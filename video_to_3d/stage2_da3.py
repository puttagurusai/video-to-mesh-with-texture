"""Stage 2 (da3 backend): camera poses, intrinsics and depth from Depth Anything 3.

Runs the existing da3_reconstruct.py (--poses-only) from the earlier project with its own venv, then converts
its camera bundle into this pipeline's formats:
  tracking/camera_poses.json   same schema as the COLMAP backend (K rescaled to keyframe resolution)
  depth/aligned/*.npy          DA3 depth upsampled to keyframe resolution, NaN where invalid
World frame = DA3's frame (first chunk), units = DA3 units (arbitrary scale, like SfM scale).

The DA3 script's depth filters (confidence percentile, depth-gradient edge removal, cross-view filter) are
switched off: the edge-gradient filter would delete exactly the depth discontinuities that edge lifting needs.
Reliability is judged afterwards by stage 3's own cross-view confidence.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json


def _frame_index(name: str) -> int:
    return int(re.search(r"frame_(\d+)", name).group(1))


def run_da3(cfg: Config, paths: Paths) -> Path:
    root = Path(os.environ.get("DA3_ROOT") or cfg.da3.root)     # env var overrides the config (different machine / folder)
    py = root / "tools" / "venv_da3" / "Scripts" / "python.exe"
    if not py.exists():                                          # portable bundle: one interpreter runs everything
        py = Path(os.environ.get("DA3_PYTHON") or sys.executable)
    script = root / "scripts" / "da3_reconstruct.py"
    for p in (py, script, root / "tools" / "da3_models" / "DA3-BASE"):
        if not p.exists():
            raise FileNotFoundError(f"DA3 install incomplete, missing: {p}\n"
                                    f"Set da3.root in pipeline.yaml or the DA3_ROOT environment variable to a folder containing "
                                    f"scripts/da3_reconstruct.py, tools/da3_src/src and tools/da3_models/DA3-BASE "
                                    f"(plus tools/venv_da3, or run the pipeline with a Python that has the DA3 dependencies)")
    out_dir = paths.root / "da3"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "scene.ply"
    cmd = [str(py), "-I", str(script), "--frames", str(paths.keyframes), "--out", str(out), "--poses-only",
           "--process-res", str(cfg.da3.process_res), "--chunk-max", str(cfg.da3.chunk_max),
           "--overlap", str(cfg.da3.overlap),
           "--edge-grad", "1e9", "--conf-percentile", "0", "--no-consist", "--max-depth", "1e9"]
    t0 = time.time()
    print(f"[stage2] DA3 poses+depth on GPU ({cfg.da3.chunk_max}-frame chunks) ...")
    # cwd is our workdir so nothing is written into the DA3 project folder. Output is streamed line by line so the
    # console (and the run log) shows progress during the minute or two DA3 takes, and kept in full in da3/da3_log.txt.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    lines: list[str] = []
    proc = subprocess.Popen(cmd, cwd=paths.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=env)
    for line in proc.stdout:
        lines.append(line)
        text = line.rstrip()
        if text and "%|" not in text and "it/s]" not in text:     # skip progress-bar redraws, keep real messages
            print(f"   DA3> {text}", flush=True)
    rc = proc.wait()
    full = "".join(lines)
    (out_dir / "da3_log.txt").write_text(full, encoding="utf-8")
    if rc != 0:
        raise RuntimeError(f"DA3 failed (exit {rc}); see da3/da3_log.txt\n{full[-1500:]}")
    print(f"[stage2] DA3 finished in {time.time() - t0:.0f}s")
    return out_dir / "scene_cams.npz"


def convert(cfg: Config, paths: Paths, npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    names, R, t, K, (H, W) = z["names"], z["R"], z["t"], z["K"], z["hw"]
    depth = z["depth"].astype(np.float32)
    meta = read_json(paths.keyframes / "meta.json")
    Wk, Hk = meta["frames"][0]["width"], meta["frames"][0]["height"]
    sx, sy = Wk / W, Hk / H
    Kk = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                   [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5],
                   [0, 0, 1.0]])

    for sub in ("aligned", "confidence"):  # drop stale maps from any earlier backend
        for f in (paths.depth / sub).glob("*.npy"):
            f.unlink()

    frames = []
    for i, nm in enumerate(names):
        k = _frame_index(str(nm))
        d = depth[i]
        valid = d > 0
        up = cv2.resize(d, (Wk, Hk), interpolation=cv2.INTER_LINEAR)
        ok = cv2.resize(valid.astype(np.uint8), (Wk, Hk), interpolation=cv2.INTER_NEAREST) > 0
        up[~ok | ~(up > 0)] = np.nan
        np.save(paths.depth__aligned / f"{paths.frame_name(k)}.npy", up.astype(np.float32))
        Rcw, tcw = R[i], t[i]
        frames.append({"frame": k, "name": f"{paths.frame_name(k)}.png", "camera_id": 1,
                       "R_cw": Rcw, "t_cw": tcw, "R_wc": Rcw.T, "t_wc": -Rcw.T @ tcw})
    frames.sort(key=lambda f: f["frame"])
    cam = {"model": "PINHOLE", "width": Wk, "height": Hk,
           "params": [Kk[0, 0], Kk[1, 1], Kk[0, 2], Kk[1, 2]], "K": Kk}
    write_json(paths.tracking / "camera_poses.json", {
        "backend": "da3", "cameras": {"1": cam}, "frames": frames,
        "scale_note": "DA3 units (arbitrary scale); world frame = DA3 first-chunk frame"})

    centres = np.array([f["t_wc"] for f in frames])
    stats = {"backend": "da3", "keyframes": len(meta["frames"]), "registered": len(frames),
             "registered_frac": len(frames) / len(meta["frames"]),
             "da3_resolution": [int(W), int(H)], "keyframe_resolution": [Wk, Hk],
             "focal_px_keyframe_res": float(Kk[0, 0]),
             "baseline_span_da3_units": float(np.linalg.norm(np.ptp(centres, axis=0))),
             "median_depth_da3_units": float(np.median(depth[depth > 0]))}
    record_metrics(paths, "stage2_poses", stats)
    return stats


def _check_disk(paths: Paths, meta: dict) -> None:
    """Stop early, with a clear message, if the depth maps for THIS video will not fit on the disk (the heavy part of the
    output): per keyframe, an aligned depth map plus a confidence map, both float32 at keyframe resolution, plus margin."""
    import shutil
    n = len(meta["frames"])
    w, h = meta["frames"][0]["width"], meta["frames"][0]["height"]
    need = int(n * (2 * w * h * 4) * 1.25) + 200 * 2**20          # +25% and 200 MB for the mesh, cloud and viewer
    free = shutil.disk_usage(paths.root).free
    print(f"[stage2] disk: about {need / 2**30:.2f} GB needed for {n} keyframes at {w}x{h}, {free / 2**30:.2f} GB free")
    if free < need:
        raise RuntimeError(
            f"Not enough disk space for this video: about {need / 2**30:.2f} GB needed, {free / 2**30:.2f} GB free at "
            f"{paths.root.anchor}. Free some space, choose another output folder (run.bat <video> <folder>), or lower "
            f"keyframes.max_side / keyframes.max_keyframes in pipeline.yaml.")


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    poses_path = paths.tracking / "camera_poses.json"
    if poses_path.exists() and not force and read_json(poses_path).get("backend") == "da3":
        print("[stage2] DA3 poses exist, skipping (use --force)")
        return
    meta = read_json(paths.keyframes / "meta.json")
    _check_disk(paths, meta)
    npz = run_da3(cfg, paths)
    stats = convert(cfg, paths, npz)
    print(f"[stage2] {stats['registered']}/{stats['keyframes']} frames posed, "
          f"focal {stats['focal_px_keyframe_res']:.0f}px at {stats['keyframe_resolution']}, "
          f"camera path spans {stats['baseline_span_da3_units']:.2f} vs median depth "
          f"{stats['median_depth_da3_units']:.2f} DA3 units "
          f"(baseline/depth = {stats['baseline_span_da3_units'] / stats['median_depth_da3_units']:.2f})")
    if stats["registered_frac"] < 0.8:
        poses_path.unlink(missing_ok=True)
        raise RuntimeError(f"Only {stats['registered_frac']:.0%} of {len(meta['frames'])} keyframes posed")
    if cfg.refine.enabled:  # DA3 poses are feed-forward; tighten them against tracked corners (see stage2b_refine)
        from . import stage2b_refine
        stage2b_refine.run(cfg, paths)
