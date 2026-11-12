"""Stage 2: COLMAP structure-from-motion via pycolmap -> camera poses + sparse landmarks."""
from __future__ import annotations

import re
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json


def _rigid_matrix(rigid) -> np.ndarray:
    """3x4 [R|t] from a pycolmap Rigid3d, tolerant of property-vs-method API differences."""
    m = rigid() if callable(rigid) else rigid
    return np.asarray(m.matrix() if callable(getattr(m, "matrix", None)) else m.matrix)


def _frame_index(name: str) -> int:
    return int(re.search(r"frame_(\d+)", name).group(1))


def run_colmap(cfg: Config, paths: Paths, names: list[str]) -> pycolmap.Reconstruction:
    work = paths.colmap
    db = work / "db.db"
    sparse = work / "sparse"
    for p in (db, sparse):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    sparse.mkdir(parents=True)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = cfg.sfm.camera_model
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = cfg.sfm.max_image_size
    ext.sift.max_num_features = cfg.sfm.max_features
    t0 = time.time()
    print(f"[stage2] extracting SIFT (cuda={pycolmap.has_cuda}) for {len(names)} images ...")
    pycolmap.extract_features(db, paths.keyframes, image_names=names,
                              camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext)
    t1 = time.time()

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = cfg.sfm.matcher_overlap
    pairing.loop_detection = False
    print(f"[stage2] sequential matching (features took {t1 - t0:.0f}s) ...")
    pycolmap.match_sequential(db, pairing_options=pairing)
    t2 = time.time()

    print(f"[stage2] incremental mapping (matching took {t2 - t1:.0f}s) ...")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.multiple_models = False  # we keep one model; don't spend minutes seeding leftover ones
    opts.random_seed = cfg.sfm.random_seed
    opts.ba_refine_extra_params = cfg.sfm.refine_distortion
    recs = pycolmap.incremental_mapping(db, paths.keyframes, sparse, opts)
    print(f"[stage2] mapping took {time.time() - t2:.0f}s")
    if not recs:
        raise RuntimeError("COLMAP produced no reconstruction; the video likely lacks parallax/texture")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    if len(recs) > 1:
        print(f"[stage2] WARNING: {len(recs)} disconnected models; using the largest "
              f"({best.num_reg_images()} images)")
    return best


def export(rec: pycolmap.Reconstruction, paths: Paths, n_keyframes: int) -> dict:
    cams = {}
    for cid, cam in rec.cameras.items():
        cams[cid] = {
            "model": cam.model_name, "width": cam.width, "height": cam.height,
            "params": list(cam.params), "K": np.asarray(cam.calibration_matrix()),
        }

    frames = []
    index_of_image_id = {}
    for iid in sorted(rec.reg_image_ids()):
        im = rec.images[iid]
        T = _rigid_matrix(im.cam_from_world)
        R_cw, t_cw = T[:, :3], T[:, 3]
        k = _frame_index(im.name)
        index_of_image_id[iid] = k
        frames.append({
            "frame": k, "image_id": iid, "name": im.name, "camera_id": im.camera_id,
            "R_cw": R_cw, "t_cw": t_cw,
            "R_wc": R_cw.T, "t_wc": -R_cw.T @ t_cw,  # t_wc = camera centre in world coords
            "num_points3D": im.num_points3D,
        })
    frames.sort(key=lambda f: f["frame"])

    xyz, err, tlen, obs = [], [], [], []
    for pi, (pid, pt) in enumerate(rec.points3D.items()):
        xyz.append(pt.xyz)
        err.append(pt.error)
        tlen.append(pt.track.length())
        for el in pt.track.elements:
            if el.image_id in index_of_image_id:
                uv = rec.images[el.image_id].points2D[el.point2D_idx].xy
                obs.append((pi, index_of_image_id[el.image_id], uv[0], uv[1]))
    np.savez_compressed(
        paths.colmap / "points3D.npz",
        xyz=np.asarray(xyz, np.float64), error=np.asarray(err, np.float32),
        track_length=np.asarray(tlen, np.int32),
        # one row per observation: point index, keyframe index, u, v (pixels, distorted image)
        obs=np.asarray(obs, np.float64),
    )

    registered = len(frames)
    mean_err = float(rec.compute_mean_reprojection_error())
    out = {"cameras": cams, "frames": frames,
           "scale_note": "arbitrary SfM scale; world frame = COLMAP model frame"}
    write_json(paths.tracking / "camera_poses.json", out)
    stats = {
        "keyframes": n_keyframes, "registered": registered,
        "registered_frac": registered / n_keyframes,
        "points3D": len(xyz), "mean_reproj_error_px": mean_err,
        "mean_track_length": float(rec.compute_mean_track_length()),
        "camera": {str(c): {"model": v["model"], "params": v["params"]} for c, v in cams.items()},
    }
    record_metrics(paths, "stage2_sfm", stats)
    return stats


def plot_cameras(paths: Paths) -> None:
    poses = read_json(paths.tracking / "camera_poses.json")
    pts = np.load(paths.colmap / "points3D.npz")["xyz"]
    centres = np.array([f["t_wc"] for f in poses["frames"]])
    fwd = np.array([np.asarray(f["R_wc"])[:, 2] for f in poses["frames"]])  # camera +z in world
    keep = np.abs(pts - np.median(pts, 0)).max(1) < 5 * np.median(np.abs(pts - np.median(pts, 0))) * 3 + 1e-9
    pts = pts[keep]
    span = np.ptp(np.vstack([pts, centres]), axis=0).max()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*pts[::max(1, len(pts) // 4000)].T, s=1, c="gray", alpha=0.5)
    ax.plot(*centres.T, "r.-")
    ax.quiver(*centres.T, *(fwd * span * 0.08).T, color="b")
    ax.set_title(f"{len(centres)} cameras, {len(pts)} sparse points (SfM scale)")
    fig.savefig(paths.viz / "sfm_cameras.png", dpi=110)
    plt.close(fig)


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    if (paths.tracking / "camera_poses.json").exists() and not force:
        print("[stage2] tracking/camera_poses.json exists, skipping (use --force)")
        return
    meta = read_json(paths.keyframes / "meta.json")
    names = [f"{paths.frame_name(f['index'])}.png" for f in meta["frames"]]
    rec = run_colmap(cfg, paths, names)
    if rec.num_points3D() == 0:
        raise RuntimeError("COLMAP model has no 3D points; initialisation failed (too little parallax?)")
    stats = export(rec, paths, len(names))
    print(f"[stage2] registered {stats['registered']}/{stats['keyframes']} frames, "
          f"{stats['points3D']} points, mean reproj {stats['mean_reproj_error_px']:.2f}px, "
          f"camera {stats['camera']}")

    sc = cfg.sfm
    problem = None
    if stats["registered_frac"] < sc.min_registered_frac:
        problem = (f"Only {stats['registered_frac']:.0%} of keyframes registered "
                   f"(need {sc.min_registered_frac:.0%}); video unsuitable or needs more overlap")
    elif stats["mean_reproj_error_px"] > sc.max_reproj_error_px:
        problem = (f"Mean reprojection error {stats['mean_reproj_error_px']:.2f}px exceeds "
                   f"{sc.max_reproj_error_px}px; poses are unreliable")
    if problem:
        # Remove the export so the skip-if-exists cache can't feed bad poses to later stages.
        (paths.tracking / "camera_poses.json").unlink(missing_ok=True)
        raise RuntimeError(problem)
    plot_cameras(paths)
