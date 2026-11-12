"""Complete scene point cloud: every keyframe's depth lifted to 3D in the shared world, plus a measured cross-view check.

  python -m video_to_3d.scene_cloud [--config pipeline_mouse.yaml]

Writes <workdir>/geometry/scene_cloud.ply (all keyframes, image colours) and scene_cloud_agreement.ply (a subset coloured by
how many other keyframes confirm each point). "Confirm" means: projected into another keyframe that sees that spot, the point's
depth matches the depth observed there (|dz|/z <= tol). Points hidden behind something in that keyframe are not counted either way.

Reading the agreement: a point seen by >= 2 other keyframes with high agreement is geometry that several independent views
place at the same spot. Low agreement means the views disagree (inconsistent poses or depth), not that the point is "wrong
in one frame".
"""
from __future__ import annotations

import argparse

import numpy as np

from .config import load_config
from .io_utils import Paths, read_json, record_metrics
from .stage10_surfaces import depth_cloud


def agreement(P: np.ndarray, src: np.ndarray, frames: dict, K: np.ndarray, paths: Paths, tol: float = 0.05):
    """Per point: (views that confirm, views that contradict). Visibility-aware: hidden points count for neither."""
    ok_n = np.zeros(len(P), np.int32)
    bad_n = np.zeros(len(P), np.int32)
    for k, f in frames.items():
        nm = paths.frame_name(k)
        Z = np.load(paths.depth__aligned / f"{nm}.npy")
        conf = np.load(paths.depth__confidence / f"{nm}.npy").astype(np.float32)
        h, w = Z.shape
        Xc = P @ np.asarray(f["R_cw"]).T + np.asarray(f["t_cw"])
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
            v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
        ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
        inside = (z > 1e-6) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (src != k)
        idx = np.flatnonzero(inside)
        zo = Z[vi[idx], ui[idx]]
        seen = np.isfinite(zo) & (conf[vi[idx], ui[idx]] >= 0.5)
        idx, zo = idx[seen], zo[seen]
        rel = (z[idx] - zo) / zo
        ok_n[idx[np.abs(rel) <= tol]] += 1
        bad_n[idx[rel < -tol]] += 1                      # point lies in front of the surface this frame observes
    return ok_n, bad_n


def build(cfg, paths: Paths, n_subset: int = 100000, export_stride: int = 8, seed: int = 0) -> dict:
    import trimesh
    sc = cfg.surface
    sc.stride = export_stride
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    K = np.asarray(next(iter(poses["cameras"].values()))["K"])
    ids = sorted(frames)
    cl = depth_cloud(paths, frames, K, ids, sc)
    P, col, src = cl["P"], cl["col"], cl["src"]
    (paths.root / "geometry").mkdir(exist_ok=True)
    trimesh.PointCloud(P, colors=col).export(paths.root / "geometry" / "scene_cloud.ply")

    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(P), size=min(n_subset, len(P)), replace=False))
    ok_n, bad_n = agreement(P[sub], src[sub], frames, K, paths)
    views = ok_n + bad_n
    support = np.where(views > 0, ok_n / np.maximum(views, 1), np.nan)
    heat = np.zeros((len(sub), 3), np.uint8)
    s = np.nan_to_num(support, nan=-1.0)
    heat[:] = (130, 130, 130)
    m = s >= 0
    heat[m, 0] = (255 * (1 - s[m])).astype(np.uint8)
    heat[m, 1] = (220 * s[m]).astype(np.uint8)
    heat[m, 2] = 30
    trimesh.PointCloud(P[sub], colors=heat).export(paths.root / "geometry" / "scene_cloud_agreement.ply")

    multi = views >= 2
    stats = {"points_total": int(len(P)), "keyframes_used": len(ids), "export_stride_px": export_stride,
             "agreement_subset": int(len(sub)),
             "points_seen_by_2plus_other_keyframes": float(multi.mean()),
             "median_support_among_2plus": float(np.nanmedian(support[multi])) if multi.any() else None,
             "share_with_support_ge_0.7_among_2plus": float((support[multi] >= 0.7).mean()) if multi.any() else None,
             "share_with_support_lt_0.3_among_2plus": float((support[multi] < 0.3).mean()) if multi.any() else None,
             "tolerance": "|depth difference| <= 5% of observed depth"}
    record_metrics(paths, "scene_cloud", stats)
    return {"P": P, "col": col, "src": src, "sub": sub, "support": support, "views": views, "stats": stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline.yaml")
    cfg = load_config(ap.parse_args().config)
    out = build(cfg, Paths(cfg))
    st = out["stats"]
    print(f"scene cloud: {st['points_total']:,} points from {st['keyframes_used']} keyframes -> geometry/scene_cloud.ply")
    print(f"  seen by >=2 other keyframes: {st['points_seen_by_2plus_other_keyframes']:.0%} of points")
    print(f"  among those: median support {st['median_support_among_2plus']:.0%}, "
          f"{st['share_with_support_ge_0.7_among_2plus']:.0%} have support >= 70%, "
          f"{st['share_with_support_lt_0.3_among_2plus']:.0%} have support < 30%")
