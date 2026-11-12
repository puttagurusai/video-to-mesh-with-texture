"""Stage 7: 3D shape classification and persistent provisional edges (Step 1 of incremental edge mapping).

  lifted 3D samples -> weighted PCA line fit -> straight / curved / piecewise / undetermined -> edge record with ID

The decision uses the 3D samples, never the 2D image alone (a straight-looking 2D edge can bend in 3D and the reverse).

Anisotropic noise model: monocular depth is uncertain ALONG the viewing ray (sigma_ray = rel_depth_sigma * z) but the
sideways position is known to about the edge-localisation accuracy (sigma_lat = lateral_px_sigma * z / f). Each
sample's perpendicular deviation from the fitted line is split along / across its viewing ray and normalised by those
two sigmas; "straight" means the confidence-weighted rms of that normalised deviation is small. A bend that is
smaller than the noise is therefore never promoted to a curve, yet sideways bends are judged at pixel precision.

When evidence is thin (contour shorter than min_len_px, too few samples, low depth confidence) the shape stays
"undetermined" rather than being guessed; the geometry is still stored.

Representation: straight -> two endpoints on the fitted line; curved -> adaptive polyline; piecewise -> only the
corner vertices; undetermined -> adaptive polyline. Every edge is "provisional": one observation cannot establish
that its 3D position is right, so confidence is capped (single_frame_conf_cap).
World coordinates = the DA3 world (first-camera frame), as in stage 5.
"""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json
from .stage5_lift import project, rdp3d
from .stage6_match import _interp_along

SHAPE_COLORS = {"straight": (255, 120, 0), "curved": (0, 140, 255), "piecewise": (0, 200, 0),
                "undetermined": (150, 150, 150)}


def dense_samples(e: dict, step_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolated world points, source uv, depth and confidence along a lifted polyline."""
    P, uv, z = (np.asarray(e[k], float) for k in ("points", "uv", "z"))
    conf = np.array([0.5 if c is None else c for c in e.get("conf", [None] * len(P))], float)
    n = np.maximum(1, np.round(np.linalg.norm(np.diff(uv, axis=0), axis=1) / step_px).astype(int))
    return _interp_along(P, n), _interp_along(uv, n), _interp_along(z, n), _interp_along(conf, n)


def fit_line(X: np.ndarray, w: np.ndarray):
    """Weighted PCA line: centroid, unit direction, signed positions along it, perpendicular residual vectors."""
    c = (w[:, None] * X).sum(0) / w.sum()
    Xc = X - c
    evals, evecs = np.linalg.eigh((Xc * w[:, None]).T @ Xc / w.sum())
    d = evecs[:, -1]
    t = Xc @ d
    return c, d, t, Xc - np.outer(t, d), float(evals[-2] / max(evals[-1], 1e-18))


def ray_line_residuals(X: np.ndarray, cam: np.ndarray, V: np.ndarray, infinite: bool = False):
    """Per sample: (sideways distance between the sample's VIEWING RAY and the stored line/polyline, depth offset of the
    sample along its ray relative to the closest approach, range).

    Depth noise moves a sample along its ray, so it must only be charged to the depth term. Taking the nearest 3D point on
    the line and splitting that offset afterwards (the first version) leaks depth error into the sideways term for any
    edge that recedes in depth, which wrongly made receding straight edges look curved and refused true matches."""
    d1 = X - cam
    rng = np.linalg.norm(d1, axis=1)
    d1 = d1 / rng[:, None]
    best_lat = np.full(len(X), np.inf)
    best_dz = np.zeros(len(X))
    segs = [(V[0], V[-1])] if infinite else list(zip(V[:-1], V[1:]))
    for A, B in segs:
        d2 = B - A
        c = float(d2 @ d2)
        if c < 1e-18:
            continue
        w0 = cam - A
        b, d, e = d1 @ d2, d1 @ w0, float(d2 @ w0)
        denom = c - b ** 2
        s = np.where(denom > 1e-9 * c, (e - b * d) / np.maximum(denom, 1e-18), 0.0)
        if not infinite:
            s = np.clip(s, 0.0, 1.0)
        t = s * b - d
        lat = np.linalg.norm(w0 + t[:, None] * d1 - s[:, None] * d2, axis=1)
        m = lat < best_lat
        best_lat[m], best_dz[m] = lat[m], (rng - t)[m]
    return best_lat, best_dz, rng


def turning_angles(V: np.ndarray) -> np.ndarray:
    a, b = V[1:-1] - V[:-2], V[2:] - V[1:-1]
    cosang = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def build_edge(e: dict, frame: int, gc, cam_center: np.ndarray, focal: float) -> dict | None:
    X, uv, z, conf = dense_samples(e, gc.dense_step_px)
    if len(X) < 3:
        return None
    zmed = float(np.median(z))
    w = np.maximum(conf, 0.05) / z ** 2
    c, d, t, res, lin_ratio = fit_line(X, w)

    # anisotropic normalised deviation: sideways error (ray-to-line distance) vs depth error (along the ray)
    lat, dz, rng = ray_line_residuals(X, cam_center, np.stack([c, c + d]), infinite=True)
    en = np.sqrt((dz / (gc.rel_depth_sigma * rng)) ** 2 + (lat / (gc.lateral_px_sigma * rng / focal)) ** 2)
    wc = np.maximum(conf, 0.05)
    rms_n = float(np.sqrt((wc * en ** 2).sum() / wc.sum()))

    length = float(t.max() - t.min())
    len2d = float(np.linalg.norm(np.diff(uv, axis=0), axis=1).sum())
    mean_conf = float(np.average(conf, weights=1 / z ** 2))
    evidence_ok = len2d >= gc.min_len_px and len(X) >= gc.min_samples and mean_conf >= gc.min_conf
    straight_ok = rms_n <= gc.straight_k and en.max() <= gc.max_dev_factor

    shape = "undetermined" if not evidence_ok else ("straight" if straight_ok else "curved")
    lat_tol = gc.lateral_px_sigma * zmed / focal
    if shape == "straight":
        lo, hi = (t.min(), t.max()) if t[-1] >= t[0] else (t.max(), t.min())   # keep original traversal order
        V = np.stack([c + d * lo, c + d * hi])
    else:
        V = X[rdp3d(X, lat_tol)]
        if shape == "curved":
            idx = rdp3d(X, 2.5 * lat_tol)
            if 3 <= len(idx) <= 6 and (turning_angles(X[idx]) >= gc.corner_deg).all():
                shape, V = "piecewise", X[idx]          # a few straight pieces joined at sharp corners
    prob = float(e.get("prob", 0.5))
    conf_edge = gc.single_frame_conf_cap * mean_conf * min(1.0, len(X) / 10) * (0.5 + 0.5 * min(prob, 1.0))
    conf_edge *= 1.0 if shape != "undetermined" else 0.6

    # 2D straightness of the same contour, to expose cases where 2D and 3D disagree
    c2 = uv.mean(0)
    u2 = np.linalg.svd(uv - c2, full_matrices=False)[2][0]
    perp2 = np.abs((uv - c2) @ np.array([-u2[1], u2[0]]))
    return {
        "type": shape, "object_id": None, "edge_kind": e["type"], "status": "provisional",
        "vertices_3d": np.round(V, 5), "confidence": round(float(conf_edge), 3),
        "observed_frames": [frame], "source": {"frame": frame, "polyline_2d": e["id"]},
        "fit": {"rms_normalized": rms_n, "max_normalized": float(en.max()), "length_3d": length,
                "length_px_2d": len2d, "z_median": zmed, "mean_depth_conf": mean_conf, "samples": int(len(X)),
                "linearity_ratio": lin_ratio, "direction": d if shape == "straight" else None,
                "straight_in_2d": bool(np.sqrt((perp2 ** 2).mean()) <= 1.5)},
        "uv_ref": np.round(uv[:: max(1, len(uv) // 12)], 1),
    }


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    gc = cfg.geometry
    out_path = paths.tracking / "edge_map_provisional.json"
    if out_path.exists() and not force:
        print("[stage7] edge_map_provisional.json exists, skipping (use --force)")
        return
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    K = np.asarray(next(iter(poses["cameras"].values()))["K"])
    ids = sorted(frames)
    edges, next_id = [], 1
    for k in ids:
        C = np.asarray(frames[k]["t_wc"])
        for run_idx, e in enumerate(read_json(paths.edges__polylines3d / f"{paths.frame_name(k)}.json")["polylines3d"]):
            rec = build_edge(e, k, gc, C, float(K[0, 0]))
            if rec is None:
                continue
            rec["source"]["run"] = run_idx      # index into the frame's lifted list (stage 8 re-reads dense samples)
            rec["id"] = next_id
            next_id += 1
            edges.append(rec)
    write_json(out_path, {"world": "DA3 first-camera frame, DA3 units", "edges": edges})

    # ---- quality numbers -----------------------------------------------------------------------------------
    shapes = [e["type"] for e in edges]
    frac = {s: shapes.count(s) / len(shapes) for s in SHAPE_COLORS}
    by_kind = {}
    for kind in ("depth_discontinuity", "crease_candidate", "texture", "unknown"):
        sub = [e["type"] for e in edges if e["edge_kind"] == kind]
        if sub:
            by_kind[kind] = {s: sub.count(s) / len(sub) for s in SHAPE_COLORS}
    cross = {"straight_2d_and_3d": 0, "straight_2d_not_3d": 0, "not_straight_2d_but_3d_straight": 0, "neither": 0}
    for e in edges:
        if e["type"] == "undetermined":
            continue
        s3, s2 = e["type"] == "straight", e["fit"]["straight_in_2d"]
        cross["straight_2d_and_3d" if s3 and s2 else "straight_2d_not_3d" if s2 else
              "not_straight_2d_but_3d_straight" if s3 else "neither"] += 1
    # fidelity of the 2-point representation: reproject the endpoints, compare to the original contour samples
    dev_px = []
    for e in edges:
        if e["type"] != "straight":
            continue
        f = frames[e["source"]["frame"]]
        uvv, _ = project(np.asarray(e["vertices_3d"]), K, np.asarray(f["R_cw"]), np.asarray(f["t_cw"]))
        a, b = uvv
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-6:
            continue
        r = np.asarray(e["uv_ref"]) - a
        dev_px.append(float(np.abs(r[:, 0] * ab[1] - r[:, 1] * ab[0]).max() / L))
    res = {"edges": len(edges), "shape_fraction": frac, "shape_fraction_by_2d_kind": by_kind,
           "straight_vs_2d_straightness": cross,
           "straight_endpoint_fidelity_px": {"median_max_dev": float(np.median(dev_px)),
                                             "p90_max_dev": float(np.percentile(dev_px, 90))} if dev_px else None,
           "median_confidence": float(np.median([e["confidence"] for e in edges])),
           "noise_model": {"rel_depth_sigma": gc.rel_depth_sigma, "lateral_px_sigma": gc.lateral_px_sigma,
                           "note": "assumed values, not measured per edge"}}
    record_metrics(paths, "stage7_edge_geometry", res)
    visualize(paths, frames, K, edges)
    print(f"[stage7] {len(edges)} provisional edges: " + ", ".join(f"{s} {frac[s]:.0%}" for s in SHAPE_COLORS))
    print("[stage7] by 2D tag, straight share: " + ", ".join(f"{k} {v['straight']:.0%}" for k, v in by_kind.items()))
    print(f"[stage7] 2D vs 3D straightness: {cross}")
    if dev_px:
        print(f"[stage7] 2-point straight edges reproject within {res['straight_endpoint_fidelity_px']['median_max_dev']:.1f}px "
              f"of the original contour (median of per-edge max, p90 {res['straight_endpoint_fidelity_px']['p90_max_dev']:.1f}px)")


def visualize(paths: Paths, frames: dict, K: np.ndarray, edges: list[dict]) -> None:
    ids = sorted(frames)
    for k in (ids[len(ids) // 3], ids[2 * len(ids) // 3]):
        img = cv2.imread(str(paths.keyframes / f"{paths.frame_name(k)}.png"))
        vis = (img * 0.45).astype(np.uint8)
        f = frames[k]
        for e in edges:
            if e["source"]["frame"] != k:
                continue
            uvv, z = project(np.asarray(e["vertices_3d"]), K, np.asarray(f["R_cw"]), np.asarray(f["t_cw"]))
            if (z <= 0).any():
                continue
            cv2.polylines(vis, [np.rint(uvv).astype(np.int32).reshape(-1, 1, 2)], False, SHAPE_COLORS[e["type"]],
                          3 if e["type"] != "undetermined" else 1, cv2.LINE_AA)
            if e["type"] == "straight":
                for p in uvv:
                    cv2.circle(vis, tuple(np.rint(p).astype(int)), 3, (255, 255, 255), -1)
        cv2.imwrite(str(paths.viz / f"geometry_overlay_{k:04d}.png"), vis)
    k = ids[len(ids) // 3]
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    for e in edges:
        if e["source"]["frame"] == k and e["type"] != "undetermined":
            V = np.asarray(e["vertices_3d"])
            col = np.array(SHAPE_COLORS[e["type"]][::-1]) / 255
            ax.plot(V[:, 0], V[:, 2], -V[:, 1], color=col, lw=1.2)
    ax.set_xlabel("x"), ax.set_ylabel("z (depth)"), ax.set_zlabel("-y")
    ax.view_init(elev=18, azim=-62)
    ax.set_title(f"frame {k} provisional 3D edges (blue straight, orange curved, green piecewise)")
    fig.savefig(paths.viz / f"geometry_3d_{k:04d}.png", dpi=90)
    plt.close(fig)
