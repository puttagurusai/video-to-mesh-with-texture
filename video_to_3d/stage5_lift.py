"""Stage 5: lift 2D edge polylines into world-space 3D with side-aware depth, then validate by reprojection.

Side-aware depth: depth is least reliable exactly at an edge (it is a blend of foreground and background), so it is
never read on the edge pixel. For each sample we read depth on both sides along the edge normal at two offsets
(side_offset_px = [near, far]) and extrapolate each side linearly to the edge:
    z0_side = z_near - slope_side * near,   slope_side = (z_far - z_near) / (far - near)
  * depth_discontinuity edges  -> the occluding (nearer) surface owns the edge; one side is chosen per polyline.
  * all other edges            -> mean of the two extrapolations when they agree (crease_side_tol), else the nearer.
Back-projection:  X_w = R_wc (z K^-1 [u,v,1]^T) + t_wc   (pixel coords: x right, y down, centre = integer).

Validation (phase-2 gate): lifted points of frame k are projected into neighbouring frames and compared with the
distance transform of the edges detected there. A control with the same points shifted 15 px sideways shows what
overlap would happen by chance (edge maps are dense, so a raw "distance is small" number would be misleading).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json
from .stage4_edges import COLORS

CONTROL_SHIFT_PX = 15.0


def resample(pts: np.ndarray, closed: bool, step: float) -> np.ndarray:
    p = np.vstack([pts, pts[:1]]) if closed else pts
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(round(s[-1] / step)) + 1)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1)


class Sampler:
    """Bilinear sampler over an image that may contain NaN. The NaN-fill and validity map are built once per
    image (they are 1.4M-pixel arrays) so that per-polyline calls only touch the few requested points."""

    def __init__(self, img: np.ndarray):
        self.h, self.w = img.shape
        valid = np.isfinite(img)
        self.filled = np.where(valid, img, 0.0).astype(np.float32)
        self.valid = valid.astype(np.float32)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """Values at (x,y); NaN if any contributing pixel is NaN or the point is outside the image."""
        coords = np.stack([pts[:, 1], pts[:, 0]])
        v = map_coordinates(self.filled, coords, order=1, mode="nearest")
        m = map_coordinates(self.valid, coords, order=1, mode="nearest")
        inside = (pts[:, 0] >= 0) & (pts[:, 0] <= self.w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= self.h - 1)
        return np.where((m > 0.999) & inside, v, np.nan)


def rdp3d(P: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker on an ordered (N,3) polyline; returns indices to keep."""
    keep = np.zeros(len(P), bool)
    keep[[0, -1]] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = P[b] - P[a]
        L = np.linalg.norm(d)
        seg = P[a + 1:b] - P[a]
        dist = np.linalg.norm(seg, axis=1) if L < 1e-12 else np.linalg.norm(np.cross(seg, d / L), axis=1)
        i = int(np.argmax(dist))
        if dist[i] > eps:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.flatnonzero(keep)


def lift_frame(polys: list[dict], Z: np.ndarray, conf: np.ndarray | None, K: np.ndarray,
               R_wc: np.ndarray, t_wc: np.ndarray, lc) -> list[dict]:
    near, far = lc.side_offset_px
    out = []
    zref = float(np.nanmedian(Z))
    zs = Sampler(Z)
    cs = Sampler(conf.astype(np.float32)) if conf is not None else None
    for e in polys:
        pts2 = resample(np.asarray(e["points"], np.float64), bool(e["closed"]), lc.step_px)
        if len(pts2) < 3:
            continue
        t = np.gradient(pts2, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        a_n, a_f = (zs(pts2 + n * d) for d in (near, far))
        b_n, b_f = (zs(pts2 - n * d) for d in (near, far))
        sa, sb = (a_f - a_n) / (far - near), (b_f - b_n) / (far - near)
        za0, zb0 = a_n - sa * near, b_n - sb * near     # each side extrapolated to the edge
        if e["type"] == "depth_discontinuity":
            # one consistent owning side per polyline: the nearer surface along most of its length
            side_a = np.nanmedian(za0) <= np.nanmedian(zb0) if np.isfinite(za0).any() and np.isfinite(zb0).any() \
                else np.isfinite(za0).sum() >= np.isfinite(zb0).sum()
            z = za0 if side_a else zb0
            side = "near_a" if side_a else "near_b"
        else:
            agree = np.abs(za0 - zb0) <= lc.crease_side_tol * np.minimum(za0, zb0)
            z = np.where(agree, (za0 + zb0) / 2, np.fmin(za0, zb0))
            side = "mid"
        ok = np.isfinite(z) & (z > 1e-6)
        if cs is not None:
            c = cs(pts2)
            ok &= np.nan_to_num(c, nan=0.0) >= lc.min_conf
        else:
            c = np.full(len(pts2), np.nan)
        # runs of consecutive valid samples become separate 3D polylines
        idx = np.flatnonzero(ok)
        if len(idx) < 3:
            continue
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) < 3:
                continue
            uv, zz = pts2[run], z[run]
            Xc = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * zz, (uv[:, 1] - K[1, 2]) / K[1, 1] * zz, zz], 1)
            Xw = Xc @ R_wc.T + t_wc
            if lc.smooth_sigma > 0 and len(Xw) >= 5:
                Xw = gaussian_filter1d(Xw, lc.smooth_sigma, axis=0, mode="nearest")
            keep = rdp3d(Xw, lc.rdp_eps_rel * zref)
            out.append({"id": e["id"], "type": e["type"], "side": side, "prob": e["prob"],
                        "points": np.round(Xw[keep], 5), "uv": np.round(uv[keep], 2),
                        "z": np.round(zz[keep], 5), "dense_samples": int(len(run)),
                        # cross-view depth confidence at each kept vertex (NaN -> null when no confidence map)
                        "conf": [None if np.isnan(v) else round(float(v), 3) for v in c[run][keep]]})
    return out


def project(X: np.ndarray, K: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X @ R_cw.T + t_cw
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]], 1)
    return uv, z


def edge_distance_transform(polys: list[dict], h: int, w: int) -> np.ndarray:
    raster = np.zeros((h, w), np.uint8)
    for e in polys:
        pts = np.rint(np.asarray(e["points"])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(raster, [pts], bool(e["closed"]), 255, 1)
    return cv2.distanceTransform(255 - raster, cv2.DIST_L2, 3)


def densify3d(e: dict, step_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Points along a lifted polyline (3D vertices + linear interpolation between them) at ~step in the source image."""
    P, uv = np.asarray(e["points"]), np.asarray(e["uv"])
    seg = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    n = np.maximum(1, np.round(seg / step_px).astype(int))
    pts = [P[i] + (P[i + 1] - P[i]) * np.linspace(0, 1, n[i], endpoint=False)[:, None] for i in range(len(seg))]
    return np.vstack(pts + [P[-1:]]), np.zeros(0)


def validate(cfg: Config, paths: Paths, ids, frames, K, lifted, polys2d, Z, h, w) -> dict:
    dts = {k: edge_distance_transform(polys2d[k], h, w) for k in ids}
    zsamp = {k: Sampler(Z[k]) for k in ids}
    rows = []
    self_err = []
    for k in ids:
        for j in (k - 1, k + 1):
            if j not in frames:
                continue
            fj = frames[j]
            for e in lifted[k]:
                X, _ = densify3d(e, 6.0)
                uv, z = project(X, K, np.asarray(fj["R_cw"]), np.asarray(fj["t_cw"]))
                inside = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
                if not inside.any():
                    continue
                zj = zsamp[j](uv[inside])
                # occlusion test: the point must sit at about the depth seen in j (blurred depth -> loose tol)
                visible = np.isfinite(zj) & (np.abs(z[inside] - zj) / np.maximum(zj, 1e-6) < 0.15)
                u, v = uv[inside][visible, 0], uv[inside][visible, 1]
                if len(u) == 0:
                    continue
                d_true = map_coordinates(dts[j], [v, u], order=1, mode="nearest")
                us = np.clip(u + CONTROL_SHIFT_PX, 0, w - 1)
                d_ctrl = map_coordinates(dts[j], [v, us], order=1, mode="nearest")
                rows.append((e["type"], k, j, len(u), float(np.mean(d_true)), float((d_true <= 2).mean()),
                             float((d_true <= 4).mean()), float((d_ctrl <= 2).mean()), float((d_ctrl <= 4).mean()),
                             float(np.median(d_true))))
        # sanity: reproject into the source frame itself against the original 2D samples
        fk = frames[k]
        for e in lifted[k]:
            uv, _ = project(np.asarray(e["points"]), K, np.asarray(fk["R_cw"]), np.asarray(fk["t_cw"]))
            self_err.append(np.linalg.norm(uv - np.asarray(e["uv"]), axis=1))
    out = {"self_reprojection_rms_px": float(np.sqrt(np.mean(np.concatenate(self_err) ** 2))) if self_err else None,
           "control_shift_px": CONTROL_SHIFT_PX, "by_type": {}}
    arr = np.array([r[3:] for r in rows], float)
    types = np.array([r[0] for r in rows])
    for tname in ["all", *COLORS]:
        m = np.ones(len(rows), bool) if tname == "all" else types == tname
        if not m.any():
            continue
        wgt = arr[m, 0]
        wavg = lambda col: float(np.average(arr[m, col], weights=wgt))  # noqa: E731
        out["by_type"][tname] = {
            "polylines": int(m.sum()), "points": int(wgt.sum()),
            "median_dist_px": float(np.median(arr[m, 6])),
            "within_2px": wavg(2), "within_4px": wavg(3),
            "control_within_2px": wavg(4), "control_within_4px": wavg(5)}
    out["_rows_sample"] = [list(map(lambda v: v if isinstance(v, str) else float(v), r)) for r in rows[:5]]
    return out


def overlay_reprojection(paths: Paths, k: int, j: int, lifted_k, frames, K) -> None:
    img = cv2.imread(str(paths.keyframes / f"{paths.frame_name(j)}.png"))
    vis = (img * 0.5).astype(np.uint8)
    for e in lifted_k:
        X, _ = densify3d(e, 6.0)
        uv, z = project(X, K, np.asarray(frames[j]["R_cw"]), np.asarray(frames[j]["t_cw"]))
        ok = z > 1e-6
        pts = np.rint(uv[ok]).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts], False, COLORS[e["type"]], 2, cv2.LINE_AA)
    cv2.imwrite(str(paths.viz / f"lift_reproj_{k:04d}_into_{j:04d}.png"), vis)


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    lc = cfg.lift
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    ids = sorted(frames)
    K = np.asarray(poses["cameras"]["1"]["K"]) if "1" in poses["cameras"] else \
        np.asarray(next(iter(poses["cameras"].values()))["K"])
    out_path = lambda k: paths.edges__polylines3d / f"{paths.frame_name(k)}.json"  # noqa: E731
    if not force and all(out_path(k).exists() for k in ids):
        print("[stage5] 3D polylines exist, skipping (use --force)")
        return

    lifted, polys2d, Z = {}, {}, {}
    h = w = 0
    for k in ids:
        nm = paths.frame_name(k)
        d = read_json(paths.edges__polylines / f"{nm}.json")
        h, w = d["height"], d["width"]
        polys2d[k] = d["polylines"]
        Z[k] = np.load(paths.depth__aligned / f"{nm}.npy")
        cp = paths.depth__confidence / f"{nm}.npy"
        conf = np.load(cp) if cp.exists() and np.load(cp, mmap_mode="r").shape == Z[k].shape else None
        f = frames[k]
        lifted[k] = lift_frame(polys2d[k], Z[k], conf, K, np.asarray(f["R_wc"]), np.asarray(f["t_wc"]), lc)
        write_json(out_path(k), {"frame": k, "polylines3d": lifted[k]})

    n_in = sum(len(v) for v in polys2d.values())
    n_out = sum(len(v) for v in lifted.values())
    print(f"[stage5] lifted {n_out} 3D polylines from {n_in} 2D polylines "
          f"({n_out / max(n_in, 1):.0%}; splits at invalid depth can exceed 100%)")
    res = validate(cfg, paths, ids, frames, K, lifted, polys2d, Z, h, w)
    for k in ids[len(ids) // 3:: max(1, len(ids) // 3)][:2]:
        if k + 1 in frames:
            overlay_reprojection(paths, k, k + 1, lifted[k], frames, K)
    res.update({"polylines_2d": n_in, "polylines_3d": n_out})
    record_metrics(paths, "stage5_lift", res)
    print(f"[stage5] self-reprojection RMS {res['self_reprojection_rms_px']:.2f}px (sanity)")
    for tname, m in res["by_type"].items():
        print(f"[stage5] into neighbours, {tname:20s}: {m['points']:6d} pts  median dist {m['median_dist_px']:.1f}px  "
              f"within 2px {m['within_2px']:.0%} (control {m['control_within_2px']:.0%})  "
              f"within 4px {m['within_4px']:.0%} (control {m['control_within_4px']:.0%})")
