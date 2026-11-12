"""Stage 2b: refine DA3's feed-forward poses + focal length against real image correspondences.

DA3 predicts poses at ~500 px in one pass, in chunks that are stitched afterwards; measured on tracked corners its
poses+depth mispredict where points move between adjacent keyframes by a few pixels. Here the DA3 depth maps stay fixed
(they define the 3D points) and we adjust, per frame, a small 6-DoF pose correction, one depth-scale factor (chunks can
disagree on scale) and one shared focal scale so that matched points land where the images say they do:

    residual = project_j( world_k(p, s_k * z_k(p)) ) - q          (p in frame k, q = matched location in frame j)

Two kinds of matches feed it: LK-tracked corners between neighbouring keyframes, and verified feature matches between
keyframes that revisit the same place far apart in time (loop closure, `loop_matches.py`). The second kind is what pulls
the two visits of an orbit or a walk-through onto each other; neighbour tracks alone cannot see drift.

Robust (soft-L1) least squares, frame 0 held fixed (pose and scale). Half of the matches (tracks and loops separately) are
held out and only used to score the result, so the reported improvement is not fit to the training residuals.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json
from .stage5_lift import Sampler


def _track_from(paths: Paths, ids_set: set, k: int, offsets, n_corners: int, min_dist: int):
    """Corners of keyframe k tracked into k+offset (forward-backward checked), with their depth at the keyframe.
    Self-contained (loads what it needs and frees it) so many keyframes can run in parallel threads."""
    gk = cv2.imread(str(paths.keyframes / f"frame_{k:04d}.png"), 0)
    p0 = cv2.goodFeaturesToTrack(gk, n_corners, 0.01, min_dist)
    if p0 is None:
        return []
    zs = Sampler(np.load(paths.depth__aligned / f"frame_{k:04d}.npy"))
    out = []
    for o in offsets:
        j = k + o
        if j not in ids_set:
            continue
        gj = cv2.imread(str(paths.keyframes / f"frame_{j:04d}.png"), 0)
        q, st, _ = cv2.calcOpticalFlowPyrLK(gk, gj, p0, None, winSize=(31, 31), maxLevel=4)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gj, gk, q, None, winSize=(31, 31), maxLevel=4)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - p0, axis=2).ravel() < 0.5)
        p, qq = p0.reshape(-1, 2)[good], q.reshape(-1, 2)[good]
        z = zs(p)
        ok = np.isfinite(z)
        out.append((np.full(int(ok.sum()), k), np.full(int(ok.sum()), j), p[ok], qq[ok], z[ok]))
    return out


def track_pairs(paths: Paths, ids: list[int], offsets, n_corners: int, min_dist: int, workers: int | None = None):
    import os
    from concurrent.futures import ThreadPoolExecutor
    ids_set = set(ids)
    workers = workers or min(8, os.cpu_count() or 4)                # 8 was faster than 16 on a 22-core machine
    with ThreadPoolExecutor(max_workers=workers) as ex:            # OpenCV's tracker releases the GIL
        results = list(ex.map(lambda k: _track_from(paths, ids_set, k, offsets, n_corners, min_dist), ids))
    parts = [part for r in results for part in r]
    if not parts:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))


def _unpack(x: np.ndarray, idx: dict, n: int, per: int):
    """Per-frame corrections from the parameter vector (frame positions missing from idx are fixed at 0)."""
    dR, dt, ls = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
    for k, pos in idx.items():
        dR[k], dt[k] = x[per * pos:per * pos + 3], x[per * pos + 3:per * pos + 6]
        if per == 7:
            ls[k] = x[per * pos + 6]
    return dR, dt, ls, float(np.exp(x[-1]))


def _poses_from_params(x: np.ndarray, idx: dict, R0: np.ndarray, t0: np.ndarray, per: int = 7):
    dR, dt, ls, s = _unpack(x, idx, len(R0), per)
    R = Rotation.from_rotvec(dR).as_matrix() @ R0
    return R, t0 + dt, np.exp(ls), s


def residuals(x, idx, R0, t0, K0, kk, jj, p, q, z, per=7):
    R, t, scale, s = _poses_from_params(x, idx, R0, t0, per)
    f, cx, cy = K0[0, 0] * s, K0[0, 2], K0[1, 2]
    zz = z * scale[kk]
    Xc = np.stack([(p[:, 0] - cx) / f * zz, (p[:, 1] - cy) / f * zz, zz], 1)
    Xw = np.einsum("nba,nb->na", R[kk], Xc - t[kk])
    Y = np.einsum("nab,nb->na", R[jj], Xw) + t[jj]
    uv = np.stack([f * Y[:, 0] / Y[:, 2] + cx, f * Y[:, 1] / Y[:, 2] + cy], 1)
    return (uv - q).ravel()


def _fit_residuals(x, idx, R0, t0, K0, kk, jj, p, q, z, per, prior_w):
    r = residuals(x, idx, R0, t0, K0, kk, jj, p, q, z, per)
    if per == 7 and prior_w > 0:                       # weak prior: depth scales stay near 1 unless the images insist
        r = np.concatenate([r, prior_w * x[:-1].reshape(-1, per)[:, 6]])
    return r


def _summary(a, b):
    return {"before_median": float(np.median(a)), "before_p90": float(np.percentile(a, 90)),
            "after_median": float(np.median(b)), "after_p90": float(np.percentile(b, 90))}


def solve_poses(R0, t0, K0, kk, jj, p, q, z, n_track, rc):
    """Core solve, independent of files. kk/jj are frame positions (0..n-1), matches [0:n_track] are neighbour tracks and
    the rest are loop-closure matches. Returns corrected poses, depth scales, focal scale and before/after held-out errors."""
    per = 7 if rc.depth_scale else 6
    n = len(R0)
    is_loop = np.arange(len(kk)) >= n_track
    # odd matches (within tracks, and within loops) are never used for fitting
    held = np.zeros(len(kk), bool)
    held[:n_track][1::2] = True
    held[n_track:][1::2] = True
    fit = ~held
    if rc.max_fit_tracks and fit.sum() > rc.max_fit_tracks:   # the solve cost scales with this; ~40k is plenty for ~1k parameters
        pick = np.random.default_rng(0).choice(np.flatnonzero(fit), size=rc.max_fit_tracks, replace=False)
        fit = np.zeros_like(fit)
        fit[pick] = True
    idx = {pos: m for m, pos in enumerate(range(1, n))}        # frame 0 fixed (pose and scale)
    nvar = per * len(idx) + 1
    x0 = np.zeros(nvar)

    def err_px(x, mask):
        r = residuals(x, idx, R0, t0, K0, kk[mask], jj[mask], p[mask], q[mask], z[mask], per).reshape(-1, 2)
        return np.linalg.norm(r, axis=1)

    held_track, held_loop = held & ~is_loop, held & is_loop
    before = {"held": err_px(x0, held), "track": err_px(x0, held_track)}
    if held_loop.any():
        before["loop"] = err_px(x0, held_loop)

    timing = {}
    t1 = time.time()
    n_fit = int(fit.sum())
    fk, fj = kk[fit], jj[fit]
    # Jacobian sparsity: each residual row depends on the `per` parameters of its two frames and the shared focal scale.
    # Built with array operations (a Python loop over ~100k correspondences took longer than the whole solve).
    col_of = np.full(n, -1)
    for pos, m in idx.items():
        col_of[pos] = per * m
    rows_l, cols_l = [], []
    for base in (col_of[fk], col_of[fj]):
        has = np.flatnonzero(base >= 0)
        cols = (base[has][:, None] + np.arange(per)[None, :]).ravel()
        for comp in (0, 1):
            rows_l.append(np.repeat(2 * has + comp, per))
            cols_l.append(cols)
    for comp in (0, 1):
        rows_l.append(2 * np.arange(n_fit) + comp)
        cols_l.append(np.full(n_fit, nvar - 1))
    n_prior = len(idx) if (per == 7 and rc.scale_prior_px > 0) else 0
    if n_prior:
        rows_l.append(2 * n_fit + np.arange(n_prior))
        cols_l.append(per * np.arange(n_prior) + 6)
    S = coo_matrix((np.ones(sum(len(r) for r in rows_l)), (np.concatenate(rows_l), np.concatenate(cols_l))),
                   shape=(2 * n_fit + n_prior, nvar)).tocsr()
    timing["sparsity_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    sol = least_squares(_fit_residuals, x0, jac_sparsity=S, loss="soft_l1", f_scale=rc.huber_px, x_scale="jac",
                        max_nfev=rc.max_nfev,
                        args=(idx, R0, t0, K0, fk, fj, p[fit], q[fit], z[fit], per, rc.scale_prior_px))
    timing["solve_s"] = round(time.time() - t1, 1)
    after = {"held": err_px(sol.x, held), "track": err_px(sol.x, held_track)}
    if held_loop.any():
        after["loop"] = err_px(sol.x, held_loop)
    R, t, scale, s = _poses_from_params(sol.x, idx, R0, t0, per)
    out = {"R": R, "t": t, "scale": scale, "s": s, "per": per, "timing": timing, "nfev": int(sol.nfev),
           "fit_tracks": int(fit.sum()), "held_out_tracks": int(held.sum()), "loop_matches": int(is_loop.sum()),
           "held_out_all_px": _summary(before["held"], after["held"]),
           "max_rotation_correction_deg": float(np.degrees(np.abs(
               sol.x[: per * len(idx)].reshape(-1, per)[:, :3]).max()))}
    # "held_out_error_px" keeps its old meaning (neighbour tracks) so older reports stay comparable
    out["held_out_error_px"] = _summary(before["track"], after["track"]) if held_track.any() else out["held_out_all_px"]
    if held_loop.any():
        out["loop_held_out_error_px"] = _summary(before["loop"], after["loop"])
    a = out["held_out_all_px"]
    out["accepted"] = bool(a["after_median"] < 0.9 * a["before_median"]) and (
        "loop_held_out_error_px" not in out
        or out["loop_held_out_error_px"]["after_median"] <= out["loop_held_out_error_px"]["before_median"])
    out["reliable"] = bool(out["accepted"] and a["after_median"] <= rc.reliable_px)
    return out


def refine(cfg: Config, paths: Paths, write: bool = True, poses_name: str = "camera_poses.json") -> dict:
    rc = cfg.refine
    poses = read_json(paths.tracking / poses_name)
    fr = sorted(poses["frames"], key=lambda f: f["frame"])
    ids = [f["frame"] for f in fr]
    pos_of = {k: i for i, k in enumerate(ids)}
    R0 = np.array([f["R_cw"] for f in fr])
    t0 = np.array([f["t_cw"] for f in fr])
    K0 = np.array(next(iter(poses["cameras"].values()))["K"])

    t_start = time.time()
    kk, jj, p, q, z = track_pairs(paths, ids, rc.offsets, rc.corners, rc.min_dist_px)
    timing = {"tracking_s": round(time.time() - t_start, 1)}
    n_track = len(kk)
    loop_info = {"matches": 0, "verified_pairs": 0}
    if rc.loop_closure:
        from .loop_matches import find_loop_pairs
        t1 = time.time()
        lk, lj, lp, lq, lz, loop_info = find_loop_pairs(paths, ids, rc)
        timing["loop_matching_s"] = round(time.time() - t1, 1)
        kk, jj = np.concatenate([kk, lk]), np.concatenate([jj, lj])
        p, q, z = np.concatenate([p, lp]), np.concatenate([q, lq]), np.concatenate([z, lz])
    kk, jj = np.array([pos_of[a] for a in kk], int), np.array([pos_of[a] for a in jj], int)

    stats = solve_poses(R0, t0, K0, kk, jj, p, q, z, n_track, rc)
    R, t, scale, s, per = stats.pop("R"), stats.pop("t"), stats.pop("scale"), stats.pop("s"), stats.pop("per")
    timing.update(stats.pop("timing"))
    stats.update({"tracks": int(n_track), "loop_pairs": int(loop_info["verified_pairs"]), "focal_scale": s,
                  "seconds": time.time() - t_start, "timing": timing,
                  "depth_scale_range": [float(scale.min()), float(scale.max())]})
    if stats["accepted"] and write:
        write_json(paths.tracking / "camera_poses_da3_raw.json", poses)
        Kn = K0.copy()
        Kn[0, 0] *= s
        Kn[1, 1] *= s
        cam_key = next(iter(poses["cameras"]))
        poses["cameras"][cam_key]["K"] = Kn
        poses["cameras"][cam_key]["params"] = [Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2]]
        for i, f in enumerate(fr):
            f["R_cw"], f["t_cw"] = R[i], t[i]
            f["R_wc"], f["t_wc"] = R[i].T, -R[i].T @ t[i]
            f["depth_scale"] = float(scale[i])      # recorded for reference; already applied to the depth files below
        poses["frames"] = fr
        poses["refined"] = True
        write_json(paths.tracking / "camera_poses.json", poses)
        if per == 7:
            # the fitted scale only makes sense together with the refined poses, so bake it into the depth maps once;
            # every later stage then reads consistent poses and depth without knowing about it
            for i, f in enumerate(fr):
                if abs(scale[i] - 1.0) > 1e-6:
                    fp = paths.depth__aligned / f"frame_{f['frame']:04d}.npy"
                    np.save(fp, (np.load(fp) * np.float32(scale[i])).astype(np.float32))
    if write:
        record_metrics(paths, "stage2b_refine", stats)
    return stats


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    st = refine(cfg, paths)
    h = st["held_out_error_px"]
    print(f"[stage2b] held-out corner error: median {h['before_median']:.2f} -> {h['after_median']:.2f}px, "
          f"p90 {h['before_p90']:.1f} -> {h['after_p90']:.1f}px (focal x{st['focal_scale']:.3f}, "
          f"{st['tracks']} tracks, {st['seconds']:.0f}s) -> {'ACCEPTED' if st['accepted'] else 'rejected, poses unchanged'}")
    if "loop_held_out_error_px" in st:
        lo = st["loop_held_out_error_px"]
        print(f"[stage2b] loop closure: {st['loop_pairs']} revisit pairs, {st['loop_matches']} matches; held-out revisit error "
              f"median {lo['before_median']:.2f} -> {lo['after_median']:.2f}px, p90 {lo['before_p90']:.1f} -> {lo['after_p90']:.1f}px; "
              f"depth scale per keyframe {st['depth_scale_range'][0]:.2f}..{st['depth_scale_range'][1]:.2f}")
    elif cfg.refine.loop_closure:
        print("[stage2b] loop closure: no revisit pairs found (footage never returns to a place it has seen)")
    if st["accepted"] and not st["reliable"]:
        print(f"[stage2b] WARNING: poses are still off by {st['held_out_all_px']['after_median']:.1f}px (median) after refinement; "
              f"the point cloud will not be consistent")
