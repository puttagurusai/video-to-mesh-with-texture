"""Synthetic check of the global pose refinement (stage2b `solve_poses`): ground truth is known, so it can be scored.

    python -m video_to_3d.selftest_refine

A camera orbits a point scene twice (so the second loop revisits the first). Poses start with accumulated drift and every
depth map has its own scale error, like chunked feed-forward output. We score recovered rotation / camera-centre error and
depth scale against the truth, with and without loop-closure matches. Takes a few seconds, needs no GPU and no video.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .config import RefineCfg
from .stage2b_refine import solve_poses


def make_case(n_per_loop=30, loops=2, drift_deg=0.6, drift_t=0.03, scale_sigma=0.08, noise_px=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_loop * loops
    W, H, f = 800, 520, 800.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    # scene: a far ring wall with texture points plus a central object, both visible from every orbit position
    ang = rng.uniform(0, 2 * np.pi, 1500)
    wall = np.stack([9 * np.cos(ang), rng.uniform(-3, 2, 1500), 9 * np.sin(ang)], 1)
    obj = rng.normal(size=(800, 3)) * [0.7, 0.5, 0.7]
    P = np.vstack([wall, obj])
    # cameras orbit at radius 3 looking at the centre; second loop slightly higher
    R_gt, t_gt = [], []
    for i in range(n):
        a = 2 * np.pi * (i % n_per_loop) / n_per_loop
        h = -0.4 - 0.15 * (i // n_per_loop)
        c = np.array([3 * np.cos(a), h, 3 * np.sin(a)])
        zc = -c / np.linalg.norm(c)
        xc = np.cross([0, 1.0, 0], zc); xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        Rcw = np.stack([xc, yc, zc])
        R_gt.append(Rcw)
        t_gt.append(-Rcw @ c)
    R_gt, t_gt = np.array(R_gt), np.array(t_gt)
    # drifted starting poses (frame 0 exact) and per-frame depth scale errors (frame 0 exact)
    dR = np.cumsum(rng.normal(size=(n, 3)) * np.radians(drift_deg), 0)
    dt = np.cumsum(rng.normal(size=(n, 3)) * drift_t, 0)
    dR[0] = dt[0] = 0
    R0 = Rotation.from_rotvec(dR).as_matrix() @ R_gt
    t0 = t_gt + dt
    s_err = np.exp(rng.normal(size=n) * scale_sigma)
    s_err[0] = 1.0

    def cam(k):
        Xc = P @ R_gt[k].T + t_gt[k]
        z = Xc[:, 2]
        uv = (Xc @ K.T)[:, :2] / np.maximum(z, 1e-9)[:, None]
        ok = (z > 0.5) & (uv[:, 0] > 20) & (uv[:, 0] < W - 20) & (uv[:, 1] > 20) & (uv[:, 1] < H - 20)
        return uv, z, ok

    proj = [cam(k) for k in range(n)]

    def matches(k, j, m=120):
        both = np.flatnonzero(proj[k][2] & proj[j][2])
        if len(both) < 20:
            return None
        pick = rng.choice(both, min(m, len(both)), replace=False)
        return (np.full(len(pick), k), np.full(len(pick), j), proj[k][0][pick],
                proj[j][0][pick] + rng.normal(size=(len(pick), 2)) * noise_px, proj[k][1][pick] * s_err[k])

    tracks = [matches(k, k + o) for k in range(n) for o in (1, 2) if k + o < n]
    loops = []
    for k in range(n_per_loop):
        for a, b in ((k, k + n_per_loop), (k + n_per_loop, k)):       # both directions of each revisit pair
            if b < n:
                loops.append(matches(a, b))

    def stack(parts):
        parts = [p for p in parts if p is not None]
        return [np.concatenate([p[i] for p in parts]) for i in range(5)]
    return dict(R0=R0, t0=t0, R_gt=R_gt, t_gt=t_gt, K=K, s_err=s_err, tracks=stack(tracks), loops=stack(loops))


def errors(R, t, Rg, tg):
    rot = np.degrees(np.linalg.norm(Rotation.from_matrix(R @ Rg.transpose(0, 2, 1)).as_rotvec(), axis=1))
    c, cg = -np.einsum("nba,nb->na", R, t), -np.einsum("nba,nb->na", Rg, tg)
    return rot, np.linalg.norm(c - cg, axis=1)


def main() -> int:
    case = make_case()
    n_track = len(case["tracks"][0])
    print(f"{len(case['R0'])} frames, {n_track} neighbour matches, {len(case['loops'][0])} loop matches")
    r0, c0 = errors(case["R0"], case["t0"], case["R_gt"], case["t_gt"])
    print(f"start      : rotation error median {np.median(r0):.2f} deg (max {r0.max():.2f}), camera-centre error median "
          f"{np.median(c0):.3f} (max {c0.max():.3f}) [orbit radius 3]")
    results = {}
    for name, loop, depth_scale in (("tracks only", False, False), ("tracks + depth scale", False, True),
                                    ("tracks + loops", True, False), ("tracks + loops + depth scale", True, True)):
        rc = RefineCfg(loop_closure=loop, depth_scale=depth_scale)
        parts = [np.concatenate([a, b]) for a, b in zip(case["tracks"], case["loops"])] if loop else case["tracks"]
        kk, jj, p, q, z = parts
        out = solve_poses(case["R0"], case["t0"], case["K"], kk.astype(int), jj.astype(int), p, q, z, n_track if loop else len(kk), rc)
        r, c = errors(out["R"], out["t"], case["R_gt"], case["t_gt"])
        rec = out["scale"] * case["s_err"]            # depth after correction relative to truth; 1.0 = recovered
        results[name] = (np.median(r), np.median(c), c.max())
        print(f"{name:30s}: rotation median {np.median(r):.2f} deg (max {r.max():.2f}), centre median {np.median(c):.3f} "
              f"(max {c.max():.3f}), depth-scale error median {np.median(np.abs(rec - 1)) * 100:.1f}%  "
              f"held-out {out['held_out_all_px']['before_median']:.1f} -> {out['held_out_all_px']['after_median']:.2f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
