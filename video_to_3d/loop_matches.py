"""Loop-closure correspondences between keyframes that are far apart in time but show the same place.

Neighbour-only tracking cannot see drift: when the camera comes back to a place it has seen before (an orbit, a room
walk-through that returns to its start), the estimated poses of the two visits disagree and nothing in the optimisation
asks them to agree. This module finds those revisits and returns verified point matches between them, in the same
format as `stage2b_refine.track_pairs`, so the global refinement can use them as extra constraints.

No learned place-recognition model and no download: ORB features, a cheap coarse match count to rank candidate pairs,
then full matching verified with a RANSAC fundamental matrix (rejects look-alike texture that does not agree
geometrically). Works on any video; nothing here depends on the footage.

Limits (measured on nothing but reasoning, so treat as unverified): strongly repeating textures can still produce a few
wrong pairs, which is why the solver downstream uses a robust loss; pure-rotation or featureless footage yields no pairs
and the refinement then behaves exactly as without loop closure.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .io_utils import Paths
from .stage5_lift import Sampler

WORK_SIDE = 800          # features are computed at this long-side size (points are scaled back to keyframe resolution)
COARSE_FEATURES = 250    # strongest features used for the cheap pair ranking


def _features(path, n_features: int):
    g = cv2.imread(str(path), 0)
    scale = min(1.0, WORK_SIDE / max(g.shape))
    if scale < 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)
    kps, des = orb.detectAndCompute(g, None)
    if des is None or len(kps) < 20:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8), scale
    order = np.argsort([-k.response for k in kps])
    pts = np.array([kps[i].pt for i in order], np.float32) / scale     # back to keyframe resolution
    return pts, des[order], scale


def _ratio_matches(d1, d2, ratio=0.8):
    if len(d1) < 8 or len(d2) < 8:
        return np.zeros(0, int), np.zeros(0, int)
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    a, b = [], []
    for pair in m:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            a.append(pair[0].queryIdx)
            b.append(pair[0].trainIdx)
    return np.array(a, int), np.array(b, int)


def find_loop_pairs(paths: Paths, ids: list[int], rc, workers: int | None = None):
    """Returns (kk, jj, p, q, z, info): for each verified match, source frame id kk, target frame id jj, pixel p in kk,
    pixel q in jj and the depth of kk at p (NaN-depth matches dropped). Both directions of every pair are returned."""
    workers = workers or min(8, os.cpu_count() or 4)
    n = len(ids)
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    info = {"keyframes": n, "candidate_pairs": 0, "verified_pairs": 0, "matches": 0}
    if n < 2 * rc.loop_min_gap:
        return (*empty, info)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        feats = list(ex.map(lambda k: _features(paths.keyframes / f"frame_{k:04d}.png", rc.loop_orb_features), ids))

    # 1) rank candidate pairs (far apart in keyframe order) by a cheap coarse match count
    cand = [(a, b) for a in range(n) for b in range(a + rc.loop_min_gap, n)]
    if len(cand) > rc.loop_max_candidates:                       # keep the cost bounded on very long videos
        pick = np.random.default_rng(0).choice(len(cand), rc.loop_max_candidates, replace=False)
        cand = [cand[i] for i in sorted(pick)]
    info["candidate_pairs"] = len(cand)

    def coarse(ab):
        a, b = ab
        ia, ib = _ratio_matches(feats[a][1][:COARSE_FEATURES], feats[b][1][:COARSE_FEATURES])
        return len(ia)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        score = np.array(list(ex.map(coarse, cand)))
    best: dict[int, list] = {i: [] for i in range(n)}
    for (a, b), s in zip(cand, score):
        if s >= rc.loop_min_coarse:
            best[a].append((s, b))
            best[b].append((s, a))
    chosen = set()
    for i, lst in best.items():
        for s, j in sorted(lst, reverse=True)[: rc.loop_pairs_per_frame]:
            chosen.add((min(i, j), max(i, j)))

    # 2) verify with full matching + RANSAC fundamental matrix
    def verify(ab):
        a, b = ab
        ia, ib = _ratio_matches(feats[a][1], feats[b][1])
        if len(ia) < rc.loop_min_inliers:
            return None
        pa, pb = feats[a][0][ia], feats[b][0][ib]
        F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5 / feats[a][2], 0.999)
        if F is None or mask is None:
            return None
        ok = mask.ravel() > 0
        if ok.sum() < rc.loop_min_inliers or ok.mean() < 0.25:
            return None
        return a, b, pa[ok], pb[ok]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verified = [v for v in ex.map(verify, sorted(chosen)) if v is not None]
    info["verified_pairs"] = len(verified)
    if not verified:
        return (*empty, info)

    # 3) attach depth (frame by frame so each depth map is loaded once) and emit both directions of each pair
    rng = np.random.default_rng(0)
    by_src: dict[int, list] = {}
    for a, b, pa, pb in verified:
        if len(pa) > rc.loop_matches_per_pair:
            s = rng.choice(len(pa), rc.loop_matches_per_pair, replace=False)
            pa, pb = pa[s], pb[s]
        by_src.setdefault(a, []).append((b, pa, pb))
        by_src.setdefault(b, []).append((a, pb, pa))
    kk, jj, P, Q, Z = [], [], [], [], []
    for a, lst in by_src.items():
        zs = Sampler(np.load(paths.depth__aligned / f"frame_{ids[a]:04d}.npy"))
        for b, pa, pb in lst:
            z = zs(pa.astype(np.float64))
            ok = np.isfinite(z)
            kk.append(np.full(int(ok.sum()), ids[a]))
            jj.append(np.full(int(ok.sum()), ids[b]))
            P.append(pa[ok])
            Q.append(pb[ok])
            Z.append(z[ok])
    kk, jj, P, Q, Z = (np.concatenate(x) for x in (kk, jj, P, Q, Z))
    info["matches"] = int(len(kk))
    return kk, jj, P.astype(np.float64), Q.astype(np.float64), Z, info
