"""Stage 8: persistent 3D edge map - project, match, fuse, keep IDs (Steps 4-5 of incremental edge mapping).

Frame by frame (camera poses are given; world = DA3 first-camera frame):
  1. every stored map edge is projected into the new frame and classified
        outside the view | occluded (behind the surface seen there) | visible
  2. visible edges are matched to detected 2D edges by pixel distance, tangent direction and depth (stage 6 matcher);
     assignment is greedy by score so a detected edge is claimed by at most one map edge, while one map edge may absorb
     several fragments of the same contour (TEED fragments contours differently in every frame)
  3. a matched observation is accepted only if its 3D samples fit the stored geometry within the depth/lateral noise
     model; otherwise it is refused (kept out, not used to drag the edge)
  4. the geometry is RE-FUSED from all accepted observations (weighted line fit for straight edges, perpendicular
     averaging against a reference polyline for curves) - the new depth never overwrites the old estimate
  5. confidence rises by noisy-OR on consistent matches; status provisional -> verified after enough observations
  6. unmatched high-confidence physical edges become NEW provisional map edges; occluded / out-of-view / missed edges
     are preserved untouched (nothing is deleted in this version)

Only "high-confidence physical" edges are tracked: determined shape, long enough, and a depth-discontinuity or crease
tag (or long and straight, since many real creases hide under the texture tag). Matching every texture line, shadow and
silhouette would create geometry nobody needs and raise false matches.

Quality is measured, not assumed: a leave-last-observation-out test compares the geometry from the first observation
alone against the multi-view fused geometry on a frame neither was fitted to.
"""
from __future__ import annotations

import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json
from .stage5_lift import Sampler, project
from .stage6_match import densify
from .stage7_edge_geometry import dense_samples, fit_line, ray_line_residuals

CONTROL_SHIFT_PX = 15.0
DIST_LOG: dict = {}   # frame-gap bucket -> [(median px to nearest detected edge, support fraction)]


# --------------------------------------------------------------------------- detected-edge index and matcher
class FrameIndex:
    """Detected 2D edges of one frame as pixels with owner polyline and local direction, plus lifted depth lookup."""

    def __init__(self, polys2d: list[dict], lifted: list[dict], h: int, w: int):
        idmap = np.zeros((h, w), np.uint16)
        ang = np.full((h, w), 255, np.uint8)
        for e in polys2d:
            pts = np.rint(np.asarray(e["points"])).astype(np.int32)
            seq = np.vstack([pts, pts[:1]]) if e["closed"] else pts
            for p, q in zip(seq[:-1], seq[1:]):
                a = int(np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])) % 180) % 180
                cv2.line(idmap, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), int(e["id"]) + 1, 1)
                cv2.line(ang, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), a, 1)
        zy, zx = np.nonzero(idmap)
        self.xy = np.stack([zx, zy], 1).astype(np.float64)
        self.pid = idmap[zy, zx].astype(np.int64) - 1
        self.ang = ang[zy, zx].astype(np.float64)
        self.tree = cKDTree(self.xy)
        uvs, zs, ids = [], [], []
        for e in lifted:
            _, uv, z = densify(e, 4.0)
            uvs.append(uv)
            zs.append(z)
            ids.append(np.full(len(uv), e["id"]))
        self.l_tree = None
        if uvs:
            self.l_uv, self.l_z, self.l_id = np.concatenate(uvs), np.concatenate(zs), np.concatenate(ids)
            self.l_tree = cKDTree(self.l_uv)


def support_match(uv: np.ndarray, z: np.ndarray, ang: np.ndarray, fi: FrameIndex, mc) -> dict:
    """A projected edge matches when enough of its samples are SUPPORTED: a detected edge pixel within gate_px, running
    in the same direction (max_angle_deg), at the same depth where the detection's depth is known. Support may come from
    any number of detected fragments; they are all returned so the caller can claim them."""
    d, i = fi.tree.query(uv)
    da = np.abs(ang - fi.ang[i])
    da = np.minimum(da, 180.0 - da)
    ok = (d <= mc.gate_px) & (da <= mc.max_angle_deg)
    if fi.l_tree is not None and ok.any():
        dl, il = fi.l_tree.query(fi.xy[i])
        known = (dl <= 6.0) & (fi.l_id[il] == fi.pid[i])
        rel = np.abs(z - fi.l_z[il]) / np.maximum(fi.l_z[il], 1e-9)
        ok &= ~known | (rel <= mc.max_depth_rel_err)
    n = int(ok.sum())
    frac = n / max(len(uv), 1)
    mean_d = float(d[ok].mean()) if n else float("inf")
    passed = frac >= mc.min_support and n >= mc.min_samples and mean_d <= mc.max_mean_dist_px
    frags = {}
    if passed:
        ids, counts = np.unique(fi.pid[i[ok]], return_counts=True)
        frags = {int(b): int(c) for b, c in zip(ids, counts) if c >= 3}
    return {"ok": bool(passed and frags), "frac": frac, "n": n, "mean_d": mean_d, "frags": frags,
            "median_d": float(np.median(d))}


# --------------------------------------------------------------------------- geometry helpers


def closest_on_polyline(P: np.ndarray, V: np.ndarray, infinite: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Closest points on polyline V (or the infinite line through V[0], V[-1]) for each row of P; any dimension."""
    best = np.full(len(P), np.inf)
    out = np.zeros_like(P, dtype=float)
    segs = [(V[0], V[-1])] if infinite else list(zip(V[:-1], V[1:]))
    for a, b in segs:
        ab = b - a
        t = ((P - a) @ ab) / max(float(ab @ ab), 1e-18)
        t = t if infinite else np.clip(t, 0.0, 1.0)
        Q = a + t[:, None] * ab
        d = np.linalg.norm(P - Q, axis=1)
        m = d < best
        best[m], out[m] = d[m], Q[m]
    return out, best


def densify_vertices(V: np.ndarray, uvV: np.ndarray, step_px: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Dense 3D points along a polyline and their image positions (3D interpolated linearly, then the caller projects)."""
    n = np.maximum(1, np.round(np.linalg.norm(np.diff(uvV, axis=0), axis=1) / step_px).astype(int))
    pts = [V[i] + (V[i + 1] - V[i]) * np.linspace(0, 1, m, endpoint=False)[:, None] for i, m in enumerate(n)]
    return np.vstack(pts + [V[-1:]]), np.zeros(0)


def reproj_dist(V: np.ndarray, shape: str, uv_obs: np.ndarray, K, R_cw, t_cw) -> np.ndarray:
    """Pixel distance from observed 2D samples to the projected edge (infinite line for straight edges: extents differ
    between views, the sideways position is what we test)."""
    uvV, z = project(V, K, R_cw, t_cw)
    if (z <= 1e-6).any():
        return np.full(len(uv_obs), np.nan)
    return closest_on_polyline(uv_obs, uvV, infinite=(shape == "straight"))[1]


# --------------------------------------------------------------------------- observations and fusion
def make_obs(rec: dict, lifted_entry: dict, cam: np.ndarray, gc, tc) -> dict:
    X, uv, z, _ = dense_samples(lifted_entry, gc.dense_step_px)
    idx = np.linspace(0, len(X) - 1, min(tc.max_obs_samples, len(X))).round().astype(int)
    return {"frame": rec["source"]["frame"], "edge": rec["id"], "X": X[idx], "uv": uv[idx], "z": z[idx],
            "w": max(rec["confidence"], 0.05) / 0.5, "cam": cam, "V": np.asarray(rec["vertices_3d"], float),
            "n_samples": int(len(X)), "len_px": rec["fit"]["length_px_2d"]}


def fuse(obs: list[dict], shape: str, tc) -> np.ndarray:
    """Geometry as a pure function of the observation set (so held-out tests can rebuild it from a subset)."""
    if shape == "straight":
        X = np.concatenate([o["X"] for o in obs])
        w = np.concatenate([np.full(len(o["X"]), o["w"]) / o["z"] ** 2 for o in obs])
        c, d, t, _, _ = fit_line(X, w)
        ref = obs[0]["X"]
        if (ref[-1] - ref[0]) @ d < 0:
            d, t = -d, -t
        return np.stack([c + d * t.min(), c + d * t.max()])
    ref = max(obs, key=lambda o: o["w"] * o["n_samples"])
    V = ref["V"].copy()
    if len(obs) == 1 or len(V) < 2:
        return V
    tau = np.gradient(V, axis=0)
    tau /= np.maximum(np.linalg.norm(tau, axis=1, keepdims=True), 1e-12)
    num, den = np.zeros_like(V), np.full(len(V), ref["w"])
    for o in obs:
        if o is ref:
            continue
        dd, ii = cKDTree(o["X"]).query(V)
        ok = dd <= tc.fuse_gate_rel * float(np.median(o["z"]))
        delta = o["X"][ii] - V
        delta -= (delta * tau).sum(1, keepdims=True) * tau          # move only sideways, never slide along the edge
        num[ok] += o["w"] * delta[ok]
        den[ok] += o["w"]
    return V + num / den[:, None]


def consistency_rms(o: dict, V: np.ndarray, shape: str, focal: float, gc) -> float:
    """Normalised rms deviation of an observation's 3D samples from the stored geometry (same noise model as stage 7)."""
    lat, dz, rng = ray_line_residuals(o["X"], o["cam"], V, infinite=(shape == "straight"))
    er, el = dz / (gc.rel_depth_sigma * rng), lat / (gc.lateral_px_sigma * rng / focal)
    return float(np.sqrt((er ** 2 + el ** 2).mean()))


def overlap_part(o: dict, V: np.ndarray, shape: str, K, Rcw, tcw, gate_px: float, min_samples: int) -> dict | None:
    """The portion of a detected observation that lies on the projected stored edge, as its own observation.
    A detected contour can be longer than the stored edge, fragmented, or wrap around a corner; only the part that
    coincides with the stored edge is evidence about it. Straight edges are tested against the infinite line but only
    within half an edge length beyond each end (so a collinear extension is allowed, a perpendicular continuation not)."""
    uvV, zV = project(V, K, Rcw, tcw)
    if (zV <= 1e-6).any():
        return None
    d = closest_on_polyline(o["uv"], uvV, infinite=(shape == "straight"))[1]
    sel = d <= gate_px
    if shape == "straight":
        ab = uvV[1] - uvV[0]
        tt = ((o["uv"] - uvV[0]) @ ab) / max(float(ab @ ab), 1e-9)
        sel &= (tt >= -0.5) & (tt <= 1.5)
    if sel.sum() < min_samples:
        return None
    part = dict(o)
    for key in ("X", "uv", "z"):
        part[key] = o[key][sel]
    part["n_samples"] = int(sel.sum())
    part["overlap_fraction"] = float(sel.mean())
    return part


def is_physical(rec: dict, tc) -> bool:
    f = rec["fit"]
    if rec["type"] == "undetermined" or f["length_px_2d"] < tc.min_len_px or rec["confidence"] < tc.min_conf:
        return False
    if rec["edge_kind"] in tc.tracked_kinds:
        return True
    return rec["type"] == "straight" and f["length_px_2d"] >= tc.long_straight_px


# --------------------------------------------------------------------------- main
def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    tc, gc, mc = cfg.track, cfg.geometry, cfg.match
    out_path = paths.tracking / "edge_map.json"
    if out_path.exists() and not force:
        print("[stage8] edge_map.json exists, skipping (use --force)")
        return
    t_start = time.time()
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    K = np.asarray(next(iter(poses["cameras"].values()))["K"])
    ids = sorted(frames)
    focal = float(K[0, 0])
    recs = read_json(paths.tracking / "edge_map_provisional.json")["edges"]
    by_frame: dict[int, dict[int, list]] = {}
    for r in recs:
        by_frame.setdefault(r["source"]["frame"], {}).setdefault(r["source"]["polyline_2d"], []).append(r)

    edges: list[dict] = []          # the persistent map
    log = []
    gap_stats: dict[str, list] = {}  # frames since last observation -> [visible, matched, no 2D candidate, control]
    group_gap: dict = {}             # (edge kind or shape, gap bucket) -> [visible, matched, control]
    next_id = 1
    for j in ids:
        nm = paths.frame_name(j)
        fj = frames[j]
        Rcw, tcw, Cj = np.asarray(fj["R_cw"]), np.asarray(fj["t_cw"]), np.asarray(fj["t_wc"])
        lifted = read_json(paths.edges__polylines3d / f"{nm}.json")["polylines3d"]
        d2 = read_json(paths.edges__polylines / f"{nm}.json")
        h, w = d2["height"], d2["width"]
        fe = FrameIndex(d2["polylines"], lifted, h, w)
        zs = Sampler(np.load(paths.depth__aligned / f"{nm}.npy"))
        polys_here = by_frame.get(j, {})
        stats = {"frame": j, "map_before": len(edges), "outside": 0, "occluded": 0, "visible": 0, "matched": 0,
                 "missed": 0, "refused_3d": 0, "too_little_overlap": 0, "shadowed": 0, "new": 0, "unreliable": 0, "no_candidate": 0}

        # ---- 1-2. project, classify, match
        pairs = []                                    # (score, map index, polyline id, coverage, mean_dist)
        vis_state = {}
        near_dist = {}                                # map index -> (median px to nearest detected edge, support fraction)
        ctrl_ok = {}                                  # same matcher on a 15 px shifted projection = chance support
        sh_uv, sh_z = [], []                          # projected samples of visible map edges, for the shadow test
        for mi, e in enumerate(edges):
            V = e["vertices_3d"]
            uvV, zV = project(V, K, Rcw, tcw)
            if (zV <= 1e-6).any():
                vis_state[mi] = "outside"
                continue
            X, _ = densify_vertices(V, uvV)
            uv, z = project(X, K, Rcw, tcw)
            inside = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
            if inside.mean() < 0.3 or inside.sum() < tc.min_visible_samples:
                vis_state[mi] = "outside"
                continue
            zj = zs(uv[inside])
            seen = np.isfinite(zj)
            rel = (z[inside] - zj) / np.maximum(zj, 1e-6)
            behind = seen & (rel > tc.visible_depth_rel)
            visible = seen & (np.abs(rel) <= tc.visible_depth_rel)
            if behind.sum() >= 0.5 * max(seen.sum(), 1):
                vis_state[mi] = "occluded"
                continue
            if visible.sum() < tc.min_visible_samples:
                vis_state[mi] = "unreliable"
                continue
            vis_state[mi] = "visible"
            tang = np.gradient(uv, axis=0)
            ang = np.degrees(np.arctan2(tang[:, 1], tang[:, 0])) % 180
            sel = np.flatnonzero(inside)[visible]
            uvs_, zs_, angs_ = uv[sel], z[sel], ang[sel]
            sm = support_match(uvs_, zs_, angs_, fe, mc)
            ctrl_ok[mi] = support_match(uvs_ + np.array([CONTROL_SHIFT_PX, 0.0]), zs_, angs_, fe, mc)["ok"]
            near_dist[mi] = (sm["median_d"], sm["frac"])
            sh_uv.append(uvs_)
            sh_z.append(zs_)
            if sm["ok"]:
                for b, cnt in sm["frags"].items():
                    score = (cnt / len(uvs_)) * (1.0 - 0.5 * min(sm["mean_d"] / mc.gate_px, 1.0))
                    pairs.append((score, mi, b, sm["frac"], sm["mean_d"]))
        pairs.sort(key=lambda p: -p[0])
        claimed: dict[int, int] = {}                  # polyline id -> map index (one owner per detected edge)
        take: dict[int, list] = {}
        for score, mi, b, cov, md in pairs:
            if b not in claimed:
                claimed[b] = mi
                take.setdefault(mi, []).append((b, score))

        # ---- 3-5. accept, fuse, update confidence/status
        absorbed = set()
        for mi, e in enumerate(edges):
            st = vis_state.get(mi, "outside")
            if st != "visible":
                stats[st] += 1
                continue
            stats["visible"] += 1
            gap = j - e["observed_frames"][-1]
            bucket = "1" if gap == 1 else "2" if gap == 2 else "3-5" if gap <= 5 else "6-10" if gap <= 10 else ">10"
            gap_stats.setdefault(bucket, [0, 0, 0, 0])[0] += 1
            gap_stats[bucket][3] += bool(ctrl_ok.get(mi))
            nobs_key = "obs:1 (never fused)" if len(e["obs"]) == 1 else "obs:2+ (fused)"
            for grp in (e["edge_kind"], "shape:" + e["type"], nobs_key):
                row = group_gap.setdefault((grp, "1-2" if gap <= 2 else "3-5" if gap <= 5 else ">5"), [0, 0, 0])
                row[0] += 1
                row[2] += bool(ctrl_ok.get(mi))
            if mi not in take:
                stats["no_candidate"] += 1
                gap_stats[bucket][2] += 1
                DIST_LOG.setdefault(bucket, []).append(near_dist[mi])
            accepted = []
            for b, score in take.get(mi, []):
                took = False
                for rec in polys_here.get(b, []):
                    o = overlap_part(make_obs(rec, lifted[rec["source"]["run"]], Cj, gc, tc), e["vertices_3d"],
                                     e["type"], K, Rcw, tcw, mc.gate_px, tc.min_visible_samples)
                    if o is None:
                        stats["too_little_overlap"] += 1
                    elif consistency_rms(o, e["vertices_3d"], e["type"], focal, gc) <= tc.gate_rms_norm:
                        o["score"] = score
                        accepted.append(o)
                        took = True
                    else:
                        stats["refused_3d"] += 1
                if took:
                    absorbed.add(b)
            if accepted:
                e["obs"].extend(accepted)
                e["vertices_3d"] = fuse(e["obs"], e["type"], tc)
                q = float(np.mean([o["score"] for o in accepted]))
                e["confidence"] = 1.0 - (1.0 - e["confidence"]) * (1.0 - 0.5 * q)
                e["observed_frames"].append(j)
                e["consecutive_missed"] = 0
                if len(e["observed_frames"]) >= tc.verify_min_obs and e["confidence"] >= tc.verify_min_conf:
                    e["status"] = "verified"
                stats["matched"] += 1
                gap_stats[bucket][1] += 1
                for grp in (e["edge_kind"], "shape:" + e["type"], nobs_key):
                    group_gap[(grp, "1-2" if gap <= 2 else "3-5" if gap <= 5 else ">5")][1] += 1
            else:
                e["consecutive_missed"] += 1
                e["missed_total"] += 1
                stats["missed"] += 1

        # ---- 6. new provisional edges from physical detections nobody claimed
        shadow = cKDTree(np.vstack(sh_uv)) if sh_uv else None
        shadow_z = np.concatenate(sh_z) if sh_z else None
        for pid, rr in polys_here.items():
            if pid in absorbed:
                continue
            for rec in rr:
                if not is_physical(rec, tc):
                    continue
                o = make_obs(rec, lifted[rec["source"]["run"]], Cj, gc, tc)
                if shadow is not None:
                    # already represented? sits on a visible map edge in the image AND at about its depth
                    dd, ii = shadow.query(o["uv"])
                    covered = (dd <= mc.gate_px) & (np.abs(o["z"] - shadow_z[ii]) / shadow_z[ii] < 0.1)
                    if covered.mean() >= 0.6:
                        stats["shadowed"] += 1
                        continue
                edges.append({"id": next_id, "type": rec["type"], "edge_kind": rec["edge_kind"], "object_id": None,
                              "status": "provisional", "confidence": rec["confidence"], "origin_frame": j,
                              "observed_frames": [j], "obs": [o], "vertices_3d": o["V"].copy(),
                              "consecutive_missed": 0, "missed_total": 0})
                next_id += 1
                stats["new"] += 1
        stats["map_after"] = len(edges)
        log.append(stats)

    # ---- evaluation ------------------------------------------------------------------------------------------
    held = {"edges": 0, "initial_median_px": None, "fused_median_px": None, "improved_fraction": None}
    e_init, e_fused = [], []
    for e in edges:
        if len(e["obs"]) < 3:
            continue
        last = e["obs"][-1]
        f = frames[last["frame"]]
        Rl, tl = np.asarray(f["R_cw"]), np.asarray(f["t_cw"])
        v0 = fuse(e["obs"][:1], e["type"], tc)
        v1 = fuse(e["obs"][:-1], e["type"], tc)
        d0 = reproj_dist(v0, e["type"], last["uv"], K, Rl, tl)
        d1 = reproj_dist(v1, e["type"], last["uv"], K, Rl, tl)
        if np.isfinite(d0).all() and np.isfinite(d1).all():
            e_init.append(float(np.median(d0)))
            e_fused.append(float(np.median(d1)))
    if e_init:
        a, b = np.array(e_init), np.array(e_fused)
        held = {"edges": len(a), "initial_median_px": float(np.median(a)), "fused_median_px": float(np.median(b)),
                "initial_p90_px": float(np.percentile(a, 90)), "fused_p90_px": float(np.percentile(b, 90)),
                "improved_fraction": float((b < a).mean())}

    # duplicate suspects: another map edge lying on top of this one (same direction, within the 3D fusion radius)
    samples, owner, dirs = [], [], {}
    for e in edges:
        V = e["vertices_3d"]
        n = max(2, int(np.linalg.norm(V[-1] - V[0]) / max(0.01 * np.median(e["obs"][0]["z"]), 1e-6)))
        n = min(n, 30)
        for a, b in zip(V[:-1], V[1:]):
            samples.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
            owner.append(np.full(n, e["id"]))
        dirs[e["id"]] = (V[-1] - V[0]) / max(np.linalg.norm(V[-1] - V[0]), 1e-12)
    S, O = np.vstack(samples), np.concatenate(owner)
    tree = cKDTree(S)
    dup = 0
    for e in edges:
        zr = float(np.median(e["obs"][0]["z"]))
        mine = O == e["id"]
        # lateral accuracy is ~2 px; radius = 3 px worth of 3D distance (parallel frame borders are NOT duplicates)
        hits = tree.query_ball_point(S[mine], r=3.0 * zr / focal)
        cnt: dict[int, int] = {}
        for hh in hits:
            for oid in {O[i] for i in hh if O[i] != e["id"]}:
                if abs(float(dirs[oid] @ dirs[e["id"]])) > 0.95:
                    cnt[oid] = cnt.get(oid, 0) + 1
        if cnt and max(cnt.values()) >= 0.5 * mine.sum():
            dup += 1

    n_obs = np.array([len(e["obs"]) for e in edges])
    verified = [e for e in edges if e["status"] == "verified"]
    res = {"seconds": time.time() - t_start, "map_edges": len(edges), "verified": len(verified),
           "verified_fraction": len(verified) / max(len(edges), 1),
           "observations_per_edge": {"median": float(np.median(n_obs)), "p90": float(np.percentile(n_obs, 90)),
                                     "max": int(n_obs.max())},
           "edges_seen_in_3plus_frames": int((n_obs >= 3).sum()),
           "held_out_last_observation": held,
           "duplicate_suspect_fraction": dup / max(len(edges), 1),
           "match_rate_by_frame_gap": {k: {"visible": v[0], "matched": v[1], "no_2d_candidate": v[2],
                                           "match_rate": v[1] / max(v[0], 1),
                                           "chance_support_rate_shifted_control": v[3] / max(v[0], 1)}
                                       for k, v in gap_stats.items()},
           "totals": {k: int(sum(s[k] for s in log)) for k in
                      ("matched", "missed", "refused_3d", "too_little_overlap", "shadowed", "occluded", "outside", "new")},
           "tracked_kinds": tc.tracked_kinds}
    # JSON-friendly map (drop bulky per-observation samples but keep what later stages need)
    out_edges = [{"id": e["id"], "type": e["type"], "edge_kind": e["edge_kind"], "object_id": None, "status": e["status"],
                  "confidence": round(float(e["confidence"]), 3), "origin_frame": e["origin_frame"],
                  "observed_frames": e["observed_frames"], "vertices_3d": np.round(e["vertices_3d"], 5),
                  "missed_total": e["missed_total"], "z_median": float(np.median(e["obs"][0]["z"])),
                  "observations": [{"frame": o["frame"], "src_edge": o["edge"], "n_samples": o["n_samples"]} for o in e["obs"]]}
                 for e in edges]
    write_json(out_path, {"world": "DA3 first-camera frame, DA3 units", "edges": out_edges, "frame_log": log})
    record_metrics(paths, "stage8_edge_tracking", res)
    plot_log(paths, log, n_obs)

    print(f"[stage8] map: {len(edges)} edges ({len(verified)} verified = {res['verified_fraction']:.0%}), "
          f"{res['edges_seen_in_3plus_frames']} seen in >=3 frames, observations/edge median {res['observations_per_edge']['median']:.0f} "
          f"max {res['observations_per_edge']['max']}, {res['seconds']:.0f}s")
    print(f"[stage8] totals over frames: {res['totals']}")
    if e_init:
        print(f"[stage8] held-out last observation ({held['edges']} edges): reprojection error "
              f"{held['initial_median_px']:.2f}px (first observation only) -> {held['fused_median_px']:.2f}px (multi-view fused); "
              f"improved on {held['improved_fraction']:.0%} of edges")
    print(f"[stage8] duplicate-suspect map edges: {res['duplicate_suspect_fraction']:.0%}")
    res["match_rate_by_kind_and_gap"] = {f"{g}|gap {b}": {"visible": v[0], "matched": v[1], "control": v[2]}
                                         for (g, b), v in sorted(group_gap.items())}
    record_metrics(paths, "stage8_edge_tracking", res)
    for g in ("depth_discontinuity", "crease_candidate", "texture", "shape:straight", "shape:curved", "shape:piecewise",
              "obs:1 (never fused)", "obs:2+ (fused)"):
        cells = [f"gap {b}: {group_gap[(g, b)][1] / max(group_gap[(g, b)][0], 1):.0%} of {group_gap[(g, b)][0]} "
                 f"(ctrl {group_gap[(g, b)][2] / max(group_gap[(g, b)][0], 1):.0%})"
                 for b in ("1-2", "3-5", ">5") if (g, b) in group_gap]
        if cells:
            print(f"[stage8]   {g:20s} " + "; ".join(cells))
    order = ["1", "2", "3-5", "6-10", ">10"]
    print("[stage8] match rate by frames since last seen: " + ", ".join(
        f"{b}: {gap_stats[b][1] / max(gap_stats[b][0], 1):.0%} of {gap_stats[b][0]} "
        f"(shifted control {gap_stats[b][3] / max(gap_stats[b][0], 1):.0%})"
        for b in order if b in gap_stats))


def plot_log(paths: Paths, log: list[dict], n_obs: np.ndarray) -> None:
    fr = [s["frame"] for s in log]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(fr, [s["map_after"] for s in log], "k-", label="map size")
    ax[0].bar(fr, [s["new"] for s in log], color="tab:red", label="new this frame")
    ax[0].set_title("map growth (new edges per frame)")
    ax[0].legend()
    for key, col in (("matched", "tab:green"), ("missed", "tab:orange"), ("refused_3d", "tab:purple")):
        ax[1].plot(fr, [s[key] for s in log], color=col, label=key)
    ax[1].set_title("visible map edges: matched / missed / refused")
    ax[1].legend()
    ax[2].hist(n_obs, bins=np.arange(1, n_obs.max() + 2) - 0.5, color="tab:blue")
    ax[2].set_title("observations per map edge")
    ax[2].set_yscale("log")
    fig.tight_layout()
    fig.savefig(paths.viz / "edge_tracking_log.png", dpi=90)
    plt.close(fig)
