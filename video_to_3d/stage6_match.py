"""Stage 6 (V1 endpoint): two-frame edge matching -> persistent edge identities.

For each adjacent keyframe pair (k -> j) every lifted 3D edge of frame k is projected into frame j and compared with the
2D edges detected there. A projected edge A and a detected edge B match when
  * enough sampled points of A fall within gate_px of B (coverage = share of A's visible samples),
  * their mean distance is small, their directions agree (PCA axis angle), and
  * the depth of A seen from camera j agrees with B's own lifted depth.
Matches are links between (frame, 2D polyline id) nodes; connected components of that link graph are the persistent
edge tracks (a fragmented contour in one frame legitimately maps several nodes onto one track). Nothing is merged or
deleted geometrically in V1 - this stage only assigns identity.

Honest scoring: the match rate on "strong" edges is reported next to a control in which the projection is shifted
15 px sideways, because edge maps are dense and some matches happen by chance.
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json
from .stage4_edges import COLORS
from .stage5_lift import Sampler, project

CONTROL_SHIFT_PX = 15.0
STEP_PX = 6.0


def _interp_along(arr: np.ndarray, n: np.ndarray) -> np.ndarray:
    parts = [arr[i] + (arr[i + 1] - arr[i]) * np.linspace(0, 1, m, endpoint=False).reshape(-1, *[1] * (arr.ndim - 1))
             for i, m in enumerate(n)]
    return np.concatenate(parts + [arr[-1:]])


def densify(e: dict, step: float = STEP_PX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense samples along a lifted polyline: world points, source-image uv, source-camera depth."""
    P, uv, z = (np.asarray(e[k], float) for k in ("points", "uv", "z"))
    n = np.maximum(1, np.round(np.linalg.norm(np.diff(uv, axis=0), axis=1) / step).astype(int))
    return _interp_along(P, n), _interp_along(uv, n), _interp_along(z, n)


def path_length(uv: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(uv, axis=0), axis=1).sum())


class FrameEdges:
    """Nearest-edge lookup over a frame's detected 2D polylines plus their lifted depth."""

    def __init__(self, polys2d: list[dict], lifted: list[dict], h: int, w: int):
        idmap = np.zeros((h, w), np.uint16)
        for e in polys2d:
            pts = np.rint(np.asarray(e["points"])).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(idmap, [pts], bool(e["closed"]), int(e["id"]) + 1, 1)
        zy, zx = np.nonzero(idmap)
        self.xy = np.stack([zx, zy], 1).astype(np.float64)
        self.pid = idmap[zy, zx].astype(np.int64) - 1
        self.tree = cKDTree(self.xy)
        uvs, zs, ids = [], [], []
        for e in lifted:
            _, uv, z = densify(e, 4.0)
            uvs.append(uv)
            zs.append(z)
            ids.append(np.full(len(uv), e["id"]))
        if uvs:
            self.l_uv, self.l_z, self.l_id = np.concatenate(uvs), np.concatenate(zs), np.concatenate(ids)
            self.l_tree = cKDTree(self.l_uv)
        else:
            self.l_tree = None


def _axis(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts - pts.mean(0)
    w, v = np.linalg.eigh(c.T @ c / max(len(pts), 1))
    return v[:, 1], float(np.sqrt(max(w[1], 0.0)))


def match_one(uv: np.ndarray, z: np.ndarray, fe: FrameEdges, mc) -> list[dict]:
    """Candidates in frame j for one projected edge (visible samples only). Returns passing candidates."""
    d, i = fe.tree.query(uv)
    near = d <= mc.gate_px
    if near.sum() < mc.min_samples:
        return []
    ids = fe.pid[i]
    out = []
    for b in np.unique(ids[near]):
        m = near & (ids == b)
        n = int(m.sum())
        if n < mc.min_samples or n / len(uv) < mc.min_coverage:
            continue
        mean_d = float(d[m].mean())
        if mean_d > mc.max_mean_dist_px:
            continue
        ax_a, spread_a = _axis(uv[m])
        nearest_b = np.unique(fe.xy[i[m]], axis=0)
        ax_b, spread_b = _axis(nearest_b) if len(nearest_b) >= 3 else (ax_a, 0.0)
        if min(spread_a, spread_b) < 2.0:       # too short to define a direction
            continue
        angle = float(np.degrees(np.arccos(np.clip(abs(ax_a @ ax_b), 0, 1))))
        if angle > mc.max_angle_deg:
            continue
        rel = np.nan
        if fe.l_tree is not None:
            dl, il = fe.l_tree.query(fe.xy[i[m]])
            ok = (dl <= 6.0) & (fe.l_id[il] == b)
            if ok.any():
                zb = fe.l_z[il[ok]]
                rel = float(np.median(np.abs(z[m][ok] - zb) / zb))
        if np.isfinite(rel) and rel > mc.max_depth_rel_err:
            continue
        out.append({"b": int(b), "n": n, "coverage": n / len(uv), "mean_dist": mean_d, "angle": angle,
                    "depth_rel_err": None if not np.isfinite(rel) else rel})
    return out


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    mc = cfg.match
    out_path = paths.tracking / "edge_tracks.json"
    if out_path.exists() and not force:
        print("[stage6] edge_tracks.json exists, skipping (use --force)")
        return
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    ids = sorted(frames)
    K = np.asarray(next(iter(poses["cameras"].values()))["K"])
    h = w = 0
    polys2d, lifted, Zs = {}, {}, {}
    for k in ids:
        nm = paths.frame_name(k)
        d2 = read_json(paths.edges__polylines / f"{nm}.json")
        h, w = d2["height"], d2["width"]
        polys2d[k] = d2["polylines"]
        lifted[k] = read_json(paths.edges__polylines3d / f"{nm}.json")["polylines3d"]
        Zs[k] = Sampler(np.load(paths.depth__aligned / f"{nm}.npy"))
    fedges = {k: FrameEdges(polys2d[k], lifted[k], h, w) for k in ids}

    dsu = DSU()
    node_type, link_scores, pair_rows = {}, [], []
    elig = {t: 0 for t in COLORS}
    hit = {t: 0 for t in COLORS}
    hit_ctrl = {t: 0 for t in COLORS}
    dup_b = dup_total = 0
    for k in ids:
        j = k + 1
        if j not in frames:
            continue
        fj = frames[j]
        b_to_a = {}
        for e in lifted[k]:
            X, uvk, _ = densify(e)
            node_type[(k, e["id"])] = e["type"]
            dsu.find((k, e["id"]))
            uv, z = project(X, K, np.asarray(fj["R_cw"]), np.asarray(fj["t_cw"]))
            inside = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
            if not inside.any():
                continue
            zj = Zs[j](uv[inside])
            vis_in = np.isfinite(zj) & (np.abs(z[inside] - zj) / np.maximum(zj, 1e-6) < 0.15)
            uvv, zv = uv[inside][vis_in], z[inside][vis_in]
            strong = path_length(uvk) >= mc.strong_min_length_px
            eligible = strong and len(uvv) >= mc.min_samples and len(uvv) / len(X) >= 0.5
            cands = match_one(uvv, zv, fedges[j], mc) if len(uvv) >= mc.min_samples else []
            if eligible:
                t = e["type"]
                elig[t] += 1
                hit[t] += bool(cands)
                shifted = uvv + np.array([CONTROL_SHIFT_PX, 0.0])
                hit_ctrl[t] += bool(match_one(shifted, zv, fedges[j], mc))
            for c in cands:
                dsu.union((k, e["id"]), (j, c["b"]))
                node_type.setdefault((j, c["b"]), polys2d[j][c["b"]]["type"])
                b_to_a.setdefault(c["b"], set()).add(e["id"])
                link_scores.append((k, j, e["id"], c["b"], c["coverage"], c["mean_dist"], c["angle"]))
        dup_total += len(b_to_a)
        dup_b += sum(len(s) > 1 for s in b_to_a.values())

    comps = {}
    for node in list(dsu.p):
        comps.setdefault(dsu.find(node), []).append(node)
    tracks = []
    for tid, nodes in enumerate(sorted(comps.values(), key=lambda n: (-len({f for f, _ in n}), n[0]))):
        fr_set = sorted({f for f, _ in nodes})
        types = [node_type.get(n, "unknown") for n in nodes]
        tracks.append({"id": tid, "n_frames": len(fr_set), "frames": fr_set,
                       "type": max(set(types), key=types.count),
                       "observations": [{"frame": f, "polyline": p} for f, p in sorted(nodes)]})
    multi = [t for t in tracks if t["n_frames"] >= 2]
    write_json(out_path, {"tracks": tracks, "links": [list(map(float, s)) for s in link_scores]})

    rate = {t: hit[t] / elig[t] for t in COLORS if elig[t]}
    ctrl = {t: hit_ctrl[t] / elig[t] for t in COLORS if elig[t]}
    n_e, n_h, n_c = sum(elig.values()), sum(hit.values()), sum(hit_ctrl.values())
    lens = np.array([t["n_frames"] for t in multi]) if multi else np.zeros(1)
    res = {"strong_eligible": n_e, "strong_match_rate": n_h / max(n_e, 1), "control_match_rate": n_c / max(n_e, 1),
           "by_type": {t: {"eligible": elig[t], "match_rate": rate.get(t), "control_rate": ctrl.get(t)} for t in COLORS},
           "duplicate_suspect_fraction": dup_b / max(dup_total, 1),
           "tracks_total": len(tracks), "tracks_multi_frame": len(multi),
           "track_length_frames": {"median": float(np.median(lens)), "p90": float(np.percentile(lens, 90)),
                                   "max": int(lens.max())},
           "nodes": len(dsu.p), "gate": {"gate_px": mc.gate_px, "angle_deg": mc.max_angle_deg,
                                         "depth_rel": mc.max_depth_rel_err}}
    record_metrics(paths, "stage6_match", res)
    visualize(paths, frames, ids, polys2d, comps, dsu)
    print(f"[stage6] strong edges matched into the next frame: {res['strong_match_rate']:.0%} "
          f"(control {res['control_match_rate']:.0%}) of {n_e} eligible")
    for t in COLORS:
        if t in rate:
            print(f"[stage6]   {t:20s}: {rate[t]:.0%} (control {ctrl[t]:.0%}) of {elig[t]}")
    print(f"[stage6] tracks: {len(multi)} span >=2 frames (median {res['track_length_frames']['median']:.0f}, "
          f"p90 {res['track_length_frames']['p90']:.0f}, max {res['track_length_frames']['max']} frames); "
          f"duplicate-suspect targets {res['duplicate_suspect_fraction']:.0%}")


def _color(tid: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((tid * 0.61803) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def visualize(paths: Paths, frames, ids, polys2d, comps, dsu) -> None:
    comp_id = {}
    for ci, nodes in enumerate(comps.values()):
        if len({f for f, _ in nodes}) >= 2:
            for n in nodes:
                comp_id[n] = ci
    for k in (ids[len(ids) // 3], ids[2 * len(ids) // 3]):
        j = k + 1
        if j not in frames:
            continue
        sides = []
        for f in (k, j):
            img = cv2.imread(str(paths.keyframes / f"{paths.frame_name(f)}.png"))
            vis = (img * 0.4).astype(np.uint8)
            for e in polys2d[f]:
                pts = np.rint(np.asarray(e["points"])).astype(np.int32).reshape(-1, 1, 2)
                c = comp_id.get((f, e["id"]))
                cv2.polylines(vis, [pts], bool(e["closed"]), _color(c) if c is not None else (90, 90, 90),
                              3 if c is not None else 1, cv2.LINE_AA)
            sides.append(cv2.resize(vis, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(paths.viz / f"matches_{k:04d}_{j:04d}.png"), np.hstack(sides))
