"""Stage 9: connect the persistent 3D edges into a graph: shared vertices, junctions, closed planar loops.

  1. endpoints of different edges that lie within a snap radius AND point in different directions become ONE vertex
     (corner) - two edges leaving a point the same way are overlapping duplicates, not a corner;
  2. a free endpoint that lands on the INTERIOR of another edge (a T-junction) splits that edge there;
  3. a free endpoint is EXTENDED along its tangent by a bounded distance to meet a nearby edge (contours stop short of
     corners by several pixels, so without this almost no loop closes);
  4. closed cycles of up to loop_max_len graph edges are candidate faces; a candidate must be planar to within the depth
     noise, simple (no self-crossing) and minimal (no other vertex inside it);
  5. every planar loop is checked against the DEPTH inside it: a real surface has depth that matches its plane, a loop
     around a doorway or window does not ("opening") and must never become a wall.

The snap radius is depth-dependent: max(gap_px * z / focal, depth_snap_rel * z). Only edges that persisted over several
frames enter (min_views). A control run displaces every edge by random vectors of 2 snap radii and rebuilds the graph;
junctions that appear anyway show how many would form by chance.
"""
from __future__ import annotations

import itertools
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json


# --------------------------------------------------------------------------- small geometry helpers
def snap_tol(z, focal, gc):
    return np.maximum(gc.gap_px * z / focal, gc.depth_snap_rel * z)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class DSU:
    def __init__(self):
        self.p: dict[int, int] = {}

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


def point_to_segments(P, A, B):
    """Closest points of each P (m,3) on each segment A-B (S,3): returns dist (m,S), t (m,S)."""
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-18)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0.0, 1.0)
    Q = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None, :] - Q, axis=2), t


def plane_fit(P):
    c = P.mean(0)
    w, v = np.linalg.eigh(np.cov((P - c).T))
    n = v[:, 0]
    return c, n, float(np.sqrt(np.mean(((P - c) @ n) ** 2)))


def plane_axes(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(n, a))
    return u, np.cross(n, u)


def seg_intersect_2d(p, q, r, s):
    d1, d2 = q - p, s - r
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-15:
        return False
    t = ((r[0] - p[0]) * d2[1] - (r[1] - p[1]) * d2[0]) / den
    u = ((r[0] - p[0]) * d1[1] - (r[1] - p[1]) * d1[0]) / den
    return 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9


def polygon_simple(poly2d):
    n = len(poly2d)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if seg_intersect_2d(poly2d[i], poly2d[(i + 1) % n], poly2d[j], poly2d[(j + 1) % n]):
                return False
    return True


def points_in_polygon(pts, poly):
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cond = (a[1] > pts[:, 1]) != (b[1] > pts[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (b[0] - a[0]) * (pts[:, 1] - a[1]) / (b[1] - a[1]) + a[0]
        inside ^= cond & (pts[:, 0] < xint)
    return inside


# --------------------------------------------------------------------------- graph construction
def build_graph(edges: list[dict], focal: float, gc) -> dict:
    """edges: [{id, P (n,3), z, shape, conf}]. Returns nodes, pieces and bookkeeping counts."""
    N = len(edges)
    for e in edges:
        seg = np.linalg.norm(np.diff(e["P"], axis=0), axis=1)
        e["s"] = np.concatenate([[0.0], np.cumsum(seg)])
        e["L"] = float(e["s"][-1])
        e["tol"] = float(snap_tol(e["z"], focal, gc))

    # endpoints: id 2k = start, 2k+1 = end of edge k; outward tangent points away from the edge
    Pe = np.zeros((2 * N, 3))
    te = np.zeros((2 * N, 3))
    tolE = np.zeros(2 * N)
    owner = np.repeat(np.arange(N), 2)
    for k, e in enumerate(edges):
        P, s = e["P"], e["s"]
        look = min(0.25 * e["L"], 5 * e["tol"])
        i0 = max(1, int(np.searchsorted(s, look)))
        i1 = max(1, int(np.searchsorted(s[-1] - s[::-1], look)))
        Pe[2 * k], Pe[2 * k + 1] = P[0], P[-1]
        te[2 * k] = unit(P[0] - P[min(i0, len(P) - 1)])
        te[2 * k + 1] = unit(P[-1] - P[max(len(P) - 1 - i1, 0)])
        tolE[2 * k] = tolE[2 * k + 1] = e["tol"]

    cos_min = np.cos(np.radians(gc.min_angle_deg))
    dsu = DSU()
    for i in range(2 * N):
        dsu.find(i)
    stats = {"corner_merges": 0, "t_junctions": 0, "extensions": 0}

    # 1. endpoint-endpoint corners
    tree = cKDTree(Pe)
    for a, b in tree.query_pairs(r=float(tolE.max())):
        if owner[a] == owner[b]:
            continue
        if np.linalg.norm(Pe[a] - Pe[b]) <= max(tolE[a], tolE[b]) and float(te[a] @ te[b]) <= cos_min:
            if dsu.find(a) != dsu.find(b):
                stats["corner_merges"] += 1
            dsu.union(a, b)
    comp_edges: dict[int, set] = {}
    for i in range(2 * N):
        comp_edges.setdefault(dsu.find(i), set()).add(owner[i])
    free = [i for i in range(2 * N) if len(comp_edges[dsu.find(i)]) == 1]

    # segment arrays of all edges
    SA, SB, SE, SS, SL = [], [], [], [], []
    for k, e in enumerate(edges):
        SA.append(e["P"][:-1]); SB.append(e["P"][1:]); SE.append(np.full(len(e["P"]) - 1, k))
        SS.append(e["s"][:-1]); SL.append(np.diff(e["s"]))
    SA, SB, SE, SS, SL = map(np.concatenate, (SA, SB, SE, SS, SL))
    SD = unit(SB - SA)

    contact_pos: dict[int, np.ndarray] = {}
    attach: dict[int, list] = {k: [] for k in range(N)}   # edge -> [(arclength, point id)]
    next_pid = 2 * N
    eff_pos = {i: Pe[i] for i in range(2 * N)}            # endpoint positions actually used (extension may move them)

    def attach_to(ep: int, seg: int, t: float, at_pos: np.ndarray, ep_pos: np.ndarray, kind: str):
        nonlocal next_pid
        k = int(SE[seg])
        s = float(SS[seg] + t * SL[seg])
        if s <= edges[k]["tol"] or s >= edges[k]["L"] - edges[k]["tol"]:     # lands on B's end: corner, not a T
            end = 0 if s <= edges[k]["tol"] else 1
            if np.linalg.norm(Pe[2 * k + end] - ep_pos) <= 2 * max(tolE[ep], edges[k]["tol"]):
                dsu.union(ep, 2 * k + end)
                eff_pos[ep] = ep_pos
                stats["corner_merges"] += 1
                return True
            return False
        pid = next_pid
        next_pid += 1
        contact_pos[pid] = at_pos
        dsu.union(ep, pid)
        eff_pos[ep] = ep_pos
        attach[k].append((s, pid))
        stats[kind] += 1
        return True

    def angle_ok(ep, seg):
        return abs(float(te[ep] @ SD[seg])) <= cos_min

    # 2. T-junctions
    free_arr = np.array(free, int)
    if len(free_arr) and gc.use_t_junctions:
        D, T = point_to_segments(Pe[free_arr], SA, SB)
        D[owner[free_arr][:, None] == SE[None, :]] = np.inf
        connected = set()
        for row, ep in enumerate(free_arr):
            j = int(np.argmin(D[row]))
            if D[row, j] <= tolE[ep] and angle_ok(ep, j):
                q = SA[j] + T[row, j] * (SB[j] - SA[j])
                if attach_to(int(ep), j, float(T[row, j]), q, Pe[ep], "t_junctions"):
                    connected.add(int(ep))
        free = [i for i in free if i not in connected]

    # 3. extension of the remaining free endpoints
    K = 10
    for ep in (free if gc.use_extension else []):
        e = edges[owner[ep]]
        Lext = gc.ext_px * e["z"] / focal
        ray = Pe[ep][None] + te[ep][None] * (Lext * np.arange(1, K + 1) / K)[:, None]
        D, T = point_to_segments(ray, SA, SB)
        D[:, SE == owner[ep]] = np.inf
        r, j = np.unravel_index(int(np.argmin(D)), D.shape)
        if D[r, j] <= max(gc.ext_tol_px * e["z"] / focal, 0.5 * tolE[ep]) and angle_ok(ep, j):
            q = SA[j] + T[r, j] * (SB[j] - SA[j])
            attach_to(ep, j, float(T[r, j]), q, ray[r], "extensions")

    # nodes = union-find components; position = mean of their member points
    members: dict[int, list] = {}
    for pid in list(dsu.p):
        members.setdefault(dsu.find(pid), []).append(pid)
    node_of, node_pos = {}, []
    for root, ms in members.items():
        pts = [eff_pos[m] if m < 2 * N else contact_pos[m] for m in ms]
        for m in ms:
            node_of[m] = len(node_pos)
        node_pos.append(np.mean(pts, axis=0))
    node_pos = np.array(node_pos)

    # pieces: each edge cut at its attachment points
    pieces = []
    for k, e in enumerate(edges):
        att = [(0.0, node_of[2 * k])] + sorted(attach[k]) and [(0.0, node_of[2 * k])] + \
              [(s, node_of[pid]) for s, pid in sorted(attach[k])] + [(e["L"], node_of[2 * k + 1])]
        for (s1, n1), (s2, n2) in zip(att[:-1], att[1:]):
            if n1 == n2 or s2 - s1 < 1e-9:
                continue
            inner = e["P"][(e["s"] > s1 + 1e-9) & (e["s"] < s2 - 1e-9)]
            poly = np.vstack([node_pos[n1][None], inner, node_pos[n2][None]])
            pieces.append({"edge": e["id"], "a": int(n1), "b": int(n2), "poly": poly, "z": e["z"], "conf": e["conf"],
                           "shape": e["shape"], "length": float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())})
    deg = np.zeros(len(node_pos), int)
    for p in pieces:
        deg[p["a"]] += 1
        deg[p["b"]] += 1
    return {"node_pos": node_pos, "degree": deg, "pieces": pieces, "stats": stats, "n_edges": N}


# --------------------------------------------------------------------------- loops
def find_loops(g: dict, gc) -> tuple[list[dict], dict]:
    import networkx as nx            # only the (optional) edge-graph stage needs it; the default pipeline imports this module for helpers
    G = nx.Graph()
    best: dict[tuple, dict] = {}
    for p in g["pieces"]:
        key = (min(p["a"], p["b"]), max(p["a"], p["b"]))
        if key not in best or p["length"] < best[key]["length"]:
            best[key] = p
    G.add_edges_from(best.keys())
    pos = g["node_pos"]
    loops, rejected = [], {"not_planar": 0, "too_small": 0, "sliver": 0, "self_crossing": 0, "not_minimal": 0}
    cycles = list(itertools.islice(nx.simple_cycles(G, length_bound=gc.loop_max_len), gc.max_cycles))
    n_cycles = len(cycles)
    for cyc in cycles:
        if len(cyc) < 3:
            continue
        ring_pieces, dense = [], []
        for u, v in zip(cyc, cyc[1:] + cyc[:1]):
            p = best[(min(u, v), max(u, v))]
            ring_pieces.append(p)
            dense.append(p["poly"] if p["a"] == u else p["poly"][::-1])
        D = np.vstack(dense)
        z = float(np.mean([p["z"] for p in ring_pieces]))
        c, n, rms = plane_fit(D)
        if rms > gc.plane_tol_rel * z:
            rejected["not_planar"] += 1
            continue
        u_ax, v_ax = plane_axes(n)
        node2d = np.stack([(pos[list(cyc)] - c) @ u_ax, (pos[list(cyc)] - c) @ v_ax], 1)
        a2 = 0.5 * np.sum(node2d[:, 0] * np.roll(node2d[:, 1], -1) - np.roll(node2d[:, 0], -1) * node2d[:, 1])
        if abs(a2) < (gc.min_area_rel * z) ** 2:
            rejected["too_small"] += 1
            continue
        perim = float(np.linalg.norm(np.diff(np.vstack([D, D[:1]]), axis=0), axis=1).sum())
        if 4 * abs(a2) / max(perim, 1e-12) < gc.min_thickness_rel * z:
            rejected["sliver"] += 1
            continue
        if not polygon_simple(node2d):
            rejected["self_crossing"] += 1
            continue
        others = np.setdiff1d(np.arange(len(pos)), cyc)
        if len(others):
            near_plane = others[np.abs((pos[others] - c) @ n) <= gc.plane_tol_rel * z]
            if len(near_plane):
                p2 = np.stack([(pos[near_plane] - c) @ u_ax, (pos[near_plane] - c) @ v_ax], 1)
                if points_in_polygon(p2, node2d).any():
                    rejected["not_minimal"] += 1
                    continue
        loops.append({"nodes": [int(x) for x in cyc], "pieces_edges": [int(p["edge"]) for p in ring_pieces],
                      "centroid": c, "normal": n, "planarity_rms": rms, "z": z, "area": float(abs(a2)),
                      "ring": D, "n_edges": len(cyc)})
    return loops, {"cycles_enumerated": n_cycles, "rejected": rejected}


def verify_loops(loops: list[dict], frames: dict, K: np.ndarray, paths: Paths, gc) -> None:
    """Classify each planar loop by the depth inside it, seen from its most frontal fully-visible keyframe."""
    zcache: dict[int, np.ndarray] = {}
    for lp in loops:
        c, n = lp["centroid"], lp["normal"]
        best = None
        for k, f in frames.items():
            Rcw, tcw, C = np.asarray(f["R_cw"]), np.asarray(f["t_cw"]), np.asarray(f["t_wc"])
            Xc = lp["ring"] @ Rcw.T + tcw
            if (Xc[:, 2] <= 1e-6).any():
                continue
            uv = (Xc @ K.T)[:, :2] / Xc[:, 2:3]
            h, w = (900, 1600) if False else (int(2 * K[1, 2]), int(2 * K[0, 2]))
            if uv.min() < 8 or uv[:, 0].max() > w - 8 or uv[:, 1].max() > h - 8:
                continue
            v = unit(c - C)
            score = abs(float(n @ v))
            if best is None or score > best[0]:
                best = (score, k, uv, Rcw, tcw, h, w)
        if best is None or best[0] < 0.25:
            lp["support"] = {"class": "unobserved", "frame": None}
            continue
        score, k, uv, Rcw, tcw, h, w = best
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(mask, [np.rint(uv - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)], 1)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(mask)
        if len(ys) < 60:
            lp["support"] = {"class": "too_small_in_image", "frame": int(k)}
            continue
        if k not in zcache:
            zcache[k] = np.load(paths.depth__aligned / f"{paths.frame_name(k)}.npy")
        Z = zcache[k][ys + y0, xs + x0]
        ok = np.isfinite(Z)
        n_c = Rcw @ n
        d_c = float(n_c @ (Rcw @ c + tcw))
        ray = np.stack([(xs + x0 - K[0, 2]) / K[0, 0], (ys + y0 - K[1, 2]) / K[1, 1], np.ones(len(xs))], 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            zp = d_c / (ray @ n_c)
        rel = (Z - zp) / zp
        ok &= np.isfinite(rel) & (zp > 0)
        if ok.sum() < 40:
            lp["support"] = {"class": "no_depth", "frame": int(k)}
            continue
        frac = float((np.abs(rel[ok]) <= gc.support_tol).mean())
        farther = float((rel[ok] > gc.support_tol).mean())
        cls = "supported" if frac >= gc.support_frac else "opening" if farther >= 0.5 else "mismatch"
        # control: the same test against a plane moved gc.depth_check_shift_rel * depth along its normal (either way).
        # If a wrong plane still passes, the loop is too small / too flat for the depth check to tell anything.
        shifted = 0.0
        for sgn in (1.0, -1.0):
            with np.errstate(divide="ignore", invalid="ignore"):
                zs = (d_c + sgn * gc.depth_check_shift_rel * lp["z"]) / (ray @ n_c)
                r2 = (Z - zs) / zs
            ok2 = ok & np.isfinite(r2) & (zs > 0)
            if ok2.sum() >= 40:
                shifted = max(shifted, float((np.abs(r2[ok2]) <= gc.support_tol).mean()))
        lp["support"] = {"class": cls, "frame": int(k), "consistent_fraction": frac, "farther_fraction": farther,
                         "shifted_plane_consistent_fraction": shifted,
                         "discriminative": bool(shifted < gc.support_frac), "frontal_score": score,
                         "pixels": int(ok.sum())}


# --------------------------------------------------------------------------- run
def load_edges(paths: Paths, gc, jitter: float | None = None, focal: float = 1.0, seed: int = 0) -> list[dict]:
    M = read_json(paths.tracking / "edge_map.json")["edges"]
    rng = np.random.default_rng(seed)
    out = []
    for e in M:
        if not (e["status"] == "verified" or len(e["observed_frames"]) >= gc.min_views):
            continue
        P = np.asarray(e["vertices_3d"], float)
        if jitter:
            d = unit(rng.normal(size=3))
            P = P + d * jitter * float(snap_tol(e["z_median"], focal, gc))
        out.append({"id": e["id"], "P": P, "z": e["z_median"], "shape": e["type"], "conf": e["confidence"],
                    "kind": e["edge_kind"]})
    return out


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    gc = cfg.graph
    out_path = paths.tracking / "edge_graph.json"
    if out_path.exists() and not force:
        print("[stage9] edge_graph.json exists, skipping (use --force)")
        return
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    K = np.asarray(next(iter(poses["cameras"].values()))["K"])
    focal = float(K[0, 0])

    edges = load_edges(paths, gc, focal=focal)
    g = build_graph(edges, focal, gc)
    deg = g["degree"]
    ctrl = build_graph(load_edges(paths, gc, jitter=gc.control_shift_tol, focal=focal), focal, gc)
    loops, loop_info = find_loops(g, gc)
    ctrl_loops, ctrl_info = find_loops(ctrl, gc)
    verify_loops(loops, frames, K, paths, gc)

    classes = {}
    for lp in loops:
        classes[lp["support"]["class"]] = classes.get(lp["support"]["class"], 0) + 1
    n_disc = sum(1 for lp in loops if lp["support"].get("discriminative"))
    n_sup_disc = sum(1 for lp in loops if lp["support"]["class"] == "supported" and lp["support"].get("discriminative"))
    degree_hist = {int(d): int((deg == d).sum()) for d in sorted(set(deg.tolist()))}
    ends_connected = 1.0 - (deg == 1).sum() / max(len(deg), 1)

    res = {"edges_in_graph": len(edges), "nodes": int(len(deg)), "graph_edges": len(g["pieces"]),
           "degree_histogram": degree_hist, "junctions_deg3plus": int((deg >= 3).sum()),
           "free_ends_deg1": int((deg == 1).sum()), "operations": g["stats"],
           "control": {"junctions_deg3plus": int((ctrl["degree"] >= 3).sum()), "free_ends_deg1": int((ctrl["degree"] == 1).sum()),
                       "operations": ctrl["stats"], "planar_loops": len(ctrl_loops),
                       "note": f"every edge displaced by {gc.control_shift_tol} snap radii in a random direction"},
           "loops": {"planar_minimal": len(loops), "by_depth_check": classes,
                     "depth_check_discriminative": n_disc, "supported_and_discriminative": n_sup_disc, "search": loop_info,
                     "edges_per_loop": {str(n): sum(l["n_edges"] == n for l in loops) for n in sorted({l['n_edges'] for l in loops})}}}
    record_metrics(paths, "stage9_edge_graph", res)
    write_json(out_path, {
        "world": "DA3 first-camera frame, DA3 units",
        "nodes": [{"id": i, "pos": np.round(p, 5), "degree": int(d)} for i, (p, d) in enumerate(zip(g["node_pos"], deg))],
        "pieces": [{"edge": p["edge"], "a": p["a"], "b": p["b"], "shape": p["shape"], "polyline": np.round(p["poly"], 5),
                    "length": p["length"]} for p in g["pieces"]],
        "loops": [{"nodes": l["nodes"], "edges": l["pieces_edges"], "planarity_rms": l["planarity_rms"], "depth": l["z"],
                   "area": l["area"], "normal": np.round(l["normal"], 4), "ring": np.round(l["ring"], 5),
                   "support": l["support"]} for l in loops]})
    visualize(paths, frames, K, g, loops)
    print(f"[stage9] {len(edges)} edges -> {len(deg)} vertices, {len(g['pieces'])} graph edges; "
          f"operations {g['stats']}; degree histogram {degree_hist}")
    print(f"[stage9] real vs control (shifted edges): junction vertices {res['junctions_deg3plus']} vs {res['control']['junctions_deg3plus']}, "
          f"corner merges {g['stats']['corner_merges']} vs {ctrl['stats']['corner_merges']}, "
          f"T-junctions {g['stats']['t_junctions']} vs {ctrl['stats']['t_junctions']}, "
          f"extensions {g['stats']['extensions']} vs {ctrl['stats']['extensions']}")
    print(f"[stage9] planar minimal loops {len(loops)} (control {len(ctrl_loops)}); depth check: {classes}; "
          f"of these the check can actually tell a wrong plane apart for {n_disc}, supported AND discriminative: {n_sup_disc}; "
          f"search {loop_info}")


def visualize(paths: Paths, frames: dict, K: np.ndarray, g: dict, loops: list[dict]) -> None:
    colors = {"supported": (0, 200, 0), "opening": (0, 0, 255), "mismatch": (0, 165, 255), "unobserved": (160, 160, 160)}
    ids = sorted(frames)
    for k in (ids[len(ids) // 4], ids[len(ids) // 2]):
        f = frames[k]
        img = cv2.imread(str(paths.keyframes / f"{paths.frame_name(k)}.png"))
        vis = (img * 0.5).astype(np.uint8)
        over = vis.copy()

        def proj(P):
            Xc = P @ np.asarray(f["R_cw"]).T + np.asarray(f["t_cw"])
            return (Xc @ K.T)[:, :2] / Xc[:, 2:3], Xc[:, 2]

        for lp in loops:
            uv, z = proj(lp["ring"])
            if (z > 1e-6).all() and np.isfinite(uv).all() and np.abs(uv).max() < 1e5:
                cv2.fillPoly(over, [np.rint(uv).astype(np.int32).reshape(-1, 1, 2)], colors[lp["support"]["class"]])
        vis = cv2.addWeighted(over, 0.45, vis, 0.55, 0)
        for p in g["pieces"]:
            uv, z = proj(p["poly"])
            if (z > 1e-6).all() and np.abs(uv).max() < 1e5:
                cv2.polylines(vis, [np.rint(uv).astype(np.int32).reshape(-1, 1, 2)], False, (255, 200, 60), 2, cv2.LINE_AA)
        uvn, zn = proj(g["node_pos"])
        for (x, y), z, d in zip(uvn, zn, g["degree"]):
            if z > 1e-6 and 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                col = (0, 0, 255) if d >= 3 else (255, 255, 255) if d == 2 else (140, 140, 140)
                cv2.circle(vis, (int(x), int(y)), 6 if d >= 3 else 4, col, -1)
        cv2.imwrite(str(paths.viz / f"graph_overlay_{k:04d}.png"), vis)
