"""Stage 10: surfaces from depth-supported planes and exact closed-loop faces.

  fused depth cloud (all keyframes, with normals)
    -> sequential normal-guided RANSAC planes -> per-plane occupancy raster -> connected patches
    -> loop faces (stage 9, depth-supported) as exact polygons that override coplanar plane cells
    -> meshes (rectangle-merged raster for planes, ear-clipped polygons for loops), vertex colours, PLY/GLB export

Nothing is invented: a plane patch exists only where depth points support it, and the depth that no plane explains is
exported separately as 'unexplained' points. Curved surfaces are NOT built in this version.

Scoring (stage-10 gate): planes are fitted on the FIRST half of the keyframes and scored on the SECOND half, against the
observed depth there. Each covered pixel is "agree" (surface depth within eval_tol of observed), "occluded" (surface lies
behind what is seen - it is simply hidden) or "contradiction" (surface floats in FRONT of observed depth - a real error).
The same scoring with every plane moved eval_shift_rel*depth along its normal is the control: it must score worse.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json
from .stage9_edge_graph import plane_axes, points_in_polygon, unit


# --------------------------------------------------------------------------- fused depth cloud
def depth_cloud(paths: Paths, frames: dict, K: np.ndarray, ids: list[int], sc) -> dict:
    P, N, Zc, Col, Src = [], [], [], [], []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for k in ids:
        nm = paths.frame_name(k)
        Z = np.load(paths.depth__aligned / f"{nm}.npy")
        conf = np.load(paths.depth__confidence / f"{nm}.npy").astype(np.float32)
        img = cv2.imread(str(paths.keyframes / f"{nm}.png"))
        h, w = Z.shape
        ys, xs = np.mgrid[sc.stride // 2:h:sc.stride, sc.stride // 2:w:sc.stride]
        z = Z[ys, xs]
        Xc = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], -1)
        dx = Xc[1:-1, 2:] - Xc[1:-1, :-2]
        dy = Xc[2:, 1:-1] - Xc[:-2, 1:-1]
        zi = z[1:-1, 1:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = (np.isfinite(dx).all(-1) & np.isfinite(dy).all(-1) & np.isfinite(zi)
                  & (np.abs(z[1:-1, 2:] - z[1:-1, :-2]) / zi < 0.06) & (np.abs(z[2:, 1:-1] - z[:-2, 1:-1]) / zi < 0.06)
                  & (conf[ys, xs][1:-1, 1:-1] >= sc.conf_min))
        n = unit(np.cross(dx, dy))
        c = Xc[1:-1, 1:-1]
        n = np.where(((n * c).sum(-1) > 0)[..., None], -n, n)          # face the camera
        f = frames[k]
        Rwc, C = np.asarray(f["R_wc"]), np.asarray(f["t_wc"])
        P.append((c[ok] @ Rwc.T + C))
        N.append(n[ok] @ Rwc.T)
        Zc.append(zi[ok])
        Col.append(img[ys[1:-1, 1:-1][ok], xs[1:-1, 1:-1][ok]][:, ::-1])
        Src.append(np.full(int(ok.sum()), k))
    return {"P": np.vstack(P), "N": np.vstack(N), "z": np.concatenate(Zc), "col": np.vstack(Col),
            "src": np.concatenate(Src)}


# --------------------------------------------------------------------------- planes
def find_planes(cl: dict, sc, seed: int = 0) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    P, N, Z = cl["P"], cl["N"], cl["z"]
    remaining = np.ones(len(P), bool)
    cos_t = np.cos(np.radians(sc.normal_deg))
    min_pts = sc.min_support_frac * len(P)
    planes = []
    for _ in range(sc.max_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_pts:
            break
        score_set = rng.choice(idx, size=min(50000, len(idx)), replace=False)
        Q, Nq, tq = P[score_set], N[score_set], sc.plane_tol_rel * Z[score_set]
        best, best_cnt = None, 0
        for c in rng.choice(idx, size=min(sc.ransac_iters, len(idx)), replace=False):
            n = N[c]
            m = (np.abs(Q @ n - P[c] @ n) <= tq) & (Nq @ n >= cos_t)
            cnt = int(m.sum())
            if cnt > best_cnt:
                best, best_cnt = (n.copy(), float(P[c] @ n)), cnt
        if best is None:
            break
        n, off = best
        for _ in range(3):                                               # least-squares refinement on all inliers
            m = remaining & (np.abs(P @ n - off) <= sc.plane_tol_rel * Z) & (N @ n >= cos_t)
            if m.sum() < 10:
                break
            cen = P[m].mean(0)
            w, v = np.linalg.eigh(np.cov((P[m] - cen).T))
            n = v[:, 0]
            if (N[m] @ n).mean() < 0:
                n = -n
            off = float(cen @ n)
        m = remaining & (np.abs(P @ n - off) <= sc.plane_tol_rel * Z) & (N @ n >= cos_t)
        if m.sum() < min_pts:
            break
        planes.append({"n": n, "off": off, "idx": np.flatnonzero(m), "support": int(m.sum())})
        remaining &= ~m
    return planes, remaining


def mask_to_rects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy merge of an occupancy raster into rectangles (r0, r1, c0, c1), end-exclusive."""
    rects, active = [], {}
    H, W = mask.shape
    for r in range(H + 1):
        runs = set()
        if r < H:
            row = np.concatenate([[0], mask[r].astype(np.int8), [0]])
            d = np.diff(row)
            runs = set(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
        for run in list(active):
            if run not in runs:
                rects.append((active.pop(run), r, run[0], run[1]))
        for run in runs:
            active.setdefault(run, r)
    return rects


def make_patches(plane: dict, cl: dict, sc, kind: str = "plane") -> list[dict]:
    n, off = plane["n"], plane["off"]
    idx = plane["idx"]
    X, Zi = cl["P"][idx], cl["z"][idx]
    p0 = X.mean(0)
    p0 = p0 - n * (p0 @ n - off)
    u, v = plane_axes(n)
    a, b = (X - p0) @ u, (X - p0) @ v
    cell = sc.cell_rel * float(np.median(Zi))
    a0, b0 = float(a.min()) - cell, float(b.min()) - cell
    W, H = int((a.max() - a0) / cell) + 2, int((b.max() - b0) / cell) + 2
    mask = np.zeros((H, W), np.uint8)
    mask[((b - b0) / cell).astype(int), ((a - a0) / cell).astype(int)] = 1
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=2)
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for c in range(1, n_cc):
        if stats[c, cv2.CC_STAT_AREA] < sc.min_cells:
            continue
        m = (lab == c)
        pts_in = (lab[((b - b0) / cell).astype(int), ((a - a0) / cell).astype(int)] == c)
        out.append({"kind": kind, "n": n, "off": off, "p0": p0, "u": u, "v": v, "a0": a0, "b0": b0, "cell": cell,
                    "mask": m, "cells": int(m.sum()), "area": float(m.sum() * cell * cell), "points": int(pts_in.sum()),
                    "z": float(np.median(Zi))})
    return out


def loop_patch(loop: dict, sc) -> dict:
    """A loop's exact polygon as a plane patch with a FINE raster (used for scoring and for overriding plane cells)."""
    ring = np.asarray(loop["ring"], float)
    n = np.asarray(loop["normal"], float)
    p0 = ring.mean(0)
    u, v = plane_axes(n)
    a, b = (ring - p0) @ u, (ring - p0) @ v
    z = float(loop["depth"])
    cell = 0.004 * z
    a0, b0 = float(a.min()) - cell, float(b.min()) - cell
    W, H = int((a.max() - a0) / cell) + 2, int((b.max() - b0) / cell) + 2
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.stack([(a - a0) / cell, (b - b0) / cell], 1).round().astype(np.int32).reshape(-1, 1, 2)], 1)
    return {"kind": "loop", "n": n, "off": float(p0 @ n), "p0": p0, "u": u, "v": v, "a0": a0, "b0": b0, "cell": cell,
            "mask": mask > 0, "cells": int(mask.sum()), "area": float(loop["area"]), "points": 0, "z": z,
            "ring2d": np.stack([a, b], 1), "ring3d": ring}


def cell_centers(p: dict) -> np.ndarray:
    H, W = p["mask"].shape
    rr, cc = np.nonzero(p["mask"])
    return p["p0"] + ((p["a0"] + (cc + 0.5) * p["cell"])[:, None]) * p["u"] + ((p["b0"] + (rr + 0.5) * p["cell"])[:, None]) * p["v"]


def remove_loop_overlap(plane_patches: list[dict], loop_patches: list[dict], sc) -> int:
    """Delete plane cells that a coplanar loop face already covers exactly (the loop's boundary is the better one)."""
    removed = 0
    for pp in plane_patches:
        for lp in loop_patches:
            if abs(float(pp["n"] @ lp["n"])) < np.cos(np.radians(10)):
                continue
            if abs(float(lp["p0"] @ pp["n"]) - pp["off"]) > sc.plane_tol_rel * lp["z"]:
                continue
            rr, cc = np.nonzero(pp["mask"])
            X = pp["p0"] + ((pp["a0"] + (cc + 0.5) * pp["cell"])[:, None]) * pp["u"] + ((pp["b0"] + (rr + 0.5) * pp["cell"])[:, None]) * pp["v"]
            q = np.stack([(X - lp["p0"]) @ lp["u"], (X - lp["p0"]) @ lp["v"]], 1)
            inside = points_in_polygon(q, lp["ring2d"])
            if inside.any():
                pp["mask"][rr[inside], cc[inside]] = False
                removed += int(inside.sum())
    return removed


# --------------------------------------------------------------------------- meshing
def ear_clip(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping for a simple 2D polygon (any orientation). Returns index triples."""
    n = len(poly)
    idx = list(range(n))
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        idx.reverse()
    tris, guard = [], 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        for i in range(len(idx)):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            pa, pb, pc = poly[a], poly[b], poly[c]
            if (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]) <= 1e-12:
                continue
            others = [j for j in idx if j not in (a, b, c)]
            if others:
                q = poly[others]
                d1 = (pb[0] - pa[0]) * (q[:, 1] - pa[1]) - (pb[1] - pa[1]) * (q[:, 0] - pa[0])
                d2 = (pc[0] - pb[0]) * (q[:, 1] - pb[1]) - (pc[1] - pb[1]) * (q[:, 0] - pb[0])
                d3 = (pa[0] - pc[0]) * (q[:, 1] - pc[1]) - (pa[1] - pc[1]) * (q[:, 0] - pc[0])
                if ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)).any():
                    continue
            tris.append((a, b, c))
            idx.pop(i)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def patch_mesh(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (m,3) and triangles (t,3) of a patch."""
    if p["kind"] == "loop":
        ring2d = p["ring2d"]
        keep = np.arange(0, len(ring2d), max(1, len(ring2d) // 48))
        poly = ring2d[keep]
        tris = ear_clip(poly)
        V = p["p0"] + poly[:, :1] * p["u"] + poly[:, 1:2] * p["v"]
        return V, np.array(tris, int).reshape(-1, 3)
    V, F = [], []
    for r0, r1, c0, c1 in mask_to_rects(p["mask"]):
        corners = [(c0, r0), (c1, r0), (c1, r1), (c0, r1)]
        base = len(V)
        for c, r in corners:
            V.append(p["p0"] + (p["a0"] + c * p["cell"]) * p["u"] + (p["b0"] + r * p["cell"]) * p["v"])
        F += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(V), np.array(F, int).reshape(-1, 3)


# --------------------------------------------------------------------------- scoring
def render_depth(patches: list[dict], R_wc, C, K, h: int, w: int, stride: int, shift_rel: float = 0.0):
    """Nearest patch hit per pixel: returns (depth along the ray, index of the patch hit; -1 where none)."""
    ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    d_cam = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, dtype=float)], -1).reshape(-1, 3)
    d_w = d_cam @ R_wc.T
    zbuf = np.full(len(d_w), np.inf)
    pid = np.full(len(d_w), -1)
    for pi, p in enumerate(patches):
        off = p["off"] + shift_rel * p["z"]
        denom = d_w @ p["n"]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (off - C @ p["n"]) / denom
        okp = np.isfinite(t) & (t > 0.02)
        X = C + t[:, None] * d_w
        ci = np.floor(((X - p["p0"]) @ p["u"] - p["a0"]) / p["cell"]).astype(int)
        ri = np.floor(((X - p["p0"]) @ p["v"] - p["b0"]) / p["cell"]).astype(int)
        H, W = p["mask"].shape
        ins = okp & (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
        hit = np.zeros(len(t), bool)
        hit[ins] = p["mask"][ri[ins], ci[ins]]
        better = hit & (t < zbuf)
        zbuf[better] = t[better]
        pid[better] = pi
    return zbuf.reshape(ys.shape), pid.reshape(ys.shape)


def score_surfaces(patches: list[dict], frames: dict, ids: list[int], K, paths: Paths, sc, shift_rel: float = 0.0) -> dict:
    agree = occl = contra = covered = valid_px = 0
    per_frame = []
    for k in ids:
        nm = paths.frame_name(k)
        Z = np.load(paths.depth__aligned / f"{nm}.npy")
        conf = np.load(paths.depth__confidence / f"{nm}.npy").astype(np.float32)
        h, w = Z.shape
        f = frames[k]
        C = np.asarray(f["t_wc"])
        zs, pid = render_depth(patches, np.asarray(f["R_wc"]), C, K, h, w, sc.eval_stride, shift_rel)
        ys, xs = np.mgrid[sc.eval_stride // 2:h:sc.eval_stride, sc.eval_stride // 2:w:sc.eval_stride]
        zo, cf = Z[ys, xs], conf[ys, xs]
        valid = np.isfinite(zo) & (cf >= sc.conf_min)
        cov = valid & np.isfinite(zs)
        # Judge by the OBSERVED 3D point's distance to the plane, not by depth along the ray: along a grazing plane
        # (floors) a tiny normal offset becomes a huge ray-depth difference and would be misread as a contradiction.
        d_w = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones(xs.shape)], -1) @ np.asarray(f["R_wc"]).T
        Xo = C + zo[..., None] * d_w
        nrm = np.stack([p["n"] for p in patches])[np.maximum(pid, 0)]
        off = np.array([p["off"] + shift_rel * p["z"] for p in patches])[np.maximum(pid, 0)]
        with np.errstate(invalid="ignore"):
            dn = np.abs((Xo * nrm).sum(-1) - off) / zo
        a_ = cov & (dn <= sc.eval_tol)
        o_ = cov & (dn > sc.eval_tol) & (zs > zo)          # observed surface is in front of the plane: plane is hidden
        c_ = cov & (dn > sc.eval_tol) & (zs <= zo)         # plane in front of the observed surface and not on it: wrong
        valid_px += int(valid.sum()); covered += int(cov.sum())
        agree += int(a_.sum()); occl += int(o_.sum()); contra += int(c_.sum())
        per_frame.append({"frame": k, "coverage": float(cov.sum() / max(valid.sum(), 1))})
    return {"valid_pixels": valid_px, "coverage_of_valid_depth": covered / max(valid_px, 1),
            "of_covered": {"agree": agree / max(covered, 1), "occluded": occl / max(covered, 1),
                           "contradiction": contra / max(covered, 1)},
            "frames": len(ids)}


def evidence_gate(patches: list[dict], frames: dict, ids: list[int], K, paths: Paths, sc) -> tuple[list[dict], list[dict]]:
    """Score every patch on its own against all given frames; keep only patches whose covered pixels mostly agree and
    rarely contradict. Returns (kept, rejected-with-reason)."""
    kept, rejected = [], []
    for i, p in enumerate(patches):
        r = score_surfaces([p], frames, ids, K, paths, sc)
        px = r["coverage_of_valid_depth"] * r["valid_pixels"]
        oc = r["of_covered"]
        seen = oc["agree"] + oc["contradiction"]          # a plane hidden behind furniture is not wrong, just unseen
        agree_visible = oc["agree"] / max(seen, 1e-9)
        p["gate"] = {"covered_px": int(px), "visible_px": int(px * seen), "agree_of_visible": agree_visible,
                     "contradiction_of_covered": oc["contradiction"], "hidden_of_covered": oc["occluded"]}
        if px * seen < sc.gate_min_px:
            rejected.append({"patch": i, "kind": p["kind"], "reason": "too few visible pixels to judge", **p["gate"]})
        elif agree_visible < sc.gate_min_agree or oc["contradiction"] > sc.gate_max_contra:
            rejected.append({"patch": i, "kind": p["kind"], "reason": "inconsistent with observed depth", **p["gate"]})
        else:
            kept.append(p)
    return kept, rejected


# --------------------------------------------------------------------------- run
def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    sc = cfg.surface
    out_json = paths.tracking / "surfaces.json"
    if out_json.exists() and not force:
        print("[stage10] surfaces.json exists, skipping (use --force)")
        return
    (paths.root / "geometry").mkdir(exist_ok=True)
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    K = np.asarray(next(iter(poses["cameras"].values()))["K"])
    ids = sorted(frames)
    half = len(ids) // 2
    graph = read_json(paths.tracking / "edge_graph.json")
    loops = [l for l in graph["loops"] if l["support"]["class"] in sc.loop_classes]

    # ---- gate: fit on the first half, score on the second half (and the control with shifted planes)
    cl_a = depth_cloud(paths, frames, K, ids[:half], sc)
    planes_a, _ = find_planes(cl_a, sc)
    patches_a = [p for pl in planes_a for p in make_patches(pl, cl_a, sc)]
    held = score_surfaces(patches_a, frames, ids[half:], K, paths, sc)
    ctrl = score_surfaces(patches_a, frames, ids[half:], K, paths, sc, shift_rel=sc.eval_shift_rel)
    ctrl2 = score_surfaces(patches_a, frames, ids[half:], K, paths, sc, shift_rel=-sc.eval_shift_rel)
    # the evidence gate, applied with the FIT frames only, then scored on the unseen half
    kept_a, rej_a = evidence_gate(patches_a, frames, ids[:half], K, paths, sc)
    held_gated = score_surfaces(kept_a, frames, ids[half:], K, paths, sc) if kept_a else None
    gate = {"fit_frames": f"{ids[0]}..{ids[half - 1]}", "scored_frames": f"{ids[half]}..{ids[-1]}",
            "planes_fitted": len(planes_a), "patches": len(patches_a), "held_out": held,
            "control_plane_shifted_plus": ctrl, "control_plane_shifted_minus": ctrl2,
            "patches_kept_by_evidence_gate": len(kept_a), "patches_rejected_by_gate": len(rej_a),
            "held_out_after_gate": held_gated}

    # ---- final surfaces from all frames
    cl = depth_cloud(paths, frames, K, ids, sc)
    planes, remaining = find_planes(cl, sc)
    plane_patches = [p for pl in planes for p in make_patches(pl, cl, sc)]
    loop_patches = [loop_patch(l, sc) for l in loops]
    removed = remove_loop_overlap(plane_patches, loop_patches, sc)
    plane_patches = [p for p in plane_patches if p["mask"].sum() >= sc.min_cells]
    built_planes, rejected = evidence_gate(plane_patches, frames, ids, K, paths, sc)
    for lp in loop_patches:                           # loop faces were already depth-checked in stage 9; record their score
        lp["gate"] = {}
    patches = built_planes + loop_patches
    in_sample = score_surfaces(patches, frames, ids, K, paths, sc)

    # ---- meshes, colours, export
    import trimesh
    Vs, Fs, Cs = [], [], []
    meta = []
    for pi, p in enumerate(patches):
        V, F = patch_mesh(p)
        if len(F) == 0:
            continue
        # colour: sample the most frontal keyframe at each vertex
        col = np.full((len(V), 3), 160, np.uint8)
        best_f, best_s = None, -1
        for k in ids:
            f = frames[k]
            C = np.asarray(f["t_wc"])
            s = abs(float(unit((p["p0"] - C)) @ p["n"]))
            if s > best_s:
                best_f, best_s = k, s
        f = frames[best_f]
        Xc = V @ np.asarray(f["R_cw"]).T + np.asarray(f["t_cw"])
        uv = (Xc @ K.T)[:, :2] / np.maximum(Xc[:, 2:3], 1e-9)
        img = cv2.imread(str(paths.keyframes / f"{paths.frame_name(best_f)}.png"))
        ok = (Xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
        col[ok] = img[uv[ok, 1].astype(int), uv[ok, 0].astype(int)][:, ::-1]
        if p["kind"] == "loop":
            col = np.clip(col.astype(int) * 0.6 + np.array([0, 90, 0]), 0, 255).astype(np.uint8)
        Fs.append(F + sum(len(v) for v in Vs))
        Vs.append(V)
        Cs.append(col)
        meta.append({"id": pi, "kind": p["kind"], "normal": np.round(p["n"], 4), "offset": round(p["off"], 5),
                     "area": round(p["area"], 5), "depth": round(p["z"], 4), "support_points": p["points"],
                     "triangles": int(len(F)), "color_frame": int(best_f)})
    Vall, Fall, Call = np.vstack(Vs), np.vstack(Fs), np.vstack(Cs)
    mesh = trimesh.Trimesh(vertices=Vall, faces=Fall, vertex_colors=np.hstack([Call, np.full((len(Call), 1), 255, np.uint8)]),
                           process=False)
    mesh.export(paths.root / "geometry" / "surfaces.ply")
    glb_ok = True
    try:
        mesh.export(paths.root / "geometry" / "surfaces.glb")
    except Exception as exc:                          # GLB is a convenience; PLY is the primary export
        glb_ok = False
        print(f"[stage10] GLB export failed ({type(exc).__name__}: {exc}); PLY written")
    resid = np.flatnonzero(remaining)
    trimesh.PointCloud(cl["P"][resid], colors=cl["col"][resid]).export(paths.root / "geometry" / "unexplained_points.ply")

    explained = 1.0 - len(resid) / len(cl["P"])
    rng_col = np.random.default_rng(3)
    palette = rng_col.integers(60, 255, size=(len(patches), 3))
    for k in (ids[len(ids) // 4], ids[len(ids) // 2]):
        f = frames[k]
        img = cv2.imread(str(paths.keyframes / f"{paths.frame_name(k)}.png"))
        h, w = img.shape[:2]
        st = 3
        zb, pid = render_depth(patches, np.asarray(f["R_wc"]), np.asarray(f["t_wc"]), K, h, w, st)
        col = np.zeros((*zb.shape, 3), np.uint8)
        for pi, p in enumerate(patches):
            col[pid == pi] = (0, 200, 0) if p["kind"] == "loop" else palette[pi][::-1]
        colf = cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)
        m = colf.sum(-1) > 0
        vis = img.copy()
        vis[m] = (0.5 * img[m] + 0.5 * colf[m]).astype(np.uint8)
        cv2.putText(vis, f"frame {k}: {len(built_planes)} plane patches (colour per patch), loop faces green", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(paths.viz / f"surfaces_overlay_{k:04d}.png"), vis)
    # quads for the viewer: rectangle-merged plane patches only
    quads = []
    for p, m in zip(patches, [None] * len(patches)):
        if p["kind"] != "plane":
            continue
        col = tuple(int(x) for x in np.clip(np.abs(p["n"]) * 200 + 40, 0, 255))
        for r0, r1, c0, c1 in mask_to_rects(p["mask"]):
            cs = [(c0, r0), (c1, r0), (c1, r1), (c0, r1)]
            quads.append({"c": [np.round(p["p0"] + (p["a0"] + c * p["cell"]) * p["u"] + (p["b0"] + r * p["cell"]) * p["v"], 4)
                                for c, r in cs], "col": col})
    res = {"cloud_points": int(len(cl["P"])), "planes": len(planes), "plane_patches_before_gate": len(plane_patches),
           "plane_patches_built": len(built_planes), "plane_patches_rejected": rejected,
           "loop_faces": len(loop_patches), "plane_cells_replaced_by_loop_faces": removed,
           "cloud_points_explained_by_planes": explained, "triangles": int(len(Fall)), "vertices": int(len(Vall)),
           "glb_written": glb_ok, "held_out_gate": gate, "in_sample": in_sample,
           "not_built": "curved surfaces; boundaries come from where depth support ends (not edge-snapped)"}
    record_metrics(paths, "stage10_surfaces", res)
    write_json(out_json, {"patches": meta, "quads": quads, "planes": [
        {"normal": np.round(pl["n"], 4), "offset": round(pl["off"], 5), "support": pl["support"]} for pl in planes]})
    print(f"[stage10] {len(planes)} planes -> {len(plane_patches)} patches, {len(built_planes)} pass the evidence gate "
          f"({len(rejected)} rejected: {[r['reason'][:12] for r in rejected]}) + {len(loop_patches)} loop faces; "
          f"{explained:.0%} of cloud points explained by planes; {len(Fall)} triangles -> geometry/surfaces.ply")
    rows = [("held-out, ungated (fit frames " + gate["fit_frames"] + ", scored " + gate["scored_frames"] + ")", held)]
    if held_gated:
        rows.append(("held-out, after evidence gate (gated on fit frames only)", held_gated))
    rows += [("control: planes shifted +10% depth", ctrl), ("control: planes shifted -10% depth", ctrl2),
             ("final surfaces, in-sample (all frames)", in_sample)]
    for name, g in rows:
        oc = g["of_covered"]
        print(f"[stage10] {name:58s} covers {g['coverage_of_valid_depth']:.0%} of valid depth; of covered: "
              f"agree {oc['agree']:.0%}, hidden-behind {oc['occluded']:.0%}, CONTRADICTION {oc['contradiction']:.0%}")
