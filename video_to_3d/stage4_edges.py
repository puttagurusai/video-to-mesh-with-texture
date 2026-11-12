"""Stage 4: TEED edge probability -> hysteresis -> thin -> junction-split ordered chains -> 2D polylines.

Output per keyframe: edges/probability/frame_XXXX.png (uint8 probability) and
edges/polylines/frame_XXXX.json with polylines in pixel coordinates (x right, y down, centre-of-pixel = integer).

Edge type is *provisional metadata* from depth behaviour across the edge (not a decision):
  depth_discontinuity  big relative depth step between the two sides (occlusion boundary / silhouette)
  crease_candidate     no step but a slope change of depth across the edge (corner / fold)
  texture              depth is smooth through the edge (paint, print, shadow)
  unknown              no usable depth around the edge
Nothing is discarded here; later stages decide using multi-view evidence.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json
from .models.teed.ted import TED

WEIGHTS = "video_to_3d/models/teed/5_model.pth"
MEAN_BGR = np.array([104.007, 116.669, 122.679], np.float32)  # BIPED mean TEED was trained with
# ring order for crossing-number: N, NE, E, SE, S, SW, W, NW
RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
COLORS = {"depth_discontinuity": (0, 0, 255), "crease_candidate": (0, 200, 0),
          "texture": (255, 120, 0), "unknown": (160, 160, 160)}


def load_model(device: str) -> TED:
    model = TED()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def edge_probability(model: TED, img_bgr: np.ndarray, device: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x = torch.from_numpy(img_bgr.astype(np.float32) - MEAN_BGR).permute(2, 0, 1)[None].to(device)
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    with torch.no_grad():
        out = model(x)[-1]                      # fused edge map
    p = torch.sigmoid(out)[0, 0, :h, :w].float().cpu().numpy()
    # This checkpoint's output rests on a flat background (~0.44) instead of ~0, so rescale each frame:
    # background (median, edges are a tiny fraction of pixels) -> 0, saturated edge -> 1.
    bg = float(np.median(p))
    return np.clip((p - bg) / (1.0 - bg), 0.0, 1.0)


def hysteresis(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Keep connected regions of (p > lo) that contain at least one pixel > hi."""
    low = (p > lo).astype(np.uint8)
    n, lab = cv2.connectedComponents(low, connectivity=8)
    if n <= 1:
        return low.astype(bool)
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, lab.ravel(), p.ravel())
    keep = peak > hi
    keep[0] = False
    return keep[lab]


def _crossing_number(skel: np.ndarray) -> np.ndarray:
    """Number of 0->1 transitions around each pixel's 8-neighbour ring (junction if >= 3)."""
    s = np.pad(skel.astype(np.uint8), 1)
    h, w = skel.shape
    ring = [s[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING]
    cn = np.zeros(skel.shape, np.int8)
    for i in range(8):
        cn += ((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.int8)
    return cn


def trace_chains(skel: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Split a 1-px skeleton at junctions and return ordered pixel chains [(N,2 array of (x,y), closed)].
    Chain ends are extended by one adjacent junction pixel so touching chains share an endpoint."""
    cn = _crossing_number(skel)
    junction = skel & (cn >= 3)
    body = skel & ~junction
    n, lab = cv2.connectedComponents(body.astype(np.uint8), connectivity=8)
    h, w = skel.shape
    chains = []
    ys_all, xs_all = np.nonzero(body)
    order = np.argsort(lab[ys_all, xs_all], kind="stable")
    ys_all, xs_all = ys_all[order], xs_all[order]
    ids = lab[ys_all, xs_all]
    bounds = np.flatnonzero(np.diff(ids)) + 1
    for ys, xs in zip(np.split(ys_all, bounds), np.split(xs_all, bounds)):
        if len(ys) == 0:
            continue
        cid = lab[ys[0], xs[0]]
        pix = set(zip(ys.tolist(), xs.tolist()))

        def nbrs(p, pool):
            y, x = p
            four = [(y + dy, x + dx) for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1))]
            diag = [(y + dy, x + dx) for dy, dx in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
            return [q for q in four + diag if q in pool]  # 4-neighbours first -> no corner shortcuts

        # A connected piece can still branch (T-junctions the crossing-number test misses). Walking ONE path per piece
        # silently dropped every side branch (43% of skeleton pixels on the test frame), so keep tracing from what is
        # left until every pixel belongs to a chain; branches become their own chains.
        remaining = set(pix)
        while remaining:
            start = next((p for p in remaining if len(nbrs(p, remaining)) <= 1), None)
            closed = start is None            # only loops are left
            if closed:
                start = next(iter(remaining))
            path, cur = [start], start
            remaining.discard(start)
            while True:
                nxt = nbrs(cur, remaining)
                if not nxt:
                    break
                cur = nxt[0]
                path.append(cur)
                remaining.discard(cur)
            if closed and len(path) > 2:
                closed = abs(path[0][0] - path[-1][0]) <= 1 and abs(path[0][1] - path[-1][1]) <= 1
            for end_idx in (0, -1):  # attach adjacent junction pixel at each end
                y, x = path[end_idx]
                for dy, dx in RING:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and junction[yy, xx]:
                        path.insert(0, (yy, xx)) if end_idx == 0 else path.append((yy, xx))
                        closed = False
                        break
            chains.append((np.array([(x, y) for y, x in path], np.float32), closed))
    return chains


def _sample_nan(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Nearest-pixel sample of img at float (x,y) points; out-of-image -> NaN."""
    h, w = img.shape
    x, y = np.rint(pts[:, 0]).astype(int), np.rint(pts[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full(len(pts), np.nan, np.float32)
    out[ok] = img[y[ok], x[ok]]
    return out


def classify(pix: np.ndarray, Z: np.ndarray | None, ec) -> tuple[str, float, float]:
    """(type, relative depth jump across the edge, depth slope kink) from the dense chain pixels."""
    if Z is None or len(pix) < 7:
        return "unknown", float("nan"), float("nan")
    t = np.empty_like(pix)
    t[3:-3] = pix[6:] - pix[:-6]
    t[:3], t[-3:] = t[3], t[-4]
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], 1) / np.maximum(norm, 1e-6)
    near, far = ec.classify_offsets
    a_n, a_f = (_sample_nan(Z, pix + n * d) for d in (near, far))
    b_n, b_f = (_sample_nan(Z, pix - n * d) for d in (near, far))
    # Extrapolate each side's depth linearly to the edge itself. On one continuous (even slanted) surface both
    # extrapolations agree, so the step only responds to a real depth discontinuity; the sum of the two
    # outward slopes is zero on a smooth surface and non-zero where the surface bends (crease).
    sa, sb = (a_f - a_n) / (far - near), (b_f - b_n) / (far - near)
    za0, zb0 = a_n - sa * near, b_n - sb * near
    zmean = (a_n + a_f + b_n + b_f) / 4
    jump = np.abs(za0 - zb0) / np.minimum(np.minimum(a_n, b_n), zmean)
    kink = np.abs(sa + sb) * far / zmean
    if np.isfinite(jump).sum() < 3:
        return "unknown", float("nan"), float("nan")
    j, k = float(np.nanmedian(jump)), float(np.nanmedian(kink))
    if j > ec.jump_thresh:
        return "depth_discontinuity", j, k
    if k > ec.kink_thresh:
        return "crease_candidate", j, k
    return "texture", j, k


def polylines_for_frame(prob: np.ndarray, Z: np.ndarray | None, ec) -> list[dict]:
    mask = hysteresis(prob, ec.hysteresis_low, ec.hysteresis_high)
    skel = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    out = []
    for pix, closed in trace_chains(skel):
        if len(pix) < ec.min_length_px:
            continue
        simp = cv2.approxPolyDP(pix.reshape(-1, 1, 2), ec.simplify_eps_px, closed).reshape(-1, 2)
        if len(simp) < 2:
            continue
        etype, jump, kink = classify(pix, Z, ec)
        xi, yi = np.rint(pix[:, 0]).astype(int), np.rint(pix[:, 1]).astype(int)
        out.append({"points": np.round(simp, 2), "prob": float(prob[yi, xi].mean()),
                    "length_px": float(len(pix)), "type": etype, "depth_jump": jump, "kink": kink,
                    "closed": bool(closed)})
    for i, e in enumerate(out):
        e["id"] = i
    return out


def draw_overlay(img: np.ndarray, polys: list[dict]) -> np.ndarray:
    vis = (img * 0.55).astype(np.uint8)
    for e in polys:
        pts = np.rint(np.asarray(e["points"])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], bool(e["closed"]), COLORS[e["type"]], 2, cv2.LINE_AA)
    return vis


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    ec = cfg.edges
    poses = read_json(paths.tracking / "camera_poses.json")
    ids = sorted(f["frame"] for f in poses["frames"])
    out_json = lambda k: paths.edges__polylines / f"{paths.frame_name(k)}.json"  # noqa: E731
    if not force and all(out_json(k).exists() for k in ids):
        print("[stage4] polylines exist, skipping (use --force)")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    stats, t0, overlays = [], time.time(), []
    for k in ids:
        name = paths.frame_name(k)
        img = cv2.imread(str(paths.keyframes / f"{name}.png"))
        prob = edge_probability(model, img, device)
        cv2.imwrite(str(paths.edges__probability / f"{name}.png"), (prob * 255).astype(np.uint8))
        zp = paths.depth__aligned / f"{name}.npy"
        Z = np.load(zp) if zp.exists() else None
        if Z is not None and Z.shape != prob.shape:
            Z = None
        polys = polylines_for_frame(prob, Z, ec)
        h, w = prob.shape
        write_json(out_json(k), {"frame": k, "width": w, "height": h, "polylines": polys})
        types = {t: sum(e["type"] == t for e in polys) for t in COLORS}
        stats.append({"frame": k, "polylines": len(polys), "total_length_px": float(sum(e["length_px"] for e in polys)),
                      "types": types})
        if k % 6 == 0:
            overlays.append(draw_overlay(img, polys))
            cv2.imwrite(str(paths.viz / f"edges_overlay_{k:04d}.png"), overlays[-1])
    if overlays:
        small = [cv2.resize(o, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for o in overlays[:6]]
        cv2.imwrite(str(paths.viz / "edges_overlay_sheet.png"), np.hstack(small))
    tot = {t: sum(s["types"][t] for s in stats) for t in COLORS}
    summary = {"frames": len(ids), "seconds": time.time() - t0,
               "median_polylines_per_frame": float(np.median([s["polylines"] for s in stats])),
               "median_total_length_px": float(np.median([s["total_length_px"] for s in stats])),
               "type_totals": tot, "thresholds": {"hysteresis": [ec.hysteresis_low, ec.hysteresis_high]}}
    record_metrics(paths, "stage4_edges", summary)
    print(f"[stage4] {len(ids)} frames in {summary['seconds']:.0f}s, median "
          f"{summary['median_polylines_per_frame']:.0f} polylines/frame; types {tot}")
