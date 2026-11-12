"""Coverage diagnostic: which visible edges are missing from the 3D edge set, and at which stage were they lost?

  python -m video_to_3d.coverage --frame 13

Reference for "visible edges" is a plain Canny map (components >= 30 px), a proxy, not ground truth: it also contains
texture, noise and shading. Each reference pixel is assigned to the FIRST funnel stage that fails to keep it:
  covered_3d          a stored 3D edge projects within 3 px of it
  lost_in_lifting     a 2D polyline exists there but no 3D edge was produced (depth/confidence problem)
  short_or_untraced   TEED + hysteresis kept it, but chain tracing dropped it (shorter than min_length_px)
  hysteresis_dropped  TEED responds weakly (above the low threshold) but never reaches the high one
  teed_silent         TEED has no response there at all
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from .config import load_config
from .io_utils import Paths, read_json
from .stage4_edges import hysteresis
from .stage5_lift import Sampler, resample

CATS = ["covered_3d", "lost_in_lifting", "short_or_untraced", "hysteresis_dropped", "teed_silent"]
COLORS = {"covered_3d": (0, 200, 0), "lost_in_lifting": (255, 0, 255), "short_or_untraced": (0, 230, 255),
          "hysteresis_dropped": (0, 140, 255), "teed_silent": (0, 0, 255)}


def skeleton(mask: np.ndarray) -> np.ndarray:
    return cv2.ximgproc.thinning(mask.astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0


def draw(polys, h, w, key="points") -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for e in polys:
        pts = np.rint(np.asarray(e[key])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [pts], bool(e.get("closed", False)), 255, 1)
    return m > 0


def near(mask: np.ndarray, r: int = 3) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def canny_reference(gray: np.ndarray, min_component: int = 30) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 1.2)
    v = float(np.median(g))
    e = cv2.Canny(g, 0.66 * v, 1.33 * v)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component
    return keep[lab]


def lifting_loss_reasons(polys2d, Z, conf, cfg, lifted_mask) -> dict:
    """Pixel-length of 2D polylines per reason their samples could not be lifted (replicates stage 5's validity test)."""
    lc = cfg.lift
    near_px, far_px = lc.side_offset_px
    zs, cs = Sampler(Z), Sampler(conf.astype(np.float32)) if conf is not None else None
    h, w = Z.shape
    reasons = {"side_depth_outside_image": 0, "side_depth_invalid_nan": 0, "low_confidence": 0,
               "valid_but_runs_too_short": 0, "lifted": 0}
    for e in polys2d:
        p = resample(np.asarray(e["points"], float), bool(e["closed"]), lc.step_px)
        if len(p) < 3:
            continue
        t = np.gradient(p, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        side_pts = [p + n * d for d in (near_px, far_px)] + [p - n * d for d in (near_px, far_px)]
        outside = np.zeros(len(p), bool)
        for q in side_pts:
            outside |= (q[:, 0] < 0) | (q[:, 0] > w - 1) | (q[:, 1] < 0) | (q[:, 1] > h - 1)
        nan_side = np.zeros(len(p), bool)
        for q in side_pts:
            nan_side |= ~np.isfinite(zs(q))
        lowc = (np.nan_to_num(cs(p), nan=0.0) < lc.min_conf) if cs is not None else np.zeros(len(p), bool)
        ok = ~(outside | nan_side) & ~lowc
        # same run logic as stage 5: runs shorter than 3 samples are discarded
        idx = np.flatnonzero(ok)
        usable = np.zeros(len(p), bool)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []:
            if len(run) >= 3:
                usable[run] = True
        step = lc.step_px
        reasons["lifted"] += int(usable.sum() * step)
        bad = ~usable
        reasons["side_depth_outside_image"] += int((bad & outside).sum() * step)
        reasons["side_depth_invalid_nan"] += int((bad & ~outside & nan_side).sum() * step)
        reasons["low_confidence"] += int((bad & ~outside & ~nan_side & lowc).sum() * step)
        reasons["valid_but_runs_too_short"] += int((bad & ~outside & ~nan_side & ~lowc).sum() * step)
    return reasons


def run(frame: int, cfg=None, paths: Paths | None = None) -> dict:
    cfg = cfg or load_config()
    paths = paths or Paths(cfg)
    nm = paths.frame_name(frame)
    img = cv2.imread(str(paths.keyframes / f"{nm}.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    prob = cv2.imread(str(paths.edges__probability / f"{nm}.png"), 0).astype(np.float32) / 255.0
    ec = cfg.edges
    polys2d = read_json(paths.edges__polylines / f"{nm}.json")["polylines"]
    lifted = read_json(paths.edges__polylines3d / f"{nm}.json")["polylines3d"]
    Z = np.load(paths.depth__aligned / f"{nm}.npy")
    cp = paths.depth__confidence / f"{nm}.npy"
    conf = np.load(cp) if cp.exists() else None

    ref = canny_reference(gray)
    low_skel = skeleton(prob > ec.hysteresis_low)
    hyst_skel = skeleton(hysteresis(prob, ec.hysteresis_low, ec.hysteresis_high))
    m_poly, m_lift = draw(polys2d, h, w), draw(lifted, h, w, key="uv")
    r = 3
    cat = np.full((h, w), -1, np.int8)
    todo = ref.copy()
    for ci, mask in enumerate([near(m_lift, r), near(m_poly, r), near(hyst_skel, r), near(low_skel, r), np.ones_like(ref)]):
        hit = todo & mask
        cat[hit] = ci
        todo &= ~hit
    counts = {c: int((cat == i).sum()) for i, c in enumerate(CATS)}
    total = sum(counts.values())

    # pixel funnel of OUR detector (independent of the Canny reference)
    funnel = {"teed_low_skeleton_px": int(low_skel.sum()), "after_hysteresis_px": int(hyst_skel.sum()),
              "polyline_px": int(sum(e["length_px"] for e in polys2d)),
              "polyline_px_lifted_to_3d": int(m_lift.sum())}
    reasons = lifting_loss_reasons(polys2d, Z, conf, cfg, m_lift)

    vis = (img * 0.35).astype(np.uint8)
    for i, c in enumerate(CATS):
        vis[cat == i] = COLORS[c]
    vis = cv2.dilate(vis, np.ones((2, 2), np.uint8))
    out = {"frame": frame, "reference_canny_px": total,
           "reference_share": {c: counts[c] / total for c in CATS}, "funnel": funnel, "lifting_loss_by_reason_px": reasons,
           "note": "Canny is a proxy for visible edges; it includes texture/noise. Categories are mutually exclusive, first failing stage wins."}
    cv2.imwrite(str(paths.viz / f"coverage_frame{frame:04d}.png"), vis)
    json.dump(out, open(paths.viz / f"coverage_frame{frame:04d}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=13)
    res = run(ap.parse_args().frame)
    print(f"frame {res['frame']}: {res['reference_canny_px']} reference (Canny) edge pixels")
    for c in CATS:
        print(f"  {c:20s} {res['reference_share'][c]:6.1%}")
    print("our detector funnel (px):", res["funnel"])
    print("lifting loss by reason (px of 2D polyline):", res["lifting_loss_by_reason_px"])
