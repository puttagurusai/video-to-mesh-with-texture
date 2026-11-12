"""Stage 1: pick sharp, overlapping keyframes with real camera parallax from the video."""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .io_utils import Paths, record_metrics, write_json

ANALYSIS_SIDE = 800  # long side used for sharpness / optical flow (cheap); flow is rescaled to max_side


def _resize_long(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _median_parallax(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, int]:
    """Median LK displacement (px, at analysis scale) of corners tracked from prev to gray."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 20:
        return 0.0, 0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    if ok.sum() < 20:
        return 0.0, int(ok.sum())
    d = np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)
    return float(np.median(d)), int(ok.sum())


def _select(grays: list[np.ndarray], sharp: np.ndarray, thresh_analysis: float,
            window: int, min_sharp: float) -> list[int]:
    """Greedy selection: when parallax vs the last keyframe passes the threshold, take the sharpest
    nearby frame. Frames blurrier than min_sharp are never chosen."""
    n = len(grays)
    sel = [int(np.argmax(sharp[: max(window, 1)]))]
    for i in range(sel[0] + 1, n):
        par, tracked = _median_parallax(grays[sel[-1]], grays[i])
        # Tracking collapse (tracked<20) means large motion; treat as parallax passed.
        if par >= thresh_analysis or tracked < 20:
            lo, hi = max(sel[-1] + 1, i - window // 2), min(n, i + window // 2 + 1)
            cand = [j for j in range(lo, hi) if sharp[j] >= min_sharp]
            if cand:
                sel.append(max(cand, key=lambda j: sharp[j]))
    return sel


def _select_overlap(grays: list[np.ndarray], sharp: np.ndarray, kc, min_sharp: float) -> tuple[list[int], list[dict]]:
    """Overlap-driven keyframes: no fixed count. Corners of the current keyframe are tracked frame by frame (forward-backward
    checked); the next keyframe is taken just before the shared content falls below overlap_target (share of the keyframe's
    corners still tracked and inside the image), the image has moved more than max_shift_frac, or max_gap frames passed.
    Slow pans therefore get few keyframes and fast or rotating footage gets many, with a guaranteed overlap between neighbours."""
    n = len(grays)
    h, w = grays[0].shape
    side = max(h, w)
    max_gap = max(kc.min_gap_frames + 1, min(kc.max_gap_frames, n // max(kc.min_keyframes, 1)))
    lk = dict(winSize=(21, 21), maxLevel=3)

    def corners(i):
        p = cv2.goodFeaturesToTrack(grays[i], maxCorners=500, qualityLevel=0.01, minDistance=8)
        if p is None or len(p) < 30:      # featureless frame: fall back to a coarse grid so tracking can still report loss
            gy, gx = np.mgrid[h // 12:h:h // 6, w // 12:w:w // 6]
            p = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
        return p

    first = int(np.argmax(sharp[: max(kc.sharpness_window, 1)]))
    sel, stats = [first], [{"survival": 1.0, "shift_px": 0.0, "reason": "first"}]
    pts0 = corners(first)
    cur, alive, last_ok, j = pts0.copy(), np.ones(len(pts0), bool), first, first + 1
    while j < n:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[j - 1], grays[j], cur, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(grays[j], grays[j - 1], nxt, None, **lk)
        fb = np.linalg.norm((back - cur).reshape(-1, 2), axis=1)
        x, y = nxt[:, 0, 0], nxt[:, 0, 1]
        alive &= (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        cur = nxt
        survival = float(alive.mean())
        shift = float(np.median(np.linalg.norm((cur - pts0).reshape(-1, 2)[alive], axis=1))) if alive.any() else float("inf")
        reason = ("overlap" if survival < kc.overlap_target else "shift" if shift > kc.max_shift_frac * side
                  else "gap" if j - sel[-1] >= max_gap else None)
        if reason is None:
            last_ok, j = j, j + 1
            continue
        lo = max(sel[-1] + kc.min_gap_frames, last_ok - kc.sharpness_window + 1)
        hi = min(max(last_ok, sel[-1] + kc.min_gap_frames), n - 1)
        cands = [k for k in range(lo, hi + 1) if sharp[k] >= min_sharp]
        pick = max(cands, key=lambda k: sharp[k]) if cands else min(max(last_ok, sel[-1] + 1), n - 1)
        sel.append(pick)
        stats.append({"survival": survival, "shift_px": shift, "reason": reason})
        pts0 = corners(pick)
        cur, alive, last_ok, j = pts0.copy(), np.ones(len(pts0), bool), pick, pick + 1
    return sel, stats


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    meta_path = paths.keyframes / "meta.json"
    if meta_path.exists() and not force:
        print("[stage1] keyframes/meta.json exists, skipping (use --force)")
        return
    kc = cfg.keyframes
    if not paths.video.exists():
        raise FileNotFoundError(f"Input video not found: {paths.video}")

    cap = cv2.VideoCapture(str(paths.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # long videos: analyse an evenly spaced subset so memory stays bounded (every frame is kept as a small grey image)
    step = max(1, -(-total // kc.max_analysis_frames)) if total else 1
    if step > 1:
        print(f"[stage1] {total} frames: analysing every {step}th frame to bound memory")
    grays, sharp = [], []
    full_size = None
    n_read = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n_read % step == 0:
            full_size = frame.shape[1], frame.shape[0]
            g = cv2.cvtColor(_resize_long(frame, ANALYSIS_SIDE), cv2.COLOR_BGR2GRAY)
            grays.append(g)
            sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
        n_read += 1
    cap.release()
    if len(grays) < 10:
        raise RuntimeError(f"Video decoded only {len(grays)} analysable frames ({n_read} read)")
    sharp = np.asarray(sharp)
    min_sharp = kc.min_sharpness_ratio * float(np.median(sharp))

    # parallax threshold is specified at max_side scale; convert to analysis scale
    keyframe_scale = min(1.0, kc.max_side / max(full_size))
    analysis_scale = min(1.0, ANALYSIS_SIDE / max(full_size))
    scale = analysis_scale / keyframe_scale
    selection_info = {"mode": kc.selection}
    if kc.selection == "overlap":
        sel, sel_stats = _select_overlap(grays, sharp, kc, min_sharp)
        reasons = {r: sum(s["reason"] == r for s in sel_stats) for r in ("overlap", "shift", "gap")}
        selection_info.update({"overlap_target": kc.overlap_target, "max_shift_frac": kc.max_shift_frac,
                               "keyframes": len(sel), "triggers": reasons,
                               "boundary_survival_median": float(np.median([s["survival"] for s in sel_stats[1:]] or [1.0])),
                               "boundary_survival_min": float(np.min([s["survival"] for s in sel_stats[1:]] or [1.0]))})
        if len(sel) > kc.max_keyframes:
            print(f"[stage1] WARNING: footage needs {len(sel)} keyframes, capped at {kc.max_keyframes} "
                  f"(neighbour overlap will fall below target)")
            pick = np.linspace(0, len(sel) - 1, kc.max_keyframes).round().astype(int)
            sel = [sel[k] for k in pick]
    else:
        thresh = kc.min_parallax_px * scale
        sel = _select(grays, sharp, thresh, kc.sharpness_window, min_sharp)
        while len(sel) < kc.target_min and thresh > 2.0:
            thresh *= 0.7
            print(f"[stage1] only {len(sel)} keyframes, lowering parallax threshold to {thresh / scale:.1f}px")
            sel = _select(grays, sharp, thresh, kc.sharpness_window, min_sharp)
        if len(sel) > kc.target_max:
            pick = np.linspace(0, len(sel) - 1, kc.target_max).round().astype(int)
            sel = [sel[k] for k in pick]
    del grays
    sel = [s * step for s in sel]                      # analysis-frame index -> source video frame

    # second pass: write the chosen frames at max_side resolution. Remove old keyframes first: a retry with FEWER keyframes
    # would otherwise leave stale images from the previous attempt that later stages (DA3 reads the whole folder) pick up.
    for old in paths.keyframes.glob("frame_*.png"):
        old.unlink()
    chosen = set(sel)
    cap = cv2.VideoCapture(str(paths.video))
    saved, thumbs = [], []
    idx = 0
    while len(saved) < len(sel):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in chosen:
            img = _resize_long(frame, kc.max_side)
            k = len(saved)
            cv2.imwrite(str(paths.keyframes / f"{paths.frame_name(k)}.png"), img)
            saved.append({"index": k, "source_frame": idx, "time_s": idx / fps,
                          "sharpness": float(sharp[idx // step]), "width": img.shape[1], "height": img.shape[0]})
            thumbs.append(_resize_long(img, 240))
        idx += 1
    cap.release()

    if thumbs:
        cols = min(8, len(thumbs))
        rows = -(-len(thumbs) // cols)
        h, w = thumbs[0].shape[:2]
        sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
        for k, t in enumerate(thumbs):
            r, c = divmod(k, cols)
            sheet[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
        cv2.imwrite(str(paths.viz / "keyframes_contact_sheet.png"), sheet)

    write_json(meta_path, {"video": str(paths.video), "fps": fps, "frames": saved, "selection": selection_info})
    record_metrics(paths, "stage1_keyframes", {
        "selection": selection_info, "video_frames": int(n_read), "analysed_every": step, "keyframes": len(saved),
        "median_sharpness_all": float(np.median(sharp)),
        "median_sharpness_selected": float(np.median([s["sharpness"] for s in saved])),
    })
    print(f"[stage1] wrote {len(saved)} keyframes from {n_read} video frames "
          f"(contact sheet: viz/keyframes_contact_sheet.png)")
