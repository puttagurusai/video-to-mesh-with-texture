"""Run the Phase-1 tracking demo on the room video ("live cam" = the video, frame by frame).

  python -I tracking_demo/run_demo.py [--video video/hotel_room.mp4] [--work-width 540]

Writes viz/tracking_demo/: one MP4 per method (cube from the method = green, reference cube = red),
errors.png, contact_sheet.png and metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracking_demo.common import Reference, cube_in_view, cube_points, draw_cube, evaluate, read_frames
from tracking_demo.slam import run_slam
from tracking_demo.vo import run_vo, run_vo_keyframe


def render(video, ref, poses, scale, out_path, label, extras=None):
    """Draw the method's cube (green) and the reference cube (red) on every frame; returns sampled frames."""
    w, h = ref.size
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    shots = {}
    for idx, _, bgr in read_frames(video, ref.size, ref.start, ref.end):
        R, t = ref.at(idx)
        draw_cube(bgr, cube_points(ref.center, ref.edge), R, t, ref.K, (0, 0, 255), 1)
        p = poses.get(idx)
        ok = p is not None and draw_cube(bgr, cube_points(scale * ref.center, scale * ref.edge), p[0], p[1],
                                         ref.K, (0, 220, 0), 3)
        txt = f"{label} f{idx} " + ("tracked" if p is not None else "LOST") + ("" if ok or p is None else " (cube off-screen)")
        if extras:
            txt += " " + extras(idx)
        cv2.putText(bgr, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(bgr)
        shots[idx] = bgr
    vw.release()
    return shots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "video" / "hotel_room.mp4"))
    ap.add_argument("--work-width", type=int, default=540)
    ap.add_argument("--out", default=str(ROOT / "viz" / "tracking_demo"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(a.video)
    ref = Reference(ROOT, a.work_width)
    print(f"frames {ref.start}..{ref.end}, working size {ref.size}, K fx={ref.K[0, 0]:.0f} "
          f"(from refined DA3 focal, NOT a checkerboard calibration)")

    grays = [(i, g, None) for i, g, _ in read_frames(video, ref.size, ref.start, ref.end)]
    results, timings, runs = {}, {}, {}

    t0 = time.time()
    runs["vo_every_frame"] = run_vo(grays, ref.K, stride=1)
    timings["vo_every_frame"] = time.time() - t0
    t0 = time.time()
    runs["vo_stride3"] = run_vo(grays, ref.K, stride=3)
    timings["vo_stride3"] = time.time() - t0
    t0 = time.time()
    runs["vo_keyframe_guarded"] = run_vo_keyframe(grays, ref.K)
    timings["vo_keyframe_guarded"] = time.time() - t0
    t0 = time.time()
    runs["slam"] = run_slam(grays, ref.K)
    timings["slam"] = time.time() - t0

    sheets = {}
    for name, (poses, info) in runs.items():
        res = evaluate(ref, poses)
        res["seconds"] = timings[name]
        res["fps_processing"] = len(grays) / timings[name]
        a_fit = res.get("fitted_unit_scale", 1.0)
        if name == "slam":
            res["first_tracked_frame"] = info["first_tracked_after_init"]
            lm = [v for v in info["landmarks"].values()]
            res["landmarks_final"] = lm[-1]
            res["landmarks_max"] = max(lm)
            ex = lambda i, info=info: f"inl {info['inliers'][i]} lm {info['landmarks'][i]}"  # noqa: E731
        else:
            ex = lambda i, info=info: f"matches {info['matches'].get(i, '-')}"  # noqa: E731
        sheets[name] = render(video, ref, poses, a_fit, out / f"{name}.mp4", name, ex)
        results[name] = res
        c, r = res.get("cube_corner_error_px"), res.get("rotation_error_deg")
        print(f"[{name}] tracked {res['frames_tracked']}/{res['frames_total']} frames, {res['fps_processing']:.0f} fps | "
              + (f"rotation err median {r['median']:.1f} deg (max {r['max']:.1f}, final {r['final']:.1f}) | cube corner err "
                 + (f"median {c['median']:.0f}px p90 {c['p90']:.0f}px max {c['max']:.0f}px over "
                    f"{res['cube_visible_keyframes']} in-view keyframes (frame width {c['frame_width_px']}px)" if c
                    else "n/a (cube never in view)")
                 if r else res.get("note", "")))

    json.dump(results, open(out / "metrics.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, res in results.items():
        pk = res.get("per_keyframe")
        if pk:
            ax[0].plot(pk["frame"], pk["rot_deg"], label=name)
            ax[1].plot(pk["frame"], [np.nan if v is None else v for v in pk["cube_px"]], label=name)
    ax[0].set_title("rotation error vs reference (deg)")
    ax[1].set_title("cube corner error (px)")
    for x in ax:
        x.set_xlabel("video frame")
        x.legend()
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=90)
    plt.close(fig)

    in_view = [f for f in range(ref.start, ref.end + 1) if cube_in_view(ref, *ref.at(f))]
    picks = [int(f) for f in np.linspace(in_view[0], in_view[-1], 4)] + [ref.end] if in_view else \
        [int(f) for f in np.linspace(ref.start, ref.end, 5)]
    rows = []
    for name in runs:
        row = [cv2.resize(sheets[name][min(sheets[name], key=lambda k: abs(k - p))], None, fx=0.45, fy=0.45) for p in picks]
        rows.append(np.hstack(row))
    cv2.imwrite(str(out / "contact_sheet.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
