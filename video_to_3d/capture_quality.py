"""Capture-quality assessment: did this video give poses and depth that agree between views, and if not, why?

Measured from the pipeline's own outputs after stages 1-3 (so it can decide whether to add keyframes and retry):
  * depth consistency between neighbouring keyframes   (stage 3)
  * pose accuracy on tracked corners after refinement   (stage 2b, held-out points)
  * rotation and camera movement between keyframes      (poses)
  * feature matches shared by neighbouring keyframes    (SIFT + RANSAC on sampled pairs)

Verdict thresholds are calibrated on two videos only (a slow room pan and a fast orbit around a small object):
  GOOD  consistency >= 95% and refined pose error <= 1.0 px   (hotel: 99%, 0.24 px)
  FAIR  consistency >= 80% and refined pose error <= 4.0 px   (mouse, 150 keyframes: 84%, 3.1 px)
  POOR  otherwise                                              (mouse, 40 keyframes: 36%, 19 px)
Treat them as a starting point, not a guarantee, and check the report's advice lines.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json

GOOD_CONSISTENCY, GOOD_POSE_PX = 0.95, 1.0
FAIR_CONSISTENCY, FAIR_POSE_PX = 0.80, 4.0


def neighbour_matches(paths: Paths, n_pairs: int = 8) -> dict:
    meta = read_json(paths.keyframes / "meta.json")["frames"]
    n = len(meta)
    idx = np.unique(np.linspace(0, n - 2, min(n_pairs, n - 1)).round().astype(int))
    sift = cv2.SIFT_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    counts = []
    for i in idx:
        a = cv2.imread(str(paths.keyframes / f"frame_{i:04d}.png"), 0)
        b = cv2.imread(str(paths.keyframes / f"frame_{i + 1:04d}.png"), 0)
        ka, da = sift.detectAndCompute(a, None)
        kb, db = sift.detectAndCompute(b, None)
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            counts.append(0)
            continue
        ms = [m[0] for m in bf.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(ms) < 12:
            counts.append(len(ms))
            continue
        p = np.float32([ka[m.queryIdx].pt for m in ms])
        q = np.float32([kb[m.trainIdx].pt for m in ms])
        _, mk = cv2.findFundamentalMat(p, q, cv2.FM_RANSAC, 1.5, 0.99)
        counts.append(int(mk.sum()) if mk is not None else 0)
    return {"median": float(np.median(counts)), "min": int(np.min(counts)), "pairs_tested": len(counts)}


def assess(cfg: Config, paths: Paths) -> dict:
    val = read_json(paths.viz / "validation.json")
    s3, s2b = val["stage3_depth"], val.get("stage2b_refine")
    poses = read_json(paths.tracking / "camera_poses.json")
    fr = sorted(poses["frames"], key=lambda f: f["frame"])
    R = [np.asarray(f["R_cw"]) for f in fr]
    C = [np.asarray(f["t_wc"]) for f in fr]
    rot = [np.degrees(Rotation.from_matrix(R[i + 1] @ R[i].T).magnitude()) for i in range(len(R) - 1)]
    depth = float(np.nanmedian(np.load(paths.depth__aligned / f"{paths.frame_name(fr[0]['frame'])}.npy")))
    span = float(np.linalg.norm(np.ptp(np.array(C), axis=0)))
    consistency = float(s3["mean_frac_consistent_neighbors"])
    worst = float(s3.get("min_frame_frac_consistent", consistency))
    pose_px = float(s2b["held_out_error_px"]["after_median"]) if s2b else float("nan")
    match = neighbour_matches(paths)
    meta = read_json(paths.keyframes / "meta.json")

    if consistency >= GOOD_CONSISTENCY and pose_px <= GOOD_POSE_PX:
        verdict = "GOOD"
    elif consistency >= FAIR_CONSISTENCY and pose_px <= FAIR_POSE_PX:
        verdict = "FAIR"
    else:
        verdict = "POOR"
    m = {"verdict": verdict, "keyframes": len(fr), "selection": meta.get("selection"),
         "depth_consistency_between_neighbours": consistency, "worst_frame_consistency": worst,
         "refined_pose_error_px": pose_px,
         "rotation_between_keyframes_deg": {"median": float(np.median(rot)), "p90": float(np.percentile(rot, 90))},
         "camera_path_span_over_depth": span / depth, "neighbour_feature_matches": match,
         "score": consistency - min(pose_px, 50.0) / 50.0}

    advice = []
    if m["rotation_between_keyframes_deg"]["p90"] > 10:
        advice.append(f"The camera turns quickly: 10% of keyframe steps rotate more than "
                      f"{m['rotation_between_keyframes_deg']['p90']:.0f} deg. Rotate and move more slowly and steadily.")
    if match["median"] < 300 or match["min"] < 100:
        advice.append(f"Neighbouring views share few features (median {match['median']:.0f}, fewest {match['min']}). "
                      f"Keep at least 70-80% of the picture the same between moments; add texture and structure to the scene.")
    if consistency < FAIR_CONSISTENCY:
        advice.append(f"Only {consistency:.0%} of depth agrees between neighbouring views: poses/depth are unreliable for this footage.")
    if m["camera_path_span_over_depth"] < 0.03:
        advice.append("The camera barely translates (almost a pure turn on the spot). Depth structure is weakly constrained; "
                      "walk sideways while filming.")
    if meta.get("selection", {}).get("keyframes", 0) >= cfg.keyframes.max_keyframes:
        advice.append("The footage needed more keyframes than the cost cap allows; film slower so fewer are needed.")
    if not advice and verdict == "GOOD":
        advice.append("Capture looks good: neighbouring views overlap well and agree on depth and pose.")
    m["advice"] = advice
    record_metrics(paths, "capture_quality", m)
    write_json(paths.viz / "capture_report.json", m)
    return m


def print_report(m: dict, label: str = "") -> None:
    print(f"[capture] {label}{m['verdict']}: {m['keyframes']} keyframes, depth consistency {m['depth_consistency_between_neighbours']:.0%}, "
          f"refined pose error {m['refined_pose_error_px']:.2f}px, rotation/keyframe median "
          f"{m['rotation_between_keyframes_deg']['median']:.1f} deg (p90 {m['rotation_between_keyframes_deg']['p90']:.1f}), "
          f"neighbour matches median {m['neighbour_feature_matches']['median']:.0f}")
    for a in m["advice"]:
        print(f"[capture]   - {a}")
