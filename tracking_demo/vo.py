"""Visual odometry exactly as described in the Phase-1 document:
ORB + BFMatcher -> essential matrix (RANSAC) -> recoverPose -> accumulate with a CONSTANT translation step.

recoverPose only yields a unit translation direction, so the per-step scale (STEP_SCALE) is arbitrary. That is the
method's known weakness and the reason the cube drifts when the true motion is not steady.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_MATCHES = 30
MAX_FEATURES = 1500


def run_vo_keyframe(frames, K: np.ndarray, step_scale: float = 0.03, new_kf_px: float = 20.0,
                    max_step_deg: float = 30.0) -> tuple[dict, dict]:
    """Guarded variant of the document's VO (NOT in the document): match against the last keyframe instead of the
    previous frame, start a new keyframe only once the image has moved >= new_kf_px, and reject any estimate whose
    rotation exceeds max_step_deg (the twisted-pair flip that destroys the plain version)."""
    from scipy.spatial.transform import Rotation
    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    T_kf = np.eye(4)        # pose of the current keyframe
    T_cur = np.eye(4)
    poses, info = {}, {"matches": {}, "rejected_flips": 0}
    kf = None
    for idx, gray, _ in frames:
        kp, des = orb.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        used = 0
        if kf is None:
            kf = (pts, des)
        elif des is not None and len(des) > 8:
            good = [m[0] for m in bf.knnMatch(kf[1], des, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            used = len(good)
            if used >= MIN_MATCHES:
                p1 = kf[0][[m.queryIdx for m in good]]
                p2 = pts[[m.trainIdx for m in good]]
                disp = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
                if disp >= 6.0:
                    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
                    if E is not None and E.shape[0] >= 3:
                        _, R, t, _ = cv2.recoverPose(E[:3], p1, p2, K, mask=mask)
                        if np.degrees(Rotation.from_matrix(R).magnitude()) <= max_step_deg:
                            T_rel = np.eye(4)
                            T_rel[:3, :3], T_rel[:3, 3] = R, t.ravel() * step_scale
                            T_cur = T_rel @ T_kf
                            if disp >= new_kf_px:
                                T_kf, kf = T_cur.copy(), (pts, des)
                        else:
                            info["rejected_flips"] += 1
        info["matches"][idx] = used
        poses[idx] = (T_cur[:3, :3].copy(), T_cur[:3, 3].copy())
    return poses, info


def run_vo(frames, K: np.ndarray, stride: int = 1, step_scale: float = 0.03) -> tuple[dict, dict]:
    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    T_cw = np.eye(4)
    poses, info = {}, {"matches": {}}
    prev_des = prev_pts = None
    for n, (idx, gray, _) in enumerate(frames):
        if n % stride == 0:
            kp, des = orb.detectAndCompute(gray, None)
            pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
            used = 0
            if prev_des is not None and des is not None and len(des) > 8 and len(prev_des) > 8:
                good = [m[0] for m in bf.knnMatch(prev_des, des, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
                used = len(good)
                if used >= MIN_MATCHES:
                    p1 = prev_pts[[m.queryIdx for m in good]]
                    p2 = pts[[m.trainIdx for m in good]]
                    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
                    if E is not None and E.shape[0] >= 3:
                        _, R, t, _ = cv2.recoverPose(E[:3], p1, p2, K, mask=mask)
                        T_rel = np.eye(4)
                        T_rel[:3, :3], T_rel[:3, 3] = R, t.ravel() * step_scale
                        T_cw = T_rel @ T_cw
            prev_des, prev_pts = des, pts
            info["matches"][idx] = used
        poses[idx] = (T_cw[:3, :3].copy(), T_cw[:3, 3].copy())
    return poses, info
