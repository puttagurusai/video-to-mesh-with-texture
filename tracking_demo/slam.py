"""Map-based monocular tracker ("mini-SLAM"): keyframes + triangulated landmarks + PnP tracking.

Unlike frame-to-frame VO it keeps a persistent map, so the scale is fixed by initialisation (baseline = 1 unit) and
poses are computed against the map instead of being chained. Deliberately small: ORB features, no bundle adjustment,
no loop closure, no global relocalisation database (ORB-SLAM3 would add those and needs Linux/WSL to build).

States: init (two-view initialisation with enough parallax) -> track (PnP against landmarks, keyframes add landmarks).
New landmarks need parallax: during a pure pan nothing can be triangulated, so the map stops growing. That is a
monocular limitation, not a bug, and it is reported through the per-frame landmark / inlier counts.
"""
from __future__ import annotations

import cv2
import numpy as np


def ratio_matches(bf, d1, d2, ratio):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    return [m[0] for m in bf.knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < ratio * m[1].distance]


def triangulate(K, Ra, ta, Rb, tb, pa, pb, min_angle_deg=1.0, max_reproj=2.0):
    """Landmarks in world coords from two posed views; returns (X, keep mask over the input pairs)."""
    Pa = K @ np.hstack([Ra, np.reshape(ta, (3, 1))])
    Pb = K @ np.hstack([Rb, np.reshape(tb, (3, 1))])
    Xh = cv2.triangulatePoints(Pa, Pb, pa.T.astype(np.float64), pb.T.astype(np.float64))
    X = (Xh[:3] / Xh[3]).T
    Xa, Xb = X @ Ra.T + ta, X @ Rb.T + tb
    ok = (Xa[:, 2] > 0) & (Xb[:, 2] > 0)
    for Xc, p in ((Xa, pa), (Xb, pb)):
        uv = Xc @ K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            ok &= np.linalg.norm(uv[:, :2] / uv[:, 2:3] - p, axis=1) < max_reproj
    Ca, Cb = -Ra.T @ ta, -Rb.T @ tb
    ra, rb = X - Ca, X - Cb
    cosang = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
    ok &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) >= min_angle_deg
    return X, ok


class MiniSlam:
    def __init__(self, K: np.ndarray):
        self.K = K
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.X, self.D = np.zeros((0, 3)), np.zeros((0, 32), np.uint8)
        self.state, self.ref, self.ref_age = "init", None, 0
        self.R, self.t = np.eye(3), np.zeros(3)
        self.kf, self.kf_inliers, self.lost = None, 0, 0

    # ------------------------------------------------------------------ public
    def process(self, gray: np.ndarray) -> dict:
        kp, des = self.orb.detectAndCompute(gray, None)
        if kp is None or des is None or len(kp) < 50:
            return self._result(False)
        pts = np.float32([k.pt for k in kp])
        return self._init(pts, des) if self.state == "init" else self._track(pts, des)

    def _result(self, valid, n_inl=0):
        return {"valid": valid, "R": self.R.copy(), "t": self.t.copy(), "inliers": n_inl,
                "landmarks": len(self.X), "state": self.state}

    # ------------------------------------------------------------------ init
    def _init(self, pts, des):
        if self.ref is None:
            self.ref, self.ref_age = (pts, des), 0
            return self._result(True)               # first frame defines the world
        self.ref_age += 1
        matches = ratio_matches(self.bf, self.ref[1], des, 0.75)
        if len(matches) < 150:
            if self.ref_age > 40:
                self.ref, self.ref_age = (pts, des), 0
            return self._result(False)
        p1 = self.ref[0][[m.queryIdx for m in matches]]
        p2 = pts[[m.trainIdx for m in matches]]
        E, mask = cv2.findEssentialMat(p1, p2, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape[0] < 3:
            return self._result(False)
        _, R, t, mask2 = cv2.recoverPose(E[:3], p1, p2, self.K, mask=mask)
        inl = mask2.ravel() > 0
        if inl.sum() < 100:
            return self._result(False)
        X, ok = triangulate(self.K, np.eye(3), np.zeros(3), R, t.ravel(), p1[inl], p2[inl], min_angle_deg=1.0)
        if ok.sum() < 80:
            if self.ref_age > 60:
                self.ref, self.ref_age = (pts, des), 0
            return self._result(False)
        sel = np.flatnonzero(inl)[ok]
        self.X = X[ok]
        self.D = des[[matches[i].trainIdx for i in sel]]
        lm_ref = np.full(len(self.ref[0]), -1)
        lm_cur = np.full(len(pts), -1)
        for n, i in enumerate(sel):
            lm_ref[matches[i].queryIdx] = n
            lm_cur[matches[i].trainIdx] = n
        self.R, self.t = R, t.ravel()
        self.kf = {"pts": pts, "des": des, "R": R, "t": t.ravel(), "lm": lm_cur}
        self.kf_inliers = int(ok.sum())
        self.state = "track"
        return self._result(True, int(ok.sum()))

    # ------------------------------------------------------------------ track
    def _track(self, pts, des):
        Xc = self.X @ self.R.T + self.t
        uvw = Xc @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[:, :2] / uvw[:, 2:3]
        h_w = (self.K[1, 2] * 2, self.K[0, 2] * 2)
        cand = np.flatnonzero((Xc[:, 2] > 0.05) & (uv[:, 0] > -20) & (uv[:, 0] < h_w[1] + 20)
                              & (uv[:, 1] > -20) & (uv[:, 1] < h_w[0] + 20))
        gate = 40.0 if self.lost == 0 else (150.0 if self.lost < 10 else 1e9)
        ids = obs = tr = None
        if len(cand) >= 15:
            ms = ratio_matches(self.bf, self.D[cand], des, 0.8)
            if ms:
                q = np.array([m.queryIdx for m in ms])
                tr = np.array([m.trainIdx for m in ms])
                keep = np.linalg.norm(uv[cand[q]] - pts[tr], axis=1) < gate
                ids, obs, tr = cand[q][keep], pts[tr][keep], tr[keep]
        if ids is None or len(ids) < 12:
            self.lost += 1
            return self._result(False)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(self.X[ids], obs.astype(np.float64), self.K, None,
                                                 iterationsCount=150, reprojectionError=3.0, confidence=0.99,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 15:
            self.lost += 1
            return self._result(False)
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(self.X[ids][inl], obs[inl].astype(np.float64), self.K, None, rvec, tvec)
        self.R, self.t, self.lost = cv2.Rodrigues(rvec)[0], tvec.ravel(), 0
        cur_lm = np.full(len(pts), -1)
        cur_lm[tr[inl]] = ids[inl]
        n_inl = len(inl)
        if n_inl < 0.75 * self.kf_inliers or n_inl < 80:
            n_inl = self._new_keyframe(pts, des, cur_lm, n_inl)
        return self._result(True, n_inl)

    def _new_keyframe(self, pts, des, cur_lm, n_inl):
        k = self.kf
        a, b = np.flatnonzero(k["lm"] < 0), np.flatnonzero(cur_lm < 0)
        added = 0
        if len(a) > 10 and len(b) > 10:
            ms = ratio_matches(self.bf, k["des"][a], des[b], 0.75)
            if len(ms) >= 10:
                ia = a[[m.queryIdx for m in ms]]
                ib = b[[m.trainIdx for m in ms]]
                X, ok = triangulate(self.K, k["R"], k["t"], self.R, self.t, k["pts"][ia], pts[ib], 1.0)
                if ok.any():
                    base = len(self.X)
                    self.X = np.vstack([self.X, X[ok]])
                    self.D = np.vstack([self.D, des[ib[ok]]])
                    cur_lm[ib[ok]] = base + np.arange(int(ok.sum()))
                    added = int(ok.sum())
        self.kf = {"pts": pts, "des": des, "R": self.R.copy(), "t": self.t.copy(), "lm": cur_lm}
        self.kf_inliers = max(n_inl + added, 30)
        return n_inl


def run_slam(frames, K: np.ndarray) -> tuple[dict, dict]:
    slam = MiniSlam(K)
    poses, info = {}, {"inliers": {}, "landmarks": {}, "first_tracked_after_init": None}
    for idx, gray, _ in frames:
        r = slam.process(gray)
        poses[idx] = (r["R"], r["t"]) if r["valid"] else None
        info["inliers"][idx], info["landmarks"][idx] = r["inliers"], r["landmarks"]
        if info["first_tracked_after_init"] is None and r["state"] == "track":
            info["first_tracked_after_init"] = idx
    return poses, info
