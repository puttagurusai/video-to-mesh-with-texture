"""Shared pieces for the tracking demo: reference poses, the fixed cube, projection, drawing, scoring.

All poses are expressed relative to the FIRST camera (world = first camera): X_cam = R @ X_world + t.
Monocular methods have an arbitrary unit, so each method's cube is placed using a least-squares scale fitted
to the reference trajectory (generous to the method; see evaluate()).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from video_to_3d.io_utils import read_json

CUBE_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def cube_points(center: np.ndarray, edge: float) -> np.ndarray:
    s = np.array([[(i & 1) * 2 - 1, ((i >> 1) & 1) * 2 - 1, ((i >> 2) & 1) * 2 - 1] for i in range(8)], float)
    return center + s * edge / 2


def project(P: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = P @ R.T + np.asarray(t).ravel()
    uvw = Xc @ K.T
    return uvw[:, :2] / uvw[:, 2:3], Xc[:, 2]


def draw_cube(img, P, R, t, K, color, thickness=2) -> bool:
    uv, z = project(P, R, t, K)
    if (z <= 0.05 * abs(z).max() + 1e-9).any() or not np.isfinite(uv).all() or np.abs(uv).max() > 1e5:
        return False
    for i, j in CUBE_EDGES:
        cv2.line(img, tuple(np.rint(uv[i]).astype(int)), tuple(np.rint(uv[j]).astype(int)), color, thickness, cv2.LINE_AA)
    return True


def cube_in_view(ref: "Reference", R: np.ndarray, t: np.ndarray) -> bool:
    """True when the anchored cube's centre projects inside the image (reference camera pose R, t)."""
    uv, z = project(ref.center[None], R, t, ref.K)
    return bool(z[0] > 0 and 0 <= uv[0, 0] < ref.size[0] and 0 <= uv[0, 1] < ref.size[1])


class Reference:
    """Refined DA3 keyframe poses, re-expressed relative to keyframe 0, scaled to the working resolution."""

    def __init__(self, root: Path, work_w: int):
        poses = read_json(root / "tracking" / "camera_poses.json")
        meta = read_json(root / "keyframes" / "meta.json")
        cam = next(iter(poses["cameras"].values()))
        s = work_w / cam["width"]
        self.K = np.array(cam["K"], float)
        self.K[:2] *= s
        self.size = (work_w, int(round(cam["height"] * s)))
        src = {m["index"]: m["source_frame"] for m in meta["frames"]}
        fr = sorted(poses["frames"], key=lambda f: f["frame"])
        self.frames = np.array([src[f["frame"]] for f in fr])
        R = [np.array(f["R_cw"]) for f in fr]
        t = [np.array(f["t_cw"]) for f in fr]
        self.R = [Ri @ R[0].T for Ri in R]
        self.t = [ti - Rr @ t[0] for Rr, ti in zip(self.R, t)]
        self.C = np.array([-Rr.T @ tr for Rr, tr in zip(self.R, self.t)])
        self.slerp = Slerp(self.frames, Rotation.from_matrix(self.R))
        depth = np.load(root / "depth" / "aligned" / "frame_0000.npy")
        h, w = depth.shape
        zc = float(np.nanmedian(depth[h * 2 // 5:h * 3 // 5, w * 2 // 5:w * 3 // 5]))
        self.center, self.edge = np.array([0.0, 0.0, zc]), 0.3 * zc
        self.start, self.end = int(self.frames[0]), int(self.frames[-1])

    def at(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        f = float(np.clip(f, self.start, self.end))
        R = self.slerp([f]).as_matrix()[0]
        C = np.array([np.interp(f, self.frames, self.C[:, i]) for i in range(3)])
        return R, -R @ C


def evaluate(ref: Reference, poses: dict[int, tuple | None]) -> dict:
    """Score a method at the reference keyframes. poses: video frame -> (R, t) relative to the first camera, or None."""
    rows = [(i, f) for i, f in enumerate(ref.frames) if poses.get(int(f)) is not None]
    out = {"frames_total": int(ref.end - ref.start + 1),
           "frames_tracked": int(sum(v is not None for f, v in poses.items() if ref.start <= f <= ref.end)),
           "keyframes_scored": len(rows)}
    if len(rows) < 4:
        return {**out, "note": "too few tracked keyframes to score"}
    Cm = np.array([-poses[int(f)][0].T @ np.asarray(poses[int(f)][1]).ravel() for _, f in rows])
    Cr = ref.C[[i for i, _ in rows]]
    denom = float((Cr * Cr).sum())
    a = float((Cr * Cm).sum() / denom) if denom > 1e-12 else 1.0
    a = a if a > 1e-6 else 1e-6
    rot, px, cos = [], [], []
    for (i, f), cm, cr in zip(rows, Cm, Cr):
        Rm, tm = poses[int(f)]
        rot.append(np.degrees(Rotation.from_matrix(np.asarray(Rm) @ ref.R[i].T).magnitude()))
        ur, zr = project(cube_points(ref.center, ref.edge), ref.R[i], ref.t[i], ref.K)
        um, zm = project(cube_points(a * ref.center, a * ref.edge), Rm, tm, ref.K)
        # pixel error only while the reference cube is actually in view: once it is off-screen, projecting it
        # amplifies tiny pose differences into thousands of pixels, which says nothing about tracking quality
        vis = cube_in_view(ref, ref.R[i], ref.t[i])
        px.append(float(np.linalg.norm(ur - um, axis=1).mean()) if vis and (zm > 0).all() else np.nan)
        if np.linalg.norm(cm) > 1e-6 and np.linalg.norm(cr) > 1e-6:
            cos.append(float(cm @ cr / (np.linalg.norm(cm) * np.linalg.norm(cr))))
    px = np.array(px)
    n_vis = int(np.isfinite(px).sum())
    out.update({
        "rotation_error_deg": {"median": float(np.median(rot)), "max": float(np.max(rot)), "final": float(rot[-1])},
        "cube_visible_keyframes": n_vis,
        "cube_corner_error_px": ({"median": float(np.nanmedian(px)), "p90": float(np.nanpercentile(px, 90)),
                                  "max": float(np.nanmax(px)), "frame_width_px": ref.size[0]} if n_vis else None),
        "trajectory_direction_cosine_median": float(np.median(cos)) if cos else None,
        "fitted_unit_scale": a,
        "per_keyframe": {"frame": [int(f) for _, f in rows], "rot_deg": [float(r) for r in rot],
                         "cube_px": [None if np.isnan(p) else float(p) for p in px]},
    })
    return out


def read_frames(video: Path, size: tuple[int, int], start: int, end: int):
    """Yield (index, gray, bgr) at the working size for frames start..end."""
    cap = cv2.VideoCapture(str(video))
    i = 0
    while i <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= start:
            bgr = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr
        i += 1
    cap.release()
