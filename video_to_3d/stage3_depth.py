"""Stage 3: Depth Anything V2 Small inference, per-frame alignment to COLMAP landmarks, confidence.

Conventions
- Network output is relative *inverse* depth (larger = closer), so alignment is done in inverse depth:
      pred ~= a * (1 / z_sfm) + b            (a > 0)
  and aligned Z = 1 / ((pred - b) / a). Z is in SfM scale (arbitrary global scale).
- Pixel coordinates follow the array convention (x right, y down, pixel centre at integer coords).
  COLMAP keypoints use centre-at-0.5, so landmark observations are shifted by -0.5 before sampling.
- Lens distortion (SIMPLE_RADIAL k ~ 1e-3) is ignored here and in stage 5: pinhole K only.
"""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .io_utils import Paths, read_json, record_metrics, write_json


# ----------------------------------------------------------------------------- inference
def predict_depths(cfg: Config, paths: Paths, frames: list[int]) -> None:
    from transformers import AutoModelForDepthEstimation

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForDepthEstimation.from_pretrained(cfg.depth.model, torch_dtype=dtype).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    for k in frames:
        out_path = paths.depth__predicted / f"{paths.frame_name(k)}.npy"
        if out_path.exists():
            continue
        img = cv2.cvtColor(cv2.imread(str(paths.keyframes / f"{paths.frame_name(k)}.png")), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = cfg.depth.input_size / min(h, w)
        nh, nw = (max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14))
        x = torch.from_numpy(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)).to(dev)
        x = (x.permute(2, 0, 1)[None].float() / 255.0 - mean) / std
        with torch.no_grad():
            pred = model(pixel_values=x.to(dtype)).predicted_depth[None].float()
        pred = torch.nn.functional.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        np.save(out_path, pred[0, 0].cpu().numpy().astype(np.float32))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- alignment
def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample img (H,W) at float (x,y) array coords with bilinear interpolation (clamped)."""
    h, w = img.shape
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def fit_scale_shift(inv_sfm: np.ndarray, pred: np.ndarray, iters: int, tol_frac: float,
                    rng: np.random.Generator) -> tuple[float, float, np.ndarray] | None:
    """Robust fit pred = a*inv_sfm + b. RANSAC with 2-point models, inlier threshold =
    tol_frac * (p95 - p5 of pred) so it is stable when b ~ 0. Returns (a, b, inlier_mask)."""
    n = len(inv_sfm)
    if n < 4:
        return None
    thr = tol_frac * float(np.percentile(pred, 95) - np.percentile(pred, 5))
    best, best_cnt = None, 0
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        dx = inv_sfm[j] - inv_sfm[i]
        if abs(dx) < 1e-9:
            continue
        a = (pred[j] - pred[i]) / dx
        if a <= 0:
            continue
        b = pred[i] - a * inv_sfm[i]
        inl = np.abs(pred - (a * inv_sfm + b)) < thr
        if inl.sum() > best_cnt:
            best, best_cnt = inl, int(inl.sum())
    if best is None:
        return None
    for _ in range(3):  # least-squares refinement on inliers, re-evaluate inliers
        A = np.stack([inv_sfm[best], np.ones(best.sum())], 1)
        (a, b), *_ = np.linalg.lstsq(A, pred[best], rcond=None)
        if a <= 0:
            return None
        best = np.abs(pred - (a * inv_sfm + b)) < thr
    return float(a), float(b), best


def consistency(Z_i: torch.Tensor, K: torch.Tensor, T_i: tuple, T_j: tuple, Z_j: torch.Tensor,
                tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU warp of frame i's aligned depth into frame j (all tensors on the same device).
    Returns (score, visible): per-pixel agreement in [0,1] (1 = depth matches Z_j) and whether
    the pixel lands on a valid depth pixel in j."""
    R_i, t_i = T_i
    R_j, t_j = T_j
    h, w = Z_i.shape
    v, u = torch.meshgrid(torch.arange(h, device=Z_i.device, dtype=torch.float32),
                          torch.arange(w, device=Z_i.device, dtype=torch.float32), indexing="ij")
    P_c = torch.stack([(u - K[0, 2]) / K[0, 0] * Z_i, (v - K[1, 2]) / K[1, 1] * Z_i, Z_i], -1)
    P_j = ((P_c - t_i) @ R_i) @ R_j.T + t_j      # world = R_i^T (P_c - t_i); then into camera j
    z = P_j[..., 2]
    uj = K[0, 0] * P_j[..., 0] / z + K[0, 2]
    vj = K[1, 1] * P_j[..., 1] / z + K[1, 2]
    inside = (z > 1e-6) & (uj >= 0) & (uj <= w - 1) & (vj >= 0) & (vj <= h - 1) & torch.isfinite(Z_i)
    grid = torch.stack([uj / (w - 1) * 2 - 1, vj / (h - 1) * 2 - 1], -1)[None]
    valid_j = torch.isfinite(Z_j)
    src = torch.stack([torch.nan_to_num(Z_j, nan=0.0), valid_j.float()])[None]
    smp = torch.nn.functional.grid_sample(src, grid, mode="bilinear", padding_mode="border",
                                          align_corners=True)[0]
    zj, vj_ok = smp[0], smp[1] > 0.99
    vis = inside & vj_ok
    rel = (z - zj).abs() / zj.clamp_min(1e-6)
    score = torch.where(vis, (1.0 - rel / (3.0 * tol)).clamp(0.0, 1.0), torch.zeros_like(rel))
    return score, vis


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    dc = cfg.depth
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    cams = {int(k): v for k, v in poses["cameras"].items()}
    if cfg.backend == "da3":
        return _run_da3(cfg, paths, frames, cams, force)
    pts = np.load(paths.colmap / "points3D.npz")
    xyz, err, tlen, obs = pts["xyz"], pts["error"], pts["track_length"], pts["obs"]
    ids = sorted(frames)

    done = all((paths.depth__aligned / f"{paths.frame_name(k)}.npy").exists() for k in ids)
    if done and not force:
        print("[stage3] aligned depth exists, skipping (use --force)")
        return
    if force:
        for sub in ("aligned", "confidence"):  # predictions depend only on the keyframe pixels
            for f in (paths.depth / sub).glob("*.npy"):
                f.unlink()

    print(f"[stage3] predicting depth for {len(ids)} frames ({dc.model}) ...")
    predict_depths(cfg, paths, ids)

    rng = np.random.default_rng(0)  # fixed seed -> reproducible alignment
    Z, fits, reliable = {}, {}, {}
    stats_frames, scatter = [], {}
    for k in ids:
        f = frames[k]
        K = np.asarray(cams[f["camera_id"]]["K"])
        R, t = np.asarray(f["R_cw"]), np.asarray(f["t_cw"])
        pred = np.load(paths.depth__predicted / f"{paths.frame_name(k)}.npy")
        rows = obs[obs[:, 1] == k]
        pi = rows[:, 0].astype(int)
        good = (tlen[pi] >= 3) & (err[pi] < 1.0)
        rows, pi = rows[good], pi[good]
        zc = (xyz[pi] @ R.T + t)[:, 2]
        ok = zc > 1e-6
        rows, pi, zc = rows[ok], pi[ok], zc[ok]
        inv = 1.0 / zc
        pv = _bilinear(pred, rows[:, 2:4] - 0.5)
        res = fit_scale_shift(inv, pv, dc.ransac_iters, dc.ransac_inlier_frac_tol, rng)
        entry = {"frame": k, "landmarks": int(len(zc))}
        if res is None or res[2].sum() < dc.min_inliers:
            reliable[k] = False
            entry.update(reliable=False, inliers=int(res[2].sum()) if res else 0)
            stats_frames.append(entry)
            print(f"[stage3] frame {k}: unreliable alignment ({entry['inliers']} inliers)")
            continue
        a, b, inl = res
        fitted = a * inv + b
        rel_res = np.abs(pv - fitted) / np.maximum(np.abs(fitted), 1e-9)
        inv_al = (pred - b) / a                          # inverse depth in SfM scale
        inv_floor = 0.25 * float(np.percentile(inv, 5))  # cap Z at 4x the farthest landmark
        Zk = np.where(inv_al > inv_floor, 1.0 / np.maximum(inv_al, 1e-12), np.nan).astype(np.float32)
        Z[k], reliable[k] = Zk, True
        scatter[k] = (inv, pv, a, b, inl)
        entry.update(reliable=True, a=a, b=b, inliers=int(inl.sum()),
                     inlier_frac=float(inl.mean()),
                     median_rel_residual_inliers=float(np.median(rel_res[inl])),
                     median_rel_residual_all=float(np.median(rel_res)),
                     valid_pixel_frac=float(np.isfinite(Zk).mean()))
        stats_frames.append(entry)

    conf_means, cons_means = _confidence(dc, paths, ids, frames, cams, Z, reliable, stats_frames)

    for k in ids:
        arr = Z.get(k)
        np.save(paths.depth__aligned / f"{paths.frame_name(k)}.npy",
                arr if arr is not None else np.full((1, 1), np.nan, np.float32))

    _diagnostics(paths, ids, Z, scatter)
    ok_entries = [e for e in stats_frames if e["reliable"]]
    summary = {
        "frames": len(ids), "reliable_frames": len(ok_entries),
        "median_rel_residual_inliers": float(np.median([e["median_rel_residual_inliers"] for e in ok_entries])),
        "worst_frame_rel_residual": float(max(e["median_rel_residual_inliers"] for e in ok_entries)),
        "mean_frac_consistent_neighbors": float(np.mean(cons_means)) if cons_means else 0.0,
        "per_frame": stats_frames,
    }
    record_metrics(paths, "stage3_depth", summary)
    print(f"[stage3] {summary['reliable_frames']}/{summary['frames']} frames aligned; "
          f"median inlier residual {summary['median_rel_residual_inliers']:.1%} "
          f"(worst frame {summary['worst_frame_rel_residual']:.1%}); "
          f"{summary['mean_frac_consistent_neighbors']:.0%} of verified pixels consistent with neighbours")


def _confidence(dc, paths: Paths, ids: list[int], frames: dict, cams: dict, Z: dict, reliable: dict,
                stats_frames: list) -> tuple[list, list]:
    """Cross-view confidence on the GPU: compare each frame's depth with its 2 nearest registered
    neighbours on each side. Writes depth/confidence/*.npy and updates stats_frames in place."""
    print("[stage3] cross-view consistency ...")
    conf_means, cons_means = [], []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    Zt = {k: T(v) for k, v in Z.items()}
    for k in ids:
        if not reliable.get(k):
            np.save(paths.depth__confidence / f"{paths.frame_name(k)}.npy", np.zeros((1, 1), np.float32))
            continue
        f = frames[k]
        K = T(cams[f["camera_id"]]["K"])
        Ti = (T(f["R_cw"]), T(f["t_cw"]))
        nbrs = [j for j in (k - 2, k - 1, k + 1, k + 2) if j in Zt]
        num, den = torch.zeros_like(Zt[k]), torch.zeros_like(Zt[k])
        for j in nbrs:
            fj = frames[j]
            s, v = consistency(Zt[k], K, Ti, (T(fj["R_cw"]), T(fj["t_cw"])), Zt[j], dc.consistency_rel_tol)
            num += s
            den += v.float()
        conf_t = torch.where(den > 0, num / den.clamp_min(1), torch.full_like(num, 0.3))  # 0.3 = unverified
        conf_t[~torch.isfinite(Zt[k])] = 0.0
        conf, den = conf_t.cpu().numpy(), den.cpu().numpy()
        np.save(paths.depth__confidence / f"{paths.frame_name(k)}.npy", conf.astype(np.float16))
        verified = den > 0
        conf_means.append(float(conf[verified].mean()) if verified.any() else 0.0)
        cons_means.append(float((conf[verified] > 0.67).mean()) if verified.any() else 0.0)
        stats_frames[[e["frame"] for e in stats_frames].index(k)].update(
            mean_confidence=conf_means[-1], frac_consistent_within_tol=cons_means[-1])
    return conf_means, cons_means


def _run_da3(cfg: Config, paths: Paths, frames: dict, cams: dict, force: bool) -> None:
    """DA3 backend: stage 2 already wrote multi-view-consistent depth in the DA3 pose frame to
    depth/aligned, so only the cross-view confidence is computed here (no V2 model, no landmarks)."""
    dc = cfg.depth
    ids = sorted(frames)
    conf_done = all((paths.depth__confidence / f"{paths.frame_name(k)}.npy").exists() for k in ids)
    if conf_done and not force:
        print("[stage3] confidence maps exist, skipping (use --force)")
        return
    Z, stats_frames = {}, []
    for k in ids:
        Z[k] = np.load(paths.depth__aligned / f"{paths.frame_name(k)}.npy")
        stats_frames.append({"frame": k, "reliable": True,
                             "valid_pixel_frac": float(np.isfinite(Z[k]).mean())})
    conf_means, cons_means = _confidence(dc, paths, ids, frames, cams, Z, {k: True for k in ids}, stats_frames)
    _diagnostics(paths, ids, Z, {})
    low = [e["frame"] for e in stats_frames if e.get("frac_consistent_within_tol", 1.0) < 0.5]
    summary = {"backend": "da3", "frames": len(ids),
               "mean_frac_consistent_neighbors": float(np.mean(cons_means)),
               "min_frame_frac_consistent": float(min(cons_means)),
               "low_consistency_frames": low, "per_frame": stats_frames}
    record_metrics(paths, "stage3_depth", summary)
    print(f"[stage3] da3 depth: {summary['mean_frac_consistent_neighbors']:.0%} of verified pixels consistent "
          f"with neighbours (worst frame {summary['min_frame_frac_consistent']:.0%}); "
          f"frames below 50%: {low or 'none'}")


def _diagnostics(paths: Paths, ids: list[int], Z: dict, scatter: dict) -> None:
    show = [k for k in ids if k in Z]
    pick = show[:: max(1, len(show) // 6)][:6]
    if pick:
        fig, axes = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 8))
        axes = np.atleast_2d(axes).reshape(2, -1)
        for c, k in enumerate(pick):
            img = cv2.cvtColor(cv2.imread(str(paths.keyframes / f"{paths.frame_name(k)}.png")), cv2.COLOR_BGR2RGB)
            axes[0, c].imshow(img)
            axes[0, c].set_title(f"frame {k}")
            d = Z[k]
            axes[1, c].imshow(d, cmap="turbo", vmin=np.nanpercentile(d, 2), vmax=np.nanpercentile(d, 98))
            for r in (0, 1):
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(paths.viz / "depth_aligned_samples.png", dpi=80)
        plt.close(fig)
    if not scatter:  # DA3 backend has no landmark fits to plot
        return
    fig, axes = plt.subplots(1, len(pick) or 1, figsize=(3.2 * (len(pick) or 1), 3.2))
    for ax, k in zip(np.atleast_1d(axes), pick):
        inv, pv, a, b, inl = scatter[k]
        ax.scatter(inv[~inl], pv[~inl], s=3, c="r")
        ax.scatter(inv[inl], pv[inl], s=3, c="g")
        xs = np.array([inv.min(), inv.max()])
        ax.plot(xs, a * xs + b, "k-", lw=1)
        ax.set_title(f"frame {k}: {inl.sum()}/{len(inl)} inl")
        ax.set_xlabel("1/z (COLMAP)")
        ax.set_ylabel("DA V2 pred")
    fig.tight_layout()
    fig.savefig(paths.viz / "depth_alignment_fits.png", dpi=80)
    plt.close(fig)
