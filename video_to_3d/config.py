"""Pipeline configuration: one dataclass per stage, loaded from pipeline.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class KeyframeCfg:
    selection: str = "overlap"       # "overlap": the footage decides the count | "parallax": old fixed-threshold rule
    overlap_target: float = 0.8      # next keyframe when fewer than this share of the keyframe's corners are still shared
    max_shift_frac: float = 0.10     # ...or the image content moved more than this fraction of its long side
    max_gap_frames: int = 30         # ...or this many video frames passed (slow footage)
    min_gap_frames: int = 2
    adaptive_targets: list[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])  # tried in order by --adaptive
    max_analysis_frames: int = 2400  # long videos are analysed on an evenly spaced subset of this many frames (memory bound)
    min_keyframes: int = 12          # safety: never fewer (pose estimation needs a few views)
    max_keyframes: int = 300         # safety: cost cap; if the footage needs more, the capture itself is the problem
    max_side: int = 1600
    target_min: int = 20
    target_max: int = 40
    min_parallax_px: float = 25.0
    sharpness_window: int = 7
    min_sharpness_ratio: float = 0.5


@dataclass
class SfmCfg:
    matcher_overlap: int = 10
    camera_model: str = "SIMPLE_RADIAL"
    max_image_size: int = 1024   # SIFT runs on a downscaled copy; keypoints/K stay in keyframe pixels
    max_features: int = 4096
    random_seed: int = 0
    refine_distortion: bool = True
    min_registered_frac: float = 0.8
    max_reproj_error_px: float = 1.5


@dataclass
class Da3Cfg:
    # Existing DA3 install (code, venv, weights) from the earlier project; it is only executed, never modified.
    root: str = "C:/me/proj/3d scene reconstruction 2"
    process_res: int = 504
    chunk_max: int = 24      # frames per DA3 chunk upper bound (auto-probed against GPU memory)
    overlap: int = 5         # frames shared between chunks for Sim3 stitching


@dataclass
class RefineCfg:
    enabled: bool = True
    offsets: list[int] = field(default_factory=lambda: [1, 2])  # keyframe gaps whose tracked corners constrain poses
    corners: int = 600
    min_dist_px: int = 12
    huber_px: float = 2.0
    max_nfev: int = 60
    max_fit_tracks: int = 0          # optional cap on fitted correspondences (0 = none). A cap speeds up dense keyframe sets
                                     # (solve time scales with it) but its effect on accuracy is NOT validated yet
    depth_scale: bool = True         # also fit one depth-scale factor per keyframe (DA3 depth scale can differ between chunks)
    scale_prior_px: float = 20.0     # weak pull of each log depth scale towards 0 (px of residual per unit log-scale)
    loop_closure: bool = True        # add matches between keyframes far apart in time that show the same place (see loop_matches.py)
    loop_min_gap: int = 8            # keyframes closer than this in order count as neighbours, not loops
    loop_orb_features: int = 1500
    loop_min_coarse: int = 12        # coarse ratio-test matches (top 250 features) a pair needs to be verified
    loop_pairs_per_frame: int = 6    # best candidate partners verified per keyframe
    loop_min_inliers: int = 30       # RANSAC fundamental-matrix inliers a pair needs
    loop_matches_per_pair: int = 150
    loop_max_candidates: int = 80000  # cap on pairs scored on very long videos (random subset beyond this)
    reliable_px: float = 2.0         # held-out error above which the refined poses are flagged as unreliable


@dataclass
class DepthCfg:
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    input_size: int = 518
    ransac_iters: int = 500
    ransac_inlier_frac_tol: float = 0.05
    min_inliers: int = 20
    consistency_rel_tol: float = 0.05


@dataclass
class EdgeCfg:
    hysteresis_high: float = 0.5   # on background-normalised probability (see stage4_edges.edge_probability)
    hysteresis_low: float = 0.2
    simplify_eps_px: float = 1.0
    min_length_px: int = 10
    classify_offsets: list[int] = field(default_factory=lambda: [6, 16])  # px each side of the edge for type tagging
    jump_thresh: float = 0.04   # slope-compensated relative depth step => depth_discontinuity (provisional)
    kink_thresh: float = 0.02   # relative depth slope change => crease_candidate (provisional)


@dataclass
class LiftCfg:
    side_offset_px: list[int] = field(default_factory=lambda: [6, 16])  # [near, far] px each side of the edge
    crease_side_tol: float = 0.05
    smooth_sigma: float = 1.0
    step_px: float = 4.0         # arc-length spacing of lifted samples in the source image
    min_conf: float = 0.35       # drop samples where cross-view depth confidence is lower
    rdp_eps_rel: float = 0.002   # 3D simplification tolerance as a fraction of median scene depth


@dataclass
class GeometryCfg:
    rel_depth_sigma: float = 0.02    # assumed relative depth noise ALONG the ray (cross-view: ~99% agree within 5%)
    lateral_px_sigma: float = 2.0    # assumed sideways edge-localisation noise in pixels
    straight_k: float = 1.5          # straight if confidence-weighted rms of the normalised deviation <= this
    max_dev_factor: float = 4.0      # ...and no sample's normalised deviation exceeds this
    min_len_px: float = 20.0         # contours shorter than this (2D path length) stay undetermined
    min_samples: int = 4
    min_conf: float = 0.4            # mean depth confidence below this: shape undetermined
    corner_deg: float = 35.0         # polyline whose interior turns are all >= this is "piecewise" (sharp corners)
    dense_step_px: float = 6.0
    single_frame_conf_cap: float = 0.5   # one observation can never justify more than this confidence


@dataclass
class TrackCfg:
    tracked_kinds: list[str] = field(default_factory=lambda: ["depth_discontinuity", "crease_candidate"])
    long_straight_px: float = 150.0   # texture-tagged but long and straight edges are tracked too (real creases hide there)
    min_len_px: float = 40.0
    min_conf: float = 0.2             # stage-7 confidence needed to enter / start a map edge
    visible_depth_rel: float = 0.15   # projected depth within this of the observed depth => not occluded
    min_visible_samples: int = 5
    gate_rms_norm: float = 3.0        # new observation must fit the stored 3D geometry within this normalised rms
    fuse_gate_rel: float = 0.03       # curved-edge fusion neighbour radius, relative to depth
    max_obs_samples: int = 40
    verify_min_obs: int = 3
    verify_min_conf: float = 0.6


@dataclass
class GraphCfg:
    min_views: int = 2               # only edges seen in this many frames (or verified) enter the graph
    gap_px: float = 12.0             # endpoint snap radius in pixels at the edge's depth (contours stop short of corners)
    depth_snap_rel: float = 0.03     # ...and never less than this fraction of depth (depth noise along the ray)
    ext_px: float = 40.0             # how far a free endpoint may be extended along its tangent to meet another edge
    ext_tol_px: float = 6.0
    min_angle_deg: float = 20.0      # edges meeting at a node must differ in direction by at least this (no parallel merges)
    use_t_junctions: bool = True     # split an edge where another edge's free end lands on its interior
    use_extension: bool = True       # extend a free end along its tangent to meet a nearby edge
    loop_max_len: int = 8            # loops of up to this many graph edges
    max_cycles: int = 200000
    plane_tol_rel: float = 0.02      # loop must be planar to this fraction of its depth (rms)
    min_area_rel: float = 0.05       # loop side scale relative to depth
    min_thickness_rel: float = 0.04  # 4*area/perimeter relative to depth; thinner loops are double detections, not faces
    depth_check_shift_rel: float = 0.10   # control: plane moved this fraction of depth along its normal
    support_tol: float = 0.06        # depth inside a loop must match its plane within this relative error
    support_frac: float = 0.6        # ...for at least this share of pixels to count as a real surface
    control_shift_tol: float = 2.0   # control: displace every edge by this many snap radii in a random direction


@dataclass
class SurfaceCfg:
    stride: int = 16                  # pixel stride of the fused depth cloud
    conf_min: float = 0.5             # cross-view depth confidence needed for a cloud point
    plane_tol_rel: float = 0.02       # plane inlier distance relative to the point's depth
    normal_deg: float = 20.0          # inlier normal must agree with the plane normal within this
    max_planes: int = 24
    min_support_frac: float = 0.012   # a plane needs this share of all cloud points
    ransac_iters: int = 1500
    cell_rel: float = 0.025           # patch raster cell relative to the plane's depth
    min_cells: int = 15
    loop_classes: list[str] = field(default_factory=lambda: ["supported"])
    eval_tol: float = 0.04            # observed point lies on the plane within this fraction of its depth (2x the fit tolerance)
    eval_shift_rel: float = 0.10      # control: plane moved this fraction of depth along its normal
    gate_min_agree: float = 0.80      # a patch is kept only if this share of its covered pixels agree with observed depth...
    gate_max_contra: float = 0.10     # ...and at most this share contradict it (plane floating in front of observed depth)
    gate_min_px: int = 150            # patches seen by fewer pixels than this cannot be judged and are not built
    eval_stride: int = 4


@dataclass
class MeshCfg:
    max_dim: int = 320                 # longest side of the signed-distance volume in voxels (GPU memory ~ dim^3)
    min_views: int = 2                 # a voxel needs this many keyframes before it can become surface
    tex_side: int = 960                # long side of each keyframe texture
    silhouette_thr: float = 0.004      # depth pixels whose relative depth gradient per pixel exceeds this are not fused
                                       # (hotel sweep: off 11.3% stretched faces, 0.004 -> 2.3% for 3 points of coverage)
    min_frame_consistency: float = 0.5 # keyframes whose depth agrees with neighbours less than this are left out of fusion
    tex_min_cos: float = 0.2           # first texturing pass: view angle (cosine) a keyframe needs to texture a face
    tex_fallback_cos: float = 0.03     # second pass for grazing faces nobody saw frontally
    drop_unseen_faces: bool = True     # remove faces that point away from every keyframe camera (cannot be real surface)


@dataclass
class MatchCfg:
    gate_px: float = 3.0             # a projected sample votes for a 2D edge only if this close to it
    max_mean_dist_px: float = 4.0
    max_angle_deg: float = 15.0
    max_depth_rel_err: float = 0.08
    min_samples: int = 5
    min_coverage: float = 0.3       # share of the projected edge's visible samples that must hit the candidate
    min_support: float = 0.5        # stage 8: share of a projected edge's samples supported by ANY detected fragments
    strong_min_length_px: float = 40.0   # edges counted in the match-rate gate


@dataclass
class Config:
    video: str = "input/video.mp4"
    workdir: str = "."
    backend: str = "da3"     # "da3": poses+intrinsics+depth from Depth Anything 3 | "colmap": pycolmap SfM + DA V2
    keyframes: KeyframeCfg = field(default_factory=KeyframeCfg)
    da3: Da3Cfg = field(default_factory=Da3Cfg)
    refine: RefineCfg = field(default_factory=RefineCfg)
    sfm: SfmCfg = field(default_factory=SfmCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    edges: EdgeCfg = field(default_factory=EdgeCfg)
    lift: LiftCfg = field(default_factory=LiftCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    track: TrackCfg = field(default_factory=TrackCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    surface: SurfaceCfg = field(default_factory=SurfaceCfg)
    mesh: MeshCfg = field(default_factory=MeshCfg)
    edges_enabled: bool = False        # the edge stages (4-10) are off by default; enable with --edges or edges_enabled: true
    match: MatchCfg = field(default_factory=MatchCfg)

    @property
    def root(self) -> Path:
        return Path(self.workdir).resolve()


def _merge(obj: Any, data: dict[str, Any]) -> None:
    """Overlay YAML values onto a dataclass tree, rejecting unknown keys (typo guard)."""
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path = "pipeline.yaml") -> Config:
    cfg = Config()
    p = Path(path)
    if p.exists():
        _merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return cfg
