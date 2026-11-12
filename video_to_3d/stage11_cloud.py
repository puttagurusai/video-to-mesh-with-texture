"""Stage 11: the complete scene point cloud (every keyframe's depth in the shared world) and its cross-view agreement."""
from __future__ import annotations

from .config import Config
from .io_utils import Paths
from .scene_cloud import build


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    ply = paths.root / "geometry" / "scene_cloud.ply"
    if ply.exists() and not force:
        print("[stage11] scene_cloud.ply exists, skipping (use --force)")
        return
    st = build(cfg, paths)["stats"]
    print(f"[stage11] {st['points_total']:,} points from {st['keyframes_used']} keyframes -> geometry/scene_cloud.ply; "
          f"seen by >=2 other keyframes: {st['points_seen_by_2plus_other_keyframes']:.0%}; median agreement "
          f"{st['median_support_among_2plus']:.0%}, {st['share_with_support_ge_0.7_among_2plus']:.0%} of points confirmed by >=70% of views")
