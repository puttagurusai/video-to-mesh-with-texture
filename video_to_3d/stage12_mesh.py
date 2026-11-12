"""Stage 12: one textured surface mesh from the fused keyframe depth (TSDF + marching cubes + multi-view texture) and viewer."""
from __future__ import annotations

from .config import Config
from .io_utils import Paths
from .scene_mesh import build


def run(cfg: Config, paths: Paths, force: bool = False) -> None:
    glb = paths.root / "geometry" / "scene_mesh.glb"
    if glb.exists() and (paths.viz / "mesh_viewer.html").exists() and not force:
        print("[stage12] scene_mesh.glb exists, skipping (use --force)")
        return
    st = build(cfg, paths, label=f"({paths.root.name})")
    print(f"[stage12] mesh {st['triangles']:,} triangles, {st['textured_fraction']:.0%} textured "
          f"(voxel {100 * st['voxel_over_median_depth']:.1f}% of depth, {st['keyframes_used']} keyframes"
          + (f", {len(st['keyframes_excluded_for_inconsistency'])} excluded as inconsistent" if st["keyframes_excluded_for_inconsistency"] else "")
          + f"); {st['cloud_points_within_2_voxels_of_mesh']:.0%} of cloud points within 2 voxels of the mesh "
          f"-> geometry/scene_mesh.glb, viz/mesh_viewer.html")
