# Video to textured 3D scene

Give it one ordinary video of a scene. It gives back a **point cloud**, a **textured mesh** and an offline **3D viewer**.
No training step, no COLMAP, one GPU (a 4 GB laptop GPU is enough).

```
run.bat "C:\path\to\video.mp4"            (or drag the video onto run.bat)
python run_pipeline.py --video path\to\video.mp4 --open
```

**Everything in depth** (history and decisions, every stage and output file, settings, verification, results, troubleshooting, open work):
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

Output goes to `runs\<video name>` (or `--out FOLDER`): `viz\mesh_viewer.html` (opens by itself), `geometry\scene_mesh.glb`
(textured, for Blender), `geometry\scene_mesh.ply`, `geometry\scene_cloud.ply` (the full cloud), `summary.json` (numbers and
warnings) and `logs\<video>_<date>_<time>.log` (a new log every run; the same text is shown live in the window).

---

## What the pipeline does, step by step

| # | Step | What happens |
|---|---|---|
| 1 | **Keyframes** (`stage1_keyframes`) | The footage decides how many frames to keep: a new keyframe starts when the previous one's tracked features are leaving the picture (default: 80% still shared). Blurry frames are skipped. Long videos are analysed at a stride. |
| 2 | **Poses + depth** (`stage2_da3`) | **Depth Anything 3 (DA3-BASE)** predicts, for every keyframe, the camera pose, focal length and a depth map, in chunks that are stitched together. About a minute on the GPU. |
| 2b | **Global refinement** (`stage2b_refine`, `loop_matches`) | Corrects what a feed-forward network gets wrong. Fits, for all keyframes at once: a small pose correction, **one depth-scale factor per keyframe** (chunks can disagree on scale) and one shared focal scale. It uses two kinds of matches: tracked corners between neighbouring keyframes, and **loop-closure matches** between keyframes that revisit the same place far apart in time (ORB features + RANSAC, no extra model). Half of all matches are held out to score the result honestly. |
| 3 | **Depth confidence** (`stage3_depth`) | Each depth pixel is checked against neighbouring views; disagreeing pixels get low confidence. |
| - | **Capture check** (`capture_quality`) | Rates the footage GOOD / FAIR / POOR from pose accuracy and depth agreement and prints advice. Unless `--no-adaptive`, it retries with stricter keyframe overlap and keeps the best attempt. |
| 11 | **Point cloud** (`stage11_cloud` / `scene_cloud`) | Every keyframe's depth is lifted into one world; each point records how many other keyframes confirm it. |
| 12 | **Mesh + texture** (`stage12_mesh` / `scene_mesh`) | All keyframes are fused into one signed-distance volume on the GPU (a voxel must be seen by 2 keyframes) -> marching cubes -> face direction taken from the volume -> each triangle textured from the keyframe that sees it best (depth-tested, smoothed) -> WebGL viewer + GLB/PLY. Keyframes that disagree with their neighbours are left out; out-of-memory retries at lower resolution. |

Scale is arbitrary (DA3 units), not metres. Nothing is invented: regions no keyframe saw stay empty.

## What we learned (why it looks like this)

* **DA3, not COLMAP, gives the poses.** COLMAP here was CPU-only, slow (2-4 min) and registered only 24-32 of 40 frames on phone pans; DA3 poses
  every frame in about a minute.
* **Capture density decides everything.** On a fast orbit (`video\source.mp4`, a table in a garden, about 6 loops in 7.75 s) neighbouring frames
  share only ~20% of trackable features. Keeping every frame cut pose error 44 px -> 4.7 px; loop closure + per-frame depth scale then raised
  cross-view agreement of the cloud from 32% to 99%.
* **A mesh bug hid most of the scene.** trimesh 4.x `mesh.invert()` leaves `face_normals` stale, so the visibility filter deleted 89% of a correct mesh
  (28k -> 243k triangles after the fix). Face direction now comes from the volume itself. Never call `invert()` and then read `face_normals`.
* **DA3-Streaming (loop closure inside DA3's repo) was not usable as shipped**: its place-recognition submodule is empty and it needs `faiss`, `pypose`
  and `triton`. The loop closure here is our own and needs no download.

Measured results (each on the stated footage; a 4 GB RTX 500 Ada laptop GPU):

| Footage | Setting | Result |
|---|---|---|
| hotel room, slow pan (13 s) | default | 19 keyframes, ~100 s, clean room, 99% textured |
| 4 s clip of the hotel pan | `run.bat`, adaptive | rated GOOD, 380k points, 291k triangles, 100% textured, 120 s |
| `source.mp4` garden orbit | 92 keyframes (min gap 2) | rated POOR, 331 s; cloud agreement poor |
| `source.mp4` | every frame + loop closure | cloud agreement 99%, 243,627 triangles, 99% textured, mesh covers 84% of the cloud, 221 s |
| mouse on a checked cloth, fast orbit | before loop closure | FAIR at best (not re-run since) |

## Folder layout

```
run.bat                  drag a video here (window stays open, live steps, new log per run)
run_pipeline.py          the command-line entry point (stages, adaptive retries, summary, viewer)
check_env.py             6-second readiness check (also: run_pipeline.py --check)
pipeline.yaml            every tunable setting, in sections (keyframes, da3, refine, mesh, ...)
CAPTURE_GUIDE.md         how to film a scene so the result is good
video_to_3d/             the code (see below)
runs/                    outputs, one folder per video
video/                   input videos
viz/room_mesh_viewer.html  the finished hotel-room viewer
configs/                 an example config (hotel, adaptive)
tracking_demo/           optional: a small visual-odometry / mini-SLAM demo from the early phase
```

`video_to_3d/` main path: `stage1_keyframes`, `stage2_da3`, `stage2b_refine`, `loop_matches`, `stage3_depth`, `capture_quality`, `scene_cloud`,
`scene_mesh`, `config`, `io_utils`, `runlog`, `selftest_refine`. Optional and **off by default**: the edge-graph stages (`stage4_edges` edge
detection, `stage5_lift` to 3D, `stage6_match`, `stage7_edge_geometry`, `stage8_edge_tracking`, `stage9_edge_graph` junctions and closed loops,
`stage10_surfaces` planes and loop faces, `coverage`, `viewer`; enable with `--edges`) and the COLMAP backend (`stage2_sfm`, `backend: colmap`).
That edge route was our first plan for a "surface from edges" scene; the point cloud -> mesh route replaced it as the goal.

## Options

`--out FOLDER`, `--force` (redo finished steps), `--no-adaptive` (single pass), `--stages 1-3` (explicit stages), `--edges`, `--config FILE`,
`--open` (open the viewer), `--check` (readiness check only), `--no-log`. Modules run alone: `python -m video_to_3d.scene_mesh`, `.scene_cloud`,
`.viewer`. Self-test of the refinement on synthetic data with known truth (about 30 s, no video, no GPU):
`python -m video_to_3d.selftest_refine`.

## Requirements

Windows, an NVIDIA GPU, Python 3.13 with torch (CUDA), OpenCV, SciPy, scikit-image, trimesh, Pillow, PyYAML. DA3-BASE (Apache-2.0) is the only
model; it comes from an existing install (`scripts/da3_reconstruct.py`, `tools/da3_src`, `tools/da3_models/DA3-BASE`, optionally `tools/venv_da3`):
set `da3.root` in `pipeline.yaml` or the `DA3_ROOT` environment variable (`DA3_PYTHON` to pick the interpreter). `python run_pipeline.py --check`
reports GPU, free VRAM, missing libraries, the DA3 files and disk. Before the heavy steps the run estimates the disk this video needs and stops
early with a clear message if it will not fit. The torch build here is CUDA 11.8 with kernels up to compute capability 9.0: **RTX 50-series GPUs
are not expected to work**. Sharing the code or weights with other people needs the owner's and legal review (DA3-BASE is Apache-2.0; check every
other file you ship).

## Know the limits

* **Capture matters most** (`CAPTURE_GUIDE.md`). Slow, smooth, overlapping footage works; fast orbits and look-alike repeating texture make poses
  disagree. The summary says so when it happens.
* `keyframes.min_gap_frames: 2` caps keyframes at every second frame. Very fast footage needs `1` (that is what made the garden orbit work); it
  is not the default because it costs time on normal footage.
* Loop closure only helps when the camera revisits a place; with no revisits the code says so and refinement behaves as before. Strongly repeating
  texture can still produce a few wrong revisit pairs (a robust loss limits the damage).
* Depth is a single network's guess per keyframe; the pose refinement keeps depth fixed except for one scale per keyframe.
* The refinement is slower on dense, loop-rich footage (about 90 s solve for 183 keyframes); `refine.max_fit_tracks` can cap it but its effect on
  accuracy is not validated. Thresholds and defaults were calibrated on a handful of videos only.
* Texture comes from a different keyframe per patch, so seams and exposure differences can show. The mesh viewer on the garden run was judged from
  rendered vertices, not inspected interactively.
