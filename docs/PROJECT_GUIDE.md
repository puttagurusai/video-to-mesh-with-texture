# Project guide: video to textured 3D scene

This is the long version. `README.md` is the one-page "how to run it"; this document is for someone who needs the **whole picture**:
what the project set out to do, how it got to its current form, what every stage does, what every output file is, how the results are
judged, what failed along the way, and what is still open. Everything stated as a number was measured in this project; things that were
not measured say so.

Contents: 1 Goal - 2 How the project evolved - 3 Data flow and output folder - 4 Stage reference - 5 Capture check - 6 Settings -
7 Optional components - 8 How results are verified - 9 Measured results - 10 Troubleshooting - 11 Limits and open work - 12 Glossary

---

## 1. Goal

Input: one ordinary (monocular, handheld) video of a scene. Output: a **point cloud**, a **textured triangle mesh** and an **offline 3D
viewer**, with no model training. Priorities, in the user's words: speed first, low resources (target: a 4 GB NVIDIA laptop GPU), and a
*correct* point cloud, because the mesh and texture are only as good as the cloud they are built from.

Not goals: metric scale (units are arbitrary), moving scenes, invented geometry (regions no frame saw stay empty).

## 2. How the project evolved

| Phase | What was tried | What happened | Decision |
|---|---|---|---|
| Start | Reconstruct a scene from **edges**: detect edges, lift them to 3D, connect them, and treat closed loops as surfaces, instead of Gaussian splats (primary target was time). | A full edge route was built (stages 4-10, section 7). It found real structure, but edges were incomplete and the surface step stayed weak (held-out contradiction about 44%, an honest limitation). | Kept as an optional route; switched the goal. |
| Phase 1 demo | `tracking_demo/`: visual odometry exactly as described in a supplied document, plus a small map-based tracker ("mini-SLAM"), run on the room video as a stand-in for a live camera. | Frame-to-frame VO drifts because its per-step translation scale is arbitrary; the mini-SLAM fixes scale at initialisation but cannot add landmarks during a pure pan (no parallax). | Demonstrates the limits; not part of the main path. |
| Poses | COLMAP (pycolmap) for camera poses. | CPU-only, 2-4 minutes, non-deterministic, registered only 24-32 of 40 frames on phone pans; low resolution broke the geometry. | Replaced by **Depth Anything 3** (DA3): poses, intrinsics and depth for all keyframes in about a minute on the GPU. COLMAP stays as `backend: colmap`. |
| Goal change | The user looked at a first textured mesh of the hotel room and said "this is exactly what we needed": point cloud -> mesh -> texture, edges off. | Edge stages switched off by default. | Current default pipeline (section 4). |
| Keyframes | A fixed number of keyframes (5, then 20-40) gave a good room cloud but a bad one for an orbit around a small object. | The footage should decide the count. | Overlap-driven keyframe selection + a capture check that says GOOD / FAIR / POOR and retries with stricter overlap. |
| Mesh | Mesh smoothing, silhouette handling. | Taubin smoothing of the mesh created spikes (edges of 20 voxels); a steep-depth mask cut stretched faces from 11% to 2% for 3 points of coverage. | Smooth the **volume**, not the mesh; keep the mask (`silhouette_thr` 0.004). |
| Pose quality | Raw DA3 poses are only about 4-7 px accurate between neighbouring keyframes. | A refinement against tracked corners brought held-out error to about 0.2 px on slow footage. | `stage2b_refine` is part of stage 2. |
| Fast orbit (`source.mp4`) | A garden table filmed in about six loops in 7.75 s. | Neighbouring frames share only about 20% of trackable features. Minimum keyframe gap 2 capped the density; even so the poses disagreed (pose error 44 px, cloud agreement poor). | Every frame as keyframe (44 px -> 4.7 px), then **loop closure + a depth scale per keyframe** (cloud agreement 32% -> 99%). |
| Mesh bug | After the cloud was right, the mesh covered only 20% of it. | Cause found by probing the pipeline step by step: in trimesh 4.12 `mesh.invert()` leaves `face_normals` stale, so the visibility filter deleted 89% of a correct mesh. | Face direction now comes from the volume itself; the mesh is rebuilt with flipped faces. |
| Alternatives checked | **DVLT** (NVIDIA multi-view model; `dvlt.cu` is a third-party C++/CUDA engine): needs a CUDA compiler to build, the official Python version needs PyTorch (not smaller than ours), weights are non-commercial (research use). **DA3-Streaming** (loop closure inside DA3's repo): its place-recognition submodule is empty and it needs `faiss`, `pypose`, `triton`. Other feed-forward models (VGGT, CUT3R, MapAnything, Pi3, MASt3R-SLAM) were surveyed from papers only, not run. | None fit the 4 GB / fast / no-download goals better than the current path. | Not integrated. The DVLT clone was deleted. |
| Usability | Drag-and-drop launcher, live step output, per-run logs, early disk check, readiness check. | The first `run.bat` closed itself on paths containing `(` `)` (batch syntax error before `pause`). | `run.bat` rewritten without parenthesised blocks; `runlog`, `check_env`, disk estimate added. |

## 3. Data flow and output folder

```
video --1--> keyframes --2--> DA3 poses+depth --2b--> refined poses (+ per-keyframe depth scale, loop closure)
                                                  |
                       3 depth confidence <-------+
                              |
                  capture check (GOOD/FAIR/POOR; retries with stricter overlap)
                              |
                11 point cloud ---> 12 TSDF mesh + multi-view texture ---> viewer.html, GLB, PLY, summary.json
```

Default stages are `1,2,3,11,12`; the adaptive loop wraps 1-3. Stages 4-10 only run with `--edges`.

`runs/<video name>/` (or `--out`):

| Path | What it is |
|---|---|
| `keyframes/frame_0000.png ...` | Chosen frames, numbered in keyframe order (not video frame numbers), long side at most `max_side` (1600). |
| `keyframes/meta.json` | Per keyframe: source video frame, time, sharpness, size; plus how the selection decided (triggers: overlap / shift / gap) and boundary feature survival. |
| `da3/scene_cams.npz`, `da3/da3_log.txt` | DA3's raw output (poses, intrinsics, depth) and its full console output. |
| `depth/aligned/*.npy` | One depth map per keyframe at keyframe resolution (float32, NaN where invalid). After refinement these already include the fitted per-keyframe scale. |
| `depth/confidence/*.npy` | Cross-view agreement per depth pixel (stage 3). `depth/predicted/` is only used by the COLMAP backend. |
| `tracking/camera_poses.json` | Final poses (R, t, both directions) and the shared intrinsics; `refined: true` after stage 2b. `camera_poses_da3_raw.json` = before refinement. |
| `geometry/scene_cloud.ply` | The full point cloud (image colours, every 8th pixel). `scene_cloud_agreement.ply` = a 100,000-point subset coloured by how many other keyframes confirm each point. |
| `geometry/scene_mesh.glb` / `.ply` | The textured mesh (one texture per source keyframe) / the same mesh with vertex colours. |
| `viz/mesh_viewer.html` | Offline WebGL2 viewer (orbit, pan, zoom; modes textured / shaded / colour-by-keyframe; a slider jumps to each filming position). |
| `viz/validation.json` | Every stage's measured numbers (the source of the summary). `viz/timings.json` = seconds per stage. |
| `viz/capture_report.json`, `adaptive_attempts.json` | The capture verdict, its evidence and advice; every adaptive attempt and its score. |
| `viz/keyframes_contact_sheet.png`, `depth_aligned_samples.png` | Quick visual checks of the keyframes and depth. |
| `summary.json` | One-page result: verdict, keyframes, timings, cloud and mesh statistics, warnings, the files to open. |
| `logs/<video>_<date>_<time>.log` | A new log every run: everything the window showed. |
| `colmap/`, `edges/` | Empty unless the COLMAP backend or the edge stages are used. |

## 4. Stage reference

### Stage 1 - keyframes (`stage1_keyframes`)
Walks the video; a new keyframe starts when fewer than `overlap_target` (0.8) of the previous keyframe's tracked corners (Lucas-Kanade with a
forward-backward check) are still shared, or the content moved more than `max_shift_frac` (10%) of the long side, or `max_gap_frames` (30)
passed. Within a small window (`sharpness_window` 7 frames) the sharpest frame is taken; frames far below the video's median sharpness
(`min_sharpness_ratio` 0.5) are rejected. Safety limits: 12 to 300 keyframes; videos longer than 2400 frames are analysed on an evenly spaced
subset. Stale `frame_*.png` from an earlier attempt are deleted first. `min_gap_frames` (default 2) is the closest two keyframes may be.

### Stage 2 - poses and depth (`stage2_da3`)
Runs the DA3 runner script from the existing DA3 install as a subprocess (its output is streamed live and saved to `da3/da3_log.txt`). DA3-BASE
processes frames at 504 px in chunks of at most 24 with 5 shared frames, stitched with similarity transforms. DA3's own depth filters are switched
off (the edge-gradient filter would delete depth steps). The result is converted to `tracking/camera_poses.json` (intrinsics rescaled to keyframe
resolution) and `depth/aligned/*.npy`. Before DA3 starts, `_check_disk` estimates the disk this video needs (two float maps per keyframe plus
margin) and stops early with a clear message if it will not fit.

### Stage 2b - global refinement (`stage2b_refine`, `loop_matches`)
DA3 depth is held fixed (it defines the 3D points). For all keyframes at once, a robust (soft-L1) least squares fits: a 6-degree pose correction
per keyframe, **one depth-scale factor per keyframe** (weakly pulled to 1), and one shared focal scale; keyframe 0 is fixed. A residual is "project
a point of keyframe k into keyframe j and compare with where the images say it is". Matches come from
1. **neighbour tracks**: up to 600 tracked corners per keyframe into the next 1-2 keyframes (threaded, 8 workers);
2. **loop-closure matches** between keyframes at least 8 apart that show the same place: ORB features (1500, computed at 800 px) -> a cheap ranking of
   all candidate pairs by ratio-test matches on the 250 strongest features -> the 6 best partners per keyframe verified with full matching and a
   RANSAC fundamental matrix (1.5 px; at least 30 inliers and 25% inlier ratio) -> up to 150 matches per pair, both directions. Pair scoring is
   capped at 80,000 candidates on very long videos.

Half of the matches (tracks and loops separately) are held out and never fitted; the reported improvement is on those. The result is accepted
only if the held-out median improves by at least 10% and the held-out loop error does not get worse; it is flagged unreliable if the held-out
error is still above `reliable_px` (2 px). When accepted, the fitted depth scales are baked into `depth/aligned` once, so every later stage sees
consistent poses and depth.

### Stage 3 - depth confidence (`stage3_depth`)
GPU check of every depth pixel against the neighbouring keyframes (agree within `consistency_rel_tol`, 5% of depth). Stored per pixel and per frame
(`per_frame` fractions in `validation.json`; keyframes below `mesh.min_frame_consistency` are later left out of the mesh).

### Stage 11 - point cloud (`stage11_cloud`, `scene_cloud`)
Lifts every keyframe's depth (every 8th pixel) into the shared world with the refined poses and the image colours. For a 100,000-point subset it
counts how many other keyframes confirm each point (projected there, depth matches within 5%, occluded points not counted). Reading it: a point
seen by two or more other keyframes with high agreement is placed at the same spot by independent views; low agreement means the views disagree.

### Stage 12 - mesh and texture (`stage12_mesh`, `scene_mesh`)
1. **Volume**: a PCA-aligned box around the cloud (0.5-99.5 percentile plus 4%), voxel = longest side / `max_dim` (320). Each keyframe adds
   a truncated signed distance on the GPU (truncation = the larger of 3 voxels and 5% of the median depth); depth pixels at steep relative depth
   gradients (`silhouette_thr` 0.004) are not fused; a voxel counts only if at least `min_views` (2) keyframes saw it. Out of memory retries at 75%
   then 50% resolution. Keyframes whose depth disagrees with their neighbours are left out.
2. **Surface**: the volume is smoothed (Gaussian, 1 voxel), then marching cubes (scikit-image). Small disconnected pieces are dropped (fewer than
   max(200, 0.3% of faces)); any face with an edge over 3 voxels is removed as a safety net.
3. **Direction**: face normals are oriented to the free side of the volume (positive signed distance). Do not use `mesh.invert()` and then read
   `face_normals` (stale in trimesh 4.x); the code rebuilds the mesh instead.
4. **Texture**: per triangle, the keyframe that sees it best (frontal enough, `tex_min_cos` 0.2, close, and unoccluded by a depth test); grazing
   faces get a second pass at `tex_fallback_cos` (0.03); labels are smoothed over neighbouring faces (top 8 candidates); faces pointing away from
   every camera that looks at them are dropped (`drop_unseen_faces`). One texture image per source keyframe, written to GLB and the viewer.
5. **Statistics** go into `validation.json` / `summary.json`: triangles, textured share, share of the cloud within 2 voxels of the mesh, the
   orientation decision, excluded keyframes, timings.

### Capture check (`capture_quality`) - see section 5
### Orchestration (`run_pipeline.py`)
Parses options, creates the output tree, starts the run log, runs the stages (each prints `STEP n STARTED` / `DONE`), runs the adaptive loop,
writes `timings.json` and `summary.json`, optionally opens the viewer. Any exception becomes a printed traceback and exit code 1.

## 5. Capture check and the adaptive loop

After stages 1-3 the pipeline measures, from its own outputs: depth consistency between neighbouring keyframes, pose accuracy on held-out tracked
corners after refinement, rotation between keyframes (median and 90th percentile), and feature matches shared by neighbours (SIFT + RANSAC on 8
sampled pairs).

| Verdict | Depth consistency | Refined pose error | Calibrated on |
|---|---|---|---|
| GOOD | at least 95% | at most 1.0 px | hotel pan: 99%, 0.24 px |
| FAIR | at least 80% | at most 4.0 px | fast orbit of a mouse with 150 keyframes: 84%, 3.1 px |
| POOR | otherwise | | mouse with 40 keyframes: 36%, 19 px |

These thresholds come from two videos; treat them as a starting point. The verdict comes with plain-language advice (go slower, keep 70-80%
overlap, add structure). Unless `--no-adaptive`, a verdict below GOOD makes it redo stages 1-3 with a stricter overlap (0.8, then 0.9, 0.95),
stopping when the score gains less than 0.01 and re-running the best attempt if the last was worse. If the footage already is as dense as
`min_gap_frames` allows, stricter overlap cannot add keyframes and the retry is wasted time (seen on `source.mp4`).

## 6. Settings (`pipeline.yaml`, defaults in `video_to_3d/config.py`)

Unknown keys are rejected, so a typo cannot silently do nothing.

| Section | Main settings |
|---|---|
| `keyframes` | `overlap_target` 0.8, `max_shift_frac` 0.10, `max_gap_frames` 30, `min_gap_frames` 2, `adaptive_targets` [0.8, 0.9, 0.95], `min_keyframes` 12, `max_keyframes` 300, `max_side` 1600, `max_analysis_frames` 2400 |
| `da3` | `root` (existing DA3 install), `process_res` 504, `chunk_max` 24, `overlap` 5 |
| `refine` | `enabled`, `offsets` [1, 2], `corners` 600, `huber_px` 2.0, `max_nfev` 60, `depth_scale`, `scale_prior_px` 20, `loop_closure`, `loop_*` (min gap 8, 1500 features, 6 partners, 30 inliers, 150 matches, 80,000 candidates), `reliable_px` 2.0, `max_fit_tracks` 0 (off; effect unvalidated) |
| `mesh` | `max_dim` 320, `min_views` 2, `tex_side` 960, `silhouette_thr` 0.004, `min_frame_consistency` 0.5, `tex_min_cos` 0.2, `tex_fallback_cos` 0.03, `drop_unseen_faces` true |
| `depth`, `surface` | consistency tolerance (5%) and the confidence floor used by the mesh and cloud (`surface.conf_min`) |
| `backend`, `sfm` | `da3` (default) or `colmap` and its settings |
| `edges_enabled`, `edges`, `lift`, `geometry`, `track`, `graph`, `match` | the optional edge route |

Environment variables: `DA3_ROOT` (DA3 install), `DA3_PYTHON` (interpreter used for DA3 when there is no `tools/venv_da3`).

## 7. Optional components (off by default)

* **Edge route (stages 4-10, `--edges`)**: 4 TEED edge detection (weights in `video_to_3d/models/teed`, MIT licence) -> thinned polylines with
  provisional type tags from depth behaviour (depth step, crease, texture); 5 lift to 3D with side-aware depth (depth is read on both sides of an
  edge, never on the edge) and a control shifted 15 px; 6 two-view edge matching into persistent identities; 7 straight / curved / piecewise
  classification with an anisotropic noise model; 8 persistent 3D edge map with re-fusion over frames; 9 edge graph (shared vertices, T-junctions,
  extension, closed planar loops checked against the depth inside them); 10 surfaces from depth-supported planes plus exact loop faces, scored on
  held-out keyframes against a shifted control. Also `coverage` (where visible edges were lost) and `viewer` (a one-frame 3D edge viewer). Known
  limit: the held-out surface contradiction stayed at about 44%.
* **COLMAP backend** (`stage2_sfm`, `backend: colmap`): CPU, slow, unreliable on pans; also needs `stage3_depth`'s Depth Anything V2 alignment path.
* **`tracking_demo/`**: the Phase-1 VO and mini-SLAM demo (`python -I tracking_demo/run_demo.py`), writes comparison videos and error plots.

## 8. How results are verified

* **Held-out scoring**: pose refinement is scored on matches it never fitted (neighbour tracks and loop matches separately).
* **Controls**: edge-route stages compare against a copy with everything shifted sideways, because edge maps are dense and chance overlap is high.
* **Cross-view agreement**: the cloud's per-point confirmation by other keyframes; the mesh's share of the cloud within 2 voxels.
* **Synthetic ground truth**: `python -m video_to_3d.selftest_refine` (about 30 s, no GPU) simulates a two-loop orbit with drifted poses and per-frame
  depth scale errors and scores the refinement against the known truth. Camera-centre error (orbit radius 3): start 0.132; tracks only 0.347;
  tracks + depth scale 0.027; tracks + loops + depth scale 0.003. It assumes errors are only per-frame scale and drift and that matches have no
  wrong pairs, so it tests the maths, not real footage.
* **Readiness check**: `python run_pipeline.py --check` (GPU, free VRAM, libraries, DA3 files, disk).
* **Not covered**: there is no regression set of varied videos with known poses yet (section 11).

## 9. Measured results

All on one 4 GB RTX 500 Ada laptop GPU. Times include everything the one-command run did.

| Footage | Setting | Result |
|---|---|---|
| Hotel room, slow pan, 13 s | default | 19 keyframes, about 100 s, 99% textured; about 93% of cloud points near the mesh |
| 4 s of the hotel pan | `run.bat`, adaptive | rated GOOD; 380,155 points; 290,951 triangles, 100% textured; 120 s |
| `source.mp4`, 7.75 s garden orbit | 92 keyframes (min gap 2), adaptive | rated POOR; 331 s; 6% depth agreement, 44 px pose error; mesh 24,242 triangles |
| same | every frame (183 keyframes), 800 px, no loop closure | 4.7 px pose error, 61% depth agreement, 170 s; cloud agreement 32% |
| same | every frame + loop closure + depth scale | revisit error 15.5 -> 1.3 px; depth agreement 77%; cloud agreement 99% (94% of points confirmed by 70%+ of views); 221 s; mesh (after the orientation fix) 243,627 triangles, 99% textured, 84% of the cloud near the mesh |
| `source2.mp4` (848x480, 30 fps; the user's run) | `run.bat`, adaptive | rated POOR after two attempts (88 and 163 keyframes); 714 s total of which stage 2 took 586 s; 374,598 points; 156,650 triangles, 98% textured |
| Mouse on a checked cloth, fast orbit | earlier version | FAIR at best with 190 keyframes; not re-run since loop closure |

## 10. Troubleshooting

| You see | Likely cause | What to do |
|---|---|---|
| Window closes immediately (old launcher) | batch syntax error on a path with `(` `)` | use the current `run.bat` |
| `This computer is not ready` | `check_env` found a problem | read the `[FAIL]` line it prints |
| `DA3 install incomplete, missing: ...` | `da3.root` / `DA3_ROOT` wrong | point it at the folder with `scripts/da3_reconstruct.py`, `tools/da3_src`, `tools/da3_models/DA3-BASE` |
| `Not enough disk space for this video` | depth maps for this many keyframes will not fit | free space, use another output folder, or lower `keyframes.max_side` / `max_keyframes` |
| `GPU out of memory at N voxels, retrying coarser` | mesh volume too large for free VRAM | automatic; or lower `mesh.max_dim` |
| CUDA error on a new GPU | the bundled torch (CUDA 11.8) has no kernels for RTX 50-series | needs a newer torch build |
| Verdict POOR | footage too fast, too little overlap, repeating texture | follow the printed advice and `CAPTURE_GUIDE.md`; for very fast footage try `min_gap_frames: 1` |
| `The mesh covers only N% of the point cloud` | keyframes disagree, or (before the fix) normals were stale | check the capture verdict first; make sure `scene_mesh.py` has the volume-based orientation |
| `poses are still off by N px after refinement` | refinement could not reconcile the views | same as POOR; also see stage 2b numbers in `validation.json` |
| `no revisit pairs found` | the camera never returns to a place it saw | normal for a one-way pan; nothing is wrong |

## 11. Limits and open work

* **Capture decides the result.** Slow, smooth, overlapping footage works; fast orbits, little overlap and look-alike repeating texture do not.
* **`min_gap_frames` default 2** caps density at every second frame. Fast footage needs 1; changing the default would slow normal runs, so it is open.
* **Precheck not built**: a roughly 10-second overlap measurement that stops hopeless videos before the 5-10 minute run; also a fallback to every frame
  when the overlap target is unreachable.
* **Cost**: the refinement solve is about 90 s for 183 loop-rich keyframes; DA3 is relaunched on every adaptive retry; depth is stored upsampled to
  keyframe resolution as float32 (native resolution in float16 would be roughly ten times smaller). `refine.max_fit_tracks` could cap the solve but is
  unvalidated.
* **Mesh**: texture comes from a different keyframe per patch, so seams and exposure differences show; the garden mesh was judged from rendered vertices
  and statistics, not inspected interactively; the mesh excludes keyframes that disagree with their neighbours (10 of 183 on the garden run).
* **No regression set**: thresholds and defaults were calibrated on a handful of videos; a set of varied videos (ideally with known poses) is the way
  to know a change generalises.
* **Not built: a portable pack** that runs on another computer without installing anything. Estimated about 6 GB unpacked (PyTorch alone is 5.2 GB), needs
  an NVIDIA driver and the VC++ runtime on the target, and RTX 50-series would not work with the bundled torch. The target machine also needs the DA3
  files (or a bundled copy).
* **Licences and sharing**: DA3-BASE is Apache-2.0 and TEED is MIT; DVLT weights are non-commercial. Sharing code, weights or customer footage outside
  the company needs the owner's and legal review; this document is not legal advice.

## 12. Glossary

* **Keyframe**: a video frame kept for reconstruction. **TSDF**: truncated signed distance volume (positive in front of a surface, negative behind).
* **Pose**: camera position and rotation. **Intrinsics**: focal length and image centre. **Depth map**: distance to the scene per pixel.
* **Chunk**: a group of keyframes DA3 processes at once; chunks are stitched with similarity transforms (rotation, translation, scale).
* **Loop closure**: noticing that the camera has returned to a place seen earlier and forcing the two visits to agree.
* **Held-out**: data the optimiser never saw, used only to score it. **Soft-L1**: a loss that limits the influence of wrong matches.
* **Marching cubes**: the algorithm that turns a volume into a triangle surface. **GLB / PLY**: a textured mesh format / a simple point or vertex-colour format.
* **DA3**: Depth Anything 3. **TEED**: a small edge-detection network. **DVLT**: an NVIDIA multi-view reconstruction model that was evaluated and not used.
