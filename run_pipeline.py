"""Video -> textured 3D scene mesh.

  python run_pipeline.py --video path/to/video.mp4              # keyframes -> poses/depth -> point cloud -> mesh -> texture
  python run_pipeline.py --video v.mp4 --out runs/myscene       # choose the output folder (default: runs/<video name>)
  python run_pipeline.py --video v.mp4 --edges                  # also run the edge stages 4-10 (off by default)
  python run_pipeline.py --config pipeline.yaml --stages 1-3    # explicit stages from a config
  python run_pipeline.py --video v.mp4 --no-adaptive            # fixed keyframe overlap instead of the quality retry loop

By default the keyframe count is chosen from the footage (overlap-driven) and the result is checked: if the capture is not
GOOD, keyframes are added and stages 1-3 retried (see CAPTURE_GUIDE.md). A summary with warnings is printed and written to
<out>/summary.json. Stage 0 only prints the resolved config and creates the output tree.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
from pathlib import Path

from video_to_3d.config import load_config
from video_to_3d.io_utils import Paths

ROOT = Path(__file__).resolve().parent

STAGES = {
    1: "stage1_keyframes",
    2: "stage2_sfm",
    3: "stage3_depth",
    4: "stage4_edges",
    5: "stage5_lift",
    6: "stage6_match",
    7: "stage7_edge_geometry",
    8: "stage8_edge_tracking",
    9: "stage9_edge_graph",
    10: "stage10_surfaces",
    11: "stage11_cloud",
    12: "stage12_mesh",
}
DEFAULT_STAGES = [1, 2, 3, 11, 12]          # edges (4-10) are optional


def parse_stages(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def summarize(cfg, paths: Paths, timings: dict, adaptive: dict | None) -> dict:
    """What happened, in one place, with plain-language warnings."""
    read = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}   # noqa: E731
    val, cap = read(paths.viz / "validation.json"), read(paths.viz / "capture_report.json")
    mesh, cloud = val.get("scene_mesh", {}), val.get("scene_cloud", {})
    warn = []
    if cap and cap.get("verdict") != "GOOD":
        warn.append(f"Capture rated {cap['verdict']}: " + " ".join(cap.get("advice", [])))
    if mesh:
        if mesh.get("textured_fraction", 1) < 0.9:
            warn.append(f"Only {mesh['textured_fraction']:.0%} of the mesh could be textured from a keyframe.")
        if mesh.get("cloud_points_within_2_voxels_of_mesh", 1) < 0.6:
            warn.append(f"The mesh covers only {mesh['cloud_points_within_2_voxels_of_mesh']:.0%} of the point cloud: "
                        f"keyframes disagree about where the surfaces are, so the mesh will be fragmentary.")
        if mesh.get("keyframes_excluded_for_inconsistency"):
            warn.append(f"{len(mesh['keyframes_excluded_for_inconsistency'])} keyframes were left out of the mesh because "
                        f"their depth disagreed with their neighbours.")
    if cloud and (cloud.get("median_support_among_2plus") or 1) < 0.7:
        warn.append(f"Median cross-view agreement of the point cloud is only {cloud['median_support_among_2plus']:.0%}.")
    out = {"video": str(paths.video), "output_folder": str(paths.root), "capture_verdict": cap.get("verdict"),
           "keyframes": cap.get("keyframes"), "adaptive": adaptive, "timings_s": timings,
           "point_cloud": cloud, "mesh": mesh, "warnings": warn,
           "open_this": str(paths.viz / "mesh_viewer.html"),
           "files": {"textured_mesh_glb": str(paths.root / "geometry" / "scene_mesh.glb"),
                     "mesh_ply": str(paths.root / "geometry" / "scene_mesh.ply"),
                     "point_cloud_ply": str(paths.root / "geometry" / "scene_cloud.ply")}}
    (paths.root / "summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def print_summary(s: dict) -> None:
    print("\n==== SUMMARY ====")
    print(f"video : {s['video']}\noutput: {s['output_folder']}")
    if s["capture_verdict"]:
        print(f"capture quality: {s['capture_verdict']} ({s['keyframes']} keyframes)")
    if s["point_cloud"]:
        print(f"point cloud: {s['point_cloud']['points_total']:,} points")
    if s["mesh"]:
        m = s["mesh"]
        print(f"mesh: {m['triangles']:,} triangles, {m['textured_fraction']:.0%} textured")
    print(f"time: {s['timings_s'].get('total_seconds', '?')} s total")
    for w in s["warnings"]:
        print(f"WARNING: {w}")
    print(f"open: {s['open_this']}")


def main() -> int:
    """Runs the pipeline; any failure is printed (and logged) with its traceback and turned into exit code 1."""
    try:
        return _main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nRUN FAILED - the error is above" + (f"; full log: {LOG_PATH}" if LOG_PATH else ""))
        return 1


LOG_PATH = None


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _main() -> int:
    global LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pipeline.yaml")
    ap.add_argument("--video", help="input video (overrides the config)")
    ap.add_argument("--out", help="output folder (default: runs/<video name> when --video is given)")
    ap.add_argument("--stages", default=None, help="e.g. 1-3 or 2,4 (default: 1,2,3,11,12; 0 = scaffold check only)")
    ap.add_argument("--force", action="store_true", help="re-run stages even if outputs exist")
    ap.add_argument("--edges", action="store_true", help="also run the edge stages 4-10")
    ap.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True,
                    help="after keyframes/poses/depth, assess capture quality and retry with stricter keyframe overlap")
    ap.add_argument("--open", action="store_true", help="open the 3D viewer in the default browser when the run finishes")
    ap.add_argument("--check", action="store_true", help="only check that this computer is ready (GPU, libraries, DA3 install, disk), then exit")
    ap.add_argument("--no-log", action="store_true", help="do not write a log file (default: a new <out>/logs/<video>_<time>.log every run)")
    args = ap.parse_args()

    if args.check:
        from check_env import check
        return check(args.config)

    cfg = load_config(args.config)
    if args.video:
        video = Path(args.video).resolve()
        if not video.exists():
            print(f"ERROR: video not found: {video}")
            return 2
        cfg.video = str(video)
        cfg.workdir = args.out or str(ROOT / "runs" / video.stem)
    elif args.out:
        cfg.workdir = args.out
    paths = Paths(cfg)
    if not args.no_log:
        from video_to_3d import runlog
        LOG_PATH = runlog.start(paths.root / "logs", Path(cfg.video).stem if args.video else "run")
    print(f"video : {paths.video}")
    print(f"output: {paths.root}")
    if args.stages:
        stages = parse_stages(args.stages)
    else:
        stages = list(range(1, 13)) if (args.edges or cfg.edges_enabled) else list(DEFAULT_STAGES)

    if 0 in stages:
        print(json.dumps(dataclasses.asdict(cfg), indent=1))
        print(f"video: {paths.video} (exists={paths.video.exists()})")
        print(f"output tree created under {paths.root}")

    timings: dict[str, float] = {}
    t_all = time.time()

    def exec_stage(s: int, force: bool) -> None:
        name = "stage2_da3" if (s == 2 and cfg.backend == "da3") else STAGES[s]
        mod = importlib.import_module(f"video_to_3d.{name}")
        print(f"\n[{_stamp()}] >>> STEP {s} STARTED: {name}", flush=True)
        t0 = time.time()
        mod.run(cfg, paths, force=force)
        dt = round(time.time() - t0, 1)
        timings[f"{s}:{name}"] = round(timings.get(f"{s}:{name}", 0.0) + dt, 1)
        print(f"[{_stamp()}] <<< STEP {s} DONE in {dt:.1f}s", flush=True)

    adaptive_info = None
    have_base = ((paths.keyframes / "meta.json").exists() and (paths.tracking / "camera_poses.json").exists()
                 and any((paths.depth / "confidence").glob("*.npy")))
    if args.adaptive and all(s in stages for s in (1, 2, 3)):
        if have_base and not args.force:
            print("[adaptive] keyframes, poses and depth already exist in this folder: reusing them (use --force to redo)")
        else:
            from video_to_3d.capture_quality import assess, print_report
            attempts = []
            for target in cfg.keyframes.adaptive_targets:
                cfg.keyframes.overlap_target = target
                print(f"##### adaptive attempt: overlap_target={target}")
                for s in (1, 2, 3):
                    exec_stage(s, True)
                rep = assess(cfg, paths)
                attempts.append({"overlap_target": target, **rep})
                print_report(rep, f"overlap_target={target}: ")
                if rep["verdict"] == "GOOD":
                    break
                if len(attempts) >= 2 and attempts[-1]["score"] - attempts[-2]["score"] < 0.01:
                    print("[adaptive] stricter overlap no longer helps: stopping the retries")
                    break
            best = max(attempts, key=lambda a: a["score"])
            if attempts[-1] is not best:                  # a later, denser attempt was worse: go back to the best one
                print(f"##### last attempt was not the best; re-running overlap_target={best['overlap_target']}")
                cfg.keyframes.overlap_target = best["overlap_target"]
                for s in (1, 2, 3):
                    exec_stage(s, True)
                assess(cfg, paths)
            adaptive_info = {"chosen_overlap_target": best["overlap_target"],
                             "attempts": [{k: a[k] for k in ("overlap_target", "verdict", "keyframes", "score")} for a in attempts]}
            (paths.viz / "adaptive_attempts.json").write_text(
                json.dumps({"chosen_overlap_target": best["overlap_target"], "attempts": attempts}, indent=1, default=float),
                encoding="utf-8")
        stages = [s for s in stages if s > 3]

    for s in stages:
        if s == 0:
            continue
        exec_stage(s, args.force)
    if timings:
        timings["total_seconds"] = round(time.time() - t_all, 1)
        (paths.viz / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")
        print("TIMINGS " + json.dumps(timings))
        if any(s in (11, 12) for s in stages):
            print_summary(summarize(cfg, paths, timings, adaptive_info))
    viewer = paths.viz / "mesh_viewer.html"
    print(f"\n[{_stamp()}] FINISHED")
    print(f"  output folder: {paths.root}")
    print(f"  3D viewer    : {viewer}")
    if LOG_PATH:
        print(f"  log file     : {LOG_PATH}")
    if args.open:
        if viewer.exists():
            import webbrowser
            webbrowser.open(viewer.as_uri())
        else:
            print(f"viewer not found (the mesh stage did not produce it): {viewer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
