"""Seconds-long readiness check for the DA3-only pipeline (no stage is run, nothing is written).

    python check_env.py            # or: python run_pipeline.py --check

Verifies: Python, GPU + free VRAM, the libraries our stages import, the Depth Anything 3 install (interpreter,
runner script, source, weights, metric-depth model) and free disk. Exit code 0 = ready, 1 = something required is missing.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# modules the default pipeline (stages 1-3, 11, 12 + capture check) imports
REQUIRED = ["numpy", "scipy", "cv2", "PIL", "yaml", "torch", "skimage", "trimesh"]
OPTIONAL = {"networkx": "edge stages 4-10 only", "matplotlib": "edge stages 4-10 only",
            "transformers": "stage 3 metric-depth fallback / DA3 runner", "pycolmap": "backend: colmap only"}
MIN_FREE_GB = 2.0          # below this: warning only
HARD_MIN_FREE_GB = 0.4     # below this: not ready


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def check(config_path: str = "pipeline.yaml") -> int:
    bad: list[str] = []
    warn: list[str] = []

    def ok(msg: str) -> None:
        print(f"  [ok]   {msg}")

    def fail(msg: str) -> None:
        print(f"  [FAIL] {msg}")
        bad.append(msg)

    def note(msg: str) -> None:
        print(f"  [warn] {msg}")
        warn.append(msg)

    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    if sys.version_info < (3, 10):
        fail("Python 3.10+ required")

    print("Libraries (pipeline interpreter):")
    for m in REQUIRED:
        ok(m) if _has(m) else fail(f"missing module: {m}")
    for m, why in OPTIONAL.items():
        if not _has(m):
            note(f"{m} not installed ({why})")

    print("GPU:")
    try:
        import torch
        if not torch.cuda.is_available():
            fail("CUDA not available to torch (NVIDIA driver missing, or a CPU-only torch build)")
        else:
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            ok(f"{p.name}, {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free, compute capability {p.major}.{p.minor}")
            # a cubin built for sm_MN runs on any GPU with the same major version and minor >= N (e.g. sm_86 runs on 8.9)
            archs = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
            runs = any(int(a[3:-1]) == p.major and int(a[-1]) <= p.minor for a in archs)
            if not runs:
                note(f"this torch build ({torch.__version__}) has no kernels for compute capability {p.major}.{p.minor}; "
                     f"it will probably fail on this GPU (build supports {', '.join(torch.cuda.get_arch_list())})")
            if free < 1.5 * 2**30:
                note("under 1.5 GB of GPU memory free: close other GPU programs (the mesh stage retries at lower resolution on OOM)")
    except Exception as e:  # noqa: BLE001 - report any import problem as a failed check
        fail(f"torch not usable: {e}")

    print("Depth Anything 3:")
    try:
        from video_to_3d.config import load_config
        cfg = load_config(config_path)
        root = Path(os.environ.get("DA3_ROOT") or cfg.da3.root)
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read {config_path}: {e}")
        root = None
    if root is not None:
        venv_py = root / "tools" / "venv_da3" / "Scripts" / "python.exe"
        py = venv_py if venv_py.exists() else Path(os.environ.get("DA3_PYTHON") or sys.executable)
        for label, p in (("runner script", root / "scripts" / "da3_reconstruct.py"),
                         ("DA3 source", root / "tools" / "da3_src"),
                         ("DA3-BASE weights", root / "tools" / "da3_models" / "DA3-BASE")):
            ok(f"{label}: {p}") if p.exists() else fail(f"missing {label}: {p}")
        ok(f"interpreter for DA3: {py}") if py.exists() else fail(f"DA3 interpreter not found: {py}")
        if py.exists():
            # modules the DA3 runner script imports; checked in the interpreter that will actually run it
            mods = ["numpy", "torch", "cv2", "scipy", "skimage", "trimesh", "PIL", "transformers", "omegaconf", "addict", "einops"]
            code = ("import importlib.util as u,sys;"
                    f"print(','.join(m for m in {mods!r} if u.find_spec(m) is None))")
            try:
                r = subprocess.run([str(py), "-I", "-c", code], capture_output=True, text=True, timeout=60)
                missing = [m for m in r.stdout.strip().split(",") if m]
                (fail(f"DA3 interpreter lacks: {', '.join(missing)}") if missing
                 else ok("DA3 interpreter has the runner's libraries"))
            except Exception as e:  # noqa: BLE001
                note(f"could not inspect the DA3 interpreter: {e}")

    print("Disk:")
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    if free_gb < HARD_MIN_FREE_GB:
        fail(f"only {free_gb:.2f} GB free at {ROOT.anchor}: not enough for even a short video")
    elif free_gb < MIN_FREE_GB:
        note(f"{free_gb:.1f} GB free at {ROOT.anchor}: fine for short videos; a long or high-resolution one needs more "
             f"(the run checks the exact need for your video before the heavy steps and stops early if it will not fit)")
    else:
        ok(f"{free_gb:.1f} GB free at {ROOT.anchor}")

    print()
    if bad:
        print(f"NOT READY: {len(bad)} problem(s)")
        return 1
    print("READY" + (f" ({len(warn)} warning(s) above)" if warn else ""))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(check(sys.argv[1] if len(sys.argv) > 1 else "pipeline.yaml"))
