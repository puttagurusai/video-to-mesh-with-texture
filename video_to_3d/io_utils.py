"""Output-tree layout, JSON helpers and the validation.json metric log."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config

SUBDIRS = [
    "keyframes",
    "colmap",
    "depth/predicted",
    "depth/aligned",
    "depth/confidence",
    "edges/probability",
    "edges/polylines",
    "edges/polylines3d",
    "tracking",
    "viz",
]


class Paths:
    """Resolved locations for every pipeline artifact."""

    def __init__(self, cfg: Config):
        self.root = cfg.root
        self.video = (self.root / cfg.video).resolve()
        for sub in SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> Path:
        # Attribute access like paths.keyframes -> <root>/keyframes (name with '_' -> '/')
        if name.startswith("_"):
            raise AttributeError(name)
        return self.root / name.replace("__", "/")

    def frame_name(self, idx: int) -> str:
        return f"frame_{idx:04d}"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return super().default(o)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=_NumpyEncoder, indent=1), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def record_metrics(paths: Paths, stage: str, metrics: dict[str, Any]) -> None:
    """Accumulate per-stage gate numbers in output/validation.json-style log (viz/validation.json)."""
    log_path = paths.viz / "validation.json"
    log = read_json(log_path) if log_path.exists() else {}
    log[stage] = metrics
    write_json(log_path, log)
