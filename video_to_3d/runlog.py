"""Per-run log: everything printed to the console is also written to a new log file, every run.

`start(log_dir, stem)` replaces sys.stdout/sys.stderr with a tee, so all existing `print` calls (and tracebacks) land in
`<log_dir>/<stem>_<YYYYmmdd_HHMMSS>.log` without touching the stages. A new file is created per run; old logs are kept.
"""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path


class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        try:
            self._stream.write(s)
            self._stream.flush()                 # steps must show up live, not when the buffer fills
        except Exception:                        # a closed/odd console must never stop the run
            pass
        self._fh.write(s)
        return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        self._fh.flush()

    def __getattr__(self, name):                 # encoding, isatty, ... come from the real stream
        return getattr(self._stream, name)


def start(log_dir: Path, stem: str) -> Path:
    for s in (sys.stdout, sys.stderr):           # characters a Windows console cannot show must not crash a print
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = open(path, "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)
    print(f"# log file : {path}")
    print(f"# started  : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"# command  : {' '.join(sys.argv)}")
    print(f"# python   : {sys.version.split()[0]} at {sys.executable}")
    return path


def stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")
