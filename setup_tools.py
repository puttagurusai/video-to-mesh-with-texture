"""Download and install the third-party pieces this project needs (and the Python packages), into the tools\\ folder.

  python setup_tools.py --check                  show what is installed / missing, change nothing
  python setup_tools.py --dry-run                show exactly what would be downloaded, change nothing
  python setup_tools.py                          install everything that is missing (asks before downloading)
  python setup_tools.py --yes                    same, without the question
  python setup_tools.py --only da3-model         install just these components
  python setup_tools.py --skip deps              install everything except these
  python setup_tools.py --with-optional          also the optional parts (edge route, COLMAP backend)
  python setup_tools.py --from-existing "D:\\old\\project"   reuse tools\\da3_src and tools\\da3_models\\DA3-BASE from an
                                                 existing install (hard links, so no extra disk space) instead of downloading
  python setup_tools.py --force --only da3-src   reinstall one component (the old folder is renamed, never deleted)

Components (names for --only / --skip):
  deps           Python packages from requirements.txt (pip, this Python; torch and numpy are pinned so pip cannot replace them)
  da3-src        Depth Anything 3 source at a pinned commit (Apache-2.0)        -> tools\\da3_src
  da3-model      DA3-BASE weights (Apache-2.0), SHA-256 verified                -> tools\\da3_models\\DA3-BASE
  dav2-metric    Depth-Anything-V2-Metric-Indoor-Small, SHA-256 verified        -> the Hugging Face cache (see note)
Optional (only with --with-optional or --only):
  optional-deps  networkx + pycolmap from requirements-optional.txt              (edge route / COLMAP backend)
  dav2-small     Depth-Anything-V2-Small (Apache-2.0), SHA-256 verified          -> the Hugging Face cache (COLMAP backend only)

What is NOT downloaded here: the DA3 runner scripts (they are in this repo: scripts\\) and the TEED edge weights (in the repo:
video_to_3d\\models\\teed, edge route only). The NVIDIA driver and the Visual C++ runtime must be installed on the computer.

Why the Depth-Anything-V2 metric model: the DA3 runner (scripts\\da3_reconstruct.py) loads it, offline, to estimate an absolute
scale even in --poses-only mode. This project does not use that scale (units stay arbitrary), but the runner still needs the model.
The scripts load it by name from the standard Hugging Face cache (%USERPROFILE%\\.cache\\huggingface\\hub, or HF_HUB_CACHE /
HF_HOME if set), so this script writes it there with real file copies (no symlinks; those need a privilege many Windows accounts lack).

Licences (from the Hugging Face model metadata read when this was written): DA3-BASE apache-2.0; Depth-Anything-V2-Small apache-2.0;
Depth-Anything-V2-Metric-Indoor-Small states NO licence in its metadata, so check its model card before sharing or shipping it.
The larger DA3 checkpoints (Large/Giant/Nested) are NOT Apache-2.0 and are not used. Not legal advice: the owner and legal review
decide what may be shared.

Safety: downloads come only from the URLs pinned below (a GitHub archive of a fixed commit, Hugging Face repos, PyPI / the PyTorch
wheel index through pip). Files with a known SHA-256 are verified before use; a mismatch aborts that component. A GitHub archive of
a commit has no stable hash, so it is only checked structurally. Nothing is uploaded anywhere. Standard library only (the
Hugging Face downloads import huggingface_hub, which the 'deps' step installs).

Tested when written: --check, --dry-run, linking an existing install, the archive download path (DA3 source), the Hugging Face
download path on small files, and the pinned SHA-256 values against the Hugging Face API. A full install on a clean machine
(PyTorch wheels, the 520 MB model download) has NOT been run.
"""
import argparse
import hashlib
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
REQ = ROOT / "requirements.txt"
REQ_OPT = ROOT / "requirements-optional.txt"

DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_SRC_URL = f"https://github.com/ByteDance-Seed/Depth-Anything-3/archive/{DA3_COMMIT}.zip"
DA3_MODEL = dict(repo="depth-anything/DA3-BASE", patterns=["config.json", "model.safetensors", "*.md"],
                 sha256="e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5", size_mb=517)
HF_NEEDS = ("config.json", "model.safetensors", "preprocessor_config.json")
HF_PATTERNS = ["config.json", "model.safetensors", "preprocessor_config.json", "*.md"]
DAV2 = {
    "dav2-metric": dict(repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", size_mb=95,
                        sha256="e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3"),
    "dav2-small": dict(repo="depth-anything/Depth-Anything-V2-Small-hf", size_mb=95,
                       sha256="3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"),
}
ORDER = ["deps", "da3-src", "da3-model", "dav2-metric", "optional-deps", "dav2-small"]
OPTIONAL = {"optional-deps", "dav2-small"}
APPROX_MB = {"deps": 4000, "da3-src": 60, "da3-model": 517, "dav2-metric": 95, "optional-deps": 100, "dav2-small": 95}


# ---------------------------------------------------------------------------------------------------- helpers
def say(msg=""):
    print(msg, flush=True)


def sha256_of(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest, expected_sha=None):
    """Download to <dest>.part, verify SHA-256 if given, then rename. Raises on any problem."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "video-to-3d-setup"})
    t0 = last = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 5:
                last = time.time()
                pct = f"{100 * got / total:.0f}%" if total else ""
                say(f"    {got / 2**20:7.1f} MB {pct}  ({got / 2**20 / max(time.time() - t0, 1e-6):.1f} MB/s)")
    if expected_sha:
        got_sha = sha256_of(part)
        if got_sha.lower() != expected_sha.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {url}\n  expected {expected_sha}\n  got      {got_sha}\n  (file deleted; nothing installed)")
        say("    SHA-256 verified")
    else:
        say("    (no pinned hash for this download; the structure is checked instead)")
    part.replace(dest)
    return dest


def safe_extract(zip_path, dest):
    """Extract a zip, refusing members that would land outside `dest` (zip-slip)."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            out = (dest / m.filename).resolve()
            if dest != out and dest not in out.parents:
                raise RuntimeError(f"unsafe path in archive: {m.filename}")
        z.extractall(dest)


def single_top_dir(tree):
    items = list(Path(tree).iterdir())
    return items[0] if len(items) == 1 and items[0].is_dir() else Path(tree)


def rename_old(target):
    old = target.with_name(target.name + ".bak_" + time.strftime("%Y%m%d_%H%M%S"))
    target.rename(old)
    say(f"    existing folder renamed to {old.name} (delete it yourself when you no longer need it)")


def move_into_place(src, target, force):
    """Move `src` to `target`. An existing target is renamed (never deleted) and reported."""
    target = Path(target)
    if target.exists():
        if not force:
            raise RuntimeError(f"{target} already exists (use --force to replace it; the old one is renamed, not deleted)")
        rename_old(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(target))


def run(cmd, **kw):
    say("    $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def req_pins(path):
    """{normalised name: version} for lines like name==version (ignores comments, options, unpinned lines)."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#;]+)", line)
        if m:
            pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


def installed_version(name, python=None):
    """Version of a package in `python` (default: this interpreter), or None."""
    if python is None or Path(python).resolve() == Path(sys.executable).resolve():
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None
    r = subprocess.run([str(python), "-I", "-c", f"import importlib.metadata as m;print(m.version({name!r}))"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def hf_cache_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_has(repo, cache=None, needs=HF_NEEDS):
    """True if refs/main of the repo resolves to a snapshot folder holding the needed files (what offline loading needs)."""
    base = Path(cache or hf_cache_dir()) / ("models--" + repo.replace("/", "--"))
    ref = base / "refs" / "main"
    if not ref.is_file():
        return False
    snap = base / "snapshots" / ref.read_text().strip()
    return all((snap / n).is_file() for n in needs)


def install_into_hf_cache(repo, src_dir, sha, cache=None):
    """Copy files (real copies, no symlinks) into the HF cache layout so offline from_pretrained(repo) works."""
    base = Path(cache or hf_cache_dir()) / ("models--" + repo.replace("/", "--"))
    snap = base / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    for p in Path(src_dir).iterdir():
        if p.is_file():
            shutil.copy2(p, snap / p.name)
    (base / "refs").mkdir(parents=True, exist_ok=True)
    (base / "refs" / "main").write_text(sha)
    return snap


def free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


def read_git_head(folder):
    """Commit id of a git checkout without running git (reads .git/HEAD), or None."""
    head = Path(folder) / ".git" / "HEAD"
    try:
        text = head.read_text().strip()
        if text.startswith("ref:"):
            ref = Path(folder) / ".git" / text.split(None, 1)[1]
            return ref.read_text().strip() if ref.is_file() else None
        return text
    except OSError:
        return None


def link_tree(src, dst):
    """Recreate `src` under `dst` with hard links (no extra disk space); copy when linking is not possible."""
    src, dst = Path(src), Path(dst)
    linked = copied = 0
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, out)
                linked += 1
            except OSError:
                shutil.copy2(p, out)
                copied += 1
    return linked, copied


# ------------------------------------------------------------------------------------------- component status
def status(key, python):
    """(ok, detail) for one component; never changes anything."""
    if key in ("deps", "optional-deps"):
        req = REQ if key == "deps" else REQ_OPT
        bad = []
        for name, want in req_pins(req).items():
            have = installed_version(name, python)
            if have is None:
                bad.append(f"{name} missing")
            elif have.split("+")[0] != want.split("+")[0]:
                bad.append(f"{name} {have} (tested {want})")
        return (not bad, "all pinned packages present" if not bad else "; ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))
    if key == "da3-src":
        ok = (TOOLS / "da3_src" / "src" / "depth_anything_3").is_dir()
        head = read_git_head(TOOLS / "da3_src")
        note = "" if head in (None, DA3_COMMIT) else f" (commit {head[:8]}, tested {DA3_COMMIT[:8]})"
        return (ok, ("tools\\da3_src present" + note) if ok else "missing: tools\\da3_src\\src\\depth_anything_3")
    if key == "da3-model":
        f = TOOLS / "da3_models" / "DA3-BASE" / "model.safetensors"
        return (f.is_file() and (TOOLS / "da3_models" / "DA3-BASE" / "config.json").is_file(),
                "tools\\da3_models\\DA3-BASE present" if f.is_file() else "missing: tools\\da3_models\\DA3-BASE\\model.safetensors")
    if key in DAV2:
        repo = DAV2[key]["repo"]
        ok = hf_cache_has(repo)
        return (ok, f"in the Hugging Face cache ({hf_cache_dir()})" if ok else f"missing in the Hugging Face cache: {repo}")
    raise KeyError(key)


# ------------------------------------------------------------------------------------------- component installers
def install_deps(python, req, tmp):
    say(f"  pip install -r {req.name}   (python: {python}; torch / torchvision / numpy pinned so pip cannot replace them)")
    cons = tmp / "constraints.txt"
    tmp.mkdir(parents=True, exist_ok=True)
    pins = req_pins(REQ)
    cons.write_text("".join(f"{n}=={v}\n" for n, v in pins.items() if n in ("torch", "torchvision", "numpy")), encoding="ascii")
    r = run([python, "-m", "pip", "install", "--disable-pip-version-check", "-c", cons, "-r", req])
    if r.returncode != 0:
        raise RuntimeError("pip install failed (see the output above)")


def install_da3_src(force, tmp):
    say(f"  Depth Anything 3 source, commit {DA3_COMMIT[:8]}: {DA3_SRC_URL}")
    tmp.mkdir(parents=True, exist_ok=True)
    zpath = download(DA3_SRC_URL, tmp / "da3_src.zip", None)
    ex = tmp / "x_da3"
    shutil.rmtree(ex, ignore_errors=True)
    safe_extract(zpath, ex)
    top = single_top_dir(ex)
    if not (top / "src" / "depth_anything_3").is_dir():
        raise RuntimeError("unexpected archive layout: src/depth_anything_3 not found")
    move_into_place(top, TOOLS / "da3_src", force)
    zpath.unlink(missing_ok=True)
    shutil.rmtree(ex, ignore_errors=True)


def hf_download(repo, patterns, dest):
    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub is not installed: run the 'deps' step first")
    sha = HfApi().model_info(repo).sha
    snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=patterns)
    return sha


def verify_file(path, sha):
    got = sha256_of(path)
    if got.lower() != sha.lower():
        raise RuntimeError(f"SHA-256 of {Path(path).name} differs from the recorded one:\n  expected {sha}\n  got      {got}\n"
                           "  The file was kept for inspection; the repo may have been updated. Ask the project owner before using it.")
    say("    SHA-256 verified")


def install_da3_model(force):
    say(f"  {DA3_MODEL['repo']} (Apache-2.0) -> tools\\da3_models\\DA3-BASE  (about {DA3_MODEL['size_mb']} MB)")
    dest = TOOLS / "da3_models" / "DA3-BASE"
    if dest.exists() and force:
        rename_old(dest)
    # local_dir download: avoids the Hugging Face cache symlinks (WinError 1314 on accounts without the privilege)
    hf_download(DA3_MODEL["repo"], DA3_MODEL["patterns"], dest)
    f = dest / "model.safetensors"
    if not f.is_file():
        raise RuntimeError("model.safetensors was not downloaded")
    verify_file(f, DA3_MODEL["sha256"])


def install_dav2(key, force, tmp):
    m = DAV2[key]
    say(f"  {m['repo']} -> Hugging Face cache {hf_cache_dir()}  (about {m['size_mb']} MB)")
    if hf_cache_has(m["repo"]) and not force:
        say("    already in the cache")
        return
    tmp.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dav2_", dir=tmp))
    try:
        sha = hf_download(m["repo"], HF_PATTERNS, work)
        if not all((work / n).is_file() for n in HF_NEEDS):
            raise RuntimeError(f"{m['repo']}: expected files missing after download")
        verify_file(work / "model.safetensors", m["sha256"])
        snap = install_into_hf_cache(m["repo"], work, sha)
        say(f"    installed into {snap}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def link_existing(key, existing, force):
    """Reuse da3_src / DA3-BASE from an existing install by hard-linking them into tools\\."""
    existing = Path(existing)
    if key == "da3-src":
        src, dst = existing / "tools" / "da3_src", TOOLS / "da3_src"
        if not (src / "src" / "depth_anything_3").is_dir():
            raise RuntimeError(f"{src} does not look like the Depth Anything 3 source (src\\depth_anything_3 missing)")
        head = read_git_head(src)
        if head and head != DA3_COMMIT:
            say(f"    WARNING: that copy is at commit {head[:8]}, not the tested {DA3_COMMIT[:8]}")
    else:
        src, dst = existing / "tools" / "da3_models" / "DA3-BASE", TOOLS / "da3_models" / "DA3-BASE"
        if not (src / "model.safetensors").is_file():
            raise RuntimeError(f"{src}\\model.safetensors not found")
    say(f"  linking {src} -> {dst}")
    if dst.exists():
        if not force:
            raise RuntimeError(f"{dst} already exists (use --force to replace it; the old one is renamed, not deleted)")
        rename_old(dst)
    linked, copied = link_tree(src, dst)
    say(f"    {linked} files hard-linked, {copied} copied (hard links use no extra disk space)")
    if key == "da3-model":
        verify_file(dst / "model.safetensors", DA3_MODEL["sha256"])


# ------------------------------------------------------------------------------------------------------- main
def main():
    global TOOLS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="show the status of every component and exit")
    ap.add_argument("--dry-run", action="store_true", help="show what would be downloaded/installed and exit")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--only", default="", help="comma-separated components: " + ", ".join(ORDER))
    ap.add_argument("--skip", default="", help="comma-separated components to skip")
    ap.add_argument("--with-optional", action="store_true", help="also install the optional components (edge route, COLMAP backend)")
    ap.add_argument("--force", action="store_true", help="reinstall selected components (old folders are renamed, not deleted)")
    ap.add_argument("--from-existing", default="", metavar="DIR", help="reuse tools\\da3_src and tools\\da3_models\\DA3-BASE of an existing install")
    ap.add_argument("--tools-dir", default="", help="install into this folder instead of <project>\\tools (for testing)")
    ap.add_argument("--python", default=sys.executable, help="Python used for the pip steps (default: this one)")
    a = ap.parse_args()
    if a.tools_dir:
        TOOLS = Path(a.tools_dir).resolve()
    tmp = TOOLS / "_setup_tmp"          # scratch space on the same drive as the targets; removed afterwards

    if os.name != "nt":
        say("NOTE: this project has only been run on Windows; the steps below are not tested elsewhere.")
    if sys.version_info < (3, 10):
        say(f"WARNING: Python {sys.version.split()[0]} is older than anything tested (3.13.7).")
    elif sys.version_info[:2] != (3, 13):
        say(f"NOTE: tested with Python 3.13.7; you have {sys.version.split()[0]}. The pinned wheels may not exist for your version.")

    only = {x.strip() for x in a.only.split(",") if x.strip()}
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    for name in only | skip:
        if name not in ORDER:
            raise SystemExit(f"unknown component '{name}'. Choose from: {', '.join(ORDER)}")
    selected = [k for k in ORDER if (not only or k in only) and k not in skip]
    if not only and not a.with_optional:
        selected = [k for k in selected if k not in OPTIONAL]

    say(f"Project: {ROOT}\nTools folder: {TOOLS}\n")
    st = {k: status(k, a.python) for k in ORDER}
    say("Status:")
    for k in ORDER:
        ok, detail = st[k]
        say(f"  [{'OK     ' if ok else 'MISSING'}] {k:14s} {detail}{'   (optional)' if k in OPTIONAL else ''}")
    if a.check:
        return 0

    todo = [k for k in selected if a.force or not st[k][0]]
    if not todo:
        say("\nNothing to do: everything selected is already installed.")
        return 0
    from_existing = bool(a.from_existing)
    def est_mb(k):
        if from_existing and k in ("da3-src", "da3-model"):
            return 0
        if k == "deps" and installed_version("torch", a.python):     # PyTorch is the bulk of the 'deps' download
            return 150
        return APPROX_MB[k]
    need_mb = sum(est_mb(k) for k in todo)
    free = free_gb(TOOLS)
    say(f"\nWill install: {', '.join(todo)}   (download about {need_mb / 1024:.1f} GB; free disk {free:.1f} GB)")
    if free * 1024 < need_mb * 1.6 + 512:
        say("WARNING: free disk looks too small for this (installed size is larger than the download).")
    for k in todo:
        say(f"- {k}")
        if k in ("deps", "optional-deps"):
            say(f"  pip packages from {REQ.name if k == 'deps' else REQ_OPT.name} (PyPI and download.pytorch.org)")
        elif k in ("da3-src", "da3-model") and from_existing:
            say(f"  link from {a.from_existing}")
        elif k == "da3-src":
            say(f"  {DA3_SRC_URL}")
        elif k == "da3-model":
            say(f"  Hugging Face: {DA3_MODEL['repo']}")
        else:
            say(f"  Hugging Face: {DAV2[k]['repo']}")
    if a.dry_run:
        say("\n--dry-run: nothing was downloaded or changed.")
        return 0
    if not a.yes:
        if input("\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
            say("Cancelled. Nothing was changed.")
            return 1

    failures = []
    for k in todo:
        say(f"\n== {k}")
        try:
            if k == "deps":
                install_deps(a.python, REQ, tmp)
            elif k == "optional-deps":
                install_deps(a.python, REQ_OPT, tmp)
            elif k in ("da3-src", "da3-model") and from_existing:
                link_existing(k, a.from_existing, a.force)
            elif k == "da3-src":
                install_da3_src(a.force, tmp)
            elif k == "da3-model":
                install_da3_model(a.force)
            else:
                install_dav2(k, a.force, tmp)
        except Exception as e:  # report and carry on with the independent components
            failures.append((k, str(e)))
            say(f"  FAILED: {e}")
    shutil.rmtree(tmp, ignore_errors=True)

    say("\n== final status")
    st = {k: status(k, a.python) for k in ORDER}
    for k in ORDER:
        say(f"  [{'OK     ' if st[k][0] else 'MISSING'}] {k:14s} {st[k][1]}")
    if failures:
        say("\nFailed components: " + ", ".join(k for k, _ in failures) +
            "\nFix the cause (see the messages above) and run this script again; finished components are skipped.")
        return 1
    say("\nDone. Check the computer with:  python run_pipeline.py --check     then run:  run.bat \"path\\to\\video.mp4\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
