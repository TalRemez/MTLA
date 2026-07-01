"""Shared helpers for the dataset-preparation scripts (scripts/prepare_*.py).

Small, dependency-light download + extract utilities so each prepare script stays readable.
Default output root is the repo-relative ``data/`` directory (gitignored); pass ``--out`` to
override. The prepared files match exactly what the dataset adapters' ``load_items`` expect and
what ``configs/*.yaml`` point at.
"""
from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")


def _progress(done, total):
    if total <= 0:
        return
    pct = 100 * done / total
    sys.stdout.write(f"\r    {pct:5.1f}%  ({done >> 20} / {total >> 20} MiB)")
    sys.stdout.flush()


def download(url: str, dest: str, skip_if_exists: bool = True) -> str:
    """Download ``url`` to ``dest`` (skipping if it already exists). Returns ``dest``."""
    if skip_if_exists and os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [skip] {dest} already exists")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk); done += len(chunk); _progress(done, total)
    print()
    os.replace(tmp, dest)
    return dest


def unzip(zip_path: str, dest_dir: str, skip_marker: str | None = None) -> str:
    """Extract ``zip_path`` into ``dest_dir``. If ``skip_marker`` (a path) exists, skip."""
    if skip_marker and os.path.exists(skip_marker):
        print(f"  [skip] {skip_marker} already present")
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    print(f"  extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    return dest_dir


def untar(tar_path: str, dest_dir: str, skip_marker: str | None = None) -> str:
    """Extract ``tar_path`` (``.tar`` / ``.tar.gz``) into ``dest_dir``. If ``skip_marker`` exists, skip."""
    if skip_marker and os.path.exists(skip_marker):
        print(f"  [skip] {skip_marker} already present")
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    print(f"  extracting {tar_path} -> {dest_dir}")
    with tarfile.open(tar_path, "r:*") as t:
        t.extractall(dest_dir)
    return dest_dir


def out_dir(args_out: str | None, name: str) -> str:
    """Resolve the per-dataset output directory (``--out`` override or ``<repo>/data/<name>``)."""
    root = os.path.abspath(os.path.expanduser(args_out)) if args_out else os.path.join(DATA_ROOT, name)
    os.makedirs(root, exist_ok=True)
    return root


def done_banner(name: str, lines: list[str]) -> None:
    print(f"\n[{name}] ready. Point your config at:")
    for ln in lines:
        print(f"    {ln}")
