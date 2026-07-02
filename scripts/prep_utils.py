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
    """Stream a URL to a local file, with a progress readout and atomic replace.

    Downloads to a ``.part`` temp file first and only renames it into place on
    completion, so an interrupted download never leaves a truncated file at ``dest``.

    Args:
        url: the source URL to fetch.
        dest: the destination file path (parent dirs are created).
        skip_if_exists: if ``True`` and ``dest`` already exists and is non-empty, skip
            the download and return immediately.

    Returns:
        The ``dest`` path (whether freshly downloaded or already present).
    """
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
                f.write(chunk)
                done += len(chunk)
                _progress(done, total)
    print()
    os.replace(tmp, dest)
    return dest


def unzip(zip_path: str, dest_dir: str, skip_marker: str | None = None) -> str:
    """Extract a zip archive into a directory, optionally skipping if already done.

    Args:
        zip_path: path to the ``.zip`` archive to extract.
        dest_dir: directory to extract into (created if missing).
        skip_marker: an optional path that, if it already exists, indicates a prior
            successful extraction and causes this call to be skipped.

    Returns:
        The ``dest_dir`` path.
    """
    if skip_marker and os.path.exists(skip_marker):
        print(f"  [skip] {skip_marker} already present")
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    print(f"  extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    return dest_dir


def untar(tar_path: str, dest_dir: str, skip_marker: str | None = None) -> str:
    """Extract a tar archive into a directory, optionally skipping if already done.

    Args:
        tar_path: path to the archive (``.tar`` or ``.tar.gz``; opened with ``r:*``).
        dest_dir: directory to extract into (created if missing).
        skip_marker: an optional path that, if it already exists, indicates a prior
            successful extraction and causes this call to be skipped.

    Returns:
        The ``dest_dir`` path.
    """
    if skip_marker and os.path.exists(skip_marker):
        print(f"  [skip] {skip_marker} already present")
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    print(f"  extracting {tar_path} -> {dest_dir}")
    with tarfile.open(tar_path, "r:*") as t:
        t.extractall(dest_dir)
    return dest_dir


def out_dir(args_out: str | None, name: str) -> str:
    """Resolve and create the output directory for a dataset prep script.

    Args:
        args_out: the script's ``--out`` value; if given it is expanded and made
            absolute, otherwise the default ``<repo>/data/<name>`` is used.
        name: the dataset's directory name under ``data/`` when ``args_out`` is None.

    Returns:
        The absolute output directory path (created if missing).
    """
    root = (
        os.path.abspath(os.path.expanduser(args_out))
        if args_out
        else os.path.join(DATA_ROOT, name)
    )
    os.makedirs(root, exist_ok=True)
    return root


def done_banner(name: str, lines: list[str]) -> None:
    """Print a closing banner telling the user which config paths to set.

    Args:
        name: the dataset name shown in the banner header.
        lines: the ``key: path`` lines to print (the prepared files and where the
            config should point at them).

    Returns:
        None. Side effect: prints to stdout.
    """
    print(f"\n[{name}] ready. Point your config at:")
    for ln in lines:
        print(f"    {ln}")
