"""Prepare QVHighlights data for the video_span benchmark.

Downloads the official QVHighlights moment-retrieval annotations (from the Moment-DETR repo) and
places `highlight_val_release.jsonl` where the `qvhighlights` adapter loads it
(`QVHighlightsDataset.load_items` reads it line-by-line: each has `qid`, `vid`, `query`,
`duration`, `relevant_windows`). The jsonl is used as-is — no transformation.

Produces, under `--out` (default `data/qvhighlights/`):
  - highlight_val_release.jsonl    val annotations          (config `paths.ann`)
  - videos/                        the clips as `{vid}.mp4`  (config `paths.video_dir`)

The raw videos are one large tarball (~134GB, all splits) from the QVHighlights authors; it is
downloaded + extracted by default. Pass `--skip-videos` to skip it (annotations only) and provide
the clips yourself under the `videos/` dir.

    python -m scripts.prepare_qvhighlights                # annotations + videos (~134GB)
    python -m scripts.prepare_qvhighlights --skip-videos  # annotations only
"""

import argparse
import glob
import os

from scripts.prep_utils import download, untar, out_dir, done_banner

# Official annotations ship in the Moment-DETR repo's data.zip (MIT). This is the upstream raw URL.
ANNO_ZIP_URL = "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl"
VAL_JSONL = "highlight_val_release.jsonl"
# Raw videos (all splits) from the QVHighlights authors; the tarball extracts to `videos/{vid}.mp4`.
VIDEOS_URL = "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"


def main():
    """Download the QVHighlights val annotations and (optionally) the clips.

    Fetches ``highlight_val_release.jsonl`` (used as-is by the ``qvhighlights`` adapter)
    into the ``--out`` directory; unless ``--skip-videos``, also downloads and extracts
    the raw video tarball (~134GB, all splits), then prints the config paths to set.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", default=None, help="output dir (default: <repo>/data/qvhighlights)"
    )
    ap.add_argument(
        "--skip-videos",
        action="store_true",
        help="skip the ~134GB raw-video download (annotations only)",
    )
    args = ap.parse_args()
    root = out_dir(args.out, "qvhighlights")

    ann = os.path.join(root, VAL_JSONL)
    try:
        download(ANNO_ZIP_URL, ann)
    except Exception as e:
        print(f"  [warn] could not fetch annotations automatically ({e}).")
        print(
            "        Get highlight_val_release.jsonl from https://github.com/jayleicn/moment_detr"
            f"\n        (data/ dir) and place it at {ann}"
        )

    videos = os.path.join(root, "videos")
    if args.skip_videos:
        os.makedirs(videos, exist_ok=True)
        print(
            f"\n  [skip] videos (--skip-videos); place the {{vid}}.mp4 clips under: {videos}"
        )
    elif glob.glob(os.path.join(videos, "*.mp4")):
        print(f"\n  [skip] videos already present under {videos}")
    else:
        # tarball extracts `videos/{vid}.mp4` into the dataset root -> exactly paths.video_dir
        tar = download(VIDEOS_URL, os.path.join(root, "qvhilights_videos.tar.gz"))
        untar(tar, root)

    done_banner(
        "QVHighlights", [f"paths.ann:       {ann}", f"paths.video_dir: {videos}"]
    )


if __name__ == "__main__":
    main()
