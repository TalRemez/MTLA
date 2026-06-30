"""Prepare QVHighlights data for the video_span benchmark.

Downloads the official QVHighlights moment-retrieval annotations (from the Moment-DETR repo) and
places `highlight_val_release.jsonl` where the `qvhighlights` adapter loads it
(`QVHighlightsDataset.load_items` reads it line-by-line: each has `qid`, `vid`, `query`,
`duration`, `relevant_windows`). The jsonl is used as-is — no transformation.

Produces, under `--out` (default `data/qvhighlights/`):
  - highlight_val_release.jsonl    val annotations          (config `paths.ann`)
  - videos/                        you provide the clips    (config `paths.video_dir`)

Videos are large and distributed separately (Google Drive); see the printed instructions and the
Moment-DETR README. Place the val clips as `{vid}.mp4` under the `videos/` dir.

    python scripts/prepare_qvhighlights.py        # -> data/qvhighlights/
"""
import argparse
import os

from _prep_utils import download, out_dir, done_banner

# Official annotations ship in the Moment-DETR repo's data.zip (MIT). This is the upstream raw URL.
ANNO_ZIP_URL = "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl"
VAL_JSONL = "highlight_val_release.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default: <repo>/data/qvhighlights)")
    args = ap.parse_args()
    root = out_dir(args.out, "qvhighlights")

    ann = os.path.join(root, VAL_JSONL)
    try:
        download(ANNO_ZIP_URL, ann)
    except Exception as e:
        print(f"  [warn] could not fetch annotations automatically ({e}).")
        print("        Get highlight_val_release.jsonl from https://github.com/jayleicn/moment_detr"
              f"\n        (data/ dir) and place it at {ann}")

    videos = os.path.join(root, "videos")
    os.makedirs(videos, exist_ok=True)
    print(f"\n  Videos: QVHighlights clips are distributed separately (large; Google Drive).")
    print(f"  Follow the Moment-DETR README data instructions, then place the val clips as")
    print(f"  {{vid}}.mp4 under: {videos}")

    done_banner("QVHighlights", [f"paths.ann:       {ann}",
                                 f"paths.video_dir: {videos}"])


if __name__ == "__main__":
    main()
