"""Prepare Charades-STA data for the video_span benchmark.

Builds the Charades-STA test parquet the `charades` adapter loads (`CharadesDataset.load_items`
reads it with pandas; each row: `video` = `{id}.mp4`, `caption`, `timestamp = [start, end]`).

The original Charades-STA test annotations (Gao et al., TALL) are open, so by default the script
downloads them and builds the parquet with no arguments. Each line is `VIDID start end##caption`.
Two overrides:
  - `--from-annotations PATH` — convert a local copy of `charades_sta_test.txt` instead of fetching.
  - `--hf` — fetch the HuggingFace `lmms-lab/Charades-STA` parquet (gated — needs `huggingface_hub`
    + an HF token / accepted terms): `pip install huggingface_hub` then `huggingface-cli login`.

Produces, under `--out` (default `data/charades/`):
  - charades_sta_test.parquet      test annotations         (config `paths.data`)
  - Charades_v1_480/               the clips as `{id}.mp4`   (config `paths.video_dir`)

The Charades_v1_480 clips (~15GB zip, open) from the Charades authors are downloaded + extracted by
default. Pass `--skip-videos` to skip them (annotations only) and provide the clips yourself.

    python -m scripts.prepare_charades                       # annotations + videos (~15GB)
    python -m scripts.prepare_charades --skip-videos         # annotations only
    python -m scripts.prepare_charades --from-annotations charades_sta_test.txt   # local ann copy
    python -m scripts.prepare_charades --hf                  # HF lmms-lab/Charades-STA (needs auth)
"""
import argparse
import glob
import os
import re

from scripts._prep_utils import download, unzip, out_dir, done_banner

# Original Charades-STA test split (Gao et al., TALL), `VIDID start end##caption`. Open, no auth.
STA_TEST_URL = "https://raw.githubusercontent.com/26hzhang/VSLNet/master/data/dataset/charades/charades_sta_test.txt"
# Charades_v1_480 clips (~15GB, open S3); the zip extracts to `Charades_v1_480/{id}.mp4`.
VIDEOS_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
HF_REPO = "lmms-lab/Charades-STA"
HF_FILE = "data/test-00000-of-00001.parquet"


def from_annotations(txt_path: str, out_parquet: str) -> int:
    """Convert original Charades-STA `VIDID start end##caption` lines -> the test parquet."""
    import pandas as pd
    rows = []
    with open(txt_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            head, _, caption = ln.partition("##")
            m = re.match(r"(\S+)\s+([\d.]+)\s+([\d.]+)", head)
            if not m:
                continue
            vid, s, e = m.group(1), float(m.group(2)), float(m.group(3))
            rows.append({"video": f"{vid}.mp4", "caption": caption.strip(),
                         "timestamp": [s, e]})
    df = pd.DataFrame(rows, columns=["video", "caption", "timestamp"])
    df.to_parquet(out_parquet)
    return len(df)


def from_hf(out_parquet: str) -> int:
    """Fetch the lmms-lab/Charades-STA test parquet via huggingface_hub (needs auth/accepted terms)."""
    import shutil
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    shutil.copy(src, out_parquet)
    import pandas as pd
    return len(pd.read_parquet(out_parquet))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default: <repo>/data/charades)")
    ap.add_argument("--from-annotations", default=None,
                    help="convert a LOCAL charades_sta_test.txt instead of downloading it")
    ap.add_argument("--hf", action="store_true",
                    help="fetch from HuggingFace lmms-lab/Charades-STA (needs HF auth)")
    ap.add_argument("--skip-videos", action="store_true",
                    help="skip the ~15GB Charades_v1_480 video download (annotations only)")
    args = ap.parse_args()
    root = out_dir(args.out, "charades")
    parquet = os.path.join(root, "charades_sta_test.parquet")

    n = None
    if args.hf:
        try:
            n = from_hf(parquet)
            print(f"  fetched {parquet} from HF {HF_REPO}  ({n} queries)")
        except Exception as e:
            print(f"  [error] HF fetch failed ({type(e).__name__}: {str(e)[:120]}).")
            print(f"          {HF_REPO} is gated — accept its terms + `huggingface-cli login`,")
            print(f"          or drop --hf to download the open Charades-STA annotations instead.")
    else:
        # Default: use the open annotations. Download them unless a local copy was given.
        txt = args.from_annotations
        if txt is None:
            txt = os.path.join(root, "charades_sta_test.txt")
            download(STA_TEST_URL, txt)
        n = from_annotations(txt, parquet)
        print(f"  built {parquet} from {txt}  ({n} queries)")

    videos = os.path.join(root, "Charades_v1_480")
    if args.skip_videos:
        os.makedirs(videos, exist_ok=True)
        print(f"\n  [skip] videos (--skip-videos); place the {{id}}.mp4 clips under: {videos}")
    elif glob.glob(os.path.join(videos, "*.mp4")):
        print(f"\n  [skip] videos already present under {videos}")
    else:
        # zip extracts `Charades_v1_480/{id}.mp4` into the dataset root -> exactly paths.video_dir
        zp = download(VIDEOS_URL, os.path.join(root, "Charades_v1_480.zip"))
        unzip(zp, root)

    done_banner("Charades-STA", [f"paths.data:      {parquet}",
                                 f"paths.video_dir: {videos}"])


if __name__ == "__main__":
    main()
