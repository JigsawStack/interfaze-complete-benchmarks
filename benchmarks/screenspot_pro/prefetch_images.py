"""
Pre-fetch all 1,581 ScreenSpot-Pro images sequentially into the HF cache.

Running 8 parallel benchmark scripts each fetching uncached images causes HF
to rate-limit us with 429s + 112s backoffs. This script fetches everything
once with a single connection so the parallel benchmark runs all hit the
local cache instantly.

Usage:
    uv run -m benchmarks.screenspot_pro.prefetch_images
"""

import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.screenspot_pro.screenspot_pro_multi import (  # noqa: E402
    ANNOTATION_FILES, DATASET_REPO,
)


def main():
    # Fetch all annotation files first (these were already cached during smoke).
    print(f"Fetching {len(ANNOTATION_FILES)} annotation files...")
    samples = []
    for fname in tqdm(ANNOTATION_FILES, desc="annotations"):
        p = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset",
                            filename=f"annotations/{fname}")
        samples.extend(json.load(open(p)))
    img_filenames = sorted({s["img_filename"] for s in samples})
    print(f"Total unique images: {len(img_filenames)}")

    failures = []
    for img in tqdm(img_filenames, desc="images"):
        for attempt in range(5):
            try:
                hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset",
                                filename=f"images/{img}")
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "Too Many Requests" in msg:
                    wait = 30 * (attempt + 1)
                    tqdm.write(f"[429] {img} attempt {attempt+1}, sleeping {wait}s")
                    time.sleep(wait)
                else:
                    tqdm.write(f"[error] {img}: {type(e).__name__}: {e}")
                    failures.append((img, str(e)))
                    break
        else:
            failures.append((img, "exceeded retries"))

    print(f"\nDone. {len(img_filenames) - len(failures)} cached, {len(failures)} failed.")
    if failures:
        for img, err in failures[:10]:
            print(f"  FAIL {img}: {err}")


if __name__ == "__main__":
    main()
