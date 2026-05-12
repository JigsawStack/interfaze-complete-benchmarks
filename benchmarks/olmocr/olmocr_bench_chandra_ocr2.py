"""
OlmOCR Benchmark for Chandra OCR 2 (datalab-to/chandra-ocr-2) on Modal.

Mirrors olmocr_bench_openai_mini.py but dispatches each rendered PDF page to a
Modal-hosted Chandra OCR 2 endpoint (see ~/jigsawstack-ocr/chandra2.py).

Required env:
    CHANDRA_MODAL_URL          Base URL of the deployed Modal app, e.g.
                               https://<workspace>--mlt-chandra-ocr-chandraocr-api.modal.run
    CHANDRA_MODAL_ADMIN_KEY    Value of the admin-key secret used by the Modal app.

Usage:
    uv run -m benchmarks.olmocr.olmocr_bench_chandra_ocr2
    uv run -m benchmarks.olmocr.olmocr_bench_chandra_ocr2 --sample
    uv run -m benchmarks.olmocr.olmocr_bench_chandra_ocr2 --skip-generation
    uv run -m benchmarks.olmocr.olmocr_bench_chandra_ocr2 --generate-only
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "bench" / "sample_data"
FULL_DATA_DIR = Path(__file__).resolve().parent / "bench" / "full_data"
CANDIDATE_NAME = "chandra_ocr2"
RATE_LIMIT = 50  # requests admitted per second (token-bucket pacing)
MAX_RETRIES = 3

HF_REPO = "allenai/olmOCR-bench"
SPLITS = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RateLimiter:
    def __init__(self, rate: int):
        self.rate = rate
        self.tokens = rate
        self.last_refill = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                elapsed = now - self.last_refill
                self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
                self.last_refill = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            await asyncio.sleep(1 / self.rate)


def download_full_dataset():
    data_dir = FULL_DATA_DIR
    pdf_dir = data_dir / "pdfs"
    all_pdfs = set()
    for split in SPLITS:
        jsonl_dest = data_dir / f"{split}.jsonl"
        if jsonl_dest.exists():
            with open(jsonl_dest) as f:
                tests = [json.loads(l) for l in f if l.strip()]
        else:
            print(f"  Downloading {split}.jsonl...")
            src = hf_hub_download(HF_REPO, f"bench_data/{split}.jsonl", repo_type="dataset")
            with open(src) as f:
                tests = [json.loads(l) for l in f if l.strip()]
            data_dir.mkdir(parents=True, exist_ok=True)
            with open(jsonl_dest, "w") as f:
                for t in tests:
                    f.write(json.dumps(t) + "\n")
        print(f"    {split}: {len(tests)} tests")
        for t in tests:
            all_pdfs.add(t["pdf"])

    print(f"\n  Total unique PDFs to download: {len(all_pdfs)}")
    downloaded = 0
    skipped = 0
    for pdf_rel in sorted(all_pdfs):
        local_path = pdf_dir / pdf_rel
        if local_path.exists():
            skipped += 1
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            src = hf_hub_download(HF_REPO, f"bench_data/pdfs/{pdf_rel}", repo_type="dataset")
            os.symlink(src, str(local_path))
            downloaded += 1
        except Exception as e:
            print(f"    Failed to download {pdf_rel}: {e}")
    print(f"  PDFs: {downloaded} downloaded, {skipped} already existed")
    return data_dir


async def process_page(pdf_path, page_num, output_path, rate_limiter):
    from olmocr.bench.runners.run_chandra_ocr2 import run_chandra_ocr2

    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            result = await asyncio.to_thread(run_chandra_ocr2, pdf_path, page_num)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
            else:
                print(f"Failed after {MAX_RETRIES} attempts: {pdf_path} page {page_num}: {e}")
                return False


async def generate_outputs(data_dir: Path, limit: int | None = None):
    pdf_folder = data_dir / "pdfs"
    output_folder = data_dir / CANDIDATE_NAME

    pdf_pages = set()
    for jsonl_file in data_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                pdf_pages.add((t["pdf"], t["page"]))

    if limit is not None and limit > 0:
        pdf_pages = set(sorted(pdf_pages)[:limit])
        print(f"--limit {limit}: capping to first {len(pdf_pages)} (pdf, page) pairs")

    print(f"Found {len(pdf_pages)} unique (pdf, page) pairs to process")

    rate_limiter = RateLimiter(RATE_LIMIT)
    tasks = []
    for pdf_rel, page in sorted(pdf_pages):
        pdf_path = str(pdf_folder / pdf_rel)
        if not os.path.exists(pdf_path):
            continue
        base_name = os.path.splitext(os.path.basename(pdf_rel))[0]
        parent_dir = os.path.dirname(pdf_rel)
        md_filename = f"{base_name}_pg{page}_repeat1.md"
        if parent_dir:
            out_path = str(output_folder / parent_dir / md_filename)
        else:
            out_path = str(output_folder / md_filename)
        if os.path.exists(out_path):
            continue
        tasks.append(process_page(pdf_path, page, out_path, rate_limiter))

    if not tasks:
        print("All outputs already exist, skipping generation.")
        return True
    print(f"Processing {len(tasks)} pages...")
    results = await tqdm_asyncio.gather(*tasks, desc=f"Generating {CANDIDATE_NAME} outputs")
    num_success = sum(1 for r in results if r)
    num_failed = len(results) - num_success
    print(f"Done: {num_success} succeeded, {num_failed} failed")
    return num_failed == 0


def run_evaluation(data_dir: Path):
    from olmocr.bench.benchmark import main as bench_main
    sys.argv = ["benchmark", "--dir", str(data_dir), "--candidate", CANDIDATE_NAME, "--force"]
    bench_main()


async def main():
    parser = argparse.ArgumentParser(description=f"Run OlmOCR benchmark with {CANDIDATE_NAME}")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test mode: only generate outputs for the first N (pdf, page) pairs.",
    )
    args = parser.parse_args()

    if args.sample:
        data_dir = SAMPLE_DATA_DIR
        print("=== Using sample data ===")
    else:
        print("=== Downloading full olmOCR-bench dataset from HuggingFace ===")
        data_dir = download_full_dataset()

    if not args.skip_generation:
        print(f"\n=== Generating {CANDIDATE_NAME} outputs ===")
        await generate_outputs(data_dir, limit=args.limit)

    if not args.generate_only:
        print("\n=== Running OlmOCR Benchmark Evaluation ===")
        run_evaluation(data_dir)


if __name__ == "__main__":
    asyncio.run(main())
