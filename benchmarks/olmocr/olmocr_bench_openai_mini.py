"""
OlmOCR Benchmark for OpenAI gpt-5.4-mini.

Mirrors olmocr_bench.py (interfaze) but uses gpt-5.4-mini via OpenAI.

Usage:
    uv run -m benchmarks.olmocr.olmocr_bench_openai_mini
    uv run -m benchmarks.olmocr.olmocr_bench_openai_mini --sample
    uv run -m benchmarks.olmocr.olmocr_bench_openai_mini --skip-generation
    uv run -m benchmarks.olmocr.olmocr_bench_openai_mini --generate-only
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
CANDIDATE_NAME = "openai_mini"
RATE_LIMIT = 25
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
    from olmocr.bench.runners.run_openai_mini import run_openai_mini

    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            result = await asyncio.to_thread(run_openai_mini, pdf_path, page_num)
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


async def generate_outputs(data_dir: Path):
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
    args = parser.parse_args()

    if args.sample:
        data_dir = SAMPLE_DATA_DIR
        print("=== Using sample data ===")
    else:
        print("=== Downloading full olmOCR-bench dataset from HuggingFace ===")
        data_dir = download_full_dataset()

    if not args.skip_generation:
        print(f"\n=== Generating {CANDIDATE_NAME} outputs ===")
        await generate_outputs(data_dir)

    if not args.generate_only:
        print("\n=== Running OlmOCR Benchmark Evaluation ===")
        run_evaluation(data_dir)


if __name__ == "__main__":
    asyncio.run(main())
