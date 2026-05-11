"""
VoxPopuli-Cleaned-AA ASR benchmark for Interfaze.

628 English speech samples drawn from European Parliament recordings, with
re-cleaned ground-truth transcripts curated by Artificial Analysis. This is
one of the three datasets that make up AA-WER v2.0 (the ASR benchmark Gemini 3
Pro and other frontier audio models report on). The other two are
Earnings22-Cleaned-AA (open, 6 samples) and AA-AgentTalk (proprietary).

Dataset: https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA

Metric: WER (Word Error Rate) with Whisper-style text normalization —
        lowercase, strip punctuation, NFKC, collapse whitespace.

Usage:
    uv run -m benchmarks.asr.voxpopuli_aa
    uv run -m benchmarks.asr.voxpopuli_aa --limit 5
    uv run -m benchmarks.asr.voxpopuli_aa --evaluate-only
"""

import sys
import os
import re
import json
import time
import base64
import asyncio
import argparse
import traceback
import unicodedata
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from jiwer import wer, cer
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.commons import invoke_interfaze  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_ID = "ArtificialAnalysis/VoxPopuli-Cleaned-AA"
SPLIT = "test"
REASONING_EFFORT = None   # off
TEMPERATURE = 0.0         # deterministic
RATE_LIMIT = 25
MAX_RETRIES = 3

PROMPT = (
    "Transcribe the following audio. Fix anything that needs fixing — "
    "disfluencies, stutters, obvious misspeaks, garbled words, or misheard "
    "named entities — so the transcription reads as the speaker clearly "
    "intended. Output ONLY the cleaned transcription, no commentary, labels, "
    "speaker tags, or timestamps."
)


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


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, record: dict):
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                tqdm.write(f"[resume] skipping malformed line {line_no} in {path}")
                continue
            if rec.get("response"):
                done.add(str(rec["id"]))
    return done


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    by_id: dict[str, dict] = {}
    for r in records:
        by_id[str(r["id"])] = r
    return list(by_id.values())


_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9' ]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Whisper-style light normalization for WER: NFKC, lowercase, strip
    punctuation (keeping apostrophes for contractions), collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _NON_ALNUM_SPACE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def fetch_audio_bytes(file_name: str) -> bytes:
    """Pull a single audio/*.wav from the HF dataset repo (cached locally)."""
    path = hf_hub_download(
        repo_id=DATASET_ID, repo_type="dataset",
        filename=f"audio/{file_name}",
    )
    with open(path, "rb") as f:
        return f.read()


def build_sample(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "file_name": row["file_name"],
        "transcript": row["transcript"],
        "duration": row.get("duration"),
        "gender": row.get("gender"),
        "language": row.get("language"),
    }


async def process_sample(sample: dict, rate_limiter, writer: JsonlWriter,
                         progress: dict) -> dict | None:
    last_error: str | None = None

    # Fetch audio once (outside the retry loop — it's cached by hf_hub_download).
    try:
        audio_bytes = await asyncio.to_thread(fetch_audio_bytes, sample["file_name"])
    except Exception as e:
        tqdm.write(f"[fetch error] id={sample['id']}: {type(e).__name__}: {e}")
        progress["failed"] += 1
        return None

    b64 = base64.b64encode(audio_bytes).decode("ascii")
    data_url = f"data:audio/wav;base64,{b64}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "file", "file": {"filename": sample["file_name"], "file_data": data_url}},
        ],
    }]

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                invoke_interfaze,
                messages,
                reasoning_effort=REASONING_EFFORT,
                temperature=TEMPERATURE,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            content = (response.choices[0].message.content or "").strip()
            request_id = getattr(response, "id", None)
            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            pred_norm = normalize_text(content)
            gt_norm = normalize_text(sample["transcript"])
            try:
                sample_wer = float(wer(gt_norm, pred_norm)) if gt_norm else float("inf")
            except Exception:
                sample_wer = float("inf")
            try:
                sample_cer = float(cer(gt_norm, pred_norm)) if gt_norm else float("inf")
            except Exception:
                sample_cer = float("inf")

            record = {
                "id": sample["id"],
                "file_name": sample["file_name"],
                "duration": sample["duration"],
                "gender": sample["gender"],
                "language": sample["language"],
                "transcript": sample["transcript"],
                "transcript_normalized": gt_norm,
                "prediction": content,
                "prediction_normalized": pred_norm,
                "wer": sample_wer,
                "cer": sample_cer,
                "response": content,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "attempts": attempt,
            }
            await writer.append(record)

            progress["done"] += 1
            progress["sum_wer"] += sample_wer if sample_wer != float("inf") else 0
            tqdm.write(
                f"[{progress['done']}/{progress['total']}] WER={sample_wer:.3f} "
                f"CER={sample_cer:.3f} id={sample['id']} dur={sample['duration']}s "
                f"latency={latency_ms}ms req_id={request_id}"
            )
            return record

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            last_error = f"{type(e).__name__}: {e}"
            tqdm.write(
                f"[error] id={sample['id']} attempt={attempt}/{MAX_RETRIES} "
                f"latency={latency_ms}ms error={last_error}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))

    progress["failed"] += 1
    tqdm.write(f"[FAILED] id={sample['id']} after {MAX_RETRIES} attempts: {last_error}")
    return None


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    # Aggregate WER/CER the correct way: per-ref length weighting — not a simple
    # mean of per-sample WERs (long clips shouldn't count the same as short ones).
    refs = [r["transcript_normalized"] for r in results if r.get("transcript_normalized")]
    hyps = [r["prediction_normalized"] for r in results if r.get("transcript_normalized")]
    corpus_wer = float(wer(refs, hyps)) if refs else float("inf")
    corpus_cer = float(cer(refs, hyps)) if refs else float("inf")

    # Simple per-sample mean for reference.
    mean_wer = sum(r["wer"] for r in results if r["wer"] != float("inf")) / max(1, len(results))
    mean_cer = sum(r["cer"] for r in results if r["cer"] != float("inf")) / max(1, len(results))

    # Time-weighted WER (the AA-WER convention within a dataset)
    total_dur = sum(r.get("duration", 0) or 0 for r in results)
    time_weighted_wer = (
        sum((r["wer"] if r["wer"] != float("inf") else 0) * (r.get("duration", 0) or 0)
            for r in results) / total_dur if total_dur > 0 else float("inf")
    )

    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int)]
    latency_stats = {}
    if latencies:
        lats = sorted(latencies)
        n = len(lats)
        latency_stats = {
            "count": n, "mean_ms": sum(lats) / n,
            "p50_ms": lats[n // 2],
            "p90_ms": lats[min(n - 1, int(n * 0.9))],
            "p99_ms": lats[min(n - 1, int(n * 0.99))],
            "max_ms": lats[-1],
        }

    return {
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "mean_sample_wer": mean_wer,
        "mean_sample_cer": mean_cer,
        "time_weighted_wer": time_weighted_wer,
        "num_samples": len(results),
        "total_duration_s": total_dur,
        "latency": latency_stats,
    }


def print_summary(metrics: dict):
    print(f"\n{'=' * 60}")
    print(f"VoxPopuli-Cleaned-AA Results (Interfaze, reasoning={REASONING_EFFORT}, temp={TEMPERATURE})")
    print(f"{'=' * 60}")
    print(f"Samples               : {metrics['num_samples']}")
    print(f"Total audio duration  : {metrics['total_duration_s']:.1f}s")
    print(f"Corpus WER            : {metrics['corpus_wer']:.4f}  ← primary metric")
    print(f"Corpus CER            : {metrics['corpus_cer']:.4f}")
    print(f"Time-weighted WER     : {metrics['time_weighted_wer']:.4f}  ← AA-WER convention")
    print(f"Mean per-sample WER   : {metrics['mean_sample_wer']:.4f}")
    print(f"Mean per-sample CER   : {metrics['mean_sample_cer']:.4f}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(f"Latency               : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
              f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms max={lat['max_ms']}ms")


async def run_predictions(pred_path: Path, limit: int | None = None):
    print(f"Loading {DATASET_ID}, split={SPLIT}...")
    dataset = load_dataset(DATASET_ID, split=SPLIT)
    print(f"Loaded {len(dataset)} samples")

    samples = [build_sample(dict(row)) for row in dataset]
    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
        print(f"--limit applied: will run at most {limit} sample(s)")
    print(f"Resume: {len(done_ids)} already completed, {len(pending)} remaining "
          f"(checkpoint: {pred_path})")
    if not pending:
        return

    writer = JsonlWriter(pred_path)
    rate_limiter = RateLimiter(RATE_LIMIT)
    progress = {"total": len(pending), "done": 0, "failed": 0, "sum_wer": 0.0}

    tasks = [process_sample(s, rate_limiter, writer, progress) for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc=f"{DATASET_ID.split('/')[-1]}/{SPLIT}")
    except Exception:
        traceback.print_exc()
    print(f"\nRun finished: {progress['done']}/{progress['total']} answered, "
          f"{progress['failed']} failed.")


def run_evaluation(pred_path: Path, metrics_path: Path):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)

    for r in results:
        if r.get("prediction_normalized") is None and r.get("response"):
            r["prediction_normalized"] = normalize_text(r["response"])
        if r.get("transcript_normalized") is None and r.get("transcript"):
            r["transcript_normalized"] = normalize_text(r["transcript"])

    metrics = compute_metrics(results)
    print_summary(metrics)
    output = {
        **metrics,
        "dataset": DATASET_ID, "split": SPLIT,
        "reasoning_effort": REASONING_EFFORT, "temperature": TEMPERATURE,
        "rate_limit": RATE_LIMIT, "model": "interfaze-beta",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="VoxPopuli-Cleaned-AA ASR benchmark")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pred_path = RESULTS_DIR / "voxpopuli_aa_responses.jsonl"
    metrics_path = RESULTS_DIR / "voxpopuli_aa_metrics.json"

    if args.evaluate_only:
        run_evaluation(pred_path, metrics_path)
    elif args.predict_only:
        asyncio.run(run_predictions(pred_path, limit=args.limit))
    else:
        asyncio.run(run_predictions(pred_path, limit=args.limit))
        run_evaluation(pred_path, metrics_path)


if __name__ == "__main__":
    main()
