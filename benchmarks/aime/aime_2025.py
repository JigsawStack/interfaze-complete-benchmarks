"""
AIME 2025 benchmark for Interfaze.

30 problems from the 2025 American Invitational Mathematics Examination
(AIME I + II, 15 each). Answers are non-negative integers.

Dataset: https://huggingface.co/datasets/MathArena/aime_2025

Usage:
    uv run -m benchmarks.aime.aime_2025
    uv run -m benchmarks.aime.aime_2025 --predict-only
    uv run -m benchmarks.aime.aime_2025 --evaluate-only
    uv run -m benchmarks.aime.aime_2025 --limit 5

Checkpointing:
    Each successful sample is appended to results/aime_2025_responses.jsonl
    as it finishes, so reruns only query problems still missing.
    Failed-after-retries samples are NOT written, so they retry next run.
"""

import sys
import json
import os
import re
import time
import asyncio
import argparse
import traceback
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.commons import invoke_interfaze  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_ID = "MathArena/aime_2025"
REASONING_EFFORT = "high"
RATE_LIMIT = 10
MAX_RETRIES = 3

QUESTION_TEMPLATE = """Solve the following problem. The last line of your response should be of the following format: 'Answer: $ANSWER' (without quotes) where $ANSWER is the non-negative integer answer to the problem. Think step by step before answering.

{problem}"""

# Primary regex matches "Answer: 123" (with optional LaTeX $ wrapping).
# Fallbacks recognize \boxed{...} and "final answer is N" forms that reasoning
# models commonly emit — scanned only in the last 600 chars to avoid picking
# up random integers from intermediate reasoning.
ANSWER_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\s*(-?\d+)\s*\$?")
FALLBACK_PATTERNS = [
    re.compile(r"\\boxed\{\s*(-?\d+)\s*\}"),
    re.compile(r"\\boxed\{\s*\\text\{\s*(-?\d+)\s*\}\s*\}"),
    re.compile(r"(?i)\bfinal\s+answer\b[^\d\n-]{0,30}(-?\d+)"),
]
FALLBACK_TAIL_CHARS = 600


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


def build_sample(row: dict) -> dict:
    problem = str(row.get("problem", "")).strip()
    answer = row.get("answer")
    problem_types = row.get("problem_type") or []
    if isinstance(problem_types, str):
        problem_types = [problem_types]
    idx = row.get("problem_idx")
    sample_id = f"aime_2025_{idx}" if idx is not None else f"aime_2025_{hash(problem)}"
    prompt = QUESTION_TEMPLATE.format(problem=problem)
    return {
        "id": sample_id,
        "problem_idx": idx,
        "problem_type": problem_types,
        "problem": problem,
        "correct_answer": int(answer),
        "prompt": prompt,
    }


def extract_answer(text: str) -> int | None:
    """Primary: simple-evals style `Answer: N`. Fallbacks: `\\boxed{N}` and
    'final answer ... N' near end of response."""
    if not text:
        return None
    match = ANSWER_PATTERN.search(text)
    if match:
        return int(match.group(1))
    tail = text[-FALLBACK_TAIL_CHARS:]
    for pattern in FALLBACK_PATTERNS:
        matches = list(pattern.finditer(tail))
        if matches:
            return int(matches[-1].group(1))
    return None


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


async def process_sample(sample: dict, rate_limiter: RateLimiter,
                         writer: JsonlWriter, progress: dict) -> dict | None:
    messages = [{"role": "user", "content": sample["prompt"]}]
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                invoke_interfaze,
                messages,
                reasoning_effort=REASONING_EFFORT,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)

            content = (response.choices[0].message.content or "").strip()
            request_id = getattr(response, "id", None)

            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            predicted = extract_answer(content)
            record = {
                "id": sample["id"],
                "problem_idx": sample["problem_idx"],
                "problem_type": sample["problem_type"],
                "problem": sample["problem"],
                "correct_answer": sample["correct_answer"],
                "predicted_answer": predicted,
                "response": content,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "attempts": attempt,
            }
            await writer.append(record)

            progress["done"] += 1
            is_correct = predicted == sample["correct_answer"]
            if is_correct:
                progress["correct"] += 1
                mark = "OK"
            else:
                mark = "X "
            tqdm.write(
                f"[{progress['done']}/{progress['total']}] {mark} "
                f"id={sample['id']} pred={predicted} gt={sample['correct_answer']} "
                f"latency={latency_ms}ms req_id={request_id} attempt={attempt}"
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
    tqdm.write(
        f"[FAILED] id={sample['id']} after {MAX_RETRIES} attempts: {last_error}"
    )
    return None


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.get("predicted_answer") == r["correct_answer"])
    unparsed = sum(1 for r in results if r.get("predicted_answer") is None)

    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int)]
    latency_stats = {}
    if latencies:
        lats = sorted(latencies)
        n = len(lats)
        latency_stats = {
            "count": n,
            "mean_ms": sum(lats) / n,
            "p50_ms": lats[n // 2],
            "p90_ms": lats[min(n - 1, int(n * 0.9))],
            "p99_ms": lats[min(n - 1, int(n * 0.99))],
            "max_ms": lats[-1],
        }

    # Per-category breakdown. Problems may have multiple types; count once per type.
    by_type: dict[str, list[int]] = {}
    for r in results:
        types = r.get("problem_type") or ["Unknown"]
        for t in types:
            by_type.setdefault(t, []).append(
                1 if r.get("predicted_answer") == r["correct_answer"] else 0
            )
    per_type = {
        t: {"accuracy": sum(v) / len(v), "count": len(v)}
        for t, v in by_type.items()
    }

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "unparsed": unparsed,
        "by_problem_type": per_type,
        "latency": latency_stats,
    }


def print_summary(metrics: dict):
    print(f"\n{'=' * 60}")
    print(f"AIME 2025 Results (Interfaze — reasoning={REASONING_EFFORT})")
    print(f"{'=' * 60}")
    print(f"Accuracy : {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"Unparsed : {metrics['unparsed']}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(
            f"Latency  : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
            f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms max={lat['max_ms']}ms"
        )
    print(f"\n{'Problem type':<25} {'Accuracy':>10} {'Count':>8}")
    print("-" * 45)
    for t in sorted(metrics["by_problem_type"]):
        m = metrics["by_problem_type"][t]
        print(f"{t:<25} {m['accuracy']:>10.4f} {m['count']:>8}")


async def run_predictions(pred_path: Path, limit: int | None = None):
    print(f"Loading AIME 2025 ({DATASET_ID}) from HuggingFace...")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Loaded {len(dataset)} problems")

    samples = [build_sample(dict(row)) for row in dataset]
    samples.sort(key=lambda s: (s["problem_idx"] is None, s["problem_idx"] or 0))

    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
        print(f"--limit applied: will run at most {limit} sample(s)")
    print(
        f"Resume: {len(done_ids)} already completed, {len(pending)} remaining "
        f"(checkpoint: {pred_path})"
    )
    if not pending:
        return

    writer = JsonlWriter(pred_path)
    rate_limiter = RateLimiter(RATE_LIMIT)
    progress = {"total": len(pending), "done": 0, "correct": 0, "failed": 0}

    tasks = [process_sample(s, rate_limiter, writer, progress) for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc="AIME 2025")
    except Exception:
        traceback.print_exc()

    print(
        f"\nRun finished: {progress['done']}/{progress['total']} answered, "
        f"{progress['correct']} correct, {progress['failed']} failed "
        f"(failed samples will retry on next run)."
    )


def run_evaluation(pred_path: Path, metrics_path: Path):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)

    for r in results:
        if r.get("response") and r.get("predicted_answer") is None:
            r["predicted_answer"] = extract_answer(r["response"])

    metrics = compute_metrics(results)
    print_summary(metrics)

    output = {
        **metrics,
        "dataset": DATASET_ID,
        "reasoning_effort": REASONING_EFFORT,
        "model": "interfaze-beta",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="AIME 2025 benchmark for Interfaze")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N unanswered samples")
    args = parser.parse_args()

    pred_path = RESULTS_DIR / "aime_2025_responses.jsonl"
    metrics_path = RESULTS_DIR / "aime_2025_metrics.json"

    if args.evaluate_only:
        run_evaluation(pred_path, metrics_path)
    elif args.predict_only:
        asyncio.run(run_predictions(pred_path, limit=args.limit))
    else:
        asyncio.run(run_predictions(pred_path, limit=args.limit))
        run_evaluation(pred_path, metrics_path)


if __name__ == "__main__":
    main()
