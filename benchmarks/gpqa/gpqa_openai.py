"""
GPQA Diamond benchmark for OpenAI GPT-5.x.

198 graduate-level multiple-choice questions in physics, chemistry, and biology
written and validated by domain experts. The "Diamond" subset is the hardest
slice of GPQA — questions where both expert validators answered correctly and
the majority of non-experts answered incorrectly.

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa (config: gpqa_diamond)
Paper:   https://arxiv.org/abs/2311.12022

Methodology:
  - For each question we deterministically shuffle the 4 answer choices using
    Record ID as the seed. This avoids both position bias (always-A) and
    pure-random eval-to-eval drift.
  - Single-shot, pass@1, no chain-of-thought prompted. The model is asked to
    output a single letter A/B/C/D.
  - Metric: exact-match accuracy on the predicted letter, reported overall and
    per high-level domain.

Default model: gpt-5.5. Run gpt-5.4-mini with `--model gpt-5.4-mini`.
Reasoning defaults to fully off (`reasoning_effort="none"`); temperature 0.0.

Usage:
    uv run -m benchmarks.gpqa.gpqa_openai
    uv run -m benchmarks.gpqa.gpqa_openai --model gpt-5.4-mini
    uv run -m benchmarks.gpqa.gpqa_openai --limit 5
    uv run -m benchmarks.gpqa.gpqa_openai --evaluate-only

Env: OPENAI_API_KEY must be set.
"""

import os
import re
import sys
import json
import time
import random
import asyncio
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_ID = "Idavidrein/gpqa"
CONFIG = "gpqa_diamond"
SPLIT = "train"  # GPQA Diamond ships as a single 'train' split with 198 rows.

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "none"
TEMPERATURE = 0.0
CONCURRENCY = 10
MAX_RETRIES = 5
RETRY_BACKOFF_CAP_S = 30.0

MODEL = DEFAULT_MODEL
REASONING_EFFORT = DEFAULT_REASONING_EFFORT

PROMPT_TEMPLATE = (
    "The following is a multiple choice question (with answers). Respond with "
    "only the single letter (A, B, C, or D) corresponding to the correct "
    "answer. Do not show your work.\n\n"
    "Question: {question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n\n"
    "Answer:"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to .env "
        "(get one from https://platform.openai.com/api-keys)."
    )

# Bypass the .env's interfaze base_url override.
openai_client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPENAI_API_KEY,
)


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def invoke_openai(messages: list[dict]):
    # GPT-5.x rejects temperature!=default when reasoning is engaged
    # ("Unsupported value: 'temperature' does not support 0.0 with this model.
    # Only the default (1) value is supported."). With reasoning off ('none'),
    # temperature=0 is accepted and gives us deterministic decoding.
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "reasoning_effort": REASONING_EFFORT,
    }
    if REASONING_EFFORT == "none":
        kwargs["temperature"] = TEMPERATURE
    return openai_client.chat.completions.create(**kwargs)


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
            if rec.get("response") is not None:
                done.add(str(rec["id"]))
    return done


def load_records(path: Path) -> list[dict]:
    by_id: dict[str, dict] = {}
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                by_id[str(rec["id"])] = rec
            except json.JSONDecodeError:
                continue
    return list(by_id.values())


_LETTER_RE = re.compile(r"\b([ABCD])\b")
_FIRST_LETTER_RE = re.compile(r"[ABCD]")


def parse_letter(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if len(s) == 1 and s.upper() in "ABCD":
        return s.upper()
    m = _LETTER_RE.search(s.upper())
    if m:
        return m.group(1)
    m = _FIRST_LETTER_RE.search(s.upper())
    if m:
        return m.group(0)
    return None


def _clean(s) -> str:
    return str(s).strip() if s is not None else ""


def build_sample(row: dict) -> dict:
    """Shuffle the 4 choices deterministically using Record ID as seed; track
    which letter ended up holding the correct answer."""
    record_id = str(row.get("Record ID") or row.get("Question", ""))
    correct = _clean(row["Correct Answer"])
    incorrect = [
        _clean(row["Incorrect Answer 1"]),
        _clean(row["Incorrect Answer 2"]),
        _clean(row["Incorrect Answer 3"]),
    ]
    choices = [(correct, True)] + [(x, False) for x in incorrect]
    rng = random.Random(record_id)
    rng.shuffle(choices)
    letters = ["A", "B", "C", "D"]
    correct_letter = letters[next(i for i, (_, is_c) in enumerate(choices) if is_c)]
    return {
        "id": record_id,
        "question": _clean(row["Question"]),
        "a": choices[0][0],
        "b": choices[1][0],
        "c": choices[2][0],
        "d": choices[3][0],
        "correct_letter": correct_letter,
        "correct_answer_text": correct,
        "domain": _clean(row.get("High-level domain")),
        "subdomain": _clean(row.get("Subdomain")),
    }


async def process_sample(sample: dict, semaphore: asyncio.Semaphore,
                         writer: JsonlWriter, progress: dict) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        question=sample["question"],
        a=sample["a"], b=sample["b"], c=sample["c"], d=sample["d"],
    )
    messages = [{"role": "user", "content": prompt}]
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            async with semaphore:
                start = time.perf_counter()
                response = await asyncio.to_thread(invoke_openai, messages)
                latency_ms = int((time.perf_counter() - start) * 1000)
            content = (response.choices[0].message.content or "").strip()
            request_id = getattr(response, "id", None)
            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            predicted = parse_letter(content)
            correct = predicted == sample["correct_letter"]

            record = {
                "id": sample["id"],
                "domain": sample["domain"],
                "subdomain": sample["subdomain"],
                "correct_letter": sample["correct_letter"],
                "prediction": predicted,
                "correct": correct,
                "response": content,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "attempts": attempt,
            }
            await writer.append(record)

            progress["done"] += 1
            if correct:
                progress["correct"] += 1
            mark = "OK" if correct else "X "
            tqdm.write(
                f"[{progress['done']}/{progress['total']}] {mark} "
                f"id={sample['id']} domain={sample['domain']:10} "
                f"gold={sample['correct_letter']} pred={predicted or '?'} "
                f"latency={latency_ms}ms attempt={attempt}"
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
                await asyncio.sleep(min(2 ** (attempt - 1), RETRY_BACKOFF_CAP_S))

    progress["failed"] += 1
    tqdm.write(f"[FAILED] id={sample['id']} after {MAX_RETRIES} attempts: {last_error}")
    return None


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    total = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    n_unparseable = sum(1 for r in results if r.get("prediction") is None)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_domain[r.get("domain") or "Unknown"].append(r)
    per_domain = {
        d: {
            "n": len(rows),
            "accuracy": sum(1 for r in rows if r.get("correct")) / len(rows),
        }
        for d, rows in by_domain.items()
    }

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
        "accuracy": n_correct / total,
        "correct": n_correct,
        "total": total,
        "unparseable": n_unparseable,
        "per_domain": per_domain,
        "latency": latency_stats,
    }


def print_summary(metrics: dict):
    print(f"\n{'=' * 60}")
    print(f"GPQA Diamond — {DATASET_ID}/{CONFIG} ({MODEL}, reasoning={REASONING_EFFORT})")
    print(f"{'=' * 60}")
    print(f"Accuracy   : {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"Unparseable: {metrics['unparseable']}")
    print("\nPer high-level domain:")
    for d in sorted(metrics["per_domain"]):
        v = metrics["per_domain"][d]
        print(f"  {d:12} n={v['n']:>3} acc={v['accuracy']:.4f}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(f"\nLatency    : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
              f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms max={lat['max_ms']}ms")


async def run_predictions(pred_path: Path, limit: int | None):
    print(f"Loading {DATASET_ID}/{CONFIG} (split={SPLIT})...")
    ds = load_dataset(DATASET_ID, CONFIG, split=SPLIT)
    print(f"Loaded {len(ds)} rows")
    samples = [build_sample(dict(row)) for row in ds]

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
    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress = {"total": len(pending), "done": 0, "correct": 0, "failed": 0}

    tasks = [process_sample(s, semaphore, writer, progress) for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc=f"GPQA Diamond / {MODEL}")
    except Exception:
        traceback.print_exc()
    acc = progress["correct"] / progress["done"] if progress["done"] else 0.0
    print(f"\nRun finished: {progress['done']}/{progress['total']} answered, "
          f"{progress['correct']} correct (acc={acc:.4f}), {progress['failed']} failed.")


def run_evaluation(pred_path: Path, metrics_path: Path):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)
    for r in results:
        if r.get("prediction") is None and r.get("response"):
            r["prediction"] = parse_letter(r["response"])
        if r.get("correct") is None and r.get("prediction") is not None:
            r["correct"] = r["prediction"] == r.get("correct_letter")

    metrics = compute_metrics(results)
    print_summary(metrics)
    output = {
        **metrics,
        "dataset": DATASET_ID,
        "config": CONFIG,
        "split": SPLIT,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "concurrency": CONCURRENCY,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    global MODEL, REASONING_EFFORT
    parser = argparse.ArgumentParser(description="GPQA Diamond benchmark for OpenAI GPT-5.x")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="OpenAI model id (e.g. gpt-5.5, gpt-5.4-mini)")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        help="reasoning_effort param ('none', 'minimal', 'low', 'medium', 'high', etc.)")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N unanswered samples")
    args = parser.parse_args()

    MODEL = args.model
    REASONING_EFFORT = args.reasoning_effort

    tag = f"{model_slug(MODEL)}_reasoning{REASONING_EFFORT}_gpqa_diamond"
    pred_path = RESULTS_DIR / f"{tag}_responses.jsonl"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"

    if args.evaluate_only:
        run_evaluation(pred_path, metrics_path)
    elif args.predict_only:
        asyncio.run(run_predictions(pred_path, limit=args.limit))
    else:
        asyncio.run(run_predictions(pred_path, limit=args.limit))
        run_evaluation(pred_path, metrics_path)


if __name__ == "__main__":
    main()
