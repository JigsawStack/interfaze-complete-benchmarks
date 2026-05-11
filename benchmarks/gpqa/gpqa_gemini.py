"""
GPQA Diamond benchmark for Gemini.

198 graduate-level multiple-choice questions in physics, chemistry, and biology.
Mirrors `gpqa_openai.py` exactly — same prompt, same deterministic shuffle of
choices, same scoring — only the inference call differs.

Default model: gemini-3.1-pro-preview. Thinking is left ON at the model's
default level (the Pro family is reasoning-first; we don't pass a
thinking_config). Temperature pinned to 0.0.

Usage:
    uv run -m benchmarks.gpqa.gpqa_gemini
    uv run -m benchmarks.gpqa.gpqa_gemini --model gemini-3.1-pro-preview
    uv run -m benchmarks.gpqa.gpqa_gemini --thinking-level minimal
    uv run -m benchmarks.gpqa.gpqa_gemini --limit 5
    uv run -m benchmarks.gpqa.gpqa_gemini --evaluate-only

Env: GEMINI_KEY must be set in .env.
"""

import os
import re
import sys
import json
import time
import asyncio
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse parsing/sample-builder/scoring from the OpenAI variant — guarantees
# identical prompts and answer-letter parsing across providers.
from benchmarks.gpqa.gpqa_openai import (  # noqa: E402
    DATASET_ID,
    CONFIG,
    SPLIT,
    PROMPT_TEMPLATE,
    JsonlWriter,
    build_sample,
    compute_metrics,
    load_completed_ids,
    load_records,
    parse_letter,
)

load_dotenv()

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_THINKING_LEVEL: str | None = None  # None => model default (Pro: thinking on)
TEMPERATURE = 0.0
CONCURRENCY = 30  # Flash has much higher headroom than Pro.
MAX_RETRIES = 6
RETRY_BACKOFF_CAP_S = 30.0

MODEL = DEFAULT_MODEL
THINKING_LEVEL: str | None = DEFAULT_THINKING_LEVEL

GEMINI_KEY = (
    os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
)
if not GEMINI_KEY:
    raise RuntimeError(
        "GEMINI_KEY is not set. Add it to .env "
        "(get one from https://aistudio.google.com/app/apikey)."
    )

gemini_client = genai.Client(api_key=GEMINI_KEY)


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def invoke_gemini(prompt: str):
    config_kwargs = {"temperature": TEMPERATURE}
    if THINKING_LEVEL is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    config = types.GenerateContentConfig(**config_kwargs)
    return gemini_client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_text(text=prompt)],
        config=config,
    )


async def process_sample(sample: dict, semaphore: asyncio.Semaphore,
                         writer: JsonlWriter, progress: dict) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        question=sample["question"],
        a=sample["a"], b=sample["b"], c=sample["c"], d=sample["d"],
    )
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            async with semaphore:
                start = time.perf_counter()
                response = await asyncio.to_thread(invoke_gemini, prompt)
                latency_ms = int((time.perf_counter() - start) * 1000)
            content = (response.text or "").strip()
            request_id = getattr(response, "response_id", None)
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


def print_summary(metrics: dict):
    print(f"\n{'=' * 60}")
    print(
        f"GPQA Diamond — {DATASET_ID}/{CONFIG} ({MODEL}, "
        f"thinking={'default' if THINKING_LEVEL is None else THINKING_LEVEL}, "
        f"temp={TEMPERATURE})"
    )
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
        "thinking_level": THINKING_LEVEL,
        "temperature": TEMPERATURE,
        "concurrency": CONCURRENCY,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    global MODEL, THINKING_LEVEL
    parser = argparse.ArgumentParser(description="GPQA Diamond benchmark for Gemini")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Gemini model id (e.g. gemini-3.1-pro-preview)")
    parser.add_argument("--thinking-level", default=DEFAULT_THINKING_LEVEL,
                        help="thinking_level ('minimal'|'low'|'high'); omit for model default")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N unanswered samples")
    args = parser.parse_args()

    MODEL = args.model
    THINKING_LEVEL = args.thinking_level

    thinking_slug = THINKING_LEVEL or "default"
    tag = f"{model_slug(MODEL)}_thinking{thinking_slug}_gpqa_diamond"
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
