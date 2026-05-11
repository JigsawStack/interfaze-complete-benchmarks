"""
Spider 2.0-Lite (SQLite-only) benchmark for OpenAI.

Mirrors spider2_lite_sqlite.py but uses gpt-5.4 via the OpenAI Chat Completions API.

Usage:
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite_openai
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite_openai --predict-only
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite_openai --evaluate-only
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite_openai --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

PRED_SQL_DIR = RESULTS_DIR / "spider2_lite_sqlite_openai_5_5_predictions"
PRED_META_PATH = RESULTS_DIR / "spider2_lite_sqlite_openai_5_5_predictions.json"
EXEC_CSV_DIR = RESULTS_DIR / "spider2_lite_sqlite_openai_5_5_exec"
EVAL_OUTPUT = RESULTS_DIR / "spider2_lite_sqlite_openai_5_5_scored.json"
METRICS_OUTPUT = RESULTS_DIR / "spider2_lite_sqlite_openai_5_5_metrics.json"

sys.path.insert(0, str(PROJECT_ROOT))
from src.commons_openai import openai_client  # noqa: E402
from benchmarks.spider2_lite_sqlite.spider2_lite_sqlite import (  # noqa: E402
    MAX_RETRIES,
    RateLimiter,
    build_prompt,
    run_evaluation,
    run_predictions_for,
)
from benchmarks.spider2_lite_sqlite.eval_helpers import extract_sql_query  # noqa: E402

OPENAI_MODEL = "gpt-5.5"
TEMPERATURE = 0
REASONING_EFFORT = "none"


async def predict_one(item: dict, rate_limiter: RateLimiter) -> dict:
    # OpenAI gpt-5.x: reasoning_effort='minimal' to suppress thinking, temperature=0.
    prompt, missing = build_prompt(item)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            resp = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                reasoning_effort=REASONING_EFFORT,
            )
            content = resp.choices[0].message.content or ""
            return {
                "instance_id": item["instance_id"],
                "db": item["db"],
                "question": item["question"],
                "predict_raw": content,
                "predict_sql": extract_sql_query(content),
                "missing_context": missing,
                "error": None,
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {
                    "instance_id": item["instance_id"],
                    "db": item["db"],
                    "question": item["question"],
                    "predict_raw": "",
                    "predict_sql": "",
                    "missing_context": missing,
                    "error": str(e),
                }
            await asyncio.sleep(2**attempt)
    return {}


async def run_predictions(limit: int | None = None) -> None:
    await run_predictions_for(predict_one, PRED_META_PATH, PRED_SQL_DIR, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spider 2.0-Lite SQLite-only benchmark for OpenAI"
    )
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    eval_kwargs = dict(
        pred_meta_path=PRED_META_PATH,
        exec_csv_dir=EXEC_CSV_DIR,
        eval_output=EVAL_OUTPUT,
        metrics_output=METRICS_OUTPUT,
        model_name=OPENAI_MODEL,
    )

    if args.evaluate_only:
        run_evaluation(**eval_kwargs)
    elif args.predict_only:
        asyncio.run(run_predictions(limit=args.limit))
    else:
        asyncio.run(run_predictions(limit=args.limit))
        run_evaluation(**eval_kwargs)


if __name__ == "__main__":
    main()
