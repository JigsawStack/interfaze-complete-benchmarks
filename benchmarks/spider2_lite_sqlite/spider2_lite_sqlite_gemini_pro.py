"""
Spider 2.0-Lite (SQLite-only) benchmark for Gemini 2.5 Pro.

Mirrors spider2_lite_sqlite.py but uses gemini-2.5-pro via google-genai.

Usage:
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite_gemini_pro
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite_gemini_pro --predict-only
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite_gemini_pro --evaluate-only
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite_gemini_pro --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

PRED_SQL_DIR = RESULTS_DIR / "spider2_lite_sqlite_gemini_pro_predictions"
PRED_META_PATH = RESULTS_DIR / "spider2_lite_sqlite_gemini_pro_predictions.json"
EXEC_CSV_DIR = RESULTS_DIR / "spider2_lite_sqlite_gemini_pro_exec"
EVAL_OUTPUT = RESULTS_DIR / "spider2_lite_sqlite_gemini_pro_scored.json"
METRICS_OUTPUT = RESULTS_DIR / "spider2_lite_sqlite_gemini_pro_metrics.json"

sys.path.insert(0, str(PROJECT_ROOT))
from src.commons_gemini import gemini_client  # noqa: E402
from google.genai import types as genai_types  # noqa: E402
from benchmarks.spider2_lite_sqlite.spider2_lite_sqlite import (  # noqa: E402
    MAX_RETRIES,
    RateLimiter,
    build_prompt,
    run_evaluation,
    run_predictions_for,
)
from benchmarks.spider2_lite_sqlite.eval_helpers import extract_sql_query  # noqa: E402

GEMINI_MODEL = "gemini-2.5-pro"
TEMPERATURE = 0.0
# gemini-2.5-pro rejects thinking_budget=0 ("This model only works in thinking
# mode"). Thinking stays on by API necessity; only temperature is constrained.
DISABLE_THINKING = False


async def predict_one(item: dict, rate_limiter: RateLimiter) -> dict:
    prompt, missing = build_prompt(item)
    config_kwargs = {"temperature": TEMPERATURE}
    if DISABLE_THINKING:
        config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
    config = genai_types.GenerateContentConfig(**config_kwargs)
    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            resp = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            content = (resp.text or "") if hasattr(resp, "text") else ""
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
        description="Spider 2.0-Lite SQLite-only benchmark for Gemini 2.5 Pro"
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
        model_name=GEMINI_MODEL,
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
