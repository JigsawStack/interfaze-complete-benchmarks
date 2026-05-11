"""
RefCOCO benchmark for xAI Grok 4.3 via OpenRouter.

Strict head-to-head with `refcoco.py` (interfaze): same prompt, same parser,
same IoU, same checkpointing — only the inference call changes. Mirrors
`refcoco_kimi.py` but routed at xAI's Grok 4.3.

Reasoning: Grok 4.3's OpenRouter endpoint rejects `reasoning.enabled=false`
("Reasoning is mandatory for this endpoint"), so the closest analog to the
other refcoco runs' "reasoning off" is the lowest accepted effort tier,
`reasoning.effort="minimal"`. The reasoning_effort knob is exposed via
--reasoning-effort. Temperature pinned to 0.0.

Usage:
    uv run -m benchmarks.obj_detection.refcoco_grok --split testA
    uv run -m benchmarks.obj_detection.refcoco_grok --split testA --limit 20
    uv run -m benchmarks.obj_detection.refcoco_grok --split testA --evaluate-only

Env: OPENROUTER_API_KEY must be set (loaded from .env).
"""

import sys
import json
import os
import time
import asyncio
import argparse
import traceback
import concurrent.futures
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.obj_detection.refcoco import (  # noqa: E402
    JsonlWriter,
    PROMPT_TEMPLATE,
    build_samples,
    coco_bbox_to_xyxy,
    compute_iou,
    compute_metrics,
    load_completed_ids,
    load_records,
    parse_box,
    pil_to_data_url,
)

load_dotenv()

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATASET = "lmms-lab/RefCOCO"
DEFAULT_SPLIT = "val"
MODEL = "x-ai/grok-4.3"
DEFAULT_REASONING_EFFORT = "minimal"
TEMPERATURE = 0.0
CONCURRENCY = 10
MAX_RETRIES = 5
RETRY_BACKOFF_CAP_S = 30.0
IOU_THRESHOLD = 0.5

REASONING_EFFORT = DEFAULT_REASONING_EFFORT

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
if not OPENROUTER_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to .env "
        "(get one from https://openrouter.ai/keys)."
    )

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

def invoke_grok(messages: list[dict]):
    """Single Grok 4.3 chat.completions call via OpenRouter."""
    return openrouter_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        extra_body={"reasoning": {"effort": REASONING_EFFORT}},
    )


async def process_sample(sample: dict, semaphore: asyncio.Semaphore,
                         writer: JsonlWriter, progress: dict) -> dict | None:
    orig_w, orig_h = sample["image"].size
    data_url, sent_w, sent_h = pil_to_data_url(sample["image"])
    sx = sent_w / orig_w
    sy = sent_h / orig_h
    gt_xyxy = coco_bbox_to_xyxy(sample["bbox_xywh"], sx, sy)

    prompt = PROMPT_TEMPLATE.format(
        width=sent_w, height=sent_h, expression=sample["expression"]
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }]
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            async with semaphore:
                start = time.perf_counter()
                response = await asyncio.to_thread(invoke_grok, messages)
                latency_ms = int((time.perf_counter() - start) * 1000)
            content = (response.choices[0].message.content or "").strip()
            request_id = getattr(response, "id", None)
            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            pred_box = parse_box(content, sent_w, sent_h)
            iou = compute_iou(pred_box, gt_xyxy) if pred_box else 0.0
            correct = pred_box is not None and iou >= IOU_THRESHOLD

            record = {
                "id": sample["id"],
                "question_id": sample["question_id"],
                "file_name": sample["file_name"],
                "expression": sample["expression"],
                "all_expressions": sample["all_expressions"],
                "image_width": sent_w,
                "image_height": sent_h,
                "gt_bbox_xyxy": gt_xyxy,
                "pred_bbox_xyxy": pred_box,
                "iou": iou,
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
                mark = "OK"
            else:
                mark = "X "
            tqdm.write(
                f"[{progress['done']}/{progress['total']}] {mark} "
                f"id={sample['id']} iou={iou:.3f} "
                f"pred={pred_box} gt={[round(x,1) for x in gt_xyxy]} "
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
                await asyncio.sleep(min(2 ** (attempt - 1), RETRY_BACKOFF_CAP_S))

    progress["failed"] += 1
    tqdm.write(f"[FAILED] id={sample['id']} after {MAX_RETRIES} attempts: {last_error}")
    return None


def print_summary(metrics: dict, dataset_name: str, split: str):
    print(f"\n{'=' * 60}")
    print(f"Grounding Results — {dataset_name} / {split} ({MODEL}, reasoning_effort={REASONING_EFFORT})")
    print(f"{'=' * 60}")
    print(f"Acc@IoU=0.5 : {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"Mean IoU    : {metrics['mean_iou']:.4f}")
    print(f"Unparsed    : {metrics['unparsed']}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(f"Latency     : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
              f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms max={lat['max_ms']}ms")
    print("\nIoU thresholds:")
    for t, acc in metrics["iou_thresholds"].items():
        print(f"  {t}: {acc:.4f}")


def build_tag(dataset: str, split: str) -> str:
    ds_slug = dataset.split("/")[-1].lower().replace("+", "plus")
    return f"grok43_reasoning{REASONING_EFFORT}_{ds_slug}_{split}"


async def run_predictions(dataset_name: str, split: str, pred_path: Path, limit: int | None):
    print(f"Loading {dataset_name}, split={split}...")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} rows")

    samples = build_samples(dataset)
    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
        print(f"--limit applied: will run at most {limit} sample(s)")
    print(f"Resume: {len(done_ids)} already completed, {len(pending)} remaining "
          f"(checkpoint: {pred_path})")
    if not pending:
        return

    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY)
    )

    writer = JsonlWriter(pred_path)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress = {"total": len(pending), "done": 0, "correct": 0, "failed": 0}
    tasks = [process_sample(s, semaphore, writer, progress) for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc=f"{dataset_name}/{split}")
    except Exception:
        traceback.print_exc()
    print(f"\nRun finished: {progress['done']}/{progress['total']} answered, "
          f"{progress['correct']} correct, {progress['failed']} failed.")


def run_evaluation(dataset_name: str, split: str, pred_path: Path, metrics_path: Path):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)

    metrics = compute_metrics(results)
    print_summary(metrics, dataset_name, split)
    output = {
        **metrics,
        "dataset": dataset_name, "split": split,
        "reasoning_effort": REASONING_EFFORT, "temperature": TEMPERATURE,
        "concurrency": CONCURRENCY, "iou_threshold": IOU_THRESHOLD,
        "model": MODEL, "provider": "openrouter",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    global REASONING_EFFORT
    parser = argparse.ArgumentParser(
        description="RefCOCO benchmark for xAI Grok 4.3 via OpenRouter "
                    "(strict head-to-head with refcoco.py — same prompt, parser, IoU)"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="lmms-lab/RefCOCO | lmms-lab/RefCOCO+ | lmms-lab/RefCOCOg")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help="val | testA | testB | test (availability varies by dataset)")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        choices=["minimal", "low", "medium", "high"],
                        help="OpenRouter reasoning.effort (Grok 4.3 cannot fully disable; "
                             "minimal is the lowest accepted tier)")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N unanswered samples")
    args = parser.parse_args()
    REASONING_EFFORT = args.reasoning_effort

    tag = build_tag(args.dataset, args.split)
    pred_path = RESULTS_DIR / f"{tag}_responses.jsonl"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"

    if args.evaluate_only:
        run_evaluation(args.dataset, args.split, pred_path, metrics_path)
    elif args.predict_only:
        asyncio.run(run_predictions(args.dataset, args.split, pred_path, limit=args.limit))
    else:
        asyncio.run(run_predictions(args.dataset, args.split, pred_path, limit=args.limit))
        run_evaluation(args.dataset, args.split, pred_path, metrics_path)


if __name__ == "__main__":
    main()
