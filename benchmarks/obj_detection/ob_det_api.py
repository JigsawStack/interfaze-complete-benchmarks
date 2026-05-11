"""
RefCOCO benchmark for the JigsawStack /v1/object_detection API.

Mirrors the protocol used by `refcoco.py` (interfaze-beta) so numbers are
directly comparable on the same splits at Acc@IoU=0.5. The only thing that
changes is the inference call: instead of asking a VLM to emit a `Box: [...]`
line, we hit JigsawStack's object_detection endpoint with the referring
expression as a prompt and parse `objects[0].bounds`.

Per-sample we record both:
    - acc / iou           — IoU of the FIRST returned object vs. the GT box.
                            This is the honest, deployable number.
    - oracle_iou / oracle_correct — best IoU across ALL returned objects.
                            Diagnostic upper bound — tells us whether the
                            right box was in the response and we just picked
                            wrong.

Datasets (lmms-lab packaging — same data as standard RefCOCO splits):
    lmms-lab/RefCOCO    val 8811, test 5000, testA 1975, testB 1810
    lmms-lab/RefCOCO+   val 8823, testA 1975, testB 1798
    lmms-lab/RefCOCOg   val 7573, test 9602  (UMD split)

Usage:
    uv run -m benchmarks.obj_detection.ob_det_api                    # RefCOCO val
    uv run -m benchmarks.obj_detection.ob_det_api --split testA
    uv run -m benchmarks.obj_detection.ob_det_api --dataset lmms-lab/RefCOCO+ --split testB
    uv run -m benchmarks.obj_detection.ob_det_api --limit 20
    uv run -m benchmarks.obj_detection.ob_det_api --evaluate-only

Env: JIGSAWSTACK_API_KEY must be set (loaded from .env).

Checkpointing: results/jigsawstack_<dataset>_<split>_responses.jsonl —
each successful sample is appended as it finishes; reruns skip completed ids.
"""

import sys
import json
import os
import time
import base64
import asyncio
import argparse
import traceback
import concurrent.futures
from io import BytesIO
from pathlib import Path

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse sample construction, IoU, and resume logic from refcoco.py — keeps
# scoring byte-for-byte identical to the interfaze run.
from benchmarks.obj_detection.refcoco import (  # noqa: E402
    JsonlWriter,
    build_samples,
    coco_bbox_to_xyxy,
    compute_iou,
    load_completed_ids,
    load_records,
    pil_to_data_url,
)

load_dotenv()

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATASET = "lmms-lab/RefCOCO"
DEFAULT_SPLIT = "val"
CONCURRENCY = 10  # JigsawStack rate-limits more aggressively than interfaze
MAX_RETRIES = 3
IOU_THRESHOLD = 0.5
REQUEST_TIMEOUT = 120.0

JIGSAWSTACK_URL = "http://localhost:3000/api/v1/object_detection"
JIGSAWSTACK_API_KEY = os.getenv("JIGSAWSTACK_API_KEY")
if not JIGSAWSTACK_API_KEY:
    raise RuntimeError(
        "JIGSAWSTACK_API_KEY is not set. Add it to .env "
        "(get one from https://jigsawstack.com/dashboard)."
    )


def invoke_jigsawstack_obj_detection(image_data_url: str, expression: str) -> dict:
    """POST to /v1/object_detection with the referring expression as a prompt.

    `annotated_image=False` skips the rendered overlay we don't need, which
    saves both bandwidth and server-side time. `x-jigsaw-skip-cache: true`
    matches the example curl and ensures we measure real inference latency
    rather than cache hits."""
    payload = {
        "url": image_data_url,
        "annotated_image": False,
        "features": ["object"],
        "prompts": [expression],
    }
    headers = {
        "x-api-key": JIGSAWSTACK_API_KEY,
        "x-jigsaw-skip-cache": "true",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(JIGSAWSTACK_URL, json=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    if not data.get("success", False):
        raise RuntimeError(f"API error: {str(data)[:500]}")
    return data


def bounds_to_xyxy(bounds: dict) -> list[float] | None:
    """Convert JigsawStack's `bounds` object to [x1, y1, x2, y2].

    The API returns four corners (top_left/top_right/bottom_left/bottom_right);
    we only need the diagonal pair. Falls back to width/height if a corner is
    missing."""
    tl = bounds.get("top_left") or {}
    br = bounds.get("bottom_right") or {}
    if "x" in tl and "y" in tl and "x" in br and "y" in br:
        return [float(tl["x"]), float(tl["y"]), float(br["x"]), float(br["y"])]
    if "x" in tl and "y" in tl and "width" in bounds and "height" in bounds:
        x1, y1 = float(tl["x"]), float(tl["y"])
        return [x1, y1, x1 + float(bounds["width"]), y1 + float(bounds["height"])]
    return None


def extract_boxes(api_response: dict) -> list[list[float]]:
    """Return all detected boxes as [x1,y1,x2,y2], in the order returned."""
    boxes: list[list[float]] = []
    for obj in api_response.get("objects") or []:
        bounds = obj.get("bounds")
        if not bounds:
            continue
        xyxy = bounds_to_xyxy(bounds)
        if xyxy is not None:
            boxes.append(xyxy)
    return boxes


async def process_sample(
    sample: dict, semaphore: asyncio.Semaphore, writer: JsonlWriter, progress: dict
) -> dict | None:
    orig_w, orig_h = sample["image"].size
    data_url, sent_w, sent_h = pil_to_data_url(sample["image"])
    sx = sent_w / orig_w
    sy = sent_h / orig_h
    gt_xyxy = coco_bbox_to_xyxy(sample["bbox_xywh"], sx, sy)

    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            async with semaphore:
                start = time.perf_counter()
                api_response = await asyncio.to_thread(
                    invoke_jigsawstack_obj_detection,
                    data_url,
                    sample["expression"],
                )
                latency_ms = int((time.perf_counter() - start) * 1000)

            boxes = extract_boxes(api_response)
            num_objects = len(boxes)

            # Honest pick: first object, matching how a downstream caller
            # would use the API without ground truth.
            pred_box = boxes[0] if boxes else None
            iou = compute_iou(pred_box, gt_xyxy) if pred_box else 0.0
            correct = pred_box is not None and iou >= IOU_THRESHOLD

            # Oracle: best IoU across all returned boxes — tells us how often
            # the right answer was in the response but we picked wrong.
            if boxes:
                oracle_iou = max(compute_iou(b, gt_xyxy) for b in boxes)
            else:
                oracle_iou = 0.0
            oracle_correct = oracle_iou >= IOU_THRESHOLD

            log_id = api_response.get("log_id")
            usage = api_response.get("_usage")

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
                "all_pred_boxes": boxes,
                "num_objects": num_objects,
                "iou": iou,
                "correct": correct,
                "oracle_iou": oracle_iou,
                "oracle_correct": oracle_correct,
                "response": api_response,
                "log_id": log_id,
                "usage": usage,
                "latency_ms": latency_ms,
                "attempts": attempt,
            }
            await writer.append(record)

            progress["done"] += 1
            if correct:
                progress["correct"] += 1
                mark = "OK"
            elif oracle_correct:
                mark = "o "  # right box was returned, just not first
            else:
                mark = "X "
            tqdm.write(
                f"[{progress['done']}/{progress['total']}] {mark} "
                f"id={sample['id']} iou={iou:.3f} oracle={oracle_iou:.3f} "
                f"n={num_objects} pred={pred_box} "
                f"gt={[round(x, 1) for x in gt_xyxy]} "
                f"latency={latency_ms}ms log_id={log_id} attempt={attempt}"
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
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    oracle_correct = sum(1 for r in results if r.get("oracle_correct"))
    no_objects = sum(1 for r in results if not r.get("num_objects"))
    ious = [r["iou"] for r in results if isinstance(r.get("iou"), (int, float))]
    oracle_ious = [
        r["oracle_iou"]
        for r in results
        if isinstance(r.get("oracle_iou"), (int, float))
    ]
    latencies = [
        r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int)
    ]
    n_objects = [
        r["num_objects"] for r in results if isinstance(r.get("num_objects"), int)
    ]

    thresholds = [0.3, 0.5, 0.7, 0.75, 0.9]
    at_threshold = {
        f"acc@{t}": (sum(1 for iou in ious if iou >= t) / len(ious) if ious else 0)
        for t in thresholds
    }
    oracle_at_threshold = {
        f"oracle_acc@{t}": (
            sum(1 for iou in oracle_ious if iou >= t) / len(oracle_ious)
            if oracle_ious
            else 0
        )
        for t in thresholds
    }

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

    return {
        "accuracy": correct / total if total else 0.0,
        "oracle_accuracy": oracle_correct / total if total else 0.0,
        "correct": correct,
        "oracle_correct": oracle_correct,
        "total": total,
        "no_objects_returned": no_objects,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "mean_oracle_iou": sum(oracle_ious) / len(oracle_ious) if oracle_ious else 0.0,
        "mean_objects_per_response": sum(n_objects) / len(n_objects)
        if n_objects
        else 0.0,
        "iou_thresholds": at_threshold,
        "oracle_iou_thresholds": oracle_at_threshold,
        "latency": latency_stats,
    }


def print_summary(metrics: dict, dataset_name: str, split: str):
    print(f"\n{'=' * 60}")
    print(
        f"Grounding Results — {dataset_name} / {split} (JigsawStack object_detection)"
    )
    print(f"{'=' * 60}")
    print(
        f"Acc@IoU=0.5        : {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})"
    )
    print(
        f"Oracle Acc@IoU=0.5 : {metrics['oracle_accuracy']:.4f} ({metrics['oracle_correct']}/{metrics['total']})"
    )
    print(f"Mean IoU           : {metrics['mean_iou']:.4f}")
    print(f"Mean Oracle IoU    : {metrics['mean_oracle_iou']:.4f}")
    print(f"Avg objects / resp : {metrics['mean_objects_per_response']:.2f}")
    print(f"No-object responses: {metrics['no_objects_returned']}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(
            f"Latency            : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
            f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms max={lat['max_ms']}ms"
        )
    print("\nIoU thresholds (first-object pick):")
    for t, acc in metrics["iou_thresholds"].items():
        print(f"  {t}: {acc:.4f}")
    print("\nIoU thresholds (oracle / best-of-N):")
    for t, acc in metrics["oracle_iou_thresholds"].items():
        print(f"  {t}: {acc:.4f}")


def build_tag(dataset: str, split: str) -> str:
    ds_slug = dataset.split("/")[-1].lower().replace("+", "plus")
    return f"jigsawstack_{ds_slug}_{split}"


async def run_predictions(
    dataset_name: str, split: str, pred_path: Path, limit: int | None
):
    print(f"Loading {dataset_name}, split={split}...")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} rows")

    samples = build_samples(dataset)
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
    print(
        f"\nRun finished: {progress['done']}/{progress['total']} answered, "
        f"{progress['correct']} correct, {progress['failed']} failed."
    )


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
        "dataset": dataset_name,
        "split": split,
        "concurrency": CONCURRENCY,
        "iou_threshold": IOU_THRESHOLD,
        "model": "jigsawstack:object_detection",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RefCOCO / RefCOCO+ / RefCOCOg benchmark for the JigsawStack "
        "object_detection API (Acc@IoU=0.5, comparable to refcoco.py)"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="lmms-lab/RefCOCO | lmms-lab/RefCOCO+ | lmms-lab/RefCOCOg",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="val | testA | testB | test (availability varies by dataset)",
    )
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N unanswered samples",
    )
    args = parser.parse_args()

    tag = build_tag(args.dataset, args.split)
    pred_path = RESULTS_DIR / f"{tag}_responses.jsonl"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"

    if args.evaluate_only:
        run_evaluation(args.dataset, args.split, pred_path, metrics_path)
    elif args.predict_only:
        asyncio.run(
            run_predictions(args.dataset, args.split, pred_path, limit=args.limit)
        )
    else:
        asyncio.run(
            run_predictions(args.dataset, args.split, pred_path, limit=args.limit)
        )
        run_evaluation(args.dataset, args.split, pred_path, metrics_path)


if __name__ == "__main__":
    main()
