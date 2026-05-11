"""
RefCOCO benchmark for Gemini 2.5 Pro.

Mirrors the protocol used by `refcoco.py` (interfaze-beta) so numbers are
directly comparable on the same splits at Acc@IoU=0.5. Only the inference
call changes — same prompt, same parser, same IoU, same checkpointing.

Reasoning: Gemini 2.5 Pro cannot fully disable thinking (min budget = 128).
We use thinking_budget=128 to approximate the interfaze "reasoning off"
default. Temperature pinned to 0.0 to match.

Datasets (lmms-lab packaging — same as standard RefCOCO splits):
    lmms-lab/RefCOCO    val 8811, test 5000, testA 1975, testB 1810
    lmms-lab/RefCOCO+   val 8823, testA 1975, testB 1798
    lmms-lab/RefCOCOg   val 7573, test 9602  (UMD split)

Usage:
    uv run -m benchmarks.obj_detection.refcoco_gemini --split testA
    uv run -m benchmarks.obj_detection.refcoco_gemini --split testA --limit 20
    uv run -m benchmarks.obj_detection.refcoco_gemini --split testA --evaluate-only

Env: GEMINI_KEY must be set (loaded from .env).
"""

import sys
import json
import os
import re
import time
import asyncio
import argparse
import traceback
import concurrent.futures
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
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
MODEL = "gemini-2.5-pro"
THINKING_BUDGET = 128  # Pro min — Pro API rejects 0 ("only works in thinking mode")
TEMPERATURE = 0.0
CONCURRENCY = 5  # Pro 503s ("model overloaded") under heavier load
MAX_RETRIES = 6
RETRY_BACKOFF_CAP_S = 30.0
IOU_THRESHOLD = 0.5

GEMINI_KEY = os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError(
        "GEMINI_KEY is not set. Add it to .env "
        "(get one from https://aistudio.google.com/app/apikey)."
    )

# google-genai's Client is thread-safe; one shared instance is fine.
gemini_client = genai.Client(api_key=GEMINI_KEY)


def invoke_gemini(prompt: str, jpeg_bytes: bytes):
    """Single Gemini 2.5 Pro generate_content call. Returns the SDK response."""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            ],
        ),
    ]
    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
    )
    return gemini_client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )


def data_url_to_jpeg_bytes(data_url: str) -> bytes:
    """Strip the `data:image/jpeg;base64,` prefix and decode."""
    import base64
    _, b64 = data_url.split(",", 1)
    return base64.b64decode(b64)


# Gemini's native bounding-box convention: [ymin, xmin, ymax, xmax] normalized
# to 0-1000 — documented in https://ai.google.dev/gemini-api/docs/image-understanding
# (and the format every Gemini grounding example uses). Asking for this format
# is the fair head-to-head equivalent of the Qwen / InternVL / interfaze prompts:
# each model is queried in the coordinate convention it was trained on.
PROMPT_GEMINI_TEMPLATE = """Locate the single region in the image described by the following expression and return its bounding box.

Expression: "{expression}"

Output ONLY a JSON object on the last line in this exact format:
{{"box_2d": [ymin, xmin, ymax, xmax]}}

Coordinates must be normalized to 0-1000 where (0,0) is the top-left corner and (1000,1000) is the bottom-right corner of the image. Think step by step before answering."""


JSON_BOX_PATTERN = re.compile(
    r'"box_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)
BARE_4TUPLE_PATTERN = re.compile(
    r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def parse_box_gemini(text: str, width: int, height: int) -> list[float] | None:
    """Parse Gemini's native [ymin, xmin, ymax, xmax] in 0-1000 → pixel xyxy.

    Tries `"box_2d": [...]` first, falls back to the last bare 4-tuple in the
    response. Returns [x1, y1, x2, y2] in pixel space, or None if unparseable.
    """
    if not text:
        return None
    m = None
    for match in JSON_BOX_PATTERN.finditer(text):
        m = match
    if m is None:
        for match in BARE_4TUPLE_PATTERN.finditer(text):
            m = match
    if m is None:
        return None
    ymin, xmin, ymax, xmax = (float(m.group(i)) for i in range(1, 5))
    # Gemini almost always emits 0-1000 normalized; guard against the rare
    # case where it emits raw pixels by checking whether values fit in [0, 1000].
    max_coord = max(abs(ymin), abs(xmin), abs(ymax), abs(xmax))
    if max_coord <= 1000:
        x1 = xmin * width / 1000.0
        y1 = ymin * height / 1000.0
        x2 = xmax * width / 1000.0
        y2 = ymax * height / 1000.0
    else:
        x1, y1, x2, y2 = xmin, ymin, xmax, ymax
    return [x1, y1, x2, y2]


async def process_sample(sample: dict, semaphore: asyncio.Semaphore,
                         writer: JsonlWriter, progress: dict) -> dict | None:
    orig_w, orig_h = sample["image"].size
    data_url, sent_w, sent_h = pil_to_data_url(sample["image"])
    jpeg_bytes = data_url_to_jpeg_bytes(data_url)
    sx = sent_w / orig_w
    sy = sent_h / orig_h
    gt_xyxy = coco_bbox_to_xyxy(sample["bbox_xywh"], sx, sy)

    prompt = PROMPT_TEMPLATE.format(
        width=sent_w, height=sent_h, expression=sample["expression"]
    )
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            async with semaphore:
                start = time.perf_counter()
                response = await asyncio.to_thread(invoke_gemini, prompt, jpeg_bytes)
                latency_ms = int((time.perf_counter() - start) * 1000)
            content = (response.text or "").strip()
            request_id = getattr(response, "response_id", None)
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
    print(f"Grounding Results — {dataset_name} / {split} ({MODEL}, thinking_budget={THINKING_BUDGET})")
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
    return f"gemini25pro_{ds_slug}_{split}"


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
        "thinking_budget": THINKING_BUDGET, "temperature": TEMPERATURE,
        "concurrency": CONCURRENCY, "iou_threshold": IOU_THRESHOLD, "model": MODEL,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RefCOCO / RefCOCO+ / RefCOCOg benchmark for Gemini 2.5 Pro "
                    "(Acc@IoU=0.5, comparable to refcoco.py interfaze run)"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="lmms-lab/RefCOCO | lmms-lab/RefCOCO+ | lmms-lab/RefCOCOg")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help="val | testA | testB | test (availability varies by dataset)")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N unanswered samples")
    args = parser.parse_args()

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
