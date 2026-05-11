"""
RefCOCO (+ optionally RefCOCO+ / RefCOCOg) benchmark for Interfaze.

This is the canonical VLM grounding benchmark: given an image and a referring
expression, the model outputs a bounding box. A prediction is considered
correct if its IoU with the ground-truth box > 0.5 (Acc@IoU=0.5). This is
exactly the metric reported by Qwen3-VL, DeepSeek-VL2, InternVL 2.5/3,
GLM-4.x-V, CogVLM-Grounding, Kosmos-2 and essentially every VLM paper.

Datasets (all lmms-lab packaging — same underlying RefCOCO data as the
original Kazemzadeh et al. 2014 / Mao et al. 2016 splits):
    lmms-lab/RefCOCO    val 8811, test 5000, testA 1975, testB 1810
    lmms-lab/RefCOCO+   val 8823, testA 1975, testB 1798
    lmms-lab/RefCOCOg   val 7573, test 9602  (UMD split)

Usage:
    uv run -m benchmarks.obj_detection.refcoco                    # RefCOCO val
    uv run -m benchmarks.obj_detection.refcoco --split testA
    uv run -m benchmarks.obj_detection.refcoco --dataset lmms-lab/RefCOCO+ --split testB
    uv run -m benchmarks.obj_detection.refcoco --limit 20
    uv run -m benchmarks.obj_detection.refcoco --evaluate-only

Checkpointing:
    Each successful sample is appended to results/<tag>_responses.jsonl
    as it finishes. Reruns only query samples still missing. Failed-after-
    retries samples are NOT written, so they retry on next run.
"""

import sys
import json
import os
import re
import time
import base64
import asyncio
import argparse
import traceback
import concurrent.futures
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.commons import invoke_interfaze  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATASET = "lmms-lab/RefCOCO"
DEFAULT_SPLIT = "val"
REASONING_EFFORT = None  # None => omit the reasoning_effort param entirely (interfaze default = off)
TEMPERATURE = 0.0  # deterministic decoding — matches standard VLM grounding eval protocol
CONCURRENCY = 25
MAX_RETRIES = 3
IOU_THRESHOLD = 0.5

# Canonical RefCOCO grounding prompt — wording from lmms-eval's `refcoco_rec`
# task (InternVL / Qwen-VL convention) plus the single output-format line that
# published "general VLM on RefCOCO" tables (LLaVA, GPT-4V, Claude) use to
# disambiguate xyxy vs yxyx ordering. No coord-space pinning, no step-by-step.
# `{width}` and `{height}` are accepted but unused so callers don't break.
PROMPT_TEMPLATE = (
    "Please provide the bounding box coordinate of the region this sentence describes: "
    "{expression}\n\n"
    "Output the coordinates in the format [x_min, y_min, x_max, y_max]."
)


BOX_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:box|answer|bounding\s*box)[\s:]*"
    r"\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\]?"
)
BOXED_PATTERN = re.compile(
    r"\\boxed\{\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\}"
)
BARE_4TUPLE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


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


def pil_to_data_url(image, max_side: int = 1024) -> tuple[str, int, int]:
    """Encode PIL image as data URL. Returns (url, width, height) of the
    image as sent to the model — coords are interpreted in this space."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h))
        w, h = new_w, new_h
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", w, h


def coco_bbox_to_xyxy(bbox, scale_x: float = 1.0, scale_y: float = 1.0) -> list[float]:
    """RefCOCO ground truth is COCO format [x, y, w, h]. Convert to
    [x1, y1, x2, y2] and optionally rescale into a resized image space."""
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y]


JSON_TLBR_PATTERN = re.compile(
    r'"top_left"\s*:\s*\{[^}]*?"x"\s*:\s*(-?\d+(?:\.\d+)?)[^}]*?"y"\s*:\s*(-?\d+(?:\.\d+)?)'
    r'[^}]*?\}[^}]*?"bottom_right"\s*:\s*\{[^}]*?"x"\s*:\s*(-?\d+(?:\.\d+)?)'
    r'[^}]*?"y"\s*:\s*(-?\d+(?:\.\d+)?)',
    re.DOTALL,
)
JSON_BOX2D_PATTERN = re.compile(
    r'"box_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def parse_box(text: str, width: int, height: int) -> list[float] | None:
    """Extract a [x1, y1, x2, y2] bounding box in pixel space from the model
    response. Robust to the formats general VLMs commonly emit on RefCOCO:

      - "Box: [a,b,c,d]" / "\\boxed{[a,b,c,d]}"  → [x1,y1,x2,y2]
      - bare last [a,b,c,d] in tail              → [x1,y1,x2,y2]
      - {"top_left":{"x","y"}, "bottom_right":{"x","y"}}
      - {"box_2d":[ymin,xmin,ymax,xmax]}  (Gemini convention, 0-1000)

    Coordinate space is auto-detected: 0-1.0 floats, 0-1000 normalized, or
    raw pixels — picks whichever fits the image dimensions.
    """
    if not text:
        return None

    # 1. JSON object with top_left / bottom_right corners — already xyxy.
    m = None
    for match in JSON_TLBR_PATTERN.finditer(text):
        m = match
    if m is not None:
        return _normalize_to_pixels(
            [float(m.group(i)) for i in range(1, 5)], width, height, order="xyxy"
        )

    # 2. Gemini-style {"box_2d": [...]} — [ymin, xmin, ymax, xmax].
    m = None
    for match in JSON_BOX2D_PATTERN.finditer(text):
        m = match
    if m is not None:
        return _normalize_to_pixels(
            [float(m.group(i)) for i in range(1, 5)], width, height, order="yxyx"
        )

    # 3. Box: / Answer: / \boxed{} prefixed line — assume xyxy.
    for pattern in (BOX_LINE_PATTERN, BOXED_PATTERN):
        matches = list(pattern.finditer(text))
        if matches:
            m = matches[-1]
            return _normalize_to_pixels(
                [float(m.group(i)) for i in range(1, 5)], width, height, order="xyxy"
            )

    # 4. Fallback — last bare 4-tuple in the tail.
    tail = text[-800:]
    matches = list(BARE_4TUPLE.finditer(tail))
    if matches:
        m = matches[-1]
        return _normalize_to_pixels(
            [float(m.group(i)) for i in range(1, 5)], width, height, order="xyxy"
        )

    return None


def _normalize_to_pixels(
    box: list[float], width: int, height: int, order: str
) -> list[float]:
    """Convert a 4-tuple to pixel-space [x1, y1, x2, y2] given its order
    ('xyxy' or 'yxyx') and auto-detected coordinate scale (0-1.0 floats /
    0-1000 normalized / raw pixels).
    """
    max_coord = max(abs(c) for c in box)
    # Scale detection: pick the smallest scale whose interpretation fits
    # the image, preferring 0-1 then 0-1000 then pixel.
    if max_coord <= 1.0:
        sx, sy = float(width), float(height)
    elif max_coord <= 1000 and (
        max(width, height) > 1000
        or box[0] > width or box[2] > width
        or box[1] > height or box[3] > height
    ):
        sx, sy = width / 1000.0, height / 1000.0
    else:
        sx = sy = 1.0

    if order == "xyxy":
        return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
    # yxyx (Gemini box_2d) → swap to xyxy and apply per-axis scale
    return [box[1] * sx, box[0] * sy, box[3] * sx, box[2] * sy]


def compute_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # Normalize in case of (x1,y1,x2,y2) with swapped corners.
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_samples(dataset) -> list[dict]:
    """One sample per row, using the first referring expression. Matches
    the convention used by lmms-eval / InternVL grounding evaluation."""
    samples = []
    for i, row in enumerate(dataset):
        answers = row.get("answer")
        if isinstance(answers, str):
            expressions = [answers]
        elif isinstance(answers, list) and answers:
            expressions = [str(a) for a in answers if str(a).strip()]
        else:
            expressions = []
        if not expressions:
            continue
        samples.append({
            "id": f"{row.get('question_id', i)}_{i}",
            "question_id": row.get("question_id"),
            "file_name": row.get("file_name"),
            "image": row["image"],
            "expression": expressions[0],
            "all_expressions": expressions,
            "bbox_xywh": list(row["bbox"]),
        })
    return samples


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
                await asyncio.sleep(2 ** (attempt - 1))

    progress["failed"] += 1
    tqdm.write(f"[FAILED] id={sample['id']} after {MAX_RETRIES} attempts: {last_error}")
    return None


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    unparsed = sum(1 for r in results if r.get("pred_bbox_xyxy") is None)
    ious = [r["iou"] for r in results if isinstance(r.get("iou"), (int, float))]
    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int)]

    # Threshold sweep for sanity
    thresholds = [0.3, 0.5, 0.7, 0.75, 0.9]
    at_threshold = {f"acc@{t}": sum(1 for iou in ious if iou >= t) / len(ious) if ious else 0
                    for t in thresholds}

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
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "unparsed": unparsed,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "iou_thresholds": at_threshold,
        "latency": latency_stats,
    }


def print_summary(metrics: dict, dataset_name: str, split: str):
    print(f"\n{'=' * 60}")
    print(f"Grounding Results — {dataset_name} / {split} (Interfaze, reasoning={REASONING_EFFORT})")
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
    return f"{ds_slug}_{split}"


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

    # Default asyncio thread pool is min(32, cpu_count + 4) — typically 12-16
    # on macOS, which would cap real in-flight HTTP calls below CONCURRENCY.
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
        "reasoning_effort": REASONING_EFFORT, "concurrency": CONCURRENCY,
        "iou_threshold": IOU_THRESHOLD, "model": "interfaze-beta",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RefCOCO / RefCOCO+ / RefCOCOg benchmark for Interfaze "
                    "(Acc@IoU=0.5, comparable to Qwen3-VL / DeepSeek-VL2 / InternVL / GLM-V)"
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
