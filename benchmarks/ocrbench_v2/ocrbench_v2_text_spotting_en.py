"""
OCRBench v2 benchmark for Interfaze — Text Spotting EN only.
Runs predictions and evaluation exclusively for the "text spotting en" task type.

Usage:
    uv run python benchmarks/ocrbench_v2/ocrbench_v2_text_spotting_en.py
    uv run python benchmarks/ocrbench_v2/ocrbench_v2_text_spotting_en.py --predict-only
    uv run python benchmarks/ocrbench_v2/ocrbench_v2_text_spotting_en.py --evaluate-only
"""

import sys
import json
import asyncio
import argparse
import base64
from pathlib import Path
from io import BytesIO

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PRED_OUTPUT = RESULTS_DIR / "ocrbench_v2_text_spotting_en_predictions.json"
EVAL_OUTPUT = RESULTS_DIR / "ocrbench_v2_text_spotting_en_scored.json"

sys.path.insert(0, str(PROJECT_ROOT))
from src.commons import invoke_interfaze  # noqa: E402

RATE_LIMIT = 25
MAX_RETRIES = 3
TARGET_TYPE = "text spotting en"

TEXT_SPOTTING_PROMPT_TEMPLATE = """Use OCR on this image to spot all text at {level}. The OCR tool returns each detected text region with its text content and four corner coordinates: top_left, top_right, bottom_left, bottom_right (each as an x,y pixel pair).

Then use run code to write a Python script that takes those OCR results and:
1. For each text region, compute the axis-aligned bounding box from the four corners:
   - x1 = min of all x coordinates (leftmost)
   - y1 = min of all y coordinates (topmost)
   - x2 = max of all x coordinates (rightmost)
   - y2 = max of all y coordinates (bottommost)
2. Normalize each coordinate to the range 0-1000 by dividing by the image width (for x) or height (for y) and multiplying by 1000, then rounding to an integer.
3. Print the results as a Python list.

Your final answer must be ONLY a Python list in this exact format, with no markdown, no code fences, no explanation:
[(x1, y1, x2, y2, "text"), (x1, y1, x2, y2, "text"), ...]"""


def get_spotting_prompt(original_question: str) -> str:
    """Preserve word-level vs line-level from the original question."""
    if "line-level" in original_question:
        return TEXT_SPOTTING_PROMPT_TEMPLATE.format(level="line-level")
    return TEXT_SPOTTING_PROMPT_TEMPLATE.format(level="word-level")


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


def pil_to_data_url(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(question: str, image_url: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


async def process_sample(sample_meta: dict, rate_limiter):
    messages = build_messages(sample_meta["question"], sample_meta["image_url"])

    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            response = await asyncio.to_thread(invoke_interfaze, messages)
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
            else:
                print(
                    f"Failed after {MAX_RETRIES} attempts for id={sample_meta['id']}: {e}"
                )
                return ""


BATCH_SIZE = 20


async def run_predictions():
    print("Loading OCRBench v2 from HuggingFace...")
    dataset = load_dataset("lmms-lab/OCRBench-v2", split="test")
    total = len(dataset)
    print(f"Loaded {total} samples")

    # Filter to only "text spotting en" samples
    spotting_indices = [i for i in range(total) if dataset[i]["type"] == TARGET_TYPE]
    print(f"Filtered to {len(spotting_indices)} '{TARGET_TYPE}' samples")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Always run fresh
    if PRED_OUTPUT.exists():
        PRED_OUTPUT.unlink()
        print("Cleared previous predictions — running fresh")

    rate_limiter = RateLimiter(RATE_LIMIT)
    output_data = {}
    num_retried = 0

    # Import eval functions for realtime scoring
    eval_scripts_dir = BENCHMARK_DIR / "eval_scripts"
    sys.path.insert(0, str(eval_scripts_dir))
    import os

    original_cwd = os.getcwd()
    os.chdir(BENCHMARK_DIR)
    from eval_text_spotting_en import (
        extract_bounding_boxes,
        spotting_evaluation_normalized,
    )

    def parse_gt(answers):
        """Parse ground truth polygons into axis-aligned bboxes + text."""
        bboxes, contents = [], []
        for line in answers[0].strip().split("\n"):
            parts = line.split(",")
            num_coords = 0
            for p in parts:
                try:
                    int(p.strip())
                    num_coords += 1
                except ValueError:
                    break
            if num_coords >= 8 and num_coords < len(parts):
                coords = [int(p.strip()) for p in parts[:num_coords]]
                text = ",".join(parts[num_coords:])
                x_coords = coords[0::2]
                y_coords = coords[1::2]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                bboxes.append([x1, y1, x2, y1, x2, y2, x1, y2])
                contents.append(text)
            elif num_coords >= 8 and num_coords == len(parts) and num_coords % 2 == 1:
                coords = [int(p.strip()) for p in parts[:num_coords - 1]]
                text = parts[num_coords - 1].strip()
                x_coords = coords[0::2]
                y_coords = coords[1::2]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                bboxes.append([x1, y1, x2, y1, x2, y2, x1, y2])
                contents.append(text)
        return bboxes, contents

    def quick_score(predict_str, answers):
        """Score a single prediction inline."""
        try:
            bboxes, contents = parse_gt(answers)
            if not bboxes:
                return 0.0
            pred_boxes = extract_bounding_boxes(predict_str)
            if not pred_boxes:
                return 0.0
            return spotting_evaluation_normalized(
                pred_boxes, {"bbox": bboxes, "content": contents}
            )
        except Exception:
            return 0.0

    from tqdm import tqdm

    all_scores = []
    num_batches = (len(spotting_indices) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(num_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(spotting_indices))
        batch_indices = spotting_indices[batch_start:batch_end]

        # Prepare all samples in this batch
        samples = []
        for idx in batch_indices:
            row = dataset[idx]
            image_url = pil_to_data_url(row["image"])
            samples.append({
                "id": row["id"],
                "dataset_name": row["dataset_name"],
                "type": row["type"],
                "question": get_spotting_prompt(row["question"]),
                "answers": row["answers"],
                "image_url": image_url,
            })

        # Fire all requests in parallel
        print(f"Batch {batch_num + 1}/{num_batches} — sending {len(samples)} requests in parallel...")
        tasks = [process_sample(s, rate_limiter) for s in samples]
        predictions = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Batch {batch_num + 1}/{num_batches}",
            leave=True,
        )

        # Score all results
        batch_scores = []
        for sample, pred in zip(samples, predictions):
            num_retried += 1
            score = quick_score(pred, sample["answers"])
            batch_scores.append(score)
            all_scores.append(score)

            output_data[sample["id"]] = {
                "id": sample["id"],
                "dataset_name": sample["dataset_name"],
                "type": sample["type"],
                "question": sample["question"],
                "answers": sample["answers"],
                "predict": pred,
            }

        # Save after each batch
        final_data = [output_data[i] for i in sorted(output_data.keys())]
        with open(PRED_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        batch_avg = sum(batch_scores) / len(batch_scores) if batch_scores else 0.0
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(
            f"  Batch {batch_num + 1} done — batch H-mean: {batch_avg:.4f} | overall H-mean: {overall_avg:.4f}"
        )

    os.chdir(original_cwd)

    final_data = [output_data[i] for i in sorted(output_data.keys())]
    num_failures = sum(1 for d in final_data if d.get("predict", "") == "")
    final_hmean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\nPredictions saved to {PRED_OUTPUT}")
    print(
        f"Total: {len(spotting_indices)} | Retried: {num_retried} | Remaining failures: {num_failures}"
    )
    print(f"Final H-mean: {final_hmean:.4f}"
    )


def run_evaluation():
    if not PRED_OUTPUT.exists():
        print(f"No predictions found at {PRED_OUTPUT}")
        print("Run with --predict-only first, or without flags to do both.")
        sys.exit(1)

    eval_scripts_dir = BENCHMARK_DIR / "eval_scripts"
    sys.path.insert(0, str(eval_scripts_dir))

    # spotting_metric.py uses relative paths like ./eval_scripts/spotting_eval/submit
    # so we must chdir to the benchmark dir for the RRC evaluation to work
    import os

    original_cwd = os.getcwd()
    os.chdir(BENCHMARK_DIR)

    # Step 1: Score each sample using eval_text_spotting_en.py
    print("Step 1: Scoring text spotting EN samples...")
    from eval_text_spotting_en import process_predictions  # noqa: E402

    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    process_predictions(str(PRED_OUTPUT), str(EVAL_OUTPUT))

    os.chdir(original_cwd)
    print(f"Scored results saved to {EVAL_OUTPUT}")

    # Step 2: Compute text spotting metrics
    print("\nStep 2: Computing text spotting EN metrics...")
    with open(EVAL_OUTPUT) as f:
        scored_data = json.load(f)

    scores = []
    for item in scored_data:
        if "ignore" in item:
            continue
        if item["type"] == TARGET_TYPE:
            scores.append(item["score"])

    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'=' * 60}")
    print("OCRBench v2 — Text Spotting EN Results (Interfaze)")
    print(f"{'=' * 60}")
    print(f"  Samples:  {len(scores)}")
    print(f"  H-mean:   {avg_score:.4f}")
    print(f"{'=' * 60}")

    # Save metrics
    metrics = {
        "text_spotting_en": {
            "avg": avg_score,
            "count": len(scores),
        },
        "model": "interfaze-beta",
    }
    metrics_path = RESULTS_DIR / "ocrbench_v2_text_spotting_en_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Also update the main metrics file's text_spotting entry
    main_metrics_path = RESULTS_DIR / "ocrbench_v2_metrics.json"
    if main_metrics_path.exists():
        with open(main_metrics_path) as f:
            main_metrics = json.load(f)
        main_metrics["en_scores"]["text_spotting"] = {
            "avg": avg_score,
            "count": len(scores),
        }
        # Recompute en_overall
        en_avgs = {
            k: v["avg"]
            for k, v in main_metrics["en_scores"].items()
            if v.get("count", 0) > 0
        }
        main_metrics["en_overall"] = (
            sum(en_avgs.values()) / len(en_avgs) if en_avgs else 0.0
        )
        with open(main_metrics_path, "w") as f:
            json.dump(main_metrics, f, indent=2)
        print(f"Updated text_spotting in {main_metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OCRBench v2 — Text Spotting EN benchmark for Interfaze"
    )
    parser.add_argument(
        "--predict-only", action="store_true", help="Only generate predictions"
    )
    parser.add_argument(
        "--evaluate-only", action="store_true", help="Only run evaluation"
    )
    args = parser.parse_args()

    if args.evaluate_only:
        run_evaluation()
    elif args.predict_only:
        asyncio.run(run_predictions())
    else:
        asyncio.run(run_predictions())
        run_evaluation()


if __name__ == "__main__":
    main()
