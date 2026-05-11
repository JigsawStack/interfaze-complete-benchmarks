"""
OCRBench v2 benchmark for OpenAI.
10,000 QA pairs across 30 task types (EN + CN).

Usage:
    uv run -m benchmarks.ocrbench_v2.ocrbench_v2_openai
    uv run -m benchmarks.ocrbench_v2.ocrbench_v2_openai --predict-only
    uv run -m benchmarks.ocrbench_v2.ocrbench_v2_openai --evaluate-only
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
PRED_OUTPUT = RESULTS_DIR / "ocrbench_v2_openai_predictions.json"
EVAL_OUTPUT = RESULTS_DIR / "ocrbench_v2_openai_scored.json"

sys.path.insert(0, str(PROJECT_ROOT))
from src.commons_openai import invoke_openai  # noqa: E402

MODEL = "gpt-5.4"
RATE_LIMIT = 25
MAX_RETRIES = 3

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
            response = await asyncio.to_thread(invoke_openai, messages, MODEL)
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
            else:
                print(
                    f"Failed after {MAX_RETRIES} attempts for id={sample_meta['id']}: {e}"
                )
                return ""


BATCH_SIZE = 100


async def run_predictions():
    print("Loading OCRBench v2 from HuggingFace...")
    dataset = load_dataset("lmms-lab/OCRBench-v2", split="test")
    total = len(dataset)
    print(f"Loaded {total} samples")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing = {}
    if PRED_OUTPUT.exists():
        with open(PRED_OUTPUT) as f:
            for item in json.load(f):
                if item.get("predict", "") != "":
                    existing[item["id"]] = item
        print(f"Resuming: {len(existing)} successful predictions found, skipping them")

    rate_limiter = RateLimiter(RATE_LIMIT)
    output_data = {}
    output_data.update(existing)
    num_retried = 0

    from tqdm import tqdm

    for batch_start in tqdm(range(0, total, BATCH_SIZE), desc="Batches"):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = dataset[batch_start:batch_end]

        samples = []
        for i in range(len(batch["id"])):
            sample_id = batch["id"][i]
            if sample_id in existing:
                continue
            image_url = pil_to_data_url(batch["image"][i])
            question = batch["question"][i]
            if batch["type"][i] == "text spotting en":
                question = get_spotting_prompt(question)
            samples.append(
                {
                    "id": sample_id,
                    "dataset_name": batch["dataset_name"][i],
                    "type": batch["type"][i],
                    "question": question,
                    "answers": batch["answers"][i],
                    "image_url": image_url,
                }
            )
        del batch

        if not samples:
            continue

        num_retried += len(samples)
        tasks = [process_sample(s, rate_limiter) for s in samples]
        predictions = await tqdm_asyncio.gather(
            *tasks, desc=f"Predicting {batch_start}-{batch_end}", leave=False
        )

        for sample, pred in zip(samples, predictions):
            output_data[sample["id"]] = {
                "id": sample["id"],
                "dataset_name": sample["dataset_name"],
                "type": sample["type"],
                "question": sample["question"],
                "answers": sample["answers"],
                "predict": pred,
            }

        del samples, predictions

    final_data = [output_data[i] for i in sorted(output_data.keys())]
    with open(PRED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    num_failures = sum(1 for d in final_data if d.get("predict", "") == "")
    print(f"\nPredictions saved to {PRED_OUTPUT}")
    print(
        f"Total: {total} | Retried: {num_retried} | Remaining failures: {num_failures}"
    )


def run_evaluation():
    if not PRED_OUTPUT.exists():
        print(f"No predictions found at {PRED_OUTPUT}")
        print("Run with --predict-only first, or without flags to do both.")
        sys.exit(1)

    eval_scripts_dir = BENCHMARK_DIR / "eval_scripts"
    sys.path.insert(0, str(eval_scripts_dir))

    import os
    original_cwd = os.getcwd()
    os.chdir(BENCHMARK_DIR)

    print("Step 1: Scoring individual samples...")
    from benchmarks.ocrbench_v2.eval_scripts.eval import process_predictions  # noqa: E402

    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    process_predictions(str(PRED_OUTPUT), str(EVAL_OUTPUT))

    os.chdir(original_cwd)
    print(f"Scored results saved to {EVAL_OUTPUT}")

    print("\nStep 2: Computing overall metrics...")
    with open(EVAL_OUTPUT) as f:
        scored_data = json.load(f)

    en_scores = {
        "text_recognition": [],
        "text_detection": [],
        "text_spotting": [],
        "relationship_extraction": [],
        "element_parsing": [],
        "mathematical_calculation": [],
        "visual_text_understanding": [],
        "knowledge_reasoning": [],
    }
    cn_scores = {
        "text_recognition": [],
        "relationship_extraction": [],
        "element_parsing": [],
        "visual_text_understanding": [],
        "knowledge_reasoning": [],
    }

    type_to_en = {
        "text recognition en": "text_recognition",
        "fine-grained text recognition en": "text_recognition",
        "full-page OCR en": "text_recognition",
        "text grounding en": "text_detection",
        "VQA with position en": "text_detection",
        "text spotting en": "text_spotting",
        "key information extraction en": "relationship_extraction",
        "key information mapping en": "relationship_extraction",
        "document parsing en": "element_parsing",
        "chart parsing en": "element_parsing",
        "table parsing en": "element_parsing",
        "formula recognition en": "element_parsing",
        "math QA en": "mathematical_calculation",
        "text counting en": "mathematical_calculation",
        "document classification en": "visual_text_understanding",
        "cognition VQA en": "visual_text_understanding",
        "diagram QA en": "visual_text_understanding",
        "reasoning VQA en": "knowledge_reasoning",
        "science QA en": "knowledge_reasoning",
        "APP agent en": "knowledge_reasoning",
        "ASCII art classification en": "knowledge_reasoning",
    }
    type_to_cn = {
        "full-page OCR cn": "text_recognition",
        "key information extraction cn": "relationship_extraction",
        "handwritten answer extraction cn": "relationship_extraction",
        "document parsing cn": "element_parsing",
        "table parsing cn": "element_parsing",
        "formula recognition cn": "element_parsing",
        "cognition VQA cn": "visual_text_understanding",
        "reasoning VQA cn": "knowledge_reasoning",
        "text translation cn": "knowledge_reasoning",
    }

    for item in scored_data:
        if "ignore" in item:
            continue
        t = item["type"]
        if t in type_to_en:
            en_scores[type_to_en[t]].append(item["score"])
        elif t in type_to_cn:
            cn_scores[type_to_cn[t]].append(item["score"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    en_avgs = {k: avg(v) for k, v in en_scores.items() if v}
    cn_avgs = {k: avg(v) for k, v in cn_scores.items() if v}
    en_overall = avg(list(en_avgs.values()))
    cn_overall = avg(list(cn_avgs.values()))

    print(f"\n{'=' * 60}")
    print(f"OCRBench v2 Results ({MODEL})")
    print(f"{'=' * 60}")
    print(f"\n{'Category':<30} {'EN':>8} {'CN':>8}")
    print("-" * 48)
    all_cats = sorted(set(list(en_scores.keys()) + list(cn_scores.keys())))
    for cat in all_cats:
        en_val = f"{en_avgs[cat]:.3f}" if cat in en_avgs else "  -"
        cn_val = f"{cn_avgs[cat]:.3f}" if cat in cn_avgs else "  -"
        print(f"{cat:<30} {en_val:>8} {cn_val:>8}")
    print("-" * 48)
    print(f"{'OVERALL':<30} {en_overall:>8.3f} {cn_overall:>8.3f}")

    metrics = {
        "en_scores": {
            k: {"avg": avg(v), "count": len(v)} for k, v in en_scores.items()
        },
        "cn_scores": {
            k: {"avg": avg(v), "count": len(v)} for k, v in cn_scores.items()
        },
        "en_overall": en_overall,
        "cn_overall": cn_overall,
        "model": MODEL,
    }
    metrics_path = RESULTS_DIR / "ocrbench_v2_openai_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="OCRBench v2 benchmark for OpenAI")
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
