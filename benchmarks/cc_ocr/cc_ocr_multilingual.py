import sys
import json
import re
import asyncio
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from src.commons import invoke_interfaze

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
CHAR_LEVEL_LANGUAGES = {"Arabic", "Japanese", "Korean"}
RATE_LIMIT = 25  # requests per second
MAX_RETRIES = 3


class RateLimiter:
    """Token-bucket rate limiter: allows RATE_LIMIT requests per second."""

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


def token_normalize(token_text, is_lower=False, is_alphanum_only=False):
    if is_lower:
        token_text = token_text.lower()
    if is_alphanum_only:
        token_text = re.sub('[^A-Za-z0-9]+', '', token_text)
    return token_text


def text_normalize_and_tokenize(text, is_keep_blank=True, is_lower=True, is_alphanum_only=False):
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r'\s+', ' ', text)
    if not is_keep_blank:
        text = text.replace(" ", "")
    text_tokens = text.split(" ") if is_keep_blank else list(text)
    text_token_normalized = [token_normalize(t, is_lower, is_alphanum_only) for t in text_tokens]
    text_token_normalized = [x for x in text_token_normalized if len(x) > 0]
    return text_token_normalized


def evaluate_single_sample(gts, preds):
    right_num = 0
    gt_counter_info = dict(Counter(gts))
    pdt_counter_info = dict(Counter(preds))
    for gt_token, gt_count in gt_counter_info.items():
        pred_count = pdt_counter_info.get(gt_token, 0)
        right_num += min(gt_count, pred_count)
    return right_num


def calculate_metrics(response_info, gt_info, is_verbose=False):
    macro_recall_list, macro_precision_list, macro_f1_list = [], [], []
    total_gt_num, total_pred_num, total_right_num = 0, 0, 0
    for file_name, fullbox_gts in gt_info.items():
        fullbox_preds = response_info.get(file_name, [])
        right_num = evaluate_single_sample(fullbox_gts, fullbox_preds)
        total_right_num += right_num
        total_gt_num += len(fullbox_gts)
        total_pred_num += len(fullbox_preds)

        macro_recall = right_num / (len(fullbox_gts) + 1e-9)
        macro_precision = right_num / (len(fullbox_preds) + 1e-9)
        macro_f1 = 2 * macro_recall * macro_precision / (macro_recall + macro_precision + 1e-9)
        macro_recall_list.append(macro_recall)
        macro_precision_list.append(macro_precision)
        macro_f1_list.append(macro_f1)

    final_macro_recall = sum(macro_recall_list) / (len(macro_recall_list) + 1e-9)
    final_macro_precision = sum(macro_precision_list) / (len(macro_precision_list) + 1e-9)
    final_macro_f1 = sum(macro_f1_list) / (len(macro_f1_list) + 1e-9)

    recall_acc = total_right_num / (total_gt_num + 1e-9)
    preci_acc = total_right_num / (total_pred_num + 1e-9)
    hmean = 2 * recall_acc * preci_acc / (recall_acc + preci_acc + 1e-9)

    vbs_eval_result = {
        'macro_recall': final_macro_recall, 'macro_precision': final_macro_precision, 'macro_f1_score': final_macro_f1,
        'micro_recall': recall_acc, 'micro_precision': preci_acc, 'micro_f1_score': hmean
    }
    eval_result = vbs_eval_result if is_verbose else {'macro_f1_score': final_macro_f1, 'micro_f1_score': hmean}
    return eval_result


def build_messages(sample) -> list[dict]:
    image_url = f"data:image/jpeg;base64,{sample['image']}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["question"]},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


async def process_sample(sample, rate_limiter):
    messages = build_messages(sample)
    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            response = await asyncio.to_thread(invoke_interfaze, messages)
            if isinstance(response, dict) and not response:
                raise RuntimeError("Empty response from API")
            text = response.choices[0].message.content
            return {
                "image_name": sample["image_name"],
                "prediction": text,
                "answer": sample["answer"],
                "l2-category": sample["l2-category"],
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"Failed after {MAX_RETRIES} attempts for {sample['image_name']}: {e}")
                return {
                    "image_name": sample["image_name"],
                    "prediction": "",
                    "answer": sample["answer"],
                    "l2-category": sample["l2-category"],
                }


def get_eval_config(language: str):
    is_word_level = language not in CHAR_LEVEL_LANGUAGES
    is_lower = True
    is_alphanum_only = False
    return is_word_level, is_lower, is_alphanum_only


def evaluate_results(results: list[dict]) -> dict:
    # Group by language
    by_language = {}
    for r in results:
        lang = r["l2-category"]
        by_language.setdefault(lang, []).append(r)

    # Tokenize all samples with per-language config and build overall dicts
    all_pred_tokens = {}
    all_gt_tokens = {}
    per_lang_metrics = {}

    for lang, samples in by_language.items():
        is_word_level, is_lower, is_alphanum_only = get_eval_config(lang)
        pred_info = {}
        gt_info = {}
        for s in samples:
            name = s["image_name"]
            pred_info[name] = text_normalize_and_tokenize(
                str(s["prediction"]).strip(), is_word_level, is_lower, is_alphanum_only
            )
            gt_info[name] = text_normalize_and_tokenize(
                str(s["answer"]).strip(), is_word_level, is_lower, is_alphanum_only
            )
        all_pred_tokens.update(pred_info)
        all_gt_tokens.update(gt_info)
        per_lang_metrics[lang] = calculate_metrics(pred_info, gt_info, is_verbose=True)

    overall_metrics = calculate_metrics(all_pred_tokens, all_gt_tokens, is_verbose=True)
    return {"overall": overall_metrics, "by_language": per_lang_metrics}


def print_summary(metrics: dict, num_samples: int, num_failures: int):
    print("CC-OCR Benchmark Results (Interfaze)")
    print(f"Samples: {num_samples} | Failures: {num_failures}")
    print(f"{'Language':<15} {'Macro F1':>10} {'Micro F1':>10} {'Precision':>10} {'Recall':>10}")
    for lang in sorted(metrics["by_language"]):
        m = metrics["by_language"][lang]
        print(f"{lang:<15} {m['macro_f1_score']:>10.4f} {m['micro_f1_score']:>10.4f} {m['micro_precision']:>10.4f} {m['micro_recall']:>10.4f}")
    o = metrics["overall"]
    print(f"{'OVERALL':<15} {o['macro_f1_score']:>10.4f} {o['micro_f1_score']:>10.4f} {o['micro_precision']:>10.4f} {o['micro_recall']:>10.4f}")


async def run_benchmark():
    print("Loading CC-OCR dataset (multi_lan_ocr)...")
    dataset = load_dataset("wulipc/CC-OCR", "multi_lan_ocr")
    test_data = dataset["test"]
    print(f"Loaded {len(test_data)} samples")

    rate_limiter = RateLimiter(RATE_LIMIT)
    tasks = [process_sample(sample, rate_limiter) for sample in test_data]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing samples")

    num_failures = sum(1 for r in results if r["prediction"] == "")
    metrics = evaluate_results(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cc_ocr_multilingual_responses.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    output = {
        **metrics,
        "num_samples": len(results),
        "num_failures": num_failures,
        "model": "interfaze-beta",
    }
    with open(RESULTS_DIR / "cc_ocr_multilingual_metrics.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_summary(metrics, len(results), num_failures)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    asyncio.run(run_benchmark())
