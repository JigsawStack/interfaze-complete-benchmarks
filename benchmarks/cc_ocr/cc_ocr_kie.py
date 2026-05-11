"""
KIE (Key Information Extraction) benchmark for CC-OCR.
Evaluation based on Donut (Copyright (c) 2022-present NAVER Corp., MIT License).

Metrics:
- F1 score: field-level micro-averaged F1
- Accuracy: normalized tree edit distance (nTED) based accuracy
"""
import sys
import json
import re
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Union

import zss
from zss import Node
from nltk import edit_distance
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from src.commons import invoke_interfaze

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
RATE_LIMIT = 25
MAX_RETRIES = 3


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


# --- JSON post-processing ---

def post_process_to_json(text: str) -> dict | None:
    try:
        if "```json" in text:
            if "```" not in text.split("```json", 1)[1]:
                text += "```"
            match = re.search(r'```json(.*?)```', text, re.DOTALL)
            json_str = match.group(1).strip().replace("\n", "")
        else:
            json_str = text.strip().replace("\n", "")
        return json.loads(json_str)
    except Exception:
        return None


# --- Text normalization ---

def fullwidth_to_halfwidth(text: str) -> str:
    result = ''
    for char in text:
        cp = ord(char)
        if cp == 0x3000:
            cp = 0x0020
        elif 0xFF01 <= cp <= 0xFF5E:
            cp -= 0xFEE0
        result += chr(cp)
    return result.replace("\u3001", ",")


def remove_unnecessary_spaces(text: str) -> str:
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[a-zA-Z0-9])', '', text)
    text = re.sub(r'(?<=[a-zA-Z0-9])\s+(?=[\u4e00-\u9fff])', '', text)
    text = re.sub(r'(?<![0-9])\s*([,.!?:;])\s*', r'\1 ', text)
    text = re.sub(r'(?<=[0-9])(?=[a-zA-Z])', ' ', text)
    text = re.sub(r'(?<=[a-zA-Z])(?=[0-9])', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_text(text: str) -> str:
    return remove_unnecessary_spaces(fullwidth_to_halfwidth(str(text)))


def normalize_values_of_nested_dict(d, normalize_func):
    if isinstance(d, dict):
        return {k: normalize_values_of_nested_dict(v, normalize_func) for k, v in d.items()}
    elif isinstance(d, list):
        return [normalize_values_of_nested_dict(x, normalize_func) if isinstance(x, dict) else x for x in d]
    elif isinstance(d, str):
        return normalize_func(d)
    return d


# --- Dict normalization & flattening ---

def normalize_dict(data: Union[dict, list, Any]):
    if isinstance(data, dict):
        new_data = {}
        for key in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = normalize_dict(data[key])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new_data[key] = value
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            new_data = [normalize_dict(item) for item in data if normalize_dict(item)]
        else:
            new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
    else:
        new_data = [str(data).strip()]
    return new_data


def flatten(data: dict) -> list:
    flatten_data = []

    def _flatten(value, key=""):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                _flatten(child_value, f"{key}.{child_key}" if key else child_key)
        elif isinstance(value, list):
            for value_item in value:
                _flatten(value_item, key)
        else:
            flatten_data.append((key, value))

    _flatten(data)
    return flatten_data


# --- F1 score ---

def cal_f1_all(preds: dict, answers: dict):
    total_tp, total_fn_or_fp = 0, 0
    for file_name, answer in answers.items():
        pred = preds.get(file_name, {})
        pred_flat = flatten(normalize_dict(pred))
        answer_flat = flatten(normalize_dict(answer))
        for field in pred_flat:
            if field in answer_flat:
                total_tp += 1
                answer_flat.remove(field)
            else:
                total_fn_or_fp += 1
        total_fn_or_fp += len(answer_flat)

    f1 = total_tp / (total_tp + total_fn_or_fp / 2 + 1e-6)
    return f1


# --- Tree edit distance accuracy ---

def update_cost(node1: Node, node2: Node):
    label1, label2 = node1.label, node2.label
    label1_leaf = "<leaf>" in label1
    label2_leaf = "<leaf>" in label2
    if label1_leaf and label2_leaf:
        return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
    elif not label1_leaf and label2_leaf:
        return 1 + len(label2.replace("<leaf>", ""))
    elif label1_leaf and not label2_leaf:
        return 1 + len(label1.replace("<leaf>", ""))
    return int(label1 != label2)


def insert_and_remove_cost(node: Node):
    label = node.label
    if "<leaf>" in label:
        return len(label.replace("<leaf>", ""))
    return 1


def construct_tree_from_dict(data: Union[dict, list], node_name: str = None):
    if node_name is None:
        node_name = "<root>"
    node = Node(node_name)
    if isinstance(data, dict):
        for key, value in data.items():
            node.addkid(construct_tree_from_dict(value, key))
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            for item in data:
                node.addkid(construct_tree_from_dict(item, "<subtree>"))
        else:
            for item in data:
                node.addkid(Node(f"<leaf>{item}"))
    else:
        raise ValueError(f"Unexpected data type: {type(data)}, node_name: {node_name}")
    return node


def cal_acc(pred: dict, answer: dict) -> float:
    pred_tree = construct_tree_from_dict(normalize_dict(pred))
    answer_tree = construct_tree_from_dict(normalize_dict(answer))
    empty_tree = construct_tree_from_dict(normalize_dict({}))
    dist_args = dict(
        get_children=zss.Node.get_children,
        insert_cost=insert_and_remove_cost,
        remove_cost=insert_and_remove_cost,
        update_cost=update_cost,
        return_operations=False,
    )
    dist = zss.distance(pred_tree, answer_tree, **dist_args)
    max_dist = zss.distance(empty_tree, answer_tree, **dist_args)
    return max(0, 1 - dist / max_dist)


def _cal_acc_pair(args):
    pred, answer = args
    return cal_acc(pred, answer)


def cal_acc_all(pred_info: dict, answer_info: dict) -> float:
    pairs = [(pred_info.get(name, {}), answer) for name, answer in answer_info.items()]
    with ProcessPoolExecutor() as executor:
        acc_values = list(executor.map(_cal_acc_pair, pairs))
    return sum(acc_values) / (len(acc_values) + 1e-6)


# --- API + benchmark pipeline ---

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
                "split": sample["split"],
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
                    "split": sample["split"],
                }


def evaluate_results(results: list[dict]) -> dict:
    # Build pred/gt dicts: image_name -> parsed JSON
    pred_info = {}
    gt_info = {}
    for r in results:
        name = r["image_name"]
        # Parse prediction JSON
        parsed = post_process_to_json(r["prediction"]) if r["prediction"] else None
        pred_info[name] = parsed if parsed is not None else {}
        # Parse ground truth (may be string JSON)
        gt = r["answer"]
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except (json.JSONDecodeError, TypeError):
                gt = {}
        gt_info[name] = gt

    # Normalize text values
    pred_info = normalize_values_of_nested_dict(pred_info, normalize_text)
    gt_info = normalize_values_of_nested_dict(gt_info, normalize_text)

    # Overall metrics
    f1 = cal_f1_all(pred_info, gt_info)
    acc = cal_acc_all(pred_info, gt_info)

    # Per-split metrics
    by_split = {}
    for r in results:
        by_split.setdefault(r["split"], []).append(r["image_name"])

    per_split = {}
    for sp, names in by_split.items():
        sp_pred = {n: pred_info[n] for n in names}
        sp_gt = {n: gt_info[n] for n in names}
        per_split[sp] = {
            "f1_score": cal_f1_all(sp_pred, sp_gt),
            "accuracy": cal_acc_all(sp_pred, sp_gt),
            "count": len(names),
        }

    # Per l2-category
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["l2-category"], []).append(r["image_name"])

    per_category = {}
    for cat, names in by_cat.items():
        cat_pred = {n: pred_info[n] for n in names}
        cat_gt = {n: gt_info[n] for n in names}
        per_category[cat] = {
            "f1_score": cal_f1_all(cat_pred, cat_gt),
            "accuracy": cal_acc_all(cat_pred, cat_gt),
            "count": len(names),
        }

    return {
        "overall": {"f1_score": f1, "accuracy": acc},
        "by_category": per_category,
        "by_split": per_split,
    }


def print_summary(metrics: dict, num_samples: int, num_failures: int):
    print("CC-OCR KIE Benchmark Results (Interfaze)")
    print(f"Samples: {num_samples} | Failures: {num_failures}")
    print(f"\n{'Category':<25} {'F1':>10} {'Accuracy':>10} {'Count':>8}")
    print("-" * 55)
    for cat in sorted(metrics["by_category"]):
        m = metrics["by_category"][cat]
        print(f"{cat:<25} {m['f1_score']:>10.4f} {m['accuracy']:>10.4f} {m['count']:>8}")
    print("-" * 55)
    o = metrics["overall"]
    print(f"{'OVERALL':<25} {o['f1_score']:>10.4f} {o['accuracy']:>10.4f} {num_samples:>8}")
    print(f"\n{'Split':<25} {'F1':>10} {'Accuracy':>10} {'Count':>8}")
    print("-" * 55)
    for sp in sorted(metrics["by_split"]):
        m = metrics["by_split"][sp]
        print(f"{sp:<25} {m['f1_score']:>10.4f} {m['accuracy']:>10.4f} {m['count']:>8}")


async def run_benchmark():
    print("Loading CC-OCR dataset (kie)...")
    dataset = load_dataset("wulipc/CC-OCR", "kie")
    test_data = dataset["test"]
    print(f"Loaded {len(test_data)} samples")

    rate_limiter = RateLimiter(RATE_LIMIT)
    tasks = [process_sample(sample, rate_limiter) for sample in test_data]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing samples")

    num_failures = sum(1 for r in results if r["prediction"] == "")
    metrics = evaluate_results(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cc_ocr_kie_responses.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    output = {
        **metrics,
        "num_samples": len(results),
        "num_failures": num_failures,
        "model": "interfaze-beta",
    }
    with open(RESULTS_DIR / "cc_ocr_kie_metrics.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_summary(metrics, len(results), num_failures)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    asyncio.run(run_benchmark())
