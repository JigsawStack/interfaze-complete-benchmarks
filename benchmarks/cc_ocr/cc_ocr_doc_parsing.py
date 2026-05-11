import sys
import json
import re
import asyncio
import unicodedata
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nltk
from lxml import html, etree
from apted import APTED, Config
from apted.helpers import Tree
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from src.commons import invoke_interfaze

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
RATE_LIMIT = 25
MAX_RETRIES = 3

# LaTeX commands to strip from doc predictions
LATEX_STRIP_PATTERNS = [
    r'\\documentclass\{.*?\}',
    r'\\usepackage\[.*?\]\{.*?\}',
    r'\\usepackage\{.*?\}',
    r'\\geometry\{.*?\}',
    r'\\begin\{document\}',
    r'\\end\{document\}',
    r'\\noindent',
]


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


# --- Utility ---

def convert_to_halfwidth(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# --- Table evaluation (TEDS) ---

def extract_and_clean_tables(text):
    if '</table>' not in text:
        text += '</table>'
    tables = re.findall(r'<table.*?>.*?</table>', text, re.DOTALL)
    clean_tables = []
    for table in tables:
        table_content = re.sub(r'<table.*?>', '<table>', table)
        table_content = re.sub(r'>\s+<', '><', table_content)
        table_content = re.sub(
            r'>(.*?)<',
            lambda m: '>' + m.group(1).replace('\n', '').replace(' ', '') + '<',
            table_content, flags=re.DOTALL,
        )
        table_content = table_content.replace('\n', '').strip()
        clean_tables.append(table_content)
    return ''.join(clean_tables)


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag, self.colspan, self.rowspan, self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    def rename(self, node1, node2):
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return nltk.edit_distance(node1.content, node2.content) / max(
                    len(node1.content), len(node2.content)
                )
        return 0.0


class TEDS:
    def __init__(self, structure_only=False):
        self.structure_only = structure_only
        self.__tokens__ = []

    def tokenize(self, node):
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell, *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            n_nodes = max(len(pred.xpath(".//*")), len(true.xpath(".//*")))
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        return 0.0


# --- Per-category evaluation functions ---

def eval_doc_sample(gt: str, pred: str) -> float:
    for pattern in LATEX_STRIP_PATTERNS:
        pred = re.sub(pattern, '', pred)
    try:
        pred = re.search(r'```latex(.+?)```', pred, re.DOTALL).group(1)
    except AttributeError:
        if '```latex' in pred:
            pred = pred.split('```latex')[1]
    pred = pred.replace(' ', '').replace('\n', '')
    gt = gt.replace(' ', '').replace('\n', '')
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return 1 - nltk.edit_distance(pred, gt) / max(len(pred), len(gt))


def eval_table_sample(gt: str, pred: str) -> float:
    teds = TEDS(structure_only=False)
    try:
        pred = re.search(r'```html(.+?)```', pred, re.DOTALL).group(1)
    except AttributeError:
        if '```html' in pred:
            pred = pred.split('```html')[1]
    pred = convert_to_halfwidth(extract_and_clean_tables(pred))
    gt = convert_to_halfwidth(extract_and_clean_tables(gt))
    return teds.evaluate(
        '<html><body>{}</body></html>'.format(pred),
        '<html><body>{}</body></html>'.format(gt),
    )


def eval_formula_sample(gt: str, pred: str, op_name: str = "formula") -> float:
    if op_name == "formula":
        pred = pred.replace("\n", " ").replace("```latex", "").replace("```", "").replace("\t", " ").replace(" ", "")
        gt = gt.replace(" ", "")
    elif op_name == "molecular":
        pred = pred.replace("\n", "").replace(" ", "").replace("<smiles>", "").replace("</smiles>", "")
        gt = gt.replace(" ", "")
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return 1 - nltk.edit_distance(pred, gt) / max(len(pred), len(gt))


def eval_formula_sample_wrapper(gt, pred):
    return eval_formula_sample(gt, pred, "formula")


def eval_molecular_sample_wrapper(gt, pred):
    return eval_formula_sample(gt, pred, "molecular")


EVAL_FUNCS = {
    "doc": eval_doc_sample,
    "table": eval_table_sample,
    "formula": eval_formula_sample_wrapper,
    "molecular": eval_molecular_sample_wrapper,
}


def _eval_worker(args):
    cat, gt, pred = args
    return EVAL_FUNCS[cat](gt, pred)


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
    # Parallel evaluation of all samples
    work_items = [(r["l2-category"], r["answer"], r["prediction"]) for r in results]
    with ProcessPoolExecutor() as executor:
        all_scores = list(executor.map(_eval_worker, work_items))

    # Per-category scores
    by_category = {}
    for r, score in zip(results, all_scores):
        cat = r["l2-category"]
        by_category.setdefault(cat, []).append(score)

    per_category_scores = {}
    for cat, scores in by_category.items():
        per_category_scores[cat] = {"score": sum(scores) / len(scores) if scores else 0.0, "count": len(scores)}

    # Per-split scores
    by_split = {}
    for r, score in zip(results, all_scores):
        sp = r["split"]
        by_split.setdefault(sp, []).append(score)

    per_split_scores = {}
    for sp, scores in by_split.items():
        per_split_scores[sp] = {"score": sum(scores) / len(scores) if scores else 0.0, "count": len(scores)}

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "overall": {"score": overall},
        "by_category": per_category_scores,
        "by_split": per_split_scores,
    }


def print_summary(metrics: dict, num_samples: int, num_failures: int):
    print("CC-OCR Doc Parsing Benchmark Results (Interfaze)")
    print(f"Samples: {num_samples} | Failures: {num_failures}")
    print(f"\n{'Category':<20} {'Score':>10} {'Count':>8}")
    print("-" * 40)
    for cat in sorted(metrics["by_category"]):
        m = metrics["by_category"][cat]
        print(f"{cat:<20} {m['score']:>10.4f} {m['count']:>8}")
    print("-" * 40)
    print(f"{'OVERALL':<20} {metrics['overall']['score']:>10.4f} {num_samples:>8}")
    print(f"\n{'Split':<25} {'Score':>10} {'Count':>8}")
    print("-" * 45)
    for sp in sorted(metrics["by_split"]):
        m = metrics["by_split"][sp]
        print(f"{sp:<25} {m['score']:>10.4f} {m['count']:>8}")


async def run_benchmark():
    print("Loading CC-OCR dataset (doc_parsing)...")
    dataset = load_dataset("wulipc/CC-OCR", "doc_parsing")
    test_data = dataset["test"]
    print(f"Loaded {len(test_data)} samples")

    rate_limiter = RateLimiter(RATE_LIMIT)
    tasks = [process_sample(sample, rate_limiter) for sample in test_data]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing samples")

    num_failures = sum(1 for r in results if r["prediction"] == "")
    metrics = evaluate_results(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cc_ocr_doc_parsing_responses.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    output = {
        **metrics,
        "num_samples": len(results),
        "num_failures": num_failures,
        "model": "interfaze-beta",
    }
    with open(RESULTS_DIR / "cc_ocr_doc_parsing_metrics.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_summary(metrics, len(results), num_failures)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    asyncio.run(run_benchmark())
