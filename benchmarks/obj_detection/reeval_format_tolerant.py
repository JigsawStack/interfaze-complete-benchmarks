"""
Format-tolerant re-evaluation of RefCOCO grounding runs.

Rationale: under the canonical eval prompt, different models emit boxes in
different conventions — xyxy vs yxyx ordering, 0-1.0 floats / 0-1000
normalized / raw pixels. The sample-by-sample correct-vs-wrong split under
a single-interpretation parser conflates two very different things:

  1. Did the model find the right region? (grounding capability — what we
     actually want to measure)
  2. Did the model emit it in the format we happened to parse? (format
     compliance — a separate, prompt-dependent concern)

This script re-scores existing `*_responses.jsonl` files by, for each
sample, trying every reasonable interpretation of every 4-tuple in the
response and keeping the one whose IoU with GT is highest. That isolates
(1) by taking (2) out of the equation — applied uniformly to every model
so the comparison stays apples-to-apples.

This is oracle parsing — it uses GT to disambiguate, so the resulting
numbers are *upper bounds* on production performance. Single-interpretation
numbers stay in the originals.

Usage:
    uv run -m benchmarks.obj_detection.reeval_format_tolerant
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.obj_detection.refcoco import (  # noqa: E402
    compute_iou, compute_metrics,
)

RESULTS_DIR = PROJECT_ROOT / "results"
TUPLE_PATTERN = re.compile(
    r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)
PAREN_PAIRS = re.compile(
    r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*[^()]*?\s*'
    r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
)
TLBR_JSON = re.compile(
    r'"top_left"\s*:\s*\{[^}]*?"x"\s*:\s*(-?\d+(?:\.\d+)?)[^}]*?"y"\s*:\s*'
    r'(-?\d+(?:\.\d+)?)[^}]*?\}[^}]*?"bottom_right"\s*:\s*\{[^}]*?"x"\s*:\s*'
    r'(-?\d+(?:\.\d+)?)[^}]*?"y"\s*:\s*(-?\d+(?:\.\d+)?)',
    re.DOTALL,
)


def extract_tuples(text: str) -> list[tuple[float, float, float, float]]:
    """Pull every plausible 4-tuple out of the response — bracket lists,
    parenthesized (x,y) pairs, and JSON top_left/bottom_right blocks.
    Order is preserved (so callers can prefer the last match if needed)."""
    out = []
    for m in TUPLE_PATTERN.finditer(text):
        out.append(tuple(float(g) for g in m.groups()))
    for m in TLBR_JSON.finditer(text):
        # already xyxy in pixel space — emit as-is.
        out.append(tuple(float(g) for g in m.groups()))
    for m in PAREN_PAIRS.finditer(text):
        out.append(tuple(float(g) for g in m.groups()))
    return out


def all_interpretations(nums, w, h):
    """Yield (label, [x1,y1,x2,y2]) for every plausible interpretation
    of a 4-tuple under the canonical RefCOCO eval — covers xyxy/yxyx
    order × {raw pixel, 0-1000 normalized, 0-1.0 float} scale."""
    n0, n1, n2, n3 = nums
    mx = max(abs(c) for c in nums)
    yield "pixel-xyxy", [n0, n1, n2, n3]
    yield "pixel-yxyx", [n1, n0, n3, n2]
    if mx <= 1000:
        yield "norm1000-xyxy", [n0 * w / 1000, n1 * h / 1000, n2 * w / 1000, n3 * h / 1000]
        yield "norm1000-yxyx", [n1 * w / 1000, n0 * h / 1000, n3 * w / 1000, n2 * h / 1000]
    if mx <= 1.0:
        yield "norm1-xyxy", [n0 * w, n1 * h, n2 * w, n3 * h]
        yield "norm1-yxyx", [n1 * w, n0 * h, n3 * w, n2 * h]


def best_box(response: str, w: int, h: int, gt: list[float]):
    """Return (best_box_xyxy, best_iou, label) — picks the interpretation
    of any 4-tuple in the response that maximizes IoU vs GT."""
    best_iou = 0.0
    best_box = None
    best_label = None
    for nums in extract_tuples(response):
        for label, box in all_interpretations(nums, w, h):
            v = compute_iou(box, gt)
            if v > best_iou:
                best_iou, best_box, best_label = v, box, label
    return best_box, best_iou, best_label


def reeval_file(path: Path) -> tuple[list[dict], dict]:
    records = []
    label_counts = {}
    n_changed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gt = r["gt_bbox_xyxy"]
            w = r["image_width"]
            h = r["image_height"]
            box, iou, label = best_box(r["response"], w, h, gt)
            new_correct = box is not None and iou >= 0.5
            if r.get("correct") != new_correct:
                n_changed += 1
            r["pred_bbox_xyxy"] = box
            r["iou"] = iou
            r["correct"] = new_correct
            r["interpretation"] = label
            records.append(r)
            if box is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
    return records, {"n_changed": n_changed, "label_counts": label_counts}


def write_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    targets = [
        ("interfaze", "refcoco_testA"),
        ("gpt-5.5",   "gpt55_refcoco_testA"),
        ("kimi-k2.6", "kimi_k26_refcoco_testA"),
    ]
    print(f"{'model':<14} {'orig_acc':>10} {'oracle_acc':>11} {'mean_iou':>10} {'changed':>9}")
    print("-" * 60)
    for label, tag in targets:
        src = RESULTS_DIR / f"{tag}_responses.jsonl"
        if not src.exists():
            print(f"{label:<14} (missing {src.name})")
            continue
        # Original score (before re-eval)
        with open(src) as f:
            orig_records = [json.loads(l) for l in f if l.strip()]
        orig_correct = sum(1 for r in orig_records if r.get("correct"))
        orig_total = len(orig_records)

        records, info = reeval_file(src)
        out_responses = RESULTS_DIR / f"{tag}_oracle_responses.jsonl"
        out_metrics = RESULTS_DIR / f"{tag}_oracle_metrics.json"
        write_records(records, out_responses)

        metrics = compute_metrics(records)
        metrics["interpretation_counts"] = info["label_counts"]
        metrics["records_changed_from_orig"] = info["n_changed"]
        metrics["model"] = label
        with open(out_metrics, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"{label:<14} {orig_correct/orig_total:>10.4f} "
              f"{metrics['accuracy']:>11.4f} {metrics['mean_iou']:>10.4f} "
              f"{info['n_changed']:>9d}")
        print(f"  interpretations used: {dict(sorted(info['label_counts'].items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
