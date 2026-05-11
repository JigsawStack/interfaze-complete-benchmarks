"""
Re-score existing screenspot-pro JSONLs with the current parser (no API calls).

Use this after changing parse_click() to validate that the fix doesn't regress
already-saved responses, and to compare old-vs-new accuracy for the same data.

Usage:
    uv run -m benchmarks.screenspot_pro.reeval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.screenspot_pro.screenspot_pro_multi import (  # noqa: E402
    parse_click_candidates, point_in_bbox,
)

RESULTS = PROJECT_ROOT / "results"

MODELS = [
    ("interfaze", "interfaze_interfaze-beta"),
    ("gemini",    "gemini_gemini-3-flash-preview"),
    ("gemini",    "gemini_gemini-2-5-pro"),
    ("openai",    "openai_gpt-5-4-mini"),
    ("anthropic", "anthropic_claude-sonnet-4-6"),
    ("openai",    "openai_gpt-5-4"),
    ("openai",    "openai_gpt-5-5"),
    ("gemini",    "gemini_gemini-3-1-pro-preview"),
]


def reeval_one(jsonl: Path, model_lower: str) -> dict:
    n, old_correct, new_correct, old_none, new_none = 0, 0, 0, 0, 0
    flipped_to_correct: list[dict] = []
    flipped_to_wrong: list[dict] = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        if rec.get("correct"):
            old_correct += 1
        if rec.get("pred_point") is None:
            old_none += 1
        cands = parse_click_candidates(
            rec.get("response", ""),
            rec.get("image_width_sent"),
            rec.get("image_height_sent"),
            model_lower,
        )
        gt = rec.get("gt_bbox_sent")
        winner = next((c for c in cands if point_in_bbox(c, gt)), None)
        new_c = winner is not None
        if not cands:
            new_none += 1
        if new_c:
            new_correct += 1
        if new_c and not rec.get("correct"):
            flipped_to_correct.append({
                "id": rec["id"], "old_pred": rec.get("pred_point"),
                "new_pred": list(winner) if winner else None,
                "gt": [round(x) for x in gt or []],
                "n_cands": len(cands),
            })
        if not new_c and rec.get("correct"):
            flipped_to_wrong.append({
                "id": rec["id"], "old_pred": rec.get("pred_point"),
                "new_pred": list(cands[0]) if cands else None,
                "gt": [round(x) for x in gt or []],
            })
    return {
        "n": n,
        "old_correct": old_correct, "new_correct": new_correct,
        "old_none": old_none, "new_none": new_none,
        "flipped_to_correct": flipped_to_correct,
        "flipped_to_wrong": flipped_to_wrong,
    }


def main() -> None:
    print(f"{'model':50} {'n':>4} {'old_acc':>8} {'new_acc':>8} {'Δ':>6} {'oldN':>5} {'newN':>5}")
    print("-" * 90)
    for provider, slug in MODELS:
        jsonl = RESULTS / f"screenspotpro_{slug}_reasoningoff_responses.jsonl"
        model_lower = slug.split("_", 1)[1]
        info = reeval_one(jsonl, model_lower)
        old_acc = info["old_correct"] / info["n"] if info["n"] else 0.0
        new_acc = info["new_correct"] / info["n"] if info["n"] else 0.0
        delta = (new_acc - old_acc) * 100
        print(f"{slug:50} {info['n']:>4} {old_acc:>7.1%} {new_acc:>7.1%} "
              f"{delta:>+5.1f} {info['old_none']:>5} {info['new_none']:>5}")
        for item in info["flipped_to_correct"][:3]:
            print(f"    +correct id={item['id']}  old_pred={item['old_pred']}  "
                  f"new_pred={item['new_pred']}  gt={item['gt']}")
        for item in info["flipped_to_wrong"][:3]:
            print(f"    -correct id={item['id']}  old_pred={item['old_pred']}  "
                  f"new_pred={item['new_pred']}  gt={item['gt']}")


if __name__ == "__main__":
    main()
