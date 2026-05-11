"""
Audit the screenspot-pro parser against existing response JSONLs.

For each (provider/model) it reports:
  - parser miss rate (pred_point is None)
  - correct rate
  - 5 sample raw responses where parser returned None
  - 5 sample raw responses where parser returned a point but answer was wrong
    (helps spot coord-system surprises like "1280 1024" vs "[1280, 1024]")

Usage: uv run -m benchmarks.screenspot_pro.audit_parser
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.screenspot_pro.screenspot_pro_multi import parse_click  # noqa: E402

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


def truncate(s: str, n: int = 240) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"…(+{len(s)-n} chars)"


def audit_one(jsonl: Path, model_lower: str) -> dict:
    none_examples: list[dict] = []
    wrong_examples: list[dict] = []
    n, none_count, wrong_count, correct_count = 0, 0, 0, 0
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        if rec.get("correct"):
            correct_count += 1
            continue
        pp = rec.get("pred_point")
        if pp is None:
            none_count += 1
            if len(none_examples) < 5:
                none_examples.append(rec)
        else:
            wrong_count += 1
            if len(wrong_examples) < 5:
                wrong_examples.append(rec)
    return {
        "n": n,
        "correct": correct_count,
        "none": none_count,
        "wrong": wrong_count,
        "none_examples": none_examples,
        "wrong_examples": wrong_examples,
    }


def reparse_and_recheck(rec: dict, model_lower: str):
    """Run the current parser on the recorded response — sanity check that
    audit and bench agree on what the parser sees."""
    pp = parse_click(
        rec.get("response", ""),
        rec.get("image_width_sent"),
        rec.get("image_height_sent"),
        model_lower,
    )
    return pp


def main() -> None:
    overall_none = 0
    overall_n = 0
    for provider, slug in MODELS:
        jsonl = RESULTS / f"screenspotpro_{slug}_reasoningoff_responses.jsonl"
        model_lower = slug.split("_", 1)[1]
        info = audit_one(jsonl, model_lower)
        overall_n += info["n"]
        overall_none += info["none"]
        print()
        print("=" * 80)
        print(f"{provider} / {slug}")
        print(f"  n={info['n']}  correct={info['correct']}  parser_none={info['none']}  "
              f"parsed_but_wrong={info['wrong']}")

        if info["none_examples"]:
            print(f"\n  -- {len(info['none_examples'])} samples where parser returned None --")
            for rec in info["none_examples"]:
                rerun = reparse_and_recheck(rec, model_lower)
                print(f"    id={rec['id']}  reparse_now={rerun}  resp={truncate(rec.get('response',''))}")

        if info["wrong_examples"]:
            print(f"\n  -- {len(info['wrong_examples'])} samples parsed-but-wrong --")
            for rec in info["wrong_examples"]:
                gt = [round(x) for x in (rec.get("gt_bbox_sent") or [])]
                pp = [round(x, 1) for x in (rec.get("pred_point") or [])]
                print(f"    id={rec['id']}  pred={pp}  gt={gt}  resp={truncate(rec.get('response',''))}")

    print()
    print("=" * 80)
    print(f"OVERALL: parser-none = {overall_none} / {overall_n} "
          f"({100*overall_none/max(overall_n,1):.1f}%)")


if __name__ == "__main__":
    main()
