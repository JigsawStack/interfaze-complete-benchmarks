"""
MMMLU (Multilingual MMLU) benchmark for Interfaze.

OpenAI's professional-translation version of the MMLU test set across 14
languages — the same benchmark Gemini 3 Pro reports as "Multilingual Q&A"
(91.8% in their Nov 2025 model card). Each language subset is the full MMLU
test split (~14,042 four-choice questions across 57 subjects) translated by
human translators.

Dataset: https://huggingface.co/datasets/openai/MMMLU

Languages (14):
    AR_XY  Arabic
    BN_BD  Bengali
    DE_DE  German
    ES_LA  Spanish (Latin America)
    FR_FR  French
    HI_IN  Hindi
    ID_ID  Indonesian
    IT_IT  Italian
    JA_JP  Japanese
    KO_KR  Korean
    PT_BR  Brazilian Portuguese
    SW_KE  Swahili
    YO_NG  Yoruba
    ZH_CN  Simplified Chinese

Methodology (matches Gemini 3 Pro model card):
    pass@1 — single attempt, no majority voting, no parallel test-time compute.
    Default sampling, single trial (large benchmark, no trial averaging).
    Headline number = macro-average accuracy across the 14 languages.

Metric: exact-match accuracy on the predicted answer letter (A/B/C/D).
        Reported per-language, per-subject, and macro-averaged.

Usage:
    uv run -m benchmarks.mmmlu.mmmlu                         # full run, all 14 langs
    uv run -m benchmarks.mmmlu.mmmlu --limit 50              # smoke test: 50 per lang
    uv run -m benchmarks.mmmlu.mmmlu --languages DE_DE FR_FR # subset of langs
    uv run -m benchmarks.mmmlu.mmmlu --evaluate-only         # rescore existing preds
"""

import sys
import os
import re
import json
import time
import asyncio
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.commons import invoke_interfaze  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_ID = "openai/MMMLU"
SPLIT = "test"

LANGUAGES = [
    "AR_XY", "BN_BD", "DE_DE", "ES_LA", "FR_FR", "HI_IN", "ID_ID",
    "IT_IT", "JA_JP", "KO_KR", "PT_BR", "SW_KE", "YO_NG", "ZH_CN",
]

REASONING_EFFORT = None   # off — Gemini 3 Pro reports MMMLU as a non-thinking eval
TEMPERATURE = 0.0         # deterministic; repo convention (Gemini uses default sampling)
RATE_LIMIT = 25
MAX_RETRIES = 3

# English instruction is intentional and standard for MMMLU evaluation: the
# question + options are in the target language, but the meta-instruction
# ("answer with a single letter") is held constant so the parser can rely on
# Latin A-D output regardless of the language of the question.
PROMPT_TEMPLATE = (
    "The following is a multiple choice question. Respond with only a single "
    "letter (A, B, C, or D) corresponding to the correct answer. Do not "
    "explain your reasoning.\n\n"
    "Question: {question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n\n"
    "Answer:"
)


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
            if rec.get("response") is not None:
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


_LETTER_RE = re.compile(r"\b([ABCD])\b")
_FIRST_LETTER_RE = re.compile(r"[ABCD]")


def parse_answer(text: str) -> str | None:
    """Extract a single A/B/C/D letter from the model's response.

    Try a word-boundary match first (handles "A", "A.", "(A)", "Answer: A").
    Fall back to the first standalone-looking letter; return None if nothing matches."""
    if not text:
        return None
    stripped = text.strip()
    # Direct hit: response is just the letter.
    if len(stripped) == 1 and stripped.upper() in "ABCD":
        return stripped.upper()
    m = _LETTER_RE.search(stripped.upper())
    if m:
        return m.group(1)
    m = _FIRST_LETTER_RE.search(stripped.upper())
    if m:
        return m.group(0)
    return None


def build_sample(row: dict, language: str) -> dict:
    idx = row.get("Unnamed: 0")
    return {
        "id": f"{language}:{idx}",
        "language": language,
        "row_index": idx,
        "subject": row["Subject"],
        "question": row["Question"],
        "a": row["A"],
        "b": row["B"],
        "c": row["C"],
        "d": row["D"],
        "answer": str(row["Answer"]).strip().upper(),
    }


async def process_sample(sample: dict, rate_limiter: RateLimiter,
                         writer: JsonlWriter, progress: dict) -> dict | None:
    last_error: str | None = None

    prompt = PROMPT_TEMPLATE.format(
        question=sample["question"],
        a=sample["a"], b=sample["b"], c=sample["c"], d=sample["d"],
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        start = time.perf_counter()
        try:
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

            predicted = parse_answer(content)
            correct = predicted == sample["answer"]

            record = {
                "id": sample["id"],
                "language": sample["language"],
                "row_index": sample["row_index"],
                "subject": sample["subject"],
                "answer": sample["answer"],
                "prediction": predicted,
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
            if predicted is None:
                progress["unparseable"] += 1
            tqdm.write(
                f"[{progress['done']}/{progress['total']}] "
                f"{sample['language']} subj={sample['subject'][:18]:18} "
                f"gold={sample['answer']} pred={predicted or '?'} "
                f"{'OK' if correct else 'X '} latency={latency_ms}ms"
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
    if not results:
        return {}

    by_lang: dict[str, list[dict]] = defaultdict(list)
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)
        by_subject[r["subject"]].append(r)

    per_language: dict[str, dict] = {}
    for lang, rows in by_lang.items():
        n = len(rows)
        n_correct = sum(1 for r in rows if r.get("correct"))
        n_unparseable = sum(1 for r in rows if r.get("prediction") is None)
        per_language[lang] = {
            "n": n,
            "accuracy": n_correct / n if n else 0.0,
            "unparseable": n_unparseable,
        }

    per_subject: dict[str, dict] = {}
    for subj, rows in by_subject.items():
        n = len(rows)
        n_correct = sum(1 for r in rows if r.get("correct"))
        per_subject[subj] = {
            "n": n,
            "accuracy": n_correct / n if n else 0.0,
        }

    # Macro-average across languages — this is the headline MMMLU number.
    lang_accs = [v["accuracy"] for v in per_language.values()]
    macro_accuracy = sum(lang_accs) / len(lang_accs) if lang_accs else 0.0

    # Micro-accuracy = pooled across all samples (depends on per-lang counts).
    n_total = sum(v["n"] for v in per_language.values())
    n_total_correct = sum(int(v["accuracy"] * v["n"]) for v in per_language.values())
    micro_accuracy = n_total_correct / n_total if n_total else 0.0

    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int)]
    latency_stats = {}
    if latencies:
        lats = sorted(latencies)
        n = len(lats)
        latency_stats = {
            "count": n,
            "mean_ms": sum(lats) / n,
            "p50_ms": lats[n // 2],
            "p90_ms": lats[min(n - 1, int(n * 0.9))],
            "p99_ms": lats[min(n - 1, int(n * 0.99))],
            "max_ms": lats[-1],
        }

    return {
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
        "num_samples": n_total,
        "per_language": per_language,
        "per_subject": per_subject,
        "latency": latency_stats,
    }


def print_summary(metrics: dict):
    print(f"\n{'=' * 68}")
    print(f"MMMLU Results (Interfaze, reasoning={REASONING_EFFORT}, temp={TEMPERATURE})")
    print(f"{'=' * 68}")
    print(f"Samples              : {metrics['num_samples']}")
    print(f"Macro-avg accuracy   : {metrics['macro_accuracy']:.4f}  ← headline (Gemini 3 Pro: 0.918)")
    print(f"Micro-avg accuracy   : {metrics['micro_accuracy']:.4f}")
    print()
    print("Per-language accuracy:")
    print(f"  {'lang':6} {'n':>6} {'acc':>8} {'unparseable':>12}")
    for lang in sorted(metrics["per_language"].keys()):
        v = metrics["per_language"][lang]
        print(f"  {lang:6} {v['n']:>6} {v['accuracy']:>8.4f} {v['unparseable']:>12}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(f"\nLatency              : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
              f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms max={lat['max_ms']}ms")


def load_all_samples(languages: list[str], limit: int | None) -> list[dict]:
    """Load samples from each language subset, optionally capping per language."""
    all_samples: list[dict] = []
    for lang in languages:
        print(f"Loading {DATASET_ID}/{lang} (split={SPLIT})...")
        ds = load_dataset(DATASET_ID, lang, split=SPLIT)
        rows = [build_sample(dict(row), lang) for row in ds]
        if limit is not None:
            rows = rows[:limit]
        print(f"  -> {len(rows)} samples")
        all_samples.extend(rows)
    return all_samples


async def run_predictions(pred_path: Path, languages: list[str], limit: int | None):
    samples = load_all_samples(languages, limit)
    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    print(f"Total samples: {len(samples)}")
    print(f"Resume: {len(done_ids)} already completed, {len(pending)} remaining "
          f"(checkpoint: {pred_path})")
    if not pending:
        return

    writer = JsonlWriter(pred_path)
    rate_limiter = RateLimiter(RATE_LIMIT)
    progress = {
        "total": len(pending),
        "done": 0,
        "correct": 0,
        "unparseable": 0,
        "failed": 0,
    }

    tasks = [process_sample(s, rate_limiter, writer, progress) for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc="MMMLU")
    except Exception:
        traceback.print_exc()

    acc = progress["correct"] / progress["done"] if progress["done"] else 0.0
    print(
        f"\nRun finished: {progress['done']}/{progress['total']} answered "
        f"({progress['failed']} failed, {progress['unparseable']} unparseable). "
        f"Pooled accuracy on this run: {acc:.4f}"
    )


def run_evaluation(pred_path: Path, metrics_path: Path):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)

    # Re-derive `correct` and `prediction` if missing (e.g., re-scoring an old run).
    for r in results:
        if r.get("prediction") is None and r.get("response"):
            r["prediction"] = parse_answer(r["response"])
        if r.get("correct") is None and r.get("prediction") is not None:
            r["correct"] = r["prediction"] == r.get("answer")

    metrics = compute_metrics(results)
    print_summary(metrics)
    output = {
        **metrics,
        "dataset": DATASET_ID,
        "split": SPLIT,
        "languages": sorted({r["language"] for r in results}),
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "rate_limit": RATE_LIMIT,
        "model": "interfaze-beta",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="MMMLU benchmark for Interfaze")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--languages", nargs="+", default=LANGUAGES,
        choices=LANGUAGES,
        help="Subset of languages to run (default: all 14).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap samples per language (smoke test).",
    )
    args = parser.parse_args()

    pred_path = RESULTS_DIR / "mmmlu_responses.jsonl"
    metrics_path = RESULTS_DIR / "mmmlu_metrics.json"

    if args.evaluate_only:
        run_evaluation(pred_path, metrics_path)
    elif args.predict_only:
        asyncio.run(run_predictions(pred_path, args.languages, args.limit))
    else:
        asyncio.run(run_predictions(pred_path, args.languages, args.limit))
        run_evaluation(pred_path, metrics_path)


if __name__ == "__main__":
    main()
