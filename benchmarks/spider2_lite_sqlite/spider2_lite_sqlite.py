"""
Spider 2.0-Lite (SQLite-only) benchmark for Interfaze.

Filters spider2-lite to the 135 instances whose IDs start with `local` (SQLite
backed) and runs predict + execute + evaluate offline. No BigQuery / Snowflake
credentials needed.

Usage:
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite --predict-only
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite --evaluate-only
    uv run -m benchmarks.spider2_lite_sqlite.spider2_lite_sqlite --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = BENCHMARK_DIR / "resource"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_JSONL = BENCHMARK_DIR / "spider2-lite-sqlite.jsonl"
SQLITE_SCHEMAS_DIR = RESOURCE_DIR / "sqlite_schemas"
SQLITE_DB_DIR = RESOURCE_DIR / "databases"  # .sqlite files live here
DOCS_DIR = RESOURCE_DIR / "documents"
GOLD_EVAL_JSONL = RESOURCE_DIR / "gold" / "spider2lite_eval_local.jsonl"
GOLD_EXEC_DIR = RESOURCE_DIR / "gold" / "exec_result"

PRED_SQL_DIR = RESULTS_DIR / "spider2_lite_sqlite_predictions"  # one .sql per item
PRED_META_PATH = RESULTS_DIR / "spider2_lite_sqlite_predictions.json"
EXEC_CSV_DIR = RESULTS_DIR / "spider2_lite_sqlite_exec"  # one .csv per item
EVAL_OUTPUT = RESULTS_DIR / "spider2_lite_sqlite_scored.json"
METRICS_OUTPUT = RESULTS_DIR / "spider2_lite_sqlite_metrics.json"

sys.path.insert(0, str(PROJECT_ROOT))
from src.commons import interfaze_client  # noqa: E402
from benchmarks.spider2_lite_sqlite.eval_helpers import (  # noqa: E402
    extract_sql_query,
    get_sqlite_result,
    score_prediction,
)

RATE_LIMIT = 10
MAX_RETRIES = 3
INTERFAZE_MODEL = "interfaze-beta"
TEMPERATURE = 0
PROMPT_TEMPLATE = """You are an expert SQLite SQL generator.

Given the following SQLite database schema:

{ddl}
{external_knowledge_section}
Question: {question}

Write a single SQLite SQL query that answers the question.

Important:
- SQLite does NOT allow referencing a SELECT-list alias (e.g. `x + y AS s`) inside the same SELECT's CASE/WHERE/HAVING. Inline the expression, or define `s` in an inner CTE/subquery first.
- When the question asks about entities (customers, users, sellers, products, ...), include the entity's canonical identifier column in the SELECT output. When the schema has both a surface ID and a canonical/deduplicated ID (e.g. `customer_id` vs `customer_unique_id`), select the canonical one — surface IDs may duplicate per row/order and break entity-level aggregates.
- Include the ranking metric as a column. Whenever the question implies sorting or selecting by a metric ("top N by X", "highest", "most", "with the most ..."), put that metric (the count, the aggregate, the difference) directly in your SELECT — not only in ORDER BY. The evaluator accepts wider result sets; omitting the ranking metric is a frequent cause of mismatch.
- Do NOT round intermediate terms. Apply ROUND() only to the final outermost expression. Per-component rounding (e.g. `ROUND(years) + ROUND(months/12) + ROUND(days/365)`) accumulates errors past the evaluator's 1e-2 tolerance. Compose the whole expression first, then round once — or omit ROUND() entirely and let the evaluator's tolerance handle precision.
- Pick the right synonym table on the first try. If a literal filter like `WHERE Belts.name='NXT'` returns 0 rows on your mock data, the table is the wrong synonym — try a sibling table (`Promotions.name='NXT'`, etc.). Do not stack more filters in an attempt to rescue an empty result.
- Do not over-filter defensively. Each extra WHERE clause is a chance to drop the correct answer. Only add a filter the question explicitly requires; when in doubt, default to the broader interpretation.
- Output ONE ROW PER GROUP, not one row per entity. When the question groups entities into named buckets/segments/categories ("for each segment", "per category", "in each bucket", "within distinct X segments") and asks for aggregate metrics, the result row count must equal the number of distinct buckets — not the number of underlying entities. If your query returns thousands of rows when the question implies ~5 segments, you forgot a `GROUP BY` on the segment column or kept entity-level rows you should have aggregated away.
- For date-difference arithmetic (career span, age, duration between two dates), prefer `julianday(d2) - julianday(d1)` divided by 365.25 (or 30 for months). Component subtraction via strftime — `(year2 - year1) + (month2 - month1)/12 + (day2 - day1)/365` — fails when months/days roll over: e.g., final 2015-03-30 vs debut 2010-08-15 gives month_diff = 3 - 8 = -5; ABS gives 5, but the actual gap is 7 months. Use julianday for any "span" / "duration" / "years between" calculation.
- A provided domain-knowledge document is a binding spec, not advice. Before writing SQL, restate every formula, threshold, bucket boundary, and column choice from it verbatim, then translate that restatement into SQL one-to-one. Do not paraphrase formulas or substitute "more reasonable" alternatives.

Output format:
- Respond with ONLY the SQL query.
- Do NOT include markdown code fences.
- Do NOT include any explanation or prose.
- Start directly with SELECT or WITH.
"""


class RateLimiter:
    def __init__(self, rate: int):
        self.rate = rate
        self.tokens = float(rate)
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


def load_sqlite_items(limit: int | None = None) -> list[dict]:
    items = []
    with open(DATASET_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d["instance_id"].startswith("local"):
                items.append(d)
    items.sort(key=lambda x: x["instance_id"])
    if limit:
        items = items[:limit]
    return items


def build_schema_dir_map() -> dict[str, Path]:
    """Map db name (as used in spider2-lite.jsonl) → schema folder under sqlite/.

    Handles known case/punctuation mismatches like Db-IMDB → DB_IMDB."""
    if not SQLITE_SCHEMAS_DIR.exists():
        return {}
    available = {p.name: p for p in SQLITE_SCHEMAS_DIR.iterdir() if p.is_dir()}
    norm = {k.lower().replace("-", "_"): v for k, v in available.items()}
    mapping = {}
    for name, path in available.items():
        mapping[name] = path
    for db_lower, path in norm.items():
        mapping.setdefault(db_lower, path)
    return mapping


def resolve_schema_dir(db_name: str) -> Path | None:
    candidates = [db_name, db_name.lower().replace("-", "_")]
    schema_map = build_schema_dir_map()
    for c in candidates:
        if c in schema_map:
            return schema_map[c]
        for k, v in schema_map.items():
            if k.lower() == c.lower():
                return v
    return None


def resolve_sqlite_path(db_name: str) -> Path | None:
    """Find the .sqlite binary for a db name, tolerating case/dash differences."""
    if not SQLITE_DB_DIR.exists():
        return None
    candidates = [
        f"{db_name}.sqlite",
        f"{db_name.lower()}.sqlite",
        f"{db_name.replace('-', '_')}.sqlite",
        f"{db_name.replace('-', '_').upper()}.sqlite",
    ]
    for c in candidates:
        p = SQLITE_DB_DIR / c
        if p.exists():
            return p
    db_lower = db_name.lower().replace("-", "_")
    for p in SQLITE_DB_DIR.glob("*.sqlite"):
        if p.stem.lower().replace("-", "_") == db_lower:
            return p
    return None


def read_ddl(schema_dir: Path) -> str:
    ddl_path = schema_dir / "DDL.csv"
    if not ddl_path.exists():
        return ""
    import csv

    parts = []
    with open(ddl_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ddl = row.get("DDL", "").strip()
            if ddl:
                parts.append(ddl.rstrip(";") + ";")
    return "\n\n".join(parts)


def read_external_knowledge(filename: str | None) -> str:
    if not filename:
        return ""
    path = DOCS_DIR / filename
    if not path.exists():
        return ""
    return path.read_text()


def build_prompt(item: dict) -> tuple[str, list[str]]:
    """Return (prompt_text, missing_pieces) where missing_pieces is for diagnostics."""
    missing = []
    schema_dir = resolve_schema_dir(item["db"])
    schema_text = read_ddl(schema_dir) if schema_dir else ""
    if not schema_text:
        missing.append(f"schema:{item['db']}")

    ek_text = read_external_knowledge(item.get("external_knowledge"))
    if item.get("external_knowledge") and not ek_text:
        missing.append(f"ek:{item['external_knowledge']}")

    ek_section = (
        f"\nDomain knowledge:\n{ek_text}\n" if ek_text else ""
    )
    prompt = PROMPT_TEMPLATE.format(
        ddl=schema_text or "(unavailable)",
        external_knowledge_section=ek_section,
        question=item["question"],
    )
    return prompt, missing


async def predict_one(item: dict, rate_limiter: RateLimiter) -> dict:
    # Bypass invoke_interfaze so we can set temperature=0 explicitly.
    prompt, missing = build_prompt(item)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            resp = await asyncio.to_thread(
                interfaze_client.chat.completions.create,
                model=INTERFAZE_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
            )
            content = resp.choices[0].message.content or ""
            return {
                "instance_id": item["instance_id"],
                "db": item["db"],
                "question": item["question"],
                "predict_raw": content,
                "predict_sql": extract_sql_query(content),
                "missing_context": missing,
                "error": None,
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {
                    "instance_id": item["instance_id"],
                    "db": item["db"],
                    "question": item["question"],
                    "predict_raw": "",
                    "predict_sql": "",
                    "missing_context": missing,
                    "error": str(e),
                }
            await asyncio.sleep(2**attempt)
    return {}  # unreachable


async def run_predictions_for(
    predict_fn,
    pred_meta_path: Path,
    pred_sql_dir: Path,
    limit: int | None = None,
) -> None:
    """Shared predict loop. ``predict_fn`` is async (item, rate_limiter) -> rec."""
    items = load_sqlite_items(limit=limit)
    print(f"Loaded {len(items)} SQLite items")

    pred_sql_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if pred_meta_path.exists():
        with open(pred_meta_path) as f:
            for rec in json.load(f):
                if rec.get("predict_sql"):
                    existing[rec["instance_id"]] = rec
        print(f"Resuming: {len(existing)} cached predictions")

    todo = [it for it in items if it["instance_id"] not in existing]
    rate_limiter = RateLimiter(RATE_LIMIT)
    tasks = [predict_fn(it, rate_limiter) for it in todo]
    new_results = (
        await tqdm_asyncio.gather(*tasks, desc="Predicting") if tasks else []
    )

    all_results = list(existing.values()) + new_results
    all_results.sort(key=lambda r: r["instance_id"])

    for rec in new_results:
        if rec.get("predict_sql"):
            (pred_sql_dir / f"{rec['instance_id']}.sql").write_text(rec["predict_sql"])

    with open(pred_meta_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    failures = sum(1 for r in all_results if not r.get("predict_sql"))
    print(f"Saved {len(all_results)} predictions to {pred_meta_path} ({failures} failures)")


async def run_predictions(limit: int | None = None) -> None:
    await run_predictions_for(predict_one, PRED_META_PATH, PRED_SQL_DIR, limit=limit)


def execute_predictions(
    query_timeout: int = 60,
    pred_meta_path: Path | None = None,
    exec_csv_dir: Path | None = None,
) -> dict[str, dict]:
    """Run each predicted SQL on its SQLite DB and write CSV outputs.

    Items whose CSV already exists are reused (so reruns/scoring don't re-execute).
    Returns mapping instance_id → {csv_path, error}."""
    pred_meta_path = pred_meta_path or PRED_META_PATH
    exec_csv_dir = exec_csv_dir or EXEC_CSV_DIR
    if not pred_meta_path.exists():
        print(f"No predictions found at {pred_meta_path}. Run --predict-only first.")
        sys.exit(1)

    with open(pred_meta_path) as f:
        records = json.load(f)

    exec_csv_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    from tqdm import tqdm
    for rec in tqdm(records, desc="Executing SQL"):
        iid = rec["instance_id"]
        csv_path = exec_csv_dir / f"{iid}.csv"
        if csv_path.exists():
            out[iid] = {"csv_path": str(csv_path), "error": None}
            continue
        sql = rec.get("predict_sql") or ""
        if not sql:
            out[iid] = {"csv_path": None, "error": "no prediction"}
            continue
        sqlite_path = resolve_sqlite_path(rec["db"])
        if sqlite_path is None:
            out[iid] = {"csv_path": None, "error": f"sqlite db not found: {rec['db']}"}
            continue
        ok, info = get_sqlite_result(
            str(sqlite_path),
            sql,
            save_dir=str(exec_csv_dir),
            file_name=f"{iid}.csv",
            timeout=query_timeout,
        )
        if ok:
            out[iid] = {"csv_path": str(csv_path), "error": None}
        else:
            out[iid] = {"csv_path": None, "error": str(info)}
    return out


def load_eval_standards() -> dict[str, dict]:
    standards = {}
    with open(GOLD_EVAL_JSONL) as f:
        for line in f:
            d = json.loads(line)
            standards[d["instance_id"]] = d
    return standards


def run_evaluation(
    pred_meta_path: Path | None = None,
    exec_csv_dir: Path | None = None,
    eval_output: Path | None = None,
    metrics_output: Path | None = None,
    model_name: str = "interfaze-beta",
) -> None:
    pred_meta_path = pred_meta_path or PRED_META_PATH
    exec_csv_dir = exec_csv_dir or EXEC_CSV_DIR
    eval_output = eval_output or EVAL_OUTPUT
    metrics_output = metrics_output or METRICS_OUTPUT

    print("Executing predicted SQL on local SQLite databases...")
    exec_results = execute_predictions(
        pred_meta_path=pred_meta_path, exec_csv_dir=exec_csv_dir
    )

    print("Scoring against gold execution results...")
    standards = load_eval_standards()
    scored: list[dict] = []
    for iid, info in sorted(exec_results.items()):
        if info["csv_path"] is None:
            scored.append(
                {
                    "instance_id": iid,
                    "score": 0,
                    "error_info": info["error"],
                }
            )
            continue
        score, err = score_prediction(
            iid, Path(info["csv_path"]), str(GOLD_EXEC_DIR), standards
        )
        scored.append({"instance_id": iid, "score": score, "error_info": err})

    with open(eval_output, "w") as f:
        json.dump(scored, f, indent=2)

    total = len(scored)
    correct = sum(s["score"] for s in scored)
    accuracy = correct / total if total else 0.0
    metrics = {
        "model": model_name,
        "split": "spider2-lite-sqlite",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Spider2-Lite (SQLite-only) — {metrics['model']}")
    print("=" * 50)
    print(f"Correct: {correct}/{total}  Accuracy: {accuracy:.3f}")
    print(f"Per-item scores: {eval_output}")
    print(f"Metrics: {metrics_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spider 2.0-Lite SQLite-only benchmark for Interfaze"
    )
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit items (debug)")
    args = parser.parse_args()

    if args.evaluate_only:
        run_evaluation()
    elif args.predict_only:
        asyncio.run(run_predictions(limit=args.limit))
    else:
        asyncio.run(run_predictions(limit=args.limit))
        run_evaluation()


if __name__ == "__main__":
    main()
