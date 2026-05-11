"""
Spider 2.0-Lite — SQLite subset — text-to-SQL benchmark for Interfaze.

This runs the 135 `local*` examples from Spider 2.0-Lite (the SQLite-backed
slice). Scoring is execution accuracy: predicted SQL is executed against the
per-example `.sqlite` file and the result DataFrame is compared to the gold
exec_result CSV(s). Comparison logic is ported from the official
`evaluation_suite/evaluate.py` so scores are directly consistent with the
reference implementation.

Important caveat: this is ~25% of the full Spider 2.0-Lite benchmark (135/547).
The BigQuery and Snowflake subsets are excluded because they need external
warehouse access. A result here is reportable as "Spider 2.0-Lite (SQLite
subset, N=135)" — NOT as the headline Spider 2.0-Lite score.

Setup (one-time):
    uv run -m benchmarks.spider2_lite.fetch_data

Usage:
    # Full run (predict + evaluate)
    uv run -m benchmarks.spider2_lite.spider2_lite

    # Prediction only
    uv run -m benchmarks.spider2_lite.spider2_lite --predict-only

    # Evaluation only
    uv run -m benchmarks.spider2_lite.spider2_lite --evaluate-only

    # Smoke test
    uv run -m benchmarks.spider2_lite.spider2_lite --limit 1

Checkpointing: each successful prediction is appended to
`results/spider2_lite_local_responses.jsonl` AND written as
`results/spider2_lite_local_sql/<instance_id>.sql`. Reruns only query
examples still missing from the JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
import sqlite3
import time
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.commons import invoke_interfaze  # noqa: E402

# Paths -----------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
SPIDER2_LITE = DATA_DIR / "Spider2" / "spider2-lite"
SQLITE_DB_DIR = SPIDER2_LITE / "resource" / "databases"
SCHEMA_DIR = SPIDER2_LITE / "resource" / "databases" / "sqlite"
DOCUMENTS_DIR = SPIDER2_LITE / "resource" / "documents"
GOLD_DIR = SPIDER2_LITE / "evaluation_suite" / "gold"
EVAL_STANDARD = GOLD_DIR / "spider2lite_eval.jsonl"
GOLD_EXEC_DIR = GOLD_DIR / "exec_result"
ALL_EXAMPLES = SPIDER2_LITE / "spider2-lite.jsonl"

RESULTS_DIR = PROJECT_ROOT / "results"
TAG = "spider2_lite_local"
RESPONSES_PATH = RESULTS_DIR / f"{TAG}_responses.jsonl"
PRED_SQL_DIR = RESULTS_DIR / f"{TAG}_sql"
METRICS_PATH = RESULTS_DIR / f"{TAG}_metrics.json"

# Config ----------------------------------------------------------------------

REASONING_EFFORT = None
TEMPERATURE = 0.0
RATE_LIMIT = 8
MAX_RETRIES = 3
# Truncate huge DDLs / external-knowledge docs so a single monster schema
# doesn't blow the context. 80k chars leaves plenty of room for reasoning
# output on a 200k-token context model.
MAX_DDL_CHARS = 80_000
MAX_EK_CHARS = 40_000

PROMPT_TEMPLATE = """You are an expert SQLite SQL developer. Write a SQL query that answers the user's question against the given database. Target dialect: SQLite.

### Database Schema
{schema}
{external_knowledge_section}
### Question
{question}

Return ONLY the final SQL query, wrapped in a fenced code block like:
```sql
SELECT ...
```
Do not include any explanation before or after the code block."""


# -----------------------------------------------------------------------------
# Utilities shared with other benches
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def load_local_examples() -> list[dict]:
    if not ALL_EXAMPLES.exists():
        raise FileNotFoundError(
            f"Missing {ALL_EXAMPLES}. Run: uv run -m benchmarks.spider2_lite.fetch_data"
        )
    with open(ALL_EXAMPLES, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r["instance_id"].startswith("local")]


def load_schema(db_name: str) -> str:
    """Concatenate every table's DDL from resource/databases/sqlite/<db>/DDL.csv.
    DDL.csv is a 2-column file: `table_name,DDL`. Some DDL strings span lines,
    so use csv.DictReader rather than hand-parsing."""
    ddl_path = SCHEMA_DIR / db_name / "DDL.csv"
    if not ddl_path.exists():
        return f"-- schema file missing: {ddl_path}"
    parts: list[str] = []
    with open(ddl_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ddl = (row.get("DDL") or "").strip()
            if ddl:
                parts.append(ddl.rstrip(";") + ";")
    schema = "\n\n".join(parts)
    if len(schema) > MAX_DDL_CHARS:
        schema = schema[:MAX_DDL_CHARS] + "\n-- [schema truncated]"
    return schema


def load_external_knowledge(filename: str | None) -> str | None:
    if not filename:
        return None
    path = DOCUMENTS_DIR / filename
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > MAX_EK_CHARS:
        text = text[:MAX_EK_CHARS] + "\n... [truncated]"
    return text


def build_prompt(example: dict) -> str:
    schema = load_schema(example["db"])
    ek = load_external_knowledge(example.get("external_knowledge"))
    ek_section = f"\n### External Knowledge\n{ek}\n" if ek else "\n"
    return PROMPT_TEMPLATE.format(
        schema=schema,
        external_knowledge_section=ek_section,
        question=example["question"],
    )


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------

_SQL_FENCE = re.compile(r"```sql\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ANY_FENCE = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)


def extract_sql(text: str) -> str:
    """Mirrors the behavior of the official evaluate.py: prefer an ```sql
    fenced block, otherwise treat the whole response as SQL. Also tolerate
    a plain ``` fence without the `sql` tag."""
    if not text:
        return ""
    m = _SQL_FENCE.search(text)
    if m:
        return m.group(1).strip()
    m = _ANY_FENCE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("pred_sql"):
                done.add(str(rec["instance_id"]))
    return done


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    by_id: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_id[str(rec["instance_id"])] = rec
    return list(by_id.values())


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------

async def process_example(example: dict, rate_limiter: RateLimiter,
                          writer: JsonlWriter, progress: dict) -> dict | None:
    instance_id = example["instance_id"]
    prompt = build_prompt(example)
    messages = [{"role": "user", "content": prompt}]

    last_error: str | None = None
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

            pred_sql = extract_sql(content)
            # Persist per-instance .sql file so the official evaluate.py can
            # consume the same directory if someone wants to cross-check.
            PRED_SQL_DIR.mkdir(parents=True, exist_ok=True)
            (PRED_SQL_DIR / f"{instance_id}.sql").write_text(pred_sql, encoding="utf-8")

            record = {
                "instance_id": instance_id,
                "db": example["db"],
                "question": example["question"],
                "external_knowledge": example.get("external_knowledge"),
                "pred_sql": pred_sql,
                "response": content,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "attempts": attempt,
            }
            await writer.append(record)
            progress["done"] += 1
            tqdm.write(
                f"[{progress['done']}/{progress['total']}] OK "
                f"id={instance_id} db={example['db']} "
                f"latency={latency_ms}ms sql_len={len(pred_sql)} "
                f"req_id={request_id} attempt={attempt}"
            )
            return record

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            last_error = f"{type(e).__name__}: {e}"
            tqdm.write(
                f"[error] id={instance_id} attempt={attempt}/{MAX_RETRIES} "
                f"latency={latency_ms}ms error={last_error}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))

    progress["failed"] += 1
    tqdm.write(f"[FAILED] id={instance_id} after {MAX_RETRIES} attempts: {last_error}")
    return None


async def run_prediction(examples: list[dict], limit: int | None):
    done_ids = load_completed_ids(RESPONSES_PATH)
    pending = [e for e in examples if e["instance_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
        print(f"--limit applied: will run at most {limit} example(s)")
    print(f"Resume: {len(done_ids)} already completed, {len(pending)} remaining "
          f"(checkpoint: {RESPONSES_PATH})")
    if not pending:
        return

    writer = JsonlWriter(RESPONSES_PATH)
    rate_limiter = RateLimiter(RATE_LIMIT)
    progress = {"total": len(pending), "done": 0, "failed": 0}
    tasks = [process_example(e, rate_limiter, writer, progress) for e in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc="spider2-lite/local")
    except Exception:
        traceback.print_exc()
    print(f"\nPrediction finished: {progress['done']}/{progress['total']} answered, "
          f"{progress['failed']} failed.")


# -----------------------------------------------------------------------------
# Evaluation — SQLite execution + row-set comparison.
#
# Ported from spider2-lite/evaluation_suite/evaluate.py. Kept faithful to the
# original semantics (column-vector matching, float tolerance of 1e-2,
# condition_cols / ignore_order flags, multi-gold).
# -----------------------------------------------------------------------------

def _normalize(v):
    return 0 if pd.isna(v) else v


def _sort_key(x):
    return (x is None, str(x), isinstance(x, (int, float)))


def _vectors_match(v1, v2, ignore_order: bool, tol: float = 1e-2) -> bool:
    v1 = [_normalize(x) for x in v1]
    v2 = [_normalize(x) for x in v2]
    if ignore_order:
        v1 = sorted(v1, key=_sort_key)
        v2 = sorted(v2, key=_sort_key)
    if len(v1) != len(v2):
        return False
    for a, b in zip(v1, v2):
        if pd.isna(a) and pd.isna(b):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tol):
                return False
        elif a != b:
            return False
    return True


def compare_table(pred: pd.DataFrame, gold: pd.DataFrame,
                  condition_cols, ignore_order: bool) -> int:
    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    t_gold = gold_cols.transpose().values.tolist()
    t_pred = pred.transpose().values.tolist()
    for gv in t_gold:
        if not any(_vectors_match(gv, pv, ignore_order) for pv in t_pred):
            return 0
    return 1


def compare_multi(pred: pd.DataFrame, golds: list[pd.DataFrame],
                  multi_condition_cols, ignore_order: bool) -> int:
    if not golds:
        return 0
    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in golds]
    elif len(golds) > 1 and not all(isinstance(s, list) for s in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in golds]
    for gold, cc in zip(golds, multi_condition_cols):
        if compare_table(pred, gold, cc, ignore_order):
            return 1
    return 0


def resolve_gold_paths(instance_id: str) -> tuple[list[Path], bool]:
    base = GOLD_EXEC_DIR / f"{instance_id}.csv"
    if base.exists():
        return [base], True
    pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    matches = sorted(
        GOLD_EXEC_DIR / name
        for name in os.listdir(GOLD_EXEC_DIR)
        if pattern.match(name)
    )
    return matches, False


def execute_sqlite(db_path: Path, sql: str) -> tuple[bool, pd.DataFrame | str]:
    """Run `sql` against `db_path`. Returns (ok, df-or-error-string).

    We copy the on-disk DB into :memory: (same pattern as the official
    evaluate.py) — faster for repeated queries and isolates writes.
    """
    try:
        disk = sqlite3.connect(str(db_path))
        mem = sqlite3.connect(":memory:")
        try:
            disk.backup(mem)
            df = pd.read_sql_query(sql, mem)
            return True, df
        finally:
            mem.close()
            disk.close()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def load_eval_standard() -> dict[str, dict]:
    if not EVAL_STANDARD.exists():
        raise FileNotFoundError(
            f"Missing {EVAL_STANDARD}. Run: uv run -m benchmarks.spider2_lite.fetch_data"
        )
    out: dict[str, dict] = {}
    with open(EVAL_STANDARD, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["instance_id"]] = rec
    return out


def evaluate_record(record: dict, eval_std: dict) -> dict:
    instance_id = record["instance_id"]
    db_path = SQLITE_DB_DIR / f"{record['db']}.sqlite"
    if not db_path.exists():
        return {
            "instance_id": instance_id, "score": 0,
            "error": f"missing sqlite db: {db_path}",
        }
    pred_sql = record.get("pred_sql") or ""
    if not pred_sql.strip():
        return {"instance_id": instance_id, "score": 0, "error": "empty pred_sql"}

    ok, result = execute_sqlite(db_path, pred_sql)
    if not ok:
        return {"instance_id": instance_id, "score": 0, "error": f"sql error: {result}"}

    pred_df: pd.DataFrame = result  # type: ignore[assignment]
    gold_paths, is_single = resolve_gold_paths(instance_id)
    if not gold_paths:
        return {"instance_id": instance_id, "score": 0, "error": "no gold file"}

    standard = eval_std.get(instance_id, {})
    condition_cols = standard.get("condition_cols")
    ignore_order = standard.get("ignore_order", False)

    try:
        if is_single:
            gold_df = pd.read_csv(gold_paths[0])
            score = compare_table(pred_df, gold_df, condition_cols, ignore_order)
        else:
            gold_dfs = [pd.read_csv(p) for p in gold_paths]
            score = compare_multi(pred_df, gold_dfs, condition_cols, ignore_order)
    except Exception as e:
        return {"instance_id": instance_id, "score": 0, "error": f"compare: {e}"}

    return {"instance_id": instance_id, "score": score, "error": None}


def run_evaluation(total_local: int) -> None:
    records = load_records(RESPONSES_PATH)
    if not records:
        print(f"No predictions at {RESPONSES_PATH}")
        sys.exit(1)
    eval_std = load_eval_standard()

    results: list[dict] = []
    for rec in tqdm(records, desc="Evaluating"):
        res = evaluate_record(rec, eval_std)
        results.append(res)
        mark = "OK" if res["score"] == 1 else "X "
        extra = f" ({res['error']})" if res.get("error") else ""
        tqdm.write(f"  {mark} {res['instance_id']}{extra}")

    correct = sum(r["score"] for r in results)
    total = len(results)
    accuracy_of_evaluated = correct / total if total else 0.0
    accuracy_of_subset = correct / total_local if total_local else 0.0

    print(f"\n{'=' * 60}")
    print(f"Spider 2.0-Lite — SQLite subset (Interfaze, reasoning={REASONING_EFFORT})")
    print(f"{'=' * 60}")
    print(f"Correct                : {correct}/{total}")
    print(f"Accuracy (of predicted): {accuracy_of_evaluated:.4f}")
    print(f"Accuracy (of local 135): {accuracy_of_subset:.4f}")

    # Top error categories for quick eyeballing.
    errors = [r for r in results if r["score"] == 0 and r.get("error")]
    if errors:
        print("\nTop error kinds:")
        kinds: dict[str, int] = {}
        for r in errors:
            kind = (r["error"] or "").split(":", 1)[0]
            kinds[kind] = kinds.get(kind, 0) + 1
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {v:4d}  {k}")

    out = {
        "accuracy_evaluated": accuracy_of_evaluated,
        "accuracy_of_local_135": accuracy_of_subset,
        "correct": correct,
        "total_evaluated": total,
        "total_local_subset": total_local,
        "subset": "local (SQLite)",
        "benchmark": "Spider 2.0-Lite",
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "per_example": results,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Spider 2.0-Lite SQLite subset (Interfaze)")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only predict the first N pending examples")
    args = parser.parse_args()

    examples = load_local_examples()
    print(f"Loaded {len(examples)} local (SQLite) examples from {ALL_EXAMPLES.name}")

    if args.evaluate_only:
        run_evaluation(total_local=len(examples))
    elif args.predict_only:
        asyncio.run(run_prediction(examples, limit=args.limit))
    else:
        asyncio.run(run_prediction(examples, limit=args.limit))
        run_evaluation(total_local=len(examples))


if __name__ == "__main__":
    main()
