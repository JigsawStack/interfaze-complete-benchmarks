"""
Self-contained evaluation helpers for Spider 2.0-Lite SQLite tasks.

Ported from Spider2/spider2-lite/evaluation_suite/evaluate.py so we don't pull
in google-cloud-bigquery for SQLite-only runs.
"""

from __future__ import annotations

import math
import os
import re
import signal
import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd


class SQLiteTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise SQLiteTimeoutError("query exceeded timeout")


@lru_cache(maxsize=None)
def load_gold_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def extract_sql_query(pred_sql_query: str) -> str:
    pattern = r"```sql\n(.*?)\n```"
    match = re.search(pattern, pred_sql_query, re.DOTALL)
    if match:
        return match.group(1).strip()
    return pred_sql_query.strip()


def get_sqlite_result(
    db_path: str,
    query: str,
    save_dir: str | None = None,
    file_name: str = "result.csv",
    chunksize: int = 500,
    timeout: int = 60,
) -> tuple[bool, object]:
    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        conn = sqlite3.connect(db_path)
        memory_conn = sqlite3.connect(":memory:")
        conn.backup(memory_conn)
        try:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                wrote_any = False
                for i, chunk in enumerate(
                    pd.read_sql_query(query, memory_conn, chunksize=chunksize)
                ):
                    mode = "a" if i > 0 else "w"
                    header = i == 0
                    chunk.to_csv(
                        os.path.join(save_dir, file_name),
                        mode=mode,
                        header=header,
                        index=False,
                    )
                    wrote_any = True
                if not wrote_any:
                    pd.DataFrame().to_csv(
                        os.path.join(save_dir, file_name), index=False
                    )
                return True, None
            df = pd.read_sql_query(query, memory_conn)
            return True, df
        finally:
            memory_conn.close()
            conn.close()
    except SQLiteTimeoutError:
        return False, f"timeout after {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def compare_pandas_table(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    condition_cols=None,
    ignore_order: bool = False,
) -> int:
    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]
        if ignore_order_:
            v1 = sorted(
                v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))
            )
            v2 = sorted(
                v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))
            )
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

    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold

    pred_cols = pred
    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for gold_vector in t_gold_list:
        if not any(
            vectors_match(gold_vector, pred_vector, ignore_order_=ignore_order)
            for pred_vector in t_pred_list
        ):
            score = 0
            break
    return score


def compare_multi_pandas_table(
    pred: pd.DataFrame, multi_gold, multi_condition_cols=None, multi_ignore_order=False
) -> int:
    if not multi_gold:
        return 0

    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(
        isinstance(sublist, list) for sublist in multi_condition_cols
    ):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]

    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(
            pred, gold, multi_condition_cols[i], multi_ignore_order[i]
        ):
            return 1
    return 0


def resolve_gold_paths(
    instance_id: str, gold_result_dir: str
) -> tuple[list[Path], bool]:
    base_path = Path(gold_result_dir) / f"{instance_id}.csv"
    if base_path.exists():
        return [base_path], True

    pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    csv_files = sorted(
        file for file in os.listdir(gold_result_dir) if pattern.match(file)
    )
    return [Path(gold_result_dir) / file for file in csv_files], False


def score_prediction(
    instance_id: str,
    pred_csv: Path,
    gold_result_dir: str,
    eval_standard_dict: dict,
) -> tuple[int, str | None]:
    """Compare a predicted CSV to the gold CSV(s). Returns (score, error_info)."""
    try:
        pred_pd = pd.read_csv(pred_csv)
    except Exception as e:
        return 0, f"Failed to read pred CSV: {e}"

    gold_paths, is_single = resolve_gold_paths(instance_id, gold_result_dir)
    standard = eval_standard_dict.get(instance_id, {})
    condition_cols = standard.get("condition_cols")
    ignore_order = standard.get("ignore_order", False)

    if not gold_paths:
        return 0, "No matching gold file found"

    try:
        if is_single:
            gold_pd = load_gold_csv(str(gold_paths[0]))
            score = compare_pandas_table(pred_pd, gold_pd, condition_cols, ignore_order)
        else:
            gold_pds = [load_gold_csv(str(p)) for p in gold_paths]
            score = compare_multi_pandas_table(
                pred_pd, gold_pds, condition_cols, ignore_order
            )
    except Exception as e:
        return 0, f"Compare error: {e}"

    if score == 0:
        return 0, "Result mismatch"
    return 1, None
