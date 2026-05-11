"""
Fetch Spider 2.0-Lite resources needed for the SQLite subset of the benchmark.

Two downloads:
    1. The xlang-ai/Spider2 repo (shallow clone) — provides spider2-lite.jsonl,
       DDL/sample data per local db, gold exec_result CSVs, external-knowledge
       docs, and the eval standard config.
    2. The SQLite `.sqlite` database files, hosted in a Google Drive zip. The
       repo only ships tiny sample JSONs; the real per-table data lives here.

Idempotent: re-running skips work that is already done.

Usage:
    uv run -m benchmarks.spider2_lite.fetch_data
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
REPO_DIR = DATA_DIR / "Spider2"
SPIDER2_LITE = REPO_DIR / "spider2-lite"
SQLITE_DB_DIR = SPIDER2_LITE / "resource" / "databases"

# The "Download local database" link from the Spider 2.0-Lite README.
SQLITE_ZIP_DRIVE_ID = "1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG"


def ensure_repo() -> None:
    if SPIDER2_LITE.exists() and (SPIDER2_LITE / "spider2-lite.jsonl").exists():
        print(f"[skip] repo already present at {REPO_DIR}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[clone] xlang-ai/Spider2 (depth=1) -> {REPO_DIR}")
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://github.com/xlang-ai/Spider2.git",
            str(REPO_DIR),
        ],
        check=True,
    )


def ensure_sqlite_dbs() -> None:
    existing = list(SQLITE_DB_DIR.glob("*.sqlite"))
    if existing:
        print(f"[skip] {len(existing)} .sqlite files already in {SQLITE_DB_DIR}")
        return
    SQLITE_DB_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "spider2-localdb.zip"
    if not zip_path.exists():
        try:
            import gdown
        except ImportError:
            print("ERROR: `gdown` is required to download from Google Drive.", file=sys.stderr)
            print("       Add it with `uv add gdown` and re-run.", file=sys.stderr)
            sys.exit(1)
        print(f"[download] Google Drive id={SQLITE_ZIP_DRIVE_ID} -> {zip_path}")
        gdown.download(id=SQLITE_ZIP_DRIVE_ID, output=str(zip_path), quiet=False)
    print(f"[extract] .sqlite files from {zip_path} -> {SQLITE_DB_DIR}")
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.endswith(".sqlite"):
                continue
            dest = SQLITE_DB_DIR / Path(member).name
            if dest.exists():
                continue
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    print(f"[done] extracted {extracted} .sqlite files")


def main() -> None:
    ensure_repo()
    ensure_sqlite_dbs()
    print(f"\nSpider 2.0-Lite data ready under: {SPIDER2_LITE}")


if __name__ == "__main__":
    main()
