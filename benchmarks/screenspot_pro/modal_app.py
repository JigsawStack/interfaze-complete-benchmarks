"""
Modal runner for the ScreenSpot-Pro benchmark.

Persists JSONL results + HF dataset cache to a Modal Volume so runs survive
local-machine death and can be inspected mid-flight without disturbing the
running container.

Quick reference (run from repo root):

  # smoke test (5 samples, attached so you see logs):
  uv run modal run benchmarks/screenspot_pro/modal_app.py::run \
      --provider gemini --model gemini-2.5-flash --limit 5

  # full detached run (survives Ctrl-C / laptop closing):
  uv run modal run --detach benchmarks/screenspot_pro/modal_app.py::run \
      --provider gemini --model gemini-3.1-pro-preview

  # check progress (reads volume; does NOT touch the running container):
  uv run modal run benchmarks/screenspot_pro/modal_app.py::check \
      --provider gemini --model gemini-3.1-pro-preview

  # pull results back to ./results/ when done:
  uv run modal run benchmarks/screenspot_pro/modal_app.py::download \
      --provider gemini --model gemini-3.1-pro-preview

  # one-off prefetch of the 1581 images into the volume cache (recommended
  # before the first full run, avoids HF 429s):
  uv run modal run benchmarks/screenspot_pro/modal_app.py::prefetch
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "interfaze-screenspot-pro"
SECRET_NAME = "screenspot-bench"
RESULTS_VOLUME = "screenspot-results"
HF_CACHE_VOLUME = "screenspot-hf-cache"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "openai>=2.26.0",
        "anthropic>=0.96.0",
        "google-genai>=1.73.1",
        "huggingface_hub>=0.30.0",
        "datasets>=4.7.0",
        "pillow>=10.0.0",
        "python-dotenv>=1.2.2",
        "tqdm>=4.66.0",
    )
    .env({"PYTHONPATH": "/app", "HF_HOME": "/hf_cache"})
    # add_local_* must come last (Modal injects these at container startup,
    # so they can't be followed by build steps).
    .add_local_dir(
        str(REPO_ROOT / "benchmarks" / "screenspot_pro"),
        remote_path="/app/benchmarks/screenspot_pro",
    )
    .add_local_file(
        str(REPO_ROOT / "benchmarks" / "__init__.py"),
        remote_path="/app/benchmarks/__init__.py",
    )
)

results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
secret = modal.Secret.from_name(SECRET_NAME)

app = modal.App(APP_NAME)


def _slug(model: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _tag(provider: str, model: str, reasoning: str) -> str:
    return f"screenspotpro_{provider}_{_slug(model)}_reasoning{reasoning}"


# ---------------------------------------------------------------------------
# Prefetch: cache all 1,581 dataset images into the volume sequentially.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/hf_cache": hf_cache_vol},
    timeout=60 * 60 * 2,
)
def prefetch() -> None:
    import sys
    sys.path.insert(0, "/app")
    from benchmarks.screenspot_pro.prefetch_images import main as prefetch_main
    prefetch_main()
    hf_cache_vol.commit()


# ---------------------------------------------------------------------------
# Run: execute the benchmark. Streams JSONL into the results volume, with
# fsync after every record (already done in screenspot_pro_multi.JsonlWriter),
# so a separate reader function sees up-to-date progress on volume.reload().
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/results": results_vol, "/hf_cache": hf_cache_vol},
    secrets=[secret],
    timeout=60 * 60 * 24,  # 24h ceiling; full run is much shorter
    cpu=4.0,
    memory=8192,
)
def _run_remote(provider: str, model: str, reasoning: str, limit: int | None) -> dict:
    import asyncio
    import sys
    sys.path.insert(0, "/app")

    # Override RESULTS_DIR to point at the mounted volume before importing.
    import benchmarks.screenspot_pro.screenspot_pro_multi as bench
    bench.RESULTS_DIR = Path("/results")
    bench.REASONING_MODE = reasoning

    tag = _tag(provider, model, reasoning)
    pred_path = Path("/results") / f"{tag}_responses.jsonl"
    metrics_path = Path("/results") / f"{tag}_metrics.json"

    # Periodic volume commits so the reader function sees progress.
    async def _committer():
        while True:
            await asyncio.sleep(20)
            try:
                results_vol.commit()
            except Exception:
                pass

    async def _main():
        committer = asyncio.create_task(_committer())
        try:
            await bench.run(provider, model, pred_path, limit=limit)
        finally:
            committer.cancel()
            results_vol.commit()

    asyncio.run(_main())
    bench.run_evaluation(pred_path, metrics_path, provider, model)
    results_vol.commit()

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {"tag": tag, "metrics_summary": {
        k: metrics.get(k) for k in ("accuracy", "correct", "total")
    }}


@app.local_entrypoint()
def run(provider: str, model: str, reasoning: str = "off", limit: int | None = None):
    """Run the benchmark. Use `modal run --detach ...` for unattended runs."""
    result = _run_remote.remote(provider, model, reasoning, limit)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Check: ephemeral, read-only progress probe. Does NOT touch the run container.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/results": results_vol},
    timeout=120,
)
def _check_remote(provider: str, model: str, reasoning: str) -> dict:
    import sys
    sys.path.insert(0, "/app")
    from benchmarks.screenspot_pro.screenspot_pro_multi import (
        load_records, compute_metrics,
    )

    # Force-refresh the local view of the volume so we see whatever the
    # running writer has flushed so far.
    results_vol.reload()

    tag = _tag(provider, model, reasoning)
    pred_path = Path("/results") / f"{tag}_responses.jsonl"
    if not pred_path.exists():
        return {"tag": tag, "status": "no-file-yet", "path": str(pred_path)}

    records = load_records(pred_path)
    metrics = compute_metrics(records) if records else {}

    # Last record timestamp (mtime) — gives a coarse "is it still moving?" hint.
    mtime = pred_path.stat().st_mtime
    return {
        "tag": tag,
        "records": len(records),
        "accuracy": metrics.get("accuracy"),
        "correct": metrics.get("correct"),
        "total_target": 1581,
        "latency_p50_ms": (metrics.get("latency") or {}).get("p50_ms"),
        "latency_p90_ms": (metrics.get("latency") or {}).get("p90_ms"),
        "file_mtime_epoch": mtime,
    }


@app.local_entrypoint()
def check(provider: str, model: str, reasoning: str = "off"):
    """Print live progress for a (running or finished) benchmark, non-disruptively."""
    result = _check_remote.remote(provider, model, reasoning)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Download: pull the JSONL + metrics back to ./results/ on the local machine.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/results": results_vol},
    timeout=300,
)
def _list_remote() -> list[str]:
    return sorted(p.name for p in Path("/results").iterdir() if p.is_file())


@app.local_entrypoint()
def download(provider: str, model: str, reasoning: str = "off"):
    """Stream the JSONL + metrics for one run back to local ./results/."""
    tag = _tag(provider, model, reasoning)
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [f"{tag}_responses.jsonl", f"{tag}_metrics.json"]
    for fname in files:
        local = out_dir / fname
        with local.open("wb") as f:
            for chunk in results_vol.read_file(fname):
                f.write(chunk)
        print(f"wrote {local} ({local.stat().st_size} bytes)")


@app.local_entrypoint()
def ls():
    """List files currently in the results volume."""
    files = _list_remote.remote()
    for f in files:
        print(f)
