"""
Modal runner for the MMMU Pro benchmark.

Persists JSONL results + HF dataset cache to Modal Volumes so runs survive
local-machine death and can be inspected mid-flight without disturbing the
running container.

Two settings, both 1730 samples each:
  - standard: text question + up to 7 inline images
  - vision:   single rendered image of the entire question

Quick reference (run from repo root):

  # smoke test (5 samples, attached so you see logs):
  uv run modal run benchmarks/mmmu_pro/modal_app.py::run \\
      --provider gemini --model gemini-2.5-flash --setting standard --limit 5

  # full detached run (survives Ctrl-C / laptop closing):
  uv run modal run --detach benchmarks/mmmu_pro/modal_app.py::run \\
      --provider gemini --model gemini-3.1-pro-preview --setting standard

  # check progress (reads volume; does NOT touch the running container):
  uv run modal run benchmarks/mmmu_pro/modal_app.py::check \\
      --provider gemini --model gemini-3.1-pro-preview --setting standard

  # pull results back to ./results/ when done:
  uv run modal run benchmarks/mmmu_pro/modal_app.py::download \\
      --provider gemini --model gemini-3.1-pro-preview --setting standard

  # one-off prefetch of both dataset configs into the HF cache volume
  # (recommended before launching parallel runs to avoid HF 429s):
  uv run modal run benchmarks/mmmu_pro/modal_app.py::prefetch
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "interfaze-mmmu-pro"
SECRET_NAME = "screenspot-bench"  # same provider keys; reused
RESULTS_VOLUME = "mmmu-pro-results"
HF_CACHE_VOLUME = "mmmu-pro-hf-cache"

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
        str(REPO_ROOT / "benchmarks" / "mmmu_pro"),
        remote_path="/app/benchmarks/mmmu_pro",
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


def _tag(provider: str, model: str, setting: str, reasoning: str) -> str:
    return f"mmmupro_{setting}_{provider}_{_slug(model)}_reasoning{reasoning}"


# ---------------------------------------------------------------------------
# Prefetch: cache both dataset configs into the volume sequentially, so the
# parallel benchmark runs hit the local cache rather than HF directly.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/hf_cache": hf_cache_vol},
    timeout=60 * 60 * 2,
)
def prefetch() -> None:
    from datasets import load_dataset
    for cfg in ["standard (10 options)", "vision"]:
        print(f"Loading MMMU/MMMU_Pro config={cfg!r} split=test...")
        ds = load_dataset("MMMU/MMMU_Pro", cfg, split="test")
        print(f"  cached {len(ds)} rows")
    hf_cache_vol.commit()


# ---------------------------------------------------------------------------
# Run: execute the benchmark. Streams JSONL into the results volume, with
# fsync after every record (already done in mmmu_pro_multi.JsonlWriter),
# so a separate reader function sees up-to-date progress on volume.reload().
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/results": results_vol, "/hf_cache": hf_cache_vol},
    secrets=[secret],
    timeout=60 * 60 * 24,  # 24h ceiling
    cpu=4.0,
    memory=8192,
)
def _run_remote(provider: str, model: str, setting: str, reasoning: str,
                limit: int | None) -> dict:
    import asyncio
    import sys
    sys.path.insert(0, "/app")

    # Override RESULTS_DIR to point at the mounted volume before importing.
    import benchmarks.mmmu_pro.mmmu_pro_multi as bench
    bench.RESULTS_DIR = Path("/results")
    bench.REASONING_MODE = reasoning

    tag = _tag(provider, model, setting, reasoning)
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
            await bench.run(provider, model, setting, pred_path, limit=limit)
        finally:
            committer.cancel()
            results_vol.commit()

    asyncio.run(_main())
    bench.run_evaluation(pred_path, metrics_path, provider, model, setting)
    results_vol.commit()

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {"tag": tag, "metrics_summary": {
        k: metrics.get(k) for k in ("accuracy", "num_samples", "unparseable")
    }}


@app.local_entrypoint()
def run(provider: str, model: str, setting: str = "standard",
        reasoning: str = "off", limit: int | None = None):
    """Run the benchmark. Use `modal run --detach ...` for unattended runs."""
    if setting not in ("standard", "vision"):
        raise SystemExit(f"--setting must be 'standard' or 'vision', got {setting!r}")
    result = _run_remote.remote(provider, model, setting, reasoning, limit)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Check: ephemeral, read-only progress probe. Does NOT touch the run container.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/results": results_vol},
    timeout=120,
)
def _check_remote(provider: str, model: str, setting: str, reasoning: str) -> dict:
    import sys
    sys.path.insert(0, "/app")
    from benchmarks.mmmu_pro.mmmu_pro_multi import (
        load_records, compute_metrics,
    )

    # Force-refresh the local view of the volume so we see whatever the
    # running writer has flushed so far.
    results_vol.reload()

    tag = _tag(provider, model, setting, reasoning)
    pred_path = Path("/results") / f"{tag}_responses.jsonl"
    if not pred_path.exists():
        return {"tag": tag, "status": "no-file-yet", "path": str(pred_path)}

    records = load_records(pred_path)
    metrics = compute_metrics(records) if records else {}

    mtime = pred_path.stat().st_mtime
    return {
        "tag": tag,
        "records": len(records),
        "accuracy": metrics.get("accuracy"),
        "unparseable": metrics.get("unparseable"),
        "total_target": 1730,
        "latency_p50_ms": (metrics.get("latency") or {}).get("p50_ms"),
        "latency_p90_ms": (metrics.get("latency") or {}).get("p90_ms"),
        "file_mtime_epoch": mtime,
    }


@app.local_entrypoint()
def check(provider: str, model: str, setting: str = "standard",
          reasoning: str = "off"):
    """Print live progress for a (running or finished) benchmark, non-disruptively."""
    result = _check_remote.remote(provider, model, setting, reasoning)
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
def download(provider: str, model: str, setting: str = "standard",
             reasoning: str = "off"):
    """Stream the JSONL + metrics for one run back to local ./results/."""
    tag = _tag(provider, model, setting, reasoning)
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
