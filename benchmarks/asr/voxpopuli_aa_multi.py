"""
VoxPopuli-Cleaned-AA ASR benchmark — multi-provider edition.

Runs the same prompt + dataset + scoring as benchmarks.asr.voxpopuli_aa, but
against non-interfaze providers (Gemini, Anthropic, OpenAI) for head-to-head
WER comparison. Designed to run in parallel with the interfaze run without
stepping on its checkpoint file.

Output: results/voxpopuli_aa_<provider>_<model-slug>_responses.jsonl

Usage:
    uv run -m benchmarks.asr.voxpopuli_aa_multi --provider gemini --model gemini-3-flash-preview
    uv run -m benchmarks.asr.voxpopuli_aa_multi --provider gemini --model gemini-3-flash-preview --limit 5
"""

import sys
import json
import time
import asyncio
import argparse
import traceback
from pathlib import Path

from datasets import load_dataset
from jiwer import wer, cer
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse everything shared: prompt, WER normalization, JSONL writer, etc.
from benchmarks.asr.voxpopuli_aa import (  # noqa: E402
    PROMPT, DATASET_ID, SPLIT, RATE_LIMIT, MAX_RETRIES,
    RateLimiter, JsonlWriter, normalize_text, fetch_audio_bytes,
    build_sample, load_completed_ids, load_records,
    compute_metrics, print_summary,
)

RESULTS_DIR = PROJECT_ROOT / "results"


def _load_interfaze_env() -> dict:
    env = {}
    for line in (Path.home() / "interfaze" / ".env.local").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# -------- provider adapters: (audio_bytes, prompt, model, client) -> (content, req_id) --------

def call_gemini(audio_bytes: bytes, prompt_text: str, model: str, client):
    """Thinking OFF (or as close as the model allows):
       - Gemini 3.x Pro: thinking_level="low" (Pro rejects 'minimal'; min is 'low').
       - Gemini 3.x Flash: thinking_level="minimal" (true disable not supported).
       - Gemini 2.5 Pro: thinking_budget=128 (Pro can't go lower than 128).
       - Gemini 2.5 Flash: thinking_budget=0 (true disable).
       Input audio as inline bytes (Gemini accepts up to ~20 MB inline)."""
    from google.genai import types
    m = model.lower()
    if m.startswith("gemini-2.5-pro"):
        thinking = types.ThinkingConfig(thinking_budget=128)
    elif m.startswith("gemini-2.5-flash"):
        thinking = types.ThinkingConfig(thinking_budget=0)
    elif "pro" in m:
        thinking = types.ThinkingConfig(thinking_level="low")
    else:
        thinking = types.ThinkingConfig(thinking_level="minimal")
    config = types.GenerateContentConfig(
        thinking_config=thinking,
        temperature=0.0,
    )
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            prompt_text,
        ],
        config=config,
    )
    content = (resp.text or "").strip()
    request_id = getattr(resp, "response_id", None) or ""
    return content, request_id


def build_client(provider: str, env: dict):
    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=env["GEMINI_KEY"])
    raise ValueError(f"Provider not yet supported here: {provider}")


def get_call_fn(provider: str):
    return {"gemini": call_gemini}[provider]


# -------- pipeline (mirrors voxpopuli_aa.process_sample, but routed via adapter) --------

async def process_sample(sample: dict, call_fn, model: str, rate_limiter,
                         writer: JsonlWriter, progress: dict, provider: str,
                         client) -> dict | None:
    last_error: str | None = None

    try:
        audio_bytes = await asyncio.to_thread(fetch_audio_bytes, sample["file_name"])
    except Exception as e:
        tqdm.write(f"[{provider} fetch error] id={sample['id']}: {type(e).__name__}: {e}")
        progress["failed"] += 1
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        start = time.perf_counter()
        try:
            content, request_id = await asyncio.to_thread(
                call_fn, audio_bytes, PROMPT, model, client
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            pred_norm = normalize_text(content)
            gt_norm = normalize_text(sample["transcript"])
            sample_wer = float(wer(gt_norm, pred_norm)) if gt_norm else float("inf")
            sample_cer = float(cer(gt_norm, pred_norm)) if gt_norm else float("inf")

            record = {
                "id": sample["id"],
                "file_name": sample["file_name"],
                "duration": sample["duration"],
                "gender": sample["gender"],
                "language": sample["language"],
                "transcript": sample["transcript"],
                "transcript_normalized": gt_norm,
                "prediction": content,
                "prediction_normalized": pred_norm,
                "wer": sample_wer,
                "cer": sample_cer,
                "response": content,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "attempts": attempt,
                "provider": provider,
                "model": model,
            }
            await writer.append(record)

            progress["done"] += 1
            tqdm.write(
                f"[{provider} {progress['done']}/{progress['total']}] WER={sample_wer:.3f} "
                f"CER={sample_cer:.3f} id={sample['id']} dur={sample['duration']}s "
                f"latency={latency_ms}ms req_id={request_id}"
            )
            return record

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            last_error = f"{type(e).__name__}: {e}"
            tqdm.write(
                f"[{provider} error] id={sample['id']} attempt={attempt}/{MAX_RETRIES} "
                f"latency={latency_ms}ms error={last_error}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))

    progress["failed"] += 1
    tqdm.write(f"[{provider} FAILED] id={sample['id']} after {MAX_RETRIES} attempts: {last_error}")
    return None


def build_tag(provider: str, model: str) -> str:
    return f"voxpopuli_aa_{provider}_{model.replace('/', '_').replace(':', '_')}"


async def run(provider: str, model: str, pred_path: Path, limit: int | None):
    env = _load_interfaze_env()
    client = build_client(provider, env)
    call_fn = get_call_fn(provider)

    print(f"[{provider}/{model}] Loading {DATASET_ID}, split={SPLIT}...")
    dataset = load_dataset(DATASET_ID, split=SPLIT)
    samples = [build_sample(dict(row)) for row in dataset]

    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
        print(f"--limit applied: will run at most {limit} sample(s)")
    print(f"Resume: {len(done_ids)} completed, {len(pending)} remaining "
          f"(checkpoint: {pred_path})")
    if not pending:
        return

    writer = JsonlWriter(pred_path)
    rate_limiter = RateLimiter(RATE_LIMIT)
    progress = {"total": len(pending), "done": 0, "failed": 0}

    tasks = [process_sample(s, call_fn, model, rate_limiter, writer, progress, provider, client)
             for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc=f"{provider}/{model}")
    except Exception:
        traceback.print_exc()
    print(f"\n[{provider}/{model}] Run finished: {progress['done']}/{progress['total']}, "
          f"{progress['failed']} failed.")


def run_evaluation(pred_path: Path, metrics_path: Path, provider: str, model: str):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)
    for r in results:
        if r.get("prediction_normalized") is None and r.get("response"):
            r["prediction_normalized"] = normalize_text(r["response"])
        if r.get("transcript_normalized") is None and r.get("transcript"):
            r["transcript_normalized"] = normalize_text(r["transcript"])
    metrics = compute_metrics(results)
    print_summary(metrics)
    output = {
        **metrics,
        "dataset": DATASET_ID, "split": SPLIT,
        "provider": provider, "model": model,
        "rate_limit": RATE_LIMIT,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-provider VoxPopuli-Cleaned-AA eval")
    parser.add_argument("--provider", required=True, choices=["gemini"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tag = build_tag(args.provider, args.model)
    pred_path = RESULTS_DIR / f"{tag}_responses.jsonl"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"

    if args.evaluate_only:
        run_evaluation(pred_path, metrics_path, args.provider, args.model)
    elif args.predict_only:
        asyncio.run(run(args.provider, args.model, pred_path, limit=args.limit))
    else:
        asyncio.run(run(args.provider, args.model, pred_path, limit=args.limit))
        run_evaluation(pred_path, metrics_path, args.provider, args.model)


if __name__ == "__main__":
    main()
