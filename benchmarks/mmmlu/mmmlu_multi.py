"""
MMMLU benchmark — multi-provider runner.

Same dataset, prompt, parser, and scoring as benchmarks.mmmlu.mmmlu (the
interfaze run), but routed through other providers (OpenAI, Anthropic, Google)
for head-to-head comparison.

Methodology mirrors Gemini 3 Pro's published MMMLU setup:
  - pass@1, single trial, no majority voting
  - reasoning OFF where the model supports it (lowest available level)
  - temperature=0 where the API allows it (some providers reject t=0 + thinking)
  - macro-average accuracy across the 14 translated languages = headline number

Output (per provider+model, so parallel runs don't clash):
  results/mmmlu_<provider>_<model_slug>_responses.jsonl
  results/mmmlu_<provider>_<model_slug>_metrics.json

Usage:
    uv run -m benchmarks.mmmlu.mmmlu_multi --provider interfaze --model interfaze-beta
    uv run -m benchmarks.mmmlu.mmmlu_multi --provider openai    --model gpt-5.4-mini
    uv run -m benchmarks.mmmlu.mmmlu_multi --provider gemini    --model gemini-3.1-pro-preview
    uv run -m benchmarks.mmmlu.mmmlu_multi --provider anthropic --model claude-sonnet-4-6
    # Smoke (1 sample per language = 14 total):
    uv run -m benchmarks.mmmlu.mmmlu_multi --provider gemini --model gemini-3-flash-preview --limit 1

Env: OPENAI_API_KEY, GEMINI_KEY, ANTHROPIC_API_KEY, INTERFAZE_API_KEY (loaded from .env).
"""

import os
import re
import sys
import json
import time
import asyncio
import argparse
import traceback
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse helpers from the interfaze base script — guarantees identical prompt /
# parser / scoring across providers.
from benchmarks.mmmlu.mmmlu import (  # noqa: E402
    DATASET_ID,
    SPLIT,
    LANGUAGES,
    PROMPT_TEMPLATE,
    JsonlWriter,
    RateLimiter,
    build_sample,
    compute_metrics,
    load_completed_ids,
    load_records,
    parse_answer,
    print_summary as _base_print_summary,
)

load_dotenv()


def _load_interfaze_env_fallback() -> None:
    """Pull any keys from ~/interfaze/.env.local that aren't already in os.environ.
    Lets us run providers whose keys live there (e.g. ANTHROPIC_API_KEY) without
    requiring users to duplicate them into the project .env."""
    path = Path.home() / "interfaze" / ".env.local"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_interfaze_env_fallback()

RESULTS_DIR = PROJECT_ROOT / "results"
RATE_LIMIT = 50
MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.0

# Dataset variant — "full" (openai/MMMLU, ~196k) or "lite" (opencompass/mmmlu_lite,
# ~20k stratified). Lite uses a different schema (input/target vs Question/Answer).
DATASET_VARIANT = "lite"

DATASET_FULL_ID = "openai/MMMLU"
DATASET_LITE_ID = "opencompass/mmmlu_lite"

# Reasoning mode — "off" (each model at its floor) or "high" (each model at
# max). Mutated by CLI before any inference runs.
REASONING_MODE = "off"

# Anthropic thinking budget (only used when REASONING_MODE == "high"). max_tokens
# must be > budget_tokens; we set 16k cap with 10k thinking budget.
ANTHROPIC_HIGH_BUDGET_TOKENS = 10_000
ANTHROPIC_HIGH_MAX_TOKENS = 16_000
ANTHROPIC_OFF_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Dataset loading + sample building
# ---------------------------------------------------------------------------

def build_sample_lite(row: dict, language: str, row_index: int) -> dict:
    """opencompass/mmmlu_lite uses input/target/A-D/subject (no Unnamed: 0)."""
    return {
        "id": f"{language}:{row_index}",
        "language": language,
        "row_index": row_index,
        "subject": row["subject"],
        "question": row["input"],
        "a": row["A"],
        "b": row["B"],
        "c": row["C"],
        "d": row["D"],
        "answer": str(row["target"]).strip().upper(),
    }


def load_dataset_for_variant(variant: str, lang: str):
    if variant == "lite":
        return load_dataset(DATASET_LITE_ID, lang, split="test")
    return load_dataset(DATASET_FULL_ID, lang, split="test")


def build_sample_for_variant(variant: str, row: dict, language: str, row_index: int) -> dict:
    if variant == "lite":
        return build_sample_lite(row, language, row_index)
    return build_sample(row, language)


# ---------------------------------------------------------------------------
# Provider adapters: each takes (prompt, model, client) and returns
#   (content: str, request_id: str | None, usage: dict)
# `usage` keys: input_tokens, output_tokens, reasoning_tokens (or None if
# the provider doesn't report it). Lets us verify whether reasoning is
# actually OFF rather than just configured off.
# Run synchronously inside asyncio.to_thread.
# ---------------------------------------------------------------------------

def _safe_int(x):
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None

def _openai_usage(resp) -> dict:
    """Extract input/output/reasoning tokens from an OpenAI chat.completions response."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    return {
        "input_tokens": _safe_int(getattr(u, "prompt_tokens", None)),
        "output_tokens": _safe_int(getattr(u, "completion_tokens", None)),
        "reasoning_tokens": _safe_int(reasoning),
    }


def _gemini_usage(resp) -> dict:
    """Extract usage from a google-genai response. Gemini reports thinking
    tokens as `thoughts_token_count` — 0 confirms reasoning fully off."""
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}
    return {
        "input_tokens": _safe_int(getattr(u, "prompt_token_count", None)),
        "output_tokens": _safe_int(getattr(u, "candidates_token_count", None)),
        "reasoning_tokens": _safe_int(getattr(u, "thoughts_token_count", None)),
    }


def _anthropic_usage(resp) -> dict:
    """Anthropic — `output_tokens` includes thinking tokens when extended
    thinking is on. With thinking disabled, no thinking content blocks
    appear, so output_tokens is purely the visible answer."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}
    return {
        "input_tokens": _safe_int(getattr(u, "input_tokens", None)),
        "output_tokens": _safe_int(getattr(u, "output_tokens", None)),
        "reasoning_tokens": None,  # not separately reported
    }


def call_interfaze(prompt: str, model: str, client) -> tuple[str, str | None, dict]:
    """Interfaze uses 'off'|'high' for reasoning_effort (NOT 'none'; that's
    OpenAI's vocabulary). Valid values per the API: minimal|low|medium|high|on|off|auto."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": DEFAULT_TEMPERATURE,
        "reasoning_effort": "off" if REASONING_MODE == "off" else "high",
    }
    resp = client.chat.completions.create(**kwargs)
    return (
        (resp.choices[0].message.content or "").strip(),
        getattr(resp, "id", None),
        _openai_usage(resp),
    )


def call_openai(prompt: str, model: str, client) -> tuple[str, str | None, dict]:
    """GPT-5.x — reasoning off vs high. With reasoning engaged the API rejects
    temperature!=default, so we omit temperature in 'high' mode."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if REASONING_MODE == "off":
        kwargs["reasoning_effort"] = "none"
        kwargs["temperature"] = DEFAULT_TEMPERATURE
    else:
        kwargs["reasoning_effort"] = "high"
    resp = client.chat.completions.create(**kwargs)
    return (
        (resp.choices[0].message.content or "").strip(),
        getattr(resp, "id", None),
        _openai_usage(resp),
    )


def call_anthropic(prompt: str, model: str, client) -> tuple[str, str | None, dict]:
    """Claude — thinking explicitly disabled vs enabled with a high budget.
    With thinking enabled, Anthropic requires temperature=1, so we omit it."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if REASONING_MODE == "off":
        kwargs["thinking"] = {"type": "disabled"}
        kwargs["temperature"] = DEFAULT_TEMPERATURE
        kwargs["max_tokens"] = ANTHROPIC_OFF_MAX_TOKENS
    else:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": ANTHROPIC_HIGH_BUDGET_TOKENS}
        kwargs["max_tokens"] = ANTHROPIC_HIGH_MAX_TOKENS
    resp = client.messages.create(**kwargs)
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip(), resp.id, _anthropic_usage(resp)


def _openrouter_extra_body(model: str) -> dict:
    """Per-model OpenRouter extras. Models differ on what `reasoning` shapes
    they accept and which underlying provider should serve them.

    - x-ai/grok-4.3: rejects `enabled=false`, so `off` maps to the lowest
      accepted tier (`effort=minimal`); `high` = default thinking on.
    - moonshotai/kimi-k2.6: supports `enabled=false`; pin provider to Moonshot
      so we benchmark Moonshot's own deployment, not a downstream reseller.
    - default: assume the model accepts the unified `enabled` toggle.
    """
    m = model.lower()
    if m.startswith("x-ai/grok-4.3"):
        if REASONING_MODE == "off":
            return {"reasoning": {"effort": "minimal"}}
        return {"reasoning": {"enabled": True}}
    if m.startswith("moonshotai/"):
        body = {"reasoning": {"enabled": REASONING_MODE != "off"}}
        body["provider"] = {"only": ["moonshotai"]}
        return body
    return {"reasoning": {"enabled": REASONING_MODE != "off"}}


def call_openrouter(prompt: str, model: str, client) -> tuple[str, str | None, dict]:
    """OpenRouter dispatch — see _openrouter_extra_body for per-model knobs.
    temperature=0 is accepted across the models we currently route here."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=DEFAULT_TEMPERATURE,
        extra_body=_openrouter_extra_body(model),
    )
    return (
        (resp.choices[0].message.content or "").strip(),
        getattr(resp, "id", None),
        _openai_usage(resp),
    )


def call_gemini(prompt: str, model: str, client) -> tuple[str, str | None, dict]:
    """Gemini — off goes to each model's floor; high goes to max thinking.
       3.x Pro: 'low' floor / 'high' max.
       3.x Flash: 'minimal' floor / 'high' max.
       2.5 Pro: budget=128 floor / budget=-1 (dynamic, model-decides) for high.
       2.5 Flash: budget=0 floor / budget=-1 for high.
    Temperature 0 is fine with thinking on for Gemini."""
    from google.genai import types
    m = model.lower()
    if REASONING_MODE == "off":
        if m.startswith("gemini-2.5-pro"):
            thinking = types.ThinkingConfig(thinking_budget=128)
        elif m.startswith("gemini-2.5-flash"):
            thinking = types.ThinkingConfig(thinking_budget=0)
        elif "pro" in m:
            thinking = types.ThinkingConfig(thinking_level="low")
        else:
            thinking = types.ThinkingConfig(thinking_level="minimal")
    else:
        if m.startswith("gemini-2.5"):
            thinking = types.ThinkingConfig(thinking_budget=-1)
        else:
            thinking = types.ThinkingConfig(thinking_level="high")
    config = types.GenerateContentConfig(
        temperature=DEFAULT_TEMPERATURE,
        thinking_config=thinking,
    )
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_text(text=prompt)],
        config=config,
    )
    return (
        (resp.text or "").strip(),
        getattr(resp, "response_id", None),
        _gemini_usage(resp),
    )


def build_client(provider: str):
    if provider == "interfaze":
        from openai import OpenAI
        api_key = os.getenv("INTERFAZE_API_KEY")
        if not api_key:
            raise RuntimeError("INTERFAZE_API_KEY missing from .env")
        # Force the production endpoint — the project .env's OPENAI_BASE_URL
        # may point at a dev/staging Cloudflare Worker, which we explicitly do
        # not want for benchmark numbers.
        return OpenAI(base_url="https://api.interfaze.ai/v1", api_key=api_key)
    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing from .env")
        # Force the real OpenAI endpoint — .env's OPENAI_BASE_URL points at interfaze.
        return OpenAI(base_url="https://api.openai.com/v1", api_key=api_key)
    if provider == "anthropic":
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing from .env")
        return Anthropic(api_key=api_key)
    if provider == "gemini":
        from google import genai
        api_key = (
            os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise RuntimeError("GEMINI_KEY missing from .env")
        return genai.Client(api_key=api_key)
    if provider == "openrouter":
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing from .env")
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    raise ValueError(f"unknown provider: {provider}")


def get_call_fn(provider: str):
    return {
        "interfaze": call_interfaze,
        "openai": call_openai,
        "anthropic": call_anthropic,
        "gemini": call_gemini,
        "openrouter": call_openrouter,
    }[provider]


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def process_sample(sample: dict, call_fn, model: str, rate_limiter: RateLimiter,
                         writer: JsonlWriter, progress: dict, provider: str,
                         client) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        question=sample["question"],
        a=sample["a"], b=sample["b"], c=sample["c"], d=sample["d"],
    )
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        start = time.perf_counter()
        try:
            content, request_id, usage = await asyncio.to_thread(call_fn, prompt, model, client)
            latency_ms = int((time.perf_counter() - start) * 1000)
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
                "provider": provider,
                "model": model,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
            }
            await writer.append(record)

            progress["done"] += 1
            if correct:
                progress["correct"] += 1
            if predicted is None:
                progress["unparseable"] += 1
            rt = usage.get("reasoning_tokens")
            ot = usage.get("output_tokens")
            tqdm.write(
                f"[{provider}/{model} {progress['done']}/{progress['total']}] "
                f"{sample['language']} subj={sample['subject'][:18]:18} "
                f"gold={sample['answer']} pred={predicted or '?'} "
                f"{'OK' if correct else 'X '} latency={latency_ms}ms "
                f"reasoning_tok={rt if rt is not None else '?'} out_tok={ot if ot is not None else '?'}"
            )
            return record

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            last_error = f"{type(e).__name__}: {e}"
            tqdm.write(
                f"[{provider}/{model} error] id={sample['id']} attempt={attempt}/{MAX_RETRIES} "
                f"latency={latency_ms}ms error={last_error}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))

    progress["failed"] += 1
    tqdm.write(f"[{provider}/{model} FAILED] id={sample['id']} after {MAX_RETRIES} attempts: {last_error}")
    return None


def load_all_samples(languages: list[str], limit: int | None) -> list[dict]:
    """Loads from openai/MMMLU (full) or opencompass/mmmlu_lite based on
    DATASET_VARIANT global."""
    ds_id = DATASET_LITE_ID if DATASET_VARIANT == "lite" else DATASET_FULL_ID
    all_samples: list[dict] = []
    for lang in languages:
        print(f"Loading {ds_id}/{lang} (split=test)...")
        ds = load_dataset_for_variant(DATASET_VARIANT, lang)
        rows = [
            build_sample_for_variant(DATASET_VARIANT, dict(row), lang, i)
            for i, row in enumerate(ds)
        ]
        if limit is not None:
            rows = rows[:limit]
        all_samples.extend(rows)
    return all_samples


async def run(provider: str, model: str, languages: list[str], pred_path: Path,
              limit: int | None):
    client = build_client(provider)
    call_fn = get_call_fn(provider)

    samples = load_all_samples(languages, limit)
    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    print(f"[{provider}/{model}] Total samples: {len(samples)}")
    print(f"[{provider}/{model}] Resume: {len(done_ids)} done, {len(pending)} pending "
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

    tasks = [process_sample(s, call_fn, model, rate_limiter, writer, progress, provider, client)
             for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc=f"{provider}/{model}")
    except Exception:
        traceback.print_exc()

    acc = progress["correct"] / progress["done"] if progress["done"] else 0.0
    print(
        f"\n[{provider}/{model}] Run finished: "
        f"{progress['done']}/{progress['total']} answered "
        f"({progress['failed']} failed, {progress['unparseable']} unparseable). "
        f"Pooled accuracy on this run: {acc:.4f}"
    )


def run_evaluation(pred_path: Path, metrics_path: Path, provider: str, model: str):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)
    for r in results:
        if r.get("prediction") is None and r.get("response"):
            r["prediction"] = parse_answer(r["response"])
        if r.get("correct") is None and r.get("prediction") is not None:
            r["correct"] = r["prediction"] == r.get("answer")

    metrics = compute_metrics(results)
    _base_print_summary(metrics)
    output = {
        **metrics,
        "dataset": DATASET_ID,
        "split": SPLIT,
        "languages": sorted({r["language"] for r in results}),
        "provider": provider,
        "model": model,
        "rate_limit": RATE_LIMIT,
        "temperature": DEFAULT_TEMPERATURE,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    global REASONING_MODE, DATASET_VARIANT
    parser = argparse.ArgumentParser(description="Multi-provider MMMLU runner")
    parser.add_argument("--provider", required=True,
                        choices=["interfaze", "openai", "anthropic", "gemini", "openrouter"])
    parser.add_argument("--model", required=True, help="Provider-specific model id")
    parser.add_argument("--reasoning", default="off", choices=["off", "high"],
                        help="off = each model at its floor; high = each at max thinking")
    parser.add_argument("--dataset-variant", default="lite", choices=["lite", "full"],
                        help="lite = opencompass/mmmlu_lite (~20k); full = openai/MMMLU (~196k)")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES,
                        help="Subset of languages (default: all 14)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per language (smoke test)")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    REASONING_MODE = args.reasoning
    DATASET_VARIANT = args.dataset_variant
    variant_slug = "lite" if DATASET_VARIANT == "lite" else "full"
    tag = f"mmmlu{variant_slug}_{args.provider}_{model_slug(args.model)}_reasoning{REASONING_MODE}"
    pred_path = RESULTS_DIR / f"{tag}_responses.jsonl"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"

    if args.evaluate_only:
        run_evaluation(pred_path, metrics_path, args.provider, args.model)
    elif args.predict_only:
        asyncio.run(run(args.provider, args.model, args.languages, pred_path, limit=args.limit))
    else:
        asyncio.run(run(args.provider, args.model, args.languages, pred_path, limit=args.limit))
        run_evaluation(pred_path, metrics_path, args.provider, args.model)


if __name__ == "__main__":
    main()
