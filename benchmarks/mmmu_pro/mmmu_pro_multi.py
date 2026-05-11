"""
MMMU Pro benchmark — multi-provider runner.

Two settings, both 10-option MCQ (A-J):
  - standard: text question + up to 7 inline images (image_1..image_7)
  - vision:   single rendered image of the entire question (image)

Methodology mirrors the MMMU Pro paper:
  - pass@1, single attempt, no majority voting
  - reasoning OFF (provider floor) by default; --reasoning high also supported
  - temperature=0 where the API allows it (some providers reject t=0 + thinking)
  - headline numbers are per-setting accuracy; the published "MMMU Pro" score
    is the average of (standard, vision)

Output (per provider+model+setting+reasoning):
  results/mmmupro_<setting>_<provider>_<modelslug>_reasoning<mode>_responses.jsonl
  results/mmmupro_<setting>_<provider>_<modelslug>_reasoning<mode>_metrics.json

Usage:
  uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider gemini \\
      --model gemini-3.1-pro-preview --setting standard
  uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider openai \\
      --model gpt-5.5 --setting vision --limit 5

Env: OPENAI_API_KEY, GEMINI_KEY, ANTHROPIC_API_KEY, INTERFAZE_API_KEY (.env).
"""

import os
import re
import io
import sys
import ast
import json
import time
import base64
import asyncio
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()


def _load_interfaze_env_fallback() -> None:
    path = Path.home() / "interfaze" / ".env.local"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_interfaze_env_fallback()


RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_REPO = "MMMU/MMMU_Pro"
SETTINGS = {
    "standard": "standard (10 options)",
    "vision": "vision",
}
SPLIT = "test"

REASONING_MODE = "off"
DEFAULT_TEMPERATURE = 0.0
RATE_LIMIT = 25
MAX_RETRIES = 3

ANTHROPIC_OFF_MAX_TOKENS = 1024
ANTHROPIC_HIGH_BUDGET_TOKENS = 10_000
ANTHROPIC_HIGH_MAX_TOKENS = 16_000

# Cap any individual image side to keep token cost in check; questions occasionally
# include large diagrams. 1536 keeps detail without paying for >2k-side renders.
MAX_IMAGE_SIDE = 1536

LETTERS_10 = list("ABCDEFGHIJ")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

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
                if rec.get("response") is not None:
                    done.add(str(rec["id"]))
            except json.JSONDecodeError:
                continue
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
                by_id[str(rec["id"])] = rec
            except json.JSONDecodeError:
                continue
    return list(by_id.values())


# ---------------------------------------------------------------------------
# Dataset / sample building
# ---------------------------------------------------------------------------

def parse_options(raw: str) -> list[str]:
    """`options` is a stringified Python list. Use literal_eval (safe)."""
    if isinstance(raw, list):
        return list(raw)
    return list(ast.literal_eval(raw))


def build_standard_sample(row: dict) -> dict:
    images: list[Image.Image] = []
    for i in range(1, 8):
        img = row.get(f"image_{i}")
        if img is not None:
            images.append(img)
    return {
        "id": row["id"],
        "setting": "standard",
        "question": row["question"],
        "options": parse_options(row["options"]),
        "images": images,
        "answer": str(row["answer"]).strip().upper(),
        "subject": row.get("subject"),
        "topic_difficulty": row.get("topic_difficulty"),
    }


def build_vision_sample(row: dict) -> dict:
    return {
        "id": row["id"],
        "setting": "vision",
        "image": row["image"],
        "options": parse_options(row["options"]),
        "answer": str(row["answer"]).strip().upper(),
        "subject": row.get("subject"),
    }


def load_samples(setting: str, limit: int | None) -> list[dict]:
    cfg = SETTINGS[setting]
    print(f"Loading {DATASET_REPO} config={cfg!r} split={SPLIT}...")
    ds = load_dataset(DATASET_REPO, cfg, split=SPLIT)
    rows = list(ds)
    if limit is not None:
        rows = rows[:limit]
    if setting == "standard":
        return [build_standard_sample(dict(r)) for r in rows]
    return [build_vision_sample(dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, MAX_IMAGE_SIDE / max(w, h))
    if scale < 1.0:
        image = image.resize((int(round(w * scale)), int(round(h * scale))))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_options_block(options: list[str]) -> str:
    lines = []
    for letter, opt in zip(LETTERS_10, options):
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


PROMPT_STANDARD = (
    "Answer the following multiple-choice question. The question may reference "
    "images via tags like \"<image 1>\", \"<image 2>\". The corresponding images "
    "are attached in order.\n\n"
    "Respond with ONLY a single letter (A through J) corresponding to the correct "
    "option. Do not explain.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Answer:"
)

PROMPT_VISION = (
    "The attached image renders a multiple-choice question with its options. "
    "Respond with ONLY a single letter (A through J) corresponding to the correct "
    "option. Do not explain.\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"\b([A-J])\b")
_FALLBACK_RE = re.compile(r"[A-J]")


def parse_answer(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if len(s) == 1 and s.upper() in LETTERS_10:
        return s.upper()
    m = _LETTER_RE.search(s.upper())
    if m:
        return m.group(1)
    m = _FALLBACK_RE.search(s.upper())
    if m:
        return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

def _safe_int(x):
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _openai_usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}
    details = getattr(u, "completion_tokens_details", None)
    rt = getattr(details, "reasoning_tokens", None) if details else None
    return {
        "input_tokens": _safe_int(getattr(u, "prompt_tokens", None)),
        "output_tokens": _safe_int(getattr(u, "completion_tokens", None)),
        "reasoning_tokens": _safe_int(rt),
    }


def _gemini_usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}
    return {
        "input_tokens": _safe_int(getattr(u, "prompt_token_count", None)),
        "output_tokens": _safe_int(getattr(u, "candidates_token_count", None)),
        "reasoning_tokens": _safe_int(getattr(u, "thoughts_token_count", None)),
    }


def _anthropic_usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}
    return {
        "input_tokens": _safe_int(getattr(u, "input_tokens", None)),
        "output_tokens": _safe_int(getattr(u, "output_tokens", None)),
        "reasoning_tokens": None,
    }


def _images_for_sample(sample: dict) -> list[Image.Image]:
    if sample["setting"] == "vision":
        return [sample["image"]]
    return sample["images"]


def _build_prompt(sample: dict) -> str:
    if sample["setting"] == "vision":
        return PROMPT_VISION
    return PROMPT_STANDARD.format(
        question=sample["question"],
        options=build_options_block(sample["options"]),
    )


def _openai_style_content(prompt: str, images: list[Image.Image]) -> list[dict]:
    """OpenAI/Interfaze chat content blocks: text + image_url(data URL)."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        b64 = base64.b64encode(image_to_jpeg_bytes(img)).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


def call_interfaze(sample: dict, model: str, client) -> tuple[str, str | None, dict]:
    content = _openai_style_content(_build_prompt(sample), _images_for_sample(sample))
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": DEFAULT_TEMPERATURE,
        "reasoning_effort": "off" if REASONING_MODE == "off" else "high",
    }
    resp = client.chat.completions.create(**kwargs)
    return (
        (resp.choices[0].message.content or "").strip(),
        getattr(resp, "id", None),
        _openai_usage(resp),
    )


def call_openai(sample: dict, model: str, client) -> tuple[str, str | None, dict]:
    content = _openai_style_content(_build_prompt(sample), _images_for_sample(sample))
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
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


def call_anthropic(sample: dict, model: str, client) -> tuple[str, str | None, dict]:
    prompt = _build_prompt(sample)
    parts: list[dict] = []
    for img in _images_for_sample(sample):
        b64 = base64.b64encode(image_to_jpeg_bytes(img)).decode("utf-8")
        parts.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    parts.append({"type": "text", "text": prompt})
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
    }
    if REASONING_MODE == "off":
        kwargs["thinking"] = {"type": "disabled"}
        kwargs["temperature"] = DEFAULT_TEMPERATURE
        kwargs["max_tokens"] = ANTHROPIC_OFF_MAX_TOKENS
    else:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": ANTHROPIC_HIGH_BUDGET_TOKENS}
        kwargs["max_tokens"] = ANTHROPIC_HIGH_MAX_TOKENS
    resp = client.messages.create(**kwargs)
    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(text_parts).strip(), resp.id, _anthropic_usage(resp)


def _openrouter_extra_body(model: str) -> dict:
    """Per-model OpenRouter extras. See benchmarks.mmmlu.mmmlu_multi for details.
    Vision-capable models we currently route: x-ai/grok-4.3, moonshotai/kimi-k2.6."""
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


def call_openrouter(sample: dict, model: str, client) -> tuple[str, str | None, dict]:
    """OpenRouter dispatch (vision-aware). Vision: pass images as data: URLs
    (OpenAI-shaped). See _openrouter_extra_body for per-model knobs."""
    content = _openai_style_content(_build_prompt(sample), _images_for_sample(sample))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=DEFAULT_TEMPERATURE,
        extra_body=_openrouter_extra_body(model),
    )
    return (
        (resp.choices[0].message.content or "").strip(),
        getattr(resp, "id", None),
        _openai_usage(resp),
    )


def call_gemini(sample: dict, model: str, client) -> tuple[str, str | None, dict]:
    from google.genai import types
    prompt = _build_prompt(sample)
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
    contents: list = []
    for img in _images_for_sample(sample):
        contents.append(types.Part.from_bytes(
            data=image_to_jpeg_bytes(img), mime_type="image/jpeg"
        ))
    contents.append(prompt)
    resp = client.models.generate_content(model=model, contents=contents, config=config)
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
            raise RuntimeError("INTERFAZE_API_KEY missing")
        return OpenAI(base_url="https://api.interfaze.ai/v1", api_key=api_key)
    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return OpenAI(base_url="https://api.openai.com/v1", api_key=api_key)
    if provider == "anthropic":
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        return Anthropic(api_key=api_key)
    if provider == "gemini":
        from google import genai
        api_key = os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_KEY missing")
        return genai.Client(api_key=api_key)
    if provider == "openrouter":
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing")
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
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        start = time.perf_counter()
        try:
            content, request_id, usage = await asyncio.to_thread(call_fn, sample, model, client)
            latency_ms = int((time.perf_counter() - start) * 1000)
            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            predicted = parse_answer(content)
            correct = predicted == sample["answer"]

            record = {
                "id": sample["id"],
                "setting": sample["setting"],
                "subject": sample.get("subject"),
                "topic_difficulty": sample.get("topic_difficulty"),
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
                f"[{provider}/{model} {sample['setting']} {progress['done']}/{progress['total']}] "
                f"id={sample['id']} subj={(sample.get('subject') or '')[:18]:18} "
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


async def run(provider: str, model: str, setting: str, pred_path: Path, limit: int | None):
    client = build_client(provider)
    call_fn = get_call_fn(provider)

    samples = load_samples(setting, limit)
    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    print(f"[{provider}/{model} {setting}] Total: {len(samples)}, "
          f"resume: {len(done_ids)} done / {len(pending)} pending "
          f"(checkpoint: {pred_path})")
    if not pending:
        return

    writer = JsonlWriter(pred_path)
    rate_limiter = RateLimiter(RATE_LIMIT)
    progress = {"total": len(pending), "done": 0, "correct": 0, "unparseable": 0, "failed": 0}

    tasks = [process_sample(s, call_fn, model, rate_limiter, writer, progress, provider, client)
             for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc=f"{provider}/{model}/{setting}")
    except Exception:
        traceback.print_exc()

    acc = progress["correct"] / progress["done"] if progress["done"] else 0.0
    print(
        f"\n[{provider}/{model} {setting}] Run finished: "
        f"{progress['done']}/{progress['total']} answered "
        f"({progress['failed']} failed, {progress['unparseable']} unparseable). "
        f"Pooled accuracy on this run: {acc:.4f}"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    by_subject: dict[str, list[dict]] = defaultdict(list)
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("subject") is not None:
            by_subject[r["subject"]].append(r)
        if r.get("topic_difficulty") is not None:
            by_difficulty[r["topic_difficulty"]].append(r)

    n_total = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    n_unparseable = sum(1 for r in results if r.get("prediction") is None)

    per_subject = {}
    for subj, rows in by_subject.items():
        n = len(rows)
        c = sum(1 for r in rows if r.get("correct"))
        per_subject[subj] = {"n": n, "accuracy": c / n if n else 0.0}

    per_difficulty = {}
    for diff, rows in by_difficulty.items():
        n = len(rows)
        c = sum(1 for r in rows if r.get("correct"))
        per_difficulty[diff] = {"n": n, "accuracy": c / n if n else 0.0}

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
        "accuracy": n_correct / n_total if n_total else 0.0,
        "num_samples": n_total,
        "unparseable": n_unparseable,
        "per_subject": per_subject,
        "per_difficulty": per_difficulty,
        "latency": latency_stats,
    }


def print_summary(metrics: dict, setting: str, provider: str, model: str):
    print(f"\n{'=' * 68}")
    print(f"MMMU Pro [{setting}] — {provider}/{model} (reasoning={REASONING_MODE}, temp={DEFAULT_TEMPERATURE})")
    print(f"{'=' * 68}")
    print(f"Samples              : {metrics['num_samples']}")
    print(f"Accuracy             : {metrics['accuracy']:.4f}")
    print(f"Unparseable          : {metrics['unparseable']}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(f"Latency              : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
              f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms")


def run_evaluation(pred_path: Path, metrics_path: Path, provider: str, model: str, setting: str):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        return
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        return
    for r in results:
        if r.get("prediction") is None and r.get("response"):
            r["prediction"] = parse_answer(r["response"])
        if r.get("correct") is None and r.get("prediction") is not None:
            r["correct"] = r["prediction"] == r.get("answer")

    metrics = compute_metrics(results)
    print_summary(metrics, setting, provider, model)
    output = {
        **metrics,
        "dataset": DATASET_REPO,
        "config": SETTINGS[setting],
        "setting": setting,
        "split": SPLIT,
        "provider": provider,
        "model": model,
        "reasoning_mode": REASONING_MODE,
        "rate_limit": RATE_LIMIT,
        "temperature": DEFAULT_TEMPERATURE,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    global REASONING_MODE
    parser = argparse.ArgumentParser(description="MMMU Pro multi-provider runner")
    parser.add_argument("--provider", required=True,
                        choices=["interfaze", "openai", "anthropic", "gemini", "openrouter"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--setting", required=True, choices=list(SETTINGS.keys()))
    parser.add_argument("--reasoning", default="off", choices=["off", "high"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples (smoke test)")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    REASONING_MODE = args.reasoning
    tag = f"mmmupro_{args.setting}_{args.provider}_{model_slug(args.model)}_reasoning{REASONING_MODE}"
    pred_path = RESULTS_DIR / f"{tag}_responses.jsonl"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"

    if args.evaluate_only:
        run_evaluation(pred_path, metrics_path, args.provider, args.model, args.setting)
    elif args.predict_only:
        asyncio.run(run(args.provider, args.model, args.setting, pred_path, limit=args.limit))
    else:
        asyncio.run(run(args.provider, args.model, args.setting, pred_path, limit=args.limit))
        run_evaluation(pred_path, metrics_path, args.provider, args.model, args.setting)


if __name__ == "__main__":
    main()
