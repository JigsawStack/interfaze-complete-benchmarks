"""
RefCOCO benchmark against multiple VLM providers (OpenAI, Anthropic, Google).

Designed for head-to-head comparison with the interfaze-beta numbers from
`refcoco.py` on the same splits (Acc@IoU=0.5). Each provider runs with
thinking/reasoning DISABLED to match the interfaze run's `reasoning_effort=None`.

Usage:
    # Single-model run:
    uv run -m benchmarks.obj_detection.refcoco_multi --provider openai --model gpt-5.4
    uv run -m benchmarks.obj_detection.refcoco_multi --provider anthropic --model claude-sonnet-4-6
    uv run -m benchmarks.obj_detection.refcoco_multi --provider gemini --model gemini-3.0-flash

    # Smoke test with --limit 1:
    uv run -m benchmarks.obj_detection.refcoco_multi --provider openai --model gpt-5.4 --limit 1

Keys are loaded from ~/interfaze/.env.local (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_KEY).
"""

import sys
import json
import time
import base64
import asyncio
import argparse
import traceback
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse parsing/IoU from the interfaze script — guarantees identical scoring.
from benchmarks.obj_detection.refcoco import (  # noqa: E402
    parse_box, compute_iou, coco_bbox_to_xyxy,
    JsonlWriter, build_samples, load_completed_ids, load_records,
    compute_metrics, print_summary, PROMPT_TEMPLATE, IOU_THRESHOLD,
)


class RateLimiter:
    """Simple async token-bucket. Local to this script — refcoco.py does not
    export one, so we don't try to import it."""
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

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATASET = "lmms-lab/RefCOCO"
DEFAULT_SPLIT = "testA"
RATE_LIMIT = 25
MAX_RETRIES = 3


def _load_interfaze_env() -> dict:
    """Parse ~/interfaze/.env.local and return a dict of keys."""
    path = Path.home() / "interfaze" / ".env.local"
    if not path.exists():
        raise FileNotFoundError(f"Expected keys at {path}")
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        env[k.strip()] = v
    return env


# --------------------------------------------------------------------------
# Provider adapters
# --------------------------------------------------------------------------
# Each adapter is a function: (image_pil, prompt, model) -> (content, request_id)
# Runs synchronously inside asyncio.to_thread.

def _image_to_jpeg_bytes(image, max_side: int = 1024) -> tuple[bytes, int, int]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h))
        w, h = new_w, new_h
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), w, h


def call_openai(image, prompt: str, model: str, client) -> tuple[str, str, int, int]:
    """Returns (content, request_id, width, height).

    For GPT-5.4 (and 5.2+) reasoning is OFF by default ("none"). We still pass
    it explicitly to defend against future default changes. NOTE: "minimal"
    is NOT a valid value on GPT-5.4 — valid are none/low/medium/high/xhigh.
    """
    img_bytes, w, h = _image_to_jpeg_bytes(image)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"
    kwargs = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    }
    if model.startswith("gpt-5") or model.startswith("o"):
        kwargs["reasoning_effort"] = "none"
    resp = client.chat.completions.create(**kwargs)
    content = (resp.choices[0].message.content or "").strip()
    return content, resp.id, w, h


def call_anthropic(image, prompt: str, model: str, client) -> tuple[str, str, int, int]:
    """Extended thinking explicitly disabled — belt-and-suspenders even though
    omitting `thinking` is off by default for Sonnet 4.6."""
    img_bytes, w, h = _image_to_jpeg_bytes(image)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        thinking={"type": "disabled"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    content = "\n".join(parts).strip()
    return content, resp.id, w, h


def call_gemini(image, prompt: str, model: str, client) -> tuple[str, str, int, int]:
    """Gemini 3.x has thinking ON by default (HIGH for Pro, can't disable).
    We always override to the LOWEST supported level for the chosen model:
       gemini-3*-pro*: 'low'     (Pro rejects 'minimal'; min is 'low')
       gemini-3*-flash*: 'minimal' (Flash supports 'minimal' as the floor)
    Gemini 2.5 used thinking_budget; we don't support that path here."""
    from google.genai import types
    img_bytes, w, h = _image_to_jpeg_bytes(image)
    m = model.lower()
    if "pro" in m:
        thinking_level = "low"
    else:
        thinking_level = "minimal"
    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=config,
    )
    content = (resp.text or "").strip()
    request_id = getattr(resp, "response_id", None) or ""
    return content, request_id, w, h


# --------------------------------------------------------------------------
# Provider-agnostic pipeline (mirrors refcoco.py's process_sample)
# --------------------------------------------------------------------------

async def process_sample(sample: dict, call_fn, model: str, rate_limiter,
                         writer: JsonlWriter, progress: dict, provider: str,
                         client) -> dict | None:
    """Pre-resize the image to know the exact dims that will be sent, embed
    those dims in the prompt, then call the provider adapter."""
    tmp_bytes, sent_w, sent_h = _image_to_jpeg_bytes(sample["image"])
    del tmp_bytes
    orig_w, orig_h = sample["image"].size
    sx = sent_w / orig_w
    sy = sent_h / orig_h
    gt_xyxy = coco_bbox_to_xyxy(sample["bbox_xywh"], sx, sy)

    prompt = PROMPT_TEMPLATE.format(
        width=sent_w, height=sent_h, expression=sample["expression"]
    )

    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        start = time.perf_counter()
        try:
            content, request_id, _, _ = await asyncio.to_thread(
                call_fn, sample["image"], prompt, model, client
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            pred_box = parse_box(content, sent_w, sent_h)
            iou = compute_iou(pred_box, gt_xyxy) if pred_box else 0.0
            correct = pred_box is not None and iou >= IOU_THRESHOLD

            record = {
                "id": sample["id"],
                "question_id": sample["question_id"],
                "file_name": sample["file_name"],
                "expression": sample["expression"],
                "all_expressions": sample["all_expressions"],
                "image_width": sent_w,
                "image_height": sent_h,
                "gt_bbox_xyxy": gt_xyxy,
                "pred_bbox_xyxy": pred_box,
                "iou": iou,
                "correct": correct,
                "response": content,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "attempts": attempt,
                "provider": provider,
                "model": model,
            }
            await writer.append(record)
            progress["done"] += 1
            if correct:
                progress["correct"] += 1
                mark = "OK"
            else:
                mark = "X "
            tqdm.write(
                f"[{provider} {progress['done']}/{progress['total']}] {mark} "
                f"id={sample['id']} iou={iou:.3f} "
                f"pred={pred_box} gt={[round(x,1) for x in gt_xyxy]} "
                f"latency={latency_ms}ms req_id={request_id} attempt={attempt}"
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_tag(dataset: str, split: str, provider: str, model: str) -> str:
    ds_slug = dataset.split("/")[-1].lower().replace("+", "plus")
    model_slug = model.replace("/", "_").replace(":", "_")
    return f"{ds_slug}_{split}_{provider}_{model_slug}"


def build_client(provider: str, env: dict):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=env["OPENAI_API_KEY"])
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=env["ANTHROPIC_API_KEY"])
    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=env["GEMINI_KEY"])
    raise ValueError(f"unknown provider: {provider}")


def get_call_fn(provider: str):
    return {
        "openai": call_openai,
        "anthropic": call_anthropic,
        "gemini": call_gemini,
    }[provider]


async def run(provider: str, model: str, dataset_name: str, split: str,
              pred_path: Path, limit: int | None):
    env = _load_interfaze_env()
    client = build_client(provider, env)
    call_fn = get_call_fn(provider)

    print(f"[{provider}/{model}] Loading {dataset_name}, split={split}...")
    dataset = load_dataset(dataset_name, split=split)
    samples = build_samples(dataset)

    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
        print(f"--limit applied: will run at most {limit} sample(s)")
    print(f"Resume: {len(done_ids)} already completed, {len(pending)} remaining "
          f"(checkpoint: {pred_path})")
    if not pending:
        return

    writer = JsonlWriter(pred_path)
    rate_limiter = RateLimiter(RATE_LIMIT)
    progress = {"total": len(pending), "done": 0, "correct": 0, "failed": 0}
    tasks = [process_sample(s, call_fn, model, rate_limiter, writer, progress, provider, client)
             for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc=f"{provider}/{model}")
    except Exception:
        traceback.print_exc()
    print(f"\n[{provider}/{model}] Run finished: {progress['done']}/{progress['total']} answered, "
          f"{progress['correct']} correct, {progress['failed']} failed.")


def run_evaluation(dataset_name: str, split: str, pred_path: Path, metrics_path: Path,
                   provider: str, model: str):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)
    metrics = compute_metrics(results)
    print_summary(metrics, dataset_name, split)
    output = {
        **metrics,
        "dataset": dataset_name, "split": split,
        "provider": provider, "model": model,
        "rate_limit": RATE_LIMIT, "iou_threshold": IOU_THRESHOLD,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-provider RefCOCO eval")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini"])
    parser.add_argument("--model", required=True, help="Provider-specific model id")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tag = build_tag(args.dataset, args.split, args.provider, args.model)
    pred_path = RESULTS_DIR / f"{tag}_responses.jsonl"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"

    if args.evaluate_only:
        run_evaluation(args.dataset, args.split, pred_path, metrics_path,
                       args.provider, args.model)
    elif args.predict_only:
        asyncio.run(run(args.provider, args.model, args.dataset, args.split,
                        pred_path, limit=args.limit))
    else:
        asyncio.run(run(args.provider, args.model, args.dataset, args.split,
                        pred_path, limit=args.limit))
        run_evaluation(args.dataset, args.split, pred_path, metrics_path,
                       args.provider, args.model)


if __name__ == "__main__":
    main()
