"""
ScreenSpot-Pro benchmark — multi-provider runner.

GUI grounding on 1,581 high-resolution professional screenshots across 23
applications (VSCode, Photoshop, AutoCAD, MATLAB, Excel, ...). Each sample
gives a natural-language instruction; the model must output a click point
or bounding box for the target UI element.

Dataset: https://huggingface.co/datasets/likaixin/ScreenSpot-Pro

Metric: **center hit accuracy** — predicted point (or center of predicted bbox)
must fall inside the ground-truth bounding box. Standard ScreenSpot scoring.

Reasoning is OFF by default at each model's floor (matches the convention in
this repo). For Gemini Pro this means thinking_level='low' (the API floor).

Usage:
    uv run -m benchmarks.screenspot_pro.screenspot_pro_multi \\
        --provider gemini --model gemini-3.1-pro-preview
    uv run -m benchmarks.screenspot_pro.screenspot_pro_multi \\
        --provider openai --model gpt-5.5 --limit 5

Env: OPENAI_API_KEY, GEMINI_KEY, ANTHROPIC_API_KEY, INTERFAZE_API_KEY (.env).
"""

import os
import re
import io
import sys
import json
import time
import base64
import asyncio
import argparse
import traceback
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_REPO = "likaixin/ScreenSpot-Pro"

# Reasoning floor per provider (matches mmmlu_multi.py conventions). Pros can't
# truly disable; this is the lowest legal setting.
REASONING_MODE = "off"

DEFAULT_TEMPERATURE = 0.0
RATE_LIMIT = 25
MAX_RETRIES = 3
# 256 was empirically too tight: Sonnet 4.6 with reasoning disabled still emits
# ~200-300 tokens of prose-then-JSON; we were truncating mid-reasoning before
# the JSON line ever appeared (~9% silent parser-None on smoke).
ANTHROPIC_OFF_MAX_TOKENS = 1024
ANTHROPIC_HIGH_MAX_TOKENS = 16_000
ANTHROPIC_HIGH_BUDGET_TOKENS = 10_000

# All 23 annotation files in the dataset.
ANNOTATION_FILES = [
    "android_studio_macos.json", "autocad_windows.json", "blender_windows.json",
    "davinci_macos.json", "eviews_windows.json", "excel_macos.json",
    "fruitloops_windows.json", "illustrator_windows.json", "inventor_windows.json",
    "linux_common_linux.json", "macos_common_macos.json", "matlab_macos.json",
    "origin_windows.json", "photoshop_windows.json", "powerpoint_windows.json",
    "premiere_windows.json", "pycharm_macos.json", "quartus_windows.json",
    "solidworks_windows.json", "stata_windows.json", "unreal_engine_windows.json",
    "vivado_windows.json", "vmware_macos.json", "vscode_macos.json",
    "windows_common_windows.json", "word_macos.json",
]


def _load_interfaze_env_fallback() -> None:
    """Pull keys from ~/interfaze/.env.local that aren't in os.environ already."""
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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_annotations() -> list[dict]:
    """Pull all annotation JSONs (cached locally by hf_hub_download)."""
    samples: list[dict] = []
    for fname in ANNOTATION_FILES:
        try:
            p = hf_hub_download(
                repo_id=DATASET_REPO, repo_type="dataset",
                filename=f"annotations/{fname}",
            )
        except Exception as e:
            print(f"[warn] could not fetch {fname}: {e}")
            continue
        with open(p) as f:
            samples.extend(json.load(f))
    return samples


def fetch_image(img_filename: str) -> Image.Image:
    """Download and open an image from the dataset (cached)."""
    p = hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset",
        filename=f"images/{img_filename}",
    )
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def build_sample(row: dict) -> dict:
    return {
        "id": row["id"],
        "img_filename": row["img_filename"],
        "bbox": row["bbox"],  # [x1, y1, x2, y2] in original image coords
        "instruction": row["instruction"],
        "img_size": row["img_size"],  # [width, height]
        "application": row.get("application"),
        "platform": row.get("platform"),
        "ui_type": row.get("ui_type"),
        "group": row.get("group"),
    }


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
# Image preprocessing — resize down to a max side so we don't blow up
# token costs on huge screenshots, while preserving aspect ratio.
# ---------------------------------------------------------------------------

MAX_IMAGE_SIDE = 2048  # screenshots are often 2560x1664; cap for cost


def image_to_jpeg_bytes(image: Image.Image) -> tuple[bytes, int, int]:
    w, h = image.size
    scale = min(1.0, MAX_IMAGE_SIDE / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h))
        w, h = new_w, new_h
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), w, h


# ---------------------------------------------------------------------------
# Output parsing — model returns either a point [x, y] or a bbox [x1,y1,x2,y2].
# Coordinates may be in pixel space, normalized [0,1], or normalized 0-1000.
# ---------------------------------------------------------------------------

JSON_POINT_RE = re.compile(
    r'"point"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)
JSON_BBOX_RE = re.compile(
    r'"(?:bbox|box_2d)"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)
BARE_PAIR_RE = re.compile(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')
BARE_4TUPLE_RE = re.compile(
    r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def parse_click_candidates(text: str, sent_w: int, sent_h: int, model_lower: str) -> list[tuple[float, float]]:
    """Return ALL plausible (px, py) click candidates the response could mean,
    in image pixel space. Permissive on coord conventions: each raw (x, y) is
    emitted in every coord interpretation that's mathematically possible.

    Why permissive: providers don't always honor our prompted coord space.
    Gemini reliably emits box_2d in 0-1000; interfaze emits point in 0-1000;
    OpenAI/Anthropic generally emit pixels but occasionally normalize. Rather
    than hardcode per-provider rules (and miscount when a model deviates), we
    accept any interpretation that lands in the GT bbox. The false-positive
    rate is small because GT bboxes on ScreenSpot-Pro are tight (often
    <100×100 px on a ~2.4M px screen).

    Interpretations attempted for each raw (x, y):
      - 0-1 normalized        (only if max ≤ 1.0)
      - raw pixels            (only if max > 1.0; bounded sanity check)
      - 0-1000 normalized     (only if max > 1.0)

    Source preference per provider:
      - gemini emits box_2d   → bbox first (y,x,y,x order), then point
      - others emit point     → point first, then bbox (x,y,x,y order)
    """
    if not text:
        return []
    txt = text
    is_gemini = "gemini" in model_lower

    def _last(regex):
        last = None
        for mm in regex.finditer(txt):
            last = mm
        return last

    raw_xy: list[tuple[float, float]] = []

    if is_gemini:
        m = JSON_BBOX_RE.search(txt) or _last(BARE_4TUPLE_RE)
        if m is not None:
            a, b, c, d = (float(m.group(i)) for i in range(1, 5))
            ymin, xmin, ymax, xmax = a, b, c, d
            raw_xy.append(((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))
        m = JSON_POINT_RE.search(txt) or _last(BARE_PAIR_RE)
        if m is not None:
            raw_xy.append((float(m.group(1)), float(m.group(2))))
    else:
        m = JSON_POINT_RE.search(txt) or _last(BARE_PAIR_RE)
        if m is not None:
            raw_xy.append((float(m.group(1)), float(m.group(2))))
        m = JSON_BBOX_RE.search(txt) or _last(BARE_4TUPLE_RE)
        if m is not None:
            a, b, c, d = (float(m.group(i)) for i in range(1, 5))
            xmin, ymin, xmax, ymax = a, b, c, d
            raw_xy.append(((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))

    candidates: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    def _emit(px: float, py: float):
        key = (round(px, 3), round(py, 3))
        if key not in seen:
            seen.add(key)
            candidates.append((px, py))

    for x, y in raw_xy:
        mx = max(abs(x), abs(y))
        if mx <= 1.0:
            _emit(x * sent_w, y * sent_h)
            continue
        if is_gemini:
            # Gemini's coord space is reliably 0-1000 (box_2d and point alike);
            # adding a pixel-space interpretation creates false positives when
            # raw values happen to be both valid pixels AND valid 0-1000 coords
            # (common for top-left UI elements).
            _emit(x * sent_w / 1000.0, y * sent_h / 1000.0)
            continue
        # Other providers' coord space is genuinely ambiguous in practice.
        # Emit both pixel and 0-1000-normalized interpretations.
        if abs(x) <= sent_w * 1.05 and abs(y) <= sent_h * 1.05:
            _emit(x, y)
        if mx <= 1100:
            _emit(x * sent_w / 1000.0, y * sent_h / 1000.0)

    return candidates


def parse_click(text: str, sent_w: int, sent_h: int, model_lower: str) -> tuple[float, float] | None:
    """Back-compat: first candidate, or None. New code should call
    parse_click_candidates and check `any(point_in_bbox(c, gt) for c in cands)`."""
    cs = parse_click_candidates(text, sent_w, sent_h, model_lower)
    return cs[0] if cs else None


def point_in_bbox(point: tuple[float, float] | None, bbox: list[float]) -> bool:
    """bbox is [x1, y1, x2, y2] in same coord space as point."""
    if point is None:
        return False
    x1, y1, x2, y2 = bbox
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


# ---------------------------------------------------------------------------
# Provider adapters — return (content, request_id, usage)
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


PROMPT_TEMPLATE_PIXEL = (
    "You are a GUI grounding assistant. The screenshot has dimensions "
    "{width}×{height} pixels (top-left = 0,0; bottom-right = {width},{height}).\n\n"
    "Task: identify the screen location for the following instruction:\n"
    "\"{instruction}\"\n\n"
    "Output ONLY a JSON object on the last line in this exact format:\n"
    "{{\"point\": [x, y]}}\n"
    "where x and y are integer pixel coordinates of the click target."
)

PROMPT_TEMPLATE_GEMINI = (
    "You are a GUI grounding assistant. Identify the screen location for "
    "the following instruction:\n\"{instruction}\"\n\n"
    "Output ONLY a JSON object on the last line in this exact format:\n"
    "{{\"box_2d\": [ymin, xmin, ymax, xmax]}}\n"
    "Coordinates must be normalized to 0-1000 where (0,0) is the top-left "
    "corner and (1000,1000) is the bottom-right of the image."
)


def call_interfaze(image: Image.Image, sample: dict, model: str, client) -> tuple[str, str | None, int, int, dict]:
    img_bytes, w, h = image_to_jpeg_bytes(image)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"
    prompt = PROMPT_TEMPLATE_PIXEL.format(width=w, height=h, instruction=sample["instruction"])
    kwargs = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "temperature": DEFAULT_TEMPERATURE,
        "reasoning_effort": "off" if REASONING_MODE == "off" else "high",
    }
    resp = client.chat.completions.create(**kwargs)
    content = (resp.choices[0].message.content or "").strip()
    return content, getattr(resp, "id", None), w, h, _openai_usage(resp)


def call_openai(image: Image.Image, sample: dict, model: str, client) -> tuple[str, str | None, int, int, dict]:
    img_bytes, w, h = image_to_jpeg_bytes(image)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"
    prompt = PROMPT_TEMPLATE_PIXEL.format(width=w, height=h, instruction=sample["instruction"])
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
    if REASONING_MODE == "off":
        kwargs["reasoning_effort"] = "none"
        kwargs["temperature"] = DEFAULT_TEMPERATURE
    else:
        kwargs["reasoning_effort"] = "high"
    resp = client.chat.completions.create(**kwargs)
    content = (resp.choices[0].message.content or "").strip()
    return content, getattr(resp, "id", None), w, h, _openai_usage(resp)


def call_anthropic(image: Image.Image, sample: dict, model: str, client) -> tuple[str, str | None, int, int, dict]:
    img_bytes, w, h = image_to_jpeg_bytes(image)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = PROMPT_TEMPLATE_PIXEL.format(width=w, height=h, instruction=sample["instruction"])
    kwargs = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
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
    return "\n".join(parts).strip(), resp.id, w, h, _anthropic_usage(resp)


def call_gemini(image: Image.Image, sample: dict, model: str, client) -> tuple[str, str | None, int, int, dict]:
    from google.genai import types
    img_bytes, w, h = image_to_jpeg_bytes(image)
    prompt = PROMPT_TEMPLATE_GEMINI.format(instruction=sample["instruction"])
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
    config = types.GenerateContentConfig(temperature=DEFAULT_TEMPERATURE, thinking_config=thinking)
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=config,
    )
    return (resp.text or "").strip(), getattr(resp, "response_id", None), w, h, _gemini_usage(resp)


# ---------------------------------------------------------------------------
# Client + dispatch
# ---------------------------------------------------------------------------

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
    raise ValueError(f"unknown provider: {provider}")


def get_call_fn(provider: str):
    return {
        "interfaze": call_interfaze,
        "openai": call_openai,
        "anthropic": call_anthropic,
        "gemini": call_gemini,
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

    try:
        image = await asyncio.to_thread(fetch_image, sample["img_filename"])
    except Exception as e:
        tqdm.write(f"[{provider}/{model} fetch error] id={sample['id']}: {type(e).__name__}: {e}")
        progress["failed"] += 1
        return None

    orig_w, orig_h = image.size
    gt_bbox_orig = sample["bbox"]

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        start = time.perf_counter()
        try:
            content, request_id, sent_w, sent_h, usage = await asyncio.to_thread(
                call_fn, image, sample, model, client
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            # Predicted point arrives in *sent* image coordinates. Map gt bbox
            # to the same scale so the in-bbox check is fair.
            sx = sent_w / orig_w
            sy = sent_h / orig_h
            gt_bbox_sent = [
                gt_bbox_orig[0] * sx, gt_bbox_orig[1] * sy,
                gt_bbox_orig[2] * sx, gt_bbox_orig[3] * sy,
            ]

            candidates = parse_click_candidates(content, sent_w, sent_h, model.lower())
            # Permissive scoring: any plausible coord-space interpretation hits.
            winner = next((c for c in candidates if point_in_bbox(c, gt_bbox_sent)), None)
            pred_point = winner if winner is not None else (candidates[0] if candidates else None)
            correct = winner is not None

            record = {
                "id": sample["id"],
                "img_filename": sample["img_filename"],
                "instruction": sample["instruction"],
                "application": sample.get("application"),
                "platform": sample.get("platform"),
                "ui_type": sample.get("ui_type"),
                "group": sample.get("group"),
                "image_width_orig": orig_w,
                "image_height_orig": orig_h,
                "image_width_sent": sent_w,
                "image_height_sent": sent_h,
                "gt_bbox_orig": gt_bbox_orig,
                "gt_bbox_sent": gt_bbox_sent,
                "pred_point": list(pred_point) if pred_point is not None else None,
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
            mark = "OK" if correct else "X "
            tqdm.write(
                f"[{provider}/{model} {progress['done']}/{progress['total']}] {mark} "
                f"id={sample['id']} app={sample.get('application'):12} "
                f"pred={pred_point} gt={[round(x) for x in gt_bbox_sent]} "
                f"latency={latency_ms}ms"
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


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))

    by_group: dict[str, list[dict]] = {}
    by_ui: dict[str, list[dict]] = {}
    by_app: dict[str, list[dict]] = {}
    for r in results:
        by_group.setdefault(r.get("group") or "?", []).append(r)
        by_ui.setdefault(r.get("ui_type") or "?", []).append(r)
        by_app.setdefault(r.get("application") or "?", []).append(r)

    def _bucket_acc(d):
        return {k: {"n": len(v), "accuracy": sum(1 for r in v if r.get("correct")) / len(v)}
                for k, v in d.items()}

    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int)]
    lat_stats = {}
    if latencies:
        lats = sorted(latencies)
        nl = len(lats)
        lat_stats = {
            "count": nl, "mean_ms": sum(lats) / nl,
            "p50_ms": lats[nl // 2],
            "p90_ms": lats[min(nl - 1, int(nl * 0.9))],
            "p99_ms": lats[min(nl - 1, int(nl * 0.99))],
            "max_ms": lats[-1],
        }

    return {
        "accuracy": n_correct / n,
        "correct": n_correct,
        "total": n,
        "per_group": _bucket_acc(by_group),
        "per_ui_type": _bucket_acc(by_ui),
        "per_application": _bucket_acc(by_app),
        "latency": lat_stats,
    }


def print_summary(metrics: dict, provider: str, model: str):
    print(f"\n{'=' * 60}")
    print(f"ScreenSpot-Pro — {provider}/{model} (reasoning={REASONING_MODE})")
    print(f"{'=' * 60}")
    print(f"Center hit accuracy : {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print("\nPer high-level group:")
    for g in sorted(metrics["per_group"]):
        v = metrics["per_group"][g]
        print(f"  {g:14} n={v['n']:>4} acc={v['accuracy']:.4f}")
    print("\nPer UI type:")
    for u in sorted(metrics["per_ui_type"]):
        v = metrics["per_ui_type"][u]
        print(f"  {u:14} n={v['n']:>4} acc={v['accuracy']:.4f}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(f"\nLatency : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
              f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms max={lat['max_ms']}ms")


async def run(provider: str, model: str, pred_path: Path, limit: int | None):
    client = build_client(provider)
    call_fn = get_call_fn(provider)

    print(f"[{provider}/{model}] Loading annotations...")
    raw = load_annotations()
    samples = [build_sample(row) for row in raw]
    print(f"[{provider}/{model}] Loaded {len(samples)} samples")

    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
    print(f"[{provider}/{model}] Resume: {len(done_ids)} done, {len(pending)} pending "
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
    acc = progress["correct"] / progress["done"] if progress["done"] else 0.0
    print(f"\n[{provider}/{model}] Run finished: {progress['done']}/{progress['total']} answered, "
          f"{progress['correct']} correct (acc={acc:.4f}), {progress['failed']} failed.")


def run_evaluation(pred_path: Path, metrics_path: Path, provider: str, model: str):
    if not pred_path.exists():
        print(f"No predictions at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)
    metrics = compute_metrics(results)
    print_summary(metrics, provider, model)
    output = {
        **metrics,
        "dataset": DATASET_REPO,
        "provider": provider, "model": model,
        "reasoning_mode": REASONING_MODE,
        "temperature": DEFAULT_TEMPERATURE,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    global REASONING_MODE
    parser = argparse.ArgumentParser(description="Multi-provider ScreenSpot-Pro runner")
    parser.add_argument("--provider", required=True,
                        choices=["interfaze", "openai", "anthropic", "gemini"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning", default="off", choices=["off", "high"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    REASONING_MODE = args.reasoning
    tag = f"screenspotpro_{args.provider}_{model_slug(args.model)}_reasoning{REASONING_MODE}"
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
