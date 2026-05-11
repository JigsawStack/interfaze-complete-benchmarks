"""One-off probe: run the single VoxPopuli-AA sample that all three Gemini runs
dropped, and dump the raw response (text + finish reason + prompt/safety
feedback) so we can see why it was rejected."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from google import genai
from google.genai import types

from benchmarks.asr.voxpopuli_aa import PROMPT, DATASET_ID, SPLIT, build_sample, fetch_audio_bytes
from benchmarks.asr.voxpopuli_aa_multi import _load_interfaze_env

TARGET_ID = "20150527-0900-PLENARY-14-en_20150527-21:41:31_1"
MODELS = ["gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3.1-pro-preview"]


def thinking_for(model: str) -> types.ThinkingConfig:
    m = model.lower()
    if m.startswith("gemini-2.5-pro"):
        return types.ThinkingConfig(thinking_budget=128)
    if m.startswith("gemini-2.5-flash"):
        return types.ThinkingConfig(thinking_budget=0)
    if "pro" in m:
        return types.ThinkingConfig(thinking_level="low")
    return types.ThinkingConfig(thinking_level="minimal")


def main():
    env = _load_interfaze_env()
    client = genai.Client(api_key=env["GEMINI_KEY"])

    print(f"Loading dataset {DATASET_ID} split={SPLIT}...")
    ds = load_dataset(DATASET_ID, split=SPLIT)
    sample = None
    for row in ds:
        s = build_sample(dict(row))
        if s["id"] == TARGET_ID:
            sample = s
            break
    if sample is None:
        print(f"!! sample {TARGET_ID} not found in dataset"); sys.exit(1)

    print(f"\nSample: id={sample['id']} dur={sample['duration']}s lang={sample['language']}")
    print(f"GT transcript: {sample['transcript']!r}\n")

    audio_bytes = fetch_audio_bytes(sample["file_name"])
    print(f"Audio fetched: {len(audio_bytes)} bytes\n")

    for model in MODELS:
        print("=" * 80)
        print(f"MODEL: {model}")
        print("=" * 80)
        config = types.GenerateContentConfig(
            thinking_config=thinking_for(model),
            temperature=0.0,
        )
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    PROMPT,
                ],
                config=config,
            )
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}\n")
            continue

        text = (resp.text or "").strip() if hasattr(resp, "text") else ""
        print(f"  resp.text         : {text!r}")
        print(f"  response_id       : {getattr(resp, 'response_id', None)}")

        pf = getattr(resp, "prompt_feedback", None)
        print(f"  prompt_feedback   : {pf}")

        cands = getattr(resp, "candidates", None) or []
        print(f"  num candidates    : {len(cands)}")
        for i, c in enumerate(cands):
            fr = getattr(c, "finish_reason", None)
            fm = getattr(c, "finish_message", None)
            sr = getattr(c, "safety_ratings", None)
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            part_texts = [getattr(p, "text", None) for p in (parts or [])]
            print(f"    candidate[{i}].finish_reason  : {fr}")
            print(f"    candidate[{i}].finish_message : {fm}")
            print(f"    candidate[{i}].safety_ratings : {sr}")
            print(f"    candidate[{i}].part texts     : {part_texts}")

        usage = getattr(resp, "usage_metadata", None)
        print(f"  usage_metadata    : {usage}\n")


if __name__ == "__main__":
    main()
