"""Probe what reasoning settings x-ai/grok-4.3 accepts on OpenRouter."""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
MODEL = "x-ai/grok-4.3"
PROMPT = "What is 2+2? Reply with only the number."

trials = [
    ("no_reasoning_field", {}),
    ("reasoning.enabled=False", {"reasoning": {"enabled": False}}),
    ("reasoning.enabled=True", {"reasoning": {"enabled": True}}),
    ("reasoning.effort=minimal", {"reasoning": {"effort": "minimal"}}),
    ("reasoning.effort=low", {"reasoning": {"effort": "low"}}),
    ("reasoning.effort=medium", {"reasoning": {"effort": "medium"}}),
    ("reasoning.effort=high", {"reasoning": {"effort": "high"}}),
    ("reasoning.max_tokens=1", {"reasoning": {"max_tokens": 1}}),
    ("reasoning.max_tokens=128", {"reasoning": {"max_tokens": 128}}),
]

for label, extra in trials:
    print("=" * 70)
    print(f"trial: {label}  body={extra}")
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.0,
            extra_body=extra,
        )
        msg = r.choices[0].message
        content = (getattr(msg, "content", None) or "").strip()
        reasoning = getattr(msg, "reasoning", None)
        usage = getattr(r, "usage", None)
        print(f"  OK content={content!r}")
        print(f"  reasoning preview: {(reasoning or '')[:120]!r}")
        if usage is not None:
            ud = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            print(f"  usage: {json.dumps(ud, default=str)}")
    except Exception as e:
        print(f"  ERROR {type(e).__name__}: {e}")
