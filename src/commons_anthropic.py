import os

import anthropic
from dotenv import load_dotenv

load_dotenv()


if (ANTHROPIC_API_KEY := os.getenv("ANTHROPIC_API_KEY", None)) is None:
    raise ValueError(
        "ANTHROPIC_API_KEY is not set in environment variables get it from https://console.anthropic.com/"
    )

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def invoke_anthropic(
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    system: str | None = None,
):
    """Invoke Anthropic Messages API with thinking disabled (omitted).

    Example image message:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_str}},
                {"type": "text", "text": "What is in this image?"},
            ],
        }]
    """
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system
    try:
        return anthropic_client.messages.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Error invoking Anthropic API: {e}") from e
