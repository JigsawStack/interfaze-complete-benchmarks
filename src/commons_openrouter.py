import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


if (OPENROUTER_API_KEY := os.getenv("OPENROUTER_API_KEY", None)) is None:
    raise ValueError(
        "OPENROUTER_API_KEY is not set in environment variables get it from https://openrouter.ai/keys"
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

openrouter_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def invoke_openrouter(
    messages: list[dict],
    model: str,
    temperature: float | None = None,
    extra_body: dict | None = None,
):
    """Invoke any OpenRouter-hosted model via the OpenAI-compatible chat-completions API."""
    kwargs: dict = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    try:
        return openrouter_client.chat.completions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Error invoking OpenRouter API: {e}") from e
