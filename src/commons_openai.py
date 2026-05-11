import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


if (OPENAI_API_KEY := os.getenv("OPENAI_API_KEY", None)) is None:
    raise ValueError(
        "OPENAI_API_KEY is not set in environment variables get it from https://platform.openai.com/api-keys"
    )

# Explicit so we don't inherit OPENAI_BASE_URL from commons.py (Interfaze).
# Override with OPENAI_API_BASE_URL for Azure, proxies, or compatible endpoints.
OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def invoke_openai(
    messages: list[dict],
    model: str = "gpt-5.4",
    stream: bool = False,
):
    """Invoke OpenAI Chat Completions API (non-reasoning model, no thinking).

    Example image message:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            ],
        }]
    """
    try:
        return openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
        )
    except Exception as e:
        raise RuntimeError(f"Error invoking OpenAI API: {e}") from e
