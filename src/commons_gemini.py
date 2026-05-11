import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


if (GEMINI_API_KEY := os.getenv("GEMINI_API_KEY", None)) is None:
    raise ValueError(
        "GEMINI_API_KEY is not set in environment variables get it from https://aistudio.google.com/apikey"
    )

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def invoke_gemini(
    contents,
    model: str = "gemini-3.1-pro-preview",
    system: str | None = None,
):
    """Invoke Gemini generate_content.

    Note: Gemini 3.x Pro models require thinking mode (reject thinking_budget=0).
    We let the API use its default thinking level rather than forcing it off.
    If you switch to a Flash variant and want thinking disabled, add
    `thinking_config=types.ThinkingConfig(thinking_budget=0)` to the config below.

    `contents` can be a list mixing strings and PIL.Image / types.Part objects, e.g.:
        contents = [question_text, pil_image]
    """
    config = types.GenerateContentConfig(
        system_instruction=system,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        return gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as e:
        raise RuntimeError(f"Error invoking Gemini API: {e}") from e
