from pydantic import BaseModel
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()


if (INTERFAZE_API_KEY := os.getenv("INTERFAZE_API_KEY", None)) is None:
    raise ValueError(
        "INTERFAZE_API_KEY is not set in environment variables get it from https://interfaze.ai/dashboard"
    )

INTERFAZE_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.interfaze.ai/v1")

interfaze_client = OpenAI(
    base_url=INTERFAZE_BASE_URL, api_key=os.getenv("INTERFAZE_API_KEY")
)


def invoke_interfaze(
    messages: list[dict],
    model: str = "interfaze-beta",
    stream: bool = False,
    structured_response: bool = False,
    structure_definition: BaseModel | None = None,
) -> dict:
    """For invoking interfaze with images you can use
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is in this image?", "image_url": {image_url: {"url": "https://example.com/image.jpg"}}},
    ]

    2. For structured response you can define a pydantic model and pass it as structure_definition argument. For example:
    class ResponseModel(BaseModel):
        capital: str
    Then you can invoke the function as follows:
    response = invoke_interfaze(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        structured_response=True,
        structure_definition=ResponseModel
    )
    """
    try:
        response = interfaze_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
        )

        return response
    except Exception as e:
        raise RuntimeError(f"Error invoking Interfaze API: {e}") from e
