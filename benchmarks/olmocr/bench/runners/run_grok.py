import os

from openai import OpenAI

from benchmarks.olmocr.data.renderpdf import render_pdf_to_base64png

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY not set — get it from https://openrouter.ai/keys"
            )
        _client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _client


def run_grok(
    pdf_path: str,
    page_num: int = 1,
    model: str = "x-ai/grok-4.3",
    target_longest_image_dim: int = 2048,
) -> str:
    """Convert a PDF page to markdown via Grok 4.3 with reasoning_effort='low'."""
    image_base64 = render_pdf_to_base64png(
        pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim
    )

    client = _get_client()

    prompt = (
        "Below is the image of one page of a PDF document. "
        "Just return the plain text representation of this document as if you were reading it naturally.\n"
        "Turn equations into LaTeX using \\( \\) for inline math and \\[ \\] for display math. "
        "Never describe equations in words — always use LaTeX notation. "
        "Turn tables into markdown format.\n"
        "Remove the headers and footers completely — do not include any text "
        "that appears at the very top or very bottom of the page outside the main body content. "
        "This includes page numbers, journal names, author names in running headers, "
        "copyright lines, DOI lines, citation requests, institutional addresses in margins, "
        "and download dates. Keep references and footnotes that are part of the body.\n"
        "For multi-column layouts, read each column top to bottom before moving to the next.\n"
        "Read any natural handwriting.\n"
        "This is likely one page out of several in the document, so be sure to preserve "
        "any sentences that come from the previous page, or continue onto the next page, exactly as they are.\n"
        "If there is no text at all that you think you should read, you can output null.\n"
        "Do not hallucinate."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        reasoning_effort="low",
        max_completion_tokens=20000,
    )

    assert len(response.choices) > 0
    raw = response.choices[0].message.content
    if raw is None or raw.strip().lower() in ("null", "none", "n/a", ""):
        return ""
    return raw
