import os
from typing import Iterable

import httpx
from reducto import Reducto


_client: Reducto | None = None

# Mirror of the LLM runners' prompt (run_interfaze / run_openai_mini / run_grok / etc.).
# Reducto's enhance.agentic[scope=text] takes a `prompt` that steers its VLM
# second-pass — this is where the equivalent text instruction goes.
PARSE_PROMPT = (
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


def _get_client() -> Reducto:
    global _client
    if _client is None:
        api_key = os.getenv("REDUCTO_API_KEY")
        if not api_key:
            raise SystemExit(
                "REDUCTO_API_KEY not set — get it from https://platform.reducto.ai"
            )
        _client = Reducto(api_key=api_key)
    return _client


def _join_chunks_from_url(url: str, timeout: float) -> str:
    data = httpx.get(url, timeout=timeout).json()
    chunks = data.get("chunks", [])
    return "\n\n".join((c.get("content") or "") for c in chunks).strip()


def run_reducto(
    pdf_path: str,
    page_num: int = 1,
    timeout: float = 600.0,
) -> str:
    """Parse a single PDF page through Reducto and return the markdown content."""
    client = _get_client()

    with open(pdf_path, "rb") as fh:
        upload = client.upload(file=fh, timeout=timeout)

    response = client.parse.run(
        input=f"reducto://{upload.file_id}",
        settings={
            "page_range": [page_num],
            "extraction_mode": "hybrid",
            "ocr_system": "standard",
            "deep_extract": True,
        },
        enhance={
            "agentic": [
                {"scope": "figure", "advanced_chart_agent": True},
                {"scope": "table"},
                {"scope": "text", "prompt": PARSE_PROMPT},
            ],
            "intelligent_ordering": True,
            "summarize_figures": False,
        },
        formatting={
            "table_output_format": "html",
            "merge_tables": False,
        },
        retrieval={"filter_blocks": ["Header", "Footer", "Page Number"]},
        timeout=timeout,
    )
    try:
        from src.commons_reducto import record_usage
        record_usage("parse", getattr(response, "usage", None))
    except Exception:
        pass

    result = response.result
    chunks = getattr(result, "chunks", None)
    if chunks is None:
        url = getattr(result, "url", None)
        text = _join_chunks_from_url(url, timeout) if url else ""
    else:
        text = "\n\n".join((c.content or "") for c in chunks).strip()

    if not text or text.strip().lower() in ("null", "none", "n/a"):
        return ""
    return text
