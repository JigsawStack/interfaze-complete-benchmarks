import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import httpx
from dotenv import load_dotenv
from reducto import Reducto

load_dotenv()


if (REDUCTO_API_KEY := os.getenv("REDUCTO_API_KEY", None)) is None:
    raise ValueError(
        "REDUCTO_API_KEY is not set in environment variables — get it from https://platform.reducto.ai"
    )


reducto_client = Reducto(api_key=REDUCTO_API_KEY)


# ---------- Usage tracking (thread-safe) ----------------------------------

_usage_lock = threading.Lock()
_usage_state: dict = {
    "calls": 0,
    "pages": 0,
    "credits": 0.0,
    "fields": 0,
    "by_endpoint": {},
}


def record_usage(endpoint: str, usage_obj) -> None:
    """Tally a single API response's usage. Safe to call from many threads."""
    if usage_obj is None:
        return
    pages = getattr(usage_obj, "num_pages", 0) or 0
    credits = getattr(usage_obj, "credits", 0) or 0
    fields = getattr(usage_obj, "num_fields", 0) or 0
    with _usage_lock:
        _usage_state["calls"] += 1
        _usage_state["pages"] += pages
        _usage_state["credits"] += credits
        _usage_state["fields"] += fields
        bucket = _usage_state["by_endpoint"].setdefault(
            endpoint, {"calls": 0, "pages": 0, "credits": 0.0, "fields": 0}
        )
        bucket["calls"] += 1
        bucket["pages"] += pages
        bucket["credits"] += credits
        bucket["fields"] += fields


def get_usage_snapshot() -> dict:
    with _usage_lock:
        return copy.deepcopy(_usage_state)


def reset_usage() -> None:
    with _usage_lock:
        _usage_state["calls"] = 0
        _usage_state["pages"] = 0
        _usage_state["credits"] = 0.0
        _usage_state["fields"] = 0
        _usage_state["by_endpoint"] = {}


def write_usage_snapshot(path) -> None:
    """Atomically persist the current usage tally to a JSON file."""
    snap = get_usage_snapshot()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2))
    tmp.replace(p)


def _join_chunks_from_url(url: str, timeout: float) -> str:
    data = httpx.get(url, timeout=timeout).json()
    chunks = data.get("chunks", [])
    return "\n\n".join((c.get("content") or "") for c in chunks).strip()


def upload_file(path: str, timeout: float = 600.0) -> str:
    """Upload a local file and return its reducto file_id."""
    with open(path, "rb") as fh:
        upload = reducto_client.upload(file=fh, timeout=timeout)
    return upload.file_id


def parse_file(
    file_id: str,
    table_format: str = "md",
    timeout: float = 600.0,
) -> tuple[str, str | None]:
    """Parse a previously-uploaded file. Returns (joined_markdown, parse_job_id).

    parse_job_id is None when the response is large (URL result) — extract
    chaining still works against `reducto://<file_id>` in that case.
    """
    response = reducto_client.parse.run(
        input=f"reducto://{file_id}",
        timeout=timeout,
        enhance={
            "agentic": [{"scope": "text"}],
            "intelligent_ordering": True,
        },
        formatting={"table_output_format": table_format},
    )
    record_usage("parse", getattr(response, "usage", None))
    job_id = getattr(response, "job_id", None)
    result = response.result
    chunks = getattr(result, "chunks", None)
    if chunks is None:
        url = getattr(result, "url", None)
        text = _join_chunks_from_url(url, timeout) if url else ""
    else:
        text = "\n\n".join((c.content or "") for c in chunks).strip()
    return text, job_id


def extract_with_schema(
    parse_job_id: str | None,
    file_id: str,
    schema: dict,
    system_prompt: str,
    timeout: float = 600.0,
) -> Any:
    """Run extract over an existing parse job (or a file_id fallback).

    Returns the extract.result (dict or list[dict]).
    """
    extract_input = (
        f"jobid://{parse_job_id}" if parse_job_id else f"reducto://{file_id}"
    )
    response = reducto_client.extract.run(
        input=extract_input,
        instructions={"schema": schema, "system_prompt": system_prompt},
        timeout=timeout,
    )
    record_usage("extract", getattr(response, "usage", None))
    return response.result


def parse_file_to_markdown(
    path: str,
    pages: Iterable[int] | None = None,
    timeout: float = 600.0,
) -> str:
    """Upload a local file (PDF / image) to Reducto, parse it, return joined markdown.

    pages: 1-indexed page numbers to include (None = all pages).

    Quality settings (per Reducto's best-practice docs):
      - enhance.agentic = [{"scope": "text"}]: VLM second pass; fixes math notation,
        handwriting, special characters.
      - enhance.intelligent_ordering = True: improves reading order on multi-column.
      - formatting.table_output_format = "md": markdown tables (matches our pipelines).
    """
    with open(path, "rb") as fh:
        upload = reducto_client.upload(file=fh, timeout=timeout)

    kwargs: dict = {
        "input": f"reducto://{upload.file_id}",
        "timeout": timeout,
        "enhance": {
            "agentic": [{"scope": "text"}, {"scope": "table"}],
            "intelligent_ordering": True,
        },
        "formatting": {"table_output_format": "md"},
    }
    if pages is not None:
        kwargs["settings"] = {"page_range": list(pages)}

    response = reducto_client.parse.run(**kwargs)
    record_usage("parse", getattr(response, "usage", None))

    result = response.result
    chunks = getattr(result, "chunks", None)
    if chunks is None:
        url = getattr(result, "url", None)
        return _join_chunks_from_url(url, timeout) if url else ""

    return "\n\n".join((c.content or "") for c in chunks).strip()
