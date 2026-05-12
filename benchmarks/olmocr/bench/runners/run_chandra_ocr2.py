import os

import requests

from benchmarks.olmocr.data.renderpdf import render_pdf_to_base64png


def run_chandra_ocr2(
    pdf_path: str,
    page_num: int = 1,
    target_longest_image_dim: int = 2048,
    task: str = "ocr_layout",
    max_tokens: int = 12384,
) -> str:
    """Convert a PDF page to markdown via Chandra OCR 2 deployed on Modal.

    Apples-to-apples with other olmOCR runners: temperature=0, no
    reasoning/thinking mode (Chandra OCR 2 is a fine-tuned OCR VLM with no
    such mode), no chain-of-thought.
    """
    url = os.getenv("CHANDRA_MODAL_URL")
    key = os.getenv("CHANDRA_MODAL_ADMIN_KEY") or os.getenv("ADMIN_KEY")
    if not url:
        raise SystemExit(
            "CHANDRA_MODAL_URL not set — point this at the Modal endpoint base URL "
            "(no trailing /ocr)."
        )
    if not key:
        raise SystemExit(
            "CHANDRA_MODAL_ADMIN_KEY (or ADMIN_KEY) not set — required by the Modal app."
        )

    image_base64 = render_pdf_to_base64png(
        pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim
    )

    resp = requests.post(
        f"{url.rstrip('/')}/ocr",
        headers={
            "x-api-admin-key": key,
            "Content-Type": "application/json",
        },
        json={
            "inputs": [f"data:image/png;base64,{image_base64}"],
            "task": task,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=600,
    )
    resp.raise_for_status()
    payload = resp.json()

    md = payload.get("markdown", "") if isinstance(payload, dict) else ""
    if not md or md.strip().lower() in ("null", "none", "n/a", ""):
        return ""
    return md
