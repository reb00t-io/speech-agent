"""Markdown → PDF document publishing.

Converts a Markdown document to a PDF, stores it under a secure random path,
and exposes helpers to serve it back. Uses pure-Python dependencies (markdown +
fpdf2) so it works in the slim container image without native libraries.
"""
from __future__ import annotations

import logging
import os
import re
import secrets
from pathlib import Path

import markdown as markdown_lib
from fpdf import FPDF

logger = logging.getLogger(__name__)

DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "data/downloads"))

# Download filenames are random tokens; only these may be served.
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_-]+\.pdf$")

# fpdf2's core fonts are Latin-1 only. Normalise the most common Unicode
# punctuation the model emits so reports render cleanly; anything still outside
# Latin-1 is replaced rather than crashing PDF generation.
_UNICODE_REPLACEMENTS = {
    "—": "-", "–": "-", "‒": "-", "−": "-",
    "‘": "'", "’": "'", "‚": ",", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "…": "...", " ": " ", "•": "-", "·": "-",
    "→": "->", "←": "<-", "≤": "<=", "≥": ">=",
}


def _to_latin1(text: str) -> str:
    for needle, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(needle, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def markdown_to_pdf_bytes(markdown_text: str, title: str = "") -> bytes:
    """Render Markdown to PDF bytes."""
    html = markdown_lib.markdown(
        markdown_text or "",
        extensions=["extra", "sane_lists", "tables"],
    )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    clean_title = _to_latin1((title or "").strip())
    if clean_title:
        pdf.set_font("Helvetica", "B", 18)
        pdf.multi_cell(0, 9, clean_title)
        pdf.ln(3)

    pdf.set_font("Helvetica", size=11)
    pdf.write_html(_to_latin1(html))

    output = pdf.output()
    return bytes(output)


def publish_markdown(markdown_text: str, title: str = "") -> dict:
    """Convert Markdown to a PDF saved under a random path; return a download link."""
    if not (markdown_text or "").strip():
        return {"error": "Missing required argument: markdown"}

    try:
        pdf_bytes = markdown_to_pdf_bytes(markdown_text, title)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("PDF generation failed")
        return {"error": f"Failed to generate PDF: {exc}"}

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{secrets.token_urlsafe(16)}.pdf"
    (DOWNLOADS_DIR / filename).write_bytes(pdf_bytes)
    logger.info("Published document %r (%d bytes) as %s", title or "untitled", len(pdf_bytes), filename)

    return {
        "download_url": f"/download/{filename}",
        "filename": filename,
        "title": title or "",
        "bytes": len(pdf_bytes),
    }


def resolve_download(name: str) -> Path | None:
    """Return the on-disk path for a download name, or None if invalid/missing."""
    if not SAFE_FILENAME_RE.match(name or ""):
        return None
    path = DOWNLOADS_DIR / name
    return path if path.is_file() else None
