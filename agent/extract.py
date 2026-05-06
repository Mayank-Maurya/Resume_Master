import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_pdf_text(path: str | Path) -> str:
    logger.info("Extracting PDF text from %s", path)
    reader = PdfReader(str(path))
    pages = reader.pages
    text = "\n".join((page.extract_text() or "") for page in pages)
    logger.info("Extracted %d chars from %d pages", len(text), len(pages))
    return text


def read_tex(path: str | Path) -> str:
    logger.info("Reading .tex file: %s", path)
    content = Path(path).read_text(encoding="utf-8")
    logger.info("Read %d chars", len(content))
    return content
