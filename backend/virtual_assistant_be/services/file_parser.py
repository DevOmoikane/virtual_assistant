from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_HAS_FITZ = False
_HAS_DOCX = False
_HAS_ODF = False

try:
    import fitz

    _HAS_FITZ = True
except ImportError:
    pass

try:
    from docx import Document

    _HAS_DOCX = True
except ImportError:
    pass

try:
    from odf import text, teletype
    from odf.opendocument import load as odf_load

    _HAS_ODF = True
except ImportError:
    pass


SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".doc": "Word (legacy)",
    ".odt": "Open Document",
    ".md": "Markdown",
    ".txt": "Plain Text",
}


def extract_text(filepath: str) -> str | None:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _extract_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return _extract_docx(filepath)
    elif ext == ".odt":
        return _extract_odt(filepath)
    elif ext in (".md", ".txt"):
        return _extract_text(filepath)

    log.warning("Unsupported extension: %s", ext)
    return None


def _extract_pdf(filepath: str) -> str | None:
    if not _HAS_FITZ:
        log.warning("PyMuPDF not installed")
        return None
    try:
        doc = fitz.open(filepath)
        parts = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(parts)
    except Exception:
        log.warning("Failed to extract PDF", exc_info=True)
        return None


def _extract_docx(filepath: str) -> str | None:
    if not _HAS_DOCX:
        log.warning("python-docx not installed")
        return None
    try:
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        log.warning("Failed to extract DOCX", exc_info=True)
        return None


def _extract_odt(filepath: str) -> str | None:
    if not _HAS_ODF:
        log.warning("odfpy not installed")
        return None
    try:
        doc = odf_load(filepath)
        paras = doc.getElementsByType(text.P)
        return "\n".join(teletype.extractText(p) for p in paras)
    except Exception:
        log.warning("Failed to extract ODT", exc_info=True)
        return None


def _extract_text(filepath: str) -> str | None:
    try:
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, encoding="latin-1") as f:
                return f.read()
        except Exception:
            log.warning("Failed to read text file", exc_info=True)
            return None
    except Exception:
        log.warning("Failed to read text file", exc_info=True)
        return None
