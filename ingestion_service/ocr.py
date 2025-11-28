from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore
from typing import Dict, List, Optional, Tuple

import requests
import json
from pathlib import Path

import config
from utils import bytes_sha256

logger = logging.getLogger("ingestion.ocr")

_OCR_REQUEST_GUARD = BoundedSemaphore(max(1, config.OCR_MAX_PARALLEL))


def split_pdf_by_pages(file_path: str) -> List[bytes]:
    """Split PDF into single-page PDF byte blobs, reading from file."""
    try:
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore

        # Open file in binary mode
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            pages: List[bytes] = []
            for index in range(len(reader.pages)):
                writer = PdfWriter()
                writer.add_page(reader.pages[index])
                buffer = io.BytesIO()
                writer.write(buffer)
                pages.append(buffer.getvalue())
            return pages
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to split PDF into pages: %s. Falling back to full document.", exc)
        try:
            return [Path(file_path).read_bytes()]
        except Exception:
            return []


def _cache_file(cache_key: str) -> Path:
    return config.OCR_CACHE_DIR / f"{cache_key}.json"


def load_cached_page(cache_key: str) -> Optional[str]:
    """Return cached OCR result if present."""
    cache_path = _cache_file(cache_key)
    if not cache_path.exists():
        return None
    try:
        raw = cache_path.read_text(encoding="utf-8")
        if not raw:
            return ""
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload.get("text") or ""
        except json.JSONDecodeError:
            pass
        return raw
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to read OCR cache %s: %s", cache_path, exc)
        return None


def write_cached_page(cache_key: str, text: str) -> None:
    """Persist OCR page result to cache."""
    cache_path = _cache_file(cache_key)
    try:
        cache_path.write_text(text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to write OCR cache %s: %s", cache_path, exc)


def _ocr_single_page(
    page_bytes: bytes,
    page_index: int,
    total_pages: int,
    filename: str,
) -> Tuple[int, str]:
    cache_key = f"{bytes_sha256(page_bytes)}_{page_index}"

    cached = load_cached_page(cache_key)
    if cached is not None:
        logger.info("OCR cache hit %s page %d/%d", filename, page_index + 1, total_pages)
        return page_index, cached

    files = {"file": (f"{filename}_p{page_index + 1}.pdf", page_bytes, "application/pdf")}
    retries = 3
    backoff = 3

    for attempt in range(1, retries + 1):
        try:
            with _OCR_REQUEST_GUARD:
                logger.debug("Sending OCR request to %s with file: %s (%d bytes)", config.OCR_SERVICE_URL, f"{filename}_p{page_index + 1}.pdf", len(page_bytes))
                response = requests.post(config.OCR_SERVICE_URL, files=files, timeout=300)
            response.raise_for_status()
            payload = response.json()
            text = payload.get("text", "") if isinstance(payload, dict) else ""
            write_cached_page(cache_key, text)
            logger.info("OCR page %d/%d for %s completed (%d chars)", page_index + 1, total_pages, filename, len(text))
            return page_index, text
        except requests.Timeout:
            logger.warning(
                "OCR timeout on %s page %d (attempt %d/%d). Retrying after %ds.",
                filename,
                page_index + 1,
                attempt,
                retries,
                backoff,
            )
        except requests.RequestException as exc:
            logger.warning(
                "OCR request failed for %s page %d (attempt %d/%d): %s",
                filename,
                page_index + 1,
                attempt,
                retries,
                exc,
            )
        time.sleep(backoff)

    logger.error("OCR failed for %s page %d after %d attempts.", filename, page_index + 1, retries)
    write_cached_page(cache_key, "")
    return page_index, ""


def extract_text_via_ocr(file_path: str, filename: str) -> str:
    """Run the external OCR service page-by-page with caching and concurrency guards."""
    if not config.OCR_SERVICE_URL:
        logger.warning("OCR_SERVICE_URL is not configured. Skipping OCR for %s.", filename)
        return ""

    pages = split_pdf_by_pages(file_path)
    total_pages = len(pages)
    if total_pages == 0:
        logger.warning("No pages extracted from %s", filename)
        return ""

    logger.info("Submitting %s (%d pages) to OCR service at %s.", filename, total_pages, config.OCR_SERVICE_URL)

    if total_pages == 1:
        _, text = _ocr_single_page(pages[0], 0, 1, filename)
        return text.strip()

    results: Dict[int, str] = {}
    max_workers = min(config.OCR_MAX_PARALLEL, max(1, total_pages))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_ocr_single_page, page_bytes, index, total_pages, filename): index
                for index, page_bytes in enumerate(pages)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    page_index, text = future.result()
                    results[page_index] = text
                except Exception as exc:  # noqa: BLE001
                    logger.error("Unexpected OCR failure for %s page %d: %s", filename, index + 1, exc)
                    results[index] = ""
    except Exception as exc:  # noqa: BLE001
        logger.error("OCR service error for %s: %s. Returning empty text.", filename, exc)
        return ""

    combined = []
    for idx in range(total_pages):
        page_text = results.get(idx, "")
        if page_text:
            combined.append(page_text)

    output = "\n\n".join(combined).strip()
    if not output:
        logger.warning("OCR returned no text for %s", filename)
    else:
        logger.info("OCR finished for %s: %d characters extracted.", filename, len(output))
    return output
