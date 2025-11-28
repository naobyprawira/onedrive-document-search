from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config
from embeddings import EmbeddingTask, embed_texts, summarise_document
from ocr import extract_text_via_ocr
from storage import replace_document
from utils import bytes_sha256, file_sha256, split_into_chunks, text_sha256

logger = logging.getLogger("ingestion.pipeline")


@dataclass
class ProcessResult:
    file_id: str
    file_name: str
    success: bool
    chunk_count: int
    summary: str
    dry_run_payload: Optional[dict] = None
    error: Optional[str] = None


def process_metadata_only_document(file_meta: dict, *, dry_run: bool) -> ProcessResult:
    """
    Process non-PDF/non-image files as metadata-only entries without downloading.
    Creates searchable document entry using filename and path only.
    """
    file_name = file_meta.get("name", "")
    file_id = file_meta.get("id", "")
    drive_path = file_meta.get("drivePath") or ""
    path_segments = [segment for segment in drive_path.split("/") if segment]
    
    # Use filename + path as searchable text for embedding
    searchable_text = f"{file_name} {drive_path}"
    
    logger.info("Processing metadata-only (no download) for %s", file_name)
    
    try:
        # Generate embedding from filename and path
        doc_vectors = embed_texts([searchable_text], task=EmbeddingTask.DOCUMENT)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Embedding failed for metadata-only %s: %s", file_name, exc)
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=False,
            chunk_count=0,
            summary=config.SUMMARY_FALLBACK_TEXT,
            error=f"embedding error: {exc}",
        )
    
    if not doc_vectors or len(doc_vectors[0]) != config.EMBED_DIM:
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=False,
            chunk_count=0,
            summary=config.SUMMARY_FALLBACK_TEXT,
            error="document embedding missing",
        )
    
    doc_vector = doc_vectors[0]
    
    # Use fallback summary for non-PDF files, but include path info for BM25 search
    summary_text = config.SUMMARY_FALLBACK_TEXT
    # Store the searchable text (filename + path) in a separate field for BM25
    bm25_searchable_text = searchable_text
    
    doc_payload = {
        "fileId": file_id,
        "fileName": file_name,
        "drivePath": drive_path,
        "pathSegments": path_segments,
        "summary": summary_text,
        "bm25SearchText": bm25_searchable_text,  # Add this for better keyword search
        "webUrl": file_meta.get("webUrl"),
        "size": file_meta.get("size", 0),
        "lastModified": file_meta.get("lastModifiedDateTime"),
        "chunkCount": 0,
        "contentHash": text_sha256(searchable_text),
        "sourceSha256": "",  # No file bytes for metadata-only
        "fileType": "metadata-only",
    }
    
    if not dry_run:
        # No chunks for metadata-only documents
        replace_document(file_id, doc_payload, doc_vector, [], [])
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=True,
            chunk_count=0,
            summary=summary_text,
        )
    
    sample_payload = {
        "document_payload": doc_payload,
        "document_vector_preview": doc_vector[:8],
        "total_chunks": 0,
    }
    return ProcessResult(
        file_id=file_id,
        file_name=file_name,
        success=True,
        chunk_count=0,
        summary=summary_text,
        dry_run_payload=sample_payload,
    )


def process_document(file_meta: dict, file_path: str, *, dry_run: bool) -> ProcessResult:
    file_name = file_meta.get("name", "document.pdf")
    file_id = file_meta.get("id", "")
    mime_type = file_meta.get("mimeType", "")
    
    if not file_id:
        return ProcessResult(file_id="", file_name=file_name, success=False, chunk_count=0, summary="", error="missing file id")

    # Check if this is an image - treat like PDF (send to OCR)
    if mime_type and mime_type.startswith("image/"):
        logger.info("Processing image file %s through OCR", file_name)
        # Images go through OCR like PDFs
        
    # Check if this is a non-PDF, non-image file - metadata only
    elif mime_type != "application/pdf":
        logger.warning("Non-PDF/non-image file %s should not have been downloaded, skipping", file_name)
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=False,
            chunk_count=0,
            summary="",
            error="non-PDF/non-image file should use metadata-only processing"
        )

    # Original PDF/Image processing logic
    try:
        text = extract_text_via_ocr(file_path, filename=file_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OCR failed for %s: %s", file_name, exc)
        return ProcessResult(file_id=file_id, file_name=file_name, success=False, chunk_count=0, summary="", error=str(exc))

    if not text.strip():
        logger.warning("No text extracted for %s.", file_name)
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=False,
            chunk_count=0,
            summary="",
            error="empty text after OCR",
        )

    summary_text = summarise_document(text)
    chunks = split_into_chunks(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        logger.warning("Chunking produced no chunks for %s.", file_name)
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=False,
            chunk_count=0,
            summary=summary_text,
            error="no chunks generated",
        )

    try:
        doc_vectors = embed_texts([text], task=EmbeddingTask.DOCUMENT)
        chunk_vectors = embed_texts(chunks, task=EmbeddingTask.DOCUMENT)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Embedding failed for %s: %s", file_name, exc)
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=False,
            chunk_count=len(chunks),
            summary=summary_text,
            error=f"embedding error: {exc}",
        )

    if not doc_vectors or len(doc_vectors[0]) != config.EMBED_DIM:
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=False,
            chunk_count=len(chunks),
            summary=summary_text,
            error="document embedding missing",
        )

    if len(chunk_vectors) != len(chunks):
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=False,
            chunk_count=len(chunks),
            summary=summary_text,
            error="chunk embedding count mismatch",
        )

    doc_vector = doc_vectors[0]

    drive_path = file_meta.get("drivePath") or ""
    path_segments = [segment for segment in drive_path.split("/") if segment]

    doc_payload = {
        "fileId": file_id,
        "fileName": file_name,
        "drivePath": drive_path,
        "pathSegments": path_segments,
        "summary": summary_text,
        "webUrl": file_meta.get("webUrl"),
        "size": file_meta.get("size", 0),
        "lastModified": file_meta.get("lastModifiedDateTime"),
        "chunkCount": len(chunks),
        "contentHash": text_sha256(text),
        "sourceSha256": file_sha256(file_path),
    }

    chunk_payloads: List[dict] = []
    for index, chunk_text in enumerate(chunks):
        chunk_payloads.append(
            {
                "docId": file_id,
                "chunkNo": index,
                "text": chunk_text,
                "textHash": text_sha256(chunk_text),
                "drivePath": drive_path,
                "pathSegments": path_segments,
                "fileName": file_name,
            }
        )

    if not dry_run:
        replace_document(file_id, doc_payload, doc_vector, chunk_payloads, chunk_vectors)
        return ProcessResult(
            file_id=file_id,
            file_name=file_name,
            success=True,
            chunk_count=len(chunks),
            summary=summary_text,
        )

    sample_payload = {
        "document_payload": doc_payload,
        "document_vector_preview": doc_vector[:8],
        "chunk_preview": [
            {
                "payload": payload,
                "vector_preview": vector[:8],
            }
            for payload, vector in list(zip(chunk_payloads, chunk_vectors))[:3]
        ],
        "total_chunks": len(chunks),
    }
    return ProcessResult(
        file_id=file_id,
        file_name=file_name,
        success=True,
        chunk_count=len(chunks),
        summary=summary_text,
        dry_run_payload=sample_payload,
    )


def process_document_from_file(file_meta: dict, file_path: str, *, dry_run: bool) -> ProcessResult:
    """Process a document from a local file path.
    
    This variant is used when the file is already downloaded to temp storage.
    It reads the file and delegates to the standard processing pipeline.
    """
    file_name = file_meta.get("name", "document.pdf")
    file_id = file_meta.get("id", "")
    if not file_id:
        return ProcessResult(file_id="", file_name=file_name, success=False, chunk_count=0, summary="", error="missing file id")

    return process_document(file_meta, file_path, dry_run=dry_run)

