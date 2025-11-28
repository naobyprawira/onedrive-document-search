from __future__ import annotations

import hashlib
from typing import Iterable, List


def sha1_to_int(value: str) -> int:
    """Derive a deterministic 64-bit integer from a string."""
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into character-based chunks with configurable overlap."""
    if not text:
        return []

    cleaned = text.strip()
    if not cleaned:
        return []

    chunk_size = max(1, chunk_size)
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap if chunk_size > overlap else 1

    chunks: List[str] = []
    for start in range(0, len(cleaned), step):
        piece = cleaned[start : start + chunk_size]
        if piece.strip():
            chunks.append(piece)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def text_sha256(value: str) -> str:
    """Return hex encoded SHA256 for text values."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bytes_sha256(data: bytes) -> str:
    """Return hex encoded SHA256 for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def file_sha256(file_path: str) -> str:
    """Return hex encoded SHA256 for a file, reading in chunks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
