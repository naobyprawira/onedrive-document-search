from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)
# Workaround for tqdm 'no attribute _lock' issue in fastembed
import tqdm
import tqdm.auto
from threading import Lock

def _patch_tqdm(t):
    if not hasattr(t, '_lock'):
        try:
            t._lock = Lock()
        except (TypeError, AttributeError):
            pass

_patch_tqdm(tqdm.tqdm)
_patch_tqdm(tqdm.auto.tqdm)

from fastembed import SparseTextEmbedding

import config
from utils import sha1_to_int

logger = logging.getLogger("ingestion.storage")

# BM25 model for sparse vectors (lazy loaded)
_bm25_model = None
BM25_VECTOR_NAME = "v_bm25"


def _get_bm25_model():
    """Lazy load BM25 model."""
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("Loaded BM25 model for sparse vectors")
    return _bm25_model


def generate_bm25_vector(text: str) -> Dict:
    """Generate BM25 sparse vector from text."""
    if not text or not text.strip():
        return {"indices": [], "values": []}
    
    try:
        model = _get_bm25_model()
        embeddings = list(model.embed([text]))
        if not embeddings:
            return {"indices": [], "values": []}
        
        sparse_vector = embeddings[0]
        return {
            "indices": sparse_vector.indices.tolist(),
            "values": sparse_vector.values.tolist(),
        }
    except Exception as exc:
        logger.warning(f"Failed to generate BM25 vector: {exc}")
        return {"indices": [], "values": []}

# Initialize Qdrant client
if not config.QDRANT_URL:
    raise ValueError("QDRANT_URL is not set. Please configure it for Qdrant Cloud.")

logger.info("Connecting to Qdrant Cloud at %s", config.QDRANT_URL)
qdrant_client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY or None
)


def _collection_exists(name: str) -> bool:
    url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/collections/{name}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to verify Qdrant collection %s: %s", name, exc)
        raise
    return False


def ensure_collections() -> None:
    """
    Ensure that both collections required by the ingestion pipeline exist.
    """

    collections = [
        (
            config.DOC_COLLECTION,
            {config.DOC_VECTOR_NAME: VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE)},
            {BM25_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False))},
        ),
        (
            config.CHUNK_COLLECTION,
            {config.CHUNK_VECTOR_NAME: VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE)},
            {BM25_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False))},
        ),
    ]

    for name, vectors, sparse_vectors in collections:
        if _collection_exists(name):
            logger.debug("Qdrant collection %s already exists.", name)
            continue
        logger.info("Creating Qdrant collection %s.", name)
        try:
            qdrant_client.create_collection(
                collection_name=name,
                vectors_config=vectors,
                sparse_vectors_config=sparse_vectors,
            )
        except UnexpectedResponse as exc:
            message = str(exc)
            if "already exists" in message or "409" in message:
                logger.info("Collection %s already existed (409).", name)
            else:
                raise


def get_local_inventory() -> Dict[str, dict]:
    """Return a mapping of fileId to stored payload for documents."""
    inventory: Dict[str, dict] = {}
    next_offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=config.DOC_COLLECTION,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=next_offset,
        )
        for point in points:
            payload = point.payload or {}
            file_id = payload.get("fileId")
            if file_id:
                inventory[file_id] = payload
        if not next_offset:
            break
    return inventory


def delete_document_and_chunks(file_id: str) -> None:
    """Remove a document and associated chunks from Qdrant."""
    logger.info("Removing document %s and associated chunks from Qdrant.", file_id)
    qdrant_client.delete(
        collection_name=config.CHUNK_COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="docId", match=MatchValue(value=file_id))]),
        wait=True,
    )
    qdrant_client.delete(
        collection_name=config.DOC_COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="fileId", match=MatchValue(value=file_id))]),
        wait=True,
    )


def replace_document(
    file_id: str,
    doc_payload: dict,
    doc_vector: List[float],
    chunk_payloads: List[dict],
    chunk_vectors: List[List[float]],
) -> None:
    """Replace document payload and chunks within Qdrant."""
    if len(chunk_payloads) != len(chunk_vectors):
        raise ValueError("Chunk payloads/vector length mismatch.")

    qdrant_client.delete(
        collection_name=config.CHUNK_COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="docId", match=MatchValue(value=file_id))]),
        wait=True,
    )

    chunk_points: List[PointStruct] = []
    for idx, (payload, vector) in enumerate(zip(chunk_payloads, chunk_vectors)):
        payload = dict(payload)
        payload.setdefault("docId", file_id)
        payload.setdefault("chunkNo", idx)
        point_id = sha1_to_int(f"{file_id}::{idx}")
        
        # Generate BM25 vector for chunk text
        chunk_text = payload.get("text", "")
        bm25_vector = generate_bm25_vector(chunk_text)
        
        chunk_points.append(
            PointStruct(
                id=point_id,
                vector={
                    config.CHUNK_VECTOR_NAME: vector,
                    BM25_VECTOR_NAME: bm25_vector,
                },
                payload=payload,
            )
        )

    if chunk_points:
        qdrant_client.upsert(
            collection_name=config.CHUNK_COLLECTION,
            points=chunk_points,
            wait=True,
        )

    # Generate BM25 vector for document
    # Use bm25SearchText if available (for metadata-only docs), otherwise fileName + summary
    doc_text_parts = []
    if doc_payload.get("bm25SearchText"):
        # Metadata-only documents have pre-built search text (filename + path)
        doc_text_parts.append(doc_payload["bm25SearchText"])
    else:
        # Regular documents use fileName + summary
        if doc_payload.get("fileName"):
            doc_text_parts.append(doc_payload["fileName"])
        if doc_payload.get("summary"):
            doc_text_parts.append(doc_payload["summary"])
    doc_bm25_vector = generate_bm25_vector(" ".join(doc_text_parts))

    doc_point = PointStruct(
        id=sha1_to_int(f"doc::{file_id}"),
        vector={
            config.DOC_VECTOR_NAME: doc_vector,
            BM25_VECTOR_NAME: doc_bm25_vector,
        },
        payload=doc_payload,
    )
    qdrant_client.upsert(
        collection_name=config.DOC_COLLECTION,
        points=[doc_point],
        wait=True,
    )
