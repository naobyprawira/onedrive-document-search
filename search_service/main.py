from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, List

import google.genai as genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from fastapi import FastAPI, HTTPException, Query
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

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

DOC_COLLECTION = os.getenv("DOC_COLLECTION", "documents_v2")
CHUNK_COLLECTION = os.getenv("CHUNK_COLLECTION", "chunks_v2")
DOC_VECTOR_NAME = "v_doc"
CHUNK_VECTOR_NAME = "v_chunk"
BM25_VECTOR_NAME = "v_bm25"
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

logger = logging.getLogger("search")
# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/search.log")
    ]
)

qdrant_client = QdrantClient(
    host=QDRANT_HOST, 
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY or None
)
app = FastAPI(title="Search Service", version="2.0.0")

@app.on_event("startup")
async def startup_event():
    os.makedirs("/app/logs", exist_ok=True)

_client: genai.Client | None = None
_bm25_model: SparseTextEmbedding | None = None


def _get_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _get_bm25_model() -> SparseTextEmbedding:
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


def embed_query(text: str) -> List[float]:
    client = _get_client()
    trimmed = (text or "").strip()
    if not trimmed:
        raise ValueError("Query must not be empty.")

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[trimmed],
                config=genai_types.EmbedContentConfig(
                    output_dimensionality=EMBED_DIM,
                    task_type="retrieval_query",
                ),
            )
            embeddings = response.embeddings or []
            if not embeddings or not embeddings[0].values:
                raise RuntimeError("Empty embedding returned from Gemini.")
            return embeddings[0].values
        except (genai_errors.ClientError, genai_errors.APIError, genai_errors.ServerError) as exc:
            if attempt >= attempts:
                raise RuntimeError(f"Gemini embedding failed: {exc}") from exc
            wait = 2**attempt + random.uniform(0, 1)
            logger.warning("Embedding retry %d/%d due to %s. Sleeping %.1fs", attempt, attempts, exc, wait)
            time.sleep(wait)

    raise RuntimeError("Failed to generate embedding for query.")


@app.get("/search", tags=["search"])
def search(
    query: str = Query(..., description="The search query string"),
    top_k: int = Query(5, ge=1, le=50, description="Number of top results to return"),
    chunk_candidates: int = Query(50, ge=1, le=200, description="Number of chunks to search before deduplication"),
) -> Dict[str, List[Dict]]:
    """
    Hybrid search (dense + sparse):
    1. Generate both semantic (dense) and BM25 (sparse) vectors for query
    2. Search chunks collection with hybrid query (for documents with chunks)
    3. Search documents collection directly (catches metadata-only docs with 0 chunks)
    4. Merge and deduplicate results
    5. Return top K results with document info + best chunk snippet (if available)
    """
    try:
        # Generate dense vector (semantic)
        query_vector = embed_query(query)
        
        # Generate sparse vector (BM25 keyword)
        bm25_vector = generate_bm25_vector(query)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Step 1: Hybrid search on chunks collection using prefetch + fusion
    try:
        response = qdrant_client.query_points(
            collection_name=CHUNK_COLLECTION,
            prefetch=[
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=bm25_vector["indices"],
                        values=bm25_vector["values"],
                    ),
                    using=BM25_VECTOR_NAME,
                    limit=chunk_candidates,
                ),
                qmodels.Prefetch(
                    query=query_vector,
                    using=CHUNK_VECTOR_NAME,
                    limit=chunk_candidates,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=chunk_candidates,
        )
        chunk_results = response.points if hasattr(response, 'points') else response
    except Exception as exc:
        logger.error(f"Hybrid search on chunks failed: {exc}")
        # Fallback to dense-only search
        try:
            chunk_results = qdrant_client.search(
                collection_name=CHUNK_COLLECTION,
                query_vector=(CHUNK_VECTOR_NAME, query_vector),
                limit=chunk_candidates,
            )
        except Exception:
            chunk_results = []
    
    # Step 2: Also search documents collection directly (for metadata-only docs)
    doc_direct_results = []
    try:
        response = qdrant_client.query_points(
            collection_name=DOC_COLLECTION,
            prefetch=[
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=bm25_vector["indices"],
                        values=bm25_vector["values"],
                    ),
                    using=BM25_VECTOR_NAME,
                    limit=top_k * 2,  # Get more candidates
                ),
                qmodels.Prefetch(
                    query=query_vector,
                    using=DOC_VECTOR_NAME,
                    limit=top_k * 2,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=top_k * 2,
        )
        doc_direct_results = response.points if hasattr(response, 'points') else response
    except Exception as exc:
        logger.error(f"Hybrid search on documents failed: {exc}")
        # Fallback to dense-only search
        try:
            doc_direct_results = qdrant_client.search(
                collection_name=DOC_COLLECTION,
                query_vector=(DOC_VECTOR_NAME, query_vector),
                limit=top_k * 2,
            )
        except Exception:
            doc_direct_results = []
    
    # If both searches failed, return empty
    if not chunk_results and not doc_direct_results:
        return {"results": []}

    # If both searches failed, return empty
    if not chunk_results and not doc_direct_results:
        return {"results": []}

    # Step 3: Group chunks by docId and keep only the best chunk per document
    doc_best_chunks: Dict[str, Dict] = {}  # docId -> {chunk_info, score}
    
    for chunk in chunk_results:
        chunk_payload = chunk.payload or {}
        doc_id = chunk_payload.get("docId")
        
        if not doc_id:
            continue
        
        # Keep only the best (first/highest scored) chunk per document
        if doc_id not in doc_best_chunks:
            snippet = chunk_payload.get("text", "")
            doc_best_chunks[doc_id] = {
                "docId": doc_id,
                "chunkNo": chunk_payload.get("chunkNo"),
                "snippet": snippet[:512] + ("..." if len(snippet) > 512 else ""),
                "score": chunk.score,
                "fileName": chunk_payload.get("fileName"),
                "source": "chunk",
            }
    
    # Step 4: Add documents found directly (with score adjustment to keep competitive)
    for doc in doc_direct_results:
        doc_payload = doc.payload or {}
        file_id = doc_payload.get("fileId")
        
        if not file_id:
            continue
        
        # If we already have this doc from chunks, keep the chunk version (better snippet)
        # But if chunk score is significantly lower, use doc score
        if file_id in doc_best_chunks:
            # If direct doc search scored much higher, update the score
            if doc.score > doc_best_chunks[file_id]["score"] * 1.2:
                doc_best_chunks[file_id]["score"] = doc.score
        else:
            # New document found only in direct search (likely metadata-only)
            doc_best_chunks[file_id] = {
                "docId": file_id,
                "chunkNo": None,
                "snippet": doc_payload.get("summary", "")[:512],  # Use summary as snippet
                "score": doc.score,
                "fileName": doc_payload.get("fileName"),
                "source": "document",
                # Include doc payload directly to avoid another query
                "drivePath": doc_payload.get("drivePath"),
                "summary": doc_payload.get("summary"),
                "webUrl": doc_payload.get("webUrl"),
            }
    
    # Step 5: For documents found via chunks, retrieve full document metadata
    chunk_source_docs = {doc_id: info for doc_id, info in doc_best_chunks.items() if info.get("source") == "chunk"}
    
    if chunk_source_docs:
        doc_ids = list(chunk_source_docs.keys())
        
        # Retrieve documents by filtering on fileId
        doc_points, _ = qdrant_client.scroll(
            collection_name=DOC_COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="fileId",
                        match=qmodels.MatchAny(any=doc_ids),
                    )
                ]
            ),
            limit=len(doc_ids),
            with_payload=True,
            with_vectors=False,
        )
        
        # Update chunk results with full document metadata
        for doc_point in doc_points:
            doc_payload = doc_point.payload or {}
            file_id = doc_payload.get("fileId")
            
            if file_id in chunk_source_docs:
                doc_best_chunks[file_id].update({
                    "drivePath": doc_payload.get("drivePath"),
                    "summary": doc_payload.get("summary"),
                    "webUrl": doc_payload.get("webUrl"),
                })
    
    # Step 6: Build final results list
    results: List[Dict] = []
    
    for file_id, info in doc_best_chunks.items():
        results.append({
            "fileId": file_id,
            "fileName": info.get("fileName"),
            "drivePath": info.get("drivePath"),
            "summary": info.get("summary"),
            "webUrl": info.get("webUrl"),
            "chunkNo": info.get("chunkNo"),
            "snippet": info.get("snippet"),
            "score": info["score"],
        })
    
    # Sort by score (highest first) and return top K
    results.sort(key=lambda item: item["score"], reverse=True)
    return {"results": results[:top_k]}
