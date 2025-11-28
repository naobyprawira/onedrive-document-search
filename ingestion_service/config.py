from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "documents_v2")
CHUNK_COLLECTION = os.getenv("CHUNK_COLLECTION", "chunks_v2")
DOC_VECTOR_NAME = "v_doc"
CHUNK_VECTOR_NAME = "v_chunk"

# Google Gemini / GenAI configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemini-2.5-flash")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

# Summary model provider (GEMINI or OPENROUTER)
SUMMARY_PROVIDER = os.getenv("SUMMARY_PROVIDER", "GEMINI").upper()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# OneDrive / Microsoft Graph configuration
MS_TENANT_ID = os.getenv("MS_TENANT_ID", "").strip()
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID", "").strip()
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET", "").strip()
ONEDRIVE_DRIVE_ID = os.getenv("ONEDRIVE_DRIVE_ID", "").strip()
ONEDRIVE_ROOT_PATH = os.getenv("ONEDRIVE_ROOT_PATH", "AI/Document Filing").strip()
GRAPH_SCOPE = "https://graph.microsoft.com/.default"
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

# Scheduler options
SCHEDULE_CRON = os.getenv("SCHEDULE_CRON", "0 0 * * *")

# OCR configuration
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "").strip()
OCR_MAX_PARALLEL = max(1, int(os.getenv("OCR_MAX_PARALLEL", "5")))
OCR_CACHE_DIR = Path(os.getenv("OCR_CACHE_DIR", Path(tempfile.gettempdir()) / "document_search_ocr_cache"))
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking configuration (character based)
CHUNK_SIZE = max(1, int(os.getenv("CHUNK_SIZE", "2000")))
CHUNK_OVERLAP = max(0, int(os.getenv("CHUNK_OVERLAP", "200")))

# Operational flags
DEFAULT_DRY_RUN = os.getenv("DRY_RUN", "false").lower() in {"true", "1", "yes"}
SKIP_SUMMARY = os.getenv("SKIP_SUMMARY", "false").lower() in {"true", "1", "yes"}
MAX_DOCUMENTS = max(0, int(os.getenv("MAX_DOCUMENTS", "0")))  # 0 = no limit
INGESTION_WORKERS = max(1, int(os.getenv("INGESTION_WORKERS", "3")))
EMBED_BATCH_SIZE = max(1, int(os.getenv("EMBED_BATCH_SIZE", "16")))
EMBED_MAX_RETRIES = max(1, int(os.getenv("EMBED_MAX_RETRIES", "3")))
SUMMARY_MAX_RETRIES = max(1, int(os.getenv("SUMMARY_MAX_RETRIES", "3")))

# Summary fallback text
SUMMARY_FALLBACK_TEXT = "Ringkasan dokumen ini tidak ditemukan."

