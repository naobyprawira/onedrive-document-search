from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
from embeddings import ensure_embeddings_ready, ensure_summarizer_ready
from graph import download_file_to_path, get_graph_access_token, list_onedrive_recursive
from pipeline import ProcessResult, process_document_from_file, process_metadata_only_document
from state_tracker import cleanup_completed, is_file_busy, set_file_state
from storage import delete_document_and_chunks, ensure_collections, get_local_inventory


def is_processable_with_ocr(mime_type: str) -> bool:
    """Check if file type can be processed with OCR (PDF or image)."""
    return mime_type in [
        "application/pdf",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff",
    ]

logger = logging.getLogger("ingestion")
logging.basicConfig(level=logging.INFO)

# Suppress verbose warnings from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
logging.getLogger("PyPDF2.generic._data_structures").setLevel(logging.ERROR)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

app = FastAPI(title="Ingestion Service", version="2.0.0")
scheduler = BackgroundScheduler()


class IngestNowRequest(BaseModel):
    dry_run: Optional[bool] = None


def _start_worker_threads(worker_count: int, queue: Queue, dry_run: bool, results: list[ProcessResult]) -> list[Thread]:
    def worker() -> None:
        while True:
            work_item = queue.get()
            if work_item is None:
                queue.task_done()
                break

            file_meta, temp_file_path = work_item
            file_id = file_meta.get("id", "")
            file_name = file_meta.get("name", "unknown")
            try:
                set_file_state(file_id, "processing", file_name)
                
                # Check if this is metadata-only (no temp file)
                if temp_file_path is None:
                    # Process metadata-only (no download needed)
                    from pipeline import process_metadata_only_document
                    result = process_metadata_only_document(file_meta, dry_run=dry_run)
                else:
                    # Process with file content (PDF or image)
                    if not os.path.exists(temp_file_path):
                        raise RuntimeError(f"Temp file not found: {temp_file_path}")
                    result = process_document_from_file(file_meta, temp_file_path, dry_run=dry_run)
                
                results.append(result)
                
                if result.success:
                    set_file_state(file_id, "completed", file_name)
                else:
                    set_file_state(file_id, "failed", file_name)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to process %s: %s", file_name, exc)
                set_file_state(file_id, "failed", file_name)
                results.append(
                    ProcessResult(
                        file_id=file_id or "",
                        file_name=file_name,
                        success=False,
                        chunk_count=0,
                        summary="",
                        error=str(exc),
                    )
                )
            finally:
                # Clean up temp file after processing (if it exists)
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.info("Cleaned up temp file: %s", temp_file_path)
                    except Exception as cleanup_exc:  # noqa: BLE001
                        logger.warning("Failed to clean up temp file %s: %s", temp_file_path, cleanup_exc)
                queue.task_done()

    threads = [Thread(target=worker, name=f"ingestion-worker-{i}", daemon=True) for i in range(worker_count)]
    for thread in threads:
        thread.start()
    return threads


def ingestion_job(dry_run: Optional[bool] = None) -> None:
    dry_run = config.DEFAULT_DRY_RUN if dry_run is None else bool(dry_run)
    logger.info("Ingestion job started. dry_run=%s", dry_run)

    try:
        ensure_embeddings_ready()
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini client initialisation failed: %s", exc)
        return

    try:
        ensure_summarizer_ready()
    except Exception as exc:  # noqa: BLE001
        logger.error("Summarizer configuration error: %s", exc)
        return

    try:
        ensure_collections()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to ensure Qdrant collections: %s", exc)
        return

    if not config.ONEDRIVE_DRIVE_ID:
        logger.error("ONEDRIVE_DRIVE_ID is not configured.")
        return

    try:
        token = get_graph_access_token()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to obtain Microsoft Graph token: %s", exc)
        return

    remote_files = list_onedrive_recursive(token, config.ONEDRIVE_DRIVE_ID, config.ONEDRIVE_ROOT_PATH)
    remote_by_id = {item["id"]: item for item in remote_files if item.get("id")}
    local_inventory = get_local_inventory()

    local_ids = set(local_inventory.keys())
    remote_ids = set(remote_by_id.keys())

    to_delete = local_ids - remote_ids
    to_consider = []
    for file_id in remote_ids:
        remote = remote_by_id[file_id]
        last_remote = (remote.get("lastModifiedDateTime") or "").strip()
        last_local = (local_inventory.get(file_id, {}).get("lastModified") or "").strip()
        if file_id not in local_inventory or (last_remote and last_remote != last_local):
            to_consider.append(remote)

    if not dry_run:
        for file_id in to_delete:
            delete_document_and_chunks(file_id)
    else:
        logger.info("Dry run enabled: %d documents would be deleted.", len(to_delete))

    if not to_consider:
        logger.info("No documents require upsert. Ingestion finished.")
        return

    if dry_run:
        to_consider = to_consider[:1]

    # Clean up old completed/failed entries
    cleanup_completed()

    # Filter out files that are already being processed
    busy_count = 0
    to_process = []
    for file_meta in to_consider:
        file_id = file_meta.get("id", "")
        file_name = file_meta.get("name", "unknown")
        if is_file_busy(file_id):
            logger.info("Skipping %s (already busy in current/previous ingestion run)", file_name)
            busy_count += 1
        else:
            to_process.append(file_meta)

    if busy_count > 0:
        logger.info("Skipped %d files that are currently being processed.", busy_count)

    if not to_process:
        logger.warning("All files are already being processed or pending. No work to do.")
        return

    # Apply MAX_DOCUMENTS limit (for development/testing)
    if config.MAX_DOCUMENTS > 0 and len(to_process) > config.MAX_DOCUMENTS:
        logger.info("Limiting ingestion to %d documents (MAX_DOCUMENTS=%d)", config.MAX_DOCUMENTS, config.MAX_DOCUMENTS)
        to_process = to_process[:config.MAX_DOCUMENTS]

    # Phase 1: Download all files to temp directory
    logger.info("Phase 1: Downloading %d files to temp directory...", len(to_process))
    temp_dir = tempfile.gettempdir()
    work_items: list[tuple[dict, str]] = []
    metadata_only_items: list[dict] = []

    for file_meta in to_process:
        file_id = file_meta.get("id", "")
        file_name = file_meta.get("name", "document.pdf")
        mime_type = file_meta.get("mimeType", "")
        
        # Check if this is a metadata-only file (no download needed)
        if not is_processable_with_ocr(mime_type):
            logger.info("Metadata-only file (no download): %s", file_name)
            metadata_only_items.append(file_meta)
            set_file_state(file_id, "enqueued", file_name)
            continue
        
        # Download files that need OCR processing (PDFs and images)
        try:
            set_file_state(file_id, "downloading", file_name)
            download_url = file_meta.get("@microsoft.graph.downloadUrl") or file_meta.get("downloadUrl")
            if not download_url:
                logger.warning("Missing downloadUrl for %s, skipping.", file_name)
                set_file_state(file_id, "failed", file_name)
                continue

            temp_file_path = os.path.join(temp_dir, f"{file_id}_{file_name}")
            download_file_to_path(download_url, temp_file_path)
            set_file_state(file_id, "enqueued", file_name)
            work_items.append((file_meta, temp_file_path))
            logger.info("Downloaded %s -> %s", file_name, temp_file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download %s: %s", file_name, exc)
            set_file_state(file_id, "failed", file_name)

    logger.info("Phase 1 complete. Downloaded %d files, %d metadata-only files.", 
                len(work_items), len(metadata_only_items))

    # Phase 2: Process metadata-only files (no download, no OCR)
    logger.info("Phase 2: Processing %d metadata-only files...", len(metadata_only_items))
    results: list[ProcessResult] = []
    
    for file_meta in metadata_only_items:
        file_id = file_meta.get("id", "")
        file_name = file_meta.get("name", "unknown")
        try:
            set_file_state(file_id, "processing", file_name)
            result = process_metadata_only_document(file_meta, dry_run=dry_run)
            results.append(result)
            
            if result.success:
                set_file_state(file_id, "completed", file_name)
            else:
                set_file_state(file_id, "failed", file_name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process metadata-only %s: %s", file_name, exc)
            set_file_state(file_id, "failed", file_name)
            results.append(
                ProcessResult(
                    file_id=file_id or "",
                    file_name=file_name,
                    success=False,
                    chunk_count=0,
                    summary="",
                    error=str(exc),
                )
            )
    
    if not work_items:
        logger.info("No OCR files to process. Ingestion finished.")
        success_count = sum(1 for result in results if result.success)
        failure_count = len(results) - success_count
        logger.info("Metadata-only processing complete. Success=%d Failures=%d", success_count, failure_count)
        return

    # Phase 3: Enqueue for worker processing (OCR files)
    logger.info("Phase 3: Enqueueing %d OCR files for processing...", len(work_items))
    queue: Queue = Queue()

    worker_count = min(config.INGESTION_WORKERS, max(1, len(work_items)))
    threads = _start_worker_threads(worker_count, queue, dry_run, results)

    for work_item in work_items:
        queue.put(work_item)

    for _ in threads:
        queue.put(None)

    queue.join()

    for thread in threads:
        thread.join()

    success_count = sum(1 for result in results if result.success)
    failure_count = len(results) - success_count
    logger.info("Ingestion job finished. Success=%d Failures=%d", success_count, failure_count)

    if dry_run and results:
        sample = results[0].dry_run_payload
        if sample:
            logger.info("Dry run payload sample:\n%s", json.dumps(sample, indent=2, ensure_ascii=False))
        
        logger.info("Dry run complete. Shutting down service...")
        os._exit(0)


# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/ingestion.log")
    ]
)

# Suppress verbose warnings from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
logging.getLogger("PyPDF2.generic._data_structures").setLevel(logging.ERROR)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

app = FastAPI(title="Ingestion Service", version="2.0.0")
scheduler = BackgroundScheduler()


@app.on_event("startup")
async def startup_event() -> None:
    # Ensure logs directory exists
    os.makedirs("/app/logs", exist_ok=True)
    
    cron_expr = config.SCHEDULE_CRON or "0 */3 * * *"
    try:
        ensure_embeddings_ready()
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini configuration error: %s", exc)
    try:
        ensure_summarizer_ready()
    except Exception as exc:  # noqa: BLE001
        logger.error("Summarizer configuration error: %s", exc)
    try:
        ensure_collections()
    except Exception as exc:  # noqa: BLE001
        logger.error("Unable to initialise Qdrant collections: %s", exc)
    try:
        scheduler.start()
        scheduler.add_job(
            ingestion_job,
            CronTrigger.from_crontab(cron_expr),
            id="ingest_onedrive",
            replace_existing=True,
        )
        logger.info("Scheduler started with cron %s", cron_expr)
        
        # Trigger immediate run in background
        from datetime import datetime, timedelta
        scheduler.add_job(
            ingestion_job,
            trigger="date",
            run_date=datetime.now() + timedelta(seconds=10),
            id="ingest_immediate_startup",
            replace_existing=True,
        )
        logger.info("Scheduled immediate ingestion run (in 10s).")
        
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start scheduler: %s", exc)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "qdrant": f"{config.QDRANT_HOST}:{config.QDRANT_PORT}"}


@app.post("/admin/ingest-now")
def ingest_now(background_tasks: BackgroundTasks, request: IngestNowRequest | None = None):
    dry_run = request.dry_run if request else None
    background_tasks.add_task(ingestion_job, dry_run=dry_run)
    effective_mode = config.DEFAULT_DRY_RUN if dry_run is None else bool(dry_run)
    return JSONResponse({"status": "queued", "dry_run": effective_mode})


if __name__ == "__main__":
    ingestion_job(dry_run=config.DEFAULT_DRY_RUN)
