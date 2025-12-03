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

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/ingestion.log")
    ]
)

logger = logging.getLogger("ingestion")

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


def is_archive(mime_type: str, file_name: str) -> bool:
    """Check if file is an archive (ZIP, RAR, 7Z)."""
    return (mime_type in [
        "application/zip", 
        "application/x-zip-compressed", 
        "application/x-rar-compressed", 
        "application/x-7z-compressed"
    ]) or file_name.lower().endswith((".zip", ".rar", ".7z"))

def ingestion_job(dry_run: Optional[bool] = None) -> None:
    dry_run = config.DEFAULT_DRY_RUN if dry_run is None else bool(dry_run)
    logger.info("Ingestion job started. dry_run=%s", dry_run)

    try:
        ensure_embeddings_ready()
        ensure_summarizer_ready()
        ensure_collections()
    except Exception as exc:
        logger.error("Initialization failed: %s", exc)
        return

    if not config.ONEDRIVE_DRIVE_ID:
        logger.error("ONEDRIVE_DRIVE_ID is not configured.")
        return

    try:
        token = get_graph_access_token()
    except Exception as exc:
        logger.error("Failed to obtain Microsoft Graph token: %s", exc)
        return

    # --- Discovery Phase ---
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
        logger.info("Dry run: %d documents would be deleted.", len(to_delete))

    if not to_consider:
        logger.info("No documents require upsert. Ingestion finished.")
        return

    if dry_run:
        to_consider = to_consider[:1]

    cleanup_completed()

    # Filter busy files
    to_process = []
    busy_count = 0
    for file_meta in to_consider:
        if is_file_busy(file_meta.get("id", "")):
            busy_count += 1
        else:
            to_process.append(file_meta)
    
    if busy_count:
        logger.info("Skipped %d busy files.", busy_count)
    
    if not to_process:
        logger.info("No work to do.")
        return

    if config.MAX_DOCUMENTS > 0:
        to_process = to_process[:config.MAX_DOCUMENTS]

    logger.info("Starting processing of %d files...", len(to_process))

    # --- Producer-Consumer Setup ---
    # Use a bounded queue to control memory usage and download rate (effectively batching)
    queue: Queue = Queue(maxsize=10) 
    results: list[ProcessResult] = []
    
    # Start worker threads immediately
    worker_count = min(config.INGESTION_WORKERS, max(1, len(to_process)))
    threads = _start_worker_threads(worker_count, queue, dry_run, results)

    temp_dir = tempfile.gettempdir()
    import patoolib
    import shutil

    # --- Producer Loop ---
    for file_meta in to_process:
        file_id = file_meta.get("id", "")
        file_name = file_meta.get("name", "unknown")
        mime_type = file_meta.get("mimeType", "")
        
        # Case 1: Metadata-only (non-processable, non-archive)
        if not is_processable_with_ocr(mime_type) and not is_archive(mime_type, file_name):
            logger.info("Processing metadata-only: %s", file_name)
            try:
                set_file_state(file_id, "processing", file_name)
                res = process_metadata_only_document(file_meta, dry_run=dry_run)
                results.append(res)
                set_file_state(file_id, "completed" if res.success else "failed", file_name)
            except Exception as exc:
                logger.error("Metadata processing failed for %s: %s", file_name, exc)
                set_file_state(file_id, "failed", file_name)
            continue

        # Case 2: Download required (PDF/Image/Archive)
        try:
            set_file_state(file_id, "downloading", file_name)
            download_url = file_meta.get("@microsoft.graph.downloadUrl") or file_meta.get("downloadUrl")
            
            if not download_url:
                logger.warning("No download URL for %s", file_name)
                set_file_state(file_id, "failed", file_name)
                continue

            temp_file_path = os.path.join(temp_dir, f"{file_id}_{file_name}")
            
            # Optimized Download: Check existence first
            if os.path.exists(temp_file_path):
                # Optional: Check size if possible, but for now assume existence = valid cache
                logger.info("File exists in temp, skipping download: %s", file_name)
            else:
                download_file_to_path(download_url, temp_file_path)
                logger.info("Downloaded %s", file_name)

            # Case 2a: Archive Handling
            if is_archive(mime_type, file_name):
                logger.info("Extracting archive: %s", file_name)
                extract_dir = os.path.join(temp_dir, f"extract_{file_id}")
                os.makedirs(extract_dir, exist_ok=True)
                try:
                    patoolib.extract_archive(temp_file_path, outdir=extract_dir, verbosity=-1)
                    
                    # Walk extracted files
                    for root, _, files in os.walk(extract_dir):
                        for sub_file in files:
                            sub_path = os.path.join(root, sub_file)
                            rel_path = os.path.relpath(sub_path, extract_dir)
                            
                            # Skip MacOS resource forks etc
                            if sub_file.startswith(".") or "__MACOSX" in sub_path:
                                continue
                                
                            # Only process supported types inside archive
                            # Simple mime inference by extension
                            sub_ext = os.path.splitext(sub_file)[1].lower()
                            if sub_ext not in [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                                continue
                                
                            # Create synthetic metadata
                            syn_id = f"{file_id}::{rel_path}"
                            syn_meta = file_meta.copy()
                            syn_meta.update({
                                "id": syn_id,
                                "name": f"{file_name}/{rel_path}",
                                "drivePath": f"{file_meta.get('drivePath', '')}/{rel_path}",
                                "mimeType": "application/pdf" if sub_ext == ".pdf" else "image/jpeg"
                            })
                            
                            # Queue extracted file
                            # Note: We pass sub_path. Worker will clean it up, so we might need to copy it 
                            # or ensure worker doesn't delete the whole extract dir prematurely.
                            # Actually, worker deletes temp_file_path. 
                            # We should probably copy to a unique temp file for the worker to own.
                            worker_temp_path = os.path.join(temp_dir, f"worker_{syn_id.replace('::', '_')}_{sub_file}")
                            shutil.copy2(sub_path, worker_temp_path)
                            
                            set_file_state(syn_id, "enqueued", syn_meta["name"])
                            queue.put((syn_meta, worker_temp_path))
                            
                except Exception as exc:
                    logger.error("Archive extraction failed for %s: %s", file_name, exc)
                finally:
                    # Cleanup archive and extraction dir
                    if os.path.exists(extract_dir):
                        shutil.rmtree(extract_dir, ignore_errors=True)
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    set_file_state(file_id, "completed", file_name) # Mark archive itself as done

            # Case 2b: Regular PDF/Image
            else:
                set_file_state(file_id, "enqueued", file_name)
                queue.put((file_meta, temp_file_path))

        except Exception as exc:
            logger.error("Download/Dispatch failed for %s: %s", file_name, exc)
            set_file_state(file_id, "failed", file_name)

    # --- Shutdown ---
    # Signal workers to stop
    for _ in threads:
        queue.put(None)
    
    # Wait for all tasks to complete
    queue.join()
    for thread in threads:
        thread.join()

    success_count = sum(1 for r in results if r.success)
    logger.info("Ingestion job finished. Success=%d Failures=%d", success_count, len(results) - success_count)
    
    if dry_run and results:
        logger.info("Dry run complete.")
        os._exit(0)




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
