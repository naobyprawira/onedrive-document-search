from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple
from urllib.parse import quote

import requests

import config

logger = logging.getLogger("ingestion.graph")


def normalize_drive_path(path: str) -> str:
    """Normalise input paths for Graph API calls."""
    cleaned = (path or "").strip()
    if cleaned.lower().startswith("root/"):
        cleaned = cleaned[5:]
    return cleaned.strip("/")


def get_graph_access_token() -> str:
    """Obtain an Azure AD access token via client credentials."""
    if not (config.MS_TENANT_ID and config.MS_CLIENT_ID and config.MS_CLIENT_SECRET):
        raise RuntimeError("Microsoft Graph credentials are not fully configured.")

    token_url = f"https://login.microsoftonline.com/{config.MS_TENANT_ID}/oauth2/v2.0/token"
    payload = {
        "client_id": config.MS_CLIENT_ID,
        "client_secret": config.MS_CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": config.GRAPH_SCOPE,
    }

    try:
        response = requests.post(token_url, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        access_token = data.get("access_token")
        if not access_token:
            raise RuntimeError("Graph token response did not include an access_token.")
        return access_token
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to obtain Graph token: {exc}") from exc


def _graph_get(url: str, token: str, params: dict | None = None) -> dict:
    """Perform a GET request with simple retry handling for 429 codes."""
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(5):
        response = requests.get(url, headers=headers, params=params, timeout=500)
        if response.status_code == 429:
            wait_seconds = int(response.headers.get("Retry-After", "2"))
            logger.warning("Graph 429 on %s. Waiting %s seconds before retry.", url, wait_seconds)
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.json()

    response.raise_for_status()
    return {}


def list_onedrive_recursive(token: str, drive_id: str, root_path: str) -> List[dict]:
    """
    Depth-first traversal of OneDrive folders under root_path, returning PDF file entries.
    """
    results: List[dict] = []
    normalized_root = normalize_drive_path(root_path)
    encoded_root = quote(normalized_root, safe="/")

    root_url = f"{config.GRAPH_BASE_URL}/drives/{drive_id}/root:/{encoded_root}"
    root_item = _graph_get(root_url, token)
    if "id" not in root_item:
        logger.error("OneDrive root path not found: %s", root_path)
        return results

    stack: List[Tuple[str, str]] = [(root_item["id"], normalized_root)]
    while stack:
        parent_id, parent_path = stack.pop()
        next_url = f"{config.GRAPH_BASE_URL}/drives/{drive_id}/items/{parent_id}/children"

        while next_url:
            data = _graph_get(next_url, token)
            for item in data.get("value", []):
                name = item.get("name", "")
                item_id = item.get("id")
                if not item_id:
                    continue

                child_path = f"{parent_path}/{name}"
                if item.get("folder"):
                    stack.append((item_id, child_path))
                    continue

                # Include all file types, not just PDFs
                mime_type = item.get("file", {}).get("mimeType") if item.get("file") else None
                
                results.append(
                    {
                        "id": item_id,
                        "name": name,
                        "webUrl": item.get("webUrl"),
                        "downloadUrl": item.get("@microsoft.graph.downloadUrl"),
                        "size": item.get("size", 0),
                        "mimeType": mime_type,
                        "lastModifiedDateTime": item.get("lastModifiedDateTime"),
                        "drivePath": child_path,
                    }
                )

            next_url = data.get("@odata.nextLink")

    return results


def download_file_to_path(download_url: str, target_path: str) -> None:
    """Download file to target path using streaming to avoid loading into memory."""
    with requests.get(download_url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

