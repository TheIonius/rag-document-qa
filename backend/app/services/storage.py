import hashlib
import os
import re
from pathlib import Path
from typing import Optional, Tuple
from app.config import settings
from app.database import get_db

def sanitize_filename(filename: str) -> str:
    # Strip any path separators and unsafe characters
    base = os.path.basename(filename)
    safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    return safe[:100]

def sanitize_workspace_id(workspace_id: str) -> str:
    if not workspace_id:
        return "ws_default"
    if ".." in workspace_id or "/" in workspace_id or "\\" in workspace_id:
        raise ValueError(f"Invalid path traversal attempt for workspace: '{workspace_id}'")
    clean = re.sub(r'[^a-zA-Z0-9_-]', '', workspace_id)
    if not clean:
        raise ValueError(f"Invalid workspace identifier: '{workspace_id}'")
    return clean

class StorageService:
    def __init__(self, base_storage_dir: Optional[Path] = None):
        self.base_dir = (base_storage_dir or settings.storage_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_workspace_dir(self, workspace_id: str) -> Path:
        clean_ws = sanitize_workspace_id(workspace_id)
        ws_dir = (self.base_dir / clean_ws).resolve()
        try:
            # Enforce strictly that ws_dir is inside self.base_dir
            if os.path.commonpath([str(self.base_dir), str(ws_dir)]) != str(self.base_dir):
                raise ValueError(f"Path traversal detected for workspace: {workspace_id}")
        except ValueError:
            raise ValueError(f"Invalid path traversal attempt for workspace: {workspace_id}")
        ws_dir.mkdir(parents=True, exist_ok=True)
        return ws_dir

    def compute_sha256(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def check_duplicate(self, workspace_id: str, sha256_checksum: str) -> Optional[dict]:
        clean_ws = sanitize_workspace_id(workspace_id)
        with get_db() as conn:
            row = conn.execute("""
                SELECT id, filename, title, version, created_at, status, file_type, file_size, chunk_count, page_count
                FROM documents
                WHERE workspace_id = ? AND sha256_checksum = ? AND status != 'DELETED'
                LIMIT 1
            """, (clean_ws, sha256_checksum)).fetchone()
            if row:
                return dict(row)
        return None

    def save_file(
        self,
        workspace_id: str,
        filename: str,
        content: bytes
    ) -> Tuple[str, str, Optional[dict]]:
        """
        Saves uploaded file content to disk under workspace partition.
        Enforces path containment and checks for duplicates.
        Returns: (storage_path, sha256_checksum, existing_duplicate_doc)
        """
        checksum = self.compute_sha256(content)
        existing_doc = self.check_duplicate(workspace_id, checksum)

        safe_name = sanitize_filename(filename)
        ws_dir = self._get_workspace_dir(workspace_id)

        storage_filename = f"{checksum[:16]}_{safe_name}"
        storage_path = (ws_dir / storage_filename).resolve()

        if os.path.commonpath([str(ws_dir), str(storage_path)]) != str(ws_dir):
            raise ValueError("Path traversal attempt in storage filename")

        if not storage_path.exists():
            with open(storage_path, "wb") as f:
                f.write(content)

        return str(storage_path), checksum, existing_doc

    def read_file(self, storage_path: str) -> bytes:
        p = Path(storage_path).resolve()
        if os.path.commonpath([str(self.base_dir), str(p)]) != str(self.base_dir):
            raise PermissionError("Access denied: path outside storage directory")
        if not p.exists():
            raise FileNotFoundError(f"Storage file not found: {storage_path}")
        return p.read_bytes()

    def delete_file(self, storage_path: Optional[str]):
        if storage_path:
            p = Path(storage_path).resolve()
            if os.path.commonpath([str(self.base_dir), str(p)]) != str(self.base_dir):
                return
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

storage_service = StorageService()
