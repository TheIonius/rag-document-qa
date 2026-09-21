import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional
from app.config import settings

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_db_connection(db_path: Path = None) -> sqlite3.Connection:
    target = db_path or settings.db_path
    conn = sqlite3.connect(str(target), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

@contextmanager
def get_db(db_path: Path = None) -> Generator[sqlite3.Connection, None, None]:
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db(db_path: Path = None):
    target = db_path or settings.db_path
    with get_db(target) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS organizations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            hashed_password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            is_superuser INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            name TEXT NOT NULL,
            slug TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS workspace_memberships (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
            UNIQUE(user_id, workspace_id)
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            page_count INTEGER NOT NULL DEFAULT 1,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            collection TEXT DEFAULT 'General Documentation',
            version INTEGER NOT NULL DEFAULT 1,
            sha256_checksum TEXT,
            storage_path TEXT,
            status TEXT NOT NULL DEFAULT 'READY',
            effective_from TEXT,
            effective_until TEXT,
            uploaded_by_id TEXT,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            section_title TEXT,
            page_number INTEGER NOT NULL DEFAULT 1,
            text TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            embedding_blob BLOB,
            created_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            document_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'QUEUED',
            progress_pct INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS query_logs (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            user_id TEXT,
            query TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            retrieved_chunks_json TEXT,
            answer TEXT NOT NULL,
            citations_json TEXT,
            confidence REAL NOT NULL,
            is_out_of_scope INTEGER NOT NULL DEFAULT 0,
            duration_ms INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            user_id TEXT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS thread_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations_json TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            actor_id TEXT,
            actor_email TEXT,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            ip_address TEXT,
            details_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            query_log_id TEXT,
            user_id TEXT,
            is_positive INTEGER NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id TEXT PRIMARY KEY,
            workspace_id TEXT,
            run_timestamp TEXT NOT NULL,
            total_cases INTEGER NOT NULL,
            passed_cases INTEGER NOT NULL,
            avg_latency_ms REAL NOT NULL,
            avg_groundedness REAL NOT NULL,
            results_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_logs(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_thread_messages_thread ON thread_messages(thread_id, created_at ASC);
        CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_events_created ON audit_events(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_doc ON ingestion_jobs(document_id);

        -- Enforce immutable, append-only compliance for audit ledger at database engine level
        CREATE TRIGGER IF NOT EXISTS trg_audit_events_no_update
        BEFORE UPDATE ON audit_events
        BEGIN
            SELECT RAISE(FAIL, 'Audit events are immutable and cannot be updated');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_audit_events_no_delete
        BEFORE DELETE ON audit_events
        BEGIN
            SELECT RAISE(FAIL, 'Audit events are immutable and cannot be deleted');
        END;
        """)

        # Add backwards-compatible column migrations for existing SQLite databases
        for col_def in [
            ("documents", "workspace_id TEXT"),
            ("documents", "version INTEGER DEFAULT 1"),
            ("documents", "sha256_checksum TEXT"),
            ("documents", "storage_path TEXT"),
            ("documents", "status TEXT DEFAULT 'READY'"),
            ("documents", "effective_from TEXT"),
            ("documents", "effective_until TEXT"),
            ("documents", "uploaded_by_id TEXT"),
            ("documents", "collection TEXT DEFAULT 'General Documentation'"),
            ("chunks", "embedding_blob BLOB"),
            ("chunks", "embedding_model TEXT DEFAULT 'BAAI/bge-small-en-v1.5'"),
            ("query_logs", "workspace_id TEXT"),
            ("query_logs", "user_id TEXT"),
            ("threads", "workspace_id TEXT"),
            ("threads", "user_id TEXT"),
        ]:
            table_name, col = col_def
            try:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col};")
            except sqlite3.OperationalError:
                pass

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_workspace ON documents(workspace_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_workspace ON query_logs(workspace_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_workspace ON threads(workspace_id);")
        except sqlite3.OperationalError:
            pass

        # Bootstrap default Organization, Workspace, and Admin User if none exist
        now = utc_now_iso()
        org_row = conn.execute("SELECT id FROM organizations WHERE slug = 'default'").fetchone()
        if not org_row:
            conn.execute("""
                INSERT INTO organizations (id, name, slug, created_at)
                VALUES ('org_default', 'Default Enterprise', 'default', ?)
            """, (now,))

        ws_row = conn.execute("SELECT id FROM workspaces WHERE id = 'ws_default'").fetchone()
        if not ws_row:
            conn.execute("""
                INSERT INTO workspaces (id, organization_id, name, slug, created_at)
                VALUES ('ws_default', 'org_default', 'Production Workspace', 'default', ?)
            """, (now,))

        # Update existing records to link to ws_default if NULL
        conn.execute("UPDATE documents SET workspace_id = 'ws_default' WHERE workspace_id IS NULL")
        conn.execute("UPDATE query_logs SET workspace_id = 'ws_default' WHERE workspace_id IS NULL")
        conn.execute("UPDATE threads SET workspace_id = 'ws_default' WHERE workspace_id IS NULL")

        admin_row = conn.execute("SELECT id FROM users WHERE email = 'admin@enterprise.local'").fetchone()
        if not admin_row:
            import os, secrets
            from app.security.auth import hash_password
            admin_pwd_env = os.getenv("CORTEX_ADMIN_PASSWORD")
            is_testing = bool(os.getenv("PYTEST_CURRENT_TEST"))
            if admin_pwd_env:
                admin_pwd = admin_pwd_env
            elif is_testing:
                admin_pwd = "TestAdminSecure123!"
            else:
                admin_pwd = secrets.token_urlsafe(16)
                bootstrap_file = settings.data_dir / ".admin_bootstrap"
                bootstrap_file.write_text(f"admin@enterprise.local:{admin_pwd}\n", encoding="utf-8")

            hashed_admin_pwd = hash_password(admin_pwd)
            conn.execute("""
                INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at)
                VALUES ('usr_admin', 'admin@enterprise.local', ?, 'Enterprise Administrator', 1, 1, ?)
            """, (hashed_admin_pwd, now))

            conn.execute("""
                INSERT OR IGNORE INTO workspace_memberships (id, user_id, workspace_id, role, created_at)
                VALUES ('mem_admin_default', 'usr_admin', 'ws_default', 'owner', ?)
            """, (now,))

def record_audit_event(
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    workspace_id: Optional[str] = "ws_default",
    actor_id: Optional[str] = None,
    actor_email: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[dict] = None
):
    """Writes an immutable audit ledger entry for security, compliance, and governance tracking."""
    event_id = f"aud_{uuid.uuid4().hex[:12]}"
    now = utc_now_iso()
    details_str = json.dumps(details or {})
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO audit_events (
                    id, workspace_id, actor_id, actor_email, action,
                    resource_type, resource_id, ip_address, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id, workspace_id, actor_id, actor_email, action,
                resource_type, resource_id, ip_address, details_str, now
            ))
    except Exception as e:
        print(f"[Audit Warning] Failed to write audit event: {e}")
