import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from app.database import get_db, record_audit_event, utc_now_iso
from app.models.schemas import (
    ResearchThread,
    ThreadDetail,
    ThreadMessage,
    ThreadCreateRequest,
    ThreadUpdateRequest,
    Citation
)
from app.security.auth import get_optional_user

router = APIRouter(prefix="/threads", tags=["threads"])

@router.get("", response_model=List[ResearchThread])
def list_threads(
    workspace_id: Optional[str] = Query(None),
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: Optional[dict] = Depends(get_optional_user)
):
    active_ws = workspace_id or x_workspace_id or "ws_default"
    with get_db() as conn:
        if active_ws != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to view threads.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], active_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this workspace.")
        if active_ws == "ws_default":
            if user:
                if user.get("is_superuser"):
                    rows = conn.execute("""
                        SELECT 
                            t.id, t.title, t.created_at, t.updated_at, t.workspace_id,
                            COUNT(m.id) as message_count,
                            COALESCE(SUM(CASE WHEN m.citations_json IS NOT NULL AND m.citations_json != '[]' THEN 1 ELSE 0 END), 0) as citations_count
                        FROM threads t
                        LEFT JOIN thread_messages m ON t.id = m.thread_id
                        WHERE t.workspace_id = 'ws_default' OR t.workspace_id IS NULL
                        GROUP BY t.id
                        ORDER BY t.updated_at DESC
                    """).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT 
                            t.id, t.title, t.created_at, t.updated_at, t.workspace_id,
                            COUNT(m.id) as message_count,
                            COALESCE(SUM(CASE WHEN m.citations_json IS NOT NULL AND m.citations_json != '[]' THEN 1 ELSE 0 END), 0) as citations_count
                        FROM threads t
                        LEFT JOIN thread_messages m ON t.id = m.thread_id
                        WHERE (t.workspace_id = 'ws_default' OR t.workspace_id IS NULL)
                          AND (t.user_id = ? OR t.user_id IS NULL)
                        GROUP BY t.id
                        ORDER BY t.updated_at DESC
                    """, (user["id"],)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT 
                        t.id, t.title, t.created_at, t.updated_at, t.workspace_id,
                        COUNT(m.id) as message_count,
                        COALESCE(SUM(CASE WHEN m.citations_json IS NOT NULL AND m.citations_json != '[]' THEN 1 ELSE 0 END), 0) as citations_count
                    FROM threads t
                    LEFT JOIN thread_messages m ON t.id = m.thread_id
                    WHERE (t.workspace_id = 'ws_default' OR t.workspace_id IS NULL)
                      AND t.user_id IS NULL
                    GROUP BY t.id
                    ORDER BY t.updated_at DESC
                """).fetchall()
        else:
            rows = conn.execute("""
                SELECT 
                    t.id, t.title, t.created_at, t.updated_at, t.workspace_id,
                    COUNT(m.id) as message_count,
                    COALESCE(SUM(CASE WHEN m.citations_json IS NOT NULL AND m.citations_json != '[]' THEN 1 ELSE 0 END), 0) as citations_count
                FROM threads t
                LEFT JOIN thread_messages m ON t.id = m.thread_id
                WHERE t.workspace_id = ?
                GROUP BY t.id
                ORDER BY t.updated_at DESC
            """, (active_ws,)).fetchall()

    results = []
    for r in rows:
        results.append(ResearchThread(
            id=r["id"],
            title=r["title"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            message_count=r["message_count"],
            citations_count=r["citations_count"],
            workspace_id=r["workspace_id"] or "ws_default"
        ))
    return results

@router.post("", response_model=ResearchThread, status_code=status.HTTP_201_CREATED)
def create_thread(
    req: ThreadCreateRequest,
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: Optional[dict] = Depends(get_optional_user)
):
    active_ws = req.workspace_id or x_workspace_id or "ws_default"
    thread_id = f"th_{uuid.uuid4().hex[:12]}"
    now = utc_now_iso()
    title = req.title.strip() if req.title and req.title.strip() else "New Investigation"
    user_id = user["id"] if user else None

    with get_db() as conn:
        conn.execute("""
            INSERT INTO threads (id, workspace_id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (thread_id, active_ws, user_id, title, now, now))

    record_audit_event(
        action="thread_create",
        resource_type="thread",
        resource_id=thread_id,
        workspace_id=active_ws,
        details={"title": title}
    )

    return ResearchThread(
        id=thread_id,
        title=title,
        created_at=now,
        updated_at=now,
        message_count=0,
        citations_count=0,
        workspace_id=active_ws
    )

@router.get("/{thread_id}", response_model=ThreadDetail)
def get_thread_detail(thread_id: str, user: Optional[dict] = Depends(get_optional_user)):
    with get_db() as conn:
        thread_row = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not thread_row:
            raise HTTPException(status_code=404, detail="Thread not found")

        th_ws = thread_row["workspace_id"] or "ws_default"
        if th_ws != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to view this thread.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], th_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this thread.")

        msg_rows = conn.execute("""
            SELECT id, thread_id, role, content, citations_json, metadata_json, created_at
            FROM thread_messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
        """, (thread_id,)).fetchall()

    messages = []
    total_citations = 0
    for mr in msg_rows:
        citations = []
        if mr["citations_json"]:
            try:
                raw_cites = json.loads(mr["citations_json"])
                citations = [Citation(**c) for c in raw_cites]
                total_citations += len(citations)
            except Exception:
                citations = []

        meta = {}
        if mr["metadata_json"]:
            try:
                meta = json.loads(mr["metadata_json"])
            except Exception:
                meta = {}

        messages.append(ThreadMessage(
            id=mr["id"],
            thread_id=mr["thread_id"],
            role=mr["role"],
            content=mr["content"],
            citations=citations,
            metadata=meta,
            created_at=mr["created_at"]
        ))

    thread_model = ResearchThread(
        id=thread_row["id"],
        title=thread_row["title"],
        created_at=thread_row["created_at"],
        updated_at=thread_row["updated_at"],
        message_count=len(messages),
        citations_count=total_citations,
        workspace_id=thread_row["workspace_id"] or "ws_default"
    )

    return ThreadDetail(thread=thread_model, messages=messages)

@router.patch("/{thread_id}", response_model=ResearchThread)
def update_thread(thread_id: str, req: ThreadUpdateRequest, user: Optional[dict] = Depends(get_optional_user)):
    now = utc_now_iso()
    new_title = req.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Thread title cannot be empty")

    with get_db() as conn:
        row = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Thread not found")

        th_ws = row["workspace_id"] or "ws_default"
        if th_ws != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to modify this thread.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], th_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this thread.")

        conn.execute("UPDATE threads SET title = ?, updated_at = ? WHERE id = ?", (new_title, now, thread_id))

    return ResearchThread(
        id=thread_id,
        title=new_title,
        created_at=row["created_at"],
        updated_at=now,
        message_count=0,
        citations_count=0,
        workspace_id=row["workspace_id"] or "ws_default"
    )

@router.delete("/{thread_id}")
def delete_thread(thread_id: str, user: Optional[dict] = Depends(get_optional_user)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Thread not found")

        th_ws = row["workspace_id"] or "ws_default"
        if th_ws != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to delete this thread.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], th_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this thread.")

        conn.execute("DELETE FROM thread_messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))

    record_audit_event(
        action="thread_delete",
        resource_type="thread",
        resource_id=thread_id,
        workspace_id=row["workspace_id"] or "ws_default",
        actor_id=user["id"] if user else None,
        actor_email=user["email"] if user else None
    )

    return {"status": "success", "deleted_thread_id": thread_id}
