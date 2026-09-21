import json
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from app.database import get_db
from app.models.schemas import AuditEventResponse
from app.security.auth import get_current_user

router = APIRouter(prefix="/audit", tags=["audit"])

@router.get("", response_model=List[AuditEventResponse])
def get_audit_ledger(
    limit: int = Query(50, ge=1, le=200),
    action: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: dict = Depends(get_current_user)
):
    active_ws = workspace_id or x_workspace_id
    query_sql = "SELECT * FROM audit_events WHERE 1=1"
    params = []

    with get_db() as conn:
        if not user.get("is_superuser"):
            # Check user membership
            if active_ws:
                mem = conn.execute("""
                    SELECT id FROM workspace_memberships WHERE workspace_id = ? AND user_id = ?
                """, (active_ws, user["id"])).fetchone()
                if not mem:
                    raise HTTPException(status_code=403, detail="Access denied: you do not have permission to view audit logs for this workspace.")
                query_sql += " AND workspace_id = ?"
                params.append(active_ws)
            else:
                user_workspaces = [
                    r["workspace_id"] for r in conn.execute(
                        "SELECT workspace_id FROM workspace_memberships WHERE user_id = ?",
                        (user["id"],)
                    ).fetchall()
                ]
                if not user_workspaces:
                    return []
                placeholders = ",".join("?" for _ in user_workspaces)
                query_sql += f" AND (workspace_id IN ({placeholders}) OR workspace_id IS NULL)"
                params.extend(user_workspaces)
        else:
            if active_ws:
                query_sql += " AND (workspace_id = ? OR workspace_id IS NULL)"
                params.append(active_ws)

    if action:
        query_sql += " AND action = ?"
        params.append(action)

    query_sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query_sql, params).fetchall()

    events = []
    for r in rows:
        details_obj = None
        if r["details_json"]:
            try:
                details_obj = json.loads(r["details_json"])
            except Exception:
                details_obj = None

        events.append(AuditEventResponse(
            id=r["id"],
            workspace_id=r["workspace_id"],
            actor_id=r["actor_id"],
            actor_email=r["actor_email"],
            action=r["action"],
            resource_type=r["resource_type"],
            resource_id=r["resource_id"],
            ip_address=r["ip_address"],
            details=details_obj,
            created_at=r["created_at"]
        ))

    return events
