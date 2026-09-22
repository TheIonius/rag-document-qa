import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_db, record_audit_event, utc_now_iso
from app.models.schemas import (
    WorkspaceCreateRequest,
    WorkspaceMemberAdd,
    WorkspaceMemberResponse,
    WorkspaceResponse
)
from app.security.auth import (
    get_current_user,
    get_optional_user,
    require_role
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

@router.get("", response_model=List[WorkspaceResponse])
def list_workspaces(user: dict = Depends(get_current_user)):
    with get_db() as conn:
        if user.get("is_superuser"):
            rows = conn.execute("""
                SELECT w.id, w.organization_id, w.name, w.slug, w.created_at, 'owner' as role
                FROM workspaces w
                ORDER BY w.created_at ASC
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT w.id, w.organization_id, w.name, w.slug, w.created_at, m.role
                FROM workspaces w
                JOIN workspace_memberships m ON w.id = m.workspace_id
                WHERE m.user_id = ?
                ORDER BY w.created_at ASC
            """, (user["id"],)).fetchall()
            if not any(r["id"] == "ws_default" for r in rows):
                demo_ws = conn.execute("""
                    SELECT id, organization_id, name || ' (Demo Sandbox)' as name, slug, created_at, 'viewer' as role
                    FROM workspaces WHERE id = 'ws_default'
                """).fetchone()
                if demo_ws:
                    rows = list(rows) + [demo_ws]

    return [WorkspaceResponse.model_validate(dict(r)) for r in rows]

@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(req: WorkspaceCreateRequest, user: dict = Depends(get_current_user)):
    now = utc_now_iso()
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    mem_id = f"mem_{uuid.uuid4().hex[:12]}"
    slug = req.slug.strip().lower() if req.slug else req.name.strip().lower().replace(" ", "-")

    with get_db() as conn:
        # Find user's organization
        first_ws = conn.execute("""
            SELECT w.organization_id FROM workspaces w
            JOIN workspace_memberships m ON w.id = m.workspace_id
            WHERE m.user_id = ? LIMIT 1
        """, (user["id"],)).fetchone()
        org_id = first_ws["organization_id"] if first_ws else "org_default"

        conn.execute("""
            INSERT INTO workspaces (id, organization_id, name, slug, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (ws_id, org_id, req.name.strip(), slug, now))

        conn.execute("""
            INSERT INTO workspace_memberships (id, user_id, workspace_id, role, created_at)
            VALUES (?, ?, ?, 'owner', ?)
        """, (mem_id, user["id"], ws_id, now))

    record_audit_event(
        action="workspace_create",
        resource_type="workspace",
        resource_id=ws_id,
        workspace_id=ws_id,
        actor_id=user["id"],
        actor_email=user["email"],
        details={"name": req.name, "slug": slug}
    )

    return WorkspaceResponse(
        id=ws_id,
        organization_id=org_id,
        name=req.name.strip(),
        slug=slug,
        created_at=now,
        role="owner"
    )

@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(workspace_id: str, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Workspace not found")

        # Verify tenant isolation membership
        if not user.get("is_superuser"):
            mem = conn.execute("""
                SELECT role FROM workspace_memberships WHERE workspace_id = ? AND user_id = ?
            """, (workspace_id, user["id"])).fetchone()
            if not mem:
                raise HTTPException(status_code=403, detail="Access denied to this workspace.")

        data = dict(row)
        data["role"] = "owner" if user.get("is_superuser") else mem["role"]
        return WorkspaceResponse.model_validate(data)

@router.get("/{workspace_id}/members", response_model=List[WorkspaceMemberResponse])
def get_workspace_members(workspace_id: str, user: dict = Depends(get_current_user)):
    with get_db() as conn:
        # Check that caller belongs to this workspace
        if not user.get("is_superuser"):
            caller_mem = conn.execute("""
                SELECT id FROM workspace_memberships WHERE workspace_id = ? AND user_id = ?
            """, (workspace_id, user["id"])).fetchone()
            if not caller_mem:
                raise HTTPException(status_code=403, detail="Access denied: you are not a member of this workspace.")

        rows = conn.execute("""
            SELECT m.id, m.user_id, m.workspace_id, m.role, m.created_at,
                   u.email, u.full_name
            FROM workspace_memberships m
            JOIN users u ON m.user_id = u.id
            WHERE m.workspace_id = ?
            ORDER BY m.created_at ASC
        """, (workspace_id,)).fetchall()

    return [
        WorkspaceMemberResponse(
            id=r["id"],
            user_id=r["user_id"],
            workspace_id=r["workspace_id"],
            role=r["role"],
            email=r["email"],
            full_name=r["full_name"],
            created_at=r["created_at"]
        )
        for r in rows
    ]

@router.post("/{workspace_id}/members", response_model=WorkspaceMemberResponse, status_code=status.HTTP_201_CREATED)
def add_workspace_member(
    workspace_id: str,
    req: WorkspaceMemberAdd,
    admin_auth: dict = Depends(require_role("admin"))
):
    now = utc_now_iso()
    mem_id = f"mem_{uuid.uuid4().hex[:12]}"
    email = req.email.strip().lower()

    with get_db() as conn:
        user_row = conn.execute("SELECT id, email, full_name FROM users WHERE email = ?", (email,)).fetchone()
        if not user_row:
            raise HTTPException(status_code=404, detail="User with this email not found. They must register first.")

        user_id = user_row["id"]
        existing = conn.execute("""
            SELECT id FROM workspace_memberships WHERE workspace_id = ? AND user_id = ?
        """, (workspace_id, user_id)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="User is already a member of this workspace.")

        conn.execute("""
            INSERT INTO workspace_memberships (id, user_id, workspace_id, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (mem_id, user_id, workspace_id, req.role, now))

    caller_user = admin_auth.get("user", {})
    record_audit_event(
        action="workspace_member_add",
        resource_type="workspace",
        resource_id=workspace_id,
        workspace_id=workspace_id,
        actor_id=caller_user.get("id"),
        actor_email=caller_user.get("email"),
        details={"member_email": email, "role": req.role}
    )

    return WorkspaceMemberResponse(
        id=mem_id,
        user_id=user_id,
        workspace_id=workspace_id,
        role=req.role,
        email=user_row["email"],
        full_name=user_row["full_name"],
        created_at=now
    )

@router.delete("/{workspace_id}/members/{member_id}", status_code=status.HTTP_200_OK)
def remove_workspace_member(
    workspace_id: str,
    member_id: str,
    admin_auth: dict = Depends(require_role("admin"))
):
    caller_user = admin_auth.get("user", {})
    with get_db() as conn:
        mem = conn.execute("""
            SELECT id, user_id, role FROM workspace_memberships WHERE id = ? AND workspace_id = ?
        """, (member_id, workspace_id)).fetchone()
        if not mem:
            raise HTTPException(status_code=404, detail="Workspace membership record not found.")
        if mem["role"] == "owner" and not caller_user.get("is_superuser"):
            raise HTTPException(status_code=400, detail="Cannot remove workspace owner.")
        if mem["user_id"] == caller_user.get("id"):
            raise HTTPException(status_code=400, detail="Cannot remove yourself from the workspace.")
        conn.execute("DELETE FROM workspace_memberships WHERE id = ?", (member_id,))

    record_audit_event(
        action="workspace_member_remove",
        resource_type="workspace",
        resource_id=workspace_id,
        workspace_id=workspace_id,
        actor_id=caller_user.get("id"),
        actor_email=caller_user.get("email"),
        details={"removed_member_id": member_id}
    )
    return {"status": "success", "message": "Member successfully removed from workspace."}

