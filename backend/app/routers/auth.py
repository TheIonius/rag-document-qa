import uuid
from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_db, record_audit_event, utc_now_iso
from app.models.schemas import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse
)
from app.security.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password
)

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register_user(req: UserRegisterRequest):
    raw_identifier = req.email.strip().lower()
    if not raw_identifier:
        raise HTTPException(status_code=400, detail="Username or email cannot be empty.")
    email = raw_identifier if "@" in raw_identifier else f"{raw_identifier}@enterprise.local"
    display_name = (req.full_name or "").strip() or raw_identifier.title()

    now = utc_now_iso()
    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    org_id = f"org_{uuid.uuid4().hex[:12]}"
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    mem_id = f"mem_{uuid.uuid4().hex[:12]}"

    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ? OR email = ?", (email, raw_identifier)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="A user with this username or email already exists.")

        user_count = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"]
        is_superuser = 1 if user_count == 0 else 0

        hashed = hash_password(req.password)
        conn.execute("""
            INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
        """, (user_id, email, hashed, display_name, is_superuser, now))

        # Create personal organization & workspace
        org_name = f"{display_name}'s Organization"
        org_slug = f"org-{uuid.uuid4().hex[:8]}"
        conn.execute("""
            INSERT INTO organizations (id, name, slug, created_at)
            VALUES (?, ?, ?, ?)
        """, (org_id, org_name, org_slug, now))

        ws_name = f"{display_name}'s Workspace"
        ws_slug = f"ws-{uuid.uuid4().hex[:8]}"
        conn.execute("""
            INSERT INTO workspaces (id, organization_id, name, slug, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (ws_id, org_id, ws_name, ws_slug, now))

        conn.execute("""
            INSERT INTO workspace_memberships (id, user_id, workspace_id, role, created_at)
            VALUES (?, ?, ?, 'owner', ?)
        """, (mem_id, user_id, ws_id, now))

    record_audit_event(
        action="user_register",
        resource_type="user",
        resource_id=user_id,
        workspace_id=ws_id,
        actor_id=user_id,
        actor_email=email
    )

    token = create_access_token({"sub": user_id, "email": email})
    user_resp = UserResponse(
        id=user_id,
        email=email,
        full_name=display_name,
        is_active=True,
        is_superuser=bool(is_superuser),
        created_at=now
    )
    return TokenResponse(access_token=token, token_type="bearer", user=user_resp)

@router.post("/login", response_model=TokenResponse)
def login_user(req: UserLoginRequest):
    raw_identifier = req.email.strip().lower()
    email = raw_identifier if "@" in raw_identifier else f"{raw_identifier}@enterprise.local"
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ? OR email = ?", (email, raw_identifier)).fetchone()
        if not row or not verify_password(req.password, row["hashed_password"]):
            record_audit_event(
                action="login_failed",
                resource_type="user",
                resource_id=row["id"] if row else None,
                actor_email=raw_identifier,
                details={"reason": "invalid_credentials"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials. Please verify your email and password.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        if not row["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your user account has been deactivated."
            )

        user_id = row["id"]
        full_name = row["full_name"]
        user_email = row["email"]
        is_super = bool(row["is_superuser"])
        created_at = row["created_at"]

    record_audit_event(
        action="user_login",
        resource_type="user",
        resource_id=user_id,
        actor_id=user_id,
        actor_email=user_email
    )

    token = create_access_token({"sub": user_id, "email": user_email})
    user_resp = UserResponse(
        id=user_id,
        email=user_email,
        full_name=full_name,
        is_active=True,
        is_superuser=is_super,
        created_at=created_at
    )
    return TokenResponse(access_token=token, token_type="bearer", user=user_resp)

@router.get("/me", response_model=UserResponse)
def get_me(user: dict = Depends(get_current_user)):
    return UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        is_active=bool(user.get("is_active", True)),
        is_superuser=bool(user.get("is_superuser", False)),
        created_at=user["created_at"]
    )
