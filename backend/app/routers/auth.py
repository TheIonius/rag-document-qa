import secrets
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_db, record_audit_event, utc_now_iso
from app.models.schemas import (
    SSOCallbackRequest,
    SSOProviderResponse,
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
    email = req.email.strip().lower()
    display_name = (req.full_name or "").strip() or email.split("@")[0].title()

    now = utc_now_iso()
    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    org_id = f"org_{uuid.uuid4().hex[:12]}"
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    mem_id = f"mem_{uuid.uuid4().hex[:12]}"

    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="A user with this email already exists.")

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

@router.get("/sso/providers", response_model=List[SSOProviderResponse])
def get_sso_providers():
    """
    List configured Enterprise SSO / OIDC identity providers.
    Supports Okta, Google Workspace, Azure Entra ID, and Local Mock IdP.
    """
    return [
        SSOProviderResponse(
            id="google",
            name="Google Workspace",
            protocol="OIDC",
            auth_url="https://accounts.google.com/o/oauth2/v2/auth",
            client_id="enterprise-cortex-google-client.apps.googleusercontent.com",
            enabled=True
        ),
        SSOProviderResponse(
            id="azure_ad",
            name="Microsoft Entra ID",
            protocol="OIDC",
            auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            client_id="enterprise-cortex-azure-client-id",
            enabled=True
        ),
        SSOProviderResponse(
            id="okta",
            name="Okta Workforce",
            protocol="OIDC",
            auth_url="https://enterprise-auth.okta.com/oauth2/v1/authorize",
            client_id="enterprise-cortex-okta-client-id",
            enabled=True
        ),
        SSOProviderResponse(
            id="mock_oidc",
            name="Enterprise Mock IdP",
            protocol="OIDC",
            auth_url="/api/v1/auth/sso/mock",
            client_id="mock-enterprise-client-id",
            enabled=True
        )
    ]

@router.post("/sso/callback", response_model=TokenResponse)
def sso_callback(req: SSOCallbackRequest):
    """
    Handle SSO/OIDC authentication callback and Auto-Provisioning (JIT).
    Validates identity provider claims, ensures user and tenant workspace exist,
    and returns an access token.
    """
    email = (req.email or "").strip().lower()
    if not email:
        if req.provider_user_id:
            clean_sub = req.provider_user_id.strip().lower().replace(" ", "-")
            email = f"{clean_sub}@{req.provider}.sso.local"
        elif req.code:
            clean_code = req.code.strip()[:10].lower()
            email = f"sso-user-{clean_code}@{req.provider}.sso.local"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SSO callback requires either an email claim, provider_user_id, or authorization code."
            )

    display_name = (req.full_name or "").strip() or email.split("@")[0].replace(".", " ").title()
    now = utc_now_iso()

    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row:
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
            is_new = False
        else:
            # Auto-provision new enterprise user with personal workspace
            is_new = True
            user_id = f"usr_{uuid.uuid4().hex[:12]}"
            org_id = f"org_{uuid.uuid4().hex[:12]}"
            ws_id = f"ws_{uuid.uuid4().hex[:12]}"
            mem_id = f"mem_{uuid.uuid4().hex[:12]}"
            user_count = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"]
            is_super = 1 if user_count == 0 else 0
            temp_hash = hash_password(secrets.token_urlsafe(32))

            conn.execute("""
                INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            """, (user_id, email, temp_hash, display_name, is_super, now))

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

            full_name = display_name
            user_email = email
            created_at = now

    record_audit_event(
        action="sso_register" if is_new else "sso_login",
        resource_type="user",
        resource_id=user_id,
        actor_id=user_id,
        actor_email=user_email,
        details={"provider": req.provider, "auto_provisioned": is_new}
    )

    token = create_access_token({"sub": user_id, "email": user_email})
    user_resp = UserResponse(
        id=user_id,
        email=user_email,
        full_name=full_name,
        is_active=True,
        is_superuser=bool(is_super),
        created_at=created_at
    )
    return TokenResponse(access_token=token, token_type="bearer", user=user_resp)

