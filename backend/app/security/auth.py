from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.config import settings

password_hasher = PasswordHash((Argon2Hasher(),))
security = HTTPBearer(auto_error=False)

ROLE_RANKS = {
    "owner": 4,
    "admin": 3,
    "editor": 2,
    "viewer": 1,
}

def hash_password(password: str) -> str:
    return password_hasher.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return password_hasher.verify(plain_password, hashed_password)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user_from_db(user_id: str):
    from app.database import get_db
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account associated with this token was not found.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = dict(row)
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated.",
            )
        return user

def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    if not credentials or not credentials.credentials:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            return None
        return get_current_user_from_db(user_id)
    except HTTPException:
        return None

def get_user_from_token(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        return get_current_user_from_db(user_id)
    except Exception:
        return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token: missing subject.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return get_current_user_from_db(user_id)

def require_workspace_membership(
    workspace_id: Optional[str] = None,
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: dict = Depends(get_current_user),
    min_role: str = "viewer",
) -> dict:
    target_ws = workspace_id or x_workspace_id
    if user.get("is_superuser"):
        return {"user": user, "role": "owner", "workspace_id": target_ws or "ws_default"}

    if not target_ws:
        from app.database import get_db
        with get_db() as conn:
            m = conn.execute(
                "SELECT workspace_id, role FROM workspace_memberships WHERE user_id = ? ORDER BY created_at ASC LIMIT 1",
                (user["id"],),
            ).fetchone()
            if not m:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User does not belong to any active workspace.",
                )
            target_ws = m["workspace_id"]
            role = m["role"]
    else:
        from app.database import get_db
        with get_db() as conn:
            m = conn.execute(
                "SELECT role FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                (user["id"], target_ws),
            ).fetchone()
            if not m:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied to workspace '{target_ws}'.",
                )
            role = m["role"]

    required_rank = ROLE_RANKS.get(min_role.lower(), 1)
    user_rank = ROLE_RANKS.get(role.lower(), 0)

    if user_rank < required_rank:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions: requires '{min_role}' role (current role: '{role}').",
        )

    return {"user": user, "role": role, "workspace_id": target_ws}

def require_role(min_role: str):
    def dependency(
        workspace_id: Optional[str] = None,
        x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
        user: dict = Depends(get_current_user),
    ):
        return require_workspace_membership(
            workspace_id=workspace_id,
            x_workspace_id=x_workspace_id,
            user=user,
            min_role=min_role
        )
    return dependency
