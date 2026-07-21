"""Simple username/password auth with admin vs user roles.

When ``AUTH_ENABLED=false`` (default), all requests are treated as admin so
local demos keep working with zero login. Enable auth for deployed buyers.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

Role = Literal["admin", "user"]

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthUser:
    username: str
    role: Role


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    )
    return digest.hex()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    return hmac.compare_digest(_hash_password(password, salt), password_hash)


def make_password_hash(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    return salt, _hash_password(password, salt)


def create_token(user: AuthUser, secret: str, ttl_seconds: int = 60 * 60 * 24) -> str:
    payload = {
        "u": user.username,
        "r": user.role,
        "exp": int(time.time()) + ttl_seconds,
    }
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{body.decode('ascii')}.{sig}"


def decode_token(token: str, secret: str) -> AuthUser | None:
    try:
        body_b64, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    body = body_b64.encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(body))
    except (json.JSONDecodeError, ValueError):
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    role = payload.get("r")
    username = payload.get("u")
    if role not in ("admin", "user") or not username:
        return None
    return AuthUser(username=str(username), role=role)  # type: ignore[arg-type]


def built_in_users(settings) -> dict[str, dict[str, str]]:
    """Seed users from env (passwords hashed at check time via stored plain compare).

    For simplicity we store salted hashes derived once per process from env
    passwords — buyers set ADMIN_PASSWORD / USER_PASSWORD in ``.env``.
    """
    users: dict[str, dict[str, str]] = {}
    admin_user = settings.admin_username.strip() or "admin"
    user_user = settings.user_username.strip() or "user"
    admin_salt, admin_hash = make_password_hash(settings.admin_password or "admin")
    user_salt, user_hash = make_password_hash(settings.user_password or "user")
    users[admin_user] = {"role": "admin", "salt": admin_salt, "hash": admin_hash}
    # Avoid clobbering if usernames collide.
    if user_user != admin_user:
        users[user_user] = {"role": "user", "salt": user_salt, "hash": user_hash}
    return users


class AuthService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._users = built_in_users(settings)
        self._secret = settings.auth_secret or secrets.token_hex(32)

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        record = self._users.get(username)
        if not record:
            return None
        if not verify_password(password, record["salt"], record["hash"]):
            return None
        return AuthUser(username=username, role=record["role"])  # type: ignore[arg-type]

    def issue_token(self, user: AuthUser) -> str:
        return create_token(user, self._secret, ttl_seconds=self.settings.auth_token_ttl_seconds)

    def user_from_token(self, token: str) -> AuthUser | None:
        return decode_token(token, self._secret)


_auth_service: AuthService | None = None


def get_auth_service():
    from app.config import get_settings

    global _auth_service
    settings = get_settings()
    if _auth_service is None or _auth_service.settings is not settings:
        _auth_service = AuthService(settings)
    return _auth_service


def reset_auth_service() -> None:
    global _auth_service
    _auth_service = None


def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthUser:
    """Resolve the caller. When auth is disabled, return a synthetic admin."""
    from app.config import get_settings

    settings = get_settings()
    if not settings.auth_enabled:
        return AuthUser(username="local", role="admin")

    token = None
    if creds and creds.credentials:
        token = creds.credentials
    elif request.cookies.get("privaterag_token"):
        token = request.cookies.get("privaterag_token")
    elif request.headers.get("X-API-Token"):
        token = request.headers.get("X-API-Token")

    if not token:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")

    user = get_auth_service().user_from_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please log in again.")
    return user


def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required for this action.")
    return user


def require_user(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """Any authenticated role (admin or user)."""
    return user
