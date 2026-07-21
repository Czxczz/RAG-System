"""Auth, roles, and admin config API tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from unittest.mock import MagicMock

from app.config import Settings, get_settings
from app.core.auth import reset_auth_service
from app.core.runtime_config import load_overrides, public_config_view, save_overrides
from app.dependencies import get_orchestrator
from app.main import app


@pytest.fixture()
def auth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "uploads").mkdir()

    settings = Settings(
        data_dir=data_dir,
        auth_enabled=True,
        auth_secret="test-secret",
        admin_username="admin",
        admin_password="admin-pass",
        user_username="user",
        user_password="user-pass",
        rerank_enabled=False,
        query_rewrite_enabled=False,
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.routes.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.get_orchestrator", lambda: MagicMock())
    reset_auth_service()

    orch = MagicMock()
    orch.registry.count = 0
    orch.store.num_chunks = 0
    orch.registry.list.return_value = []
    app.dependency_overrides[get_orchestrator] = lambda: orch

    with TestClient(app) as c:
        yield c, settings

    app.dependency_overrides.clear()
    reset_auth_service()
    get_settings.cache_clear()


def test_auth_status_reports_enabled(auth_client):
    client, _ = auth_client
    res = client.get("/auth/status")
    assert res.status_code == 200
    assert res.json()["auth_enabled"] is True


def test_login_admin_and_user(auth_client):
    client, _ = auth_client
    bad = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert bad.status_code == 401

    admin = client.post(
        "/auth/login", json={"username": "admin", "password": "admin-pass"}
    )
    assert admin.status_code == 200
    assert admin.json()["role"] == "admin"
    token = admin.json()["token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["role"] == "admin"

    user = client.post(
        "/auth/login", json={"username": "user", "password": "user-pass"}
    )
    assert user.status_code == 200
    assert user.json()["role"] == "user"


def test_user_cannot_delete_or_edit_config(auth_client):
    client, _ = auth_client
    login = client.post(
        "/auth/login", json={"username": "user", "password": "user-pass"}
    )
    token = login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    deleted = client.delete("/documents/abc", headers=headers)
    assert deleted.status_code == 403

    cfg = client.get("/admin/config", headers=headers)
    assert cfg.status_code == 403


def test_admin_can_read_config(auth_client):
    client, _ = auth_client
    login = client.post(
        "/auth/login", json={"username": "admin", "password": "admin-pass"}
    )
    token = login.json()["token"]
    res = client.get(
        "/admin/config", headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    assert "top_k" in res.json()["config"]


def test_runtime_overrides_roundtrip(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_overrides(data_dir, {"top_k": 7, "openai_api_key": "sk-test-key"})
    loaded = load_overrides(data_dir)
    assert loaded["top_k"] == 7
    assert loaded["openai_api_key"] == "sk-test-key"

    settings = Settings(data_dir=data_dir, top_k=3, openai_api_key="")
    view = public_config_view(settings.model_copy(update=loaded))
    assert view["top_k"] == 7
    assert view["openai_api_key_set"] is True
    assert "sk-test" not in view["openai_api_key"] or "••••" in view["openai_api_key"]
