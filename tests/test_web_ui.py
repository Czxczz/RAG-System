"""Smoke tests for the Web UI static routes."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ui_root_returns_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "PrivateRAG" in res.text


def test_static_js_available():
    res = client.get("/static/js/app.js")
    assert res.status_code == 200
    assert "chatStream" in res.text
    assert "upload/stream" in res.text
    assert "admin/config" in res.text


def test_ui_includes_progress_and_login():
    res = client.get("/")
    assert res.status_code == 200
    assert "upload-progress" in res.text
    assert "login-screen" in res.text
    assert "admin-modal" in res.text
    assert "open-admin-btn" in res.text
