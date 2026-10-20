"""SSRF protection and API-level controls."""
import importlib

import pytest
from fastapi.testclient import TestClient

from deepdive import config, sources


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://localhost:8000/", "http://169.254.169.254/latest/meta-data",
                                 "http://10.0.0.5/", "http://192.168.1.1/", "file:///etc/passwd", "ftp://example.com/x",
                                 "http://[::1]/"])
def test_ssrf_blocked(url):
    """Private, loopback, metadata, non-http schemes must never be fetched."""
    assert sources.is_safe_url(url) is False
    assert sources.fetch(url) is None


def test_public_ip_allowed():
    """A public address passes the check (literal IP, so no DNS needed)."""
    assert sources.is_safe_url("http://93.184.216.34/") is True


@pytest.fixture
def client(monkeypatch, tmp_path):
    """FastAPI test client backed by a temporary database."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "api.db"))
    app_module = importlib.import_module("app")
    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_health_and_metrics(client):
    """Liveness and metrics endpoints respond without authentication."""
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/metrics").status_code == 200


def test_api_key_enforced(client, monkeypatch):
    """When an API key is configured, requests without it are rejected."""
    monkeypatch.setattr(config, "API_KEY", "secret")
    assert client.post("/api/runs", data={"question": "What are EV battery risks?"}).status_code == 401
    assert client.get("/api/runs/1").status_code == 401


def test_input_validation(client):
    """Too-short questions and unsupported uploads are rejected before any model is called."""
    assert client.post("/api/runs", data={"question": "hi"}).status_code == 422
    r = client.post("/api/runs", data={"question": "What are EV battery risks?"},
                    files={"files": ("x.exe", b"abc")})
    assert r.status_code == 422


def test_unknown_run_404(client):
    """Unknown run ids return 404."""
    assert client.get("/api/runs/99999").status_code == 404
