"""
test_system.py — System / End-to-End Tests for XMedFusion API
Tests the live FastAPI server via HTTP (requires: uvicorn app:app running, or uses TestClient).

Run (with live server started separately on port 8000):
    cd backend
    pytest tests/test_system.py -v

Or using FastAPI TestClient (no separate server needed):
    pytest tests/test_system.py -v -k "not live"
"""
import os
import sys
import json
import time
import pytest
import requests
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "http://127.0.0.1:8000"


def _server_is_up():
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _image_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════
# 1. HEALTH ENDPOINT
# ══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _server_is_up(), reason="API server not running on port 8000")
class TestHealthEndpoint:
    """System tests for GET /api/health."""

    def test_health_returns_200(self):
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        assert r.status_code == 200

    def test_health_body_has_required_fields(self):
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        data = r.json()
        for key in ("status", "uptime_seconds", "cpu_percent", "memory_used_mb", "gpu_available"):
            assert key in data, f"Missing field: {key}"

    def test_health_status_is_healthy(self):
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        assert r.json()["status"] == "healthy"

    def test_health_gpu_field_is_bool(self):
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        assert isinstance(r.json()["gpu_available"], bool)

    def test_health_uptime_is_positive(self):
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        assert r.json()["uptime_seconds"] >= 0

    def test_health_response_under_2_seconds(self):
        t0 = time.time()
        requests.get(f"{API_BASE}/api/health", timeout=5)
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"Health check took {elapsed:.2f}s (expected < 2s)"


# ══════════════════════════════════════════════════════════════════
# 2. SYNTHESIZE ENDPOINT — Valid X-ray
# ══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _server_is_up(), reason="API server not running on port 8000")
class TestSynthesizeValidXray:
    """System tests for POST /api/synthesize-report with a real X-ray."""

    def test_endpoint_accepts_png_upload(self, real_xray_path):
        with open(real_xray_path, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/synthesize-report",
                files={"file": ("xray.png", f, "image/png")},
                stream=True,
                timeout=180,
            )
        assert r.status_code == 200

    def test_response_is_ndjson_stream(self, real_xray_path):
        with open(real_xray_path, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/synthesize-report",
                files={"file": ("xray.png", f, "image/png")},
                stream=True,
                timeout=180,
            )
        chunks = []
        for line in r.iter_lines():
            if line:
                chunks.append(json.loads(line))
                if chunks[-1].get("status") in ("complete", "error"):
                    break
        assert len(chunks) > 0

    def test_complete_chunk_has_final_report(self, real_xray_path):
        with open(real_xray_path, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/synthesize-report",
                files={"file": ("xray.png", f, "image/png")},
                stream=True,
                timeout=180,
            )
        complete_chunk = None
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("status") == "complete":
                    complete_chunk = data
                    break
        assert complete_chunk is not None
        assert "final_report" in complete_chunk
        assert len(complete_chunk["final_report"]) > 10

    def test_complete_chunk_has_knowledge_graph(self, real_xray_path):
        with open(real_xray_path, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/synthesize-report",
                files={"file": ("xray.png", f, "image/png")},
                stream=True,
                timeout=180,
            )
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("status") == "complete":
                    assert "knowledge_graph" in data
                    return
        pytest.fail("No complete chunk received")


# ══════════════════════════════════════════════════════════════════
# 3. SYNTHESIZE ENDPOINT — Invalid (Non-X-ray) Image
# ══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _server_is_up(), reason="API server not running on port 8000")
class TestSynthesizeInvalidImage:
    """System tests for POST /api/synthesize-report with a non-medical image."""

    def test_non_xray_returns_200_but_error_status(self, non_xray_path):
        with open(non_xray_path, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/synthesize-report",
                files={"file": ("photo.jpg", f, "image/jpeg")},
                stream=True,
                timeout=30,
            )
        assert r.status_code == 200
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("status") == "error":
                    assert "X-Ray" in data["message"] or "not a medical" in data["message"].lower()
                    return
        pytest.fail("Expected error status chunk for non-X-ray image")

    def test_no_report_generated_for_invalid_image(self, non_xray_path):
        with open(non_xray_path, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/synthesize-report",
                files={"file": ("photo.jpg", f, "image/jpeg")},
                stream=True,
                timeout=30,
            )
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("status") == "complete":
                    pytest.fail("Should NOT have received 'complete' for a non-X-ray image")


# ══════════════════════════════════════════════════════════════════
# 4. FASTAPI TESTCLIENT (no running server required)
# ══════════════════════════════════════════════════════════════════
class TestAPIWithTestClient:
    """
    Uses FastAPI TestClient so no external server is needed.
    LLM calls are mocked to keep these fast.
    """

    @pytest.fixture(scope="class")
    def client(self):
        from unittest.mock import patch, MagicMock, AsyncMock
        import httpx

        # Patch the heavy models before importing app
        with patch("synthesis.LocalSynthesisAgent.generate_final_report") as mock_gen, \
             patch("vision.VisionEncoder.__init__", return_value=None):

            async def fake_gen(*args, **kwargs):
                yield json.dumps({"status": "validating", "message": "Checking..."}) + "\n"
                yield json.dumps({"status": "complete", "final_report": "Normal study.",
                                  "knowledge_graph": None, "explainability": {},
                                  "explainable_image_path": None}) + "\n"

            mock_gen.side_effect = fake_gen

            try:
                from fastapi.testclient import TestClient
                from app import app
                return TestClient(app)
            except Exception:
                pytest.skip("Could not load app for TestClient (heavy model imports)")

    def test_health_endpoint_with_testclient(self, client):
        if client is None:
            pytest.skip("TestClient unavailable")
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_unknown_route_returns_404(self, client):
        if client is None:
            pytest.skip("TestClient unavailable")
        r = client.get("/api/nonexistent")
        assert r.status_code == 404
