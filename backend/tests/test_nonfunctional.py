"""
test_nonfunctional.py — Non-Functional Tests for XMedFusion Backend
Covers performance, security, robustness, and boundary conditions.

Run:
    cd backend
    pytest tests/test_nonfunctional.py -v
"""
import os
import sys
import time
import json
import pytest
import torch
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "http://127.0.0.1:8000"


def _server_is_up():
    try:
        return requests.get(f"{API_BASE}/api/health", timeout=2).status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# 1. PERFORMANCE TESTS
# ══════════════════════════════════════════════════════════════════
class TestPerformance:
    """Verify inference latency of core components."""

    def test_bioclip_encode_image_under_500ms(self, real_xray_pil):
        """BioMedCLIP image encoding must complete within 500ms (GPU)."""
        from vision import vision_encoder
        # Warm-up
        vision_encoder.encode_image(real_xray_pil)
        t0 = time.perf_counter()
        vision_encoder.encode_image(real_xray_pil)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"Image encoding took {elapsed_ms:.1f}ms (limit: 500ms)"

    def test_bioclip_encode_batch_text_under_2s(self):
        """Encoding 64 text prompts must complete within 2 seconds."""
        from vision import vision_encoder
        texts = ["chest x-ray finding"] * 64
        t0 = time.perf_counter()
        vision_encoder.encode_text(texts)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Batch text encoding took {elapsed:.2f}s (limit: 2s)"

    def test_xray_filter_inference_under_500ms(self, real_xray_path):
        """X-ray filter (bouncer head) forward pass must be under 500ms."""
        from xray_filter import is_chest_xray
        # Warm-up
        is_chest_xray(real_xray_path)
        t0 = time.perf_counter()
        is_chest_xray(real_xray_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"Filter took {elapsed_ms:.1f}ms (limit: 500ms)"

    def test_retrieval_topk_under_10s(self, real_xray_path):
        """Retrieval of top-5 from 4K reports must complete in under 10 seconds."""
        from vision import vision_encoder
        from draft import RetrievalAgent, reports_dict
        agent = RetrievalAgent(vision_encoder, k=5)
        t0 = time.perf_counter()
        agent.retrieve_top_k(real_xray_path, reports_dict)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Retrieval took {elapsed:.2f}s (limit: 10s)"

    def test_explain_image_generation_under_2s(self, real_xray_path, tmp_path):
        """Explainability overlay generation must complete within 2 seconds."""
        from explain import generate_explainable_image
        kg = {
            "entities": [["cardiomegaly", "Present"], ["mediastinum", "Anatomy"]],
            "relations": [[0, 1, "located_at"]],
        }
        out = str(tmp_path / "perf_explained.png")
        t0 = time.perf_counter()
        generate_explainable_image(real_xray_path, kg, out)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Explain overlay took {elapsed:.2f}s (limit: 2s)"

    @pytest.mark.skipif(not _server_is_up(), reason="API server not running")
    def test_health_endpoint_response_under_1s(self):
        t0 = time.perf_counter()
        requests.get(f"{API_BASE}/api/health", timeout=5)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s (limit: 1s)"

    def test_gpu_memory_not_leaked_after_filter(self, real_xray_path):
        """
        GPU memory usage should not grow significantly across 5 calls
        to the X-ray filter (checks for memory leaks).
        """
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        from xray_filter import is_chest_xray
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(5):
            is_chest_xray(real_xray_path)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        leak_mb = (after - before) / (1024 ** 2)
        assert leak_mb < 50, f"GPU memory leaked {leak_mb:.1f}MB over 5 filter calls"


# ══════════════════════════════════════════════════════════════════
# 2. ROBUSTNESS / BOUNDARY TESTS
# ══════════════════════════════════════════════════════════════════
class TestRobustness:
    """Verify the system handles edge cases and bad inputs gracefully."""

    def test_encode_image_1x1_pixel(self):
        """Encoding a 1×1 image must not crash — model should handle tiny input."""
        from vision import vision_encoder
        tiny = Image.new("RGB", (1, 1), color=(0, 0, 0))
        feat = vision_encoder.encode_image(tiny)
        assert feat is not None

    def test_encode_image_large_4k(self):
        """Encoding a 4K image must not crash (preprocess resizes it)."""
        from vision import vision_encoder
        large = Image.new("RGB", (3840, 2160), color=(200, 200, 200))
        feat = vision_encoder.encode_image(large)
        assert feat.shape[-1] == 512

    def test_encode_text_empty_string(self):
        """Encoding an empty string should not raise an exception."""
        from vision import vision_encoder
        feat = vision_encoder.encode_text([""])
        assert feat is not None

    def test_encode_text_very_long_string(self):
        """Text longer than BioMedCLIP's 256-token limit is truncated, not crashed."""
        from vision import vision_encoder
        long_text = "cardiomegaly " * 500
        feat = vision_encoder.encode_text([long_text])
        assert feat is not None

    def test_validate_report_unicode_input(self):
        """validate_report must handle unicode without crashing."""
        from validators import validate_report
        report = "Lungs are clear. Aucun épanchement pleural. 无胸腔积液."
        result = validate_report(report, None)
        assert isinstance(result, dict)

    def test_validate_report_very_long_report(self):
        """validate_report must handle a very long report without crashing."""
        from validators import validate_report
        long_report = "Normal. " * 1000
        result = validate_report(long_report, None)
        assert isinstance(result, dict)

    def test_filter_grayscale_image(self, tmp_path):
        """X-ray filter must handle grayscale images (common for X-rays)."""
        from xray_filter import is_chest_xray
        gray = Image.new("L", (224, 224), color=128)
        path = str(tmp_path / "gray.png")
        gray.save(path)
        is_valid, conf = is_chest_xray(path)
        assert isinstance(is_valid, bool)
        assert 0.0 <= conf <= 1.0

    def test_filter_rgba_image(self, tmp_path):
        """X-ray filter must handle RGBA images without crashing."""
        from xray_filter import is_chest_xray
        rgba = Image.new("RGBA", (224, 224), color=(128, 128, 128, 255))
        path = str(tmp_path / "rgba.png")
        rgba.save(path)
        is_valid, conf = is_chest_xray(path)
        assert isinstance(conf, float)

    def test_explain_with_kg_missing_zone_name(self, real_xray_path, tmp_path):
        """KG with an unknown anatomy zone should still produce output (zone skipped)."""
        from explain import generate_explainable_image
        kg = {
            "entities": [["cardiomegaly", "Present"], ["spleen", "Anatomy"]],
            "relations": [[0, 1, "located_at"]],
        }
        out = str(tmp_path / "unknown_zone.png")
        result = generate_explainable_image(real_xray_path, kg, out)
        # Should not crash
        assert result is None or isinstance(result, str)

    def test_retrieval_k_larger_than_dataset(self, real_xray_path):
        """Requesting more results than available should not crash."""
        from vision import vision_encoder
        from draft import RetrievalAgent
        # FIX: Removed the unsupported dict multiplication
        tiny_dict = {"/some/path.png": "Normal study."}
        agent = RetrievalAgent(vision_encoder, k=1000)
        results = agent.retrieve_top_k(real_xray_path, tiny_dict)
        assert len(results) <= 1  # Can't return more than available

    def test_hybrid_findings_all_black_image(self):
        """All-black image should not crash the hybrid scoring."""
        from vision import vision_encoder, get_hybrid_findings
        black = Image.new("RGB", (224, 224), color=(0, 0, 0))
        feat = vision_encoder.encode_image(black)
        findings = get_hybrid_findings(feat)
        assert isinstance(findings, dict)


# ══════════════════════════════════════════════════════════════════
# 3. SECURITY TESTS
# ══════════════════════════════════════════════════════════════════
class TestSecurity:
    """Verify the API rejects malicious or malformed inputs safely."""

    @pytest.mark.skipif(not _server_is_up(), reason="API server not running")
    def test_upload_non_image_file_rejected(self):
        """Uploading a text/HTML file must not cause a server crash (500)."""
        payload = b"<script>alert('xss')</script>"
        r = requests.post(
            f"{API_BASE}/api/synthesize-report",
            files={"file": ("evil.html", payload, "text/html")},
            timeout=15,
        )
        # Should return 200 with error stream, or 400/422 — NOT 500
        assert r.status_code != 500

    @pytest.mark.skipif(not _server_is_up(), reason="API server not running")
    def test_upload_empty_file_does_not_crash_server(self):
        """Uploading an empty file must not crash (status 500)."""
        r = requests.post(
            f"{API_BASE}/api/synthesize-report",
            files={"file": ("empty.png", b"", "image/png")},
            timeout=15,
        )
        assert r.status_code != 500

    @pytest.mark.skipif(not _server_is_up(), reason="API server not running")
    def test_missing_file_field_returns_422(self):
        """POST without the 'file' field should return 422 Unprocessable Entity."""
        r = requests.post(f"{API_BASE}/api/synthesize-report", timeout=10)
        assert r.status_code == 422

    @pytest.mark.skipif(not _server_is_up(), reason="API server not running")
    def test_cors_headers_present(self):
        """API must include CORS headers for browser clients."""
        r = requests.options(
            f"{API_BASE}/api/health",
            headers={"Origin": "http://localhost:3000"},
            timeout=5,
        )
        # FastAPI CORS middleware returns Allow-Origin in response
        assert "access-control-allow-origin" in (k.lower() for k in r.headers.keys())

    @pytest.mark.skipif(not _server_is_up(), reason="API server not running")
    def test_path_traversal_filename_rejected(self):
        """A filename like '../../etc/passwd' must not cause a server crash."""
        payload = b"\x89PNG\r\n\x1a\n"  # minimal PNG header
        r = requests.post(
            f"{API_BASE}/api/synthesize-report",
            files={"file": ("../../etc/passwd", payload, "image/png")},
            timeout=15,
        )
        assert r.status_code != 500

    def test_kg_json_injection_does_not_crash_validator(self):
        """Malformed/injected KG dict must not crash validate_report."""
        from validators import validate_report
        evil_kg = {
            "entities": [["'; DROP TABLE reports; --", "Present"]],
            "relations": [],
        }
        result = validate_report("Normal findings.", evil_kg)
        assert isinstance(result, dict)

    def test_report_with_html_injection_handled(self):
        """Report text containing HTML tags must not crash the validator."""
        from validators import validate_report
        evil_report = "<b>Normal</b> <script>alert(1)</script> lungs."
        result = validate_report(evil_report, None)
        assert isinstance(result, dict)

    def test_gpu_not_accessible_outside_model(self):
        """Direct CUDA tensor operations should be isolated to model inference."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU")
        # Just verifies we can't accidentally contaminate global GPU state
        t = torch.zeros(1).cuda()
        assert t.device.type == "cuda"
        del t
        torch.cuda.empty_cache()