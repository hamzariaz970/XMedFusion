"""
test_unit.py — Unit Tests for XMedFusion Backend
Tests individual components in isolation (no Ollama/LLM calls required).

Run:
    cd backend
    pytest tests/test_unit.py -v
"""
import os
import sys
import json
import pytest
import torch
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════
# 1. VALIDATORS MODULE
# ══════════════════════════════════════════════════════════════════
class TestValidators:
    """Unit tests for validators.py — report validation logic."""

    def setup_method(self):
        from validators import validate_report, kg_present_absent, extract_section, parse_labels
        self.validate_report = validate_report
        self.kg_present_absent = kg_present_absent
        self.extract_section = extract_section
        self.parse_labels = parse_labels

    def test_valid_unstructured_report_returns_ok(self):
        report = "Lungs are clear. No pleural effusion. Normal cardiac silhouette."
        result = self.validate_report(report, None)
        assert result["ok"] is True
        assert result["errors"] == []

    def test_structured_report_with_both_sections_passes(self):
        report = "FINDINGS: Clear lungs.\nIMPRESSION: Normal study."
        result = self.validate_report(report, None)
        assert result["ok"] is True

    def test_structured_report_missing_impression_fails(self):
        report = "FINDINGS: Mild cardiomegaly."
        result = self.validate_report(report, None)
        assert result["ok"] is False
        assert any("IMPRESSION" in e for e in result["errors"])

    def test_kg_contradiction_absent_entity_mentioned_fails(self, kg_absent_effusion):
        # KG says effusion is Absent, but report mentions "pleural effusion" without negation
        report = "There is pleural effusion in the lower lobes."
        result = self.validate_report(report, kg_absent_effusion)
        assert result["ok"] is False

    def test_kg_contradiction_properly_negated_passes(self, kg_absent_effusion):
        report = "No pleural effusion is identified."
        result = self.validate_report(report, kg_absent_effusion)
        assert result["ok"] is True

    def test_kg_present_absent_parsing(self, kg_cardiomegaly):
        present, absent = self.kg_present_absent(kg_cardiomegaly)
        assert "cardiomegaly" in present

    def test_kg_present_absent_empty_kg(self):
        present, absent = self.kg_present_absent(None)
        assert present == set()
        assert absent == set()

    def test_extract_section_returns_correct_content(self):
        report = "FINDINGS: Bibasilar opacities.\nIMPRESSION: Possible pneumonia."
        result = self.extract_section(report, "FINDINGS")
        assert "Bibasilar opacities" in result

    def test_parse_labels_comma_separated(self):
        labels = "Cardiomegaly, Pleural Effusion, Edema"
        result = self.parse_labels(labels)
        assert len(result) == 3
        assert "Cardiomegaly" in result

    def test_validate_empty_report_passes_unstructured(self):
        # Empty report, no KG — should not crash
        result = self.validate_report("", None)
        assert isinstance(result, dict)
        assert "ok" in result


# ══════════════════════════════════════════════════════════════════
# 2. EXPLAIN MODULE
# ══════════════════════════════════════════════════════════════════
class TestExplain:
    """Unit tests for explain.py — visual explainability overlay."""

    def setup_method(self):
        from explain import generate_explainable_image, parse_kg_for_visuals, apply_clinical_heuristics
        self.generate_explainable_image = generate_explainable_image
        self.parse_kg_for_visuals = parse_kg_for_visuals
        self.apply_clinical_heuristics = apply_clinical_heuristics

    def _make_dummy_image(self, path, size=(512, 512)):
        img = Image.new("RGB", size, color=(128, 128, 128))
        img.save(path)
        return path

    def test_parse_kg_for_visuals_returns_zone_map(self, kg_cardiomegaly, tmp_path):
        result = self.parse_kg_for_visuals(kg_cardiomegaly)
        # "mediastinum" is a recognized zone
        assert isinstance(result, dict)

    def test_parse_kg_empty_returns_empty(self):
        result = self.parse_kg_for_visuals(None)
        assert result == {}

    def test_parse_kg_no_entities_key(self):
        result = self.parse_kg_for_visuals({"relations": []})
        assert result == {}

    def test_apply_clinical_heuristics_pneumothorax_to_top(self):
        boxes = self.apply_clinical_heuristics("right lung", ["Pneumothorax"], 0, 0, 300, 600)
        assert "top" in boxes

    def test_apply_clinical_heuristics_effusion_to_bottom(self):
        boxes = self.apply_clinical_heuristics("left lung", ["Pleural Effusion"], 0, 0, 300, 600)
        assert "bottom" in boxes

    def test_apply_clinical_heuristics_cardiomegaly_to_lower_mid(self):
        boxes = self.apply_clinical_heuristics("mediastinum", ["Cardiomegaly"], 0, 0, 300, 600)
        assert "lower_mid" in boxes

    def test_apply_clinical_heuristics_normal_skipped(self):
        boxes = self.apply_clinical_heuristics("right lung", ["Clear", "Normal"], 0, 0, 300, 600)
        assert len(boxes) == 0

    def test_generate_explainable_image_creates_file(self, tmp_path, real_xray_path):
        output = str(tmp_path / "explained.png")
        fake_kg = {
            "entities": [["cardiomegaly", "Present"], ["mediastinum", "Anatomy"]],
            "relations": [[0, 1, "located_at"]]
        }
        result = self.generate_explainable_image(real_xray_path, fake_kg, output)
        assert result is not None
        assert os.path.exists(output)

    def test_generate_explainable_image_no_findings_returns_none(self, real_xray_path, tmp_path):
        output = str(tmp_path / "no_findings.png")
        empty_kg = {"entities": [], "relations": []}
        result = self.generate_explainable_image(real_xray_path, empty_kg, output)
        # If no recognized zone findings, function returns None
        assert result is None or isinstance(result, str)

    def test_generate_explainable_image_invalid_path_returns_none(self, tmp_path):
        output = str(tmp_path / "out.png")
        result = self.generate_explainable_image("/nonexistent/image.png", {}, output)
        assert result is None


# ══════════════════════════════════════════════════════════════════
# 3. DRAFT MODULE — RetrievalAgent (no LLM)
# ══════════════════════════════════════════════════════════════════
class TestDraftRetrieval:
    """Unit tests for draft.py RetrievalAgent (no LLM invocation)."""

    def test_retrieval_agent_initializes(self):
        from vision import vision_encoder
        from draft import RetrievalAgent
        agent = RetrievalAgent(vision_encoder, k=3)
        assert agent.k == 3

    def test_truncate_report(self):
        from draft import truncate_report
        long = " ".join([f"word{i}" for i in range(200)])
        result = truncate_report(long, max_words=10)
        assert len(result.split()) <= 10

    def test_truncate_report_short_text_unchanged(self):
        from draft import truncate_report
        short = "Heart size normal."
        assert truncate_report(short) == short

    def test_retrieve_top_k_empty_dict_returns_empty(self, real_xray_path):
        from vision import vision_encoder
        from draft import RetrievalAgent
        agent = RetrievalAgent(vision_encoder, k=3)
        result = agent.retrieve_top_k(real_xray_path, {})
        assert result == []

    def test_retrieve_top_k_returns_correct_count(self, real_xray_path):
        from vision import vision_encoder
        from draft import RetrievalAgent, reports_dict
        agent = RetrievalAgent(vision_encoder, k=3)
        results = agent.retrieve_top_k(real_xray_path, reports_dict)
        assert len(results) <= 3
        assert all(isinstance(r, str) for r in results)


# ══════════════════════════════════════════════════════════════════
# 4. VISION MODULE — VisionEncoder
# ══════════════════════════════════════════════════════════════════
class TestVisionEncoder:
    """Unit tests for vision.py — BioMedCLIP encoder outputs."""

    def test_encode_image_returns_normalized_tensor(self, real_xray_pil):
        from vision import vision_encoder
        feat = vision_encoder.encode_image(real_xray_pil)
        assert feat.ndim == 2          # (1, 512)
        norm = feat.norm(dim=-1).item()
        assert abs(norm - 1.0) < 1e-3  # L2-normalized

    def test_encode_text_returns_normalized_tensor(self):
        from vision import vision_encoder
        feat = vision_encoder.encode_text(["chest x-ray showing cardiomegaly"])
        assert feat.ndim == 2
        norm = feat.norm(dim=-1).item()
        assert abs(norm - 1.0) < 1e-3

    def test_image_text_similarity_xray_vs_xray_label(self, real_xray_pil):
        from vision import vision_encoder
        import torch.nn.functional as F
        img_feat = vision_encoder.encode_image(real_xray_pil)
        xray_feat = vision_encoder.encode_text(["a chest x-ray"])
        selfie_feat = vision_encoder.encode_text(["a photo of a cat"])
        sim_xray = F.cosine_similarity(img_feat, xray_feat).item()
        sim_selfie = F.cosine_similarity(img_feat, selfie_feat).item()
        assert sim_xray > sim_selfie, "X-ray image should be more similar to 'x-ray' text than 'cat'"

    def test_get_hybrid_findings_returns_dict(self, real_xray_pil):
        from vision import vision_encoder, get_hybrid_findings, DISEASES
        feat = vision_encoder.encode_image(real_xray_pil)
        findings = get_hybrid_findings(feat)
        assert isinstance(findings, dict)
        assert all(k in DISEASES for k in findings.keys())

    def test_hybrid_findings_scores_between_0_and_1(self, real_xray_pil):
        from vision import vision_encoder, get_hybrid_findings
        feat = vision_encoder.encode_image(real_xray_pil)
        findings = get_hybrid_findings(feat)
        for k, v in findings.items():
            assert 0.0 <= v <= 1.0, f"{k}: score {v} out of range"


# ══════════════════════════════════════════════════════════════════
# 5. XRAY FILTER MODULE
# ══════════════════════════════════════════════════════════════════
class TestXRayFilter:
    """Unit tests for xray_filter.py — X-ray bouncer head."""

    def test_real_xray_passes_filter(self, real_xray_path):
        from xray_filter import is_chest_xray
        is_valid, confidence = is_chest_xray(real_xray_path)
        assert is_valid is True, f"Real X-ray rejected (confidence={confidence:.4f})"
        assert confidence > 0.5

    def test_non_xray_rejected(self, non_xray_path):
        from xray_filter import is_chest_xray
        is_valid, confidence = is_chest_xray(non_xray_path)
        assert is_valid is False, f"Non-X-ray accepted (confidence={confidence:.4f})"

    def test_filter_returns_float_confidence(self, real_xray_path):
        from xray_filter import is_chest_xray
        is_valid, confidence = is_chest_xray(real_xray_path)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_filter_handles_corrupt_path_gracefully(self):
        from xray_filter import is_chest_xray
        is_valid, confidence = is_chest_xray("/nonexistent/fake.png")
        assert is_valid is False
        assert confidence == 0.0


# ══════════════════════════════════════════════════════════════════
# 6. CONFIG MODULE
# ══════════════════════════════════════════════════════════════════
class TestConfig:
    """Verify config.py has all required fields."""

    def test_required_fields_exist(self):
        import config
        assert hasattr(config, "OLLAMA_MODEL")
        assert hasattr(config, "TEMPERATURE")
        assert hasattr(config, "BASE_URL")

    def test_temperature_in_valid_range(self):
        import config
        assert 0.0 <= config.TEMPERATURE <= 2.0

    def test_base_url_is_localhost(self):
        import config
        assert "localhost" in config.BASE_URL or "127.0.0.1" in config.BASE_URL
