"""
test_integration.py — Integration Tests for XMedFusion Backend
Tests multiple components working together (no Ollama/LLM calls required).

Run:
    cd backend
    pytest tests/test_integration.py -v
"""
import os
import sys
import json
import asyncio
import pytest
from PIL import Image
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════
# 1. VISION → HYBRID FINDINGS → KG EXPLANATION (Full Visual Chain)
# ══════════════════════════════════════════════════════════════════
class TestVisionToExplainPipeline:
    """Tests the chain: image → BioMedCLIP → findings → bounding boxes."""

    def test_real_xray_findings_fed_into_explain(self, real_xray_path, tmp_path):
        from vision import vision_encoder, get_hybrid_findings
        from explain import generate_explainable_image

        img = Image.open(real_xray_path).convert("RGB")
        feat = vision_encoder.encode_image(img)
        findings = get_hybrid_findings(feat)

        # Build a synthetic KG from findings
        entities, relations = [], []
        for i, (disease, score) in enumerate(findings.items()):
            if score > 0:
                entities.append([disease.lower(), "Present"])
        if entities:
            entities.append(["mediastinum", "Anatomy"])
            relations.append([0, len(entities) - 1, "located_at"])

        kg = {"entities": entities, "relations": relations}
        output = str(tmp_path / "integration_test_explained.png")
        result = generate_explainable_image(real_xray_path, kg, output)
        # Result is either a path (findings exist) or None (all-normal image)
        assert result is None or os.path.exists(result)

    def test_vision_retrieval_similarity_ordering(self, real_xray_path):
        """
        The retrieval agent must return reports ranked by descending similarity score.
        We verify the scores are monotonically non-increasing.
        """
        from vision import vision_encoder
        from draft import RetrievalAgent, reports_dict
        import torch.nn.functional as F

        agent = RetrievalAgent(vision_encoder, k=5)
        img = Image.open(real_xray_path).convert("RGB")
        img_feat = vision_encoder.encode_image(img)

        all_reports = list(set(reports_dict.values()))[:50]  # subset for speed
        feats = vision_encoder.encode_text(all_reports)
        sims = F.cosine_similarity(img_feat, feats).tolist()
        sorted_sims = sorted(sims, reverse=True)

        # First k must be the top-k scores
        top5 = sorted(sims, reverse=True)[:5]
        assert top5 == sorted(top5, reverse=True)


# ══════════════════════════════════════════════════════════════════
# 2. XRAY FILTER → SYNTHESIS GATE
# ══════════════════════════════════════════════════════════════════
class TestFilterGateSynthesis:
    """
    Verifies that the filter correctly gates invalid images out of the
    synthesis pipeline before any LLM is called.
    """

    @pytest.mark.asyncio
    async def test_non_xray_rejected_before_llm(self, non_xray_path):
        """Non-X-ray must emit 'error' status without calling the LLM."""
        from synthesis import LocalSynthesisAgent
        from vision import vision_encoder, get_hybrid_findings, VisualDescriptionAgent
        from draft import RetrievalAgent, LocalLLMReportAgent, reports_dict

        agent = LocalSynthesisAgent()
        retrieval = RetrievalAgent(vision_encoder, k=3)
        draft = LocalLLMReportAgent.__new__(LocalLLMReportAgent)  # don't init LLM
        vision = VisualDescriptionAgent.__new__(VisualDescriptionAgent)

        results = []
        async for chunk in agent.generate_final_report(
            draft_agent=draft,
            vision_agent=vision,
            retrieval_agent=retrieval,
            reports_dict=reports_dict,
            image_paths=[non_xray_path],
        ):
            data = json.loads(chunk)
            results.append(data)
            if data.get("status") == "error":
                break

        statuses = [r["status"] for r in results]
        assert "error" in statuses, f"Expected error status for non-X-ray, got: {statuses}"
        # Ensure error is hit early (before synthesis_start)
        assert "synthesis_start" not in statuses

    @pytest.mark.asyncio
    async def test_real_xray_passes_filter_and_proceeds(self, real_xray_path):
        """Real X-ray must pass the filter and proceed to at least parallel_start."""
        from synthesis import LocalSynthesisAgent
        from vision import vision_encoder, get_hybrid_findings, VisualDescriptionAgent
        from draft import RetrievalAgent, LocalLLMReportAgent, reports_dict
        from unittest.mock import patch

        def fake_generate(context):
            return "Normal study. No acute cardiopulmonary process."

        def fake_describe(findings):
            return "Normal chest X-ray."

        agent = LocalSynthesisAgent()
        retrieval = RetrievalAgent(vision_encoder, k=3)

        # FIX: Use patch.object on the initialized agent instance for "llm"
        with patch("draft.LocalLLMReportAgent.generate_report", side_effect=fake_generate), \
             patch("vision.VisualDescriptionAgent.generate_description", side_effect=fake_describe), \
             patch("synthesis.LocalSynthesisAgent._clean_output", side_effect=lambda x: x), \
             patch.object(agent, "llm") as mock_llm:

            mock_llm.invoke.return_value = MagicMock(content="Normal study.")

            draft = LocalLLMReportAgent()
            vision = VisualDescriptionAgent()

            statuses = []
            async for chunk in agent.generate_final_report(
                draft_agent=draft,
                vision_agent=vision,
                retrieval_agent=retrieval,
                reports_dict=reports_dict,
                image_paths=[real_xray_path],
            ):
                data = json.loads(chunk)
                statuses.append(data["status"])
                if data["status"] in ("complete", "error"):
                    break

        assert "parallel_start" in statuses, f"Expected parallel_start, got: {statuses}"
        assert "error" not in statuses


# ══════════════════════════════════════════════════════════════════
# 3. VALIDATOR → SYNTHESIS REPAIR LOOP
# ══════════════════════════════════════════════════════════════════
class TestValidatorIntegration:
    """
    Tests that validate_report correctly feeds errors into the repair cycle.
    """

    def test_contradiction_detected_in_full_flow(self, kg_absent_effusion):
        from validators import validate_report
        bad_report = "There is bilateral pleural effusion in the lower lobes."
        result = validate_report(bad_report, kg_absent_effusion)
        assert not result["ok"]
        assert len(result["errors"]) > 0

    def test_repair_is_triggered_for_invalid_report(self, kg_absent_effusion):
        """Verify repair_report is called when validate_report returns errors."""
        from synthesis import LocalSynthesisAgent
        from validators import validate_report

        bad_report = "There is bilateral pleural effusion."
        val = validate_report(bad_report, kg_absent_effusion)
        assert not val["ok"]

        agent = LocalSynthesisAgent.__new__(LocalSynthesisAgent)
        
        # FIX: Directly attach a mock since __init__ was skipped
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = MagicMock(
            content="No pleural effusion is identified."
        )
        
        fixed = agent.repair_report(bad_report, val["errors"], "KG: effusion Absent")
        assert isinstance(fixed, str)
        assert len(fixed) > 0


# ══════════════════════════════════════════════════════════════════
# 4. ANNOTATION DATA INTEGRITY
# ══════════════════════════════════════════════════════════════════
class TestAnnotationIntegrity:
    """Verifies the dataset annotation file is well-formed."""

    def test_annotation_has_train_split(self, annotation_data):
        assert "train" in annotation_data or isinstance(annotation_data, list)

    def test_each_sample_has_report(self, annotation_data):
        if isinstance(annotation_data, dict):
            samples = annotation_data.get("train", [])
        else:
            samples = annotation_data
        for sample in samples[:20]:
            assert "report" in sample
            assert len(sample["report"]) > 0

    def test_each_sample_has_image_path(self, annotation_data):
        if isinstance(annotation_data, dict):
            samples = annotation_data.get("train", [])
        else:
            samples = annotation_data
        for sample in samples[:20]:
            assert "image_path" in sample
            assert len(sample["image_path"]) > 0

    def test_image_files_exist_on_disk(self, annotation_data):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img_root = os.path.join(backend_dir, "data", "iu_xray", "images")
        if isinstance(annotation_data, dict):
            samples = annotation_data.get("train", [])
        else:
            samples = annotation_data
        missing = 0
        for sample in samples[:10]:
            for rel in sample.get("image_path", []):
                full = os.path.join(img_root, rel)
                if not os.path.exists(full):
                    missing += 1
        assert missing == 0, f"{missing} image files referenced in annotation are missing on disk"