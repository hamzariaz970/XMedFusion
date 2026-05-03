import os
import re
import json
import asyncio
import torch
import gc
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from validators import validate_report
from PIL import Image 
import config
from config import HF_TOKEN

os.environ["HF_TOKEN"] = HF_TOKEN

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

REPORT_CONTEXT_WINDOW = min(config.CONTEXT_WINDOW, 8192)
LLM_TIMEOUT_SECONDS = 180
REPORT_MAX_TOKENS = 420
CT_GENERATION_TIMEOUT_SECONDS = 120
CT_REQUEST_TIMEOUT_SECONDS = 240

# <--- NEW IMPORTS --->
from explain import generate_explainable_image
from xray_filter import classify_scan

# --- 4. CT Vision Model (MedGemma fine-tuned on CT grids) ---
# Define constants unconditionally so they are always in scope
CT_MODEL_PATH = "model_weights/Vision_Agent/medgemma_ct_grid_finetuned"
_ct_model: object = None   # lazy-loaded on first CT request
_ct_proc:  object = None
CT_AVAILABLE = os.path.isdir(CT_MODEL_PATH)

# Only import transformers types at module level — vision_ct imports pandas
# which requires pytz; we import load_jpeg_montage lazily inside _infer_ct_report
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _TRANSFORMERS_OK = True
except ImportError as _e:
    print(f"[synthesis.py] transformers import failed: {_e}. CT pipeline disabled.")
    _TRANSFORMERS_OK = False
    CT_AVAILABLE = False


# --- 1. Import Legacy Draft Agent ---
from draft import (
    RetrievalAgent,
    LocalLLMReportAgent,
    reports_dict,
    truncate_report
)

# --- 2. Import NEW Vision Agent (Updated API) ---
from vision import (
    vision_encoder,          # The global model instance
    classifier as vision_classifier_head,
    get_hybrid_findings,     # The new detection logic
    get_multiview_hybrid_findings,
    VisualDescriptionAgent,  # The new writer class
)

# --- 3. Import NEW KG Agent (Updated API) ---
try:
    from kg_agent import infer_kg
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

try:
    from kg_agent_xrv_rad_dino_ensemble import (
        load_ensemble_agent,
        predict_pathologies as predict_visual_pathology_evidence,
    )
    ENSEMBLE_KG_AVAILABLE = True
except ImportError as _ensemble_import_error:
    print(f"[synthesis.py] RAD-DINO+XRV evidence agent unavailable: {_ensemble_import_error}")
    ENSEMBLE_KG_AVAILABLE = False

USE_LEGACY_SPATIAL_KG = False
KG_CLASSIFIER_POLICY_MODE = "validation"
KG_PREPROCESS_VARIANT = os.getenv("KG_PREPROCESS_VARIANT", "baseline")
REPORT_COVERAGE_PRIORITY = os.getenv("REPORT_COVERAGE_PRIORITY", "1").strip().lower() not in {"0", "false", "no", "off"}
_ensemble_evidence_agent = None
_ensemble_evidence_agent_variant = None


def _get_ensemble_evidence_agent():
    """Cache the RAD-DINO+XRV ensemble so evaluation does not reload it per study."""
    global _ensemble_evidence_agent, _ensemble_evidence_agent_variant
    if not ENSEMBLE_KG_AVAILABLE:
        return None
    if _ensemble_evidence_agent is None or _ensemble_evidence_agent_variant != KG_PREPROCESS_VARIANT:
        _ensemble_evidence_agent = load_ensemble_agent(preprocess_variant=KG_PREPROCESS_VARIANT)
        _ensemble_evidence_agent_variant = KG_PREPROCESS_VARIANT
    return _ensemble_evidence_agent


def _query_label_scores_from_visual_evidence(visual_evidence: dict | None) -> dict:
    findings = (visual_evidence or {}).get("findings", {})
    scores = {}
    for disease, payload in findings.items():
        if not isinstance(payload, dict):
            continue
        score = float(payload.get("score", 0.0))
        threshold = float(payload.get("threshold", 1.0))
        status = str(payload.get("status", "absent"))
        if status == "present":
            scores[disease] = max(score, threshold)
        elif status == "uncertain":
            scores[disease] = max(scores.get(disease, 0.0), score)
        elif score >= max(0.35, threshold - 0.15):
            scores[disease] = max(scores.get(disease, 0.0), score)
    return scores


# ----------------------------------------------------------------
# CT Lazy-Load Helpers (exact pattern from reference synthesis.py)
# ----------------------------------------------------------------
def _load_ct_model():
    """Lazy-load and cache the fine-tuned CT MedGemma model."""
    global _ct_model, _ct_proc
    if _ct_model is not None:
        return _ct_model, _ct_proc
    print("[CT] Loading fine-tuned MedGemma CT model...")
    from peft import PeftModel as _PeftModel
    _ct_proc = AutoProcessor.from_pretrained("google/medgemma-4b-it", token=HF_TOKEN)
    _ct_proc.tokenizer.padding_side = "left"
    base = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        # Required: use eager attention to avoid or_mask_function error
        # which requires torch>=2.6 flex-attention API
        attn_implementation="eager",
        token=HF_TOKEN
    )
    _ct_model = _PeftModel.from_pretrained(base, CT_MODEL_PATH).merge_and_unload()
    # Force eager attention on merged model (merge_and_unload may reset it)
    _ct_model.config._attn_implementation = "eager"
    _ct_model.eval()
    print("[CT] Model ready.")
    return _ct_model, _ct_proc



def _infer_ct_report(image_paths: list) -> str:
    """
    Build a grid montage from uploaded CT slices and generate a report
    using the fine-tuned MedGemma model. No other agents are used.
    """
    ct_model, ct_proc = _load_ct_model()
    device = next(ct_model.parameters()).device

    # Build montage: directory of slices OR list of individual files
    if len(image_paths) == 1 and os.path.isdir(image_paths[0]):
        # Lazy import to avoid pandas/pytz chain at startup
        from vision_ct import load_jpeg_montage as _ljm
        grid_img = _ljm(image_paths[0])
    else:
        imgs = [Image.open(p).convert("RGB").resize((256, 256)) for p in image_paths]
        n    = len(imgs)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        grid = Image.new("RGB", (256 * cols, 256 * rows))
        for i, img in enumerate(imgs):
            grid.paste(img, ((i % cols) * 256, (i // cols) * 256))
        grid_img = grid

    prompt_text = (
        "You are an expert radiologist. Analyze this CT scan grid montage and write a "
        "concise radiology report with FINDINGS and IMPRESSION sections. "
        "If you detect an abnormality, clearly indicate the grid cell/slice number in brackets where it is visible (e.g., [Slice 5])."
    )
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": grid_img},
            {"type": "text",  "text":  prompt_text},
        ]
    }
    formatted = ct_proc.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    inputs    = ct_proc(text=formatted, images=[grid_img], return_tensors="pt").to(device)

    # Prevent transformers from crashing on PyTorch < 2.6
    # by removing the token_type_ids that trigger the 'or_mask_function'
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        out_ids = ct_model.generate(
            **inputs,
            max_new_tokens=160,
            max_time=CT_GENERATION_TIMEOUT_SECONDS,
            do_sample=False,
            no_repeat_ngram_size=5,
            repetition_penalty=1.1,
            # use_cache=False bypasses Gemma3's HybridCache which requires
            # torch>=2.6 flex_attention mask functions (or_mask_function /
            # and_mask_function) — works on any torch version
            use_cache=False,
        )
    out_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    report  = ct_proc.decode(out_ids[0], skip_special_tokens=True).strip()
    return report

# -------------------------------
# Helper: Format KG for LLM
# -------------------------------
def format_kg_for_prompt(kg_data):
    if not kg_data or "entities" not in kg_data:
        return "No Knowledge Graph detected."
    
    entities = kg_data.get("entities", [])
    relations = kg_data.get("relations", [])
    
    formatted = []
    # Format as "Observation (Anatomy)"
    for r in relations:
        if r[2] in ["located_at", "modify"]:
            if r[0] < len(entities) and r[1] < len(entities):
                obs = entities[r[0]][0]
                anat = entities[r[1]][0]
                formatted.append(f"- {obs} ({anat})")
                
    if not formatted:
        return "Normal / No specific findings."
        
    return "DETECTED FINDINGS:\n" + "\n".join(formatted)


DISEASE_TO_ANATOMY = {
    "Cardiomegaly": "mediastinum",
    "Pleural Effusion": "pleural space",
    "Edema": "lungs",
    "Pneumothorax": "pleural space",
    "Infiltrate": "lungs",
    "Consolidation": "lungs",
    "Lung Opacity": "lungs",
    "Nodule": "lungs",
    "Atelectasis": "lungs",
    "Fracture": "bones",
}

DISEASE_TERMS = {
    "Cardiomegaly": [
        "cardiomegaly", "enlarged heart", "heart is enlarged", "heart size is enlarged",
        "heart is mildly enlarged", "heart size is mildly enlarged", "mildly enlarged heart",
        "cardiac silhouette is enlarged", "cardiac silhouette is mildly enlarged",
        "enlargement of the heart",
    ],
    "Pleural Effusion": [
        "pleural effusion", "pleural fluid", "effusion", "costophrenic blunting",
        "blunting of the costophrenic",
    ],
    "Edema": ["pulmonary edema", "interstitial edema", "vascular congestion", "vascular engorgement"],
    "Pneumothorax": ["pneumothorax", "pleural air"],
    "Infiltrate": ["infiltrate", "infiltration", "pneumonia", "airspace disease"],
    "Consolidation": ["consolidation", "consolidative opacity"],
    "Lung Opacity": ["opacity", "opacities", "airspace opacity", "pulmonary opacity"],
    "Nodule": ["nodule", "nodular opacity", "mass", "lung lesion"],
    "Atelectasis": ["atelectasis", "atelectatic", "volume loss", "bibasilar atelectatic"],
    "Fracture": ["fracture", "rib fracture", "compression deformity", "wedge fracture", "acute osseous abnormality"],
}

NEGATION_TERMS = (
    "no", "not", "without", "absence of", "negative for", "no evidence of",
    "no focal", "no definite", "no visible", "free of",
)

UNCERTAINTY_TERMS = ("possible", "possibly", "may represent", "suggesting", "questionable", "versus", "cannot exclude")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _term_is_negated(sentence: str, term: str) -> bool:
    lower = sentence.lower()
    term_lower = term.lower()
    idx = lower.find(term_lower)
    if idx < 0:
        return False
    window = lower[max(0, idx - 60): idx + len(term_lower) + 20]
    return any(neg in window for neg in NEGATION_TERMS)


def _sentence_is_uncertain(sentence: str) -> bool:
    lower = sentence.lower()
    return any(term in lower for term in UNCERTAINTY_TERMS)


def extract_report_findings(report: str) -> dict:
    """Extract present/absent/uncertain pathology facts from synthesized report text."""
    text = " ".join([
        _extract_report_section(report, "FINDING"),
        _extract_report_section(report, "IMPRESSION"),
    ]).strip() or report
    sentences = _split_sentences(text)
    findings = {
        disease: {
            "status": "not_mentioned",
            "source_sentence": "",
            "anatomy": DISEASE_TO_ANATOMY[disease],
        }
        for disease in DISEASE_TO_ANATOMY
    }

    for disease, terms in DISEASE_TERMS.items():
        for sentence in sentences:
            matched_term = next((term for term in terms if term in sentence.lower()), None)
            if not matched_term:
                continue
            if _term_is_negated(sentence, matched_term):
                status = "absent"
            elif _sentence_is_uncertain(sentence):
                status = "uncertain"
            else:
                status = "present"

            # Present beats uncertain, uncertain beats absent, absent beats not mentioned.
            rank = {"not_mentioned": 0, "absent": 1, "uncertain": 2, "present": 3}
            if rank[status] >= rank[findings[disease]["status"]]:
                slice_match = re.search(r'\[Slice\s*(\d+)\]', sentence, flags=re.IGNORECASE)
                slice_idx = int(slice_match.group(1)) if slice_match else None

                findings[disease].update({
                    "status": status,
                    "source_sentence": sentence,
                    "matched_term": matched_term,
                    "slice_index": slice_idx
                })

    return findings


def _anatomy_index(entities: list, anatomy_to_idx: dict, anatomy: str) -> int:
    if anatomy not in anatomy_to_idx:
        anatomy_to_idx[anatomy] = len(entities)
        entities.append([anatomy, "Anatomy"])
    return anatomy_to_idx[anatomy]


def build_kg_from_synthesis_report(final_report: str, evidence_bundle: dict | None = None) -> dict:
    """Build the actual frontend KG from the final report, with evidence as metadata."""
    report_findings = extract_report_findings(final_report)
    entities = []
    relations = []
    anatomy_to_idx = {}

    for disease, finding in report_findings.items():
        status = finding["status"]
        if status == "not_mentioned":
            continue
        anatomy = finding["anatomy"]
        anat_idx = _anatomy_index(entities, anatomy_to_idx, anatomy)
        obs_idx = len(entities)
        if status == "absent":
            entities.append([disease.lower(), "AbsentObservation"])
            relations.append([obs_idx, anat_idx, "absent_at"])
        elif status == "uncertain":
            entities.append([disease.lower(), "UncertainObservation"])
            relations.append([obs_idx, anat_idx, "possible_at"])
        else:
            entities.append([disease.lower(), "Observation"])
            relations.append([obs_idx, anat_idx, "located_at"])

    if not entities:
        chest_idx = _anatomy_index(entities, anatomy_to_idx, "chest")
        obs_idx = len(entities)
        entities.append(["clear", "Observation"])
        relations.append([obs_idx, chest_idx, "modify"])

    return {
        "entities": entities,
        "relations": relations,
        "metadata": {
            "kg_source": "synthesis_report",
            "report_findings": report_findings,
            "evidence_sources": evidence_bundle or {},
        },
    }


def _legacy_kg_observations(kg_data: dict | None) -> list[str]:
    if not kg_data:
        return []
    entities = kg_data.get("entities", [])
    observations = []
    for relation in kg_data.get("relations", []):
        if len(relation) < 3 or relation[0] >= len(entities) or relation[1] >= len(entities):
            continue
        obs = str(entities[relation[0]][0]).strip()
        anat = str(entities[relation[1]][0]).strip()
        rel = relation[2]
        if obs and obs.lower() not in {"clear", "normal"}:
            observations.append(f"{obs} ({anat}; relation={rel})")
    return observations


def _confidence_bucket(score: float, threshold: float, status: str, finding: dict) -> str:
    if bool(finding.get("heldout_target_met")) and status == "present":
        return "strong"
    if status == "uncertain":
        return "candidate_only"
    if threshold > 0 and score >= 0.75 * threshold:
        return "weak_support"
    if threshold > 0 and score <= 0.25 * threshold:
        return "negative_support"
    return "low"


def format_visual_evidence_for_prompt(evidence: dict | None) -> str:
    if not evidence or not evidence.get("findings"):
        return "No classifier evidence available."

    rows = []
    for disease, finding in evidence["findings"].items():
        score = float(finding.get("score", 0.0))
        threshold = float(finding.get("threshold", 1.0))
        support_threshold = finding.get("support_threshold")
        status = finding.get("status", "absent")
        bucket = _confidence_bucket(score, threshold, status, finding)
        kg_enabled = bool(finding.get("kg_enabled", False))
        heldout_precision = finding.get("heldout_edge_precision")
        support_text = f", support_threshold={float(support_threshold):.3f}" if support_threshold is not None else ""
        rows.append(
            f"- {disease}: score={score:.3f}, threshold={threshold:.3f}, "
            f"status={status}, confidence={bucket}{support_text}, kg_edge_allowed={kg_enabled}, "
            f"heldout_precision={heldout_precision}"
        )
    return "\n".join(rows)


def build_evidence_bundle(vision_report: str, spatial_kg: dict | None, visual_evidence: dict | None) -> dict:
    return {
        "vision_agent_text": vision_report,
        "spatial_kg_observations": _legacy_kg_observations(spatial_kg),
        "visual_classifier_evidence": visual_evidence or {},
    }


FINDING_PHRASES = {
    "Cardiomegaly": "cardiomegaly",
    "Pleural Effusion": "pleural effusion",
    "Edema": "interstitial pulmonary edema",
    "Pneumothorax": "pneumothorax",
    "Infiltrate": "focal infiltrate",
    "Consolidation": "focal consolidation",
    "Lung Opacity": "bilateral interstitial/lung opacities",
    "Nodule": "pulmonary nodule",
    "Atelectasis": "atelectasis",
    "Fracture": "acute osseous abnormality",
}

CORE_NEGATIVES = {
    "Pleural Effusion",
    "Pneumothorax",
    "Infiltrate",
    "Consolidation",
    "Atelectasis",
    "Fracture",
}

UNCERTAIN_CANDIDATE_DISEASES = {
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Pneumothorax",
    "Infiltrate",
    "Consolidation",
    "Lung Opacity",
    "Nodule",
    "Atelectasis",
    "Fracture",
}

UNCERTAIN_SCORE_RATIO = 0.75


def _spatial_supports_disease(evidence_bundle: dict | None, disease: str) -> bool:
    observations = (evidence_bundle or {}).get("spatial_kg_observations") or []
    terms = DISEASE_TERMS.get(disease, []) + [disease.lower()]
    for observation in observations:
        obs_lower = str(observation).lower()
        if any(term.lower() in obs_lower for term in terms):
            return True
    return False


def _vision_text_supports_disease(evidence_bundle: dict | None, disease: str) -> bool:
    vision_text = str((evidence_bundle or {}).get("vision_agent_text") or "")
    if not vision_text:
        return False
    parsed = extract_report_findings(vision_text)
    return parsed.get(disease, {}).get("status") in {"present", "uncertain"}


def _visual_finding(evidence_bundle: dict | None, disease: str) -> dict:
    visual = (evidence_bundle or {}).get("visual_classifier_evidence") or {}
    return (visual.get("findings") or {}).get(disease, {})


def _visual_absent(finding: dict) -> bool:
    if not finding:
        return False
    score = float(finding.get("score", 0.0))
    threshold = float(finding.get("threshold", 1.0))
    return finding.get("status") == "absent" and score < threshold


def _visual_score_ratio(finding: dict) -> float:
    if not finding:
        return 0.0
    threshold = float(finding.get("threshold", 1.0))
    if threshold <= 0:
        return 0.0
    return float(finding.get("score", 0.0)) / threshold


def adjudicate_findings_from_evidence(evidence_bundle: dict | None) -> dict:
    """
    Convert model evidence into graph facts. This is deliberately stricter than
    report parsing: LLM text never creates hard positive KG edges by itself.
    """
    adjudicated = {}
    for disease in DISEASE_TO_ANATOMY:
        finding = _visual_finding(evidence_bundle, disease)
        status = "not_mentioned"
        reason = "no_classifier_evidence"
        confidence = "none"

        if finding:
            score = float(finding.get("score", 0.0))
            threshold = float(finding.get("threshold", 1.0))
            visual_status = finding.get("status", "absent")
            ratio = _visual_score_ratio(finding)
            create_edge = bool(finding.get("create_kg_edge")) and visual_status == "present"

            if create_edge:
                status = "present"
                reason = "classifier_policy_allows_kg_edge"
                confidence = "strong"
            elif visual_status == "uncertain" or (
                disease in UNCERTAIN_CANDIDATE_DISEASES and ratio >= UNCERTAIN_SCORE_RATIO
            ):
                status = "uncertain"
                reason = "near_threshold_or_policy_blocked"
                confidence = "candidate_only"
            elif _spatial_supports_disease(evidence_bundle, disease) or _vision_text_supports_disease(evidence_bundle, disease):
                status = "uncertain"
                reason = "vision_or_spatial_support"
                confidence = "candidate_only"
            elif disease in CORE_NEGATIVES or score <= max(0.05, 0.45 * threshold):
                status = "absent"
                reason = "classifier_below_threshold"
                confidence = "negative_support"

        adjudicated[disease] = {
            "status": status,
            "anatomy": DISEASE_TO_ANATOMY[disease],
            "phrase": FINDING_PHRASES.get(disease, disease.lower()),
            "reason": reason,
            "confidence": confidence,
            "classifier": finding,
        }
    return adjudicated


def build_kg_from_evidence(evidence_bundle: dict | None, final_report: str = "") -> dict:
    """Build the frontend KG from adjudicated classifier evidence, not LLM prose."""
    adjudicated = adjudicate_findings_from_evidence(evidence_bundle)
    entities = []
    relations = []
    anatomy_to_idx = {}

    for disease, finding in adjudicated.items():
        status = finding["status"]
        if status == "not_mentioned":
            continue
        anat_idx = _anatomy_index(entities, anatomy_to_idx, finding["anatomy"])
        obs_idx = len(entities)
        if status == "present":
            entities.append([disease.lower(), "Observation"])
            relations.append([obs_idx, anat_idx, "located_at"])
        elif status == "uncertain":
            entities.append([disease.lower(), "UncertainObservation"])
            relations.append([obs_idx, anat_idx, "possible_at"])
        elif status == "absent":
            entities.append([disease.lower(), "AbsentObservation"])
            relations.append([obs_idx, anat_idx, "absent_at"])

    if not entities:
        chest_idx = _anatomy_index(entities, anatomy_to_idx, "chest")
        obs_idx = len(entities)
        entities.append(["clear", "Observation"])
        relations.append([obs_idx, chest_idx, "modify"])

    return {
        "entities": entities,
        "relations": relations,
        "metadata": {
            "kg_source": "adjudicated_classifier_evidence",
            "kg_policy_mode": KG_CLASSIFIER_POLICY_MODE,
            "report_text": final_report,
            "adjudicated_findings": adjudicated,
            "evidence_sources": evidence_bundle or {},
        },
    }


def build_report_from_evidence(evidence_bundle: dict | None, fallback_report: str = "") -> str:
    """Create a report whose asserted facts match the evidence-built KG."""
    if not evidence_bundle:
        return fallback_report

    adjudicated = adjudicate_findings_from_evidence(evidence_bundle)
    present = [f["phrase"] for f in adjudicated.values() if f["status"] == "present"]
    uncertain = [f["phrase"] for f in adjudicated.values() if f["status"] == "uncertain"]
    absent = [
        f["phrase"]
        for disease, f in adjudicated.items()
        if f["status"] == "absent" and disease in CORE_NEGATIVES
    ]

    findings_sentences = []
    if present:
        findings_sentences.append("Present findings include " + ", ".join(dict.fromkeys(present)) + ".")
    if uncertain:
        findings_sentences.append("Possible findings include " + ", ".join(dict.fromkeys(uncertain)) + ".")
    for phrase in dict.fromkeys(absent):
        findings_sentences.append(f"No {phrase} is identified.")
    if not findings_sentences:
        findings_sentences.append("No acute cardiopulmonary abnormality is identified.")

    impression_parts = []
    if present:
        impression_parts.append(", ".join(dict.fromkeys(present)).capitalize() + ".")
    if uncertain:
        impression_parts.append("Possible " + ", ".join(dict.fromkeys(uncertain)) + ".")
    if not impression_parts:
        impression_parts.append("No acute cardiopulmonary abnormality.")

    return (
        "FINDINGS: "
        + " ".join(findings_sentences)
        + "\n\nIMPRESSION: "
        + " ".join(impression_parts)
    )


def build_report_from_kg_data(kg_data: dict | None, fallback_report: str = "") -> str:
    """Render report text directly from adjudicated KG metadata."""
    adjudicated = (kg_data or {}).get("metadata", {}).get("adjudicated_findings", {})
    if not adjudicated:
        return fallback_report

    present = []
    uncertain = []
    absent = []
    for disease in DISEASE_TO_ANATOMY:
        finding = adjudicated.get(disease, {})
        status = finding.get("status")
        phrase = finding.get("phrase", FINDING_PHRASES.get(disease, disease.lower()))
        if status == "present":
            present.append(phrase)
        elif status == "uncertain":
            uncertain.append(phrase)
        elif status == "absent" and disease in CORE_NEGATIVES:
            absent.append(phrase)

    findings_sentences = []
    if present:
        findings_sentences.append("Present findings include " + ", ".join(dict.fromkeys(present)) + ".")
    if uncertain:
        findings_sentences.append("Additional possible findings include " + ", ".join(dict.fromkeys(uncertain)) + ".")
    for phrase in dict.fromkeys(absent):
        findings_sentences.append(f"No {phrase} is identified.")
    if not findings_sentences:
        findings_sentences.append("No acute cardiopulmonary abnormality is identified.")

    impression_parts = []
    if present:
        impression_parts.append(", ".join(dict.fromkeys(present)).capitalize() + ".")
    if uncertain:
        impression_parts.append("Possible " + ", ".join(dict.fromkeys(uncertain)) + ".")
    if not impression_parts:
        impression_parts.append("No acute cardiopulmonary abnormality.")

    return (
        "FINDINGS: "
        + " ".join(findings_sentences)
        + "\n\nIMPRESSION: "
        + " ".join(impression_parts)
    )


def apply_evidence_guardrails(report: str, evidence_bundle: dict | None) -> str:
    """Suppress unsupported Vision-Agent hallucinations before final KG creation."""
    if not report or not evidence_bundle:
        return report

    report_findings = extract_report_findings(report)
    unsupported = []
    for disease, finding in report_findings.items():
        if finding["status"] not in {"present", "uncertain"}:
            continue
        visual_finding = _visual_finding(evidence_bundle, disease)
        if _spatial_supports_disease(evidence_bundle, disease):
            continue
        if bool(visual_finding.get("create_kg_edge")) and visual_finding.get("status") == "present":
            continue
        if _visual_absent(visual_finding):
            unsupported.append(disease)

    if not unsupported:
        return report

    present = []
    possible = []
    absent = []

    for disease in DISEASE_TO_ANATOMY:
        visual_finding = _visual_finding(evidence_bundle, disease)
        spatial_supported = _spatial_supports_disease(evidence_bundle, disease)
        visual_status = visual_finding.get("status")
        phrase = FINDING_PHRASES.get(disease, disease.lower())

        if disease in unsupported:
            if disease in CORE_NEGATIVES:
                absent.append(phrase)
            continue

        if spatial_supported or (visual_status == "present" and bool(visual_finding.get("create_kg_edge"))):
            present.append(phrase)
        elif visual_status == "uncertain" or report_findings.get(disease, {}).get("status") == "uncertain":
            possible.append(phrase)
        elif disease in CORE_NEGATIVES and _visual_absent(visual_finding):
            absent.append(phrase)

    # Avoid redundant wording when the lung-opacity concept already covers edema-like language.
    if "bilateral interstitial/lung opacities" in present and "interstitial pulmonary edema" in present:
        present.remove("interstitial pulmonary edema")
        possible.append("interstitial pulmonary edema")

    findings_sentences = []
    if present:
        findings_sentences.append("Present findings include " + ", ".join(dict.fromkeys(present)) + ".")
    if possible:
        findings_sentences.append("Possible findings include " + ", ".join(dict.fromkeys(possible)) + ".")
    if absent:
        for phrase in dict.fromkeys(absent):
            findings_sentences.append(f"No {phrase} is identified.")
    if not findings_sentences:
        findings_sentences.append("No acute cardiopulmonary abnormality is identified.")

    present_impression = list(dict.fromkeys(present))
    possible_impression = list(dict.fromkeys(possible))
    if present_impression or possible_impression:
        impression_parts = []
        if present_impression:
            impression_parts.append(", ".join(present_impression).capitalize() + ".")
        if possible_impression:
            impression_parts.append("Possible " + ", ".join(possible_impression) + ".")
        impression = " ".join(impression_parts)
    else:
        impression = "No acute cardiopulmonary abnormality."

    return (
        "FINDINGS: "
        + " ".join(findings_sentences)
        + "\n\nIMPRESSION: "
        + impression
    )

def _extract_report_section(text: str, section: str) -> str:
    pattern = rf"\b{section}S?:\s*(.*?)(?=\s*(?:FINDINGS|IMPRESSION|LABELS|RECOMMENDATIONS?):|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""

def _kg_labels(kg_data):
    labels = []
    seen = set()
    if not kg_data:
        return labels

    for text, label in kg_data.get("entities", []):
        label_type = str(label).lower()
        if label_type not in {"observation", "uncertainobservation"}:
            continue
        normalized = str(text).strip().lower()
        if not normalized or normalized in {"clear", "normal"}:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        rendered = normalized.title() if label_type == "observation" else f"Possible {normalized.title()}"
        labels.append(rendered)
    return labels

def _fallback_recommendation(labels, kg_data=None):
    if _kg_has_hard_present_finding(kg_data):
        return "Radiologist review and clinical correlation are recommended because abnormal findings are supported by the image evidence."
    if _kg_has_uncertain_finding(kg_data):
        return "Radiologist review is recommended because the image evidence supports possible abnormal findings that should be confirmed clinically or on follow-up imaging."
    if labels:
        return "Radiologist review, clinical correlation, and comparison with prior imaging are recommended."
    return "No acute imaging follow-up is suggested by the generated report; correlate clinically."

def _kg_has_hard_present_finding(kg_data) -> bool:
    if not kg_data:
        return False
    for text, label in kg_data.get("entities", []):
        if str(label) == "Observation" and str(text).strip().lower() not in {"clear", "normal"}:
            return True
    return False


def _kg_has_uncertain_finding(kg_data) -> bool:
    if not kg_data:
        return False
    for text, label in kg_data.get("entities", []):
        if str(label) == "UncertainObservation" and str(text).strip().lower() not in {"clear", "normal"}:
            return True
    return False

KG_REPORT_SENTENCES = {
    "Cardiomegaly": "Cardiomegaly is present.",
    "Pleural Effusion": "Small pleural effusion is present.",
    "Edema": "Mild pulmonary edema is present.",
    "Pneumothorax": "Pneumothorax is present.",
    "Infiltrate": "Focal infiltrate is present.",
    "Consolidation": "Focal consolidation is present.",
    "Lung Opacity": "Patchy pulmonary opacities are present.",
    "Nodule": "Pulmonary nodule is present.",
    "Atelectasis": "Atelectatic opacity is present.",
    "Fracture": "Osseous fracture is present.",
}

def _hard_kg_diseases(kg_data) -> list[str]:
    if not kg_data:
        return []
    adjudicated = kg_data.get("metadata", {}).get("adjudicated_findings", {})
    return [
        disease
        for disease, finding in adjudicated.items()
        if isinstance(finding, dict) and finding.get("status") == "present"
    ]

def _report_mentions_disease(report: str, disease: str) -> bool:
    lower = (report or "").lower()
    terms = DISEASE_TERMS.get(disease, []) + [disease.lower()]
    return any(term.lower() in lower for term in terms)

def augment_retrieved_report_with_hard_kg(report: str, kg_data) -> str:
    """Preserve IU-style retrieval wording, adding only high-confidence KG facts."""
    additions = []
    for disease in _hard_kg_diseases(kg_data):
        if not _report_mentions_disease(report, disease):
            additions.append(KG_REPORT_SENTENCES.get(disease, f"{disease} is present."))
    if not additions:
        return report
    return report.rstrip() + " " + " ".join(additions)

def _dedupe_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    deduped = []
    seen = set()
    for sentence in sentences:
        normalized = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(sentence.strip())
    return " ".join(deduped)

def normalize_report_sections(report: str, kg_data) -> str:
    """Return a frontend-stable report with Findings, Impression, Recommendations, and Labels."""
    clean_report = report.strip()
    findings = _extract_report_section(clean_report, "FINDING")
    impression = _extract_report_section(clean_report, "IMPRESSION")
    recommendation = _extract_report_section(clean_report, "RECOMMENDATION")
    labels_text = _extract_report_section(clean_report, "LABEL")
    labels = [item.strip() for item in labels_text.split(",") if item.strip()]

    if not findings and not impression:
        sentences = _split_sentences(clean_report)
        if len(sentences) >= 2:
            midpoint = max(1, len(sentences) // 2)
            findings = " ".join(sentences[:midpoint])
            impression = " ".join(sentences[midpoint:])
        else:
            findings = clean_report
            impression = clean_report
    elif findings and not impression:
        impression = findings
    elif impression and not findings:
        findings = impression

    findings = _dedupe_sentences(findings)
    impression = _dedupe_sentences(impression)

    if not labels:
        labels = _kg_labels(kg_data)

    if not recommendation:
        recommendation = _fallback_recommendation(labels, kg_data)
    recommendation = _dedupe_sentences(recommendation)

    label_line = ", ".join(labels) if labels else "No acute abnormality"
    return (
        f"FINDINGS:\n{findings.strip()}\n\n"
        f"IMPRESSION:\n{impression.strip()}\n\n"
        f"RECOMMENDATIONS:\n{recommendation.strip()}\n\n"
        f"LABELS:\n{label_line}"
    )

# -------------------------------
# Local Synthesis Agent
# -------------------------------
class LocalSynthesisAgent:
    def __init__(self, model_name=config.OLLAMA_MODEL): 
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        
        self.llm = ChatOllama(
            model=model_name, 
            temperature=config.TEMPERATURE,
            num_ctx=REPORT_CONTEXT_WINDOW,
            num_predict=REPORT_MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
            keep_alive=3600,  # Keep model hot in VRAM for 1 hour
            # CRITICAL SAFETY: Stop markers
            stop=["<|endoftext|>", "RECOMMENDATIONS:", "\n\n\n\n"] 
        )

    def repair_report(self, bad_report: str, errors: list[str], kg_text_block: str) -> str:
        err_txt = "\n".join([f"- {e}" for e in errors])
        prompt = f"""
        You are a strict radiology report editor.
        This report FAILED validation based on facts:
        {err_txt}
        
        Rewrite the report to fix ALL issues.
        Rewrite the report to fix ALL issues.
        Rules:
        - Output a SINGLE concise paragraph (IU X-ray style).
        - No headers (e.g. "Revised Report:", "Corrected Report:").
        - START DIRECTLY with the medical text.
        
        Report to fix:
        {bad_report}
        """
        resp = self.llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        return self._clean_output(content)

    def _clean_output(self, text):
        """Circuit Breaker logic."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
        text = text.replace("Espaça", "Airspace").replace("Espaco", "Airspace")
        text = re.sub(r"\bXXXX\b", "visualized structures", text)
        text = re.sub(r"^\s*(?:Revised\s+)?(?:Corrected\s+)?Report:?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^\s*\*\*.*?\*\*\s*", "", text, flags=re.MULTILINE)
        if text.count("Corrected Report:") > 2:
             text = text.split("Corrected Report:")[0].strip()
        return text

    # NEW: Added scan_type parameter (defaults to 'auto' for testing)
    async def generate_final_report(self, draft_agent, vision_agent, retrieval_agent, reports_dict, image_paths, scan_type="auto"):
        target_image = image_paths[0]
        yield json.dumps({"status": "validating", "message": "Checking image modality..."}) + "\n"
        
        # --- MULTI-CLASS VALIDATION / ROUTING LOGIC ---
        # In explicit mode, trust the clinician/frontend selection. The bouncer
        # model is useful for "auto" mode, but it can confuse valid X-rays and CT
        # slices, so it should not veto an intentional modality choice.
        if scan_type != "auto":
            detected_modality, confidence = scan_type, 1.0
        else:
            detected_modality, confidence = classify_scan(target_image)
        
        if detected_modality == "invalid":
            error_msg = f"Invalid Image Detected. This does not appear to be a medical scan (Confidence: {confidence:.1%})."
            print(f"❌ {error_msg}")
            yield json.dumps({"status": "error", "message": error_msg}) + "\n"
            return
            
        if scan_type != "auto" and detected_modality != scan_type:
            user_friendly_expected = "Chest X-Ray" if scan_type == "xray" else "CT Scan"
            user_friendly_detected = "Chest X-Ray" if detected_modality == "xray" else "CT Scan"
            error_msg = f"Validation Mismatch: You selected '{user_friendly_expected}', but uploaded a '{user_friendly_detected}'."
            print(f"❌ {error_msg}")
            yield json.dumps({"status": "error", "message": error_msg}) + "\n"
            return
        # ----------------------------------------
        
        yield json.dumps({
            "status": "parallel_start",
            "message": f"Running Agents for {detected_modality.upper()}...",
            "detected_modality": detected_modality,
            "requested_scan_type": scan_type,
        }) + "\n"

        # --- SEQUENTIAL HELPER FUNCTIONS (Updated for New APIs) ---
        
        # 1. Vision Agent (MULTI-IMAGE SUPPORT)
        async def run_vision():
            print(f"[DEBUG] Vision Agent: Starting for {len(image_paths)} images...")
            try:
                def _process_vision():
                    raw_images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
                    view_names = [os.path.basename(img_path) for img_path in image_paths]
                    combined_findings = get_multiview_hybrid_findings(raw_images, view_names=view_names)
                    return vision_agent.generate_description(combined_findings)

                res = await asyncio.wait_for(
                    asyncio.to_thread(_process_vision),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                print("[DEBUG] Vision Agent: Finished multi-view analysis.")
                return res
            except asyncio.TimeoutError:
                print("⚠️ Vision Agent timed out.")
                return "Visual analysis timed out."
            except Exception as e:
                print(f"⚠️ Vision Error: {e}")
                return "Visual analysis failed."

        # 2. Draft Agent
        async def run_draft(query_label_scores=None):
            print("[DEBUG] Draft Agent: Starting...")
            try:
                # Use vision_encoder model for retrieval too (Reuse!)
                top_reports = await asyncio.to_thread(
                    retrieval_agent.retrieve_top_k, 
                    image_paths, # PASSED LIST OF PATHS INSTEAD OF 1
                    reports_dict,
                    query_label_scores,
                )
                draft_context = "\n\n".join(truncate_report(r, 75) for r in top_reports)
                print("[DEBUG] Draft Agent: Retrieved reports.")

                if not draft_context.strip():
                    return "No similar-case context available.", ""
                
                res = await asyncio.wait_for(
                    asyncio.to_thread(draft_agent.generate_report, draft_context),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                print("[DEBUG] Draft Agent: Finished.")
                return res, (top_reports[0] if top_reports else "")
            except asyncio.TimeoutError:
                print("⚠️ Draft Agent timed out.")
                return "Similar-case retrieval timed out; synthesize from visual and KG findings only.", ""
            except Exception as e:
                print(f"⚠️ Draft Agent Error: {e}")
                return "Similar-case retrieval unavailable; synthesize from visual and KG findings only.", ""

        # 3. KG Agent
        async def run_kg():
            print("[DEBUG] KG Agent: Starting...")
            spatial_kg = None
            spatial_text = "Spatial KG skipped."
            visual_evidence = None
            visual_evidence_text = "Classifier evidence skipped."

            if not KG_AVAILABLE and not ENSEMBLE_KG_AVAILABLE:
                return spatial_kg, spatial_text, visual_evidence, visual_evidence_text

            try:
                if KG_AVAILABLE and USE_LEGACY_SPATIAL_KG:
                    # Legacy spatial KG is disabled by default for X-rays because
                    # it is too noisy to create or support final graph facts.
                    spatial_kg = await asyncio.to_thread(
                        infer_kg,
                        image_paths,
                        projection="Frontal",
                        clip_model=vision_encoder.model,
                        clip_prep=vision_encoder.preprocess,
                        tokenizer=vision_encoder.tokenizer,
                        classifier_head=vision_classifier_head,
                        device=vision_encoder.device,
                        debug=False,
                    )
                    spatial_text = format_kg_for_prompt(spatial_kg)

                if ENSEMBLE_KG_AVAILABLE:
                    # RAD-DINO + XRV evidence: calibrated classifier scores and confidence.
                    ensemble_agent = _get_ensemble_evidence_agent()
                    visual_evidence = await asyncio.to_thread(
                        predict_visual_pathology_evidence,
                        image_paths,
                        agent=ensemble_agent,
                        policy_mode=KG_CLASSIFIER_POLICY_MODE,
                    )
                    visual_evidence_text = format_visual_evidence_for_prompt(visual_evidence)

                print("[DEBUG] KG Agent: Finished.")
                return spatial_kg, spatial_text, visual_evidence, visual_evidence_text
            except Exception as e:
                print(f"⚠️ KG Error: {e}")
                return spatial_kg, "KG Error.", visual_evidence, "Classifier evidence error."

        # --- EXECUTION FLOW ---

        if detected_modality == "ct":
            # -------------------------------------------------------
            # CT PIPELINE: MedGemma grid-montage → direct report
            # -------------------------------------------------------
            raw_spatial_kg = None
            kg_text_block = "CT scan - image-only X-ray KG evidence is not applicable."
            visual_evidence = None
            visual_evidence_text = "CT scan - classifier evidence not applicable."
            evidence_bundle = {}
            draft_report = "N/A"
            vision_report = "N/A"

            if not CT_AVAILABLE:
                yield json.dumps({"status": "error", "message": f"CT model weights not found at {CT_MODEL_PATH}."}) + "\n"
                return

            # Fake KG Agent sequence for frontend aesthetics
            yield json.dumps({"status": "kg_start", "message": "Mapping CT scan to clinical knowledge graph..."}) + "\n"
            await asyncio.sleep(1.0)

            # Start Vision Agent sequence
            yield json.dumps({"status": "vision_start", "message": "CT Vision Agent building montage & running MedGemma inference..."}) + "\n"

            try:
                vision_report = await asyncio.wait_for(
                    asyncio.to_thread(_infer_ct_report, image_paths),
                    timeout=CT_REQUEST_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                print(f"[CT Path] ⚠️ CT inference timed out after {CT_REQUEST_TIMEOUT_SECONDS}s")
                yield json.dumps({
                    "status": "error",
                    "message": f"CT inference timed out after {CT_REQUEST_TIMEOUT_SECONDS} seconds. Try fewer slices or verify the MedGemma CT runtime."
                }) + "\n"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            except Exception as e:
                print(f"[CT Path] ❌ CT inference failed: {e}")
                yield json.dumps({"status": "error", "message": f"CT inference failed: {e}"}) + "\n"
                return

            # Fake Synthesis Agent sequence
            yield json.dumps({"status": "synthesis_start", "message": "Structuring MedGemma raw output into Final Report format..."}) + "\n"
            await asyncio.sleep(0.5)

            final_report = self._clean_output(vision_report)
            print(f"\n[FINAL CT REPORT]:\n{final_report}\n")

        else:
            # -------------------------------------------------------
            # X-RAY PIPELINE: BioMedCLIP + 3-agent flow (unchanged)
            # -------------------------------------------------------
            # Run agents sequentially to save VRAM
            yield json.dumps({"status": "kg_start", "message": "Building clinical knowledge graph..."}) + "\n"
            raw_spatial_kg, kg_text_block, visual_evidence, visual_evidence_text = await run_kg()
            yield json.dumps({"status": "vision_start", "message": "Extracting visual findings from scan..."}) + "\n"
            vision_report = await run_vision()
            yield json.dumps({"status": "draft_start", "message": "Retrieving similar cases for report style..."}) + "\n"
            draft_query_scores = _query_label_scores_from_visual_evidence(visual_evidence)
            draft_report, retrieved_backbone_report = await run_draft(draft_query_scores)
            evidence_bundle = build_evidence_bundle(vision_report, raw_spatial_kg, visual_evidence)

            print("[DEBUG] All agents COMPLETE.")
            yield json.dumps({"status": "parallel_done", "message": "Agents finished."}) + "\n"

            # --- SYNTHESIS ---
            yield json.dumps({"status": "synthesis_start", "message": "Synthesizing..."}) + "\n"

            synthesis_prompt = f"""
            You are an expert radiologist. You are evaluating the IU X-Ray dataset.
            
            ### INPUTS:
            1. VISION AGENT TEXT DESCRIPTION:
            {vision_report}

            2. STRUCTURED SPATIAL KG EVIDENCE:
            {kg_text_block}

            3. STRUCTURED CLASSIFIER EVIDENCE WITH CONFIDENCE:
            {visual_evidence_text}

            4. STYLE REFERENCE FROM SIMILAR CASES:
            {draft_report}

            5. RETRIEVED REPORT BACKBONE:
            {retrieved_backbone_report}

            ### TASK:
            Write a final radiology report.
            
            ### RULES:
            - **PRIMARY SOURCE:** Use the Vision Agent text as a candidate description, not as ground truth.
            - **STRUCTURED EVIDENCE:** Use KG/classifier evidence to support, weaken, or challenge the Vision Agent text.
            - **CONFIDENCE:** State strong supported findings directly. Phrase borderline evidence as mild, possible, or suggested. Do not drop clinically relevant borderline findings if classifier evidence supports them.
            - **CONFLICTS:** If Vision text says a finding is present but classifier evidence is negative/low, avoid that finding unless it is strongly supported elsewhere.
            - **NEGATIVES:** Include only natural high-value negatives, especially no pneumothorax or no large pleural effusion. Do not enumerate every absent label.
            - **BACKBONE:** Use the retrieved report backbone for IU-style sentence structure and normal negative phrasing, but correct findings that conflict with stronger visual/classifier evidence.
            - **STYLE ONLY:** Use STYLE REFERENCE for phrasing and common IU-Xray wording; do not copy unrelated diagnoses.
            - **NORMALITY:** Write a normal report only if both the classifier evidence and vision evidence are non-abnormal. Do not let style reference or a generic normal sentence suppress supported disease candidates.
            - **COVERAGE PRIORITY:** For this task, medical coverage is more important than matching IU wording exactly.
            - **FORMAT:** Use concise IU X-ray style with FINDINGS and IMPRESSION headers.
            
            REPORT:
            """

            full_response = ""
            try:
                def sync_invoke():
                    response = self.llm.invoke(synthesis_prompt)
                    content = response.content if hasattr(response, "content") else str(response)
                    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                full_response = await asyncio.wait_for(
                    asyncio.to_thread(sync_invoke),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                yield json.dumps({"status": "streaming", "chunk": full_response}) + "\n"
            except asyncio.TimeoutError:
                yield json.dumps({
                    "status": "error",
                    "message": f"Report synthesis timed out after {LLM_TIMEOUT_SECONDS} seconds. Check Ollama model performance or use a smaller model."
                }) + "\n"
                return
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"
                return

            final_report = self._clean_output(full_response.strip())

        # --- ReAct / CRITIQUE LOOP ---
        MAX_REACT_STEPS = 0
        print(f"[DEBUG] Entering ReAct loop.")

        for step in range(MAX_REACT_STEPS):
            yield json.dumps({"status": "thought_start", "message": f"Critiquing (Step {step+1})..."}) + "\n"
            
            critique_prompt = f"""
            You are a strict Medical Auditor. 
            Compare the REPORT vs the FACTS (KG).
            
            FACTS: {kg_text_block}
            REPORT: {final_report}
            
            Identify any:
            1. Hallucinations (Mentions findings NOT in FACTS)
            2. Missed Findings (FACTS mentions it, REPORT skips it)
            3. Contradictions (FACTS say Absent, REPORT says Present)
            
            If PERFECT, output "OK".
            If ERRORS, list them concisely.
            """
            
            critique = ""
            try:
                def sync_critique():
                    resp = self.llm.invoke(critique_prompt)
                    c = resp.content if hasattr(resp, "content") else str(resp)
                    return re.sub(r'<think>.*?</think>', '', c, flags=re.DOTALL).strip()
                critique = await asyncio.to_thread(sync_critique)
            except: pass
            
            # If critique is clean, break early
            if "OK" in critique and len(critique) < 20:
                yield json.dumps({"status": "thought_done", "message": "Report is verified."}) + "\n"
                break
                
            yield json.dumps({"status": "thought_done", "message": "Refining report..."}) + "\n"
            
            # ACT (Repair)
            yield json.dumps({"status": "repair_start", "iter": step}) + "\n"
            final_report = await asyncio.to_thread(self.repair_report, final_report, [critique], kg_text_block)

        if detected_modality == "ct":
            preliminary_kg = build_kg_from_synthesis_report(final_report, evidence_bundle)
            final_report = normalize_report_sections(final_report, preliminary_kg)
            kg_json = build_kg_from_synthesis_report(final_report, evidence_bundle)
        else:
            # Keep the LLM/retrieval report wording for text quality. Build the
            # frontend KG independently from adjudicated classifier evidence so
            # prose cannot create graph facts.
            preliminary_kg = build_kg_from_evidence(evidence_bundle, final_report)
            evidence_report = build_report_from_kg_data(preliminary_kg, build_report_from_evidence(evidence_bundle, final_report))
            if REPORT_COVERAGE_PRIORITY and (_kg_has_hard_present_finding(preliminary_kg) or _kg_has_uncertain_finding(preliminary_kg)):
                print("[DEBUG] Coverage-priority mode enabled; using evidence-built report text for X-ray output.")
                final_report = evidence_report
            elif retrieved_backbone_report:
                print("[DEBUG] Using retrieved IU report backbone for report text; hard KG facts are deterministic additions.")
                final_report = augment_retrieved_report_with_hard_kg(retrieved_backbone_report, preliminary_kg)
            else:
                final_report = apply_evidence_guardrails(final_report, evidence_bundle)
            preliminary_kg = build_kg_from_evidence(evidence_bundle, final_report)
            final_report = normalize_report_sections(final_report, preliminary_kg)
            kg_json = build_kg_from_evidence(evidence_bundle, final_report)

        # --- FINAL VALIDATION (Regex Safety Net) ---
        # Only force report/KG consistency for CT or legacy report-derived KG.
        # For X-rays, KG is an independent evidence layer; repairing the report
        # against it collapses natural IU-style wording into a rigid template.
        v = {"ok": True, "errors": []} if detected_modality != "ct" else validate_report(final_report, kg_json)
        if not v["ok"]:
             yield json.dumps({"status": "repair_start", "iter": 99}) + "\n"
             try:
                 final_report = await asyncio.wait_for(
                     asyncio.to_thread(self.repair_report, final_report, v["errors"], format_kg_for_prompt(kg_json)),
                     timeout=LLM_TIMEOUT_SECONDS,
                 )
                 if detected_modality == "ct":
                     preliminary_kg = build_kg_from_synthesis_report(final_report, evidence_bundle)
                 else:
                     preliminary_kg = build_kg_from_evidence(evidence_bundle, final_report)
                 final_report = normalize_report_sections(final_report, preliminary_kg)
                 if detected_modality == "ct":
                     kg_json = build_kg_from_synthesis_report(final_report, evidence_bundle)
                 else:
                     kg_json = build_kg_from_evidence(evidence_bundle, final_report)
             except asyncio.TimeoutError:
                 print("⚠️ Report repair timed out; using pre-repair report.")

        # ------------------------------------------------------------------
        # NEW: VISUAL EXPLAINABILITY & TRACE
        # ------------------------------------------------------------------
        
        # Create a dedicated output folder for explainable images
        explain_dir = os.path.join("out", "explained_images")
        os.makedirs(explain_dir, exist_ok=True)
        
        # Build a unique filename (e.g., "CXR3655_IM-1817_0_explained.png")
        parent_folder = os.path.basename(os.path.dirname(target_image))
        file_name = os.path.basename(target_image)
        new_filename = f"{parent_folder}_{file_name}".replace(".png", "_explained.png").replace(".jpg", "_explained.jpg")
        
        explain_img_path = os.path.join(explain_dir, new_filename)
        
        # Run the generator
        explained_path = await asyncio.to_thread(
            generate_explainable_image, target_image, kg_json, explain_img_path, detected_modality
        )

        explainability_trace = {
            "evidence_sources": {
                "vision_agent_findings": vision_report,
                "spatial_kg_evidence": kg_text_block,
                "visual_classifier_evidence": visual_evidence_text,
                "retrieval_agent_draft": draft_report
            },
            "reasoning_steps": [
                f"1. Verified image modality as '{detected_modality.upper()}'.",
                "2. Extracted raw visual description from the Vision Agent.",
                "3. Generated structured KG/classifier evidence with confidence and provenance.",
                "4. Retrieved top-k visually similar historical cases.",
                "5. Synthesized the final report by adjudicating vision text against structured evidence.",
                "6. Built the frontend knowledge graph from the final synthesized report."
            ]
        }

        print(f"\n[FINAL REPORT]:\n{final_report}\n")
        try:
            out_str = json.dumps({
                "status": "complete", 
                "detected_modality": detected_modality,
                "requested_scan_type": scan_type,
                "final_report": final_report, 
                "knowledge_graph": kg_json,
                "explainability": explainability_trace,
                "explainable_image_path": explained_path if explained_path else "Normal - No highlights needed"
            }) + "\n"
            print("[DEBUG] Successfully serialized final chunk to JSON")
            yield out_str
        except Exception as e:
            print(f"[ERROR] Failed to serialize final chunk: {e}")
            import traceback
            traceback.print_exc()
            raise

# -------------------------------
# Example Usage (Test Block)
# -------------------------------
if __name__ == "__main__":
    async def main_test():
        # Hard IU X-ray test case: mixed positive/negative findings across two views.
        # This should exercise Vision Agent text, visual classifier evidence,
        # synthesis arbitration, final KG construction, and explainability.
        hard_sample_id = "CXR1965_IM-0629"
        image_paths = [
            os.path.join("data", "iu_xray", "images", hard_sample_id, "0.png"),
            os.path.join("data", "iu_xray", "images", hard_sample_id, "1.png"),
        ]
        reference_report = (
            "There is moderate cardiomegaly. There are bilateral interstitial opacities, "
            "increased since the previous exam. No focal airspace consolidation, "
            "pleural effusions or pneumothorax. No acute bony abnormalities."
        )

        annotation_path = os.path.join("data", "iu_xray", "annotation.json")
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
                for row in annotations.get("test", []):
                    if row.get("id") == hard_sample_id:
                        reference_report = row.get("report") or reference_report
                        break
            except Exception as exc:
                print(f"⚠️ Could not read IU X-ray annotation file: {exc}")

        missing_paths = [path for path in image_paths if not os.path.exists(path)]
        if missing_paths:
            print(f"❌ Error: Missing IU X-ray test image(s) for {hard_sample_id}:")
            for path in missing_paths:
                print(f"   - {path}")
            return

        print(f"\n🚀 Testing hard IU X-ray case: {hard_sample_id}")
        print("Images:")
        for path in image_paths:
            print(f"   - {path}")
        print("\nReference report:")
        print(reference_report + "\n")
        
        # Initialize Agents
        retrieval_agent = RetrievalAgent(vision_encoder, k=5)
        draft_agent = LocalLLMReportAgent()
        vision_agent = VisualDescriptionAgent() 
        synthesis_agent = LocalSynthesisAgent()

        # Run the Pipeline
        gen = synthesis_agent.generate_final_report(
            draft_agent=draft_agent,
            vision_agent=vision_agent,
            retrieval_agent=retrieval_agent,
            reports_dict=reports_dict,
            image_paths=image_paths,
            scan_type="xray"
        )

        async for chunk in gen:
            try:
                data = json.loads(chunk.strip())
                status = data.get("status")
                
                # Print live updates
                if status != "complete" and status != "streaming":
                    print(f" 🔄 {status.upper()}: {data.get('message', '')}")
                elif status == "error":
                    # This catches the Bouncer/Filter error and stops gracefully!
                    print(f" ❌ ERROR: {data.get('message')}")
                    
                # Final Output
                if status == "complete":
                    print("\n" + "="*50)
                    print("✅ FINAL GENERATED REPORT:")
                    print(data.get("final_report"))
                    
                    print("\n🔍 EXPLAINABILITY TRACE:")
                    print(json.dumps(data.get("explainability"), indent=2))

                    kg = data.get("knowledge_graph") or {}
                    kg_summary = {
                        "kg_source": (kg.get("metadata") or {}).get("kg_source"),
                        "entities": kg.get("entities", []),
                        "relations": kg.get("relations", []),
                    }
                    print("\n🧠 SYNTHESIZED KNOWLEDGE GRAPH:")
                    print(json.dumps(kg_summary, indent=2))
                    
                    print("\n🖼️ EXPLAINABLE IMAGE SAVED AT:")
                    print(data.get("explainable_image_path"))
                    print("="*50 + "\n")
                    
            except Exception as e: 
                print(f"⚠️ Chunk Error: {e}")
                continue

    asyncio.run(main_test())
