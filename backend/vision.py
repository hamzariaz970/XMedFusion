import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import warnings
import re
import json
import sys
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer  # Use explicit HF Tokenizer
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

# IMPORT OPEN_CLIP
import open_clip

import config

REPORT_CONTEXT_WINDOW = min(config.CONTEXT_WINDOW, 8192)
LLM_TIMEOUT_SECONDS = 180

# Suppress warnings
warnings.filterwarnings("ignore")

# -------------------------------
# CONFIGURATION
# -------------------------------
TRAINED_HEAD_PATH = "model_weights/KG_Agent/biomed_clip/biomed_clip_head_best.pth"

DISEASES = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumothorax", 
    "Infiltrate", "Consolidation", "Lung Opacity", "Nodule", 
    "Atelectasis", "Fracture"
]

# Earlier BioMedCLIP checkpoints in this project were trained against the
# 14-label CheXpert-style target set saved alongside thresholds.json.
LEGACY_CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding"
]

CLASSIFIER_LABEL_ALIASES = {
    "Cardiomegaly": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
    "Pleural Effusion": ["Pleural Effusion", "Pleural Other"],
    "Infiltrate": ["Infiltrate", "Pneumonia"],
    "Nodule": ["Nodule", "Lung Lesion"],
}

# Zero-Shot Prompts (The "Verifier")
PATHOLOGY_CONFIG = {
    "Cardiomegaly":     {"pos": "enlarged heart cardiomegaly", "neg": "normal heart size"},
    "Pleural Effusion": {"pos": "pleural effusion fluid",      "neg": "no pleural effusion"},
    "Edema":            {"pos": "pulmonary edema",             "neg": "no pulmonary edema"},
    "Pneumothorax":     {"pos": "pneumothorax air",            "neg": "no pneumothorax"},
    "Infiltrate":       {"pos": "lung infiltrate pneumonia",   "neg": "no infiltrates"},
    "Consolidation":    {"pos": "lung consolidation",          "neg": "no consolidation"},
    "Lung Opacity":     {"pos": "lung opacity",                "neg": "clear lungs"},
    "Nodule":           {"pos": "lung nodule mass",            "neg": "no nodules"},
    "Atelectasis":      {"pos": "lung atelectasis",            "neg": "no atelectasis"},
    "Fracture":         {"pos": "bone fracture broken rib",    "neg": "normal ribs no fractures intact bones"} 
}

# -------------------------------
# Device Configuration
# -------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# -------------------------------
# Model Definitions
# -------------------------------

class VisionEncoder(nn.Module):
    """
    Standard BioMedCLIP Vision Tower + Text Encoder (for Zero-Shot)
    """
    def __init__(self, device="cuda", model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        super().__init__()
        self.device = device
        print(f"Loading BioMedCLIP Vision Encoder: {model_name}...")
        
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=device)
            self.model = model
            self.preprocess = preprocess
            
            # FIX: Use Hugging Face AutoTokenizer directly (Robust)
            hf_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            
        except Exception as e:
            print(f"Error loading OpenCLIP model: {e}")
            raise e

        # Freeze
        for p in self.model.parameters():
            p.requires_grad = False
            
    def encode_image(self, image):
        """Preprocess and Encode Image"""
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, text_list):
        """Encode Text Prompts using HF Tokenizer"""
        # Explicitly tokenize with padding/truncation to 256 (BioMedCLIP default)
        tokens = self.tokenizer(
            text_list, 
            padding="max_length", 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        )["input_ids"].to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
        return features

class BioMedCLIPClassifierHead(nn.Module):
    """Deeper classifier head with residual connection. Must match training script."""
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.residual = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.head(features) + self.residual(features)

class LegacyBioMedCLIPClassifierHead(nn.Module):
    """Older single-linear-layer head used by earlier saved checkpoints."""
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.head(features)

def _infer_classifier_output_count(state_dict):
    for key in ("head.weight", "head.8.weight", "residual.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return len(DISEASES)

def _labels_for_output_count(output_count):
    if output_count == len(DISEASES):
        return DISEASES
    if output_count == len(LEGACY_CHEXPERT_LABELS):
        return LEGACY_CHEXPERT_LABELS
    return [f"class_{idx}" for idx in range(output_count)]

def build_classifier_for_state_dict(state_dict):
    output_count = _infer_classifier_output_count(state_dict)
    if "head.weight" in state_dict and "head.0.weight" not in state_dict:
        head = LegacyBioMedCLIPClassifierHead(num_classes=output_count).to(device)
    else:
        head = BioMedCLIPClassifierHead(num_classes=output_count).to(device)
    head.label_names = _labels_for_output_count(output_count)
    return head

def _classifier_probs_by_disease(raw_probs, label_names):
    raw_probs = np.atleast_1d(raw_probs)
    label_scores = {
        label: float(raw_probs[idx])
        for idx, label in enumerate(label_names[:len(raw_probs)])
    }

    disease_scores = {}
    for disease in DISEASES:
        aliases = CLASSIFIER_LABEL_ALIASES.get(disease, [disease])
        alias_scores = [label_scores[label] for label in aliases if label in label_scores]
        disease_scores[disease] = max(alias_scores) if alias_scores else 0.0
    return disease_scores

def _threshold_for_disease(disease):
    aliases = CLASSIFIER_LABEL_ALIASES.get(disease, [disease])
    for label in aliases:
        if label in LEARNED_THRESHOLDS:
            return LEARNED_THRESHOLDS[label]
    if disease == "Fracture":
        return 0.50
    if disease == "Nodule":
        return 0.45
    return 0.40

# -------------------------------
# GLOBAL INITIALIZATION
# -------------------------------

# 1. Load Vision Encoder
try:
    vision_encoder = VisionEncoder(device=device).to(device)
    vision_encoder.eval()
except Exception as e:
    print(f"CRITICAL ERROR loading vision model: {e}")
    sys.exit(1)

# 2. Load Trained Classifier Head
classifier = BioMedCLIPClassifierHead(num_classes=len(DISEASES)).to(device)
classifier.label_names = DISEASES
if os.path.exists(TRAINED_HEAD_PATH):
    try:
        state_dict = torch.load(TRAINED_HEAD_PATH, map_location=device)
        classifier = build_classifier_for_state_dict(state_dict)
        classifier.load_state_dict(state_dict)
        print(f"✅ Loaded Trained Classifier from {TRAINED_HEAD_PATH}")
    except Exception as e:
        print(f"⚠️ Error loading classifier weights: {e}")
else:
    print(f"⚠️ Classifier weights not found at {TRAINED_HEAD_PATH}")
classifier.eval()

def reload_classifier_head():
    """Hot-reload classifier weights from disk after HIL fine-tuning."""
    global classifier
    if os.path.exists(TRAINED_HEAD_PATH):
        try:
            state_dict = torch.load(TRAINED_HEAD_PATH, map_location=device)
            classifier = build_classifier_for_state_dict(state_dict)
            classifier.load_state_dict(state_dict)
            classifier.eval()
            print(f"🔄 Hot-reloaded classifier from {TRAINED_HEAD_PATH}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to hot-reload classifier: {e}")
            return False
    print(f"⚠️ Cannot reload: weights not found at {TRAINED_HEAD_PATH}")
    return False

# 2b. Load fine-tuned backbone weights if available
TRAINED_BACKBONE_PATH = "model_weights/KG_Agent/biomed_clip/biomed_clip_backbone_finetuned.pth"
if os.path.exists(TRAINED_BACKBONE_PATH):
    try:
        unfrozen_state = torch.load(TRAINED_BACKBONE_PATH, map_location=device)
        model_state = vision_encoder.model.state_dict()
        updated = 0
        for name, value in unfrozen_state.items():
            if name in model_state:
                model_state[name] = value
                updated += 1
        if updated > 0:
            vision_encoder.model.load_state_dict(model_state)
            print(f"✅ Loaded fine-tuned backbone ({updated} params) from {TRAINED_BACKBONE_PATH}")
    except:
        pass

# 2c. Load learned thresholds
THRESHOLDS_PATH = "model_weights/KG_Agent/biomed_clip/thresholds.json"
LEARNED_THRESHOLDS = {}
if os.path.exists(THRESHOLDS_PATH):
    try:
        with open(THRESHOLDS_PATH, 'r') as f:
            LEARNED_THRESHOLDS = json.load(f)
        print(f"✅ Loaded learned thresholds from {THRESHOLDS_PATH}")
    except:
        pass

# 3. Pre-compute Text Embeddings (Zero-Shot)
disease_text_features = {}
print("Pre-computing Zero-Shot Embeddings...")
with torch.no_grad():
    for disease, prompts in PATHOLOGY_CONFIG.items():
        pos_txt = f"chest x-ray showing {prompts['pos']}"
        neg_txt = f"chest x-ray showing {prompts['neg']}"
        # Now uses our robust encode_text method
        embeds = vision_encoder.encode_text([pos_txt, neg_txt])
        disease_text_features[disease] = embeds

# -------------------------------
# Logic: Hybrid Scoring
# -------------------------------
def get_hybrid_findings(img_embedding):
    """
    Combines Trained Head (Global) + Zero-Shot (Global) to suppress hallucinations.
    Now uses learned thresholds and 50/50 weighting.
    """
    findings = {}
    
    # A. Get Trained Head Probs
    with torch.no_grad():
        logits = classifier(img_embedding)
        trained_probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        label_names = getattr(classifier, "label_names", _labels_for_output_count(np.atleast_1d(trained_probs).shape[0]))
        trained_map = _classifier_probs_by_disease(trained_probs, label_names)

    # B. Get Zero-Shot Probs & Combine
    for disease, text_embeds in disease_text_features.items():
        # 1. Zero-Shot Score
        sim = (100.0 * img_embedding @ text_embeds.T).softmax(dim=-1).squeeze()
        zs_prob = sim[0].item()
        
        # 2. Get Trained Score
        tr_prob = trained_map.get(disease, 0.0)
        
        # 3. IMPROVED: Equal hybrid weighting (trained model is better now)
        hybrid_score = (zs_prob * 0.5) + (tr_prob * 0.5)
        
        # 4. Use learned thresholds (with fallbacks)
        threshold = _threshold_for_disease(disease)
        
        findings[disease] = hybrid_score if hybrid_score > threshold else 0.0
        
    return findings

# -------------------------------
# Logic: Scan Validation (Zero-Shot)
# -------------------------------
def is_medical_scan(image, vision_encoder, threshold=0.1):
    """
    Uses BioMedCLIP zero-shot to check if image is a medical scan vs random object.
    """
    # Define labels for classification
    labels = [
        "a medical x-ray or ct scan", 
        "a photograph of a person", 
        "a photograph of an animal", 
        "a photograph of a cat",
        "a picture of furniture or household objects",
        "a landscape or nature photograph",
        "text or a document",
        "a digital illustration or icon"
    ]
    
    # Pre-compute text embeddings for these labels
    with torch.no_grad():
        # Using the robust encode_text with the HF tokenizer
        text_features = vision_encoder.encode_text(labels)
        
        # Encode image
        img_features = vision_encoder.encode_image(image)
        
        # Get similarities
        # Calculate cosine similarity (100.0 is the temperature used in CLIP)
        logits = (100.0 * img_features @ text_features.T).softmax(dim=-1).squeeze()
        
    probs = logits.cpu().numpy()
    
    # Index 0 is our "medical scan" label
    medical_prob = probs[0]
    
    # Log for debugging
    print(f"[SCAN VALIDATOR] Medical Scan Probability: {medical_prob:.4f}")
    for i, label in enumerate(labels):
        print(f"   - {label}: {probs[i]:.4f}")
        
    # Check if index 0 has the highest probability OR is above a safe threshold
    # Sometimes complex scans might have slightly lower confidence than 'clear' categories
    is_top = np.argmax(probs) == 0
    return is_top or medical_prob > threshold, medical_prob, labels[np.argmax(probs)]

# -------------------------------
# Local LLM Agent
# -------------------------------
class VisualDescriptionAgent:
    def __init__(self, model_name="MedAIBase/MedGemma1.5:4b"): 
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        self.llm = ChatOllama(
            model=model_name,
            temperature=config.TEMPERATURE,
            num_ctx=REPORT_CONTEXT_WINDOW,
            num_predict=160,
            timeout=LLM_TIMEOUT_SECONDS,
        )

    def generate_description(self, findings_dict):
        # Sort by confidence
        active_findings = [(k, v) for k, v in findings_dict.items() if v > 0]
        active_findings.sort(key=lambda x: x[1], reverse=True)
        
        if not active_findings:
            prompt = (
                "You are an expert radiologist. Describe a 'Normal Chest X-Ray'. "
                "State that the lungs are clear, heart size is normal, and no acute abnormalities are seen. "
                "Keep it concise (1 sentence)."
            )
        else:
            desc_list = []
            for k, v in active_findings:
                desc_list.append(f"{k}")
            
            input_text = ", ".join(desc_list)
            prompt = (
                "You are an expert radiologist. Write a concise Visual Description (1-2 sentences) of a Chest X-Ray based on these findings.\n"
                f"FINDINGS: {input_text}\n"
                "OUTPUT (Visual Description only):"
            )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return content.strip()

# -------------------------------
# TEST BLOCK
# -------------------------------
if __name__ == "__main__":
    print("\n--- Running Vision Agent Test (Hybrid Batch) ---")
    
    ANNO_PATH = "data/iu_xray/annotation.json"
    IMG_ROOT = "data/iu_xray/images"

    if not os.path.exists(ANNO_PATH):
        print(f"❌ Annotation file not found at {ANNO_PATH}")
        sys.exit(1)

    with open(ANNO_PATH, 'r') as f:
        data = json.load(f)

    test_samples = data.get("test", [])
    if not test_samples:
        test_samples = [x for x in data if x.get("split") == "test"]
    if not test_samples:
        test_samples = data.get("val", [])

    samples_to_run = test_samples[:20]
    print(f"Loaded {len(samples_to_run)} samples.\n")
    
    writer = VisualDescriptionAgent()

    for i, item in enumerate(samples_to_run):
        uid = item.get("id", "Unknown")
        gt_report = item.get("report", "No Report")
        
        img_list = item.get("image_path", [])
        if not img_list: continue
        img_rel_path = img_list[0] if isinstance(img_list, list) else img_list
        full_img_path = os.path.join(IMG_ROOT, img_rel_path)
        
        print(f"📸 [{i+1}/20] {uid}")
        
        if os.path.exists(full_img_path):
            try:
                raw_img = Image.open(full_img_path).convert("RGB")
                img_embedding = vision_encoder.encode_image(raw_img)
                
                findings = get_hybrid_findings(img_embedding)
                ai_description = writer.generate_description(findings)
                
                print(f"   👁️  AI Description:\n   {ai_description}")
                print(f"\n   📄 GT Report:\n   {gt_report}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            
        else:
            print(f"   ⚠️ Image not found: {full_img_path}")
        
        print("-" * 50)
