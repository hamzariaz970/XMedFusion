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
    """
    The trained head that detects diseases.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, features):
        return self.head(features)

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
if os.path.exists(TRAINED_HEAD_PATH):
    try:
        state_dict = torch.load(TRAINED_HEAD_PATH, map_location=device)
        classifier.head.load_state_dict(state_dict)
        classifier.eval()
        print(f"✅ Loaded Trained Classifier from {TRAINED_HEAD_PATH}")
    except Exception as e:
        print(f"⚠️ Error loading classifier weights: {e}")
else:
    print(f"⚠️ Classifier weights not found at {TRAINED_HEAD_PATH}")

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
    """
    findings = {}
    
    # A. Get Trained Head Probs
    with torch.no_grad():
        logits = classifier(img_embedding)
        trained_probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        trained_map = {d: p for d, p in zip(DISEASES, trained_probs)}

    # B. Get Zero-Shot Probs & Combine
    for disease, text_embeds in disease_text_features.items():
        # 1. Zero-Shot Score
        sim = (100.0 * img_embedding @ text_embeds.T).softmax(dim=-1).squeeze()
        zs_prob = sim[0].item()
        
        # 2. Get Trained Score
        tr_prob = trained_map.get(disease, 0.0)
        
        # 3. Hybrid Average (60% Zero-Shot, 40% Trained)
        hybrid_score = (zs_prob * 0.6) + (tr_prob * 0.4)
        
        # 4. TUNED THRESHOLDS (Lowered for better sensitivity)
        # Was 0.50, now 0.40 to catch "Mild" cases
        threshold = 0.40
        if disease == "Fracture": threshold = 0.60 # Keep high
        if disease == "Nodule": threshold = 0.50
        
        findings[disease] = hybrid_score if hybrid_score > threshold else 0.0
        
    return findings

# -------------------------------
# Local LLM Agent
# -------------------------------
class VisualDescriptionAgent:
    def __init__(self, model_name="MedAIBase/MedGemma1.5:4b"): 
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        self.llm = ChatOllama(model=model_name, temperature=0.1)

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