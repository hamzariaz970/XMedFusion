from __future__ import annotations

import os
import json
import re
import sys
import gc
import time
from pathlib import Path
from typing import List, Tuple, Literal, Optional, Union, Dict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import open_clip
import config

# -------------------------
# CONFIGURATION
# -------------------------
TRAINED_HEAD_PATH = "model_weights/KG_Agent/biomed_clip/biomed_clip_head_best.pth"

DISEASES = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumothorax", 
    "Infiltrate", "Consolidation", "Lung Opacity", "Nodule", 
    "Atelectasis", "Fracture"
]

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

# -------------------------
# 1. TRAINED CLASSIFIER ARCHITECTURE
# -------------------------
class BioMedCLIPClassifierHead(nn.Module):
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

# -------------------------
# 2. HELPER FUNCTIONS
# -------------------------
def get_spatial_crops(image: Image.Image) -> Dict[str, Image.Image]:
    """
    FIX: Adjusted Left Lung crop start from 0.55 -> 0.62 to exclude heart shadow.
    """
    w, h = image.size
    return {
        "Right Lung": image.crop((0, 0, int(w * 0.45), h)),
        "Mediastinum": image.crop((int(w * 0.30), 0, int(w * 0.70), h)),
        # Start at 0.62 to avoid Heart Shadow confusion
        "Left Lung": image.crop((int(w * 0.62), 0, w, h)) 
    }

def get_global_probs(image, model, classifier_head, preprocess, device):
    if classifier_head is None:
        return {}
        
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features = F.normalize(features, dim=-1)
        logits = classifier_head(features)
        probs = torch.sigmoid(logits).squeeze()
        
    return {disease: probs[idx].item() for idx, disease in enumerate(DISEASES)}

def hybrid_classify_spatial(image_path, model, classifier_head, preprocess, tokenizer, device):
    try:
        raw_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error: {e}"

    findings_list = []
    
    # Global Priors
    global_probs = get_global_probs(raw_img, model, classifier_head, preprocess, device)
    
    # Zero-Shot Embeddings
    crops = get_spatial_crops(raw_img)
    disease_embeds = {}
    with torch.no_grad():
        norm_tokens = tokenizer(["normal chest x-ray", "abnormal chest x-ray"], padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"].to(device)
        norm_embeds = F.normalize(model.encode_text(norm_tokens), dim=-1)

        for d, cfg in PATHOLOGY_CONFIG.items():
            tokens = tokenizer([f"chest x-ray showing {cfg['pos']}", f"chest x-ray showing {cfg['neg']}"], padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"].to(device)
            disease_embeds[d] = F.normalize(model.encode_text(tokens), dim=-1)

    # Scan Zones
    for zone_name, zone_img in crops.items():
        img_tensor = preprocess(zone_img).unsqueeze(0).to(device)
        zone_findings = []

        with torch.no_grad():
            img_features = F.normalize(model.encode_image(img_tensor), dim=-1)
            norm_logits = (100.0 * img_features @ norm_embeds.T).softmax(dim=-1).squeeze()
            prob_zone_normal = norm_logits[0].item()

            for disease, embeds in disease_embeds.items():
                if zone_name == "Mediastinum" and disease not in ["Cardiomegaly", "Nodule"]: continue
                if zone_name in ["Right Lung", "Left Lung"] and disease == "Cardiomegaly": continue

                # Zero-Shot & Trained Probs
                logits = (100.0 * img_features @ embeds.T).softmax(dim=-1).squeeze()
                zs_prob = logits[0].item()
                trained_prob = global_probs.get(disease, zs_prob)
                
                # --- LOGIC FIX: GLOBAL VETO ---
                # If the Trained Model is confident the disease is ABSENT (< 0.15),
                # we suppress it regardless of what the visual scanner says.
                if trained_prob < 0.15:
                    continue

                # Hybrid Score
                final_score = (zs_prob * 0.6) + (trained_prob * 0.4)
                
                # --- TUNED THRESHOLDS ---
                # Lowered from 0.45 to 0.40 to catch mild pathologies (CXR2704)
                threshold = 0.40
                
                # Keep stricter thresholds for "noisy" diseases
                if disease == "Fracture": threshold = 0.60
                if disease == "Nodule": threshold = 0.55
                
                # Special Logic: If Zero-Shot is VERY confident (>0.85), trust it even if global is lower
                if zs_prob > 0.85: final_score = max(final_score, 0.6)

                if final_score > threshold and zs_prob > (prob_zone_normal - 0.05):
                     zone_findings.append(disease)

        if zone_findings:
            findings_list.append(f"{zone_name}: {', '.join(zone_findings)}")
        else:
            findings_list.append(f"{zone_name}: Clear")
            
    return "; ".join(findings_list)

# -------------------------
# 3. GRAPH BUILDER
# -------------------------
def parse_findings_to_kg(findings_str: str) -> dict:
    entities = []
    relations = []
    zones = findings_str.split(";")
    
    for zone_part in zones:
        if ":" not in zone_part: continue
        zone_name, findings_text = zone_part.split(":", 1)
        zone_name = zone_name.strip().lower()
        findings_text = findings_text.strip()
        
        anat_idx = len(entities)
        entities.append([zone_name, "Anatomy"])
        
        if not findings_text: continue
        observations = [f.strip().lower() for f in findings_text.split(",")]
        
        for obs in observations:
            obs_idx = len(entities)
            entities.append([obs, "Observation"])
            if obs in ["normal", "clear"]:
                relations.append([obs_idx, anat_idx, "modify"])
            else:
                relations.append([obs_idx, anat_idx, "located_at"])
                
    return {"entities": entities, "relations": relations}

# -------------------------
# 4. MAIN INFERENCE
# -------------------------
def infer_kg(
    image: Union[str, Path],
    projection: str = "Frontal",
    *,
    clip_model=None, 
    clip_prep=None,
    tokenizer=None,
    classifier_head=None, 
    device=None,
    model: str = None, 
    debug: bool = True,
) -> dict:
    
    if debug: print("[KG] infer_kg called...", flush=True)
    image_path = Path(image).expanduser()
    if not image_path.exists(): raise FileNotFoundError(f"Image not found: {image_path}")
    
    owns_models = False

    # 1. Setup BioMedCLIP
    if clip_model is None or clip_prep is None or tokenizer is None:
        owns_models = True
        if debug: print("[KG] Loading BioMedCLIP...", flush=True)
        d = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        try:
            clip_model, _, clip_prep = open_clip.create_model_and_transforms(model_name, device=d)
            hf_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            clip_model.eval()
        except Exception as e:
            print(f"[KG] ❌ Error: {e}")
            return {"entities": [], "relations": []}
        device = d

    # 2. Setup Classifier Head
    if classifier_head is None:
        if os.path.exists(TRAINED_HEAD_PATH):
            try:
                classifier_head = BioMedCLIPClassifierHead(num_classes=len(DISEASES)).to(device)
                state_dict = torch.load(TRAINED_HEAD_PATH, map_location=device)
                classifier_head.head.load_state_dict(state_dict)
                classifier_head.eval()
            except:
                classifier_head = None
    
    # 3. Extract Findings
    if debug: print("[KG] Scanning...", flush=True)
    visual_findings = hybrid_classify_spatial(
        image_path, clip_model, classifier_head, clip_prep, tokenizer, device
    )
    if debug: print(f"[KG] Findings: {visual_findings}", flush=True)

    # 4. Construct Graph
    if debug: print("[KG] Constructing Graph...", flush=True)
    kg = parse_findings_to_kg(visual_findings)
    
    # 5. Cleanup
    if owns_models and device == "cuda":
        if debug: print("[KG] Cleaning up models...", flush=True)
        clip_model.to("cpu")
        if classifier_head: classifier_head.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

    if debug: print("[KG] Success.", flush=True)
    return kg

# -------------------------
# Test Block
# -------------------------
if __name__ == "__main__":
    print("\n--- Running KG Agent Test Batch ---")
    
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
    
    # Check top 30
    samples_to_run = test_samples[:30]
    
    # Init Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Models on {device}...")
    
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(model_name, device=device)
    hf_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    clip_model.eval()
    
    classifier_head = None
    if os.path.exists(TRAINED_HEAD_PATH):
        try:
            classifier_head = BioMedCLIPClassifierHead(num_classes=len(DISEASES)).to(device)
            state_dict = torch.load(TRAINED_HEAD_PATH, map_location=device)
            classifier_head.head.load_state_dict(state_dict)
            classifier_head.eval()
            print("✅ Trained Head Loaded.")
        except:
            print("⚠️ Could not load trained head.")

    print("-" * 60)

    for i, item in enumerate(samples_to_run):
        uid = item.get("id", "Unknown")
        gt_report = item.get("report", "No Report")
        
        img_list = item.get("image_path", [])
        if not img_list: continue
        img_rel_path = img_list[0] if isinstance(img_list, list) else img_list
        full_img_path = os.path.join(IMG_ROOT, img_rel_path)
        
        if not os.path.exists(full_img_path):
            continue

        print(f"\n📸 Processing {i+1}/30: {uid}")
        
        try:
            kg_result = infer_kg(
                full_img_path, 
                clip_model=clip_model,
                clip_prep=clip_prep,
                tokenizer=tokenizer,
                classifier_head=classifier_head,
                device=device,
                debug=False 
            )
            
            print(f"📄 GT Report: {gt_report}")
            
            entities = kg_result.get("entities", [])
            relations = kg_result.get("relations", [])
            grouped = {}
            for r in relations:
                if r[0] >= len(entities) or r[1] >= len(entities): continue
                if r[2] in ["located_at", "modify"]:
                    obs = entities[r[0]][0] 
                    anat = entities[r[1]][0] 
                    if anat not in grouped: grouped[anat] = []
                    grouped[anat].append(obs)

            print("🤖 AI Graph Findings:")
            if not grouped:
                print("   (No findings detected)")
            for anat, obs_list in grouped.items():
                print(f"   • {anat.title()}: {', '.join(obs_list)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 60)