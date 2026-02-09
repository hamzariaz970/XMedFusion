import os
import json
import torch
import sys
from tqdm import tqdm

# Import the KG Inference function
from kg_agent import infer_kg, BioMedCLIPClassifierHead, TRAINED_HEAD_PATH, DISEASES
import open_clip
from transformers import AutoTokenizer

# -------------------------------
# CONFIGURATION
# -------------------------------
ANNO_PATH = "data/iu_xray/annotation.json"
IMG_ROOT = "data/iu_xray/images"

# Diseases to stress test
TARGET_PATHOLOGIES = [
    "cardiomegaly", 
    "pleural effusion", 
    "opacity", 
    "edema", 
    "atelectasis",
    "pneumothorax",
    "fracture"
]

# Negation handling to ensure we pick TRULY sick patients
NEGATION_PHRASES = [
    "no ", "not ", "without ", "free of ", "negative for ", "absence of ", 
    "unremarkable", "normal", "clear", "resolved"
]

def is_actually_sick(report, disease):
    """
    Returns True only if the disease is present and NOT negated.
    """
    report = report.lower()
    disease = disease.lower()
    
    if disease not in report: return False
    
    start_idx = report.find(disease)
    context = report[max(0, start_idx - 30): start_idx]
    
    for neg in NEGATION_PHRASES:
        if neg in context: return False
            
    return True

def find_sick_patients(anno_path, pathology_list, samples_per_disease=3):
    """
    Finds verified sick patients for each disease.
    """
    if not os.path.exists(anno_path):
        print(f"❌ Annotation file not found: {anno_path}")
        return {}

    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    all_samples = []
    if isinstance(data, dict):
        for split in data:
            all_samples.extend(data[split])
    else:
        all_samples = data

    found_samples = {}
    print(f"🔍 Scanning {len(all_samples)} reports for graph validation...")
    
    for pathology in pathology_list:
        found_samples[pathology] = []
        for item in all_samples:
            report = item.get("report", "").lower()
            if is_actually_sick(report, pathology):
                found_samples[pathology].append(item)
                if len(found_samples[pathology]) >= samples_per_disease:
                    break
    return found_samples

def check_graph_for_disease(kg, disease_name):
    """
    Searches the KG nodes for the specific disease.
    """
    entities = kg.get("entities", [])
    
    # 1. Direct Match
    for entity in entities:
        node_name, node_type = entity
        if disease_name.lower() in node_name.lower():
            return True, node_name
            
    # 2. Synonym Match (Opacity coverage)
    if disease_name == "opacity":
        for entity in entities:
            n = entity[0].lower()
            if any(x in n for x in ["infiltrate", "consolidation", "pneumonia", "opacity"]):
                return True, n
                
    return False, None

def run_stress_test():
    print("\n--- 🕸️ Running KG Agent Stress Test 🕸️ ---")
    
    # 1. Load Models ONCE to save time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Loading models on {device}...")
    
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(model_name, device=device)
    hf_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    clip_model.eval()
    
    # Load Trained Head explicitly
    classifier_head = BioMedCLIPClassifierHead(num_classes=len(DISEASES)).to(device)
    if os.path.exists(TRAINED_HEAD_PATH):
        classifier_head.head.load_state_dict(torch.load(TRAINED_HEAD_PATH, map_location=device))
        classifier_head.eval()
        print("   ✅ Trained Head Loaded.")
    else:
        print("   ⚠️ Trained Head NOT FOUND (Results may be poor).")

    # 2. Find Patients
    sick_cases = find_sick_patients(ANNO_PATH, TARGET_PATHOLOGIES)
    print("-" * 60)

    # 3. Run Tests
    score_card = {"passes": 0, "fails": 0}
    
    for disease, samples in sick_cases.items():
        print(f"\n👉 Target: {disease.upper()}")
        
        for item in samples:
            uid = item.get("id", "Unknown")
            gt_report = item.get("report", "No Report")[:100] + "..."
            
            img_list = item.get("image_path", [])
            if not img_list: continue
            img_rel_path = img_list[0] if isinstance(img_list, list) else img_list
            full_img_path = os.path.join(IMG_ROOT, img_rel_path)
            
            if not os.path.exists(full_img_path): continue
            
            # --- RUN KG AGENT ---
            try:
                kg = infer_kg(
                    full_img_path,
                    clip_model=clip_model,
                    clip_prep=clip_prep,
                    tokenizer=tokenizer,
                    classifier_head=classifier_head,
                    device=device,
                    debug=False
                )
                
                # --- VERIFY GRAPH ---
                found, node_name = check_graph_for_disease(kg, disease)
                
                if found:
                    print(f"   ✅ [PASS] {uid}: Graph contains '{node_name}'")
                    score_card["passes"] += 1
                else:
                    print(f"   ❌ [FAIL] {uid}: Disease missing from graph.")
                    print(f"      GT: {gt_report}")
                    # Print what WAS found to debug
                    observations = [e[0] for e in kg['entities'] if e[1] == 'Observation']
                    print(f"      Graph found: {observations}")
                    score_card["fails"] += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {uid}: {e}")

    print("\n" + "="*40)
    print(f"📊 FINAL SCORE: {score_card['passes']} Passes / {score_card['fails']} Fails")
    print("="*40)

if __name__ == "__main__":
    run_stress_test()