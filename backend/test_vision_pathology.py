import os
import json
import torch
from PIL import Image

# Import from vision.py
from vision import (
    vision_encoder, 
    classifier, 
    get_hybrid_findings, 
    VisualDescriptionAgent, 
    DISEASES
)

# -------------------------------
# CONFIGURATION
# -------------------------------
ANNO_PATH = "data/iu_xray/annotation.json"
IMG_ROOT = "data/iu_xray/images"

TARGET_PATHOLOGIES = [
    "cardiomegaly", 
    "pleural effusion", 
    "opacity", 
    "edema", 
    "atelectasis"
]

# List of phrases that indicate the disease is NOT present
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
    
    # 1. Check if word exists
    if disease not in report:
        return False
    
    # 2. Check context (Robust Negation)
    start_idx = report.find(disease)
    # Look at the 30 characters before the disease word
    context = report[max(0, start_idx - 30): start_idx]
    
    for neg in NEGATION_PHRASES:
        if neg in context:
            return False
            
    return True

def find_sick_patients(anno_path, pathology_list, samples_per_disease=3):
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
    print(f"🔍 Scanning {len(all_samples)} reports for TRUE pathologies...")
    
    for pathology in pathology_list:
        found_samples[pathology] = []
        for item in all_samples:
            report = item.get("report", "").lower()
            
            if is_actually_sick(report, pathology):
                found_samples[pathology].append(item)
                if len(found_samples[pathology]) >= samples_per_disease:
                    break
    return found_samples

def run_test():
    print("\n--- 🏥 Running Pathology Stress Test (Strict GT) 🏥 ---")
    
    writer = VisualDescriptionAgent()
    sick_cases = find_sick_patients(ANNO_PATH, TARGET_PATHOLOGIES)
    
    total_found = sum(len(v) for v in sick_cases.values())
    print(f"✅ Found {total_found} confirmed sick cases.\n")

    for disease, samples in sick_cases.items():
        print(f"\n👉 Testing for: {disease.upper()}")
        print("=" * 60)
        
        for item in samples:
            uid = item.get("id", "Unknown")
            gt_report = item.get("report", "No Report")
            img_list = item.get("image_path", [])
            
            if not img_list: continue
            img_rel_path = img_list[0] if isinstance(img_list, list) else img_list
            full_img_path = os.path.join(IMG_ROOT, img_rel_path)
            
            if not os.path.exists(full_img_path):
                continue

            print(f"📸 Image: {uid}")
            
            try:
                # Run Vision Agent
                raw_img = Image.open(full_img_path).convert("RGB")
                with torch.no_grad():
                    embedding = vision_encoder.encode_image(raw_img)
                
                findings = get_hybrid_findings(embedding)
                
                # Filter for display (Lower threshold just to see what's happening)
                active_findings = {k: v for k, v in findings.items() if v > 0.35}
                
                description = writer.generate_description(findings)
                formatted_findings = {k: round(v, 2) for k, v in active_findings.items()}
                
                print(f"   📊 Signals: {json.dumps(formatted_findings, indent=None)}")
                print(f"   👁️  AI Desc: {description}")
                print(f"   📄 GT Report: {gt_report[:150]}...") 
                
                # Check Success
                hit = False
                if any(disease.lower() in k.lower() for k in active_findings.keys()):
                    hit = True
                    print(f"   ✅ SUCCESS: {disease} Detected.")
                elif disease == "opacity" and any(x in active_findings for x in ["Infiltrate", "Consolidation", "Lung Opacity", "Pneumonia"]):
                     hit = True
                     print("   ✅ SUCCESS: Opacity subtype detected.")
                elif disease == "edema" and "Edema" in active_findings:
                     hit = True
                     print("   ✅ SUCCESS: Edema detected.")
                elif disease == "atelectasis" and "Atelectasis" in active_findings:
                     hit = True
                     print("   ✅ SUCCESS: Atelectasis detected.")
                else:
                     print(f"   ⚠️ MISS: {disease} not found (Highest relevant: {max(findings.values()):.2f})")

            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print("-" * 60)

if __name__ == "__main__":
    run_test()