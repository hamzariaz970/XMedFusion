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
    "no ", "no evidence", "without ", "negative for", "free of",
    "absent", "not ", "denies ", "ruled out",
    "no definite", "no definitive", "no visible", "no acute", "no focal",
    "no large", "no obvious", "no significant", "no suspicious",
    "no displaced", "nondisplaced",
]

def is_actually_sick(report, disease):
    """
    Returns True only if the disease is present and NOT negated.
    Uses sentence-boundary-aware negation with 120-char context window.
    """
    report_lower = report.lower()
    disease_lower = disease.lower()
    
    if disease_lower not in report_lower:
        return False
    
    # Check ALL occurrences
    start = 0
    while True:
        pos = report_lower.find(disease_lower, start)
        if pos == -1:
            return False  # All occurrences were negated
        
        # Get context window (120 chars to cover long comma-separated lists)
        context_start = max(0, pos - 120)
        context = report_lower[context_start:pos]
        
        # Respect sentence boundaries
        last_period = max(context.rfind('.'), context.rfind('!'), context.rfind('?'))
        if last_period >= 0:
            context = context[last_period + 1:]
        
        negated = any(neg in context for neg in NEGATION_PHRASES)
        
        if not negated:
            return True  # Found a non-negated occurrence
        
        start = pos + len(disease_lower)

def find_sick_patients(anno_path, pathology_list, samples_per_disease=5):
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
                # Verify image exists
                img_list = item.get("image_path", [])
                if not img_list: continue
                img_rel = img_list[0] if isinstance(img_list, list) else img_list
                if not os.path.exists(os.path.join(IMG_ROOT, img_rel)): continue
                
                found_samples[pathology].append(item)
                if len(found_samples[pathology]) >= samples_per_disease:
                    break
        
        # Show what was selected
        count = len(found_samples[pathology])
        if count == 0:
            print(f"  ⚠️  {pathology.upper()}: No verified positive cases found!")
        else:
            print(f"  ✅ {pathology.upper()}: Found {count} verified cases")
    
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
        classifier_head.load_state_dict(torch.load(TRAINED_HEAD_PATH, map_location=device))
        classifier_head.eval()
        print("   ✅ Trained Head Loaded.")
    else:
        print("   ⚠️ Trained Head NOT FOUND (Results may be poor).")

    # 2. Find Patients (5 per disease for more robust testing)
    sick_cases = find_sick_patients(ANNO_PATH, TARGET_PATHOLOGIES, samples_per_disease=5)
    print("-" * 60)

    # 3. Run Tests
    score_card = {"passes": 0, "fails": 0}
    per_disease_scores = {}
    
    for disease, samples in sick_cases.items():
        print(f"\n👉 Target: {disease.upper()} ({len(samples)} cases)")
        per_disease_scores[disease] = {"pass": 0, "fail": 0}
        
        for item in samples:
            uid = item.get("id", "Unknown")
            gt_report = item.get("report", "No Report")
            
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
                    per_disease_scores[disease]["pass"] += 1
                else:
                    print(f"   ❌ [FAIL] {uid}: Disease missing from graph.")
                    print(f"      GT: {gt_report[:120]}...")
                    # Print what WAS found to debug
                    observations = [e[0] for e in kg['entities'] if e[1] == 'Observation']
                    print(f"      Graph found: {observations}")
                    score_card["fails"] += 1
                    per_disease_scores[disease]["fail"] += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {uid}: {e}")

    # Summary
    total = score_card['passes'] + score_card['fails']
    print("\n" + "=" * 50)
    print(f"📊 FINAL SCORE: {score_card['passes']} Passes / {score_card['fails']} Fails ({100*score_card['passes']/max(total,1):.0f}%)")
    print("=" * 50)
    print("\nPer-disease breakdown:")
    for disease, scores in per_disease_scores.items():
        p, f = scores["pass"], scores["fail"]
        total_d = p + f
        pct = 100 * p / max(total_d, 1)
        bar = "█" * p + "░" * f
        print(f"   {disease:>20s}: {bar} {p}/{total_d} ({pct:.0f}%)")

if __name__ == "__main__":
    run_stress_test()