import os
import json
import pandas as pd
import re
import torch
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Import Agents & Config
from synthesis import (
    LocalSynthesisAgent,
    RetrievalAgent,
    LocalLLMReportAgent,
    VisionLLMAgent,
    reports_dict,
    model,
    preprocess,
    device
)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
ANNOTATIONS_PATH = "data/iu_xray/annotation.json" 
IMAGES_ROOT = "data/iu_xray/images"
OUTPUT_FILE = "test_generations_formatted.json"
SCORES_FILE = "test_scores_formatted.csv"
LIMIT = 3

# ------------------------------------------------------------------
# FORMATTER AGENT (New Addition)
# ------------------------------------------------------------------
class ReportFormatter:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def format_to_ground_truth_style(self, raw_report):
        """
        Takes the detailed agent output and rewrites it as a single 
        narrative paragraph to match IU X-Ray ground truth style.
        """
        prompt = f"""
You are a medical data cleaner. Rewrite the following radiology report content into a SINGLE continuous paragraph.
- Remove all headers (FINDINGS, IMPRESSION, etc.).
- Remove bullet points.
- Remove labels.
- Make it flow naturally like a story.
- Keep it concise.

Here are examples of the target style:
1. "The heart size and pulmonary vascularity appear within normal limits. A large hiatal hernia is noted. The lungs are free of focal airspace disease."
2. "Cardiac and mediastinal contours are within normal limits. The lungs are clear. Bony structures are intact."

RAW REPORT TO REWRITE:
{raw_report}

REWRITTEN PARAGRAPH ONLY:
"""
        # Using the existing LLM agent to invoke this simple task
        response = self.llm.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        # Clean thinking tokens if using DeepSeek
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content

# ------------------------------------------------------------------
# SCORER CLASS
# ------------------------------------------------------------------
class MedicalReportEvaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

    def compute_scores(self, ref, hypo):
        score_results = {}
        for scorer, method in self.scorers:
            try:
                score, scores = scorer.compute_score(ref, hypo)
                if isinstance(method, list):
                    for m, s in zip(method, score):
                        score_results[m] = s
                else:
                    score_results[method] = score
            except Exception as e:
                # print(f"⚠️ Error in {method}: {e}")
                if isinstance(method, list):
                    for m in method: score_results[m] = 0.0
                else:
                    score_results[method] = 0.0
        return score_results

# ------------------------------------------------------------------
# MAIN EVALUATION LOOP
# ------------------------------------------------------------------
def run_evaluation():
    print(f"📂 Loading annotations from {ANNOTATIONS_PATH}...")
    
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'test' in data:
        test_data_raw = data['test']
    elif isinstance(data, list):
         test_data_raw = [x for x in data if x.get('split') == 'test']
    else:
        print("❌ Error: Could not parse annotations.")
        return

    # Flatten Data
    test_image_paths = []
    test_reports = []
    
    for ex in test_data_raw:
        img_paths = ex.get('image_path', [])
        if isinstance(img_paths, str): img_paths = [img_paths]
        
        if img_paths:
            full_path = os.path.join(IMAGES_ROOT, img_paths[0])
            test_image_paths.append(full_path)
            test_reports.append(ex.get('report', ""))

    print(f"🧪 Found {len(test_image_paths)} Total Samples.")
    print(f"⚠️ DEBUG MODE: Running only {LIMIT} samples.\n")
    
    test_image_paths = test_image_paths[:LIMIT]
    test_reports = test_reports[:LIMIT]

    # Initialize Agents
    print("🤖 Initializing Agents...")
    retrieval_agent = RetrievalAgent(model, preprocess, k=5, device=device)
    draft_agent = LocalLLMReportAgent(model_name="ollama/deepseek-r1:1.5b")
    vision_agent = VisionLLMAgent(model_name="ollama/deepseek-r1:1.5b")
    synthesis_agent = LocalSynthesisAgent(model_name="ollama/deepseek-r1:1.5b")
    
    # NEW: Initialize Formatter
    formatter = ReportFormatter(draft_agent) 

    references = {}
    hypotheses = {}
    results_log = []

    print("\n" + "="*60)
    print("🚀 VISUAL COMPARISON (With Formatting Step)")
    print("="*60)

    for idx, (img_path, ground_truth) in enumerate(tqdm(zip(test_image_paths, test_reports), total=len(test_image_paths))):
        sample_id = str(idx)

        if not os.path.exists(img_path):
            if os.path.exists(os.path.basename(img_path)):
                img_path = os.path.basename(img_path)
            else:
                continue

        try:
            # 1. GENERATE RAW OUTPUT (Structured with headers)
            gen = synthesis_agent.generate_final_report(
                draft_agent=draft_agent,
                vision_agent=vision_agent,
                retrieval_agent=retrieval_agent,
                reports_dict=reports_dict,
                image_paths=[img_path] 
            )

            raw_text = ""
            for chunk in gen:
                try:
                    chunk_data = json.loads(chunk)
                    if chunk_data.get("status") == "complete":
                        raw_text = chunk_data.get("final_report", "")
                except: pass
            
            # 2. FORMATTING STEP (New LLM Call)
            # Transforms the raw output into a single GT-style paragraph
            formatted_hyp = formatter.format_to_ground_truth_style(raw_text)
            
            # 3. NORMALIZE (Lowercase, remove extra spaces for metric calc)
            clean_ref = ground_truth.replace('\n', ' ').strip().lower()
            clean_hyp = formatted_hyp.replace('\n', ' ').strip().lower()
            
            # Safety check
            if not clean_hyp: clean_hyp = "no report generated"

            # 4. DEBUG PRINT
            print(f"\n[Sample {idx}]")
            print(f"📸 Image: {os.path.basename(img_path)}")
            print(f"📘 REF: {clean_ref[:80]}...") 
            print(f"📙 HYP: {clean_hyp[:80]}...") 

            # 5. STORE
            references[sample_id] = [clean_ref]
            hypotheses[sample_id] = [clean_hyp]

            results_log.append({
                "id": sample_id,
                "image": img_path,
                "raw_generated": raw_text,
                "final_formatted": clean_hyp,
                "ground_truth": clean_ref
            })

        except Exception as e:
            print(f"Error: {e}")
            continue

    # 5. COMPUTE SCORES
    print("\n" + "="*60)
    print("📊 CALCULATING METRICS")
    print("="*60)
    
    if not references:
        print("❌ No valid results.")
        return

    evaluator = MedicalReportEvaluator()
    scores = evaluator.compute_scores(references, hypotheses)

    for metric, score in scores.items():
        print(f"{metric:<10}: {score:.4f}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results_log, f, indent=2)
    pd.DataFrame([scores]).to_csv(SCORES_FILE, index=False)
    print(f"\n✅ Done. Full log saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()