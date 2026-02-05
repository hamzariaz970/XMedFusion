import os
import json
import pandas as pd
import re
import asyncio
import torch
import gc
import numpy as np
from tqdm.asyncio import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Import Agents & Config
import config
from synthesis import (
    LocalSynthesisAgent,
    RetrievalAgent,
    LocalLLMReportAgent,
    VisionLLMAgent,
    reports_dict,
    clip_model,
    clip_prep,
    device
)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
ANNOTATIONS_PATH = "data/iu_xray/annotation.json" 
IMAGES_ROOT = "data/iu_xray/images"
OUTPUT_FILE = "out/test_generations_judge.json"
SCORES_FILE = "out/test_scores_judge.csv"
LIMIT = 10
# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
class ReportFormatter:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def format_to_ground_truth_style(self, raw_report):
        prompt = f"Rewrite this radiology report into a single concise paragraph. Remove headers.\n\nREPORT:\n{raw_report}\n\nPARAGRAPH:"
        try:
            response = self.llm.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        except:
            return raw_report

class LLMJudge:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def evaluate_medical_accuracy(self, reference, hypothesis):
        prompt = f"""
        Rate medical accuracy (1-5) of AI Report vs Ground Truth (GT).
        GT: "{reference}"
        AI: "{hypothesis}"
        Output ONLY the integer.
        """
        try:
            response = self.llm.llm.invoke(prompt)
            score = re.search(r'\d', str(response.content))
            return int(score.group()) if score else 3
        except: return 3

class MedicalReportEvaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(), "METEOR"),  <-- DISABLED TO PREVENT JAVA CRASH
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

    def compute_scores(self, ref, hypo):
        score_results = {}
        for scorer, method in self.scorers:
            try:
                score, scores = scorer.compute_score(ref, hypo)
                if isinstance(method, list):
                    for m, s in zip(method, score): score_results[m] = s
                else:
                    score_results[method] = score
            except: 
                pass # Silently fail on individual metrics if needed
        return score_results

# ------------------------------------------------------------------
# MAIN EVALUATION LOOP
# ------------------------------------------------------------------
async def run_evaluation():
    # 1. LOAD PREVIOUS RESULTS (RESUME CAPABILITY)
    results_log = []
    processed_ids = set()
    
    if os.path.exists(OUTPUT_FILE):
        print(f"🔄 Found existing results file: {OUTPUT_FILE}")
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results_log = json.load(f)
                processed_ids = {item['id'] for item in results_log}
            print(f"✅ Loaded {len(results_log)} previously generated samples. Resuming...")
        except Exception as e:
            print(f"⚠️ Error loading file, starting fresh: {e}")

    # 2. LOAD DATASET
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)
    test_data_raw = data.get('test', []) if isinstance(data, dict) else [x for x in data if x.get('split') == 'test']
    
    # Apply Limit
    test_data_raw = test_data_raw[:LIMIT]
    
    # Check if we are already done
    if len(processed_ids) >= len(test_data_raw):
        print("🎉 All samples already generated! Skipping directly to scoring.")
    else:
        # 3. INITIALIZE AGENTS (Only if we need to generate)
        print(f"🤖 Initializing Agents ({config.OLLAMA_MODEL})...")
        retrieval_agent = RetrievalAgent(clip_model, clip_prep, k=5, device=device)
        draft_agent = LocalLLMReportAgent()
        vision_agent = VisionLLMAgent()
        synthesis_agent = LocalSynthesisAgent()
        
        formatter = ReportFormatter(draft_agent)
        judge = LLMJudge(draft_agent)

        print(f"🚀 Generating missing samples...")
        
        for idx, ex in enumerate(tqdm(test_data_raw)):
            sample_id = str(idx)
            
            # Skip if already done
            if sample_id in processed_ids:
                continue

            img_path = os.path.join(IMAGES_ROOT, ex['image_path'][0])
            ground_truth = ex['report']

            if not os.path.exists(img_path):
                if os.path.exists(os.path.basename(img_path)):
                    img_path = os.path.basename(img_path)
                else:
                    continue

            try:
                # -- GENERATION PIPELINE --
                raw_text = ""
                gen = synthesis_agent.generate_final_report(
                    draft_agent, vision_agent, retrieval_agent, reports_dict, [img_path]
                )
                async for chunk in gen:
                    d = json.loads(chunk)
                    if d.get("status") == "complete":
                        raw_text = d.get("final_report", "")

                # Format & Normalize
                formatted_hyp = await asyncio.to_thread(formatter.format_to_ground_truth_style, raw_text)
                clean_ref = ground_truth.replace('\n', ' ').strip().lower()
                clean_hyp = formatted_hyp.replace('\n', ' ').strip().lower()
                if not clean_hyp: clean_hyp = "no report generated"

                # Judge
                score = await asyncio.to_thread(judge.evaluate_medical_accuracy, clean_ref, clean_hyp)

                # Add to log
                new_entry = {
                    "id": sample_id,
                    "image": img_path,
                    "raw_generated": raw_text,
                    "final_formatted": clean_hyp,
                    "ground_truth": clean_ref,
                    "judge_score": score
                }
                results_log.append(new_entry)
                
                # --- CRITICAL: SAVE IMMEDIATELY ---
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(results_log, f, indent=2)
                # ----------------------------------

                print(f"\n[Img {idx}] Judge: {score}/5 (Saved)")
                
                # Memory Cleanup
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error {idx}: {e}")
                continue

    # 4. COMPUTE NLP SCORES (Using loaded JSON)
    print("\n" + "="*60)
    print("📊 CALCULATING AGGREGATE METRICS")
    print("="*60)
    
    if not results_log:
        print("❌ No results found to score.")
        return

    # Prepare dictionaries for pycocoevalcap
    references = {item['id']: [item['ground_truth']] for item in results_log}
    hypotheses = {item['id']: [item['final_formatted']] for item in results_log}
    judge_scores = [item['judge_score'] for item in results_log]

    evaluator = MedicalReportEvaluator()
    scores = evaluator.compute_scores(references, hypotheses)
    
    # Add Judge Score
    scores["LLM_Judge_Avg"] = np.mean(judge_scores) if judge_scores else 0

    print(f"{'METRIC':<15} {'SCORE':<10}")
    print("-" * 25)
    for metric, score in scores.items():
        print(f"{metric:<15}: {score:.4f}")

    pd.DataFrame([scores]).to_csv(SCORES_FILE, index=False)
    print(f"\n✅ All Done. Scores saved to {SCORES_FILE}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())