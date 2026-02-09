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
# from pycocoevalcap.meteor.meteor import Meteor # Disabled to prevent Java crash
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import traceback
from langchain_community.chat_models import ChatOllama

# --- IMPORTS FOR NEW PIPELINE ---
import config
from synthesis import LocalSynthesisAgent
from draft import RetrievalAgent, LocalLLMReportAgent, reports_dict
from vision import vision_encoder, VisualDescriptionAgent

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
ANNOTATIONS_PATH = "data/iu_xray/annotation.json" 
IMAGES_ROOT = "data/iu_xray/images"
OUTPUT_FILE = "out/test_generations_judge.json"
SCORES_FILE = "out/test_scores_judge.csv"

# Set to None to run ALL samples, or an integer (e.g., 20) for debugging
LIMIT = None 

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
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def evaluate_medical_accuracy(self, reference, hypothesis):
        prompt = f"""
        You are an expert radiologist and medical auditor. Evaluate the AI-generated report against the Ground Truth (GT) report.
        Rate the AI report on the following 5 dimensions strictly on a scale of 1-10 (10 is perfect/identical to GT quality):

        1. **Coverage of Key Findings**: Does the AI report include all critical clinical findings present in the GT?
        2. **Consistency**: Does the AI report contradict the GT? (10 = No contradictions).
        3. **Diagnostic Accuracy**: Is the overall clinical impression and diagnosis correct?
        4. **Stylistic Alignment**: Does the writing style match the GT (professional radiology style)?
        5. **Conciseness**: Is the report concise and free of unnecessary fluff?

        GT: "{reference}"
        AI: "{hypothesis}"

        Output ONLY valid JSON:
        {{
          "coverage": <int>,
          "consistency": <int>,
          "accuracy": <int>,
          "style": <int>,
          "conciseness": <int>
        }}
        """
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"coverage": 5, "consistency": 5, "accuracy": 5, "style": 5, "conciseness": 5}
        except: 
            return {"coverage": 5, "consistency": 5, "accuracy": 5, "style": 5, "conciseness": 5}

class MedicalReportEvaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
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
                pass 
        return score_results

# ------------------------------------------------------------------
# MAIN EVALUATION LOOP
# ------------------------------------------------------------------
async def run_evaluation():
    # 1. LOAD PREVIOUS RESULTS
    results_log = []
    processed_ids = set()
    
    if os.path.exists(OUTPUT_FILE):
        print(f"🔄 Found existing results file: {OUTPUT_FILE}")
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results_log = json.load(f)
                processed_ids = {str(item['id']) for item in results_log}
            print(f"✅ Loaded {len(results_log)} previously generated samples. Resuming...")
        except Exception as e:
            print(f"⚠️ Error loading file, starting fresh: {e}")

    # 2. LOAD DATASET
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Annotation file not found: {ANNOTATIONS_PATH}")
        return

    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)
    
    test_data_raw = data.get('test', []) if isinstance(data, dict) else [x for x in data if x.get('split') == 'test']
    
    if LIMIT is not None:
        print(f"⚠️ LIMIT set to {LIMIT}. Running partial evaluation.")
        test_data_raw = test_data_raw[:LIMIT]
    else:
        print(f"🚀 Running on FULL Test Set ({len(test_data_raw)} samples).")
    
    if len(processed_ids) >= len(test_data_raw):
        print("🎉 All samples already generated! Skipping directly to scoring.")
    else:
        # --- INITIALIZE AGENTS ---
        print("⚙️ Initializing Agents...")
        
        # FIXED: Pass model and preprocess as POSITIONAL arguments to avoid keyword errors
        retrieval_agent = RetrievalAgent(
            vision_encoder.model,       # Positional 1
            vision_encoder.preprocess,  # Positional 2
            k=3, 
            device=vision_encoder.device
        )
        
        draft_agent = LocalLLMReportAgent()
        vision_agent = VisualDescriptionAgent() 
        synthesis_agent = LocalSynthesisAgent()
        
        print(f"⚖️ Initializing Judge ({config.OLLAMA_JUDGE_MODEL})...")
        judge_llm = ChatOllama(model=config.OLLAMA_JUDGE_MODEL, temperature=0.1)
        judge = LLMJudge(judge_llm)
        
        formatter = ReportFormatter(draft_agent)

        print(f"🚀 Generating missing samples...")
        
        for idx, ex in enumerate(tqdm(test_data_raw)):
            sample_id = str(ex.get('id', idx))
            
            if sample_id in processed_ids:
                continue

            img_list = ex.get('image_path', [])
            if not img_list: continue
            
            img_rel_path = img_list[0] if isinstance(img_list, list) else img_list
            img_path = os.path.join(IMAGES_ROOT, img_rel_path)
            ground_truth = ex.get('report', "")

            if not os.path.exists(img_path):
                if os.path.exists(os.path.basename(img_path)):
                    img_path = os.path.basename(img_path)
                else:
                    continue

            # Ensure Output Directory Exists
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

            try:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                async def _generate_with_timeout():
                    raw = ""
                    accumulated = ""
                    gen = synthesis_agent.generate_final_report(
                        draft_agent, vision_agent, retrieval_agent, reports_dict, [img_path]
                    )
                    
                    async for chunk in gen:
                        d = json.loads(chunk)
                        if d.get("status") == "streaming" and "chunk" in d:
                            accumulated += d["chunk"]
                        if d.get("status") == "complete":
                            raw = d.get("final_report", "")
                    
                    if not raw and accumulated:
                        raw = accumulated
                    return raw

                # Execute with 120s (2 min) timeout
                try:
                    raw_text = await asyncio.wait_for(_generate_with_timeout(), timeout=120)
                except asyncio.TimeoutError:
                    print(f"\n⏰ TIMEOUT Sample {idx}. Skipping.")
                    clean_ref = ground_truth.replace('\n', ' ').strip().lower()
                    new_entry = {
                        "id": sample_id,
                        "image": img_path,
                        "raw_generated": "TIMEOUT_ERROR",
                        "final_formatted": "error",
                        "ground_truth": clean_ref,
                        "judge_scores": {}
                    }
                    results_log.append(new_entry)
                    with open(OUTPUT_FILE, 'w') as f:
                        json.dump(results_log, f, indent=2)
                    continue 

                formatted_hyp = await asyncio.to_thread(formatter.format_to_ground_truth_style, raw_text)
                clean_ref = ground_truth.replace('\n', ' ').strip().lower()
                clean_hyp = formatted_hyp.replace('\n', ' ').strip().lower()
                if not clean_hyp: clean_hyp = "no report generated"

                scores_dict = await asyncio.to_thread(judge.evaluate_medical_accuracy, clean_ref, clean_hyp)

                new_entry = {
                    "id": sample_id,
                    "image": img_path,
                    "raw_generated": raw_text,
                    "final_formatted": clean_hyp,
                    "ground_truth": clean_ref,
                    "judge_scores": scores_dict
                }
                results_log.append(new_entry)
                
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(results_log, f, indent=2)

                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error {idx}: {e}")
                continue

    # 4. COMPUTE SCORES
    print("\n" + "="*60)
    print("📊 CALCULATING AGGREGATE METRICS")
    print("="*60)
    
    if not results_log:
        print("❌ No results found.")
        return

    valid_logs = [x for x in results_log if x.get("final_formatted") != "error"]
    
    if not valid_logs:
        print("❌ No valid generations found.")
        return

    references = {item['id']: [item['ground_truth']] for item in valid_logs}
    hypotheses = {item['id']: [item['final_formatted']] for item in valid_logs}

    evaluator = MedicalReportEvaluator()
    scores = evaluator.compute_scores(references, hypotheses)

    scores["LLM_Judge_Msg"] = "See detailed scores below"

    print(f"{'METRIC':<25} {'SCORE':<10}")
    print("-" * 35)
    for metric, score in scores.items():
        if isinstance(score, (int, float)):
             print(f"{metric:<25}: {score:.4f}")
    
    print("-" * 35)
    print("LLM JUDGE (1-10):")
    keys = ["coverage", "consistency", "accuracy", "style", "conciseness"]
    judge_avgs = {}
    for k in keys:
        try:
             vals = [item['judge_scores'].get(k, 0) for item in valid_logs if 'judge_scores' in item]
             avg = np.mean(vals) if vals else 0
             print(f"{k.capitalize():<25}: {avg:.4f}")
             judge_avgs[k] = avg
        except: pass
    
    scores.update(judge_avgs)
    scores["total_samples"] = len(results_log)
    scores["valid_samples"] = len(valid_logs)

    pd.DataFrame([scores]).to_csv(SCORES_FILE, index=False)
    print(f"\n✅ All Done. Scores saved to {SCORES_FILE}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())