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
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import traceback
import open_clip
from PIL import Image

# Use the updated LangChain core import
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
ANNOTATIONS_PATH = "data/iu_xray/annotation.json" 
IMAGES_ROOT = "data/iu_xray/images"
OUTPUT_FILE = "out/test_generations_bioclip_judge.json"
SCORES_FILE = "out/test_scores_bioclip_judge.csv"

# Models
BIOCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
LLM_GENERATOR_MODEL = "llama3.1:8b"  # Used for synthesizing the draft
LLM_JUDGE_MODEL = "llama3.1:8b"      # Used for grading and formatting

LIMIT = None # Set to an integer (e.g., 5) for debugging
TOP_K_RETRIEVALS = 3
TIMEOUT_SECONDS = 120

# ------------------------------------------------------------------
# HELPERS (Formatter, Judge, Evaluator)
# ------------------------------------------------------------------
class ReportFormatter:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def format_to_ground_truth_style(self, raw_report):
        prompt = f"Rewrite this radiology report into a single concise paragraph. Remove headers.\n\nREPORT:\n{raw_report}\n\nPARAGRAPH:"
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
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

        1. **Coverage**: Does the AI report include all critical clinical findings present in the GT?
        2. **Consistency**: Does the AI report contradict the GT? (10 = No contradictions).
        3. **Accuracy**: Is the overall clinical impression and diagnosis correct?
        4. **Style**: Does the writing style match the GT (professional radiology style)?
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
            response = self.llm.invoke([HumanMessage(content=prompt)])
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
    
    all_data = data if isinstance(data, list) else data.get('train', []) + data.get('test', [])
    test_data = [x for x in all_data if x.get('split') == 'test']
    train_data = [x for x in all_data if x.get('split') == 'train']
    
    if LIMIT is not None:
        print(f"⚠️ LIMIT set to {LIMIT}. Running partial evaluation.")
        test_data = test_data[:LIMIT]
    else:
        print(f"🚀 Running on FULL Test Set ({len(test_data)} samples).")
    
    if len(processed_ids) >= len(test_data):
        print("🎉 All samples already generated! Skipping directly to scoring.")
    else:
        # 3. INITIALIZE MODELS & AGENTS
        print("⚙️ Initializing BioCLIP and LLMs...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _, preprocess = open_clip.create_model_and_transforms(BIOCLIP_MODEL, device=device)
        tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL)
        clip_model.eval()

        generator_llm = ChatOllama(model=LLM_GENERATOR_MODEL, temperature=0.2)
        judge_llm = ChatOllama(model=LLM_JUDGE_MODEL, temperature=0.1)
        
        formatter = ReportFormatter(judge_llm)
        judge = LLMJudge(judge_llm)

        # 4. BUILD RETRIEVAL POOL
        print("📚 Building text retrieval pool from training data...")
        unique_reports = list(set([ex['report'] for ex in train_data if ex.get('report')]))
        text_features_list = []
        batch_size = 128
        
        with torch.no_grad():
            for i in tqdm(range(0, len(unique_reports), batch_size), desc="Encoding text"):
                batch_texts = unique_reports[i:i+batch_size]
                text_tokens = tokenizer(batch_texts, context_length=256).to(device)
                features = clip_model.encode_text(text_tokens)
                features /= features.norm(dim=-1, keepdim=True)
                text_features_list.append(features)
            all_text_features = torch.cat(text_features_list, dim=0)

        # 5. GENERATION & JUDGING LOOP
        print(f"🚀 Generating missing samples...")
        
        for idx, ex in enumerate(tqdm(test_data)):
            sample_id = str(ex.get('id', idx))
            if sample_id in processed_ids: continue

            img_list = ex.get('image_path', [])
            if not img_list: continue
            
            img_rel_path = img_list[0] if isinstance(img_list, list) else img_list
            img_path = os.path.join(IMAGES_ROOT, img_rel_path)
            ground_truth = ex.get('report', "")

            if not os.path.exists(img_path): continue
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

            try:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                # --- A. BioCLIP Retrieval ---
                with torch.no_grad():
                    image = Image.open(img_path).convert('RGB')
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    image_features = clip_model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    similarities = (image_features @ all_text_features.T).squeeze(0)
                    top_k_indices = similarities.topk(TOP_K_RETRIEVALS).indices.tolist()
                    retrieved_reports = [unique_reports[i] for i in top_k_indices]

                # --- B. LLM Synthesize Draft ---
                reports_text = "\n".join([f"- {r}" for r in retrieved_reports])
                prompt = (
                    "You are an expert radiologist. Based on the visual similarity to historical cases, "
                    f"the current X-ray matches these findings:\n\n{reports_text}\n\n"
                    "Synthesize a single, professional radiology report for the current patient. "
                    "Do not mention historical cases."
                )

                async def _generate_draft():
                    response = await asyncio.to_thread(generator_llm.invoke, [HumanMessage(content=prompt)])
                    return response.content

                try:
                    raw_text = await asyncio.wait_for(_generate_draft(), timeout=TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    print(f"\n⏰ TIMEOUT Sample {idx}. Skipping.")
                    clean_ref = ground_truth.replace('\n', ' ').strip().lower()
                    new_entry = {
                        "id": sample_id, "image": img_path, "raw_generated": "TIMEOUT_ERROR",
                        "final_formatted": "error", "ground_truth": clean_ref, "judge_scores": {}
                    }
                    results_log.append(new_entry)
                    with open(OUTPUT_FILE, 'w') as f: json.dump(results_log, f, indent=2)
                    continue 

                # --- C. Format & Judge ---
                formatted_hyp = await asyncio.to_thread(formatter.format_to_ground_truth_style, raw_text)
                clean_ref = ground_truth.replace('\n', ' ').strip().lower()
                clean_hyp = formatted_hyp.replace('\n', ' ').strip().lower()
                if not clean_hyp: clean_hyp = "no report generated"

                scores_dict = await asyncio.to_thread(judge.evaluate_medical_accuracy, clean_ref, clean_hyp)

                # --- D. Save ---
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

            except Exception as e:
                print(f"Error {idx}: {e}")
                continue

    # 6. COMPUTE FINAL METRICS (PyCOCO & Average Judge)
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




