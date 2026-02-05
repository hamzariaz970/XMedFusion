import os
import json
import pandas as pd
import re
import asyncio
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
OUTPUT_FILE = "test_generations_judge.json"
SCORES_FILE = "test_scores_judge.csv"
LIMIT = 5  # Small limit to test without waiting too long

# ------------------------------------------------------------------
# FORMATTER AGENT (Style Matcher)
# ------------------------------------------------------------------
class ReportFormatter:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def format_to_ground_truth_style(self, raw_report):
        prompt = f"""
You are a medical data cleaner. Rewrite the following radiology report content into a SINGLE continuous paragraph.
- Remove all headers (FINDINGS, IMPRESSION, etc.).
- Remove bullet points.
- Remove labels.
- Make it flow naturally like a story.
- Keep it concise.

RAW REPORT:
{raw_report}

REWRITTEN PARAGRAPH ONLY:
"""
        response = self.llm.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content

# ------------------------------------------------------------------
# LLM-AS-A-JUDGE (Medical Accuracy Scorer)
# ------------------------------------------------------------------
class LLMJudge:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def evaluate_medical_accuracy(self, reference, hypothesis):
        """
        Asks the LLM to score the medical accuracy between 1-5.
        Inspired by RadGraph/RadCheck methodologies.
        """
        prompt = f"""
You are a senior radiologist auditing AI reports. 
Compare the Ground Truth report to the AI Generated report.

Ground Truth: "{reference}"
AI Generated: "{hypothesis}"

Task: Rate the AI report's MEDICAL ACCURACY on a scale of 1 to 5.
1: Completely wrong / hallucinated findings.
2: Missed major findings or added major false positives.
3: Captured some findings but missed others or minor errors.
4: Medically accurate but minor style/phrasing differences.
5: Perfect medical equivalent (findings match exactly).

Output ONLY the number (1-5).
"""
        try:
            response = self.llm.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            # Extract the first digit found
            score = re.search(r'\d', content)
            return int(score.group()) if score else 3
        except:
            return 3 # Default to average if parsing fails

# ------------------------------------------------------------------
# SCORER CLASS (NLP Metrics)
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
            except Exception:
                if isinstance(method, list):
                    for m in method: score_results[m] = 0.0
                else:
                    score_results[method] = 0.0
        return score_results

# ------------------------------------------------------------------
# MAIN EVALUATION LOOP (Async)
# ------------------------------------------------------------------
async def run_evaluation():
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

    # Initialize Agents (Using Global Config)
    print(f"🤖 Initializing Agents ({config.OLLAMA_MODEL})...")
    
    retrieval_agent = RetrievalAgent(clip_model, clip_prep, k=5, device=device)
    draft_agent = LocalLLMReportAgent()     # Defaults to config model
    vision_agent = VisionLLMAgent()         # Defaults to config model
    synthesis_agent = LocalSynthesisAgent() # Defaults to config model
    
    # Initialize Evaluators
    formatter = ReportFormatter(draft_agent) 
    judge = LLMJudge(draft_agent) # Use the same LLM to judge

    references = {}
    hypotheses = {}
    results_log = []
    judge_scores = []

    print("\n" + "="*60)
    print("🚀 STARTING ASYNC EVALUATION")
    print("="*60)

    # Use tqdm async for progress bar
    for idx, (img_path, ground_truth) in enumerate(tqdm(zip(test_image_paths, test_reports), total=len(test_image_paths))):
        sample_id = str(idx)

        if not os.path.exists(img_path):
            if os.path.exists(os.path.basename(img_path)):
                img_path = os.path.basename(img_path)
            else:
                continue

        try:
            # 1. RUN PIPELINE (Async Generator)
            # We consume the generator to get the final output
            raw_text = ""
            kg_data = None
            
            gen = synthesis_agent.generate_final_report(
                draft_agent=draft_agent,
                vision_agent=vision_agent,
                retrieval_agent=retrieval_agent,
                reports_dict=reports_dict,
                image_paths=[img_path] 
            )

            async for chunk in gen:
                try:
                    chunk_data = json.loads(chunk)
                    if chunk_data.get("status") == "complete":
                        raw_text = chunk_data.get("final_report", "")
                        kg_data = chunk_data.get("knowledge_graph")
                except: pass
            
            # 2. FORMATTING STEP (Sync)
            formatted_hyp = await asyncio.to_thread(formatter.format_to_ground_truth_style, raw_text)
            
            # 3. NORMALIZE
            clean_ref = ground_truth.replace('\n', ' ').strip().lower()
            clean_hyp = formatted_hyp.replace('\n', ' ').strip().lower()
            if not clean_hyp: clean_hyp = "no report generated"

            # 4. LLM JUDGE SCORE (Sync)
            medical_score = await asyncio.to_thread(judge.evaluate_medical_accuracy, clean_ref, clean_hyp)
            judge_scores.append(medical_score)

            # 5. STORE
            references[sample_id] = [clean_ref]
            hypotheses[sample_id] = [clean_hyp]

            results_log.append({
                "id": sample_id,
                "image": img_path,
                "raw_generated": raw_text,
                "final_formatted": clean_hyp,
                "ground_truth": clean_ref,
                "judge_score": medical_score
            })

            # Live Print
            print(f"\n[Img {idx}] Judge Score: {medical_score}/5 | BLEU-4 will be calc at end.")

        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue

    # 6. COMPUTE NLP SCORES
    print("\n" + "="*60)
    print("📊 CALCULATING AGGREGATE METRICS")
    print("="*60)
    
    if not references:
        print("❌ No valid results.")
        return

    evaluator = MedicalReportEvaluator()
    scores = evaluator.compute_scores(references, hypotheses)

    # Add Judge Score Average
    avg_judge = np.mean(judge_scores) if judge_scores else 0
    scores["LLM_Judge_Avg"] = avg_judge

    # Print Final Table
    print(f"{'METRIC':<15} {'SCORE':<10}")
    print("-" * 25)
    for metric, score in scores.items():
        print(f"{metric:<15}: {score:.4f}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results_log, f, indent=2)
    
    # Save scores including the new judge metric
    df_scores = pd.DataFrame([scores])
    df_scores.to_csv(SCORES_FILE, index=False)
    
    print(f"\n✅ Done. Full log saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())