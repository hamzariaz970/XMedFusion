import os
import json
import re
import asyncio
import pandas as pd
import numpy as np
import torch
import gc
import base64
import traceback
from tqdm.asyncio import tqdm
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import config

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
ANNOTATIONS_PATH = "data/iu_xray/annotation.json"
IMAGES_ROOT = "data/iu_xray/images"
OUTPUT_FILE = "out/test_generations_llava.json"
SCORES_FILE = "out/test_scores_llava.csv"
MODEL_NAME = config.OLLAMA_LLAVA_MODEL
TIMEOUT_SECONDS = 180

# Set to None to run ALL samples, or an integer (e.g., 5) for debugging
LIMIT = None

# ------------------------------------------------------------------
# LLM JUDGE  (identical to evaluate.py)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# NLG METRICS  (union of evaluate.py + calculate_metrics.py)
# ------------------------------------------------------------------
class MedicalReportEvaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        # Optional scorers loaded lazily
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            self.scorers.append((Meteor(), "METEOR"))
        except Exception:
            print("⚠️  METEOR unavailable (needs Java). Skipping.")

    def compute_scores(self, ref, hypo):
        score_results = {}

        # 1. pycocoevalcap metrics (BLEU, ROUGE-L, CIDEr, METEOR)
        for scorer, method in self.scorers:
            try:
                score, scores = scorer.compute_score(ref, hypo)
                if isinstance(method, list):
                    for m, s in zip(method, score):
                        score_results[m] = s
                else:
                    score_results[method] = score
            except Exception as e:
                print(f"Warning: Scorer {method} failed: {e}")

        # 2. Detailed ROUGE-1 / ROUGE-2 / ROUGE-L via rouge_score
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
            r1, r2, rl = [], [], []
            for id_ in ref:
                r = ref[id_][0]
                h = hypo[id_][0]
                s = scorer.score(r, h)
                r1.append(s['rouge1'].fmeasure)
                r2.append(s['rouge2'].fmeasure)
                rl.append(s['rougeL'].fmeasure)
            score_results['ROUGE-1'] = np.mean(r1)
            score_results['ROUGE-2'] = np.mean(r2)
            score_results['ROUGE-L (Detailed)'] = np.mean(rl)
        except ImportError:
            print("⚠️  rouge-score not installed. Skipping detailed ROUGE.")
        except Exception as e:
            print(f"Error calculating detailed ROUGE: {e}")

        # 3. BERTScore
        try:
            from bert_score import score as bert_score_func
            print("Calculating BERTScore (this may take a moment)...")
            ids = list(ref.keys())
            refs_list = [ref[i][0] for i in ids]
            hyps_list = [hypo[i][0] for i in ids]
            P, R, F1 = bert_score_func(
                hyps_list, refs_list, lang="en", verbose=True, device=None
            )
            score_results['BERTScore'] = F1.mean().item()
        except ImportError:
            print("⚠️  bert-score not installed. Skipping BERTScore.")
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")

        return score_results

# ------------------------------------------------------------------
# MAIN EVALUATION LOOP
# ------------------------------------------------------------------
async def run_llava_evaluation():
    print(f"🚀 LLaVA-Med Evaluation  |  model={MODEL_NAME}  |  judge={config.OLLAMA_JUDGE_MODEL}")

    # ── 1. Resume from previous run ──────────────────────────────
    results_log = []
    processed_ids = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results_log = json.load(f)
                processed_ids = {str(item['id']) for item in results_log}
            print(f"✅ Resuming from {len(processed_ids)} previously saved samples.")
        except Exception as e:
            print(f"⚠️ Could not load previous results, starting fresh: {e}")

    # ── 2. Load test data ────────────────────────────────────────
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Annotation file not found: {ANNOTATIONS_PATH}")
        return

    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)

    test_data = (
        data.get('test', []) if isinstance(data, dict)
        else [x for x in data if x.get('split') == 'test']
    )

    if LIMIT is not None:
        print(f"⚠️ LIMIT={LIMIT}. Running partial evaluation.")
        test_data = test_data[:LIMIT]
    else:
        print(f"📋 Full test set: {len(test_data)} samples.")

    # ── 3. Generate reports ──────────────────────────────────────
    if len(processed_ids) >= len(test_data):
        print("🎉 All samples already generated! Skipping to scoring.")
    else:
        # Initialise LLaVA-Med via Ollama
        llm = ChatOllama(model=MODEL_NAME, temperature=0.2)

        # Initialise Judge
        print(f"⚖️ Initializing Judge ({config.OLLAMA_JUDGE_MODEL})...")
        judge_llm = ChatOllama(model=config.OLLAMA_JUDGE_MODEL, temperature=0.1)
        judge = LLMJudge(judge_llm)

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        for idx, ex in enumerate(tqdm(test_data)):
            sample_id = str(ex.get('id', idx))

            if sample_id in processed_ids:
                continue

            # Image path
            img_list = ex.get('image_path', [])
            if not img_list:
                continue
            img_rel = img_list[0] if isinstance(img_list, list) else img_list
            img_path = os.path.join(IMAGES_ROOT, img_rel)
            ground_truth = ex.get('report', "")

            if not os.path.exists(img_path):
                print(f"⚠️ Image not found: {img_path}")
                continue

            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ── Encode image ─────────────────────────────────
                with open(img_path, "rb") as img_f:
                    image_b64 = base64.b64encode(img_f.read()).decode('utf-8')

                prompt_text = (
                    "You are an expert radiologist. You are evaluating the IU X-Ray dataset.\n\n"
                    "### TASK:\n"
                    "Look at this chest X-ray and write a radiology report.\n\n"
                    "### RULES:\n"
                    "- Write a single concise paragraph. No headers.\n"
                    "- Use professional radiology report style.\n"
                    "- Only describe findings you can actually see in the image.\n"
                    "- If the image appears normal, write a normal report.\n\n"
                    "REPORT:"
                )
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{image_b64}",
                        },
                    ]
                )

                # ── Generate with timeout ────────────────────────
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(llm.invoke, [message]),
                        timeout=TIMEOUT_SECONDS,
                    )
                    generated_text = response.content
                except asyncio.TimeoutError:
                    print(f"\n⏰ TIMEOUT sample {idx}. Recording error.")
                    clean_ref = ground_truth.replace('\n', ' ').strip().lower()
                    results_log.append({
                        "id": sample_id,
                        "image": img_path,
                        "raw_generated": "TIMEOUT_ERROR",
                        "final_formatted": "error",
                        "ground_truth": clean_ref,
                        "judge_scores": {},
                    })
                    with open(OUTPUT_FILE, 'w') as f:
                        json.dump(results_log, f, indent=2)
                    continue

                # ── Clean hypothesis & reference ─────────────────
                clean_ref = ground_truth.replace('\n', ' ').strip().lower()
                clean_hyp = generated_text.replace('\n', ' ').strip().lower()
                if not clean_hyp:
                    clean_hyp = "no report generated"

                # ── LLM Judge ────────────────────────────────────
                judge_scores = await asyncio.to_thread(
                    judge.evaluate_medical_accuracy, clean_ref, clean_hyp
                )

                # ── Save entry ───────────────────────────────────
                entry = {
                    "id": sample_id,
                    "image": img_path,
                    "raw_generated": generated_text,
                    "final_formatted": clean_hyp,
                    "ground_truth": clean_ref,
                    "judge_scores": judge_scores,
                }
                results_log.append(entry)

                # Atomic save
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(results_log, f, indent=2)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error on sample {idx}: {e}")
                traceback.print_exc()
                continue

    # ── 4. Compute aggregate metrics ─────────────────────────────
    print("\n" + "=" * 60)
    print("📊 CALCULATING AGGREGATE METRICS")
    print("=" * 60)

    if not results_log:
        print("❌ No results found.")
        return

    valid_logs = [
        x for x in results_log
        if x.get("final_formatted") != "error"
        and "TIMEOUT_ERROR" not in x.get("raw_generated", "")
    ]

    if not valid_logs:
        print("❌ No valid generations to score.")
        return

    print(f"Total samples: {len(results_log)}  |  Valid: {len(valid_logs)}")

    references = {item['id']: [item['ground_truth']] for item in valid_logs}
    hypotheses = {item['id']: [item['final_formatted']] for item in valid_logs}

    evaluator = MedicalReportEvaluator()
    scores = evaluator.compute_scores(references, hypotheses)

    # ── NLG metric table ─────────────────────────────────────────
    print(f"\n{'METRIC':<25} {'SCORE':<10}")
    print("-" * 35)
    for metric, val in scores.items():
        if isinstance(val, (int, float)):
            print(f"{metric:<25}: {val:.4f}")

    # ── Judge averages ───────────────────────────────────────────
    print("-" * 35)
    print("LLM JUDGE (1-10):")
    judge_keys = ["coverage", "consistency", "accuracy", "style", "conciseness"]
    for k in judge_keys:
        vals = [
            item['judge_scores'].get(k, 0)
            for item in valid_logs
            if isinstance(item.get('judge_scores'), dict)
        ]
        avg = np.mean(vals) if vals else 0
        scores[k] = avg
        print(f"  {k.capitalize():<23}: {avg:.4f}")

    scores["total_samples"] = len(results_log)
    scores["valid_samples"] = len(valid_logs)

    pd.DataFrame([scores]).to_csv(SCORES_FILE, index=False)
    print(f"\n✅ Done. Scores saved to {SCORES_FILE}")
    print(f"   Generations saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(run_llava_evaluation())
