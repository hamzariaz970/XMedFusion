"""
Calculate metrics on existing LLaVA-Med generation results.
Usage:  python calculate_metrics_llava.py
        python calculate_metrics_llava.py path/to/other_file.json
"""
import json
import os
import re
import sys
import numpy as np
import pandas as pd
from langchain_community.chat_models import ChatOllama
import config
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Defaults
DEFAULT_INPUT = "out/test_generations_llava.json"
DEFAULT_SCORES = "out/test_scores_llava.csv"


# ------------------------------------------------------------------
# LLM JUDGE  (identical to evaluate.py — 5 dimensions)
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
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            self.scorers.append((Meteor(), "METEOR"))
        except Exception:
            print("⚠️  METEOR unavailable (needs Java). Skipping.")

    def compute_scores(self, ref, hypo):
        results = {}

        # 1. pycocoevalcap metrics
        for scorer, method in self.scorers:
            try:
                score, _ = scorer.compute_score(ref, hypo)
                if isinstance(method, list):
                    for m, s in zip(method, score):
                        results[m] = s
                else:
                    results[method] = score
            except Exception as e:
                print(f"Warning: {method} failed: {e}")

        # 2. Detailed ROUGE-1 / ROUGE-2 / ROUGE-L
        try:
            from rouge_score import rouge_scorer
            sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            r1, r2, rl = [], [], []
            for id_ in ref:
                s = sc.score(ref[id_][0], hypo[id_][0])
                r1.append(s['rouge1'].fmeasure)
                r2.append(s['rouge2'].fmeasure)
                rl.append(s['rougeL'].fmeasure)
            results['ROUGE-1'] = np.mean(r1)
            results['ROUGE-2'] = np.mean(r2)
            results['ROUGE-L (Detailed)'] = np.mean(rl)
        except ImportError:
            print("⚠️  rouge-score not installed. Skipping detailed ROUGE.")
        except Exception as e:
            print(f"Error calculating detailed ROUGE: {e}")

        # 3. BERTScore
        try:
            from bert_score import score as bert_score_func
            print("Calculating BERTScore (may take a moment)...")
            ids = list(ref.keys())
            refs_list = [ref[i][0] for i in ids]
            hyps_list = [hypo[i][0] for i in ids]
            P, R, F1 = bert_score_func(hyps_list, refs_list, lang="en", verbose=True, device=None)
            results['BERTScore'] = F1.mean().item()
        except ImportError:
            print("⚠️  bert-score not installed. Skipping BERTScore.")
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")

        return results


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    scores_file = input_file.replace(".json", "_scores.csv") if len(sys.argv) > 1 else DEFAULT_SCORES

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")

    # Filter out errors / timeouts
    valid = [
        x for x in data
        if x.get("final_formatted") != "error"
        and "TIMEOUT_ERROR" not in x.get("raw_generated", "")
    ]
    skipped = len(data) - len(valid)
    print(f"Skipped (timeout/error): {skipped}")
    print(f"Valid samples: {len(valid)}")

    if not valid:
        print("No valid samples to evaluate.")
        return

    # ── Run LLM Judge on entries missing scores ──────────────────
    needs_judging = [
        item for item in valid
        if not isinstance(item.get("judge_scores"), dict)
        or not item["judge_scores"]  # empty dict
    ]

    if needs_judging:
        print(f"\n⚖️ {len(needs_judging)} entries need LLM Judge scoring ({config.OLLAMA_JUDGE_MODEL})...")
        judge_llm = ChatOllama(model=config.OLLAMA_JUDGE_MODEL, temperature=0.1)
        judge = LLMJudge(judge_llm)

        for i, item in enumerate(needs_judging):
            ref = item.get("ground_truth", "")
            hyp = item.get("final_formatted", "")
            print(f"  Judging {i+1}/{len(needs_judging)} (id={item['id']})...", end=" ")
            item["judge_scores"] = judge.evaluate_medical_accuracy(ref, hyp)
            print(f"✓ {item['judge_scores']}")

            # Atomic save after each judge call
            with open(input_file, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"✅ All entries judged. Updated {input_file}")
    else:
        print("✅ All entries already have judge scores.")

    # ── Build dicts for NLG metrics ──────────────────────────────
    references = {item['id']: [item['ground_truth']] for item in valid}
    hypotheses = {item['id']: [item['final_formatted']] for item in valid}

    evaluator = MedicalReportEvaluator()
    scores = evaluator.compute_scores(references, hypotheses)

    # ── Judge averages (all 5 dimensions) ────────────────────────
    print("\n" + "-" * 35)
    print("LLM JUDGE (1-10):")
    judge_keys = ["coverage", "consistency", "accuracy", "style", "conciseness"]
    for k in judge_keys:
        vals = [
            item['judge_scores'].get(k, 0)
            for item in valid
            if isinstance(item.get('judge_scores'), dict)
        ]
        avg = np.mean(vals) if vals else 0
        scores[f"Judge_{k.capitalize()}"] = avg
        print(f"  {k.capitalize():<23}: {avg:.4f}")

    scores["total_samples"] = len(data)
    scores["valid_samples"] = len(valid)

    # Print summary
    print("\n" + "=" * 40)
    print(f"{'METRIC':<25} {'SCORE':<10}")
    print("-" * 35)
    for metric, val in scores.items():
        if isinstance(val, (int, float)):
            print(f"{metric:<25}: {val:.4f}")
    print("=" * 40)

    # Save
    pd.DataFrame([scores]).to_csv(scores_file, index=False)
    print(f"\n✅ Scores saved to {scores_file}")


if __name__ == "__main__":
    main()

