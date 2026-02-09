import json
import os
import pandas as pd
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Configuration
INPUT_FILE = "out/test_generations_judge.json"
CLEANED_FILE = "out/test_generations_cleaned.json"
SCORES_FILE = "out/test_scores_cleaned.csv"

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
        
        # 1. COCO Metrics (BLEU, METEOR, ROUGE-L, CIDEr)
        for scorer, method in self.scorers:
            try:
                score, scores = scorer.compute_score(ref, hypo)
                if isinstance(method, list):
                    for m, s in zip(method, score): score_results[m] = s
                else:
                    score_results[method] = score
            except Exception as e: 
                print(f"Warning: Scorer {method} failed: {e}")

        # 2. ROUGE-1, ROUGE-2, ROUGE-L (Detailed via rouge_score)
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            r1, r2, rl = [], [], []
            
            for id_ in ref:
                # ref[id] is a list, typically 1 element for medical reports
                r = ref[id_][0]
                h = hypo[id_][0]
                scores = scorer.score(r, h)
                r1.append(scores['rouge1'].fmeasure)
                r2.append(scores['rouge2'].fmeasure)
                rl.append(scores['rougeL'].fmeasure)

            score_results['ROUGE-1'] = np.mean(r1)
            score_results['ROUGE-2'] = np.mean(r2)
            score_results['ROUGE-L (Detailed)'] = np.mean(rl)
        except ImportError:
            print("Warning: rouge-score not installed (pip install rouge-score). Skipping detailed ROUGE.")
        except Exception as e:
            print(f"Error calculating ROUGE-1/2: {e}")

        # 3. BERTScore
        try:
            from bert_score import score as bert_score_func
            print("Calculating BERTScore (this may take a moment)...")
            
            # Convert dicts to lists aligned by ID
            ids = list(ref.keys())
            refs_list = [ref[i][0] for i in ids]
            hyps_list = [hypo[i][0] for i in ids]
            
            P, R, F1 = bert_score_func(hyps_list, refs_list, lang="en", verbose=True, device=None) # device=None extracts from current env or cpu
            score_results['BERTScore'] = F1.mean().item()
        except ImportError:
            print("Warning: bert-score not installed (pip install bert-score). Skipping BERTScore.")
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")

        return score_results

import argparse

def main():
    parser = argparse.ArgumentParser(description="Calculate Metrics for Medical Reports")
    parser.add_argument("input_file", nargs="?", default=INPUT_FILE, help="Path to input JSON file")
    args = parser.parse_args()
    
    current_input = args.input_file
    cleaned_file = current_input.replace(".json", "_cleaned.json")
    scores_file = current_input.replace(".json", "_scores.csv")

    if not os.path.exists(current_input):
        print(f"Error: {current_input} not found.")
        return

    print(f"Reading {current_input}...")
    try:
        with open(current_input, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: JSON file is likely being written to or is corrupt. Try again later.")
        return

    print(f"Original Count: {len(data)}")

    # Filter out TIMEOUT errors
    cleaned_data = []
    skipped_count = 0
    
    for item in data:
        raw_gen = item.get("raw_generated", "")
        formatted = item.get("final_formatted", "")
        
        is_error = "TIMEOUT_ERROR" in raw_gen or formatted == "error"
        
        if is_error:
            skipped_count += 1
        else:
            cleaned_data.append(item)

    print(f"Skipped (Timeout/Error): {skipped_count}")
    print(f"Valid Samples: {len(cleaned_data)}")

    # Save cleaned data
    with open(cleaned_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    print(f"Saved cleaned data to {cleaned_file}")

    if not cleaned_data:
        print("No valid samples to evaluate.")
        return

    # Compute Metrics
    print("Calculating Metrics...")
    
    # Prepare dictionaries
    references = {item['id']: [item['ground_truth']] for item in cleaned_data}
    hypotheses = {item['id']: [item['final_formatted']] for item in cleaned_data}

    evaluator = MedicalReportEvaluator()
    scores = evaluator.compute_scores(references, hypotheses)
    
    # Process Judge Scores (Handle both legacy scalar and new dict)
    judge_data = {
        "coverage": [],
        "consistency": [],
        "accuracy": [],
        "style": [],
        "conciseness": [],
        "legacy_avg": []
    }
    
    for item in cleaned_data:
        # New Format (Dict)
        if "judge_scores" in item and isinstance(item["judge_scores"], dict):
            js = item["judge_scores"]
            for k in judge_data.keys():
                if k in js: judge_data[k].append(js[k])
        
        # Legacy/Fallback (Scalar)
        elif "judge_score" in item:
            val = item["judge_score"]
            judge_data["legacy_avg"].append(val)

    # Calculate Averages and Add to Scores
    print("-" * 30)
    print("LLM JUDGE (1-10):")
    
    # New Dimensions
    has_new_scores = False
    for k in ["coverage", "consistency", "accuracy", "style", "conciseness"]:
        vals = judge_data[k]
        if vals:
            avg = np.mean(vals)
            scores[f"Judge_{k.capitalize()}"] = avg
            print(f"{k.capitalize():<15}: {avg:.4f}")
            has_new_scores = True
            
    # Legacy Average
    if judge_data["legacy_avg"]:
        avg = np.mean(judge_data["legacy_avg"])
        if not has_new_scores:
             scores["LLM_Judge_Avg"] = avg
             print(f"Average (Legacy): {avg:.4f}")

    # General Counts
    scores["Count"] = len(cleaned_data)

    print("\n" + "="*30)
    print(f"{'METRIC':<25} {'SCORE':<10}")
    print("-" * 35)
    for metric, score in scores.items():
        if isinstance(score, (int, float)):
             print(f"{metric:<25}: {score:.4f}")
    print("="*30)

    # Save Scores
    pd.DataFrame([scores]).to_csv(scores_file, index=False)
    print(f"Scores saved to {scores_file}")

if __name__ == "__main__":
    main()
