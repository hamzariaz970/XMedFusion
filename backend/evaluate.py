import argparse
import asyncio
import json
import os
import re
from typing import Dict, List

import gc
import numpy as np
import pandas as pd
import torch
from langchain_community.chat_models import ChatOllama
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from tqdm.asyncio import tqdm

import config
from draft import LocalLLMReportAgent, RetrievalAgent, reports_dict
from report_labels import DISEASES, case_profile, definite_positive_vector, label_audit_from_report
from synthesis import LocalSynthesisAgent
from vision import VisualDescriptionAgent, vision_encoder


ANNOTATIONS_PATH = "data/iu_xray/annotation.json"
IMAGES_ROOT = "data/iu_xray/images"
DEFAULT_OUTPUT_FILE = "out/test_generations_judge.json"
DEFAULT_SCORES_FILE = "out/test_scores_judge.csv"
DEFAULT_KG_SCORES_FILE = "out/test_kg_scores_judge.json"
DEFAULT_TIMEOUT_SECONDS = 120
STATUS_RANK = {"not_mentioned": 0, "absent": 1, "uncertain": 2, "present": 3}
KG_TYPE_TO_STATUS = {
    "Observation": "present",
    "AbsentObservation": "absent",
    "UncertainObservation": "uncertain",
}
KG_TEXT_TO_DISEASE = {disease.lower(): disease for disease in DISEASES}


class ReportFormatter:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def format_to_ground_truth_style(self, raw_report: str) -> str:
        """Normalize report text deterministically for lexical metrics.

        The previous evaluator used an LLM rewrite here. That made BLEU/ROUGE
        depend on a second generation step and could erase useful IU-style
        phrasing from the actual pipeline output.
        """
        text = raw_report or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(
            r"\b(?:RECOMMENDATIONS?|LABELS?)\s*:\s*.*$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"\b(?:FINDINGS?|IMPRESSION)\s*:\s*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bXXXX\b", "visualized structures", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or raw_report


class LLMJudge:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def evaluate_medical_accuracy(self, reference: str, hypothesis: str) -> Dict[str, int]:
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
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"coverage": 5, "consistency": 5, "accuracy": 5, "style": 5, "conciseness": 5}


class MedicalReportEvaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

    def compute_scores(self, ref: Dict[str, List[str]], hypo: Dict[str, List[str]]) -> Dict[str, float]:
        score_results = {}
        for scorer, method in self.scorers:
            try:
                score, _ = scorer.compute_score(ref, hypo)
                if isinstance(method, list):
                    for metric_name, metric_score in zip(method, score):
                        score_results[metric_name] = metric_score
                else:
                    score_results[method] = score
            except Exception:
                pass
        return score_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the XMedFusion synthesis pipeline on IU X-ray test samples.")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to the first N test samples.")
    parser.add_argument(
        "--sample-mode",
        choices=["head", "stratified", "hard_positive", "hard_multifinding", "rare_positive", "hard_stratified"],
        default="stratified",
        help="Subset selection strategy when --limit is used.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["inclusive", "definite"],
        default="inclusive",
        help="Whether ground-truth labels should count uncertain mentions or only definite positives.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file instead of starting fresh.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="JSON file for per-sample generations and metadata.")
    parser.add_argument("--scores-file", default=DEFAULT_SCORES_FILE, help="CSV file for aggregate text metrics.")
    parser.add_argument("--kg-scores-file", default=DEFAULT_KG_SCORES_FILE, help="JSON file for aggregate KG metrics.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Per-sample synthesis timeout.")
    return parser.parse_args()


def sanitize_text(text: str) -> str:
    return (text or "").replace("\n", " ").strip().lower()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def extract_full_image_paths(example: Dict) -> List[str]:
    rel_paths = example.get("image_path", [])
    if isinstance(rel_paths, str):
        rel_paths = [rel_paths]
    full_paths = []
    for rel_path in rel_paths:
        candidate = os.path.join(IMAGES_ROOT, rel_path)
        if os.path.exists(candidate):
            full_paths.append(candidate)
    return full_paths


def label_array_to_dict(labels: np.ndarray) -> Dict[str, int]:
    return {
        disease: int(labels[idx])
        for idx, disease in enumerate(DISEASES)
    }


def extract_ground_truth_labels(report: str, label_mode: str = "inclusive") -> np.ndarray:
    if label_mode == "definite":
        return definite_positive_vector(report).astype(np.int32)
    labels, _ = label_audit_from_report(report)
    return labels.astype(np.int32)


def select_test_subset(test_data_raw: List[Dict], limit: int | None, sample_mode: str, label_mode: str = "inclusive") -> List[Dict]:
    if limit is None or limit >= len(test_data_raw):
        if sample_mode in {"hard_positive", "hard_multifinding", "rare_positive", "hard_stratified"}:
            limit = len(test_data_raw)
        else:
            return test_data_raw
    if sample_mode == "head":
        return test_data_raw[:limit]

    enriched = []
    for example in test_data_raw:
        profile = case_profile(example.get("report", ""))
        labels = extract_ground_truth_labels(example.get("report", ""), label_mode)
        enriched.append({
            "example": example,
            "labels": labels,
            "positive_count": int(labels.sum()),
            "definite_positive_count": int(profile["definite_positive_count"]),
            "rare_positive_count": int(profile["rare_positive_count"]),
        })

    if sample_mode == "hard_positive":
        positives = [item["example"] for item in enriched if item["positive_count"] > 0]
        return positives[:limit]

    if sample_mode == "hard_multifinding":
        positives = [item for item in enriched if item["positive_count"] >= 2]
        positives.sort(key=lambda item: (item["positive_count"], item["rare_positive_count"]), reverse=True)
        return [item["example"] for item in positives[:limit]]

    if sample_mode == "rare_positive":
        rare = [item for item in enriched if item["rare_positive_count"] > 0]
        rare.sort(key=lambda item: (item["rare_positive_count"], item["positive_count"]), reverse=True)
        return [item["example"] for item in rare[:limit]]

    if sample_mode == "hard_stratified":
        enriched = [item for item in enriched if item["positive_count"] > 0]
        if not enriched:
            return []

    selected = []
    selected_ids = set()
    uncovered = {
        DISEASES[idx]
        for idx in range(len(DISEASES))
        if any(item["labels"][idx] == 1 for item in enriched)
    }

    while len(selected) < limit and uncovered:
        best = None
        best_gain = -1
        for item in enriched:
            sample_id = str(item["example"].get("id"))
            if sample_id in selected_ids:
                continue
            gain = sum(
                1
                for idx, disease in enumerate(DISEASES)
                if disease in uncovered and item["labels"][idx] == 1
            )
            if gain > best_gain or (gain == best_gain and best is not None and item["positive_count"] > best["positive_count"]):
                best = item
                best_gain = gain
        if best is None or best_gain <= 0:
            break
        selected.append(best["example"])
        selected_ids.add(str(best["example"].get("id")))
        for idx, disease in enumerate(DISEASES):
            if best["labels"][idx] == 1 and disease in uncovered:
                uncovered.remove(disease)

    remaining_positive = [
        item for item in enriched
        if str(item["example"].get("id")) not in selected_ids and item["positive_count"] > 0
    ]
    remaining_positive.sort(key=lambda item: item["positive_count"], reverse=True)
    for item in remaining_positive:
        if len(selected) >= limit:
            break
        selected.append(item["example"])
        selected_ids.add(str(item["example"].get("id")))

    if sample_mode not in {"hard_positive", "hard_multifinding", "rare_positive", "hard_stratified"}:
        remaining_negative = [
            item for item in enriched
            if str(item["example"].get("id")) not in selected_ids
        ]
        for item in remaining_negative:
            if len(selected) >= limit:
                break
            selected.append(item["example"])

    return selected[:limit]


def extract_kg_label_statuses(kg_data: Dict | None) -> Dict[str, str]:
    statuses = {disease: "not_mentioned" for disease in DISEASES}
    if not kg_data:
        return statuses

    entities = kg_data.get("entities", [])
    for entity in entities:
        if len(entity) < 2:
            continue
        text, label = entity[0], entity[1]
        disease = KG_TEXT_TO_DISEASE.get(str(text).strip().lower())
        status = KG_TYPE_TO_STATUS.get(str(label))
        if not disease or not status:
            continue
        if STATUS_RANK[status] >= STATUS_RANK[statuses[disease]]:
            statuses[disease] = status
    return statuses


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int | None]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max(1, len(y_true))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def average_metric(values: List[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def compute_kg_metrics(valid_logs: List[Dict]) -> Dict:
    if not valid_logs:
        return {"valid_samples": 0, "error": "No valid generations found."}

    gt_matrix = []
    pred_present_matrix = []
    pred_support_matrix = []
    per_label = {}

    for disease in DISEASES:
        gt = np.array([int(item["ground_truth_labels"][disease]) for item in valid_logs], dtype=np.int32)
        pred_status = [item["kg_label_statuses"].get(disease, "not_mentioned") for item in valid_logs]
        pred_present = np.array([1 if status == "present" else 0 for status in pred_status], dtype=np.int32)
        pred_support = np.array([1 if status in {"present", "uncertain"} else 0 for status in pred_status], dtype=np.int32)

        gt_positive_breakdown = {
            status: int(sum(1 for status_item, truth in zip(pred_status, gt) if truth == 1 and status_item == status))
            for status in ["present", "uncertain", "absent", "not_mentioned"]
        }
        gt_negative_breakdown = {
            status: int(sum(1 for status_item, truth in zip(pred_status, gt) if truth == 0 and status_item == status))
            for status in ["present", "uncertain", "absent", "not_mentioned"]
        }

        strict_metrics = compute_binary_metrics(gt, pred_present)
        support_metrics = compute_binary_metrics(gt, pred_support)

        per_label[disease] = {
            "support": int(gt.sum()),
            "strict_present_metrics": strict_metrics,
            "present_or_uncertain_metrics": support_metrics,
            "status_breakdown": {
                "gt_positive": gt_positive_breakdown,
                "gt_negative": gt_negative_breakdown,
            },
        }

        gt_matrix.append(gt)
        pred_present_matrix.append(pred_present)
        pred_support_matrix.append(pred_support)

    gt_matrix = np.column_stack(gt_matrix)
    pred_present_matrix = np.column_stack(pred_present_matrix)
    pred_support_matrix = np.column_stack(pred_support_matrix)

    strict_macro_precision = average_metric([per_label[d]["strict_present_metrics"]["precision"] for d in DISEASES])
    strict_macro_recall = average_metric([per_label[d]["strict_present_metrics"]["recall"] for d in DISEASES])
    strict_macro_f1 = average_metric([per_label[d]["strict_present_metrics"]["f1"] for d in DISEASES])
    support_macro_precision = average_metric([per_label[d]["present_or_uncertain_metrics"]["precision"] for d in DISEASES])
    support_macro_recall = average_metric([per_label[d]["present_or_uncertain_metrics"]["recall"] for d in DISEASES])
    support_macro_f1 = average_metric([per_label[d]["present_or_uncertain_metrics"]["f1"] for d in DISEASES])

    strict_micro = compute_binary_metrics(gt_matrix.reshape(-1), pred_present_matrix.reshape(-1))
    support_micro = compute_binary_metrics(gt_matrix.reshape(-1), pred_support_matrix.reshape(-1))

    return {
        "valid_samples": len(valid_logs),
        "strict_present_macro": {
            "precision": strict_macro_precision,
            "recall": strict_macro_recall,
            "f1": strict_macro_f1,
        },
        "strict_present_micro": strict_micro,
        "present_or_uncertain_macro": {
            "precision": support_macro_precision,
            "recall": support_macro_recall,
            "f1": support_macro_f1,
        },
        "present_or_uncertain_micro": support_micro,
        "per_label": per_label,
    }


def write_kg_scores(path: str, kg_scores: Dict) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kg_scores, f, indent=2)


async def run_evaluation(args) -> None:
    results_log = []
    processed_ids = set()

    ensure_parent_dir(args.output_file)
    ensure_parent_dir(args.scores_file)
    ensure_parent_dir(args.kg_scores_file)

    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from existing results file: {args.output_file}")
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                results_log = json.load(f)
            processed_ids = {str(item["id"]) for item in results_log}
            print(f"Loaded {len(results_log)} previous samples.")
        except Exception as exc:
            print(f"Could not load previous results, starting fresh: {exc}")
            results_log = []
            processed_ids = set()
    elif os.path.exists(args.output_file):
        print(f"Fresh run requested; ignoring existing file: {args.output_file}")

    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"Annotation file not found: {ANNOTATIONS_PATH}")
        return

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_data_raw = data.get("test", []) if isinstance(data, dict) else [x for x in data if x.get("split") == "test"]
    if args.limit is not None:
        print(f"Running partial evaluation on {args.limit} test studies with sample mode '{args.sample_mode}'.")
        test_data_raw = select_test_subset(test_data_raw, args.limit, args.sample_mode, args.label_mode)
    else:
        test_data_raw = select_test_subset(test_data_raw, args.limit, args.sample_mode, args.label_mode)
        print(f"Running full test evaluation on {len(test_data_raw)} studies.")

    if len(processed_ids) >= len(test_data_raw):
        print("All requested samples already processed; moving to scoring.")
    else:
        print("Initializing synthesis pipeline agents...")
        retrieval_label_weight = float(os.getenv("RETRIEVAL_LABEL_WEIGHT", "0.0"))
        retrieval_agent = RetrievalAgent(vision_encoder, k=3, label_weight=retrieval_label_weight)
        draft_agent = LocalLLMReportAgent()
        vision_agent = VisualDescriptionAgent()
        synthesis_agent = LocalSynthesisAgent()

        print(f"Initializing judge model: {config.OLLAMA_JUDGE_MODEL}")
        judge_llm = ChatOllama(
            model=config.OLLAMA_JUDGE_MODEL,
            temperature=config.TEMPERATURE,
            num_ctx=config.CONTEXT_WINDOW,
        )
        judge = LLMJudge(judge_llm)
        formatter = ReportFormatter(draft_agent)

        for idx, example in enumerate(tqdm(test_data_raw)):
            sample_id = str(example.get("id", idx))
            if sample_id in processed_ids:
                continue

            image_paths = extract_full_image_paths(example)
            if not image_paths:
                continue

            ground_truth_report = example.get("report", "")
            ground_truth_labels = extract_ground_truth_labels(ground_truth_report, args.label_mode)

            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                async def _generate_with_timeout():
                    complete_payload = None
                    accumulated = ""
                    gen = synthesis_agent.generate_final_report(
                        draft_agent=draft_agent,
                        vision_agent=vision_agent,
                        retrieval_agent=retrieval_agent,
                        reports_dict=reports_dict,
                        image_paths=image_paths,
                        scan_type="xray",
                    )
                    async for chunk in gen:
                        data_chunk = json.loads(chunk.strip())
                        if data_chunk.get("status") == "streaming" and "chunk" in data_chunk:
                            accumulated += data_chunk["chunk"]
                        if data_chunk.get("status") == "complete":
                            complete_payload = data_chunk
                    if complete_payload is None and accumulated:
                        complete_payload = {"final_report": accumulated}
                    return complete_payload

                try:
                    payload = await asyncio.wait_for(_generate_with_timeout(), timeout=args.timeout_seconds)
                except asyncio.TimeoutError:
                    print(f"Timeout on sample {sample_id}; recording error.")
                    results_log.append({
                        "id": sample_id,
                        "image_paths": image_paths,
                        "raw_generated": "TIMEOUT_ERROR",
                        "final_formatted": "error",
                        "ground_truth": sanitize_text(ground_truth_report),
                        "ground_truth_labels": label_array_to_dict(ground_truth_labels),
                        "judge_scores": {},
                        "knowledge_graph": {},
                        "kg_label_statuses": {},
                    })
                    with open(args.output_file, "w", encoding="utf-8") as f:
                        json.dump(results_log, f, indent=2)
                    continue

                if not payload or not payload.get("final_report"):
                    raise RuntimeError("No final report returned by synthesis pipeline.")

                raw_report = payload.get("final_report", "")
                knowledge_graph = payload.get("knowledge_graph") or {}
                kg_label_statuses = extract_kg_label_statuses(knowledge_graph)
                formatted_hypothesis = await asyncio.to_thread(formatter.format_to_ground_truth_style, raw_report)
                clean_reference = sanitize_text(ground_truth_report)
                clean_hypothesis = sanitize_text(formatted_hypothesis) or "no report generated"
                judge_scores = await asyncio.to_thread(judge.evaluate_medical_accuracy, clean_reference, clean_hypothesis)

                results_log.append({
                    "id": sample_id,
                    "image_paths": image_paths,
                    "raw_generated": raw_report,
                    "final_formatted": clean_hypothesis,
                    "ground_truth": clean_reference,
                    "ground_truth_labels": label_array_to_dict(ground_truth_labels),
                    "judge_scores": judge_scores,
                    "knowledge_graph": knowledge_graph,
                    "kg_label_statuses": kg_label_statuses,
                    "detected_modality": payload.get("detected_modality"),
                })
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(results_log, f, indent=2)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                print(f"Error on sample {sample_id}: {exc}")
                continue

    print("\n" + "=" * 60)
    print("CALCULATING AGGREGATE METRICS")
    print("=" * 60)

    if not results_log:
        print("No results found.")
        return

    valid_logs = [item for item in results_log if item.get("final_formatted") != "error"]
    if not valid_logs:
        print("No valid generations found.")
        return

    references = {item["id"]: [item["ground_truth"]] for item in valid_logs}
    hypotheses = {item["id"]: [item["final_formatted"]] for item in valid_logs}

    evaluator = MedicalReportEvaluator()
    text_scores = evaluator.compute_scores(references, hypotheses)

    judge_keys = ["coverage", "consistency", "accuracy", "style", "conciseness"]
    for key in judge_keys:
        values = [item["judge_scores"].get(key, 0) for item in valid_logs if "judge_scores" in item]
        text_scores[key] = float(np.mean(values)) if values else 0.0
    text_scores["total_samples"] = len(results_log)
    text_scores["valid_samples"] = len(valid_logs)
    text_scores["requested_limit"] = args.limit
    text_scores["sample_mode"] = args.sample_mode

    kg_scores = compute_kg_metrics(valid_logs)
    kg_scores["total_samples"] = len(results_log)
    kg_scores["requested_limit"] = args.limit
    kg_scores["sample_mode"] = args.sample_mode

    print(f"{'TEXT METRIC':<25} {'SCORE':<10}")
    print("-" * 40)
    for metric_name, metric_value in text_scores.items():
        if isinstance(metric_value, (int, float)):
            print(f"{metric_name:<25}: {metric_value:.4f}")

    strict_micro = kg_scores.get("strict_present_micro", {})
    support_micro = kg_scores.get("present_or_uncertain_micro", {})
    print("-" * 40)
    print("KG METRICS")
    print(f"{'Strict present P/R/F1':<25}: {strict_micro.get('precision')} / {strict_micro.get('recall')} / {strict_micro.get('f1')}")
    print(f"{'Present+uncertain P/R/F1':<25}: {support_micro.get('precision')} / {support_micro.get('recall')} / {support_micro.get('f1')}")

    pd.DataFrame([text_scores]).to_csv(args.scores_file, index=False)
    write_kg_scores(args.kg_scores_file, kg_scores)

    print(f"\nSaved text metrics to {args.scores_file}")
    print(f"Saved KG metrics to {args.kg_scores_file}")
    print(f"Saved per-sample generations to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(run_evaluation(parse_args()))
