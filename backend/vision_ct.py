"""
XMedFusion End-to-End Report Generation Pipeline
Fine-Tuning MedGemma for CT Scans using Grid Montaging & Metadata
"""

import os
import sys
import json
import glob
import csv
import cv2
import torch
import numpy as np
import pandas as pd
import re
import random
from PIL import Image
from tqdm import tqdm
import argparse
import copy

from config import HF_TOKEN
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset as HFDataset
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText, 
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Evaluation Metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer as rs_mod
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    from langchain_ollama import ChatOllama

# ------------------------------------------------------------------
# 1. METADATA & GRID MONTAGING FOR JPEGS
# ------------------------------------------------------------------
def format_age_sex(age_str: str, sex_str: str) -> str:
    """Converts demographic codes like '036Y', 'M' -> '36-year-old Male'"""
    age = age_str.replace("Y", "").lstrip("0") if age_str else "Unknown"
    sex_map = {"M": "Male", "F": "Female"}
    sex = sex_map.get(sex_str.upper(), "Unknown sex")
    
    if age != "Unknown":
        return f"{age}-year-old {sex}"
    return f"Patient of unknown age, {sex}"

def load_metadata_dict(metadata_csv: str) -> dict:
    meta_dict = {}
    if not os.path.exists(metadata_csv):
        print(f"⚠️ Metadata file not found at {metadata_csv}. Demographics will be omitted.")
        return meta_dict
        
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vol_name = row["VolumeName"].replace(".nii.gz", "").strip()
            meta_dict[vol_name] = format_age_sex(row.get("PatientAge", ""), row.get("PatientSex", ""))
            
    return meta_dict

def load_jpeg_montage(volume_dir: str, grid_size=(2, 4), target_size=(256, 256)) -> Image.Image:
    """
    Loads N equidistant slices, resizes them, and stitches them into a single 
    grid image (e.g., 2x4 = 8 slices) to prevent VLM Out-Of-Memory errors.
    """
    jpeg_paths = sorted(glob.glob(os.path.join(volume_dir, "*.jpg")))
    if not jpeg_paths:
        raise FileNotFoundError(f"No JPEG slices found in: {volume_dir}")

    total = len(jpeg_paths)
    start = int(total * 0.15) # Skip top 15% (neck/shoulders)
    end   = int(total * 0.90) # Skip bottom 10% (lower abdomen)
    useful = jpeg_paths[start:end]

    num_images_needed = grid_size[0] * grid_size[1]
    
    # Sample slices
    actual_slices = min(num_images_needed, len(useful))
    indices = np.linspace(0, len(useful) - 1, actual_slices, dtype=int)
    
    slice_w = target_size[0] // grid_size[1]
    slice_h = target_size[1] // grid_size[0]
    
    resized_slices = []
    for idx in indices:
        img = cv2.imread(useful[idx], cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (slice_w, slice_h))
        resized_slices.append(img_resized)
        
    # Pad with blank images if the volume had too few slices
    while len(resized_slices) < num_images_needed:
        resized_slices.append(np.zeros((slice_h, slice_w, 3), dtype=np.uint8))
        
    # Stitch the grid together
    rows = []
    for i in range(grid_size[0]):
        row = np.hstack(resized_slices[i * grid_size[1] : (i + 1) * grid_size[1]])
        rows.append(row)
        
    grid_img = np.vstack(rows)
    return Image.fromarray(grid_img)

def load_and_format_dataset(csv_path: str, meta_dict: dict, jpegs_root: str, split: str = "all_data"):
    available_vols = {d for d in os.listdir(jpegs_root) if os.path.isdir(os.path.join(jpegs_root, d))}
    formatted_data = {"id": [], "image": [], "messages": [], "ground_truth": []}

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc=f"Formatting {split} dataset"):
        vol_name = row.get("VolumeName", "").replace(".nii.gz", "").strip()
        if vol_name not in available_vols:
            continue

        findings   = row.get("Findings_EN", "").strip()
        impression = row.get("Impressions_EN", "").strip()
        if not findings:
            continue

        # Filter out 90% of normal scans during training to prevent mode collapse
        if split == "train":
            lower_imp = impression.lower()
            if "within normal limits" in lower_imp or "no active infiltration" in lower_imp or impression == "":
                if random.random() > 0.10: 
                    continue

        report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        patient_demo = meta_dict.get(vol_name, "Patient demographics unknown")
        
        volume_dir = os.path.join(jpegs_root, vol_name)
        try:
            pil_img = load_jpeg_montage(volume_dir, grid_size=(4, 4))
        except FileNotFoundError:
            continue

        prompt_text = (
            "You are an expert thoracic radiologist. Analyze the attached CT scan montage "
            "(axial slices arranged in a grid, cranio-caudal order) and write a structured radiology report.\n"
            f"Clinical indication: {patient_demo}\n\n"
            "FINDINGS:\n"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": report}
                ]
            }
        ]

        formatted_data["id"].append(vol_name)
        formatted_data["image"].append(pil_img)
        formatted_data["messages"].append(messages)
        formatted_data["ground_truth"].append(report)

    print(f"  Loaded {len(formatted_data['id'])} total samples.")
    return HFDataset.from_dict(formatted_data)

# ------------------------------------------------------------------
# 2. EVALUATION SUITE
# ------------------------------------------------------------------
class LLMJudge:
    def __init__(self, model_name="gpt-oss:20b"):
        self.llm = ChatOllama(model=model_name, temperature=0.1)

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
            resp  = self.llm.invoke(prompt)
            match = re.search(r"\{.*\}", resp.content, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except Exception as e:
            print(f"  [Judge error]: {e}")
            return {}

def run_evaluation(model_id_or_path, test_dataset, processor, device="cuda", test_all=False):
    print(f"\n📊 Starting Comprehensive Evaluation on: {model_id_or_path}")

    out_json = "out/ct_grid_generations.json"
    os.makedirs("out", exist_ok=True)

    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            saved_results = json.load(f)
        done_ids = {r["id"] for r in saved_results}
        print(f"  Resuming: {len(done_ids)} / {len(test_dataset)} samples already saved.")
    else:
        saved_results = []
        done_ids = set()

    remaining_indices = [
        i for i, sid in enumerate(test_dataset["id"]) if sid not in done_ids
    ]

    # Limit to 30 samples unless test_all flag is provided
    if not test_all and len(remaining_indices) > 30:
        print(f"  --test_all not specified. Limiting evaluation to 30 samples.")
        remaining_indices = remaining_indices[:30]

    if remaining_indices:
        # Load fine-tuned model: merge LoRA into base weights to avoid device splits
        is_peft_path = os.path.isdir(model_id_or_path) and os.path.exists(
            os.path.join(model_id_or_path, "adapter_config.json")
        )
        if is_peft_path:
            from peft import PeftModel as PeftModelLoader
            print("  Loading base model + merging LoRA adapter...")
            base = AutoModelForImageTextToText.from_pretrained(
                "google/medgemma-4b-it",
                torch_dtype=torch.bfloat16,
                device_map={"":"cuda:0"},  # avoids accelerate get_balanced_memory bug
            )
            eval_model = PeftModelLoader.from_pretrained(base, model_id_or_path)
            eval_model = eval_model.merge_and_unload()
        else:
            print("  Loading base model for evaluation...")
            eval_model = AutoModelForImageTextToText.from_pretrained(
                model_id_or_path,
                torch_dtype=torch.bfloat16,
                device_map={"":"cuda:0"},
            )

        eval_model.eval()
        eval_device = next(eval_model.parameters()).device
        print(f"  Eval model on: {eval_device}")
        processor.tokenizer.padding_side = "left"

        judge = LLMJudge()
        print(f"Running inference and saving step-by-step for {len(remaining_indices)} volumes...")

        for idx in tqdm(remaining_indices, desc="Evaluating"):
            pil_img = test_dataset["image"][idx]

            # Build user-only message with image injected
            user_msg = copy.deepcopy(test_dataset["messages"][idx][0])
            for block in user_msg["content"]:
                if block["type"] == "image":
                    block["image"] = pil_img

            # Apply chat template and tokenise
            prompt_text = processor.apply_chat_template(
                [user_msg], tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=prompt_text, images=[pil_img], return_tensors="pt"
            ).to(eval_device)

            # Generate
            with torch.inference_mode():
                output_ids = eval_model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            hyp = processor.decode(generated, skip_special_tokens=True).strip()

            s_id = test_dataset["id"][idx]
            ref  = test_dataset["ground_truth"][idx]

            # LLM Judge Scoring
            j_scores = judge.evaluate_medical_accuracy(ref, hyp)

            # Save after each sample
            entry = {"id": s_id, "ref": ref, "hyp": hyp, "judge_scores": j_scores}
            saved_results.append(entry)
            with open(out_json, "w") as f:
                json.dump(saved_results, f, indent=2)

        del eval_model
        torch.cuda.empty_cache()


    print("\nComputing aggregate metrics...")
    refs_dict, hyps_dict = {}, {}
    refs_list, hyps_list = [], []
    judge_accum = {"coverage": [], "consistency": [], "accuracy": [], "style": [], "conciseness": []}

    for entry in saved_results:
        s_id = entry["id"]
        ref  = entry["ref"].lower()
        hyp  = entry["hyp"].lower()
        refs_dict[s_id] = [ref]
        hyps_dict[s_id] = [hyp]
        refs_list.append(ref)
        hyps_list.append(hyp)
        for k in judge_accum:
            if k in entry.get("judge_scores", {}):
                try:
                    judge_accum[k].append(float(entry["judge_scores"][k]))
                except (TypeError, ValueError):
                    pass

    if not refs_dict:
        print("No results to compute metrics on.")
        return

    final_scores = {}
    b_scorer = Bleu(4)
    b_score, _ = b_scorer.compute_score(refs_dict, hyps_dict)
    for i in range(4):
        final_scores[f"BLEU_{i+1}"] = b_score[i]

    c_scorer = Cider()
    final_scores["CIDEr"], _ = c_scorer.compute_score(refs_dict, hyps_dict)

    rouge_scorer = rs_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(refs_list, hyps_list):
        s = rouge_scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    final_scores.update({
        "ROUGE_1": float(np.mean(r1)),
        "ROUGE_2": float(np.mean(r2)),
        "ROUGE_L": float(np.mean(rl)),
    })

    import nltk
    for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
    meteor_scores = []
    for ref, hyp in zip(refs_list, hyps_list):
        try:
            meteor_scores.append(nltk_meteor([word_tokenize(ref)], word_tokenize(hyp)))
        except Exception:
            meteor_scores.append(0.0)
    final_scores["METEOR"] = float(np.mean(meteor_scores))

    P, R, F1 = bert_score(hyps_list, refs_list, lang="en", verbose=False)
    final_scores.update({
        "BERT_P":  P.mean().item(),
        "BERT_R":  R.mean().item(),
        "BERT_F1": F1.mean().item(),
    })

    for k, v in judge_accum.items():
        final_scores[f"Judge_{k.capitalize()}"] = float(np.mean(v)) if v else 0.0

    print("\n" + "=" * 30 + "\nFINAL RESULTS\n" + "=" * 30)
    for m, s in final_scores.items():
        print(f"{m:20}: {s:.4f}")

    out_csv = "out/comprehensive_test_scores_ct_grid.csv"
    pd.DataFrame([final_scores]).to_csv(out_csv, index=False)
    print(f"\nSaved metrics → {out_csv}")


# ------------------------------------------------------------------
# 3. TRAINING PIPELINE & COLLATOR
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run evaluation on the raw pretrained MedGemma without fine-tuning.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation on the fine-tuned model.")
    parser.add_argument("--test_all", action="store_true", help="Evaluate the entire validation subset instead of the default 30 samples.")
    args = parser.parse_args()

    model_id   = "google/medgemma-4b-it"
    output_dir = "model_weights/Vision_Agent/medgemma_ct_grid_finetuned"

    # ── HARDCODED CT-RATE PATHS ────────────────────────────────────────────
    data_root    = r"F:\XMedFusion\ct_rate"
    jpegs_root   = os.path.join(data_root, "processed_jpegs")
    val_csv      = os.path.join(data_root, "dataset", "radiology_text_reports", "validation_reports.csv")
    metadata_csv = os.path.join(data_root, "dataset", "metadata", "validation_metadata.csv")
    # ──────────────────────────────────────────────────────────────────────

    if torch.cuda.get_device_capability()[0] < 8:
        print("Warning: GPU does not officially support bfloat16 natively. Performance may be degraded.")

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    print("Loading Demographics Metadata...")
    meta_dict = load_metadata_dict(metadata_csv)

    # ── Dataset caching ────────────────────────────────────────────────────
    cache_dir = os.path.join(data_root, "cached_dataset")
    train_cache = os.path.join(cache_dir, "train")
    test_cache  = os.path.join(cache_dir, "test")

    if os.path.exists(train_cache) and os.path.exists(test_cache):
        from datasets import load_from_disk
        print("✅ Loading cached dataset from disk (skipping JPEG processing)...")
        train_dataset = load_from_disk(train_cache)
        test_dataset  = load_from_disk(test_cache)
        print(f"   Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")
    else:
        print("Building dataset from scratch (this takes ~15 min, cached afterwards)...")
        full_dataset = load_and_format_dataset(val_csv, meta_dict, jpegs_root, split="full_data")

        print("\nCreating 90/10 train/test split (seed=42)...")
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset  = split_dataset["test"]
        print(f"   Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")

        print(f"Saving dataset to {cache_dir} ...")
        train_dataset.save_to_disk(train_cache)
        test_dataset.save_to_disk(test_cache)
        print("✅ Dataset cached. Next run will load instantly.")
    # ──────────────────────────────────────────────────────────────────────

    print(f"Final Train Size: {len(train_dataset)} | Final Test Size: {len(test_dataset)}")


    if args.baseline:
        run_evaluation(model_id, test_dataset, processor, test_all=args.test_all)
        return

    if args.eval_only:
        if not os.path.exists(output_dir):
            print(f"Error: Could not find fine-tuned weights at {output_dir}")
            return
        run_evaluation(output_dir, test_dataset, processor, test_all=args.test_all)
        return


    print("\nInitializing QLoRA Model...")
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    }
    
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

    peft_config = LoraConfig(
        lora_alpha=32, # Increased to force model to respect visual features
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # Custom Collator for Single-Image Grids
    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            images.append([example["image"].convert("RGB")])
            texts.append(processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            ).strip())

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
        except KeyError:
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100 

        batch["labels"] = labels
        return batch

    # Check for existing checkpoint to resume from
    resume_from_checkpoint = False
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Found existing checkpoint at {output_dir}. Will resume training.")
        resume_from_checkpoint = True

    # Configure Trainer to evaluate and save the BEST model automatically
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,                
        per_device_train_batch_size=1,     # Must be 1 on 16GB with 4-bit + image tokens
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,     # Effective batch = 8
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        
        # Eval and Save parameters
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        learning_rate=5e-5,                # Decreased LR
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,                  # Increased warmup
        lr_scheduler_type="cosine",        # Smoother decay
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # The trainer needs to see the test set to calculate eval_loss
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    print("\n🚀 Starting Supervised Fine-Tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print(f"✅ Saving final best model to {output_dir}...")
    trainer.save_model()
    
    del model
    del trainer
    torch.cuda.empty_cache()

    run_evaluation(output_dir, test_dataset, processor, test_all=args.test_all)

if __name__ == "__main__":
    main()