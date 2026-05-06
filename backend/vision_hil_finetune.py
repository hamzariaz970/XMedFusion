"""
Lightweight HIL Vision Agent fine-tuning.

This module is imported only by the background training worker. The request path
must not preprocess scans or load model weights.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image


DEFAULT_BASE_MODEL_ID = "google/medgemma-4b-it"
DEFAULT_ACTIVE_ADAPTER_DIR = Path(os.getenv(
    "HIL_VISION_ACTIVE_ADAPTER_DIR",
    str(Path("model_weights") / "Vision_Agent" / "medgemma_ct_grid_finetuned"),
))
DEFAULT_BACKUP_ROOT = Path("model_weights") / "Vision_Agent" / "hil_backups"
DEFAULT_WORK_ROOT = Path("data") / "hil_vision_jobs"


@dataclass
class HILVisionHyperparameters:
    epochs: int = 1
    max_steps: int = 80
    learning_rate: float = 1e-5
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_samples: int = 64
    max_report_chars: int = 1800
    image_size: int = 512

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _pick_image_url(scan: dict[str, Any]) -> str:
    source_images = scan.get("source_images")
    if isinstance(source_images, list):
        for image in source_images:
            if isinstance(image, dict) and _safe_text(image.get("url")):
                return _safe_text(image.get("url"))

    original = _safe_text(scan.get("original_image_url"))
    if original:
        return next((part.strip() for part in original.split(",") if part.strip()), "")

    return _safe_text(scan.get("explainability_reference_image_url") or scan.get("heatmap_image_url"))


def build_samples_from_supabase(sb, batch_id: str, max_samples: int = 64) -> list[dict[str, Any]]:
    """Fetch batch items and build image/report training samples."""
    items_resp = (
        sb.table("hil_feedback_batch_items")
        .select("*")
        .eq("batch_id", batch_id)
        .order("item_order")
        .execute()
    )
    items = items_resp.data or []
    if not items:
        return []

    if max_samples > 0:
        items = items[:max_samples]

    feedback_ids = [item["feedback_id"] for item in items if item.get("feedback_id")]
    scan_ids = [item["scan_id"] for item in items if item.get("scan_id")]
    doctor_ids = [item["doctor_id"] for item in items if item.get("doctor_id")]
    hil_report_ids = [item["hil_report_id"] for item in items if item.get("hil_report_id")]
    hil_scan_ids = [item["hil_scan_id"] for item in items if item.get("hil_scan_id")]

    feedback_rows = {}
    if feedback_ids:
        resp = sb.table("feedback").select("*").in_("id", feedback_ids).execute()
        feedback_rows = {row["id"]: row for row in (resp.data or [])}

    scans = {}
    if scan_ids:
        resp = (
            sb.table("medical_scans")
            .select("id, patient_id, scan_type, original_image_url, source_images, heatmap_image_url, explainability_reference_image_url, created_at")
            .in_("id", scan_ids)
            .execute()
        )
        scans = {row["id"]: row for row in (resp.data or [])}

    hil_reports = {}
    if hil_report_ids:
        resp = sb.table("hil_reports").select("*").in_("id", hil_report_ids).execute()
        hil_reports = {row["id"]: row for row in (resp.data or [])}

    hil_scans = {}
    if hil_scan_ids:
        resp = sb.table("hil_scans").select("*").in_("id", hil_scan_ids).execute()
        hil_scans = {row["id"]: row for row in (resp.data or [])}

    doctors = {}
    if doctor_ids:
        resp = sb.table("doctors").select("user_id, full_name, email").in_("user_id", doctor_ids).execute()
        doctors = {row["user_id"]: row for row in (resp.data or [])}

    samples: list[dict[str, Any]] = []
    for item in items:
        doctor = doctors.get(item.get("doctor_id")) or {}
        if item.get("source_type") == "hil_report" and item.get("hil_report_id"):
            hil_report = hil_reports.get(item.get("hil_report_id"))
            hil_scan = hil_scans.get(item.get("hil_scan_id") or (hil_report or {}).get("scan_id"))
            if not hil_report or not hil_scan:
                continue
            findings = _safe_text(hil_report.get("findings"))
            impression = _safe_text(hil_report.get("impression"))
            if not hil_scan.get("image_url") or not (findings or impression):
                continue
            samples.append({
                "image_url": _safe_text(hil_scan.get("image_url")),
                "scan_type": "xray",
                "anonymous_patient_id": (hil_report.get("task_id") or "")[:8],
                "doctor_name": doctor.get("full_name") or doctor.get("email") or "Unknown",
                "original_report": "",
                "edited_report": f"FINDINGS: {findings}\nIMPRESSION: {impression}".strip(),
            })
            continue

        feedback = feedback_rows.get(item.get("feedback_id"))
        scan = scans.get(item.get("scan_id"))
        if not feedback or not scan:
            continue

        image_url = _pick_image_url(scan)
        edited_findings = _safe_text(feedback.get("edited_findings") or feedback.get("original_findings"))
        edited_impression = _safe_text(feedback.get("edited_impression") or feedback.get("original_impression"))
        if not image_url or not (edited_findings or edited_impression):
            continue

        original_report = ""
        if item.get("include_original_report", True):
            original_report = "\n".join(
                part for part in [
                    f"Original findings: {_safe_text(feedback.get('original_findings'))}",
                    f"Original impression: {_safe_text(feedback.get('original_impression'))}",
                ]
                if part and not part.endswith(":")
            ).strip()

        samples.append({
            "image_url": image_url,
            "scan_type": scan.get("scan_type") or "xray",
            "anonymous_patient_id": (scan.get("patient_id") or "")[:8],
            "doctor_name": doctor.get("full_name") or doctor.get("email") or "Unknown",
            "original_report": original_report,
            "edited_report": f"FINDINGS: {edited_findings}\nIMPRESSION: {edited_impression}".strip(),
        })

    return samples


def _download_image(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".png"
    output_path = output_dir / filename
    if output_path.exists():
        return output_path

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def _prepare_dataset(samples: list[dict[str, Any]], job_dir: Path, hp: HILVisionHyperparameters) -> Path:
    image_dir = job_dir / "images"
    records: list[dict[str, str]] = []
    for sample in samples[: hp.max_samples]:
        image_path = _download_image(sample["image_url"], image_dir)
        # Normalize images once in the worker, not in the admin request.
        with Image.open(image_path) as img:
            img.convert("RGB").resize((hp.image_size, hp.image_size)).save(image_path)

        report = _safe_text(sample["edited_report"])[: hp.max_report_chars]
        original = _safe_text(sample.get("original_report"))
        prompt = (
            "Generate a concise radiology report for this medical image. "
            f"Modality: {sample.get('scan_type', 'xray')}. "
            "Patient identity is anonymized. "
        )
        if original:
            prompt += "Use the prior AI draft only as context and correct it where needed. "
        records.append({
            "image": str(image_path),
            "prompt": prompt,
            "original_report": original[: hp.max_report_chars],
            "response": report,
        })

    dataset_path = job_dir / "train.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return dataset_path


def backup_active_adapter(active_dir: Path = DEFAULT_ACTIVE_ADAPTER_DIR, backup_root: Path = DEFAULT_BACKUP_ROOT) -> str:
    """Snapshot the current active adapter before replacing it."""
    if not active_dir.exists():
        return ""
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"hil_medgemma_active_{stamp}"
    shutil.copytree(active_dir, backup_dir)
    return str(backup_dir)


def run_hil_vision_finetune(
    samples: list[dict[str, Any]],
    job_id: str,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a conservative 16GB-friendly QLoRA adaptation for the Vision Agent.

    Hyperparameters are intentionally light:
    - 4-bit base model loading
    - LoRA rank 4 / alpha 8
    - batch size 1 with gradient accumulation 8
    - one epoch and max 80 optimizer steps
    """
    hp = HILVisionHyperparameters(**{**HILVisionHyperparameters().as_dict(), **(hyperparameters or {})})
    if len(samples) < 3:
        return {"error": f"Need at least 3 HIL samples for Vision fine-tuning. Got {len(samples)}.", "num_samples": len(samples)}

    job_dir = DEFAULT_WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = _prepare_dataset(samples, job_dir, hp)
    output_dir = job_dir / "adapter"

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        return {
            "error": f"Vision fine-tuning dependencies are missing: {exc}",
            "num_samples": len(samples),
            "dataset_path": str(dataset_path),
        }

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL_ID, token=os.getenv("HF_TOKEN"))
    model = AutoModelForImageTextToText.from_pretrained(
        DEFAULT_BASE_MODEL_ID,
        token=os.getenv("HF_TOKEN"),
        quantization_config=quantization_config if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=hp.lora_r,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        for row in batch:
            image = Image.open(row["image"]).convert("RGB")
            images.append(image)
            prompt = row["prompt"]
            if row.get("original_report"):
                prompt += f"\nPrior AI draft:\n{row['original_report']}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{prompt}\nCorrected radiology report:"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": row["response"]}]},
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

        encoded = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = encoded["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        encoded["labels"] = labels
        return encoded

    max_steps = min(hp.max_steps, max(1, len(dataset) * hp.epochs))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        learning_rate=hp.learning_rate,
        num_train_epochs=hp.epochs,
        max_steps=max_steps,
        warmup_ratio=0.05,
        weight_decay=0.0,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=0.3,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate)
    train_result = trainer.train()

    backup_dir = backup_active_adapter()
    if DEFAULT_ACTIVE_ADAPTER_DIR.exists():
        shutil.rmtree(DEFAULT_ACTIVE_ADAPTER_DIR)
    DEFAULT_ACTIVE_ADAPTER_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DEFAULT_ACTIVE_ADAPTER_DIR)
    processor.save_pretrained(DEFAULT_ACTIVE_ADAPTER_DIR)

    return {
        "status": "complete",
        "num_samples": len(dataset),
        "train_loss": float(train_result.training_loss),
        "adapter_output_dir": str(DEFAULT_ACTIVE_ADAPTER_DIR),
        "backup_dir": backup_dir,
        "hyperparameters": hp.as_dict(),
    }
