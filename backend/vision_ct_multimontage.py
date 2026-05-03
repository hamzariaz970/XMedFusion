"""
Experimental CT-RATE multi-montage pipeline.

This file is intentionally separate from the production CT route. It is meant
for experiments where an entire CT study is represented as a sequence of
montage pages instead of a single 4x4 image.

Why this exists:
- Local CT-RATE reports in backend/data/ct_rate are long-form and detailed.
- The current production CT path uses one 4x4 montage and therefore only sees
  a small subset of slices.
- This module keeps the report style closer to CT-RATE and can be used as the
  basis for fine-tuning on multi-image MedGemma inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from config import HF_TOKEN

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_DATA_ROOT = Path(r"F:\XMedFusion\ct_rate")
FALLBACK_DATA_ROOT = Path(__file__).resolve().parent / "data" / "ct_rate"
DEFAULT_GRID_SIZE = (4, 4)
DEFAULT_PAGE_SIZE = (1024, 1024)
DEFAULT_MODEL_ID = "google/medgemma-4b-it"
DEFAULT_OUTPUT_DIR = Path("model_weights") / "Vision_Agent" / "medgemma_ct_multimontage_fullvolume_finetuned"
DEFAULT_TRAIN_PAGE_SIZE = (512, 512)
DEFAULT_EVAL_PAGE_SIZE = (768, 768)


def discover_ct_rate_root(explicit_root: str | os.PathLike[str] | None = None) -> Path:
    if explicit_root:
        root = Path(explicit_root)
    else:
        root = DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else FALLBACK_DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(f"CT-RATE root not found: {root}")
    return root


def format_age_sex(age_str: str, sex_str: str) -> str:
    age = age_str.replace("Y", "").lstrip("0") if age_str else "Unknown"
    sex_map = {"M": "Male", "F": "Female"}
    sex = sex_map.get((sex_str or "").upper(), "Unknown sex")
    if age != "Unknown":
        return f"{age}-year-old {sex}"
    return f"Patient of unknown age, {sex}"


def load_metadata_dict(metadata_csv: str | os.PathLike[str]) -> dict[str, str]:
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        return {}

    meta_dict: dict[str, str] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            vol_name = row["VolumeName"].replace(".nii.gz", "").strip()
            meta_dict[vol_name] = format_age_sex(row.get("PatientAge", ""), row.get("PatientSex", ""))
    return meta_dict


def load_report_examples(
    reports_csv: str | os.PathLike[str],
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(reports_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "VolumeName": row.get("VolumeName", ""),
                    "Findings_EN": row.get("Findings_EN", "").strip(),
                    "Impressions_EN": row.get("Impressions_EN", "").strip(),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def _sorted_slice_paths(volume_dir: str | os.PathLike[str]) -> list[Path]:
    volume_path = Path(volume_dir)
    slice_paths = sorted(volume_path.glob("*.jpg"))
    if not slice_paths:
        raise FileNotFoundError(f"No JPEG slices found in {volume_path}")
    return slice_paths


def _volume_page_metadata_from_paths(
    slice_paths: Sequence[Path],
    *,
    grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
    target_size: tuple[int, int] = DEFAULT_PAGE_SIZE,
    max_montages: int | None = None,
) -> dict[str, Any]:
    rows, cols = grid_size
    slices_per_page = rows * cols
    total_pages = math.ceil(len(slice_paths) / slices_per_page)

    page_indices = list(range(total_pages))
    if max_montages is not None:
        page_indices = _page_selection_indices(total_pages, max_montages)

    page_metadata: list[dict[str, Any]] = []
    for selected_page_number, chunk_idx in enumerate(page_indices, start=1):
        start = chunk_idx * slices_per_page
        chunk = slice_paths[start : start + slices_per_page]
        cell_metadata = []
        for cell_idx in range(rows * cols):
            col = cell_idx % cols
            row = cell_idx // cols
            if cell_idx < len(chunk):
                source_path = chunk[cell_idx]
                source_filename = source_path.name
                source_order_index = start + cell_idx + 1
            else:
                source_filename = None
                source_order_index = None
            cell_metadata.append(
                {
                    "cell_index": cell_idx + 1,
                    "row": row,
                    "col": col,
                    "source_filename": source_filename,
                    "source_order_index": source_order_index,
                }
            )

        page_metadata.append(
            {
                "rows": rows,
                "cols": cols,
                "tile_width": target_size[0] // cols,
                "tile_height": target_size[1] // rows,
                "slice_cells": cell_metadata,
                "slice_start": start + 1 if chunk else None,
                "slice_end": start + len(chunk) if chunk else None,
                "slice_count": len(chunk),
                "page_index": selected_page_number,
                "source_page_index": chunk_idx + 1,
            }
        )

    return {
        "source_image_count": len(slice_paths),
        "selected_page_count": len(page_metadata),
        "total_possible_pages": total_pages,
        "grid_size": {"rows": rows, "cols": cols},
        "target_size": {"width": target_size[0], "height": target_size[1]},
        "pages": page_metadata,
    }


def _page_selection_indices(total_pages: int, max_pages: int) -> list[int]:
    if total_pages <= max_pages:
        return list(range(total_pages))
    if max_pages <= 1:
        return [total_pages // 2]
    positions = [round(i * (total_pages - 1) / (max_pages - 1)) for i in range(max_pages)]
    return sorted(dict.fromkeys(positions))


def _build_montage_page(
    slice_paths: Sequence[Path],
    *,
    grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
    target_size: tuple[int, int] = DEFAULT_PAGE_SIZE,
    global_start_index: int = 0,
) -> tuple[Image.Image, dict[str, Any]]:
    rows, cols = grid_size
    tile_w = target_size[0] // cols
    tile_h = target_size[1] // rows
    page = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    cell_metadata = []

    for cell_idx in range(rows * cols):
        col = cell_idx % cols
        row = cell_idx // cols
        x = col * tile_w
        y = row * tile_h

        if cell_idx < len(slice_paths):
            source_path = slice_paths[cell_idx]
            with Image.open(source_path) as img:
                img_rgb = img.convert("RGB").resize((tile_w, tile_h))
            page.paste(img_rgb, (x, y))
            cell_metadata.append(
                {
                    "cell_index": cell_idx + 1,
                    "row": row,
                    "col": col,
                    "source_filename": source_path.name,
                    "source_order_index": global_start_index + cell_idx + 1,
                }
            )
        else:
            cell_metadata.append(
                {
                    "cell_index": cell_idx + 1,
                    "row": row,
                    "col": col,
                    "source_filename": None,
                    "source_order_index": None,
                }
            )

    metadata = {
        "rows": rows,
        "cols": cols,
        "tile_width": tile_w,
        "tile_height": tile_h,
        "slice_cells": cell_metadata,
        "slice_start": global_start_index + 1 if slice_paths else None,
        "slice_end": global_start_index + len(slice_paths) if slice_paths else None,
        "slice_count": len(slice_paths),
    }
    return page, metadata


def build_volume_montage_pages(
    volume_dir: str | os.PathLike[str],
    *,
    grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
    target_size: tuple[int, int] = DEFAULT_PAGE_SIZE,
    max_montages: int | None = None,
) -> tuple[list[Image.Image], dict[str, Any]]:
    slice_paths = _sorted_slice_paths(volume_dir)
    rows, cols = grid_size
    slices_per_page = rows * cols

    all_chunks = [
        slice_paths[start : start + slices_per_page]
        for start in range(0, len(slice_paths), slices_per_page)
    ]
    total_pages = len(all_chunks)

    page_indices = list(range(total_pages))
    if max_montages is not None:
        page_indices = _page_selection_indices(total_pages, max_montages)

    pages: list[Image.Image] = []
    page_metadata: list[dict[str, Any]] = []
    for selected_page_number, chunk_idx in enumerate(page_indices, start=1):
        chunk = all_chunks[chunk_idx]
        global_start = chunk_idx * slices_per_page
        page_img, meta = _build_montage_page(
            chunk,
            grid_size=grid_size,
            target_size=target_size,
            global_start_index=global_start,
        )
        meta["page_index"] = selected_page_number
        meta["source_page_index"] = chunk_idx + 1
        pages.append(page_img)
        page_metadata.append(meta)

    metadata = _volume_page_metadata_from_paths(
        slice_paths,
        grid_size=grid_size,
        target_size=target_size,
        max_montages=max_montages,
    )
    return pages, metadata


def build_ct_rate_style_prompt(
    *,
    patient_demo: str,
    page_metadata: Sequence[dict[str, Any]],
) -> str:
    page_lines = []
    for page in page_metadata:
        page_lines.append(
            f"- Page {page['page_index']} covers slices {page['slice_start']} to {page['slice_end']}."
        )

    pages_text = "\n".join(page_lines)
    return (
        "You are an expert thoracic radiologist analyzing a CT-RATE style chest CT study.\n"
        "The study is provided as multiple montage pages in cranio-caudal order. Each page contains axial slices.\n"
        f"Clinical indication: {patient_demo}\n\n"
        "Montage coverage:\n"
        f"{pages_text}\n\n"
        "Write a detailed structured report that matches CT-RATE style.\n"
        "Requirements:\n"
        "1. Include a comprehensive FINDINGS section with complete sentences.\n"
        "2. Include a concise IMPRESSION section summarizing the main conclusions.\n"
        "3. Mention mediastinum, lungs, pleura, heart, visible upper abdomen, and bones when relevant.\n"
        "4. If a finding is best supported by a specific page or slice span, mention it in prose.\n"
        "5. Do not truncate the report. Finish all sentences.\n\n"
        "Output exactly:\n"
        "FINDINGS:\n"
        "<detailed findings>\n\n"
        "IMPRESSION:\n"
        "<concise impression>"
    )


def build_multimontage_user_message(
    montage_pages: Sequence[Image.Image],
    *,
    patient_demo: str,
    page_metadata: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "image", "image": page} for page in montage_pages]
    content.append(
        {
            "type": "text",
            "text": build_ct_rate_style_prompt(patient_demo=patient_demo, page_metadata=page_metadata),
        }
    )
    return {"role": "user", "content": content}


def load_and_format_multimontage_dataset(
    csv_path: str | os.PathLike[str],
    *,
    meta_dict: dict[str, str],
    jpegs_root: str | os.PathLike[str],
    max_montages: int | None = None,
) -> Any:
    from datasets import Dataset as HFDataset

    jpegs_root_path = Path(jpegs_root)
    records: list[dict[str, Any]] = []

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    missing_volumes = 0
    for row in rows:
        vol_name = row.get("VolumeName", "").replace(".nii.gz", "").strip()
        findings = row.get("Findings_EN", "").strip()
        impression = row.get("Impressions_EN", "").strip()
        if not vol_name or not findings:
            continue

        volume_dir = jpegs_root_path / vol_name
        if not volume_dir.exists():
            missing_volumes += 1
            continue

        slice_paths = _sorted_slice_paths(volume_dir)
        montage_meta = _volume_page_metadata_from_paths(slice_paths, max_montages=max_montages)
        report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}".strip()
        patient_demo = meta_dict.get(vol_name, "Patient demographics unknown")
        prompt_text = build_ct_rate_style_prompt(
            patient_demo=patient_demo,
            page_metadata=montage_meta["pages"],
        )

        records.append(
            {
                "id": vol_name,
                "volume_dir": str(volume_dir),
                "patient_demo": patient_demo,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image"} for _ in range(montage_meta["selected_page_count"])]
                        + [{"type": "text", "text": prompt_text}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": report}],
                    },
                ],
                "ground_truth": report,
                "montage_metadata": montage_meta,
            }
        )

    if records:
        print(f"Prepared {len(records)} studies from {Path(csv_path).name}; skipped {missing_volumes} missing volumes.")
    else:
        print(f"Prepared 0 studies from {Path(csv_path).name}; skipped {missing_volumes} missing volumes.")
    return HFDataset.from_list(records)


def load_combined_available_multimontage_dataset(
    *,
    reports_csvs: Sequence[Path],
    metadata_csvs: Sequence[Path],
    jpegs_root: str | os.PathLike[str],
    max_montages: int | None = None,
) -> Any:
    from datasets import Dataset as HFDataset

    combined_meta: dict[str, str] = {}
    for metadata_csv in metadata_csvs:
        combined_meta.update(load_metadata_dict(metadata_csv))

    jpegs_root_path = Path(jpegs_root)
    records_by_id: dict[str, dict[str, Any]] = {}
    missing_volumes = 0

    for reports_csv in reports_csvs:
        with reports_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        for row in rows:
            vol_name = row.get("VolumeName", "").replace(".nii.gz", "").strip()
            findings = row.get("Findings_EN", "").strip()
            impression = row.get("Impressions_EN", "").strip()
            if not vol_name or not findings:
                continue
            if vol_name in records_by_id:
                continue

            volume_dir = jpegs_root_path / vol_name
            if not volume_dir.exists():
                missing_volumes += 1
                continue

            slice_paths = _sorted_slice_paths(volume_dir)
            montage_meta = _volume_page_metadata_from_paths(slice_paths, max_montages=max_montages)
            report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}".strip()
            patient_demo = combined_meta.get(vol_name, "Patient demographics unknown")
            prompt_text = build_ct_rate_style_prompt(
                patient_demo=patient_demo,
                page_metadata=montage_meta["pages"],
            )

            records_by_id[vol_name] = {
                "id": vol_name,
                "volume_dir": str(volume_dir),
                "patient_demo": patient_demo,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image"} for _ in range(montage_meta["selected_page_count"])]
                        + [{"type": "text", "text": prompt_text}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": report}],
                    },
                ],
                "ground_truth": report,
                "montage_metadata": montage_meta,
            }

    records = list(records_by_id.values())
    print(f"Prepared {len(records)} available studies across {len(reports_csvs)} report files; skipped {missing_volumes} missing volumes.")
    if not records:
        raise RuntimeError(
            f"No usable studies were found under {jpegs_root_path}. "
            "Check that processed JPEG folders exist for the available CT-RATE reports."
        )
    return HFDataset.from_list(records)


def generate_multimontage_report(
    *,
    processor: Any,
    model: Any,
    volume_dir: str | os.PathLike[str],
    patient_demo: str = "Patient demographics unknown",
    max_montages: int | None = None,
    grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
    target_size: tuple[int, int] = DEFAULT_EVAL_PAGE_SIZE,
    max_new_tokens: int = 512,
    max_time: int = 300,
) -> dict[str, Any]:
    montage_pages, montage_meta = build_volume_montage_pages(
        volume_dir,
        grid_size=grid_size,
        target_size=target_size,
        max_montages=max_montages,
    )

    user_msg = build_multimontage_user_message(
        montage_pages,
        patient_demo=patient_demo,
        page_metadata=montage_meta["pages"],
    )

    formatted = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = processor(text=formatted, images=list(montage_pages), return_tensors="pt").to(device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        max_time=max_time,
        do_sample=False,
        no_repeat_ngram_size=5,
        repetition_penalty=1.1,
        use_cache=False,
    )
    output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    report = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return {
        "report": report,
        "montage_metadata": montage_meta,
        "page_count": len(montage_pages),
    }


def summarize_ct_rate_local_reports(
    *,
    data_root: str | os.PathLike[str] | None = None,
    split: str = "validation",
    limit: int = 3,
) -> list[dict[str, str]]:
    root = discover_ct_rate_root(data_root)
    reports_csv = root / "dataset" / "radiology_text_reports" / f"{split}_reports.csv"
    return load_report_examples(reports_csv, limit=limit)


def default_local_paths(data_root: str | os.PathLike[str] | None = None) -> dict[str, Path]:
    root = discover_ct_rate_root(data_root)
    return {
        "data_root": root,
        "jpegs_root": root / "processed_jpegs",
        "train_reports_csv": root / "dataset" / "radiology_text_reports" / "train_reports.csv",
        "valid_reports_csv": root / "dataset" / "radiology_text_reports" / "validation_reports.csv",
        "train_metadata_csv": root / "dataset" / "metadata" / "train_metadata.csv",
        "valid_metadata_csv": root / "dataset" / "metadata" / "validation_metadata.csv",
    }


def _dataset_cache_dir(data_root: Path, split_name: str, max_montages: int | None) -> Path:
    suffix = "all_pages" if max_montages is None else f"max_{max_montages}_pages"
    return data_root / "cached_dataset" / f"multimontage_{split_name}_{suffix}"


def _available_split_cache_dir(data_root: Path, split_name: str, max_montages: int | None, test_size: float, seed: int) -> Path:
    suffix = "all_pages" if max_montages is None else f"max_{max_montages}_pages"
    test_pct = int(round(test_size * 100))
    return data_root / "cached_dataset" / f"multimontage_available_{split_name}_{suffix}_test{test_pct}_seed{seed}"


def load_or_create_multimontage_dataset(
    *,
    reports_csv: Path,
    metadata_csv: Path,
    jpegs_root: Path,
    cache_dir: Path,
    max_montages: int | None,
) -> Any:
    from datasets import load_from_disk

    if cache_dir.exists():
        print(f"✅ Loading cached multi-montage dataset from {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"Building multi-montage dataset from {reports_csv.name} ...")
    meta_dict = load_metadata_dict(metadata_csv)
    dataset = load_and_format_multimontage_dataset(
        reports_csv,
        meta_dict=meta_dict,
        jpegs_root=jpegs_root,
        max_montages=max_montages,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No usable studies were found for {reports_csv.name}. "
            f"Check that processed JPEG folders exist under {jpegs_root} for this split."
        )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir))
    print(f"✅ Saved dataset cache to {cache_dir}")
    return dataset


def load_or_create_available_split_datasets(
    *,
    data_root: Path,
    reports_csvs: Sequence[Path],
    metadata_csvs: Sequence[Path],
    jpegs_root: Path,
    max_montages: int | None,
    test_size: float,
    seed: int,
) -> tuple[Any, Any]:
    from datasets import load_from_disk

    train_cache = _available_split_cache_dir(data_root, "train", max_montages, test_size, seed)
    test_cache = _available_split_cache_dir(data_root, "test", max_montages, test_size, seed)

    if train_cache.exists() and test_cache.exists():
        print(f"✅ Loading cached available-data split from {train_cache.parent}")
        return load_from_disk(str(train_cache)), load_from_disk(str(test_cache))

    print("Building available-data multi-montage dataset from downloaded CT-RATE studies ...")
    full_dataset = load_combined_available_multimontage_dataset(
        reports_csvs=reports_csvs,
        metadata_csvs=metadata_csvs,
        jpegs_root=jpegs_root,
        max_montages=max_montages,
    )
    split_dataset = full_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    train_cache.parent.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(train_cache))
    test_dataset.save_to_disk(str(test_cache))
    print(f"✅ Saved available-data split caches to {train_cache.parent}")
    return train_dataset, test_dataset


def make_multimontage_collator(
    *,
    processor: Any,
    max_montages: int | None,
    grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
    target_size: tuple[int, int] = DEFAULT_TRAIN_PAGE_SIZE,
):
    def collate_fn(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        batch_images: list[list[Image.Image]] = []

        for example in examples:
            montage_pages, montage_meta = build_volume_montage_pages(
                example["volume_dir"],
                grid_size=grid_size,
                target_size=target_size,
                max_montages=max_montages,
            )
            prompt_text = build_ct_rate_style_prompt(
                patient_demo=example["patient_demo"],
                page_metadata=montage_meta["pages"],
            )
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in montage_pages] + [{"type": "text", "text": prompt_text}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["ground_truth"]}],
                },
            ]

            texts.append(
                processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )
            batch_images.append([page.convert("RGB") for page in montage_pages])

        batch = processor(text=texts, images=batch_images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        image_token_candidates = []
        for token_name in ("boi_token", "image_token"):
            token_value = processor.tokenizer.special_tokens_map.get(token_name)
            if token_value:
                image_token_candidates.append(processor.tokenizer.convert_tokens_to_ids(token_value))
        image_token_candidates.extend([262144, 262145])

        for token_id in set(token_id for token_id in image_token_candidates if token_id is not None and token_id >= 0):
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def run_multimontage_evaluation(
    *,
    model_id_or_path: str,
    test_dataset: Any,
    processor: Any,
    output_dir: str | os.PathLike[str],
    max_montages: int | None,
    eval_limit: int = 20,
    target_size: tuple[int, int] = DEFAULT_EVAL_PAGE_SIZE,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    is_adapter_path = Path(model_id_or_path).is_dir() and (Path(model_id_or_path) / "adapter_config.json").exists()
    if is_adapter_path:
        base = AutoModelForImageTextToText.from_pretrained(
            DEFAULT_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
            token=HF_TOKEN,
        )
        model = PeftModel.from_pretrained(base, model_id_or_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
            token=HF_TOKEN,
        )
    model.eval()

    results: list[dict[str, Any]] = []
    limit = min(eval_limit, len(test_dataset))
    for idx in range(limit):
        example = test_dataset[idx]
        generation = generate_multimontage_report(
            processor=processor,
            model=model,
            volume_dir=example["volume_dir"],
            patient_demo=example["patient_demo"],
            max_montages=max_montages,
            target_size=target_size,
        )
        results.append(
            {
                "id": example["id"],
                "ground_truth": example["ground_truth"],
                "prediction": generation["report"],
                "page_count": generation["page_count"],
                "montage_metadata": generation["montage_metadata"],
            }
        )

    with (output_path / "multimontage_eval_generations.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved multi-montage evaluation generations to {output_path / 'multimontage_eval_generations.json'}")
    return results


def inspect_local_ct_rate_examples(paths: dict[str, Path]) -> None:
    examples = summarize_ct_rate_local_reports(data_root=paths["data_root"], split="validation", limit=3)
    print("Local CT-RATE examples:")
    for idx, row in enumerate(examples, start=1):
        print(f"\n[{idx}] {row['VolumeName']}")
        print(f"Findings: {row['Findings_EN'][:400]}...")
        print(f"Impression: {row['Impressions_EN'][:200]}...")

    sample_volume = paths["jpegs_root"] / "valid_1_a_1"
    _, metadata = build_volume_montage_pages(sample_volume)
    print(f"\nSample volume: {sample_volume}")
    print(f"Source slices: {metadata['source_image_count']}")
    print(f"Montage pages needed to cover all slices: {metadata['total_possible_pages']}")
    print(f"Selected pages: {metadata['selected_page_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma on CT-RATE multi-montage studies.")
    parser.add_argument("--data-root", default=None, help="CT-RATE root. Defaults to F:\\XMedFusion\\ct_rate if present.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save adapter checkpoints.")
    parser.add_argument("--train-max-montages", type=int, default=4, help="Maximum montage pages per study during training. Default 4 for 16GB GPUs.")
    parser.add_argument("--eval-max-montages", type=int, default=8, help="Maximum montage pages per study during evaluation/generation.")
    parser.add_argument("--cache-max-montages", type=int, default=8, help="Maximum montage pages stored in cached metadata. Use a value >= train/eval page limits.")
    parser.add_argument("--train-page-size", type=int, default=512, help="Square montage size used during training.")
    parser.add_argument("--eval-page-size", type=int, default=768, help="Square montage size used during evaluation/generation.")
    parser.add_argument("--baseline", action="store_true", help="Run evaluation on the raw base MedGemma model.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate the saved adapter.")
    parser.add_argument("--inspect-only", action="store_true", help="Print dataset/report examples and exit.")
    parser.add_argument("--eval-limit", type=int, default=20, help="Number of validation studies to generate during evaluation.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction of available studies reserved for test/eval.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for the available-data train/test split.")
    args = parser.parse_args()

    paths = default_local_paths(args.data_root)
    if args.inspect_only:
        inspect_local_ct_rate_examples(paths)
        return

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if torch.cuda.get_device_capability()[0] < 8:
        print("Warning: GPU does not officially support bfloat16 natively. Performance may be degraded.")

    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID, token=HF_TOKEN)
    processor.tokenizer.padding_side = "right"

    train_dataset, valid_dataset = load_or_create_available_split_datasets(
        data_root=paths["data_root"],
        reports_csvs=[paths["train_reports_csv"], paths["valid_reports_csv"]],
        metadata_csvs=[paths["train_metadata_csv"], paths["valid_metadata_csv"]],
        jpegs_root=paths["jpegs_root"],
        max_montages=args.cache_max_montages,
        test_size=args.test_size,
        seed=args.split_seed,
    )
    print(
        f"Available-data split ready: train={len(train_dataset)} | test={len(valid_dataset)} "
        f"(test_size={args.test_size}, seed={args.split_seed}, "
        f"train_max_montages={args.train_max_montages}, eval_max_montages={args.eval_max_montages}, "
        f"train_page_size={args.train_page_size}, eval_page_size={args.eval_page_size})"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline:
        run_multimontage_evaluation(
            model_id_or_path=DEFAULT_MODEL_ID,
            test_dataset=valid_dataset,
            processor=processor,
            output_dir=output_dir,
            max_montages=args.eval_max_montages,
            eval_limit=args.eval_limit,
            target_size=(args.eval_page_size, args.eval_page_size),
        )
        return

    if args.eval_only:
        if not output_dir.exists() or not any(output_dir.iterdir()):
            raise FileNotFoundError(f"No saved adapter found in {output_dir}")
        run_multimontage_evaluation(
            model_id_or_path=str(output_dir),
            test_dataset=valid_dataset,
            processor=processor,
            output_dir=output_dir,
            max_montages=args.eval_max_montages,
            eval_limit=args.eval_limit,
            target_size=(args.eval_page_size, args.eval_page_size),
        )
        return

    print("\nInitializing multi-montage QLoRA model...")
    model = AutoModelForImageTextToText.from_pretrained(
        DEFAULT_MODEL_ID,
        token=HF_TOKEN,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model.config.use_cache = False
    if hasattr(model, "vision_tower"):
        model.vision_tower.requires_grad_(False)
    if hasattr(model, "multi_modal_projector"):
        model.multi_modal_projector.requires_grad_(False)

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        r=args.lora_r,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        exclude_modules=["vision_tower", "multi_modal_projector"],
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"],
        ensure_weight_tying=True,
    )

    collate_fn = make_multimontage_collator(
        processor=processor,
        max_montages=args.train_max_montages,
        target_size=(args.train_page_size, args.train_page_size),
    )

    resume_from_checkpoint = output_dir.exists() and any(output_dir.iterdir())
    if resume_from_checkpoint:
        print(f"Found existing checkpoint at {output_dir}. Training will resume from it.")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=5,
        eval_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    print("\n🚀 Starting multi-montage fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"✅ Saving adapter to {output_dir} ...")
    trainer.save_model(str(output_dir))

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_multimontage_evaluation(
        model_id_or_path=str(output_dir),
        test_dataset=valid_dataset,
        processor=processor,
        output_dir=output_dir,
        max_montages=args.eval_max_montages,
        eval_limit=args.eval_limit,
        target_size=(args.eval_page_size, args.eval_page_size),
    )


if __name__ == "__main__":
    main()
