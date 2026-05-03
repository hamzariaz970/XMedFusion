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

import csv
import math
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from config import HF_TOKEN

os.environ["HF_TOKEN"] = HF_TOKEN

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "ct_rate"
DEFAULT_GRID_SIZE = (4, 4)
DEFAULT_PAGE_SIZE = (1024, 1024)
DEFAULT_MODEL_ID = "google/medgemma-4b-it"


def discover_ct_rate_root(explicit_root: str | os.PathLike[str] | None = None) -> Path:
    if explicit_root:
        root = Path(explicit_root)
    else:
        root = DEFAULT_DATA_ROOT
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

    metadata = {
        "source_image_count": len(slice_paths),
        "selected_page_count": len(pages),
        "total_possible_pages": total_pages,
        "grid_size": {"rows": rows, "cols": cols},
        "target_size": {"width": target_size[0], "height": target_size[1]},
        "pages": page_metadata,
    }
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

    for row in rows:
        vol_name = row.get("VolumeName", "").replace(".nii.gz", "").strip()
        findings = row.get("Findings_EN", "").strip()
        impression = row.get("Impressions_EN", "").strip()
        if not vol_name or not findings:
            continue

        volume_dir = jpegs_root_path / vol_name
        if not volume_dir.exists():
            continue

        _, montage_meta = build_volume_montage_pages(volume_dir, max_montages=max_montages)
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

    return HFDataset.from_list(records)


def generate_multimontage_report(
    *,
    processor: Any,
    model: Any,
    volume_dir: str | os.PathLike[str],
    patient_demo: str = "Patient demographics unknown",
    max_montages: int | None = None,
    grid_size: tuple[int, int] = DEFAULT_GRID_SIZE,
    target_size: tuple[int, int] = DEFAULT_PAGE_SIZE,
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


if __name__ == "__main__":
    paths = default_local_paths()
    examples = summarize_ct_rate_local_reports(data_root=paths["data_root"], split="validation", limit=3)
    print("Local CT-RATE examples:")
    for idx, row in enumerate(examples, start=1):
        print(f"\n[{idx}] {row['VolumeName']}")
        print(f"Findings: {row['Findings_EN'][:400]}...")
        print(f"Impression: {row['Impressions_EN'][:200]}...")

    sample_volume = paths["jpegs_root"] / "valid_1_a_1"
    pages, metadata = build_volume_montage_pages(sample_volume)
    print(f"\nSample volume: {sample_volume}")
    print(f"Source slices: {metadata['source_image_count']}")
    print(f"Montage pages needed to cover all slices: {metadata['total_possible_pages']}")
    print(f"Selected pages: {metadata['selected_page_count']}")
