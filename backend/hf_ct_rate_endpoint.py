"""
Run CT-RATE studies through a Hugging Face Inference Endpoint.

This script is intentionally standalone so it can be used without booting the
FastAPI app. It reuses the repository's CT-RATE assumptions:
- studies live under F:\\XMedFusion\\ct_rate by default
- each study is a folder of processed JPEG slices
- the prompt style mirrors backend/vision_ct_multimontage.py

Usage examples
--------------
Dry run against one validation study:
    python hf_ct_rate_endpoint.py --study-ids valid_1000_a_1 --dry-run

Real endpoint call:
    python hf_ct_rate_endpoint.py ^
        --study-ids valid_1000_a_1 valid_1001_a_1 ^
        --endpoint-name your-endpoint-name

Environment
-----------
- HF_TOKEN is required for real endpoint calls.
- HF_ENDPOINT_URL is required unless `--endpoint-url` is passed explicitly.
- HF_ENDPOINT_NAME is optional, but some dedicated endpoints require the model
  field in the OpenAI-compatible request body. If omitted, the script first
  tries without `model` and then raises a clear error if the server requires it.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
from PIL import Image

try:
    # backend/config.py loads .env via dotenv, so this picks up endpoint settings too.
    import config as _backend_config
except Exception:
    _backend_config = None

DEFAULT_DATA_ROOT = Path(r"F:\XMedFusion\ct_rate")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hf_ct_rate_endpoint"
DEFAULT_GRID_SIZE = (4, 4)
DEFAULT_PAGE_SIZE = (768, 768)
DEFAULT_MAX_MONTAGES = 8
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.1
DEFAULT_SELECTION_START_FRACTION = 0.15
DEFAULT_SELECTION_END_FRACTION = 0.90


@dataclass(frozen=True)
class CTRatePaths:
    data_root: Path
    jpegs_root: Path
    reports_csv: Path
    metadata_csv: Path


@dataclass(frozen=True)
class StudyRecord:
    study_id: str
    volume_dir: Path
    patient_demo: str
    findings: str
    impression: str


def normalize_endpoint_url(endpoint_url: str) -> str:
    if not endpoint_url:
        raise ValueError("HF endpoint URL is not configured. Set HF_ENDPOINT_URL in backend/.env or pass --endpoint-url.")
    base = endpoint_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def discover_paths(data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT, split: str = "validation") -> CTRatePaths:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"CT-RATE root not found: {root}")

    split_prefix = "train" if split.lower().startswith("train") else "validation"
    return CTRatePaths(
        data_root=root,
        jpegs_root=root / "processed_jpegs",
        reports_csv=root / "dataset" / "radiology_text_reports" / f"{split_prefix}_reports.csv",
        metadata_csv=root / "dataset" / "metadata" / f"{split_prefix}_metadata.csv",
    )


def format_age_sex(age_str: str, sex_str: str) -> str:
    age = age_str.replace("Y", "").lstrip("0") if age_str else "Unknown"
    sex_map = {"M": "Male", "F": "Female"}
    sex = sex_map.get((sex_str or "").upper(), "Unknown sex")
    return f"{age}-year-old {sex}" if age != "Unknown" else f"Patient of unknown age, {sex}"


def load_metadata_dict(metadata_csv: str | os.PathLike[str]) -> dict[str, str]:
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    result: dict[str, str] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            study_id = row["VolumeName"].replace(".nii.gz", "").strip()
            result[study_id] = format_age_sex(row.get("PatientAge", ""), row.get("PatientSex", ""))
    return result


def load_study_records(
    *,
    paths: CTRatePaths,
    study_ids: Sequence[str] | None = None,
    max_studies: int | None = None,
) -> list[StudyRecord]:
    requested_ids = {study_id.strip() for study_id in study_ids or [] if study_id.strip()}
    metadata = load_metadata_dict(paths.metadata_csv)
    records: list[StudyRecord] = []

    with paths.reports_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            study_id = row.get("VolumeName", "").replace(".nii.gz", "").strip()
            if not study_id:
                continue
            if requested_ids and study_id not in requested_ids:
                continue

            volume_dir = paths.jpegs_root / study_id
            if not volume_dir.exists():
                continue

            records.append(
                StudyRecord(
                    study_id=study_id,
                    volume_dir=volume_dir,
                    patient_demo=metadata.get(study_id, "Patient demographics unknown"),
                    findings=row.get("Findings_EN", "").strip(),
                    impression=row.get("Impressions_EN", "").strip(),
                )
            )

            if max_studies is not None and len(records) >= max_studies:
                break

    if requested_ids:
        found_ids = {record.study_id for record in records}
        missing = sorted(requested_ids - found_ids)
        if missing:
            raise ValueError(f"Requested study IDs not found in {paths.reports_csv.name}: {', '.join(missing)}")

    if not records:
        raise ValueError("No CT-RATE studies matched the selection.")

    return records


def sorted_slice_paths(volume_dir: str | os.PathLike[str]) -> list[Path]:
    volume_path = Path(volume_dir)
    slice_paths = sorted(volume_path.glob("*.jpg"))
    if not slice_paths:
        raise FileNotFoundError(f"No JPEG slices found in {volume_path}")
    return slice_paths


def page_selection_indices(
    total_pages: int,
    max_pages: int,
    *,
    start_fraction: float = DEFAULT_SELECTION_START_FRACTION,
    end_fraction: float = DEFAULT_SELECTION_END_FRACTION,
) -> list[int]:
    if total_pages <= max_pages:
        return list(range(total_pages))
    if max_pages <= 1:
        return [total_pages // 2]

    start_page = max(0, min(total_pages - 1, round((total_pages - 1) * start_fraction)))
    end_page = max(start_page, min(total_pages - 1, round((total_pages - 1) * end_fraction)))
    useful_span = end_page - start_page + 1

    if useful_span < max_pages:
        start_page = 0
        end_page = total_pages - 1

    positions = [
        round(start_page + i * (end_page - start_page) / (max_pages - 1))
        for i in range(max_pages)
    ]
    return sorted(dict.fromkeys(positions))


def build_montage_page(
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
    cell_metadata: list[dict[str, Any]] = []

    for cell_idx in range(rows * cols):
        row = cell_idx // cols
        col = cell_idx % cols
        x = col * tile_w
        y = row * tile_h

        if cell_idx < len(slice_paths):
            source_path = slice_paths[cell_idx]
            with Image.open(source_path) as image:
                rgb = image.convert("RGB").resize((tile_w, tile_h))
            page.paste(rgb, (x, y))
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
    max_montages: int | None = DEFAULT_MAX_MONTAGES,
) -> tuple[list[Image.Image], dict[str, Any]]:
    slice_paths = sorted_slice_paths(volume_dir)
    rows, cols = grid_size
    slices_per_page = rows * cols
    chunks = [slice_paths[start : start + slices_per_page] for start in range(0, len(slice_paths), slices_per_page)]
    total_pages = len(chunks)

    selected_page_indices = list(range(total_pages))
    if max_montages is not None:
        selected_page_indices = page_selection_indices(total_pages, max_montages)

    pages: list[Image.Image] = []
    page_metadata: list[dict[str, Any]] = []
    for selected_idx, page_idx in enumerate(selected_page_indices, start=1):
        global_start = page_idx * slices_per_page
        page, meta = build_montage_page(
            chunks[page_idx],
            grid_size=grid_size,
            target_size=target_size,
            global_start_index=global_start,
        )
        meta["page_index"] = selected_idx
        meta["source_page_index"] = page_idx + 1
        pages.append(page)
        page_metadata.append(meta)

    metadata = {
        "source_image_count": len(slice_paths),
        "selected_page_count": len(pages),
        "total_possible_pages": total_pages,
        "pages": page_metadata,
    }
    return pages, metadata


def build_ct_rate_style_prompt(*, patient_demo: str, page_metadata: Sequence[dict[str, Any]]) -> str:
    page_lines = [
        (
            f"- Page {page['page_index']} covers slices {page['slice_start']} to {page['slice_end']} "
            f"(source page {page['source_page_index']})."
        )
        for page in page_metadata
    ]
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
        "4. Prioritize high-confidence visible findings; if subtle abnormality is uncertain, describe it as possible rather than definite.\n"
        "5. Do not invent lesion counts, measurements, or laterality unless clearly visible on the provided pages.\n"
        "6. Prioritize the visible thorax; do not over-weight the first or last slices if coverage is partial.\n"
        "7. If a finding is best supported by a specific page or slice span, mention it in prose.\n"
        "8. Do not repeat the prompt, page list, or placeholder text. Output only the report body.\n"
        "9. Do not truncate the report. Finish all sentences.\n\n"
        "Output exactly:\n"
        "FINDINGS:\n"
        "<detailed findings>\n\n"
        "IMPRESSION:\n"
        "<concise impression>"
    )


def image_to_data_url(image: Image.Image, *, image_format: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if image_format.upper() == "JPEG" else f"image/{image_format.lower()}"
    return f"data:{mime};base64,{encoded}"


def build_user_content(montage_pages: Sequence[Image.Image], *, prompt_text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for page in montage_pages:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(page)}})
    return content


def build_inference_api_payload(
    *,
    prompt_text: str,
    montage_pages: Sequence[Image.Image],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    image_count = len(montage_pages)
    image_prefix = " ".join(["<start_of_image>"] * image_count).strip()
    text = f"{image_prefix}\n{prompt_text}" if image_prefix else prompt_text
    images = [image_to_data_url(page) for page in montage_pages]
    return {
        "inputs": {
            "text": text,
            "images": images,
        },
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    }


def build_chat_payload(
    *,
    content: Sequence[dict[str, Any]],
    model: str | None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": list(content)}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if model:
        payload["model"] = model
    return payload


def extract_report_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError(f"Unexpected endpoint response shape: {json.dumps(response_json)[:500]}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()


def extract_inference_api_text(response_json: Any) -> str:
    if isinstance(response_json, list) and response_json:
        first = response_json[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"]).strip()
    if isinstance(response_json, dict) and "generated_text" in response_json:
        return str(response_json["generated_text"]).strip()
    raise ValueError(f"Unexpected inference API response shape: {json.dumps(response_json)[:500]}")


def clean_generated_report(raw_text: str, *, prompt_text: str) -> str:
    text = raw_text.strip()
    if text.startswith(prompt_text):
        text = text[len(prompt_text) :].lstrip()
    marker = "FINDINGS:"
    marker_positions = []
    start = 0
    while True:
        index = text.find(marker, start)
        if index == -1:
            break
        marker_positions.append(index)
        start = index + len(marker)
    if marker_positions:
        text = text[marker_positions[-1] :]
    return text.strip()


def call_chat_completion(
    *,
    endpoint_url: str,
    hf_token: str,
    payload: dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    url = f"{normalize_endpoint_url(endpoint_url)}/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )

    if response.ok:
        return response.json()

    error_body = response.text[:1000]
    raise requests.HTTPError(
        f"Endpoint request failed with {response.status_code}: {error_body}",
        response=response,
    )


def call_inference_api(
    *,
    endpoint_url: str,
    hf_token: str,
    payload: dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> Any:
    url = endpoint_url.rstrip("/")
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )

    if response.ok:
        return response.json()

    error_body = response.text[:1000]
    raise requests.HTTPError(
        f"Endpoint request failed with {response.status_code}: {error_body}",
        response=response,
    )


def run_single_study(
    study: StudyRecord,
    *,
    endpoint_url: str,
    hf_token: str | None,
    endpoint_name: str | None,
    output_dir: Path,
    max_montages: int,
    page_size: tuple[int, int],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    dry_run: bool,
    save_montages: bool,
) -> dict[str, Any]:
    montage_pages, montage_metadata = build_volume_montage_pages(
        study.volume_dir,
        target_size=page_size,
        max_montages=max_montages,
    )
    prompt_text = build_ct_rate_style_prompt(
        patient_demo=study.patient_demo,
        page_metadata=montage_metadata["pages"],
    )
    content = build_user_content(montage_pages, prompt_text=prompt_text)
    chat_payload = build_chat_payload(
        content=content,
        model=endpoint_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    inference_payload = build_inference_api_payload(
        prompt_text=prompt_text,
        montage_pages=montage_pages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    study_output_dir = output_dir / study.study_id
    study_output_dir.mkdir(parents=True, exist_ok=True)

    if save_montages or dry_run:
        for index, page in enumerate(montage_pages, start=1):
            page.save(study_output_dir / f"montage_page_{index:02d}.jpg", quality=90)

    payload_preview = {
        "message_count": len(chat_payload["messages"]),
        "content_item_count": len(content),
        "has_model_field": "model" in chat_payload,
        "page_count": len(montage_pages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "study_id": study.study_id,
        "inference_image_count": len(inference_payload["inputs"]["images"]),
    }
    (study_output_dir / "payload_preview.json").write_text(json.dumps(payload_preview, indent=2), encoding="utf-8")
    (study_output_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (study_output_dir / "ground_truth_report.txt").write_text(
        f"FINDINGS:\n{study.findings}\n\nIMPRESSION:\n{study.impression}\n",
        encoding="utf-8",
    )
    (study_output_dir / "montage_metadata.json").write_text(json.dumps(montage_metadata, indent=2), encoding="utf-8")

    if dry_run:
        generated_report = "DRY RUN: endpoint not called."
        (study_output_dir / "generated_report.txt").write_text(generated_report, encoding="utf-8")
        return {
            "study_id": study.study_id,
            "page_count": len(montage_pages),
            "output_dir": str(study_output_dir),
            "report": generated_report,
            "dry_run": True,
        }

    if not hf_token:
        raise ValueError("HF_TOKEN is required for real endpoint calls.")

    try:
        try:
            response_json = call_chat_completion(
                endpoint_url=endpoint_url,
                hf_token=hf_token,
                payload=chat_payload,
                timeout_seconds=timeout_seconds,
            )
            report_text = clean_generated_report(
                extract_report_text(response_json),
                prompt_text=prompt_text,
            )
        except requests.HTTPError as exc:
            response = exc.response
            should_fallback = response is not None and response.status_code in {400, 404}
            if not should_fallback:
                raise
            response_json = call_inference_api(
                endpoint_url=endpoint_url,
                hf_token=hf_token,
                payload=inference_payload,
                timeout_seconds=timeout_seconds,
            )
            report_text = clean_generated_report(
                extract_inference_api_text(response_json),
                prompt_text=prompt_text,
            )
    except requests.HTTPError as exc:
        response = exc.response
        message = str(exc)
        if response is not None and response.status_code == 422 and not endpoint_name:
            message += (
                "\nThe endpoint may require a model/endpoint name in the OpenAI-compatible payload. "
                "Set HF_ENDPOINT_NAME or pass --endpoint-name from the Hugging Face Endpoint overview page."
            )
        raise RuntimeError(message) from exc

    (study_output_dir / "response.json").write_text(json.dumps(response_json, indent=2), encoding="utf-8")
    (study_output_dir / "generated_report.txt").write_text(report_text + "\n", encoding="utf-8")

    return {
        "study_id": study.study_id,
        "page_count": len(montage_pages),
        "output_dir": str(study_output_dir),
        "report": report_text,
        "dry_run": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CT reports from CT-RATE studies via a Hugging Face endpoint.")
    default_endpoint_url = ""
    if _backend_config is not None:
        default_endpoint_url = getattr(_backend_config, "HF_ENDPOINT_URL", "") or ""
    if not default_endpoint_url:
        default_endpoint_url = os.environ.get("HF_ENDPOINT_URL", "")
    parser.add_argument("--endpoint-url", default=default_endpoint_url, help="Hugging Face dedicated endpoint base URL.")
    parser.add_argument("--endpoint-name", default=os.environ.get("HF_ENDPOINT_NAME"), help="Endpoint/model name shown in the HF Endpoint overview.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="CT-RATE root directory.")
    parser.add_argument("--split", default="validation", choices=["validation", "train"], help="Dataset split CSV to use.")
    parser.add_argument("--study-ids", nargs="*", default=None, help="Specific CT-RATE study IDs, e.g. valid_1000_a_1.")
    parser.add_argument("--max-studies", type=int, default=1, help="How many studies to run if --study-ids is omitted.")
    parser.add_argument("--max-montages", type=int, default=DEFAULT_MAX_MONTAGES, help="Maximum montage pages sent per study.")
    parser.add_argument("--page-width", type=int, default=DEFAULT_PAGE_SIZE[0], help="Montage page width in pixels.")
    parser.add_argument("--page-height", type=int, default=DEFAULT_PAGE_SIZE[1], help="Montage page height in pixels.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for reports, prompts, and payload previews.")
    parser.add_argument("--save-montages", action="store_true", help="Save generated montage pages to the output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare payloads and save artifacts without calling the endpoint.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = discover_paths(args.data_root, args.split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_records = load_study_records(
        paths=paths,
        study_ids=args.study_ids,
        max_studies=None if args.study_ids else args.max_studies,
    )

    hf_token = None
    if _backend_config is not None:
        hf_token = getattr(_backend_config, "HF_TOKEN", None) or None
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or None
    summaries: list[dict[str, Any]] = []
    for study in selected_records:
        result = run_single_study(
            study,
            endpoint_url=args.endpoint_url,
            hf_token=hf_token,
            endpoint_name=args.endpoint_name,
            output_dir=output_dir,
            max_montages=args.max_montages,
            page_size=(args.page_width, args.page_height),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            dry_run=args.dry_run,
            save_montages=args.save_montages,
        )
        summaries.append(result)
        print(f"[{study.study_id}] pages={result['page_count']} output={result['output_dir']}")
        print(result["report"][:1200])
        print()

    (output_dir / "run_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
