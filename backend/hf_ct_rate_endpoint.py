from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from PIL import Image

import config


HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL", "").strip()
HF_CT_ENDPOINT_URL = os.getenv("HF_CT_ENDPOINT_URL", "").strip()
HF_CT_ENDPOINT_MODEL = os.getenv("HF_CT_ENDPOINT_MODEL", "google/medgemma-4b-it").strip()
HF_CT_ENDPOINT_TIMEOUT_SECONDS = int(os.getenv("HF_CT_ENDPOINT_TIMEOUT_SECONDS", "240"))
HF_CT_ENDPOINT_MAX_IMAGES = 1
HF_CT_ENDPOINT_IMAGE_MAX_DIM = int(os.getenv("HF_CT_ENDPOINT_IMAGE_MAX_DIM", "160"))
HF_CT_ENDPOINT_JPEG_QUALITY = int(os.getenv("HF_CT_ENDPOINT_JPEG_QUALITY", "60"))
HF_CT_ENDPOINT_MAX_NEW_TOKENS = int(os.getenv("HF_CT_ENDPOINT_MAX_NEW_TOKENS", "256"))
HF_ROUTER_URL = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions").strip()


class HFCTEndpointError(RuntimeError):
    pass


def _resolve_endpoint_url() -> str:
    if HF_CT_ENDPOINT_URL:
        return HF_CT_ENDPOINT_URL
    if HF_ENDPOINT_URL:
        return HF_ENDPOINT_URL
    return HF_ROUTER_URL


def _representative_indices(
    total: int,
    limit: int,
    *,
    trim_top: float = 0.15,
    trim_bottom: float = 0.10,
) -> list[int]:
    if total <= 0:
        return []
    if limit <= 0:
        raise ValueError("HF_CT_ENDPOINT_MAX_IMAGES must be greater than 0.")
    if total <= limit:
        return list(range(total))

    start = int(total * trim_top)
    end = int(total * (1.0 - trim_bottom))
    if end <= start:
        start, end = 0, total

    usable = list(range(start, end))
    if not usable:
        usable = list(range(total))

    if len(usable) <= limit:
        return usable
    if limit == 1:
        return [usable[len(usable) // 2]]

    positions = [round(i * (len(usable) - 1) / (limit - 1)) for i in range(limit)]
    return [usable[pos] for pos in positions]


def _select_representative_paths(image_paths: list[str], limit: int) -> tuple[list[str], list[int]]:
    selected_indexes = _representative_indices(len(image_paths), limit)
    return [image_paths[index] for index in selected_indexes], selected_indexes


def _image_to_data_url(path: str) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((HF_CT_ENDPOINT_IMAGE_MAX_DIM, HF_CT_ENDPOINT_IMAGE_MAX_DIM))
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=HF_CT_ENDPOINT_JPEG_QUALITY,
            optimize=True,
        )

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _build_prompt(slice_count: int) -> str:
    return (
        "You are an expert thoracic radiologist analyzing a CT-RATE style chest CT study. "
        f"The study is represented by {slice_count} axial slice images in cranio-caudal order. "
        "Write the report in exactly this format:\n\n"
        "FINDINGS:\n"
        "Use complete radiology sentences in the style of the CT-RATE report corpus. "
        "Write a comprehensive Findings section of roughly 110 to 180 words when the study supports that level of detail. "
        "Describe airways, lungs, pleura, heart, mediastinum, visible upper abdomen, and bones when relevant. "
        "Prefer natural chest CT phrasing such as patency of the trachea/main bronchi, mediastinal or hilar adenopathy, pleural effusion, nodules, ground-glass opacity, consolidation, atelectatic change, and osseous abnormality when supported. "
        "If a structure is only partially visible, qualify it as such instead of omitting the region entirely. "
        "Mention slice numbers in brackets only when helpful, for example [Slice 5].\n\n"
        "IMPRESSION:\n"
        "Provide a concise diagnostic impression of 1 to 3 sentences, usually about 20 to 50 words, prioritizing the dominant abnormalities or stating no acute thoracic abnormality when appropriate.\n\n"
        "Do not include markdown fences, JSON, explanations, recommendations, labels, or any text outside FINDINGS and IMPRESSION."
    )


def _endpoint_inputs_prompt(prompt: str, image_urls: list[str]) -> str:
    image_blocks = "\n\n".join(f"![]({url})" for url in image_urls)
    return f"{image_blocks}\n\n{prompt}".strip()


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "\n".join(_extract_text(item) for item in payload if item)
    if not isinstance(payload, dict):
        return str(payload or "")

    for key in ("generated_text", "text", "output_text", "report"):
        value = payload.get(key)
        if value:
            return _extract_text(value)

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        content = message.get("content")
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if content:
            return str(content)

    return json.dumps(payload)


def normalize_ct_rate_report(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s*```$", "", clean).strip()

    findings_match = re.search(
        r"FINDINGS?:\s*(.*?)(?=\n?\s*(?:IMPRESSION|IMPRESSIONS):|\Z)",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )
    impression_match = re.search(
        r"IMPRESSIONS?:\s*(.*?)(?=\n?\s*(?:RECOMMENDATIONS?|LABELS?):|\Z)",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )

    findings = findings_match.group(1).strip() if findings_match else clean.strip()
    impression = impression_match.group(1).strip() if impression_match else ""

    if not impression:
        impression = "No separate impression was provided by the CT endpoint."

    if not findings:
        findings = "No acute cardiopulmonary abnormality is identified on the provided CT slices."

    return f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}".strip()


def _router_payload(prompt: str, image_urls: list[str], *, max_new_tokens: int | None = None) -> dict[str, Any]:
    token_limit = max_new_tokens or HF_CT_ENDPOINT_MAX_NEW_TOKENS
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
    return {
        "model": HF_CT_ENDPOINT_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": token_limit,
        "temperature": 0.1,
    }


def _custom_endpoint_payload(prompt: str, image_urls: list[str], *, max_new_tokens: int | None = None) -> dict[str, Any]:
    token_limit = max_new_tokens or HF_CT_ENDPOINT_MAX_NEW_TOKENS
    return {
        "inputs": {
            "image_urls": image_urls,
            "images": image_urls,
            "prompt": prompt,
            "text": prompt,
        },
        "parameters": {
            "max_new_tokens": token_limit,
            "temperature": 0.1,
            "do_sample": False,
            "return_full_text": False,
        },
    }


def generate_ct_rate_report_from_endpoint(image_paths: list[str], *, max_new_tokens: int | None = None) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("No CT slices were provided.")

    missing_paths = [path for path in image_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"CT slice paths not found: {missing_paths[0]}")

    token = os.getenv("HF_CT_ENDPOINT_API_KEY") or os.getenv("HF_TOKEN") or config.HF_TOKEN
    if not token:
        raise RuntimeError("HF_TOKEN or HF_CT_ENDPOINT_API_KEY is required for the CT endpoint path.")

    selected_paths, selected_indexes = _select_representative_paths(image_paths, HF_CT_ENDPOINT_MAX_IMAGES)
    image_urls = [_image_to_data_url(path) for path in selected_paths]
    prompt = _build_prompt(len(selected_paths))
    endpoint_url = _resolve_endpoint_url()
    payload = (
        _custom_endpoint_payload(prompt, image_urls, max_new_tokens=max_new_tokens)
        if endpoint_url != HF_ROUTER_URL
        else _router_payload(prompt, image_urls, max_new_tokens=max_new_tokens)
    )

    response = requests.post(
        endpoint_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=HF_CT_ENDPOINT_TIMEOUT_SECONDS,
    )
    if not response.ok:
        error_text = response.text.strip()
        try:
            error_payload = response.json()
            error_text = json.dumps(error_payload)
        except Exception:
            pass
        raise HFCTEndpointError(
            f"HF endpoint request failed with status {response.status_code}: {error_text}"
        )

    raw_payload = response.json()
    raw_report = _extract_text(raw_payload)
    final_report = normalize_ct_rate_report(raw_report)

    return {
        "final_report": final_report,
        "raw_report": raw_report,
        "endpoint_url": endpoint_url,
        "model": HF_CT_ENDPOINT_MODEL,
        "selected_slice_count": len(selected_paths),
        "selected_slice_indices": selected_indexes,
        "input_slice_count": len(image_paths),
    }
