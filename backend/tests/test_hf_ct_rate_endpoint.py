import base64
import io
from pathlib import Path

from PIL import Image

from hf_ct_rate_endpoint import (
    DEFAULT_GRID_SIZE,
    StudyRecord,
    build_chat_payload,
    build_ct_rate_style_prompt,
    build_user_content,
    build_volume_montage_pages,
    discover_paths,
    extract_report_text,
    image_to_data_url,
    load_study_records,
    normalize_endpoint_url,
    page_selection_indices,
)


def _make_jpg(path: Path, value: int) -> None:
    image = Image.new("L", (64, 64), color=value)
    image.save(path, format="JPEG")


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_normalize_endpoint_url_appends_v1():
    assert normalize_endpoint_url("https://example.com") == "https://example.com/v1"
    assert normalize_endpoint_url("https://example.com/") == "https://example.com/v1"
    assert normalize_endpoint_url("https://example.com/v1") == "https://example.com/v1"


def test_page_selection_indices_spreads_pages():
    assert page_selection_indices(3, 8) == [0, 1, 2]
    assert page_selection_indices(10, 1) == [5]
    assert page_selection_indices(10, 4) == [0, 3, 6, 9]


def test_build_volume_montage_pages_limits_pages_and_tracks_metadata(tmp_path):
    volume_dir = tmp_path / "study_a"
    volume_dir.mkdir()
    for index in range(40):
        _make_jpg(volume_dir / f"slice_{index:04d}.jpg", value=index)

    pages, metadata = build_volume_montage_pages(volume_dir, max_montages=2, target_size=(400, 400))

    assert len(pages) == 2
    assert metadata["total_possible_pages"] == 3
    assert metadata["selected_page_count"] == 2
    assert metadata["pages"][0]["slice_start"] == 1
    assert metadata["pages"][1]["slice_end"] == 40


def test_build_ct_rate_style_prompt_mentions_page_ranges():
    prompt = build_ct_rate_style_prompt(
        patient_demo="65-year-old Male",
        page_metadata=[
            {"page_index": 1, "slice_start": 1, "slice_end": 16},
            {"page_index": 2, "slice_start": 17, "slice_end": 32},
        ],
    )

    assert "65-year-old Male" in prompt
    assert "Page 1 covers slices 1 to 16." in prompt
    assert "Page 2 covers slices 17 to 32." in prompt
    assert "FINDINGS:" in prompt
    assert "IMPRESSION:" in prompt


def test_image_to_data_url_returns_jpeg_prefix():
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    data_url = image_to_data_url(image)
    assert data_url.startswith("data:image/jpeg;base64,")

    payload = data_url.split(",", 1)[1]
    decoded = base64.b64decode(payload)
    reopened = Image.open(io.BytesIO(decoded))
    assert reopened.size == (32, 32)


def test_build_user_content_places_text_then_images():
    pages = [Image.new("RGB", (32, 32), color="black"), Image.new("RGB", (32, 32), color="white")]
    content = build_user_content(pages, prompt_text="Prompt text")

    assert content[0] == {"type": "text", "text": "Prompt text"}
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "image_url"


def test_build_chat_payload_optionally_includes_model():
    content = [{"type": "text", "text": "hello"}]

    payload_without_model = build_chat_payload(content=content, model=None)
    assert "model" not in payload_without_model
    assert payload_without_model["messages"][0]["content"] == content

    payload_with_model = build_chat_payload(content=content, model="endpoint-name")
    assert payload_with_model["model"] == "endpoint-name"


def test_extract_report_text_handles_string_and_list_shapes():
    string_shape = {"choices": [{"message": {"content": "FINDINGS:\nNormal.\n\nIMPRESSION:\nNormal."}}]}
    assert extract_report_text(string_shape).startswith("FINDINGS:")

    list_shape = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "FINDINGS:\nMild opacity.\n\nIMPRESSION:\nOpacity."}
                    ]
                }
            }
        ]
    }
    assert "Mild opacity" in extract_report_text(list_shape)


def test_load_study_records_reads_local_ct_rate_layout(tmp_path):
    data_root = tmp_path / "ct_rate"
    processed = data_root / "processed_jpegs" / "valid_1000_a_1"
    processed.mkdir(parents=True)
    for index in range(4):
        _make_jpg(processed / f"slice_{index:04d}.jpg", value=index)

    _write_csv(
        data_root / "dataset" / "radiology_text_reports" / "validation_reports.csv",
        "VolumeName,ClinicalInformation_EN,Technique_EN,Findings_EN,Impressions_EN",
        [
            'valid_1000_a_1.nii.gz,Not given.,CT chest,"Lungs are clear.","No acute abnormality."',
            'valid_missing_a_1.nii.gz,Not given.,CT chest,"Should be skipped.","Missing volume."',
        ],
    )
    _write_csv(
        data_root / "dataset" / "metadata" / "validation_metadata.csv",
        "VolumeName,PatientAge,PatientSex",
        ["valid_1000_a_1.nii.gz,065Y,M"],
    )

    paths = discover_paths(data_root, "validation")
    records = load_study_records(paths=paths, study_ids=["valid_1000_a_1"])

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, StudyRecord)
    assert record.study_id == "valid_1000_a_1"
    assert record.patient_demo == "65-year-old Male"
    assert record.volume_dir.name == "valid_1000_a_1"
