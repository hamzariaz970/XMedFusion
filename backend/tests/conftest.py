"""
conftest.py — Shared pytest fixtures for all XMedFusion test modules.
"""
import os
import sys
import json
import glob
import pytest
from PIL import Image

# Ensure backend root is on sys.path so imports work from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XRAY_IMAGE_DIR = os.path.join(BACKEND_DIR, "data", "iu_xray", "images")
NON_XRAY_IMAGE_DIR = os.path.join(BACKEND_DIR, "data", "test")
ANNOTATION_PATH = os.path.join(BACKEND_DIR, "data", "iu_xray", "annotation.json")


def _first_png(directory):
    """Return the first .png found inside directory tree."""
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".png"):
                return os.path.join(root, f)
    return None


def _first_jpg(directory):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                return os.path.join(root, f)
    return None


@pytest.fixture(scope="session")
def real_xray_path():
    path = _first_png(XRAY_IMAGE_DIR)
    if not path:
        pytest.skip("No X-ray images found in data/iu_xray/images/")
    return path


@pytest.fixture(scope="session")
def non_xray_path():
    path = _first_jpg(NON_XRAY_IMAGE_DIR)
    if not path:
        pytest.skip("No non-X-ray images found in data/test/")
    return path


@pytest.fixture(scope="session")
def real_xray_pil(real_xray_path):
    return Image.open(real_xray_path).convert("RGB")


@pytest.fixture(scope="session")
def non_xray_pil(non_xray_path):
    return Image.open(non_xray_path).convert("RGB")


@pytest.fixture(scope="session")
def annotation_data():
    if not os.path.exists(ANNOTATION_PATH):
        pytest.skip("annotation.json not found")
    with open(ANNOTATION_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_report():
    return (
        "The cardiomediastinal silhouette is within normal limits. "
        "Lungs are clear. No pleural effusion or pneumothorax."
    )


@pytest.fixture(scope="session")
def kg_cardiomegaly():
    """A small fake KG that marks cardiomegaly as Present."""
    return {
        "entities": [
            ["cardiomegaly", "Present"],
            ["mediastinum", "Anatomy"],
        ],
        "relations": [[0, 1, "located_at"]],
    }


@pytest.fixture(scope="session")
def kg_absent_effusion():
    """A small fake KG that marks pleural effusion as Absent."""
    return {
        "entities": [["pleural effusion", "Absent"]],
        "relations": [],
    }
