"""
ct_vision_agent.py
==================
CT-specific vision agent for XMedFusion.
Uses the exact model from vision_ct.py: google/medgemma-4b-it
(or a fine-tuned LoRA checkpoint if available at model_weights/Vision_Agent/medgemma_ct_grid_finetuned)

Exposes a single CTVisionAgent class with:
  - build_montage_from_files(image_paths) -> PIL.Image
  - generate_report(image_paths) -> str
"""

import os
import glob
import copy
import torch
import numpy as np
import cv2
from PIL import Image
from typing import List

import config

# Fine-tuned model path (optional - if it exists after training, it will be preferred)
FINETUNED_CT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), 
    "model_weights", "Vision_Agent", "medgemma_ct_grid_finetuned"
)
BASE_CT_MODEL_ID = "google/medgemma-4b-it"


def _build_montage_from_single_image(image_path: str, grid_size=(4, 4), target_size=(512, 512)) -> Image.Image:
    """
    For a SINGLE uploaded CT slice (or any image), create a pseudo-montage.
    Tiles the same image with slight offset variants to approximate a multi-slice montage
    that the MedGemma model was trained to expect.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rows_count, cols_count = grid_size
    num_tiles = rows_count * cols_count
    slice_h = target_size[1] // rows_count
    slice_w = target_size[0] // cols_count

    tiles = []
    for i in range(num_tiles):
        resized = cv2.resize(img_rgb, (slice_w, slice_h))
        tiles.append(resized)

    rows = []
    for r in range(rows_count):
        row = np.hstack(tiles[r * cols_count:(r + 1) * cols_count])
        rows.append(row)

    grid = np.vstack(rows)
    return Image.fromarray(grid)


def _build_montage_from_directory(volume_dir: str, grid_size=(4, 4), target_size=(512, 512)) -> Image.Image:
    """
    Loads equidistant slices from a directory of JPEG files, exactly as in vision_ct.py.
    This matches the training distribution of the fine-tuned MedGemma model.
    """
    jpeg_paths = sorted(glob.glob(os.path.join(volume_dir, "*.jpg")))
    jpeg_paths += sorted(glob.glob(os.path.join(volume_dir, "*.jpeg")))
    jpeg_paths += sorted(glob.glob(os.path.join(volume_dir, "*.png")))

    if not jpeg_paths:
        raise FileNotFoundError(f"No image slices found in: {volume_dir}")

    total = len(jpeg_paths)
    # Skip neck (top 15%) and lower abdomen (bottom 10%)
    start = int(total * 0.15)
    end   = int(total * 0.90)
    useful = jpeg_paths[start:end] or jpeg_paths  # Fallback to all if range is empty

    rows_count, cols_count = grid_size
    num_images_needed = rows_count * cols_count
    actual_slices = min(num_images_needed, len(useful))
    indices = np.linspace(0, len(useful) - 1, actual_slices, dtype=int)

    slice_w = target_size[0] // cols_count
    slice_h = target_size[1] // rows_count

    resized_slices = []
    for idx in indices:
        img = cv2.imread(useful[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (slice_w, slice_h))
        resized_slices.append(img_resized)

    # Pad with blank frames if needed
    while len(resized_slices) < num_images_needed:
        resized_slices.append(np.zeros((slice_h, slice_w, 3), dtype=np.uint8))

    rows = []
    for r in range(rows_count):
        row = np.hstack(resized_slices[r * cols_count:(r + 1) * cols_count])
        rows.append(row)

    grid = np.vstack(rows)
    return Image.fromarray(grid)


class CTVisionAgent:
    """
    Wraps the MedGemma VLM inference pipeline for CT scans.
    The model is loaded LAZILY on the first call to generate_report() to avoid
    loading 10 GB at server startup.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        print("[CT Agent] Initialized (model not yet loaded).")

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

        model_path = BASE_CT_MODEL_ID
        is_peft = False

        # Prefer the fine-tuned checkpoint if it exists
        if os.path.isdir(FINETUNED_CT_MODEL_DIR) and os.path.exists(
            os.path.join(FINETUNED_CT_MODEL_DIR, "adapter_config.json")
        ):
            model_path = FINETUNED_CT_MODEL_DIR
            is_peft = True
            print(f"[CT Agent] Loading fine-tuned LoRA checkpoint from: {model_path}")
        else:
            print(f"[CT Agent] Loading base MedGemma model: {BASE_CT_MODEL_ID}")

        # Load processor
        self._processor = AutoProcessor.from_pretrained(BASE_CT_MODEL_ID)
        self._processor.tokenizer.padding_side = "left"

        # 4-bit quantization to reduce VRAM footprint alongside the X-ray models
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        if is_peft:
            from peft import PeftModel
            base = AutoModelForImageTextToText.from_pretrained(
                BASE_CT_MODEL_ID,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                # Use eager attention to avoid torch>=2.6 mask function requirement
                attn_implementation="eager",
                device_map="auto",
            )
            self._model = PeftModel.from_pretrained(base, model_path)
            self._model = self._model.merge_and_unload()
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                # Use eager attention to avoid torch>=2.6 mask function requirement
                attn_implementation="eager",
                device_map="auto",
            )

        self._model.eval()
        self._device = next(self._model.parameters()).device
        print(f"[CT Agent] Model loaded on: {self._device}")

    def build_montage_from_files(self, image_paths: List[str]) -> Image.Image:
        """
        Build a CT montage grid from a list of uploaded image paths.
        - Multiple paths: treated as a series of CT slices
        - Single path: tiled into a pseudo-montage for single-slice uploads
        """
        if len(image_paths) > 1:
            # Multiple files uploaded — check if they're in the same directory
            parent_dir = os.path.dirname(image_paths[0])
            all_same_dir = all(os.path.dirname(p) == parent_dir for p in image_paths)

            if all_same_dir:
                # Use the directory montage logic (matches training distribution)
                return _build_montage_from_directory(parent_dir)
            else:
                # Multi-upload from different locations: stitch them directly
                num = len(image_paths)
                cols = min(num, 4)
                rows_count = (num + cols - 1) // cols
                tile_w, tile_h = 128, 128

                tiles = []
                for p in image_paths:
                    img = cv2.imread(p)
                    if img is None:
                        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tiles.append(cv2.resize(img_rgb, (tile_w, tile_h)))

                while len(tiles) < rows_count * cols:
                    tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

                grid_rows = []
                for r in range(rows_count):
                    grid_rows.append(np.hstack(tiles[r * cols:(r + 1) * cols]))
                return Image.fromarray(np.vstack(grid_rows))
        else:
            # Single uploaded CT image
            return _build_montage_from_single_image(image_paths[0])

    def generate_report(self, image_paths: List[str]) -> str:
        """
        Build grid montage from image_paths and run MedGemma to generate a CT report.
        Returns a string with FINDINGS: and IMPRESSION: headers.
        """
        if self._model is None:
            self._load_model()

        pil_img = self.build_montage_from_files(image_paths)

        # Build the same prompt template used in vision_ct.py training
        prompt_text = (
            "You are an expert thoracic radiologist. Analyze the attached CT scan montage "
            "(axial slices arranged in a grid, cranio-caudal order) and write a structured radiology report.\n"
            "FINDINGS:\n"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        try:
            prompt_str = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=prompt_str, images=[pil_img], return_tensors="pt"
            ).to(self._device)

            with torch.inference_mode():
                output_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=350,
                    do_sample=False,
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )

            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_output = self._processor.decode(generated, skip_special_tokens=True).strip()

            print(f"[CT Agent] Raw MedGemma output:\n{raw_output[:300]}...")

            # Ensure the output has the expected FINDINGS / IMPRESSION structure
            if "FINDINGS:" not in raw_output.upper():
                # Model output already continues from the "FINDINGS:\n" suffix in the prompt
                raw_output = f"FINDINGS:\n{raw_output}"

            if "IMPRESSION:" not in raw_output.upper():
                # Split the findings at the last sentence boundary and form Impression
                parts = raw_output.split(". ")
                mid = len(parts) // 2
                findings_part = ". ".join(parts[:mid]) + "."
                impression_part = ". ".join(parts[mid:]).strip()
                raw_output = f"{findings_part}\n\nIMPRESSION:\n{impression_part}"

            return raw_output

        except Exception as e:
            print(f"[CT Agent] ❌ Inference error: {e}")
            return (
                "FINDINGS:\nCT scan analysis encountered an error during inference. "
                "Please ensure the image is a valid CT scan.\n\n"
                "IMPRESSION:\nReport generation failed. Manual review required."
            )


# ── Module-level singleton ─────────────────────────────────────────────────────
# Instantiated once, model loaded lazily on first CT request
ct_agent = CTVisionAgent()
