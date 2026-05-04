from __future__ import annotations

import base64
import gc
import io
import os
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL_ID = os.getenv("HF_CT_BASE_MODEL_ID", "google/medgemma-4b-it").strip()
ADAPTER_DIR = os.getenv("HF_CT_ADAPTER_DIR", "").strip()
MAX_NEW_TOKENS = int(os.getenv("HF_CT_MAX_NEW_TOKENS", "256"))


def _decode_image_url(image_url: str) -> Image.Image:
    if not image_url.startswith("data:image/"):
        raise ValueError("Only data:image/* base64 URLs are supported by this endpoint handler.")
    _, encoded = image_url.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, use_fast=False)
        self.processor.tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_ID,
            dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="eager",
            device_map="auto",
        )

        if ADAPTER_DIR:
            model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            model = model.merge_and_unload()
        else:
            model = base

        model.config._attn_implementation = "eager"
        model.eval()
        self.model = model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = data.get("inputs") if isinstance(data, dict) else None
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON payload with an 'inputs' object.")

        image_urls = payload.get("image_urls") or payload.get("images")
        if not isinstance(image_urls, list) or not image_urls:
            raise ValueError("Expected inputs.image_urls to be a non-empty list.")

        prompt = str(payload.get("prompt") or payload.get("text") or "").strip()
        if not prompt:
            raise ValueError("Expected inputs.prompt to be a non-empty string.")

        parameters = data.get("parameters") if isinstance(data, dict) else {}
        max_new_tokens = int(parameters.get("max_new_tokens", MAX_NEW_TOKENS))

        images = [_decode_image_url(image_url) for image_url in image_urls]
        user_msg = {
            "role": "user",
            "content": (
                [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}]
            ),
        }

        formatted = self.processor.apply_chat_template(
            [user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.processor(
            text=formatted,
            images=images,
            return_tensors="pt",
        ).to(self.model.device)

        if "token_type_ids" in model_inputs:
            del model_inputs["token_type_ids"]

        with torch.no_grad():
            out_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=5,
                repetition_penalty=1.1,
                use_cache=False,
            )

        out_ids = out_ids[:, model_inputs["input_ids"].shape[1]:]
        generated_text = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()

        del model_inputs, out_ids, images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"generated_text": generated_text}
