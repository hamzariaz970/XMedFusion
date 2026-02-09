'''
python kg_agent.py --image "data/iu_xray/images/CXR1_1_IM-0001/1.png" --projection "Frontal" --out "out/outkg.json"
'''

# kg_agent.py
from __future__ import annotations

import os
import sys
import json
import re
from pathlib import Path
from string import Template
from typing import List, Tuple, Literal, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel

from google import genai
from google.genai import types

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# Load backend/.env (same folder as this script)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# -------------------------
# Prompt (same behavior as your current file)
# -------------------------
PROMPT_TMPL = Template(r"""You are an expert radiologist.

You MUST output a RadGraph-style KG that matches this token-level convention.

OUTPUT JSON ONLY:
{"entities": [[TEXT, LABEL], ...], "relations": [[HEAD_IDX, TAIL_IDX, TYPE], ...]}

HARD CONSTRAINTS (do not violate):
- TEXT must be short token-like spans (usually 1 word).
- Do NOT output long phrases: "no evidence of ...", "acute cardiopulmonary abnormality", etc.
- Do NOT output general extra anatomy (bones, soft tissues, diaphragm, gastric bubble, etc.)
  unless it is one of the core RadGraph tokens you chose.
- Prefer this core token set when relevant (normal CXR template):
  cardiac, silhouette, mediastinum, size, pulmonary, edema, pleural, effusion,
  pneumothorax, Normal, chest, x - XXXX
- Split combined concepts:
  "pleural effusion" -> ["pleural", ...] + ["effusion", ...]
  "pulmonary edema"  -> ["pulmonary", ...] + ["edema", ...]
- IMPORTANT label convention for this dataset:
  "size" should be Anatomy::definitely present (not Observation).
  "x - XXXX" should be Observation::definitely present (not Anatomy).

RELATION TEMPLATES (use these whenever applicable):
- silhouette -> cardiac : modify
- size -> mediastinum : modify
- edema -> pulmonary : located_at
- effusion -> pleural : located_at
- Normal -> x - XXXX : modify
- x - XXXX -> chest : located_at

ALLOWED LABELS:
- Anatomy::definitely present/absent/uncertain
- Observation::definitely present/absent/uncertain
ALLOWED RELATIONS:
- modify, located_at, suggestive_of

PROJECTION: $projection
Return ONLY the JSON object, complete and valid.
""")


# -------------------------
# Schema forced output
# -------------------------
Label = Literal[
    "Anatomy::definitely present",
    "Anatomy::definitely absent",
    "Anatomy::uncertain",
    "Observation::definitely present",
    "Observation::definitely absent",
    "Observation::uncertain",
]
RelType = Literal["modify", "located_at", "suggestive_of"]

Entity = Tuple[str, Label]
Relation = Tuple[int, int, RelType]


class KGOut(BaseModel):
    entities: List[Entity]
    relations: List[Relation]


def infer_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def filter_valid_relations(obj: dict) -> dict:
    """Drop relations with out-of-range indices (defensive)."""
    ents = obj.get("entities", [])
    rels = obj.get("relations", [])
    n = len(ents)
    good = []
    for h, t, r in rels:
        if isinstance(h, int) and isinstance(t, int) and 0 <= h < n and 0 <= t < n:
            good.append([h, t, r])
    obj["relations"] = good
    return obj


def extract_first_json_block(text: str) -> Optional[str]:
    """Fallback extractor if model ever returns extra wrapper text."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    a = s.find("{")
    b = s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    return s[a : b + 1]


def infer_kg(
    image: Union[str, Path],
    projection: str = "Frontal",
    *,
    model: str = "gemini-2.5-flash",
    max_output_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    thinking_budget: int = 0,
    debug: bool = False,
) -> dict:
    """
    Run Gemini vision KG extraction and return a Python dict:
      {"entities": [[text,label], ...], "relations": [[head,tail,type], ...]}
    """
    image_path = Path(image).expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Put GEMINI_API_KEY in backend/.env or pass api_key=...")

    prompt = PROMPT_TMPL.safe_substitute(projection=projection)

    client = genai.Client(api_key=key)

    mime = infer_mime(image_path)
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime)

    resp = client.models.generate_content(
        model=model,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            response_mime_type="application/json",
            response_json_schema=KGOut.model_json_schema(),
        ),
    )

    if debug:
        fr = None
        try:
            fr = resp.candidates[0].finish_reason
        except Exception:
            pass
        usage = getattr(resp, "usage_metadata", None)
        print(f"[DEBUG] finish_reason={fr}", file=sys.stderr)
        if usage is not None:
            print(f"[DEBUG] usage_metadata={usage}", file=sys.stderr)

    raw_text = getattr(resp, "text", None) or ""
    # Normally this should already be clean JSON due to response_mime_type + schema
    try:
        kg = KGOut.model_validate_json(raw_text).model_dump()
    except Exception:
        # fallback extraction if anything weird happens
        block = extract_first_json_block(raw_text)
        if not block:
            raise ValueError(f"Model did not return JSON.\nRaw:\n{raw_text[:1000]}")
        kg = KGOut.model_validate_json(block).model_dump()

    return filter_valid_relations(kg)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Gemini vision → RadGraph-style KG (structured JSON).")
    ap.add_argument("--image", required=True)
    ap.add_argument("--projection", default="Frontal")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--max_output_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="Optional .json file path to write pretty JSON.")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    kg = infer_kg(
        args.image,
        projection=args.projection,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
        thinking_budget=0,
        debug=args.debug,
    )

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")

    # machine-friendly stdout
    print(json.dumps(kg, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
