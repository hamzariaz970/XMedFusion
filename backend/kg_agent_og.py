'''
To run:
  --image "/Users/hamzariaz/VSCODE/FYP/Website/XMedAgent/backend/data/iu_xray/images/CXR1_1_IM-0001/1.png" \
  --projection "Frontal" \
  --out "/Users/hamzariaz/VSCODE/FYP/Website/XMedAgent/backend/out" \
  --gt_jsonl "/Users/hamzariaz/VSCODE/FYP/Website/XMedAgent/backend/iu_xray/iu_xray_kg_labels.jsonl" \
  --gt_uid 1 \
  --viz \
  --viz_format png \
  --viz_out "/Users/hamzariaz/VSCODE/FYP/Website/XMedAgent/backend/out_graph" \
  --debug
Make sure to set GEMINI_API_KEY in backend/.env
'''
from __future__ import annotations

import os
import sys
import json
import argparse
import re
from pathlib import Path
from string import Template
from typing import List, Tuple, Literal, Optional
from collections import Counter
from difflib import SequenceMatcher

from dotenv import load_dotenv
from pydantic import BaseModel

from google import genai
from google.genai import types

# Load backend/.env (same folder as this script)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# -------------------------
# RadGraph-token style prompt
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


def resolve_out_path(out_arg: Optional[str], image_path: Path) -> Optional[Path]:
    """Allow --out to be a directory OR a file path ending with .json."""
    if not out_arg:
        return None
    p = Path(out_arg).expanduser()
    if p.suffix.lower() != ".json":
        p.mkdir(parents=True, exist_ok=True)
        return p / f"kg_{image_path.stem}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_viz_path(viz_out: Optional[str], image_path: Path, fmt: str) -> Optional[Path]:
    """
    --viz_out can be:
      - a directory -> writes <dir>/kg_<stem>.<fmt>
      - a file path ending in .png/.svg/.pdf -> uses that
      - None -> returns None
    """
    if not viz_out:
        return None
    p = Path(viz_out).expanduser()
    if p.suffix.lower() in {".png", ".svg", ".pdf"}:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / f"kg_{image_path.stem}.{fmt}"


# -------------------------
# Ground truth loading
# -------------------------
def load_gt_from_jsonl(jsonl_path: Path, uid: Optional[int] = None, filename: Optional[str] = None) -> dict:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"GT jsonl not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if uid is not None and rec.get("uid") != uid:
                continue
            if filename is not None and rec.get("filename") != filename:
                continue

            tj = rec.get("target_json")
            if tj is None:
                raise ValueError("Matched record has no target_json field")

            return json.loads(tj)

    raise ValueError(f"No ground-truth record matched uid={uid} filename={filename}")


def auto_uid_from_image(image_path: Path) -> Optional[int]:
    stem = image_path.stem.strip()
    return int(stem) if stem.isdigit() else None


# -------------------------
# Fuzzy matching utilities (kept; not required for GT viz)
# -------------------------
def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-]", "", s)
    return s

def sim(a: str, b: str) -> float:
    a0, b0 = norm_text(a), norm_text(b)
    if not a0 or not b0:
        return 0.0
    seq = SequenceMatcher(None, a0, b0).ratio()
    ta, tb = set(a0.split()), set(b0.split())
    jac = (len(ta & tb) / len(ta | tb)) if (ta | tb) else 0.0
    return max(seq, jac)

def greedy_entity_alignment(pred_entities, gold_entities, threshold):
    pairs = []
    for pi, (pt, pl) in enumerate(pred_entities):
        for gi, (gt, gl) in enumerate(gold_entities):
            if pl != gl:
                continue
            score = sim(pt, gt)
            if score >= threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)

    used_p, used_g = set(), set()
    pred_to_gold = {}
    tp = 0
    for score, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pred_to_gold[pi] = gi
        tp += 1
    return pred_to_gold, tp

def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2*p*r)/(p+r) if (p+r) else 0.0
    return p, r, f1

def filter_valid_relations(obj: dict) -> dict:
    ents = obj.get("entities", [])
    rels = obj.get("relations", [])
    n = len(ents)
    good = []
    for h, t, r in rels:
        if isinstance(h, int) and isinstance(t, int) and 0 <= h < n and 0 <= t < n:
            good.append([h, t, r])
    obj["relations"] = good
    return obj


# -------------------------
# BEAUTIFUL KG VIZ (Graphviz)
# -------------------------
def _label_parts(lbl: str):
    a, b = lbl.split("::", 1)
    return a.strip(), b.strip()

def render_kg_graphviz(pred_obj: dict, out_file: Path, title: str = "") -> None:
    try:
        from graphviz import Digraph
    except Exception as e:
        raise RuntimeError(
            "Graphviz python package not available. Install: pip install graphviz\n"
            "Also install system Graphviz (macOS): brew install graphviz"
        ) from e

    entities = pred_obj.get("entities", [])
    relations = pred_obj.get("relations", [])

    certainty_style = {
        "definitely present":  {"color": "#1F7A1F", "fillcolor": "#E6F4E6", "style": "filled"},
        "definitely absent":   {"color": "#B3261E", "fillcolor": "#FCE8E6", "style": "filled"},
        "uncertain":           {"color": "#B26A00", "fillcolor": "#FFF4E0", "style": "filled,dashed"},
    }
    category_shape = {"Anatomy": "box", "Observation": "ellipse"}

    edge_style = {
        "modify":       {"color": "#5F6368", "style": "solid",  "penwidth": "1.4"},
        "located_at":   {"color": "#1A73E8", "style": "bold",   "penwidth": "1.8"},
        "suggestive_of":{"color": "#9334E6", "style": "dashed", "penwidth": "1.6"},
    }

    g = Digraph("KG", format=out_file.suffix.lstrip("."))
    g.attr(
        rankdir="LR",
        bgcolor="white",
        pad="0.2",
        nodesep="0.45",
        ranksep="0.55",
        splines="spline",
        overlap="false",
        fontname="Helvetica",
        fontsize="11",
        labelloc="t",
        label=(title if title else "Knowledge Graph"),
    )

    g.attr("node", fontname="Helvetica", fontsize="10", margin="0.12,0.08")
    g.attr("edge", fontname="Helvetica", fontsize="9", arrowsize="0.8")

    anatomy_nodes = []
    obs_nodes = []
    for i, (text, lbl) in enumerate(entities):
        cat, cert = _label_parts(lbl)
        if cat == "Anatomy":
            anatomy_nodes.append((i, text, lbl, cert))
        else:
            obs_nodes.append((i, text, lbl, cert))

    def add_cluster(name: str, color: str, rows):
        with g.subgraph(name=f"cluster_{name}") as c:
            c.attr(label=name, color=color, penwidth="1.2", style="rounded", fontname="Helvetica", fontsize="11")
            for i, text, lbl, cert in rows:
                cat, cert2 = _label_parts(lbl)
                st = certainty_style.get(cert2, certainty_style["uncertain"])
                node_id = f"n{i}"
                node_label = f"{i}: {text}\\n{lbl}"
                c.node(
                    node_id,
                    label=node_label,
                    shape=category_shape.get(cat, "box"),
                    color=st["color"],
                    fillcolor=st["fillcolor"],
                    style=st["style"],
                    penwidth="1.4",
                )

    add_cluster("Anatomy", "#8AB4F8", anatomy_nodes)
    add_cluster("Observation", "#81C995", obs_nodes)

    for (h, t, rtype) in relations:
        src = f"n{h}"
        dst = f"n{t}"
        st = edge_style.get(rtype, edge_style["suggestive_of"])
        g.edge(src, dst, label=rtype, color=st["color"], style=st["style"], penwidth=st["penwidth"])

    with g.subgraph(name="cluster_legend") as lg:
        lg.attr(label="Legend", style="rounded,dashed", color="#DADCE0", fontname="Helvetica", fontsize="11")

        lg.node("L1", label="Anatomy\\n(definitely present)", shape="box",
                color=certainty_style["definitely present"]["color"],
                fillcolor=certainty_style["definitely present"]["fillcolor"],
                style=certainty_style["definitely present"]["style"])
        lg.node("L2", label="Observation\\n(definitely absent)", shape="ellipse",
                color=certainty_style["definitely absent"]["color"],
                fillcolor=certainty_style["definitely absent"]["fillcolor"],
                style=certainty_style["definitely absent"]["style"])
        lg.node("L3", label="Uncertain", shape="ellipse",
                color=certainty_style["uncertain"]["color"],
                fillcolor=certainty_style["uncertain"]["fillcolor"],
                style=certainty_style["uncertain"]["style"])

        lg.edge("L1", "L2", label="located_at", **edge_style["located_at"])
        lg.edge("L2", "L3", label="suggestive_of", **edge_style["suggestive_of"])
        lg.edge("L1", "L3", label="modify", **edge_style["modify"])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_file.with_suffix(".dot")
    dot_path.write_text(g.source, encoding="utf-8")

    tmp_base = out_file.with_suffix("")
    g.render(filename=str(tmp_base), cleanup=True)

    produced = tmp_base.with_suffix(out_file.suffix)
    if produced.exists() and produced != out_file:
        produced.replace(out_file)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemini 2.5 Flash vision → RadGraph-style KG + optional GT viz.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--projection", default="Frontal")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--max_output_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="Output file (.json) OR directory.")
    ap.add_argument("--debug", action="store_true")

    # ground truth (for reference + viz)
    ap.add_argument("--gt_jsonl", default=None)
    ap.add_argument("--gt_uid", type=int, default=None)
    ap.add_argument("--gt_filename", default=None)

    # visualization
    ap.add_argument("--viz", action="store_true", help="Render KG image(s) using Graphviz.")
    ap.add_argument("--viz_format", default="png", choices=["png", "svg", "pdf"])
    ap.add_argument("--viz_out", default=None, help="Directory OR file path base to write KG images.")
    ap.add_argument("--viz_title", default=None, help="Optional title at top of the predicted graph.")

    args = ap.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    out_path = resolve_out_path(args.out, image_path)

    prompt = PROMPT_TMPL.safe_substitute(projection=args.projection)

    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Put GEMINI_API_KEY in backend/.env or pass --api_key.")

    client = genai.Client(api_key=api_key)

    mime = infer_mime(image_path)
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime)

    resp = client.models.generate_content(
        model=args.model,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_json_schema=KGOut.model_json_schema(),
        ),
    )

    if args.debug:
        fr = None
        try:
            fr = resp.candidates[0].finish_reason
        except Exception:
            pass
        usage = getattr(resp, "usage_metadata", None)
        print(f"[DEBUG] finish_reason={fr}", file=sys.stderr)
        if usage is not None:
            print(f"[DEBUG] usage_metadata={usage}", file=sys.stderr)

    pred_kg = KGOut.model_validate_json(resp.text)
    pred_obj = filter_valid_relations(pred_kg.model_dump())

    if out_path:
        out_path.write_text(json.dumps(pred_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.debug:
            print(f"[DEBUG] wrote pred json: {out_path}", file=sys.stderr)

    # --- Load GT (if provided) ---
    gold_obj = None
    match_uid = None
    match_fname = None
    if args.gt_jsonl:
        gt_path = Path(args.gt_jsonl).expanduser()
        match_uid = args.gt_uid
        match_fname = args.gt_filename
        if match_uid is None and match_fname is None:
            match_uid = auto_uid_from_image(image_path)

        gold_obj = load_gt_from_jsonl(gt_path, uid=match_uid, filename=match_fname)
        gold_obj = filter_valid_relations(gold_obj)

        # Write GT json next to pred for reference (pretty)
        if out_path:
            gt_json_path = out_path.with_name(out_path.stem + "_gt.json")
            gt_json_path.write_text(json.dumps(gold_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.debug:
                print(f"[DEBUG] wrote gt json: {gt_json_path}", file=sys.stderr)

    # --- Render graphs ---
    if args.viz:
        # base viz path
        viz_base = resolve_viz_path(args.viz_out, image_path, args.viz_format)
        if viz_base is None:
            viz_base = Path.cwd() / "out_graph" / f"kg_{image_path.stem}.{args.viz_format}"

        # Pred graph goes to *_pred.<fmt>
        pred_viz = viz_base.with_name(viz_base.stem + "_pred" + viz_base.suffix)
        title_pred = args.viz_title or f"Pred KG for {image_path.name} ({args.projection})"
        render_kg_graphviz(pred_obj, pred_viz, title=title_pred)
        if args.debug:
            print(f"[DEBUG] wrote pred graph: {pred_viz}", file=sys.stderr)

        # GT graph (if available) goes to *_gt.<fmt>
        if gold_obj is not None:
            gt_viz = viz_base.with_name(viz_base.stem + "_gt" + viz_base.suffix)
            key = f"uid={match_uid}" if match_uid is not None else (f"filename={match_fname}" if match_fname else "ground-truth")
            title_gt = f"GT KG ({key}) for {image_path.name} ({args.projection})"
            render_kg_graphviz(gold_obj, gt_viz, title=title_gt)
            if args.debug:
                print(f"[DEBUG] wrote gt graph: {gt_viz}", file=sys.stderr)

    # Print minified pred JSON to stdout
    print(json.dumps(pred_obj, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
