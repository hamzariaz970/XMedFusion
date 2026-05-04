"""
explain.py — Clinical explainability image generator for XMedFusion.

Generates a medically meaningful overlay on X-ray / CT scans by:
  - Using disease-specific anatomical bounding boxes calibrated to PA chest X-ray anatomy
  - Displaying classifier confidence scores for each finding
  - Color-coding by confidence tier (Present=red, Possible=amber)
  - Drawing a minimal clinical legend
"""

import os
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Disease-specific anatomical ROIs for standard PA chest X-ray
# Each disease maps to a LIST of boxes: [(x1, y1, x2, y2), ...]
# Coordinates are fractions of (width, height)
# ---------------------------------------------------------------------------
DISEASE_ROI = {
    # Heart: Centered in the lower mediastinum
    "Cardiomegaly":    [(0.35, 0.45, 0.65, 0.88)],
    
    # Pleural Effusion: Costophrenic angles (bottom corners)
    "Pleural Effusion": [(0.08, 0.78, 0.38, 0.96), (0.62, 0.78, 0.92, 0.96)],
    
    # Pneumothorax: Lung apices (top outer regions)
    "Pneumothorax":    [(0.10, 0.05, 0.40, 0.28), (0.60, 0.05, 0.90, 0.28)],
    
    # Pulmonary Edema: Perihilar regions (butterfly pattern)
    "Edema":           [(0.25, 0.35, 0.45, 0.65), (0.55, 0.35, 0.75, 0.65)],
    
    # Infiltrate / Consolidation / Opacity: Mid-to-lower lung fields
    "Infiltrate":      [(0.10, 0.35, 0.42, 0.85), (0.58, 0.35, 0.90, 0.85)],
    "Consolidation":   [(0.12, 0.40, 0.40, 0.82), (0.60, 0.40, 0.88, 0.82)],
    "Lung Opacity":    [(0.10, 0.20, 0.45, 0.88), (0.55, 0.20, 0.90, 0.88)],
    
    # Atelectasis: Lung bases (just above diaphragm)
    "Atelectasis":     [(0.10, 0.65, 0.45, 0.92), (0.55, 0.65, 0.90, 0.92)],
    
    # Nodule: Random parenchymal spots (modeled as generic lung zones)
    "Nodule":          [(0.15, 0.15, 0.40, 0.40), (0.60, 0.15, 0.85, 0.40)],
    
    # Fracture: Rib cage and clavicles (outer periphery and top)
    "Fracture":        [(0.02, 0.10, 0.15, 0.80), (0.85, 0.10, 0.98, 0.80), (0.20, 0.02, 0.80, 0.15)],
}

# ---------------------------------------------------------------------------
# Dynamic Color Palette — A set of high-contrast distinct hues
# ---------------------------------------------------------------------------
COLOR_PALETTE = [
    (231, 76,  60),   # Red
    (52,  152, 219),  # Blue
    (46,  204, 113),  # Green
    (155, 89,  182),  # Purple
    (241, 196, 15),   # Yellow
    (230, 126, 34),   # Orange
    (26,  188, 156),  # Teal
    (189, 195, 199),  # Gray
    (255, 105, 180),  # Pink
    (52,  73,  94),   # Dark Slate
    (160, 82,  45),   # Brown
    (22,  160, 133),  # Dark Cyan
]

COLOR_LABEL_BG = (10,  10,  10) # Near-black label backgrounds


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.Draw, text: str, font) -> tuple[int, int]:
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _draw_finding_box(
    draw: ImageDraw.Draw,
    overlay_draw: ImageDraw.Draw,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color_rgb: tuple,
    confidence: float,
    font_label: ImageFont.FreeTypeFont,
    font_conf: ImageFont.FreeTypeFont,
    label_y_used: dict | None = None,
    draw_label: bool = True,
):
    """Draw a single disease bounding box with visual confidence weighting."""
    # ── Compute visual weight based on confidence ──────────────────────────
    if confidence >= 0.80:
        border_w   = 5
        fill_alpha = 48  # Stronger highlight
        tick_w     = 6
    elif confidence >= 0.40:
        border_w   = 3
        fill_alpha = 28  # Standard highlight
        tick_w     = 4
    else:
        border_w   = 1
        fill_alpha = 14  # Subtle highlight for borderline findings
        tick_w     = 2

    border_color = color_rgb + (220,)
    fill_color   = color_rgb + (fill_alpha,)

    # Semi-transparent fill
    overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
    # Solid border
    draw.rectangle([x1, y1, x2, y2], outline=border_color, width=border_w)

    # Corner tick marks (emphasize the ROI)
    tick_len = 14
    for (tx1, ty1, tx2, ty2) in [
        (x1, y1, x1 + tick_len, y1), (x1, y1, x1, y1 + tick_len),
        (x2 - tick_len, y1, x2, y1), (x2, y1, x2, y1 + tick_len),
        (x1, y2 - tick_len, x1, y2), (x1, y2, x1 + tick_len, y2),
        (x2 - tick_len, y2, x2, y2), (x2, y2 - tick_len, x2, y2),
    ]:
        draw.line([(tx1, ty1), (tx2, ty2)], fill=border_color, width=tick_w)

    if not draw_label:
        return

    # ---- Unified Label pill ----
    # Strip any status prefixes from the label for a clean display
    display_label = label
    for prefix in ["Possible ", "Likely ", "Present ", "Uncertain "]:
        if display_label.startswith(prefix):
            display_label = display_label[len(prefix):]
    
    conf_pct = f"{int(confidence * 100)}%"
    main_text = f" {display_label} "
    conf_text = f" {conf_pct} "

    tw_main, th = _text_size(draw, main_text, font_label)
    tw_conf, _  = _text_size(draw, conf_text, font_conf)

    pill_h  = th + 10
    pill_w  = tw_main + tw_conf + 6
    px = max(x1, x1 + 4)

    base_py = max(4, y1 - pill_h - 4)
    if label_y_used is not None:
        x_bucket = px // 80
        occupied_y = label_y_used.get(x_bucket, -999)
        if base_py < occupied_y + pill_h + 4:
            base_py = occupied_y + pill_h + 6
        label_y_used[x_bucket] = base_py
    py = base_py

    # Unified label background
    draw.rectangle([px, py, px + tw_main, py + pill_h],
                   fill=COLOR_LABEL_BG + (230,), outline=border_color, width=2)
    draw.text((px + 2, py + 4), main_text, fill=(255, 255, 255, 255), font=font_label)

    # Percentage badge
    draw.rectangle([px + tw_main, py, px + pill_w, py + pill_h],
                   fill=color_rgb + (230,), outline=border_color, width=2)
    draw.text((px + tw_main + 2, py + 4), conf_text, fill=(255, 255, 255, 255), font=font_conf)


def generate_explainable_image(image_path, kg_data, output_path, modality="xray"):
    """
    Main entry point called by synthesis.py.

    For X-rays: renders disease-specific anatomical bounding boxes with
    classifier confidence scores extracted from kg_data["metadata"]["adjudicated_findings"].

    For CT: keeps the existing cell-highlight approach.
    """
    try:
        base_img = Image.open(image_path).convert("RGBA")
        width, height = base_img.size

        # Overlay for semi-transparent fills (alpha composite later)
        fill_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        fill_draw  = ImageDraw.Draw(fill_layer)

        # Main draw for crisp borders, labels, legend
        border_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw         = ImageDraw.Draw(border_layer)

        font_size_label = max(13, int(height * 0.022))
        font_size_conf  = max(12, int(height * 0.019))

        font_label = _load_font(font_size_label)
        font_conf  = _load_font(font_size_conf)

        has_findings = False

        # ---------------------------------------------------------------
        # CT PATH — highlight relevant montage cells (existing approach)
        # ---------------------------------------------------------------
        if modality == "ct":
            metadata       = (kg_data or {}).get("metadata", {})
            report_findings = metadata.get("report_findings", {})
            ct_highlights  = metadata.get("ct_highlights", [])
            montage_meta   = metadata.get("ct_montage", {})

            cols   = max(1, int(montage_meta.get("cols", 4)))
            rows   = max(1, int(montage_meta.get("rows", 4)))
            cell_w = width  // cols
            cell_h = height // rows

            rendered = []
            if ct_highlights:
                for item in ct_highlights:
                    if not isinstance(item, dict): continue
                    try:
                        slice_idx = int(item.get("slice_index"))
                    except (TypeError, ValueError):
                        continue
                    if slice_idx < 1: continue
                    rendered.append({
                        "label":       str(item.get("label", "Finding")).strip() or "Finding",
                        "slice_index": slice_idx,
                        "status":      str(item.get("status", "present")).lower(),
                        "confidence":  float(item.get("confidence", 0.6)),
                    })
            else:
                for disease, data in report_findings.items():
                    if data.get("status") in ["present", "uncertain"] and data.get("slice_index") is not None:
                        rendered.append({
                            "label":       disease,
                            "slice_index": data["slice_index"],
                            "status":      data.get("status", "present"),
                            "confidence":  float(data.get("confidence", 0.6)),
                        })

            for item in rendered:
                sl   = item["slice_index"]
                idx  = max(0, min((rows * cols) - 1, sl - 1))
                col  = idx % cols
                row  = idx // cols
                bx1  = col * cell_w
                by1  = row * cell_h
                bx2  = bx1 + cell_w
                by2  = by1 + cell_h

                # Use first color in palette for CT highlights
                color = COLOR_PALETTE[0]
                fill_draw.rectangle([bx1, by1, bx2, by2], fill=color + (50,))
                draw.rectangle([bx1, by1, bx2, by2], outline=color + (220,), width=2)

                label_text = f" {item['label']} ({int(item['confidence'] * 100)}%) "
                tw, th = _text_size(draw, label_text, font_label)
                draw.rectangle([bx1 + 4, by1 + 4, bx1 + tw + 8, by1 + th + 12],
                               fill=COLOR_LABEL_BG + (200,))
                draw.text((bx1 + 6, by1 + 6), label_text,
                          fill=(255, 255, 255, 255), font=font_label)
                has_findings = True

        # ---------------------------------------------------------------
        # X-RAY PATH — disease-specific anatomical ROIs + confidence
        # ---------------------------------------------------------------
        else:
            # Pull adjudicated findings from KG metadata (most informative source)
            adjudicated = (
                (kg_data or {})
                .get("metadata", {})
                .get("adjudicated_findings", {})
            )

            # Fallback: parse entities/relations if metadata is absent
            if not adjudicated:
                adjudicated = _build_adjudicated_from_entities(kg_data)

            # Collect all renderable findings first so we can sort by confidence
            # and avoid drawing zero-confidence noise
            renderable = []
            for disease, finding in adjudicated.items():
                status = finding.get("status", "not_mentioned")
                if status not in ("present", "uncertain"):
                    continue

                clf       = finding.get("classifier") or {}
                raw_score = float(clf.get("score", 0.0))
                threshold = float(clf.get("threshold", 1.0))
                conf_str  = finding.get("confidence", "none")  # "strong" | "candidate_only" | ...

                # ── Compute a normalised display confidence ───────────────
                # Strategy: use the raw score directly (it is already 0-1 from
                # the ensemble classifiers). score/threshold can exceed 1.0 when
                # the model fires well above the decision boundary — cap at 100%.
                if threshold > 0 and threshold <= 1.0:
                    display_confidence = min(1.0, raw_score / threshold)
                else:
                    display_confidence = min(1.0, raw_score)

                # For "strong" (create_kg_edge) findings, boost to at least the raw score
                if conf_str == "strong" and raw_score > 0:
                    display_confidence = min(1.0, max(display_confidence, raw_score))

                # ── FILTER: skip below 15% — pure noise ──────────────────
                MIN_DISPLAY_CONFIDENCE = 0.15
                if display_confidence < MIN_DISPLAY_CONFIDENCE:
                    continue

                # ── ASSIGN DYNAMIC COLOR ───────────────────────────────────
                # Cycle through the palette based on finding index
                color_idx = len(renderable) % len(COLOR_PALETTE)
                color = COLOR_PALETTE[color_idx]

                renderable.append({
                    "disease":    disease,
                    "confidence": display_confidence,
                    "color":      color,
                })

            # Sort high-confidence on top so red boxes are drawn last (visible)
            renderable.sort(key=lambda x: x["confidence"])

            # Track label pill y-positions per x-band to avoid overlap
            label_y_used: dict[int, int] = {}

            for item in renderable:
                disease = item["disease"]
                # Get ROIs for this disease (list of boxes)
                rois_pct = DISEASE_ROI.get(disease, [(0.05, 0.05, 0.95, 0.95)])
                
                # Draw each ROI in the set
                for i, roi_pct in enumerate(rois_pct):
                    x1 = int(roi_pct[0] * width)
                    y1 = int(roi_pct[1] * height)
                    x2 = int(roi_pct[2] * width)
                    y2 = int(roi_pct[3] * height)

                    # Only draw the label on the FIRST box of the set to avoid clutter
                    draw_label = (i == 0)

                    _draw_finding_box(
                        draw, fill_draw,
                        x1, y1, x2, y2,
                        item["disease"], 
                        item["color"], 
                        item["confidence"],
                        font_label, font_conf,
                        label_y_used=label_y_used,
                        draw_label=draw_label
                    )
                has_findings = True

        if not has_findings:
            # Draw a "Normal Study" badge in the top-right corner
            badge_text = " STUDY APPEARS NORMAL "
            tw, th = _text_size(draw, badge_text, font_label)
            margin = 20
            bx1 = width - tw - margin - 10
            by1 = margin
            bx2 = width - margin
            by2 = margin + th + 10
            
            draw.rectangle([bx1, by1, bx2, by2], fill=(46, 204, 113, 220), outline=(255, 255, 255, 255), width=2)
            draw.text((bx1 + 5, by1 + 5), badge_text, fill=(255, 255, 255, 255), font=font_label)

        # Composite: base → fill layer → border/label layer
        composed = Image.alpha_composite(base_img, fill_layer)
        composed = Image.alpha_composite(composed, border_layer)
        composed.convert("RGB").save(output_path)
        return output_path

    except Exception as e:
        print(f"[explain.py] Error generating explainability image: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Fallback parser: build a minimal adjudicated dict from raw KG entities
# ---------------------------------------------------------------------------
def _build_adjudicated_from_entities(kg_data) -> dict:
    """If adjudicated_findings metadata is absent, approximate from KG entities."""
    result = {}
    if not kg_data or "entities" not in kg_data:
        return result

    entities  = kg_data.get("entities", [])
    relations = kg_data.get("relations", [])

    for r in relations:
        if len(r) < 3 or r[0] >= len(entities) or r[1] >= len(entities):
            continue
        obs_entity = entities[r[0]]
        obs_text   = obs_entity[0].strip().title()
        obs_type   = str(obs_entity[1]).lower()
        rel        = str(r[2]).lower()

        if "absent" in obs_type or "absent" in rel:
            continue
        if obs_text.lower() in {"clear", "normal", "unremarkable", "intact"}:
            continue

        status = "uncertain" if "uncertain" in obs_type else "present"
        # Match to a known disease name
        matched_disease = None
        for known in DISEASE_ROI:
            if known.lower() in obs_text.lower() or obs_text.lower() in known.lower():
                matched_disease = known
                break
        if not matched_disease:
            matched_disease = obs_text

        if matched_disease not in result:
            result[matched_disease] = {
                "status":     status,
                "classifier": {"score": 0.55, "threshold": 1.0},
            }

    return result
