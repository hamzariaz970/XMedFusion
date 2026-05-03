import os
from PIL import Image, ImageDraw, ImageFont

# Zone Definitions (Colors and relative coordinates)
ZONES = {
    "right lung": {"x1_pct": 0.02, "y1_pct": 0.02, "x2_pct": 0.48, "y2_pct": 0.98, "color": (255, 99, 71, 220)},    # Red
    "mediastinum": {"x1_pct": 0.30, "y1_pct": 0.05, "x2_pct": 0.70, "y2_pct": 0.95, "color": (60, 179, 113, 220)}, # Green
    "left lung": {"x1_pct": 0.52, "y1_pct": 0.02, "x2_pct": 0.98, "y2_pct": 0.98, "color": (30, 144, 255, 220)},     # Blue
    "lungs": {"x1_pct": 0.02, "y1_pct": 0.02, "x2_pct": 0.98, "y2_pct": 0.98, "color": (255, 165, 0, 210)},
    "pleural space": {"x1_pct": 0.02, "y1_pct": 0.60, "x2_pct": 0.98, "y2_pct": 0.98, "color": (138, 43, 226, 210)},
    "bones": {"x1_pct": 0.02, "y1_pct": 0.02, "x2_pct": 0.98, "y2_pct": 0.98, "color": (220, 20, 60, 210)}
}

def parse_kg_for_visuals(kg_data):
    """Maps diseases to their anatomical zones using intelligent string matching."""
    if not kg_data or "entities" not in kg_data: return {}
    entities = kg_data.get("entities", [])
    relations = kg_data.get("relations", [])
    zone_findings = {}
    
    for r in relations:
        if len(r) >= 3 and r[0] < len(entities) and r[1] < len(entities):
            obs = entities[r[0]][0].title()
            anat = entities[r[1]][0].lower()
            rel = str(r[2]).lower()
            
            # Loosen the strict relation check. LLMs generate various relation names.
            is_valid_relation = any(k in rel for k in ["locate", "modify", "has", "show", "involve", "suggest", "present", "affect", "evidence"])
            is_anatomy = any(k in anat for k in ["lung", "heart", "mediastin", "pleura", "bone", "rib", "clavicle", "spine", "base", "hila", "aorta", "silhouette", "lobe", "zone"])
            
            if is_valid_relation or is_anatomy:
                # Intelligent mapping to base zones
                mapped_zone = "lungs" # Default fallback
                if any(k in anat for k in ["right", "rl"]): mapped_zone = "right lung"
                elif any(k in anat for k in ["left", "ll"]): mapped_zone = "left lung"
                elif any(k in anat for k in ["mediastin", "heart", "cardiac", "aorta", "hila", "silhouette"]): mapped_zone = "mediastinum"
                elif any(k in anat for k in ["pleura", "costophrenic", "base", "effusion", "fluid"]): mapped_zone = "pleural space"
                elif any(k in anat for k in ["bone", "rib", "clavicle", "spine", "fracture", "osseous"]): mapped_zone = "bones"
                else: mapped_zone = "lungs"
                
                # Filter out obvious non-findings
                if obs.lower() not in ["clear", "normal", "unremarkable", "intact"]:
                    if mapped_zone not in zone_findings: zone_findings[mapped_zone] = []
                    if obs not in zone_findings[mapped_zone]:
                        zone_findings[mapped_zone].append(obs)
                        
    return zone_findings

def apply_clinical_heuristics(zone_name, findings, x1, y1, x2, y2):
    """
    Applies anatomical logic based on the 10 specific diseases in the dataset.
    Groups diseases by where they logically appear in the zone.
    """
    grouped_boxes = {
        "top": {"findings": [], "coords": [x1, y1, x2, y1 + int((y2-y1)*0.35)]},          # Apices
        "bottom": {"findings": [], "coords": [x1, y1 + int((y2-y1)*0.65), x2, y2]},       # Bases
        "lower_mid": {"findings": [], "coords": [x1, y1 + int((y2-y1)*0.40), x2, y2]},    # Lower Heart
        "central": {"findings": [], "coords": [
            x1 + int((x2-x1)*0.3) if "right" in zone_name else x1, 
            y1 + int((y2-y1)*0.2), 
            x2 if "right" in zone_name else x1 + int((x2-x1)*0.7), 
            y2 - int((y2-y1)*0.2)
        ]}, # Perihilar / Inner lungs
        "full": {"findings": [], "coords": [x1+10, y1+10, x2-10, y2-10]}                  # General
    }

    for f in findings:
        f_lower = f.lower()
        if f_lower in ["clear", "normal"]: continue
        
        # 1. Air rises (Pneumothorax)
        if "pneumothorax" in f_lower:
            grouped_boxes["top"]["findings"].append(f)
        # 2. Fluid sinks (Pleural Effusion)
        elif "effusion" in f_lower:
            grouped_boxes["bottom"]["findings"].append(f)
        # 3. Heart sits low (Cardiomegaly)
        elif "cardiomegaly" in f_lower:
            grouped_boxes["lower_mid"]["findings"].append(f)
        # 4. Edema spreads from the center (Perihilar)
        elif "edema" in f_lower:
            grouped_boxes["central"]["findings"].append(f)
        # 5. Diffuse/Lobar/Random (Opacity, Infiltrate, Consolidation, Atelectasis, Nodule, Fracture)
        else:
            grouped_boxes["full"]["findings"].append(f)

    # Return only the boxes that actually have findings
    return {k: v for k, v in grouped_boxes.items() if v["findings"]}

def generate_explainable_image(image_path, kg_data, output_path, modality="xray"):
    try:
        base_img = Image.open(image_path).convert("RGBA")
        width, height = base_img.size
        
        overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Responsive Font Sizing
        font_size = max(14, int(height * 0.025))
        try: font = ImageFont.truetype("arial.ttf", font_size)
        except IOError: font = ImageFont.load_default()

        has_findings = False

        if modality == "ct":
            # CT Explainability: Highlight specific grid cells
            report_findings = kg_data.get("metadata", {}).get("report_findings", {})
            
            # Assume 4x4 grid as per vision_ct.py/synthesis.py
            cols, rows = 4, 4
            cell_w = width // cols
            cell_h = height // rows
            
            for disease, data in report_findings.items():
                if data.get("status") in ["present", "uncertain"] and data.get("slice_index") is not None:
                    slice_idx = data["slice_index"]
                    # 1-indexed to 0-indexed, and cap at 15
                    idx = max(0, min(15, slice_idx - 1))
                    col = idx % cols
                    row = idx // cols
                    
                    bx1 = col * cell_w
                    by1 = row * cell_h
                    bx2 = bx1 + cell_w
                    by2 = by1 + cell_h
                    
                    has_findings = True
                    
                    # Highlight the cell with a semi-transparent colored box
                    highlight_color = (255, 99, 71, 80) if data.get("status") == "present" else (255, 165, 0, 80)
                    draw.rectangle([bx1, by1, bx2, by2], fill=highlight_color, outline=(255, 255, 255, 200), width=2)
                    
                    # Add label
                    label_text = f"{disease} (Slice {slice_idx})"
                    try:
                        bbox = font.getbbox(label_text)
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except AttributeError:
                        text_w, text_h = draw.textsize(label_text, font=font)
                        
                    padding = 4
                    pill_x1 = bx1 + 5 
                    pill_y1 = by1 + 5
                    pill_x2 = pill_x1 + text_w + (padding * 2)
                    pill_y2 = pill_y1 + text_h + (padding * 2)
                    
                    draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=(0, 0, 0, 180), outline=(255,255,255,100), width=1)
                    draw.text((pill_x1 + padding, pill_y1 + padding), label_text, fill=(255, 255, 255, 255), font=font)
                    
        else:
            # X-ray Explainability (existing logic)
            findings_map = parse_kg_for_visuals(kg_data)
            
            for zone_name, findings_list in findings_map.items():
                if zone_name not in ZONES: continue
                
                # Get absolute base coordinates for the zone
                coords = ZONES[zone_name]
                base_x1 = int(coords["x1_pct"] * width)
                base_y1 = int(coords["y1_pct"] * height)
                base_x2 = int(coords["x2_pct"] * width)
                base_y2 = int(coords["y2_pct"] * height)
                
                # Get the smart, anatomically correct bounding boxes for this zone's diseases
                smart_boxes = apply_clinical_heuristics(zone_name, findings_list, base_x1, base_y1, base_x2, base_y2)
                
                for box_type, data in smart_boxes.items():
                    has_findings = True
                    bx1, by1, bx2, by2 = data["coords"]
                    
                    # 1. Draw the clean, clinical outline
                    draw.rectangle([bx1, by1, bx2, by2], fill=None, outline=coords["color"], width=3)
                    
                    # 2. Build the Text Label
                    label_text = "\n".join(f"• {f}" for f in data["findings"])
                    
                    # Calculate text dimensions
                    try:
                        bbox = font.getbbox(label_text)
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except AttributeError:
                        text_w, text_h = draw.textsize(label_text, font=font)

                    padding = 8
                    pill_x1 = bx1 + 5 
                    pill_y1 = by1 + 5
                    pill_x2 = pill_x1 + text_w + (padding * 2)
                    pill_y2 = pill_y1 + text_h + (padding * 2)
                    
                    # 3. Draw dark background for readability
                    draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=(0, 0, 0, 180), outline=coords["color"], width=1)
                    
                    # 4. Draw crisp white text
                    draw.text((pill_x1 + padding, pill_y1 + padding), label_text, fill=(255, 255, 255, 255), font=font)
                
        if has_findings:
            final_img = Image.alpha_composite(base_img, overlay).convert("RGB")
            final_img.save(output_path)
            return output_path
            
        return None 

    except Exception as e:
        print(f"Error generating visual explainability: {e}")
        return None
