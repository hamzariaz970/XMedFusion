import os
from PIL import Image, ImageDraw, ImageFont

# Zone Definitions (Colors and relative coordinates)
ZONES = {
    "right lung": {"x1_pct": 0.02, "y1_pct": 0.02, "x2_pct": 0.48, "y2_pct": 0.98, "color": (255, 99, 71, 220)},    # Red
    "mediastinum": {"x1_pct": 0.30, "y1_pct": 0.05, "x2_pct": 0.70, "y2_pct": 0.95, "color": (60, 179, 113, 220)}, # Green
    "left lung": {"x1_pct": 0.52, "y1_pct": 0.02, "x2_pct": 0.98, "y2_pct": 0.98, "color": (30, 144, 255, 220)}     # Blue
}

def parse_kg_for_visuals(kg_data):
    """Maps diseases to their anatomical zones."""
    if not kg_data or "entities" not in kg_data: return {}
    entities = kg_data.get("entities", [])
    relations = kg_data.get("relations", [])
    zone_findings = {}
    
    for r in relations:
        if r[2] in ["located_at", "modify"]:
            if r[0] < len(entities) and r[1] < len(entities):
                obs = entities[r[0]][0].title()
                anat = entities[r[1]][0].lower()
                if anat not in zone_findings: zone_findings[anat] = []
                zone_findings[anat].append(obs)
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

def generate_explainable_image(image_path, kg_data, output_path):
    try:
        base_img = Image.open(image_path).convert("RGBA")
        width, height = base_img.size
        
        overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Responsive Font Sizing
        font_size = max(14, int(height * 0.025))
        try: font = ImageFont.truetype("arial.ttf", font_size)
        except IOError: font = ImageFont.load_default()

        findings_map = parse_kg_for_visuals(kg_data)
        has_findings = False
        
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