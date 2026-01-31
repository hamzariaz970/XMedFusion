# synthesis.py

import os
import re
import json
import time
from pathlib import Path
from langchain_community.chat_models import ChatOllama

# NEW: validator (create validators.py as shown below, or paste it into this file)
from validators import validate_report

# Import from draft.py
from draft import (
    RetrievalAgent,
    LocalLLMReportAgent,
    reports_dict,
    model,
    preprocess,
    device,
    truncate_report,
)

# Import from vision.py
from vision import (
    get_visual_embeddings,
    embeddings_to_text,
    vision_model,
    proj_heads,
    LocalLLMReportAgent as VisionLLMAgent,
)

# Import from kg_agent.py
try:
    from kg_agent import infer_kg
    KG_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Could not import 'kg_agent'. Knowledge Graph features will be disabled.")
    KG_AVAILABLE = False


# -------------------------------
# Helper: Format KG for LLM
# -------------------------------
def format_kg_for_prompt(kg_data):
    if not kg_data or "entities" not in kg_data:
        return "No Knowledge Graph detected."

    entities = kg_data.get("entities", [])
    present_entities = []
    absent_entities = []

    for _, (text, label) in enumerate(entities):
        if "absent" in label:
            absent_entities.append(f"- {text} is ABSENT")
        elif "present" in label:
            present_entities.append(f"- {text} is PRESENT ({label.split('::')[0]})")

    kg_text = "DETECTED FINDINGS (GROUND TRUTH):\n"
    if present_entities:
        kg_text += "PRESENT:\n" + "\n".join(present_entities) + "\n"
    if absent_entities:
        kg_text += "ABSENT (Rule out):\n" + "\n".join(absent_entities)

    return kg_text


# -------------------------------
# Local Synthesis Agent
# -------------------------------
class LocalSynthesisAgent:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        self.llm = ChatOllama(model=model_name, temperature=0.1)

    def repair_report(self, bad_report: str, errors: list[str], kg_text_block: str) -> str:
        err_txt = "\n".join([f"- {e}" for e in errors])

        prompt = f"""
You are a strict radiology report editor.

This report FAILED validation:
{err_txt}

Rewrite the report to fix ALL issues.

Rules:
- Output ONLY these sections in this exact order:
FINDINGS:
IMPRESSION:
LABELS:
RECOMMENDATIONS:
- FINDINGS: one narrative paragraph, no bullets
- IMPRESSION: 1-2 sentences
- LABELS: comma-separated list ONLY (one line)
- RECOMMENDATIONS: one sentence (or "None" if normal)
- Obey KG facts strictly:
{kg_text_block}

Here is the report to fix:
{bad_report}

Return ONLY the corrected report.
"""
        resp = self.llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # strip accidental code fences if any
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
        return text

    # CHANGED: Generator (yields status)
    def generate_final_report(
        self,
        draft_agent,
        vision_agent,
        retrieval_agent,
        reports_dict,
        image_paths,
    ):
        print(f"\n🚀 Starting Synthesis Pipeline for {len(image_paths)} image(s)...")
        target_image = image_paths[0]

        # 1️⃣ Vision Agent
        print("\n👁️  Running Vision Agent...")
        yield json.dumps({"status": "vision_start", "message": "Analyzing visual features..."}) + "\n"

        img_embed = get_visual_embeddings(image_paths, vision_model, proj_heads)
        if img_embed is not None:
            vision_features = embeddings_to_text(img_embed)
            vision_report = vision_agent.generate_report(vision_features)
        else:
            vision_report = "No visual features could be extracted."

        print(f"✅ Vision Report Generated ({len(vision_report)} chars)")
        yield json.dumps({"status": "vision_done"}) + "\n"

        # 2️⃣ Draft Agent
        print("\n📚 Running Draft Agent...")
        yield json.dumps({"status": "draft_start", "message": "Retrieving similar cases..."}) + "\n"

        top_reports = retrieval_agent.retrieve_top_k(target_image, reports_dict)
        draft_context = "\n\n".join(truncate_report(r, 75) for r in top_reports)
        draft_report = draft_agent.generate_report(draft_context)

        print(f"✅ Draft Report Generated ({len(draft_report)} chars)")
        yield json.dumps({"status": "draft_done"}) + "\n"

        # 3️⃣ Knowledge Graph Agent
        print("\n🕸️  Running Knowledge Graph Agent...")
        yield json.dumps({"status": "kg_start", "message": "Constructing Knowledge Graph..."}) + "\n"

        kg_text_block = "KG generation skipped."
        kg_json = None

        if KG_AVAILABLE:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    kg_json = infer_kg(target_image, projection="Frontal", thinking_budget=0)
                    kg_text_block = format_kg_for_prompt(kg_json)
                    print("✅ Knowledge Graph Extracted successfully")
                    break
                except Exception as e:
                    if "503" in str(e) or "overloaded" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2
                            print(f"⚠️ Model overloaded. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            kg_text_block = "Error: Knowledge Graph unavailable (Server Overloaded)."
                    else:
                        print(f"❌ Error generating KG: {e}")
                        kg_text_block = "Error generating Knowledge Graph."
                        break

        yield json.dumps({"status": "kg_done"}) + "\n"

        # 4️⃣ Synthesis (Final Combined Report)
        print("\n📝 GENERATING FINAL COMBINED REPORT...")
        yield json.dumps({"status": "synthesis_start", "message": "Synthesizing final report..."}) + "\n"

        synthesis_prompt = f"""
You are an expert thoracic radiologist. Write a final radiology report.

### INPUT DATA SOURCES (Use these to form your report):
1. **KNOWLEDGE GRAPH (FACTS)**: {kg_text_block}
- STRICT RULE: If KG says "ABSENT", the finding is NOT present. If "PRESENT", it IS present.
- WARNING: "Normal cardiomegaly" is impossible. If Normal -> "Heart size is normal". If Cardiomegaly -> "Heart size is enlarged".

2. **DRAFT REPORT (STYLE)**: {draft_report}
3. **VISION REPORT (VISUALS)**: {vision_report}
- DO NOT talk about angles, lines and dimensions that come from vision report.
  You can only talk about actual medical findings.

### OUTPUT INSTRUCTIONS:
- You must output ONLY the four sections below.
- DO NOT output "Source 1", "Vision Report", "Analysis", or "Overview".
- DO NOT include any introductory text like "Here is the report".
- Make sure all the facts from the KNOWLEDGE GRAPH are accurately presented.
- Ensure that all input sources have been considered.

### REQUIRED FINAL OUTPUT FORMAT:

FINDINGS:
[Write a continuous narrative paragraph here. Describe lungs, heart, pleura, and bones clearly. Do NOT use bullet points.]

IMPRESSION:
[1-2 sentence summary of the diagnosis.]

LABELS:
[Comma-separated list of conditions found, e.g., Cardiomegaly, Pleural Effusion, Normal.]

RECOMMENDATIONS:
[One sentence recommendation. If normal, write "None". If abnormal, suggest "Clinical correlation recommended" or "Follow-up CT", or a relevant suggestion.]
"""

        full_response = ""
        for chunk in self.llm.stream(synthesis_prompt):
            content = getattr(chunk, "content", None)
            if content:
                print(content, end="", flush=True)
                full_response += content

        print("\n" + "=" * 60)

        # CLEANUP
        final_report = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        # remove accidental leading ``` blocks
        final_report = re.sub(r"^```(?:json)?\s*", "", final_report, flags=re.IGNORECASE).strip()
        final_report = re.sub(r"\s*```$", "", final_report).strip()

        # 5️⃣ AGENTIC LOOP: Validate -> Repair -> Re-validate
        MAX_ITERS = 2  # keep small for latency / streaming demo

        for it in range(MAX_ITERS + 1):
            yield json.dumps({"status": "validate_start", "iter": it}) + "\n"
            v = validate_report(final_report, kg_json)
            yield json.dumps(
                {"status": "validate_done", "iter": it, "ok": v["ok"], "errors": v["errors"]}
            ) + "\n"

            if v["ok"]:
                break

            if it == MAX_ITERS:
                break

            yield json.dumps({"status": "repair_start", "iter": it}) + "\n"
            final_report = self.repair_report(final_report, v["errors"], kg_text_block)
            yield json.dumps({"status": "repair_done", "iter": it}) + "\n"

        # 6️⃣ COMPLETE
        yield json.dumps(
            {"status": "complete", "final_report": final_report, "knowledge_graph": kg_json}
        ) + "\n"


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    print(f"Using device: {device}")

    target_image = "data/iu_xray/images/CXR3655_IM-1817/0.png"

    if not os.path.exists(target_image):
        if os.path.exists("data/iu_xray/images"):
            found = None
            for root, _, files in os.walk("data/iu_xray/images"):
                for f in files:
                    if f.endswith(".png"):
                        found = os.path.join(root, f)
                        break
                if found:
                    break
            if found:
                target_image = found

    if os.path.exists(target_image):
        image_paths = [target_image]

        retrieval_agent = RetrievalAgent(model, preprocess, k=5, device=device)
        draft_agent = LocalLLMReportAgent(model_name="deepseek-r1:1.5b")
        vision_agent = VisionLLMAgent(model_name="deepseek-r1:1.5b")
        synthesis_agent = LocalSynthesisAgent(model_name="deepseek-r1:1.5b")

        gen = synthesis_agent.generate_final_report(
            draft_agent=draft_agent,
            vision_agent=vision_agent,
            retrieval_agent=retrieval_agent,
            reports_dict=reports_dict,
            image_paths=image_paths,
        )

        output_dir = "XMedAgent/out"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "final_report.txt")

        final_text = ""
        final_payload = None
        for chunk in gen:
            try:
                data = json.loads(chunk)
            except Exception:
                continue
            if data.get("status") == "complete":
                final_payload = data
                final_text = data.get("final_report", "")
                break

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"\n📄 Final report saved to: {save_path}")
        if final_payload and final_payload.get("knowledge_graph") is not None:
            print("✅ Knowledge graph included in final payload.")
    else:
        print("❌ Error: No valid image found.")
