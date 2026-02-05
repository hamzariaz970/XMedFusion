# synthesis.py

import os
import re
import json
import time
import asyncio  # <--- REQUIRED FOR PARALLELISM
from pathlib import Path
from langchain_community.chat_models import ChatOllama

# NEW: validator (ensure validators.py exists)
from validators import validate_report
import config  # <--- GLOBAL CONFIG

# Import from draft.py
from draft import (
    RetrievalAgent,
    LocalLLMReportAgent,
    reports_dict,
    model as clip_model,      # Explicitly import CLIP model
    preprocess as clip_prep,  # Explicitly import CLIP preprocess
    device,
    truncate_report,
    zero_shot_classify        # <--- CRITICAL NEW IMPORT
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
    def __init__(self, model_name=config.OLLAMA_MODEL): 
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        
        self.llm = ChatOllama(
            model=model_name, 
            temperature=config.TEMPERATURE
        )

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
        content = resp.content if hasattr(resp, "content") else str(resp)
        
        # Standard cleanup for repair output
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"\s*```$", "", content).strip()
        
        # Post-repair headers fix
        headers = ["FINDINGS", "IMPRESSION", "LABELS", "RECOMMENDATIONS"]
        for h in headers:
            content = re.sub(f"({h})[: ]*(?=[A-Z])", f"{h}:\n", content, flags=re.IGNORECASE)
            
        return content

    # CHANGED: ASYNC Generator for Parallel Execution
    async def generate_final_report(
        self,
        draft_agent,
        vision_agent,
        retrieval_agent,
        reports_dict,
        image_paths,
    ):
        print(f"\n🚀 Starting PARALLEL Synthesis Pipeline for {len(image_paths)} image(s)...")
        target_image = image_paths[0]

        # Yield initial status
        yield json.dumps({"status": "parallel_start", "message": "Running Vision, Draft, and KG agents in parallel..."}) + "\n"

        # --- DEFINE ASYNC TASKS ---

        # Task 1: Vision (Zero-Shot)
        async def run_vision():
            return await asyncio.to_thread(
                zero_shot_classify, target_image, clip_model, clip_prep, device
            )

        # Task 2: Draft (Retrieval + LLM)
        async def run_draft():
            # CPU bound retrieval
            top_reports = await asyncio.to_thread(
                retrieval_agent.retrieve_top_k, target_image, reports_dict
            )
            draft_context = "\n\n".join(truncate_report(r, 75) for r in top_reports)
            # IO bound generation
            return await asyncio.to_thread(
                draft_agent.generate_report, draft_context
            )

        # Task 3: Knowledge Graph (API Call)
        async def run_kg():
            if not KG_AVAILABLE:
                return None, "KG generation skipped."
            
            kg_text = "KG generation skipped."
            kg_data = None
            
            # Simple retry logic inside the thread wrapper
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    kg_data = await asyncio.to_thread(
                        infer_kg, target_image, projection="Frontal", thinking_budget=0
                    )
                    kg_text = format_kg_for_prompt(kg_data)
                    break
                except Exception as e:
                    if "503" in str(e) or "overloaded" in str(e).lower():
                        if attempt < max_retries - 1:
                            await asyncio.sleep((attempt + 1) * 2)
                        else:
                            kg_text = "Error: KG Server Overloaded."
                    else:
                        print(f"KG Error: {e}")
                        kg_text = "Error generating Knowledge Graph."
                        break
            return kg_data, kg_text

        # --- EXECUTE PARALLEL GATHER ---
        # This runs all 3 agents simultaneously 
        vision_report, draft_report, (kg_json, kg_text_block) = await asyncio.gather(
            run_vision(),
            run_draft(),
            run_kg()
        )

        # Notify completion of parallel phase
        print(f"✅ Vision Detected: {vision_report}")
        print(f"✅ Draft Generated: {len(draft_report)} chars")
        if kg_json: print("✅ KG Extracted")
        
        yield json.dumps({"status": "parallel_done", "message": "Agents finished. Synthesizing..."}) + "\n"

        # ---------------------------------------------------------
        # 4️⃣ Synthesis (Final Combined Report - Sequential Phase)
        # ---------------------------------------------------------
        print("\n📝 GENERATING FINAL COMBINED REPORT...")
        yield json.dumps({"status": "synthesis_start", "message": "Synthesizing final report..."}) + "\n"

  # ReAct-Style Prompt (Updated for MedGemma / IU X-Ray Style)
        synthesis_prompt = f"""
You are an expert thoracic radiologist. Write a final radiology report.

### INPUT DATA:
1. **KNOWLEDGE GRAPH (FACTS)**: 
{kg_text_block}
(Strictly obey these findings. If ABSENT, you must rule it out.)

2. **VISION AI ANALYSIS**: 
{vision_report}
(These are probabilities. Only mention if high probability or consistent with KG.)

3. **REFERENCE CASES**: 
{draft_report}
(Use this purely for writing style and sentence structure.)

### CONFLICT RESOLUTION:
- If KG says "Normal" but Vision says "Cardiomegaly", TRUST KG.
- If Vision says "Edema" but KG says "ABSENT: Edema", TRUST KG.

### STYLE GUIDELINES (CRITICAL):
- **Systematic Approach:** You MUST write at least one sentence for EACH of these regions:
  1. **Heart & Mediastinum:** (e.g., "The cardiac silhouette is within normal limits.")
  2. **Lungs:** (e.g., "No focal consolidation or edema.")
  3. **Pleura:** (e.g., "No pleural effusion or pneumothorax.")
  4. **Bones:** (e.g., "No acute bony abnormality.")
- **Pertinent Negatives:** Do not just say "Normal." Explicitly state what is NOT there (e.g., "There is no pneumothorax").
- **Narrative Flow:** Combine these into a smooth paragraph. Do not use bullet points.

### EXAMPLES OF DESIRED OUTPUT (MIMIC THIS EXACTLY):
Example 1:
"The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema or focal consolidation. No pleural effusion or pneumothorax is seen. The osseous structures are unremarkable."

Example 2:
"The heart size is normal. The mediastinal contour is within normal limits. The lungs are clear without focal consolidation or effusion. There is no evidence of pneumothorax."

### OUTPUT INSTRUCTIONS:
- Generate the report based on the INPUT DATA but using the SYSTEMATIC APPROACH above.
- Ensure "Labels" are just a comma-separated list.

### REQUIRED OUTPUT FORMAT:

FINDINGS:
[Write the systematic narrative here.]

IMPRESSION:
[1 sentence summary, e.g., "No acute cardiopulmonary process."]

LABELS:
[Comma-separated list e.g. Cardiomegaly, Normal, etc.]

RECOMMENDATIONS:
[One sentence or "None".]
"""

        full_response = ""
        # Stream the LLM response
        async for chunk in self.llm.astream(synthesis_prompt):
            content = getattr(chunk, "content", None)
            if content:
                print(content, end="", flush=True)
                full_response += content

        print("\n" + "=" * 60)

        # === IMPROVED CLEANUP BLOCK ===
        
        # 1. Remove <think> tags
        final_report = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        
        # 2. Fix standard formatting
        final_report = re.sub(r"^```(?:json)?\s*", "", final_report, flags=re.IGNORECASE).strip()
        final_report = re.sub(r"\s*```$", "", final_report).strip()

        # 3. Force Newlines BEFORE Headers 
        # (Fixes "Thoracic RECOMMENDATIONS" -> "Thoracic\nRECOMMENDATIONS")
        headers = ["FINDINGS", "IMPRESSION", "LABELS", "RECOMMENDATIONS"]
        for h in headers:
            # Look for header (case insensitive) preceded by a word character
            final_report = re.sub(r"(?<=\w)[ \t]*(" + h + r":?)", r"\n\n\1", final_report, flags=re.IGNORECASE)

        # 4. Force Newlines AFTER Headers
        # (Fixes "LabelsScoliosis" -> "Labels:\nScoliosis")
        for h in headers:
            final_report = re.sub(f"({h})[: ]*(?=[A-Z])", f"{h}:\n", final_report, flags=re.IGNORECASE)

        # 5. Fix Specific Typos
        final_report = final_report.replace("Espaça", "Airspace")
        final_report = final_report.replace("Espaco", "Airspace")
        
        # 6. Fix "Normal Cardiomegaly" Contradiction
        if "Normal" in final_report and "Cardiomegaly" in final_report:
            if "Cardiomegaly" not in kg_text_block: 
                 final_report = final_report.replace("Cardiomegaly", "")

        # 7. Final Trim
        final_report = final_report.strip()

        # AGENTIC LOOP: Validate -> Repair -> Re-validate
        MAX_ITERS = 3

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
            # Note: repair_report is synchronous, wrap it if strictly needed, but it's fast enough usually
            final_report = await asyncio.to_thread(self.repair_report, final_report, v["errors"], kg_text_block)
            yield json.dumps({"status": "repair_done", "iter": it}) + "\n"

        # 6️⃣ COMPLETE
        yield json.dumps(
            {"status": "complete", "final_report": final_report, "knowledge_graph": kg_json}
        ) + "\n"


# -------------------------------
# Example Usage (Test Block)
# -------------------------------
if __name__ == "__main__":
    # Helper for running async main in script
    async def main_test():
        print(f"Using device: {device}")
        print(f"Global Model: {config.OLLAMA_MODEL}")

        target_image = "data/iu_xray/images/CXR3655_IM-1817/0.png"
        
        # (Image search logic omitted for brevity, assuming path exists for test)
        if not os.path.exists(target_image):
             # minimal fallback check
             if os.path.exists("data/iu_xray/images"):
                for root, _, files in os.walk("data/iu_xray/images"):
                    for f in files:
                        if f.endswith(".png"):
                            target_image = os.path.join(root, f)
                            break
                    if target_image: break

        if os.path.exists(target_image):
            image_paths = [target_image]

            # Initialize Agents (Using Global Config Defaults)
            retrieval_agent = RetrievalAgent(clip_model, clip_prep, k=5, device=device)
            draft_agent = LocalLLMReportAgent()     # Uses config.OLLAMA_MODEL
            vision_agent = VisionLLMAgent()         # Uses config.OLLAMA_MODEL
            synthesis_agent = LocalSynthesisAgent() # Uses config.OLLAMA_MODEL

            # Run Pipeline
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
            
            # Consume ASYNC Generator
            async for chunk in gen:
                try:
                    data = json.loads(chunk)
                    if "status" in data:
                        print(f"[{data['status']}] {data.get('message', '')}")
                    
                    if data.get("status") == "complete":
                        final_text = data.get("final_report", "")
                        if data.get("knowledge_graph"):
                            print("✅ Knowledge graph included.")
                except Exception:
                    continue

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(f"\n📄 Final report saved to: {save_path}")
        else:
            print("❌ Error: No valid image found.")

    # Execute Async Main
    asyncio.run(main_test())