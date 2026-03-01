import os
import re
import json
import asyncio
import torch
import gc
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from validators import validate_report
from PIL import Image 
import config

# <--- NEW IMPORTS --->
from explain import generate_explainable_image
from xray_filter import classify_scan  # Replaced is_chest_xray with classify_scan

# --- 1. Import Legacy Draft Agent ---
from draft import (
    RetrievalAgent,
    LocalLLMReportAgent,
    reports_dict,
    truncate_report
)

# --- 2. Import NEW Vision Agent (Updated API) ---
from vision import (
    vision_encoder,          # The global model instance
    get_hybrid_findings,     # The new detection logic
    VisualDescriptionAgent,  # The new writer class
)

# --- 3. Import NEW KG Agent (Updated API) ---
try:
    from kg_agent import infer_kg
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

# -------------------------------
# Helper: Format KG for LLM
# -------------------------------
def format_kg_for_prompt(kg_data):
    if not kg_data or "entities" not in kg_data:
        return "No Knowledge Graph detected."
    
    entities = kg_data.get("entities", [])
    relations = kg_data.get("relations", [])
    
    formatted = []
    # Format as "Observation (Anatomy)"
    for r in relations:
        if r[2] in ["located_at", "modify"]:
            if r[0] < len(entities) and r[1] < len(entities):
                obs = entities[r[0]][0]
                anat = entities[r[1]][0]
                formatted.append(f"- {obs} ({anat})")
                
    if not formatted:
        return "Normal / No specific findings."
        
    return "DETECTED FINDINGS:\n" + "\n".join(formatted)

# -------------------------------
# Local Synthesis Agent
# -------------------------------
class LocalSynthesisAgent:
    def __init__(self, model_name=config.OLLAMA_MODEL): 
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        
        self.llm = ChatOllama(
            model=model_name, 
            temperature=config.TEMPERATURE,
            # CRITICAL SAFETY: Stop markers
            stop=["<|endoftext|>", "RECOMMENDATIONS:", "\n\n\n\n"] 
        )

    def repair_report(self, bad_report: str, errors: list[str], kg_text_block: str) -> str:
        err_txt = "\n".join([f"- {e}" for e in errors])
        prompt = f"""
        You are a strict radiology report editor.
        This report FAILED validation based on facts:
        {err_txt}
        
        Rewrite the report to fix ALL issues.
        Rules:
        - Output a SINGLE concise paragraph (IU X-ray style).
        - No headers (e.g. "Revised Report:", "Corrected Report:").
        - START DIRECTLY with the medical text.
        
        Report to fix:
        {bad_report}
        """
        resp = self.llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        return self._clean_output(content)

    def _clean_output(self, text):
        """Circuit Breaker logic."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
        text = text.replace("Espaça", "Airspace").replace("Espaco", "Airspace")
        text = re.sub(r"^\s*(?:Revised\s+)?(?:Corrected\s+)?Report:?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^\s*\*\*.*?\*\*\s*", "", text, flags=re.MULTILINE)
        if text.count("Corrected Report:") > 2:
             text = text.split("Corrected Report:")[0].strip()
        return text

    # NEW: Added scan_type parameter (defaults to 'auto' for testing)
    async def generate_final_report(self, draft_agent, vision_agent, retrieval_agent, reports_dict, image_paths, scan_type="auto"):
        target_image = image_paths[0]
        yield json.dumps({"status": "validating", "message": "Checking image modality..."}) + "\n"
        
        # --- NEW MULTI-CLASS VALIDATION LOGIC ---
        detected_modality, confidence = classify_scan(target_image)
        
        if detected_modality == "invalid":
            error_msg = f"Invalid Image Detected. This does not appear to be a medical scan (Confidence: {confidence:.1%})."
            print(f"❌ {error_msg}")
            yield json.dumps({"status": "error", "message": error_msg}) + "\n"
            return
            
        if scan_type != "auto" and detected_modality != scan_type:
            user_friendly_expected = "Chest X-Ray" if scan_type == "xray" else "CT Scan"
            user_friendly_detected = "Chest X-Ray" if detected_modality == "xray" else "CT Scan"
            error_msg = f"Validation Mismatch: You selected '{user_friendly_expected}', but uploaded a '{user_friendly_detected}'."
            print(f"❌ {error_msg}")
            yield json.dumps({"status": "error", "message": error_msg}) + "\n"
            return
        # ----------------------------------------
        
        yield json.dumps({"status": "parallel_start", "message": f"Running Agents for {detected_modality.upper()}..."}) + "\n"

        # --- SEQUENTIAL HELPER FUNCTIONS (Updated for New APIs) ---
        
        # 1. Vision Agent (MULTI-IMAGE SUPPORT)
        async def run_vision():
            print(f"[DEBUG] Vision Agent: Starting for {len(image_paths)} images...")
            try:
                def _process_vision():
                    combined_findings = {}
                    
                    # Iterate through every uploaded image
                    for img_path in image_paths:
                        raw = Image.open(img_path).convert("RGB")
                        emb = vision_encoder.encode_image(raw)
                        # Get findings for this specific view
                        findings = get_hybrid_findings(emb)
                        
                        # Merge into combined_findings, keeping the MAXIMUM confidence score per disease
                        for disease, score in findings.items():
                            if disease not in combined_findings:
                                combined_findings[disease] = score
                            else:
                                combined_findings[disease] = max(combined_findings[disease], score)

                    # Now generate description based on the merged multi-view findings
                    return vision_agent.generate_description(combined_findings)

                res = await asyncio.to_thread(_process_vision)
                print("[DEBUG] Vision Agent: Finished multi-view analysis.")
                return res
            except Exception as e:
                print(f"⚠️ Vision Error: {e}")
                return "Visual analysis failed."

        # 2. Draft Agent
        async def run_draft():
            print("[DEBUG] Draft Agent: Starting...")
            # Use vision_encoder model for retrieval too (Reuse!)
            top_reports = await asyncio.to_thread(
                retrieval_agent.retrieve_top_k, 
                image_paths, # PASSED LIST OF PATHS INSTEAD OF 1
                reports_dict
            )
            draft_context = "\n\n".join(truncate_report(r, 75) for r in top_reports)
            print("[DEBUG] Draft Agent: Retrieved reports.")
            
            res = await asyncio.to_thread(draft_agent.generate_report, draft_context)
            print("[DEBUG] Draft Agent: Finished.")
            return res 

        # 3. KG Agent
        async def run_kg():
            print("[DEBUG] KG Agent: Starting...")
            if not KG_AVAILABLE: return None, "KG skipped."
            try:
                # Use global encoder components to save memory
                kg_data = await asyncio.to_thread(
                    infer_kg, 
                    target_image, 
                    projection="Frontal", 
                    # UPDATED ARGS: No 'thinking_budget', pass models explicitly
                    clip_model=vision_encoder.model,
                    clip_prep=vision_encoder.preprocess,
                    tokenizer=vision_encoder.tokenizer,
                    device=vision_encoder.device,
                    debug=False
                )
                print("[DEBUG] KG Agent: Finished.")
                return kg_data, format_kg_for_prompt(kg_data)
            except Exception as e:
                print(f"⚠️ KG Error: {e}")
                return None, "KG Error."

        # --- EXECUTION FLOW ---
        
        # Run agents (using await to ensure they finish one by one to save VRAM)
        kg_json, kg_text_block = await run_kg()
        vision_report = await run_vision()
        draft_report = await run_draft()
        
        print("[DEBUG] All agents COMPLETE.")
        yield json.dumps({"status": "parallel_done", "message": "Agents finished."}) + "\n"

        # --- SYNTHESIS START ---
        yield json.dumps({"status": "synthesis_start", "message": "Synthesizing..."}) + "\n"

        synthesis_prompt = f"""
        You are an expert radiologist. You are evaluating the IU X-Ray dataset.
        
        ### INPUTS:
        1. FACTS (Knowledge Graph): {kg_text_block}
        2. VISUAL ANALYSIS (AI Vision): {vision_report}
        3. STYLE REFERENCE (Similar Cases): {draft_report}

        ### TASK:
        Write a final radiology report.
        
        ### RULES:
        - **PRIMARY SOURCE:** Trust VISUAL ANALYSIS and FACTS for what is actually present.
        - **STYLE ONLY:** Use STYLE REFERENCE only for phrasing, NOT for medical findings.
        - **NORMALITY:** If Visual Analysis says "Normal" or "Clear", write a NORMAL report, ignoring any diseases in the Style Reference.
        - **FORMAT:** A single concise paragraph. No headers.
        
        REPORT:
        """
        
        full_response = ""
        try:
            def sync_invoke():
                response = self.llm.invoke(synthesis_prompt)
                content = response.content if hasattr(response, "content") else str(response)
                return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            full_response = await asyncio.to_thread(sync_invoke)
            yield json.dumps({"status": "streaming", "chunk": full_response}) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
            return

        final_report = self._clean_output(full_response.strip())
        
        # --- ReAct / CRITIQUE LOOP (RESTORED!) ---
        MAX_REACT_STEPS = 0 # Keep at 1 for speed, can increase if needed
        print(f"[DEBUG] Entering ReAct loop.")
        
        for step in range(MAX_REACT_STEPS):
            yield json.dumps({"status": "thought_start", "message": f"Critiquing (Step {step+1})..."}) + "\n"
            
            critique_prompt = f"""
            You are a strict Medical Auditor. 
            Compare the REPORT vs the FACTS (KG).
            
            FACTS: {kg_text_block}
            REPORT: {final_report}
            
            Identify any:
            1. Hallucinations (Mentions findings NOT in FACTS)
            2. Missed Findings (FACTS mentions it, REPORT skips it)
            3. Contradictions (FACTS say Absent, REPORT says Present)
            
            If PERFECT, output "OK".
            If ERRORS, list them concisely.
            """
            
            critique = ""
            try:
                def sync_critique():
                    resp = self.llm.invoke(critique_prompt)
                    c = resp.content if hasattr(resp, "content") else str(resp)
                    return re.sub(r'<think>.*?</think>', '', c, flags=re.DOTALL).strip()
                critique = await asyncio.to_thread(sync_critique)
            except: pass
            
            # If critique is clean, break early
            if "OK" in critique and len(critique) < 20:
                yield json.dumps({"status": "thought_done", "message": "Report is verified."}) + "\n"
                break
                
            yield json.dumps({"status": "thought_done", "message": "Refining report..."}) + "\n"
            
            # ACT (Repair)
            yield json.dumps({"status": "repair_start", "iter": step}) + "\n"
            final_report = await asyncio.to_thread(self.repair_report, final_report, [critique], kg_text_block)

        # --- FINAL VALIDATION (Regex Safety Net) ---
        v = validate_report(final_report, kg_json)
        if not v["ok"]:
             yield json.dumps({"status": "repair_start", "iter": 99}) + "\n"
             final_report = await asyncio.to_thread(self.repair_report, final_report, v["errors"], kg_text_block)

        # ------------------------------------------------------------------
        # NEW: VISUAL EXPLAINABILITY & TRACE
        # ------------------------------------------------------------------
        
        # Create a dedicated output folder for explainable images
        explain_dir = os.path.join("out", "explained_images")
        os.makedirs(explain_dir, exist_ok=True)
        
        # Build a unique filename (e.g., "CXR3655_IM-1817_0_explained.png")
        parent_folder = os.path.basename(os.path.dirname(target_image))
        file_name = os.path.basename(target_image)
        new_filename = f"{parent_folder}_{file_name}".replace(".png", "_explained.png").replace(".jpg", "_explained.jpg")
        
        explain_img_path = os.path.join(explain_dir, new_filename)
        
        # Run the generator
        explained_path = await asyncio.to_thread(
            generate_explainable_image, target_image, kg_json, explain_img_path
        )

        explainability_trace = {
            "evidence_sources": {
                "vision_agent_findings": vision_report,
                "knowledge_graph_rules": kg_text_block,
                "retrieval_agent_draft": draft_report
            },
            "reasoning_steps": [
                f"1. Verified image modality as '{detected_modality.upper()}'.",
                "2. Extracted raw visual features and calculated hybrid pathology scores.",
                "3. Mapped visual findings to anatomical zones to build Knowledge Graph.",
                "4. Retrieved top-k visually similar historical cases.",
                "5. Synthesized final report prioritizing KG facts, formatted in historical style."
            ]
        }

        print(f"\n[FINAL REPORT]:\n{final_report}\n")
        yield json.dumps({
            "status": "complete", 
            "final_report": final_report, 
            "knowledge_graph": kg_json,
            "explainability": explainability_trace,
            "explainable_image_path": explained_path if explained_path else "Normal - No highlights needed"
        }) + "\n"

# -------------------------------
# Example Usage (Test Block)
# -------------------------------
if __name__ == "__main__":
    async def main_test():
        
        # =================================================================
        # ⬇️ PASTE YOUR IMAGE PATH DIRECTLY HERE ⬇️
        # =================================================================
        target_image = "data/test/image10.jpg"
        
        # Verify the path exists before starting
        if not os.path.exists(target_image):
            print(f"❌ Error: The path '{target_image}' does not exist.")
            print("💡 Tip: Check for typos or make sure you included the file extension (.png, .jpg).")
            return

        print(f"\n🚀 Testing on: {target_image}\n")
        image_paths = [target_image]
        
        # Initialize Agents
        retrieval_agent = RetrievalAgent(vision_encoder, k=5)
        draft_agent = LocalLLMReportAgent()
        vision_agent = VisualDescriptionAgent() 
        synthesis_agent = LocalSynthesisAgent()

        # Run the Pipeline
        gen = synthesis_agent.generate_final_report(
            draft_agent=draft_agent,
            vision_agent=vision_agent,
            retrieval_agent=retrieval_agent,
            reports_dict=reports_dict,
            image_paths=image_paths,
            scan_type="auto" # Default for local testing
        )

        async for chunk in gen:
            try:
                data = json.loads(chunk.strip())
                status = data.get("status")
                
                # Print live updates
                if status != "complete" and status != "streaming":
                    print(f" 🔄 {status.upper()}: {data.get('message', '')}")
                elif status == "error":
                    # This catches the Bouncer/Filter error and stops gracefully!
                    print(f" ❌ ERROR: {data.get('message')}")
                    
                # Final Output
                if status == "complete":
                    print("\n" + "="*50)
                    print("✅ FINAL GENERATED REPORT:")
                    print(data.get("final_report"))
                    
                    print("\n🔍 EXPLAINABILITY TRACE:")
                    print(json.dumps(data.get("explainability"), indent=2))
                    
                    print("\n🖼️ EXPLAINABLE IMAGE SAVED AT:")
                    print(data.get("explainable_image_path"))
                    print("="*50 + "\n")
                    
            except Exception as e: 
                print(f"⚠️ Chunk Error: {e}")
                continue

    asyncio.run(main_test())