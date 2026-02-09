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

    async def generate_final_report(self, draft_agent, vision_agent, retrieval_agent, reports_dict, image_paths):
        target_image = image_paths[0]
        yield json.dumps({"status": "parallel_start", "message": "Running Agents..."}) + "\n"

        # --- SEQUENTIAL HELPER FUNCTIONS (Updated for New APIs) ---
        
        # 1. Vision Agent
        async def run_vision():
            print("[DEBUG] Vision Agent: Starting...")
            try:
                # Wrap sync calls in thread to avoid blocking loop
                def _process_vision():
                    raw = Image.open(target_image).convert("RGB")
                    # Use global encoder from vision.py (Shared Memory)
                    emb = vision_encoder.encode_image(raw)
                    # New Hybrid Findings Logic
                    findings = get_hybrid_findings(emb)
                    # New Writer Class
                    return vision_agent.generate_description(findings)

                res = await asyncio.to_thread(_process_vision)
                print("[DEBUG] Vision Agent: Finished.")
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
                target_image, 
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
        MAX_REACT_STEPS = 1 # Keep at 1 for speed, can increase if needed
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

        print(f"\n[FINAL REPORT]:\n{final_report}\n")
        yield json.dumps({"status": "complete", "final_report": final_report, "knowledge_graph": kg_json}) + "\n"

# -------------------------------
# Example Usage (Test Block)
# -------------------------------
if __name__ == "__main__":
    async def main_test():
        # Test on a known image
        target_image = "data/iu_xray/images/CXR3655_IM-1817/0.png"
        
        if not os.path.exists(target_image):
             for root, _, files in os.walk("data/iu_xray/images"):
                 for f in files:
                     if f.endswith(".png"):
                         target_image = os.path.join(root, f)
                         break
                 if target_image: break

        if os.path.exists(target_image):
            print(f"Testing on: {target_image}")
            image_paths = [target_image]
            
            # Initialize Agents
            # Note: Retrieval now uses the shared vision_encoder models
            retrieval_agent = RetrievalAgent(vision_encoder.model, vision_encoder.preprocess, k=5, device=vision_encoder.device)
            draft_agent = LocalLLMReportAgent()
            
            # New Vision Writer
            vision_agent = VisualDescriptionAgent() 
            
            synthesis_agent = LocalSynthesisAgent()

            gen = synthesis_agent.generate_final_report(
                draft_agent=draft_agent,
                vision_agent=vision_agent,
                retrieval_agent=retrieval_agent,
                reports_dict=reports_dict,
                image_paths=image_paths,
            )

            async for chunk in gen:
                try:
                    data = json.loads(chunk)
                    if data.get("status") == "complete":
                        print("\n✅ GENERATED REPORT:")
                        print(data.get("final_report"))
                except: continue
        else:
            print("❌ Error: No valid image found.")

    asyncio.run(main_test())