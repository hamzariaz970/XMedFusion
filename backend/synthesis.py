# synthesis.py

import os
import re
import json
import time
import asyncio
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from validators import validate_report
import config

from draft import (
    RetrievalAgent,
    LocalLLMReportAgent,
    reports_dict,
    model as clip_model,
    preprocess as clip_prep,
    device,
    truncate_report,
    zero_shot_classify
)

from vision import (
    get_visual_embeddings,
    embeddings_to_text,
    vision_model,
    proj_heads,
    LocalLLMReportAgent as VisionLLMAgent,
)

try:
    from kg_agent import infer_kg
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

# -------------------------------
# Helper: Format KG
# -------------------------------
def format_kg_for_prompt(kg_data):
    if not kg_data or "entities" not in kg_data:
        return "No Knowledge Graph detected."
    entities = kg_data.get("entities", [])
    present = [f"- {t} is PRESENT" for t, l in entities if "present" in l]
    absent = [f"- {t} is ABSENT" for t, l in entities if "absent" in l]
    
    kg_text = "DETECTED FINDINGS:\n"
    if present: kg_text += "PRESENT:\n" + "\n".join(present) + "\n"
    if absent: kg_text += "ABSENT:\n" + "\n".join(absent)
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
            temperature=config.TEMPERATURE,
            # CRITICAL SAFETY: Stop generating if it tries to start a new section unexpectedly
            stop=["<|endoftext|>", "RECOMMENDATIONS:", "\n\n\n\n"] 
        )

    def repair_report(self, bad_report: str, errors: list[str], kg_text_block: str) -> str:
        err_txt = "\n".join([f"- {e}" for e in errors])
        prompt = f"""
You are a strict radiology report editor.
This report FAILED validation:
{err_txt}

Rewrite the report to fix ALL issues.
Rules:
- Output ONLY these sections: FINDINGS, IMPRESSION, LABELS, RECOMMENDATIONS.
- LABELS must be a single line of unique conditions.
- RECOMMENDATIONS must be one sentence.

Report to fix:
{bad_report}

Corrected Report:
"""
        resp = self.llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        return self._clean_output(content)

    def _clean_output(self, text):
        """
        Circuit Breaker logic to prevent Infinite Loops and OOM crashes.
        """
        # 1. Remove DeepSeek thinking tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

        # 2. Fix Headers (Ensure Newlines)
        headers = ["FINDINGS", "IMPRESSION", "LABELS", "RECOMMENDATIONS"]
        for h in headers:
            # Force newline before header if it's stuck to previous text
            text = re.sub(r"(?<=\w)[ \t]*(" + h + r":?)", r"\n\n\1", text, flags=re.IGNORECASE)
            # Force newline after header
            text = re.sub(f"({h})[: ]*(?=[A-Z])", f"{h}:\n", text, flags=re.IGNORECASE)

        # 3. CRITICAL LOOP FIX: Deduplicate & Truncate Labels
        # Finds the Labels section and aggressively cleans it
        label_match = re.search(r"(LABELS:)(.*?)(?=(RECOMMENDATIONS:|IMPRESSION:|$))", text, re.DOTALL | re.IGNORECASE)
        
        if label_match:
            header = label_match.group(1)
            raw_labels = label_match.group(2).strip()
            
            # Split by comma or newline
            items = [x.strip() for x in re.split(r'[,\n]', raw_labels) if x.strip()]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_items = [x for x in items if not (x.lower() in seen or seen.add(x.lower()))]
            
            # LIMIT: Max 6 labels. This kills the infinite loop hard.
            unique_items = unique_items[:6]
            
            clean_labels_str = ", ".join(unique_items)
            
            # Reconstruct the section
            # We replace the whole messy loop with the clean version
            text = text.replace(label_match.group(0), f"{header}\n{clean_labels_str}\n")

        # 4. Fix specific typos
        text = text.replace("Espaça", "Airspace").replace("Espaco", "Airspace")
        
        return text

    async def generate_final_report(self, draft_agent, vision_agent, retrieval_agent, reports_dict, image_paths):
        target_image = image_paths[0]
        yield json.dumps({"status": "parallel_start", "message": "Running Agents..."}) + "\n"

        # --- ASYNC TASKS ---
        async def run_vision():
            return await asyncio.to_thread(zero_shot_classify, target_image, clip_model, clip_prep, device)

        async def run_draft():
            top_reports = await asyncio.to_thread(retrieval_agent.retrieve_top_k, target_image, reports_dict)
            draft_context = "\n\n".join(truncate_report(r, 75) for r in top_reports)
            return await asyncio.to_thread(draft_agent.generate_report, draft_context)

        async def run_kg():
            if not KG_AVAILABLE: return None, "KG skipped."
            try:
                kg_data = await asyncio.to_thread(infer_kg, target_image, projection="Frontal", thinking_budget=0)
                return kg_data, format_kg_for_prompt(kg_data)
            except: return None, "KG Error."

        # Execute Parallel
        vision_report, draft_report, (kg_json, kg_text_block) = await asyncio.gather(run_vision(), run_draft(), run_kg())
        yield json.dumps({"status": "parallel_done", "message": "Agents finished."}) + "\n"

        # --- SYNTHESIS ---
        yield json.dumps({"status": "synthesis_start", "message": "Synthesizing..."}) + "\n"

        synthesis_prompt = f"""
You are an expert radiologist. Write a final report.

### INPUTS:
1. FACTS (KG): {kg_text_block}
2. VISUALS: {vision_report}
3. STYLE REF: {draft_report}

### RULES:
- Systematically check Heart, Lungs, Pleura, Bones.
- State pertinent negatives (e.g., "No pneumothorax").
- LABELS section must be a comma-separated list of max 5 items.

### OUTPUT FORMAT:
FINDINGS:
[Narrative paragraph]
IMPRESSION:
[Summary]
LABELS:
[Condition1, Condition2, ...]
RECOMMENDATIONS:
[Action or None]
"""
        full_response = ""
        # We manually accumulate response to run the cleaner on the full text
        async for chunk in self.llm.astream(synthesis_prompt):
            content = getattr(chunk, "content", None)
            if content:
                full_response += content
        
        # Apply the Circuit Breaker Cleanup
        final_report = self._clean_output(full_response)
        
        # --- VALIDATION LOOP ---
        MAX_ITERS = 1 # Keep it low to prevent OOM
        for it in range(MAX_ITERS + 1):
            v = validate_report(final_report, kg_json)
            yield json.dumps({"status": "validate_done", "iter": it, "ok": v["ok"], "errors": v["errors"]}) + "\n"
            
            if v["ok"]: break
            if it == MAX_ITERS: break
            
            yield json.dumps({"status": "repair_start", "iter": it}) + "\n"
            # Use run_in_executor/to_thread for the blocking repair call
            final_report = await asyncio.to_thread(self.repair_report, final_report, v["errors"], kg_text_block)

        yield json.dumps({"status": "complete", "final_report": final_report, "knowledge_graph": kg_json}) + "\n"

# -------------------------------
# Example Usage (Test Block)
# -------------------------------
if __name__ == "__main__":
    async def main_test():
        print(f"Using device: {device}")
        target_image = "data/iu_xray/images/CXR3655_IM-1817/0.png"
        
        if not os.path.exists(target_image):
             if os.path.exists("data/iu_xray/images"):
                for root, _, files in os.walk("data/iu_xray/images"):
                    for f in files:
                        if f.endswith(".png"):
                            target_image = os.path.join(root, f)
                            break
                    if target_image: break

        if os.path.exists(target_image):
            image_paths = [target_image]
            retrieval_agent = RetrievalAgent(clip_model, clip_prep, k=5, device=device)
            draft_agent = LocalLLMReportAgent()
            vision_agent = VisionLLMAgent()
            synthesis_agent = LocalSynthesisAgent()

            gen = synthesis_agent.generate_final_report(
                draft_agent=draft_agent,
                vision_agent=vision_agent,
                retrieval_agent=retrieval_agent,
                reports_dict=reports_dict,
                image_paths=image_paths,
            )

            final_text = ""
            async for chunk in gen:
                try:
                    data = json.loads(chunk)
                    if data.get("status") == "complete":
                        final_text = data.get("final_report", "")
                        print(final_text)
                except: continue
        else:
            print("❌ Error: No valid image found.")

    asyncio.run(main_test())