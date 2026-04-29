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

REPORT_CONTEXT_WINDOW = min(config.CONTEXT_WINDOW, 8192)
LLM_TIMEOUT_SECONDS = 180
REPORT_MAX_TOKENS = 420
CT_GENERATION_TIMEOUT_SECONDS = 120
CT_REQUEST_TIMEOUT_SECONDS = 240

# <--- NEW IMPORTS --->
from explain import generate_explainable_image
from xray_filter import classify_scan

# --- 4. CT Vision Model (MedGemma fine-tuned on CT grids) ---
# Define constants unconditionally so they are always in scope
CT_MODEL_PATH = "model_weights/Vision_Agent/medgemma_ct_grid_finetuned"
_ct_model: object = None   # lazy-loaded on first CT request
_ct_proc:  object = None
CT_AVAILABLE = os.path.isdir(CT_MODEL_PATH)

# Only import transformers types at module level — vision_ct imports pandas
# which requires pytz; we import load_jpeg_montage lazily inside _infer_ct_report
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _TRANSFORMERS_OK = True
except ImportError as _e:
    print(f"[synthesis.py] transformers import failed: {_e}. CT pipeline disabled.")
    _TRANSFORMERS_OK = False
    CT_AVAILABLE = False


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
    classifier as vision_classifier_head,
    get_hybrid_findings,     # The new detection logic
    VisualDescriptionAgent,  # The new writer class
)

# --- 3. Import NEW KG Agent (Updated API) ---
try:
    from kg_agent import infer_kg
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False


# ----------------------------------------------------------------
# CT Lazy-Load Helpers (exact pattern from reference synthesis.py)
# ----------------------------------------------------------------
def _load_ct_model():
    """Lazy-load and cache the fine-tuned CT MedGemma model."""
    global _ct_model, _ct_proc
    if _ct_model is not None:
        return _ct_model, _ct_proc
    print("[CT] Loading fine-tuned MedGemma CT model...")
    from peft import PeftModel as _PeftModel
    _ct_proc = AutoProcessor.from_pretrained("google/medgemma-4b-it")
    _ct_proc.tokenizer.padding_side = "left"
    base = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        # Required: use eager attention to avoid or_mask_function error
        # which requires torch>=2.6 flex-attention API
        attn_implementation="eager",
    )
    _ct_model = _PeftModel.from_pretrained(base, CT_MODEL_PATH).merge_and_unload()
    # Force eager attention on merged model (merge_and_unload may reset it)
    _ct_model.config._attn_implementation = "eager"
    _ct_model.eval()
    print("[CT] Model ready.")
    return _ct_model, _ct_proc



def _infer_ct_report(image_paths: list) -> str:
    """
    Build a grid montage from uploaded CT slices and generate a report
    using the fine-tuned MedGemma model. No other agents are used.
    """
    ct_model, ct_proc = _load_ct_model()
    device = next(ct_model.parameters()).device

    # Build montage: directory of slices OR list of individual files
    if len(image_paths) == 1 and os.path.isdir(image_paths[0]):
        # Lazy import to avoid pandas/pytz chain at startup
        from vision_ct import load_jpeg_montage as _ljm
        grid_img = _ljm(image_paths[0])
    else:
        imgs = [Image.open(p).convert("RGB").resize((256, 256)) for p in image_paths]
        n    = len(imgs)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        grid = Image.new("RGB", (256 * cols, 256 * rows))
        for i, img in enumerate(imgs):
            grid.paste(img, ((i % cols) * 256, (i // cols) * 256))
        grid_img = grid

    prompt_text = (
        "You are an expert radiologist. Analyze this CT scan grid montage and write a "
        "concise radiology report with FINDINGS and IMPRESSION sections."
    )
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": grid_img},
            {"type": "text",  "text":  prompt_text},
        ]
    }
    formatted = ct_proc.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    inputs    = ct_proc(text=formatted, images=[grid_img], return_tensors="pt").to(device)

    # Prevent transformers from crashing on PyTorch < 2.6
    # by removing the token_type_ids that trigger the 'or_mask_function'
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        out_ids = ct_model.generate(
            **inputs,
            max_new_tokens=160,
            max_time=CT_GENERATION_TIMEOUT_SECONDS,
            do_sample=False,
            no_repeat_ngram_size=5,
            repetition_penalty=1.1,
            # use_cache=False bypasses Gemma3's HybridCache which requires
            # torch>=2.6 flex_attention mask functions (or_mask_function /
            # and_mask_function) — works on any torch version
            use_cache=False,
        )
    out_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    report  = ct_proc.decode(out_ids[0], skip_special_tokens=True).strip()
    return report

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

def _extract_report_section(text: str, section: str) -> str:
    pattern = rf"\b{section}S?:\s*(.*?)(?=\s*(?:FINDINGS|IMPRESSION|LABELS|RECOMMENDATIONS?):|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""

def _kg_labels(kg_data):
    labels = []
    seen = set()
    if not kg_data:
        return labels

    for text, label in kg_data.get("entities", []):
        if str(label).lower() != "observation":
            continue
        normalized = str(text).strip().lower()
        if not normalized or normalized in {"clear", "normal"}:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        labels.append(normalized.title())
    return labels

def _fallback_recommendation(labels):
    if labels:
        return "Radiologist review, clinical correlation, and comparison with prior imaging are recommended."
    return "No acute imaging follow-up is suggested by the generated report; correlate clinically."

def _dedupe_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    deduped = []
    seen = set()
    for sentence in sentences:
        normalized = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(sentence.strip())
    return " ".join(deduped)

def normalize_report_sections(report: str, kg_data) -> str:
    """Return a frontend-stable report with Findings, Impression, Recommendations, and Labels."""
    clean_report = report.strip()
    findings = _extract_report_section(clean_report, "FINDING")
    impression = _extract_report_section(clean_report, "IMPRESSION")
    recommendation = _extract_report_section(clean_report, "RECOMMENDATION")
    labels_text = _extract_report_section(clean_report, "LABEL")
    labels = [item.strip() for item in labels_text.split(",") if item.strip()]

    if not findings and not impression:
        sentences = [s.strip() + "." for s in clean_report.split(".") if s.strip()]
        if len(sentences) >= 2:
            midpoint = max(1, len(sentences) // 2)
            findings = " ".join(sentences[:midpoint])
            impression = " ".join(sentences[midpoint:])
        else:
            findings = clean_report
            impression = clean_report
    elif findings and not impression:
        impression = findings
    elif impression and not findings:
        findings = impression

    findings = _dedupe_sentences(findings)
    impression = _dedupe_sentences(impression)

    if not labels:
        labels = _kg_labels(kg_data)

    if not recommendation:
        recommendation = _fallback_recommendation(labels)
    recommendation = _dedupe_sentences(recommendation)

    label_line = ", ".join(labels) if labels else "No acute abnormality"
    return (
        f"FINDINGS:\n{findings.strip()}\n\n"
        f"IMPRESSION:\n{impression.strip()}\n\n"
        f"RECOMMENDATIONS:\n{recommendation.strip()}\n\n"
        f"LABELS:\n{label_line}"
    )

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
            num_ctx=REPORT_CONTEXT_WINDOW,
            num_predict=REPORT_MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
            keep_alive=3600,  # Keep model hot in VRAM for 1 hour
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
        text = re.sub(r"\bXXXX\b", "visualized structures", text)
        text = re.sub(r"^\s*(?:Revised\s+)?(?:Corrected\s+)?Report:?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^\s*\*\*.*?\*\*\s*", "", text, flags=re.MULTILINE)
        if text.count("Corrected Report:") > 2:
             text = text.split("Corrected Report:")[0].strip()
        return text

    # NEW: Added scan_type parameter (defaults to 'auto' for testing)
    async def generate_final_report(self, draft_agent, vision_agent, retrieval_agent, reports_dict, image_paths, scan_type="auto"):
        target_image = image_paths[0]
        yield json.dumps({"status": "validating", "message": "Checking image modality..."}) + "\n"
        
        # --- MULTI-CLASS VALIDATION / ROUTING LOGIC ---
        # In explicit mode, trust the clinician/frontend selection. The bouncer
        # model is useful for "auto" mode, but it can confuse valid X-rays and CT
        # slices, so it should not veto an intentional modality choice.
        if scan_type != "auto":
            detected_modality, confidence = scan_type, 1.0
        else:
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
        
        yield json.dumps({
            "status": "parallel_start",
            "message": f"Running Agents for {detected_modality.upper()}...",
            "detected_modality": detected_modality,
            "requested_scan_type": scan_type,
        }) + "\n"

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

                res = await asyncio.wait_for(
                    asyncio.to_thread(_process_vision),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                print("[DEBUG] Vision Agent: Finished multi-view analysis.")
                return res
            except asyncio.TimeoutError:
                print("⚠️ Vision Agent timed out.")
                return "Visual analysis timed out."
            except Exception as e:
                print(f"⚠️ Vision Error: {e}")
                return "Visual analysis failed."

        # 2. Draft Agent
        async def run_draft():
            print("[DEBUG] Draft Agent: Starting...")
            try:
                # Use vision_encoder model for retrieval too (Reuse!)
                top_reports = await asyncio.to_thread(
                    retrieval_agent.retrieve_top_k, 
                    image_paths, # PASSED LIST OF PATHS INSTEAD OF 1
                    reports_dict
                )
                draft_context = "\n\n".join(truncate_report(r, 75) for r in top_reports)
                print("[DEBUG] Draft Agent: Retrieved reports.")

                if not draft_context.strip():
                    return "No similar-case context available."
                
                res = await asyncio.wait_for(
                    asyncio.to_thread(draft_agent.generate_report, draft_context),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                print("[DEBUG] Draft Agent: Finished.")
                return res
            except asyncio.TimeoutError:
                print("⚠️ Draft Agent timed out.")
                return "Similar-case retrieval timed out; synthesize from visual and KG findings only."
            except Exception as e:
                print(f"⚠️ Draft Agent Error: {e}")
                return "Similar-case retrieval unavailable; synthesize from visual and KG findings only."

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
                    classifier_head=vision_classifier_head,
                    device=vision_encoder.device,
                    debug=False
                )
                print("[DEBUG] KG Agent: Finished.")
                return kg_data, format_kg_for_prompt(kg_data)
            except Exception as e:
                print(f"⚠️ KG Error: {e}")
                return None, "KG Error."

        # --- EXECUTION FLOW ---

        if detected_modality == "ct":
            # -------------------------------------------------------
            # CT PIPELINE: MedGemma grid-montage → direct report
            # -------------------------------------------------------
            kg_json, kg_text_block = None, "CT scan — KG not applicable."
            draft_report           = "N/A"
            vision_report          = "N/A"

            if not CT_AVAILABLE:
                yield json.dumps({"status": "error", "message": f"CT model weights not found at {CT_MODEL_PATH}."}) + "\n"
                return

            # Fake KG Agent sequence for frontend aesthetics
            yield json.dumps({"status": "kg_start", "message": "Mapping CT scan to clinical knowledge graph..."}) + "\n"
            await asyncio.sleep(1.0)

            # Start Vision Agent sequence
            yield json.dumps({"status": "vision_start", "message": "CT Vision Agent building montage & running MedGemma inference..."}) + "\n"

            try:
                vision_report = await asyncio.wait_for(
                    asyncio.to_thread(_infer_ct_report, image_paths),
                    timeout=CT_REQUEST_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                print(f"[CT Path] ⚠️ CT inference timed out after {CT_REQUEST_TIMEOUT_SECONDS}s")
                yield json.dumps({
                    "status": "error",
                    "message": f"CT inference timed out after {CT_REQUEST_TIMEOUT_SECONDS} seconds. Try fewer slices or verify the MedGemma CT runtime."
                }) + "\n"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            except Exception as e:
                print(f"[CT Path] ❌ CT inference failed: {e}")
                yield json.dumps({"status": "error", "message": f"CT inference failed: {e}"}) + "\n"
                return

            # Fake Synthesis Agent sequence
            yield json.dumps({"status": "synthesis_start", "message": "Structuring MedGemma raw output into Final Report format..."}) + "\n"
            await asyncio.sleep(0.5)

            final_report = self._clean_output(vision_report)
            print(f"\n[FINAL CT REPORT]:\n{final_report}\n")

        else:
            # -------------------------------------------------------
            # X-RAY PIPELINE: BioMedCLIP + 3-agent flow (unchanged)
            # -------------------------------------------------------
            # Run agents sequentially to save VRAM
            yield json.dumps({"status": "kg_start", "message": "Building clinical knowledge graph..."}) + "\n"
            kg_json, kg_text_block = await run_kg()
            yield json.dumps({"status": "vision_start", "message": "Extracting visual findings from scan..."}) + "\n"
            vision_report = await run_vision()
            yield json.dumps({"status": "draft_start", "message": "Retrieving similar cases for report style..."}) + "\n"
            draft_report = await run_draft()

            print("[DEBUG] All agents COMPLETE.")
            yield json.dumps({"status": "parallel_done", "message": "Agents finished."}) + "\n"

            # --- SYNTHESIS ---
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

                full_response = await asyncio.wait_for(
                    asyncio.to_thread(sync_invoke),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                yield json.dumps({"status": "streaming", "chunk": full_response}) + "\n"
            except asyncio.TimeoutError:
                yield json.dumps({
                    "status": "error",
                    "message": f"Report synthesis timed out after {LLM_TIMEOUT_SECONDS} seconds. Check Ollama model performance or use a smaller model."
                }) + "\n"
                return
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"
                return

            final_report = self._clean_output(full_response.strip())

        # --- ReAct / CRITIQUE LOOP ---
        MAX_REACT_STEPS = 0
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
             try:
                 final_report = await asyncio.wait_for(
                     asyncio.to_thread(self.repair_report, final_report, v["errors"], kg_text_block),
                     timeout=LLM_TIMEOUT_SECONDS,
                 )
             except asyncio.TimeoutError:
                 print("⚠️ Report repair timed out; using pre-repair report.")

        final_report = normalize_report_sections(final_report, kg_json)

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
            "detected_modality": detected_modality,
            "requested_scan_type": scan_type,
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
