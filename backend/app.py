# app.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import shutil
import uuid
import os
import json
import base64
import uvicorn
from typing import List
import config
import subprocess
import time
import requests

# <--- NEW IMPORTS --->
from xray_filter import classify_scan

# Import agents and dependencies
from synthesis import (
    LocalSynthesisAgent,
    RetrievalAgent,
    LocalLLMReportAgent,
    VisualDescriptionAgent,
    reports_dict,
    vision_encoder,
)

# --- GLOBAL AGENT STORE ---
agents = {}

# --- LIFESPAN MANAGER ---
def check_and_start_ollama():
    ollama_url = f"{config.BASE_URL}/api/tags"
    try:
        requests.get(ollama_url, timeout=1)
        print("✅ Ollama is already running.")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("⚠️ Ollama is NOT running. Starting it now...")
        try:
            subprocess.Popen(["ollama", "serve"], shell=True)
            print("⏳ Waiting for Ollama to be ready...", end="", flush=True)
            for _ in range(20):
                try:
                    requests.get(ollama_url, timeout=1)
                    print(" Done!")
                    return
                except:
                    time.sleep(1)
                    print(".", end="", flush=True)
            print("\n❌ Failed to start Ollama automatically. Please start it manually.")
        except FileNotFoundError:
             print("\n❌ 'ollama' command not found. Is it installed?")

def _normalized_ollama_model_name():
    model_name = config.OLLAMA_MODEL
    if model_name.startswith("ollama/"):
        model_name = model_name.split("/", 1)[1]
    return model_name

def _get_ollama_status():
    model_name = _normalized_ollama_model_name()
    try:
        response = requests.get(f"{config.BASE_URL}/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models if m.get("name")]
        return {
            "ollama_running": True,
            "ollama_model": model_name,
            "ollama_model_available": model_name in model_names,
            "ollama_available_models": model_names,
        }
    except Exception as exc:
        return {
            "ollama_running": False,
            "ollama_model": model_name,
            "ollama_model_available": False,
            "ollama_error": str(exc),
            "ollama_available_models": [],
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("🧹 Memory cleaned before startup...")
    
    check_and_start_ollama()
    
    device = vision_encoder.device
    print(f"\n🚀 STARTUP: Loading AI Models on {device}...")
    
    agents["retrieval"] = RetrievalAgent(vision_encoder, k=3)
    agents["draft"] = LocalLLMReportAgent()
    agents["vision"] = VisualDescriptionAgent()
    agents["synthesis"] = LocalSynthesisAgent()
    
    print("✅ All AI Agents Ready! Server is listening.")
    yield
    print("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/synthesize-report")
async def synthesize_report(
    files: List[UploadFile] = File(...),
    # Optional frontend parameter: "xray", "ct", or "auto"
    scan_type: str = Form("auto") 
):
    image_paths = []
    
    # 1. Save all uploaded files to disk
    for file in files:
        file_id = str(uuid.uuid4())
        image_path = os.path.join(UPLOAD_DIR, f"{file_id}.png")

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        image_paths.append(image_path)

    # 2. VALIDATE: Distinguish between X-Rays, CTs, and Invalid
    # We will validate based on the FIRST image out of the batch to save time.
    try:
        if len(image_paths) > 0:
            if scan_type != "auto":
                print(f"User explicitly selected scan_type: {scan_type}. Bypassing AI filter.")
                detected_modality = scan_type
                confidence = 1.0 # Forced confidence
            else:
                primary_image = image_paths[0]
                detected_modality, confidence = classify_scan(primary_image)
            
            # 1. Is it entirely invalid?
            if detected_modality == "invalid":
                for path in image_paths:
                    if os.path.exists(path):
                        os.remove(path)
                async def error_stream_invalid():
                    yield json.dumps({
                        "status": "error", 
                        "message": f"Validation Failed: The primary image does not appear to be a medical scan (Confidence: {confidence:.1%})."
                    }) + "\n"
                return StreamingResponse(error_stream_invalid(), media_type="application/x-ndjson")
                
            # 2. Does the detected scan match what the frontend expected?
            if scan_type == "auto" and detected_modality not in ["xray", "ct"]: # Just an additional safety net
                for path in image_paths:
                    if os.path.exists(path):
                        os.remove(path)
                async def error_stream_mismatch():
                    yield json.dumps({
                        "status": "error", 
                        "message": f"Validation Failed: The detected scan type ({detected_modality}) is not supported."
                    }) + "\n"
                return StreamingResponse(error_stream_mismatch(), media_type="application/x-ndjson")

    except Exception as e:
        print(f"⚠️ Validation error: {e}")

    # 3. Process Valid Scans
    report_generator = agents["synthesis"].generate_final_report(
        draft_agent=agents["draft"],
        vision_agent=agents["vision"],
        retrieval_agent=agents["retrieval"],
        reports_dict=reports_dict,
        image_paths=image_paths,
        scan_type=scan_type
    )

    async def process_stream(generator):
        try:
            async for chunk in generator:
                try:
                    data = json.loads(chunk)
                    if data.get("status") == "complete":
                        exp_path = data.get("explainable_image_path")
                        if exp_path and os.path.exists(exp_path) and exp_path != "Normal - No highlights needed":
                            with open(exp_path, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode("utf-8")
                            data["heatmap"] = f"data:image/png;base64,{b64_data}"
                        else:
                            data["heatmap"] = None
                        yield json.dumps(data) + "\n"
                    else:
                        yield chunk
                except Exception:
                    yield chunk
        finally:
            for path in image_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as cleanup_error:
                    print(f"⚠️ Could not clean upload temp file {path}: {cleanup_error}")

    return StreamingResponse(
        process_stream(report_generator), 
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ── Health / Metrics Endpoint ──────────────────────
import psutil
_server_start_time = time.time()

@app.get("/api/health")
async def health():
    import torch as _torch
    mem = psutil.virtual_memory()
    result = {
        "status": "healthy",
        "uptime_seconds": int(time.time() - _server_start_time),
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "memory_used_mb": round(mem.used / (1024 ** 2)),
        "memory_total_mb": round(mem.total / (1024 ** 2)),
        "gpu_available": _torch.cuda.is_available(),
    }
    ollama_status = _get_ollama_status()
    result.update(ollama_status)
    if not ollama_status["ollama_running"] or not ollama_status["ollama_model_available"]:
        result["status"] = "degraded"
        result["message"] = f"Ollama model '{ollama_status['ollama_model']}' is not available. Pull it or update config.OLLAMA_MODEL."
    if _torch.cuda.is_available():
        result["gpu_name"] = _torch.cuda.get_device_name(0)
        result["gpu_memory_used_mb"] = round(_torch.cuda.memory_allocated(0) / (1024 ** 2))
        result["gpu_memory_total_mb"] = round(_torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
    return result

# ── Explainability Narrative Endpoint ────────────────
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
import config

class ExplainRequest(BaseModel):
    findings: str
    impression: str

@app.post("/api/explain")
async def generate_explanation(req: ExplainRequest):
    try:
        model_name = config.OLLAMA_MODEL
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
            
        llm = ChatOllama(
            model=model_name,
            temperature=config.TEMPERATURE,
            num_ctx=config.CONTEXT_WINDOW,
            base_url=config.BASE_URL
        )
        
        prompt = f"""
        Act as an expert Radiologist and AI Explainer for the X-MedFusion application.
        
        The AI pipeline has already synthesized the following report for this patient:
        FINDINGS: {req.findings}
        IMPRESSION: {req.impression}
        
        The patient is viewing their original X-ray alongside an Insights heatmap generated by our Vision Agent.
        
        Please provide a highly-structured "Automated Diagnostic Narrative" that explains HOW the AI arrived at this conclusion. 
        Format your response EXACTLY in these three steps:
        
        **Step 1: Visual Feature Extraction**
        Explain what the Vision Agent highlighted in the heatmap.

        **Step 2: Anatomical & Clinical Context**
        Correlate those highlighted regions to the specific anatomy (e.g., lungs, heart) and explain their clinical significance based on the FINDINGS.

        **Step 3: Diagnostic Deduction**
        Explain how the findings logically lead to the final IMPRESSION.

        Add a brief, italicized disclaimer at the very end stating this is an experimental visualization tool intended to assist radiologists.
        """
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        return {"explanation": content.strip()}
    except Exception as e:
        return {"error": str(e)}

# ── HIL Fine-tuning Endpoint ──────────────────────
from pydantic import BaseModel as _BaseModel
import threading

class HILFinetuneRequest(_BaseModel):
    task_id: str

_finetune_status = {"running": False, "last_result": None}

def _get_supabase_client():
    """Create a Supabase client, reading creds from env or frontend .env."""
    import os as _os
    from supabase import create_client

    sb_url = _os.environ.get("SUPABASE_URL", "")
    sb_key = _os.environ.get("SUPABASE_SERVICE_KEY", _os.environ.get("SUPABASE_KEY", ""))

    if not sb_url or not sb_key:
        env_path = _os.path.join(_os.path.dirname(__file__), "..", "frontend", ".env")
        if _os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("VITE_SUPABASE_URL="):
                        sb_url = line.split("=", 1)[1].strip()
                    elif line.startswith("VITE_SUPABASE_ANON_KEY="):
                        sb_key = line.split("=", 1)[1].strip()

    if not sb_url or not sb_key:
        return None, None, None
    return create_client(sb_url, sb_key), sb_url, sb_key


async def _verify_admin_jwt(authorization: str | None) -> str:
    """Verify Supabase JWT and confirm the user is an admin. Returns user_id."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]

    # Decode JWT — try PyJWT first, fall back to unverified decode
    try:
        import jwt as _jwt
        jwt_secret = os.environ.get("SUPABASE_JWT_SECRET", "")
        if jwt_secret:
            payload = _jwt.decode(token, jwt_secret, algorithms=["HS256"], audience="authenticated")
        else:
            # If no secret configured, decode without verification (dev mode)
            payload = _jwt.decode(token, options={"verify_signature": False})
    except ImportError:
        # PyJWT not installed — decode without verification (dev fallback)
        import base64 as _b64
        import json as _json
        parts = token.split(".")
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="Malformed JWT")
        padded = parts[1] + "==" * (4 - len(parts[1]) % 4)
        payload = _json.loads(_b64.urlsafe_b64decode(padded))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token has no subject")

    # Verify admin role via Supabase
    sb, _, _ = _get_supabase_client()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    role_resp = sb.table("user_roles").select("role, approval_status").eq("user_id", user_id).maybe_single().execute()
    role_data = role_resp.data
    if not role_data or role_data.get("role") != "admin" or role_data.get("approval_status") != "approved":
        raise HTTPException(status_code=403, detail="Admin access required")

    return user_id


@app.post("/api/hil/finetune")
async def hil_finetune(req: HILFinetuneRequest, authorization: str | None = Header(None)):
    # ── Auth: only admins can trigger fine-tuning ──
    await _verify_admin_jwt(authorization)

    if _finetune_status["running"]:
        return {"error": "Fine-tuning is already running. Please wait."}

    try:
        sb, _, _ = _get_supabase_client()
        if not sb:
            return {"error": "Supabase credentials not configured. Set SUPABASE_URL and SUPABASE_KEY env vars."}

        # Get approved reports for this task
        reports_resp = sb.table("hil_reports").select("*").eq("task_id", req.task_id).eq("status", "approved").execute()
        reports = reports_resp.data or []

        if len(reports) == 0:
            return {"error": "No approved reports found for this task.", "num_samples": 0}

        # Minimum sample guard
        MIN_HIL_SAMPLES = 5
        if len(reports) < MIN_HIL_SAMPLES:
            return {
                "error": f"Need at least {MIN_HIL_SAMPLES} approved reports to fine-tune. Got {len(reports)}.",
                "num_samples": len(reports),
            }

        # Get scan URLs
        scan_ids = [r["scan_id"] for r in reports]
        scans_resp = sb.table("hil_scans").select("*").in_("id", scan_ids).execute()
        scans_map = {s["id"]: s["image_url"] for s in (scans_resp.data or [])}

        approved_scans = []
        for r in reports:
            image_url = scans_map.get(r["scan_id"])
            if image_url:
                approved_scans.append({
                    "image_url": image_url,
                    "findings": r.get("findings", ""),
                    "impression": r.get("impression", ""),
                })

        if len(approved_scans) == 0:
            return {"error": "No scans matched the approved reports.", "num_samples": 0}

        # Run fine-tuning in background thread
        def _run():
            _finetune_status["running"] = True
            try:
                from hil_finetune import run_hil_finetune
                result = run_hil_finetune(approved_scans)
                _finetune_status["last_result"] = result

                # Hot-reload updated model weights into the running server
                if result.get("status") == "complete":
                    try:
                        from vision import reload_classifier_head
                        reloaded = reload_classifier_head()
                        result["model_reloaded"] = reloaded
                        print("🔄 Model hot-reloaded after HIL fine-tuning" if reloaded else "⚠️ Model reload skipped")
                    except Exception as re_err:
                        print(f"⚠️ Model reload failed: {re_err}")
                        result["model_reloaded"] = False
            except Exception as e:
                _finetune_status["last_result"] = {"error": str(e)}
            finally:
                _finetune_status["running"] = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return {
            "status": "started",
            "num_samples": len(approved_scans),
            "message": f"Fine-tuning started with {len(approved_scans)} approved scans."
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/hil/finetune-status")
async def hil_finetune_status():
    return {
        "running": _finetune_status["running"],
        "last_result": _finetune_status["last_result"],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
