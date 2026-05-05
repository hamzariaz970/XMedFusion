# app.py

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
import sys
import uvicorn
from typing import List
import re
import config
import subprocess
import time
import requests
import threading
from functools import lru_cache

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# --- GLOBAL AGENT STORE ---
agents = {}
_ai_runtime = None
_ai_runtime_lock = threading.Lock()
_ai_warmup_started = False
_ai_warmup_complete = False
_ai_warmup_error = None
_optional_model_status = {}
PRELOAD_ALL_MODELS_ON_STARTUP = os.getenv("PRELOAD_ALL_MODELS_ON_STARTUP", "0").strip().lower() not in {"0", "false", "no", "off"}
_resolved_ollama_model = None


def _load_ai_runtime():
    """
    Import the heavy synthesis stack on demand.
    Importing synthesis pulls in BioMedCLIP, classifier weights, filter weights,
    and the cached report corpus, so keep it off the startup critical path.
    """
    global _ai_runtime
    if _ai_runtime is not None:
        return _ai_runtime

    with _ai_runtime_lock:
        if _ai_runtime is not None:
            return _ai_runtime

        _resolve_ollama_model_name()

        from synthesis import (
            LocalSynthesisAgent,
            RetrievalAgent,
            LocalLLMReportAgent,
            VisualDescriptionAgent,
            reports_dict,
            vision_encoder,
            warm_optional_models,
        )

        _ai_runtime = {
            "LocalSynthesisAgent": LocalSynthesisAgent,
            "RetrievalAgent": RetrievalAgent,
            "LocalLLMReportAgent": LocalLLMReportAgent,
            "VisualDescriptionAgent": VisualDescriptionAgent,
            "reports_dict": reports_dict,
            "vision_encoder": vision_encoder,
            "warm_optional_models": warm_optional_models,
        }
        return _ai_runtime


def _ensure_report_agents(*, preload_optional_models=False):
    """
    Lazily build long-lived report agents once per process.
    This keeps `python app.py` fast while still reusing models and Ollama clients
    across requests after the first initialization.
    """
    runtime = _load_ai_runtime()
    if agents:
        return runtime

    with _ai_runtime_lock:
        if agents:
            return runtime

        device = runtime["vision_encoder"].device
        print(f"\n🤖 Initializing report pipeline on {device}...")

        retrieval = runtime["RetrievalAgent"](runtime["vision_encoder"], k=3)
        agents["retrieval"] = retrieval
        agents["draft"] = runtime["LocalLLMReportAgent"]()
        agents["vision"] = runtime["VisualDescriptionAgent"]()
        agents["synthesis"] = runtime["LocalSynthesisAgent"]()

        try:
            retrieval._ensure_text_features()
        except Exception as exc:
            print(f"⚠️ Retrieval text feature warmup skipped: {exc}")

        global _optional_model_status
        if preload_optional_models:
            _optional_model_status = runtime["warm_optional_models"]()

        print("✅ Report pipeline initialized.")
        return runtime


def _warm_ai_stack():
    global _ai_warmup_started, _ai_warmup_complete, _ai_warmup_error
    if _ai_warmup_started:
        return
    _ai_warmup_started = True
    try:
        _ensure_report_agents(preload_optional_models=PRELOAD_ALL_MODELS_ON_STARTUP)
        _ai_warmup_complete = True
    except Exception as exc:
        _ai_warmup_error = str(exc)
        print(f"⚠️ Background AI warmup failed: {exc}")


@lru_cache(maxsize=1)
def _get_explain_llm():
    from langchain_community.chat_models import ChatOllama

    model_name = _resolve_ollama_model_name()

    return ChatOllama(
        model=model_name,
        temperature=config.TEMPERATURE,
        num_ctx=config.CONTEXT_WINDOW,
        base_url=config.BASE_URL,
        keep_alive=3600,
    )

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


def _resolve_ollama_model_name():
    global _resolved_ollama_model
    if _resolved_ollama_model:
        return _resolved_ollama_model

    preferred_model = _normalized_ollama_model_name()
    fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL", "").strip()

    try:
        response = requests.get(f"{config.BASE_URL}/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models if m.get("name")]

        if preferred_model in model_names:
          chosen_model = preferred_model
        elif fallback_model and fallback_model in model_names:
          chosen_model = fallback_model
          print(f"[Ollama] Preferred model '{preferred_model}' not found. Falling back to '{chosen_model}'.")
        elif model_names:
          chosen_model = model_names[0]
          print(f"[Ollama] Preferred model '{preferred_model}' not found. Falling back to '{chosen_model}'.")
        else:
          chosen_model = preferred_model

        _resolved_ollama_model = chosen_model
        config.OLLAMA_MODEL = chosen_model
        return chosen_model
    except Exception:
        _resolved_ollama_model = preferred_model
        return preferred_model

def _get_ollama_status():
    model_name = _resolve_ollama_model_name()
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


@lru_cache(maxsize=1)
def _get_scan_classifier():
    from xray_filter import classify_scan

    return classify_scan

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
    if PRELOAD_ALL_MODELS_ON_STARTUP:
        print("🚀 STARTUP: Preloading X-ray, ensemble, and CT models before serving requests...")
        _warm_ai_stack()
        print("✅ Startup model preload complete. Server is listening.")
    else:
        print("🚀 STARTUP: Server ready. Warming AI pipeline in the background...")
        threading.Thread(target=_warm_ai_stack, daemon=True).start()

    yield
    print("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan)

# Allowed origins: wildcard "*" with allow_credentials=True is invalid per the CORS spec.
# Enumerate trusted origins so auth headers/cookies work from Vercel.
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    # Vercel production deployment
    "https://x-med-fusion-8uf7.vercel.app",
]
_extra_origins = os.getenv("EXTRA_CORS_ORIGINS", "").strip()
if _extra_origins:
    CORS_ALLOWED_ORIGINS += [o.strip() for o in _extra_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
    expose_headers=["*"],
)

UPLOAD_DIR = "uploads"
CT_MAX_UPLOAD_FILES = int(os.getenv("CT_MAX_UPLOAD_FILES", "300"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text or "")]


def _representative_sample_indices(total: int, limit: int, *, trim_top: float = 0.15, trim_bottom: float = 0.10) -> list[int]:
    if total <= 0:
        return []
    if total <= limit:
        return list(range(total))

    start = int(total * trim_top)
    end = int(total * (1.0 - trim_bottom))
    if end <= start:
        start, end = 0, total

    usable = list(range(start, end))
    if not usable:
        usable = list(range(total))
    if len(usable) <= limit:
        return usable
    if limit == 1:
        return [usable[len(usable) // 2]]

    positions = [round(i * (len(usable) - 1) / (limit - 1)) for i in range(limit)]
    return [usable[pos] for pos in positions]


def _validation_sample_indexes(total_files: int, scan_type: str) -> list[int]:
    return list(range(total_files))


def _find_duplicate_names(names: list[str]) -> list[str]:
    seen = {}
    duplicates = []
    for name in names:
        normalized = (name or "").strip().lower()
        if not normalized:
            continue
        if normalized in seen:
            duplicates.append(name)
        else:
            seen[normalized] = name
    return duplicates


def _validate_scan_batch(image_paths: list[str], scan_type: str, original_names: list[str] | None = None) -> dict:
    duplicate_names = _find_duplicate_names(original_names or [])
    if duplicate_names:
        return {
            "valid": False,
            "requested_scan_type": scan_type,
            "detected_modality": None,
            "sampled_results": [],
            "sampled_count": 0,
            "counts": {"xray": 0, "ct": 0, "invalid": 0},
            "message": f"Duplicate scan names detected. Remove repeated files such as '{duplicate_names[0]}'.",
        }

    classify_scan = _get_scan_classifier()
    sampled_indexes = _validation_sample_indexes(len(image_paths), scan_type)
    sampled_paths = [image_paths[idx] for idx in sampled_indexes]
    results = []
    counts = {"xray": 0, "ct": 0, "invalid": 0}

    for idx, path in zip(sampled_indexes, sampled_paths):
        modality, confidence = classify_scan(path)
        counts[modality] = counts.get(modality, 0) + 1
        results.append(
            {
                "index": idx,
                "filename": os.path.basename(path),
                "modality": modality,
                "confidence": round(float(confidence), 4),
            }
        )

    dominant_modality = max(("xray", "ct", "invalid"), key=lambda key: counts.get(key, 0))
    has_invalid = counts["invalid"] > 0
    has_mixed_valid = counts["xray"] > 0 and counts["ct"] > 0

    if has_invalid and has_mixed_valid:
        valid = False
        message = "Uploaded files contain a mix of CT, X-ray, and invalid images. All files in the batch must belong to the same scan modality."
    elif has_invalid:
        valid = False
        invalid_files = [result["filename"] for result in results if result["modality"] == "invalid"]
        example = invalid_files[0] if invalid_files else "one or more files"
        message = f"One or more uploads do not appear to be valid medical scans, for example '{example}'."
    elif scan_type == "auto":
        valid = not has_mixed_valid
        message = (
            "Mixed scan modalities were detected in this upload batch. All uploaded files must be either CT scans or X-rays."
            if has_mixed_valid
            else f"Detected {dominant_modality.upper()} study."
        )
    else:
        valid = dominant_modality == scan_type and not has_mixed_valid
        expected = "X-ray" if scan_type == "xray" else "CT"
        actual = dominant_modality.upper()
        message = (
            f"Uploaded files appear to be mixed between CT and X-ray. All uploaded files must be {expected} images for this study."
            if has_mixed_valid
            else f"Expected a {expected} study, but sampled files look like {actual}."
        )
        if valid:
            message = f"Validated {expected} study."

    return {
        "valid": valid,
        "requested_scan_type": scan_type,
        "detected_modality": None if has_invalid or has_mixed_valid else dominant_modality,
        "sampled_results": results,
        "sampled_count": len(results),
        "counts": counts,
        "message": message,
    }


async def _store_uploads(files: List[UploadFile], scan_type: str) -> list[dict]:
    stored_uploads = []
    for file in files:
        file_id = str(uuid.uuid4())
        original_ext = os.path.splitext(file.filename or "")[1].lower()
        safe_ext = original_ext if original_ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".png"
        image_path = os.path.join(UPLOAD_DIR, f"{file_id}{safe_ext}")

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        stored_uploads.append({
            "original_name": file.filename or "",
            "image_path": image_path,
        })

    if scan_type == "ct":
        stored_uploads.sort(key=lambda item: _natural_sort_key(item["original_name"]))

    return stored_uploads


@app.post("/api/validate-upload-batch")
async def validate_upload_batch(
    files: List[UploadFile] = File(...),
    scan_type: str = Form("auto")
):
    if scan_type == "ct" and len(files) > CT_MAX_UPLOAD_FILES:
        return {
            "valid": False,
            "message": f"Upload limit exceeded. Upload up to {CT_MAX_UPLOAD_FILES} images per study.",
            "requested_scan_type": scan_type,
        }

    stored_uploads = await _store_uploads(files, scan_type)
    image_paths = [item["image_path"] for item in stored_uploads]
    original_names = [item["original_name"] for item in stored_uploads]
    try:
        return await asyncio.to_thread(_validate_scan_batch, image_paths, scan_type, original_names)
    finally:
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as cleanup_error:
                print(f"⚠️ Could not clean validation temp file {path}: {cleanup_error}")


@app.post("/api/synthesize-report")
async def synthesize_report(
    files: List[UploadFile] = File(...),
    # Optional frontend parameter: "xray", "ct", or "auto"
    scan_type: str = Form("auto") 
):
    if scan_type == "ct" and len(files) > CT_MAX_UPLOAD_FILES:
        async def too_many_ct_files():
            yield json.dumps({
                "status": "error",
                "message": f"Upload limit exceeded. Upload up to {CT_MAX_UPLOAD_FILES} images per study."
            }) + "\n"
        return StreamingResponse(too_many_ct_files(), media_type="application/x-ndjson")

    stored_uploads = await _store_uploads(files, scan_type)
    image_paths = [item["image_path"] for item in stored_uploads]

    original_names = [item["original_name"] for item in stored_uploads]
    validation = await asyncio.to_thread(_validate_scan_batch, image_paths, scan_type, original_names)
    if not validation.get("valid"):
        async def invalid_batch():
            yield json.dumps({
                "status": "error",
                "message": validation.get("message", "Uploaded files failed validation."),
                "validation": validation,
            }) + "\n"
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as cleanup_error:
                print(f"⚠️ Could not clean invalid upload temp file {path}: {cleanup_error}")
        return StreamingResponse(invalid_batch(), media_type="application/x-ndjson")

    effective_scan_type = validation.get("detected_modality") if scan_type == "auto" else scan_type

    # 2. Process scans through the shared report pipeline.
    runtime = await asyncio.to_thread(_ensure_report_agents)
    report_generator = agents["synthesis"].generate_final_report(
        draft_agent=agents["draft"],
        vision_agent=agents["vision"],
        retrieval_agent=agents["retrieval"],
        reports_dict=runtime["reports_dict"],
        image_paths=image_paths,
        scan_type=effective_scan_type
    )

    async def process_stream(generator):
        try:
            async for chunk in generator:
                try:
                    data = json.loads(chunk)
                    if data.get("status") == "complete":
                        exp_path = data.get("explainable_image_path")
                        ref_path = data.get("explainability_reference_image_path")
                        if exp_path and os.path.exists(exp_path):
                            with open(exp_path, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode("utf-8")
                            data["heatmap"] = f"data:image/png;base64,{b64_data}"
                        else:
                            data["heatmap"] = None
                        if ref_path and os.path.exists(ref_path):
                            with open(ref_path, "rb") as f:
                                ref_b64_data = base64.b64encode(f.read()).decode("utf-8")
                            data["reference_image"] = f"data:image/png;base64,{ref_b64_data}"
                        else:
                            data["reference_image"] = None
                        print("[DEBUG] Yielding final JSON chunk from process_stream")
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
        "ai_warmup_started": _ai_warmup_started,
        "ai_warmup_complete": _ai_warmup_complete,
        "ai_models_loaded": bool(agents),
    }
    result.update(_optional_model_status)
    if _ai_warmup_error:
        result["ai_warmup_error"] = _ai_warmup_error
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

class ExplainRequest(BaseModel):
    findings: str
    impression: str
    modality: str = "xray"

@app.post("/api/explain")
async def generate_explanation(req: ExplainRequest):
    try:
        llm = _get_explain_llm()
        
        prompt = f"""
        Act as an expert Radiologist and AI Explainer for the X-MedFusion application.
        
        The AI pipeline has already synthesized the following report for this patient:
        FINDINGS: {req.findings}
        IMPRESSION: {req.impression}
        
        The patient is viewing their original {req.modality.upper()} study alongside an Insights overlay generated by our Vision Agent.
        
        Please provide a highly-structured "Automated Diagnostic Narrative" that explains HOW the AI arrived at this conclusion. 
        Format your response EXACTLY in these three steps:
        
        **Step 1: Visual Feature Extraction**
        Explain what the Vision Agent highlighted in the overlay.

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
