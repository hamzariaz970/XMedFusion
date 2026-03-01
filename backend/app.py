# app.py

from fastapi import FastAPI, UploadFile, File, Form # <--- Imported Form
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
from transformers import ViTForImageClassification

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
    
    global model
    print("Loading Global Vision Model (ViT)...")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
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
            if scan_type != "auto" and detected_modality != scan_type:
                for path in image_paths:
                    if os.path.exists(path):
                        os.remove(path)
                async def error_stream_mismatch():
                    user_friendly_expected = "Chest X-Ray" if scan_type == "xray" else "CT Scan"
                    user_friendly_detected = "Chest X-Ray" if detected_modality == "xray" else "CT Scan"
                    yield json.dumps({
                        "status": "error", 
                        "message": f"Validation Failed: You selected '{user_friendly_expected}', but uploaded a '{user_friendly_detected}'."
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
        image_paths=image_paths
    )

    async def process_stream(generator):
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
    if _torch.cuda.is_available():
        result["gpu_name"] = _torch.cuda.get_device_name(0)
        result["gpu_memory_used_mb"] = round(_torch.cuda.memory_allocated(0) / (1024 ** 2))
        result["gpu_memory_total_mb"] = round(_torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)