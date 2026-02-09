# app.py

from fastapi import FastAPI, UploadFile, File
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
import config  # <--- Global Config
import subprocess
import time
import requests
from transformers import ViTForImageClassification

# Import agents and dependencies
# We import these to initialize them, but we won't run them until a request comes in
from synthesis import (
    LocalSynthesisAgent,
    RetrievalAgent,
    LocalLLMReportAgent,
    VisionLLMAgent,
    reports_dict,
    clip_model,   # Renamed in synthesis.py to be specific
    clip_prep,    # Renamed in synthesis.py to be specific
    device
)

# Import the explainability function
from explainability import run_explainability

# --- GLOBAL AGENT STORE ---
# We keep agents here so they stay in memory (RAM) and don't reload per request
agents = {}

# --- LIFESPAN MANAGER ---
# This runs once when you start the server

def check_and_start_ollama():
    """Checks if Ollama is running, and starts it if not."""
    ollama_url = f"{config.BASE_URL}/api/tags"
    try:
        requests.get(ollama_url, timeout=1)
        print("✅ Ollama is already running.")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("⚠️ Ollama is NOT running. Starting it now...")
        try:
            # Start Ollama in the background
            subprocess.Popen(["ollama", "serve"], shell=True)
            
            # Wait for it to come alive
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
    # Startup - Clean memory first
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("🧹 Memory cleaned before startup...")
    
    check_and_start_ollama()
    
    # Initialize Global Model
    global model
    print("Loading Global Vision Model (ViT)...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )
    print(f"\n🚀 STARTUP: Loading AI Models on {device}...")
    print(f"🧠 LLM Engine: {config.OLLAMA_MODEL}")
    
    # Initialize agents globally
    # Note: These classes now default to config.OLLAMA_MODEL internally
    agents["retrieval"] = RetrievalAgent(clip_model, clip_prep, k=3, device=device)
    agents["draft"] = LocalLLMReportAgent()
    agents["vision"] = VisionLLMAgent()
    agents["synthesis"] = LocalSynthesisAgent()
    
    print("✅ All AI Agents Ready! Server is listening.")
    yield
    print("🛑 Shutting down...")

# Initialize App with lifespan
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
async def synthesize_report(file: UploadFile = File(...)):
    # 1. Save the Uploaded Image
    file_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}.png")

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Generate Heatmap (Explainability)
    # We generate this first so it's ready to be injected into the stream
    heatmap_data_uri = None
    try:
        # Offload blocking heavy compute to thread to keep server responsive
        heatmap_path = await asyncio.to_thread(run_explainability, image_path, UPLOAD_DIR)
        
        # Convert to Base64 Data URI for easy frontend display
        if heatmap_path and os.path.exists(heatmap_path):
            # Using synchronous file read here is okay for small images, 
            # but ideally should be async too if images are huge.
            with open(heatmap_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            heatmap_data_uri = f"data:image/png;base64,{b64_data}"
    except Exception as e:
        print(f"⚠️ Explainability failed: {e}")

    # 3. Create the ASYNC Generator using Pre-Loaded Agents
    # We pass the globally loaded agents into the generator
    report_generator = agents["synthesis"].generate_final_report(
        draft_agent=agents["draft"],
        vision_agent=agents["vision"],
        retrieval_agent=agents["retrieval"],
        reports_dict=reports_dict,
        image_paths=[image_path]
    )

    # 4. Async Wrapper to Inject Heatmap
    # CRITICAL CHANGE: Uses 'async for' because generate_final_report is now async
    async def stream_with_heatmap(generator, heatmap_uri):
        async for chunk in generator:
            try:
                # Chunks are newline-delimited JSON strings. 
                data = json.loads(chunk)
                
                # If this chunk signals completion, inject the heatmap
                if data.get("status") == "complete":
                    data["heatmap"] = heatmap_uri
                    yield json.dumps(data) + "\n"
                else:
                    # Yield intermediate status updates as-is
                    yield chunk
            except Exception:
                # If parsing fails, just pass it through
                yield chunk

    # Return Streaming Response
    return StreamingResponse(
        stream_with_heatmap(report_generator, heatmap_data_uri), 
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    # 0.0.0.0 allows access from other devices/network
    # reload=True is helpful during development
    uvicorn.run(app, host="127.0.0.1", port=8000)