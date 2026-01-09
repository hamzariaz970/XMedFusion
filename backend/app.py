from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import uuid
import os
import json
import base64

# Import agents
from synthesis import (
    LocalSynthesisAgent,
    RetrievalAgent,
    LocalLLMReportAgent,
    VisionLLMAgent,
    reports_dict,
    model,
    preprocess,
    device
)

# Import the explainability function
from explainability import run_explainability

app = FastAPI()

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
    # We run this before the streaming starts so it's ready to be sent
    try:
        heatmap_path = run_explainability(image_path, output_dir=UPLOAD_DIR)
        
        # Convert to Base64 Data URI for easy frontend display
        with open(heatmap_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        heatmap_data_uri = f"data:image/png;base64,{b64_data}"
    except Exception as e:
        print(f"⚠️ Explainability failed: {e}")
        heatmap_data_uri = None

    # 3. Initialize Agents
    retrieval_agent = RetrievalAgent(model, preprocess, k=5, device=device)
    draft_agent = LocalLLMReportAgent(model_name="ollama/deepseek-r1:1.5b")
    vision_agent = VisionLLMAgent(model_name="ollama/deepseek-r1:1.5b")
    synthesis_agent = LocalSynthesisAgent(model_name="ollama/deepseek-r1:1.5b")

    # 4. Create the base Generator
    report_generator = synthesis_agent.generate_final_report(
        draft_agent=draft_agent,
        vision_agent=vision_agent,
        retrieval_agent=retrieval_agent,
        reports_dict=reports_dict,
        image_paths=[image_path]
    )

    # 5. Create a Wrapper to Inject Heatmap into the Final Response
    def stream_with_heatmap(generator, heatmap_uri):
        for chunk in generator:
            try:
                # Chunks are newline-delimited JSON strings. 
                # We check if this chunk is the "complete" status.
                data = json.loads(chunk)
                
                if data.get("status") == "complete":
                    # Inject the pre-calculated heatmap
                    data["heatmap"] = heatmap_uri
                    yield json.dumps(data) + "\n"
                else:
                    # Yield intermediate status updates as-is
                    yield chunk
            except Exception:
                # If parsing fails (e.g. empty lines), just pass it through
                yield chunk

    # Return Streaming Response
    return StreamingResponse(
        stream_with_heatmap(report_generator, heatmap_data_uri), 
        media_type="application/x-ndjson"
    )