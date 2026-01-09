# draft.py

import os
import json
import pickle
import torch
import clip
import re
import warnings
from pathlib import Path
from PIL import Image
from torch.nn.functional import cosine_similarity
# CHANGED: Using native ChatOllama for better handling of deepseek outputs
from langchain_community.chat_models import ChatOllama 

# Suppress specific torch load warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# Device
# -------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# -------------------------------
# Load fine-tuned CLIP
# -------------------------------
clip_model_path = r"model_weights/Draft_Agent/clip_medical.pth"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

# Check if weights file exists before loading to avoid generic errors
if not os.path.exists(clip_model_path):
    raise FileNotFoundError(f"Model weights not found at: {clip_model_path}")

# safe_globals or strict loading might be needed in future torch versions, 
# but for now we suppress the warning via filterwarnings above.
model.load_state_dict(torch.load(clip_model_path, map_location=device))
model.eval()
print("✅ Fine-tuned CLIP loaded successfully")

# -------------------------------
# Load dataset and map image paths to reports (with caching)
# -------------------------------
annotation_path = r"data/iu_xray/annotation.json"
image_base_dir = r"data/iu_xray/images"
BASE = Path(__file__).resolve().parent
cache_file = BASE / "data" / "cache" / "reports_dict.pkl"
cache_file.parent.mkdir(parents=True, exist_ok=True)


if os.path.exists(cache_file):
    # Load cached mapping
    with open(cache_file, "rb") as f:
        reports_dict = pickle.load(f)
    print(f"✅ Loaded cached reports_dict with {len(reports_dict)} entries")
else:
    # Build mapping
    if not os.path.exists(annotation_path):
        print(f"⚠️ Annotation file not found at {annotation_path}. skipping map build.")
        reports_dict = {}
    else:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)["train"]

        reports_dict = {}
        for item in annotations:
            report_text = item["report"]
            for rel_path in item["image_path"]:
                # Build full path
                png_path = os.path.join(image_base_dir, rel_path.replace("/", os.sep))
                if os.path.exists(png_path):
                    reports_dict[png_path] = report_text

        # Save cache
        with open(cache_file, "wb") as f:
            pickle.dump(reports_dict, f)
        print(f"✅ Built and cached reports_dict with {len(reports_dict)} entries")

# -------------------------------
# Helper functions
# -------------------------------
def truncate_report(text, max_words=75):
    tokens = text.split()
    return " ".join(tokens[:max_words])

def encode_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
    return img_feat

def encode_texts(texts):
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        txt_feat = model.encode_text(tokens)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    return txt_feat

# -------------------------------
# Retrieval Agent
# -------------------------------
class RetrievalAgent:
    def __init__(self, clip_model, preprocess, k=5, device='cuda'):
        self.model = clip_model
        self.preprocess = preprocess
        self.device = device
        self.k = k
        self.model.eval()
        print(f"✅ Retrieval Agent initialized. Top-k={self.k}")

    def retrieve_top_k(self, image_path, reports_dict):
        all_reports = list(set(reports_dict.values()))
        if not all_reports:
            print("⚠️ No reports found in dictionary.")
            return []
            
        text_features = encode_texts(all_reports)
        img_feat = encode_image(image_path)
        sims = cosine_similarity(img_feat, text_features)
        
        # Determine actual k based on available reports
        actual_k = min(self.k, len(all_reports))
        if actual_k == 0:
            return []
            
        topk_idx = sims.topk(actual_k).indices.tolist()
        top_reports = [all_reports[i] for i in topk_idx]
        top_scores = [sims[i].item() for i in topk_idx]

        for i, (rep, score) in enumerate(zip(top_reports, top_scores)):
            print(f"[{i+1}] Score: {score:.4f}\n{rep}\n{'-'*50}")
        return top_reports

# -------------------------------
# Local LLM Agent using Ollama (Corrected)
# -------------------------------
class LocalLLMReportAgent:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        # Strip 'ollama/' prefix if present, ChatOllama expects just the model name
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
            
        self.llm = ChatOllama(
            model=model_name, 
            temperature=0.1  # Low temp for deterministic medical reports
        )

    def generate_report(self, visual_description):
        prompt = (
            "You are an expert thoracic radiologist. "
            "Using the following X-ray image features, generate a formal radiology report "
            "in narrative style (no bullet points). "
            "Organize the report into sections:\n"
            "FINDINGS: Describe abnormalities and normal structures.\n"
            "IMPRESSION: Summarize diagnostic conclusions.\n"
            "LABELS: List present conditions as words separated by commas.\n\n"
            f"Visual Features: {visual_description}\n"
            "\n=== Start Formal Radiology Report ==="
        )
        
        # Single invocation
        print("⏳ Generating report with Local LLM...")
        response = self.llm.invoke(prompt)
        
        content = response.content if hasattr(response, "content") else str(response)

        # CLEANUP: Remove <think> tags from DeepSeek/reasoning models
        # This regex removes everything between <think> and </think> including newlines
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        return clean_content


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example image (replace with any preprocessed image from dataset)
    image_path = os.path.join(image_base_dir, "CXR10_IM-0002", "0.png")

    if not os.path.exists(image_path):
        # Fallback for testing if specific image doesn't exist
        print(f"⚠️ Image not found: {image_path}")
        # Try to find the first available image in the base dir for demo purposes
        if os.path.exists(image_base_dir):
            for root, dirs, files in os.walk(image_base_dir):
                for file in files:
                    if file.endswith(".png"):
                        image_path = os.path.join(root, file)
                        print(f"⚠️ Switching to available image: {image_path}")
                        break
                if image_path.endswith(".png"): break

    if os.path.exists(image_path) and reports_dict:
        # 1️⃣ Retrieve top-k similar reports
        retrieval_agent = RetrievalAgent(model, preprocess, k=5, device=device)
        top_reports = retrieval_agent.retrieve_top_k(image_path, reports_dict)

        # 2️⃣ Generate structured report using local LLM
        # Note: You can pass just "deepseek-r1:1.5b" now
        llm_agent = LocalLLMReportAgent(model_name="deepseek-r1:1.5b")
        visual_desc = "\n\n".join(truncate_report(r, 75) for r in top_reports)
        
        draft_report = llm_agent.generate_report(visual_desc)

        print("\n📝 Draft Radiology Report:\n")
        print(draft_report)
    else:
        print("❌ Could not run pipeline: Missing image or empty reports dictionary.")