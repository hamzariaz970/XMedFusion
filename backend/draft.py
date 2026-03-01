# draft.py

import os
import json
import pickle
import torch
import re
import warnings
from pathlib import Path
from PIL import Image
from torch.nn.functional import cosine_similarity
# CHANGED: Using native ChatOllama for better handling of deepseek outputs
from langchain_community.chat_models import ChatOllama 

import config

# Suppress specific torch load warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

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

# -------------------------------
# Retrieval Agent (Uses shared BioMedCLIP VisionEncoder)
# -------------------------------
class RetrievalAgent:
    """
    Retrieves top-k similar reports using a shared VisionEncoder (BioMedCLIP).
    The encoder provides encode_image() and encode_text() in the same embedding space.
    """
    def __init__(self, vision_encoder, k=3):
        self.encoder = vision_encoder
        self.k = k
        print(f"✅ Retrieval Agent initialized (BioMedCLIP). Top-k={self.k}")

    def retrieve_top_k(self, image_paths, reports_dict):
        all_reports = list(set(reports_dict.values()))
        if not all_reports:
            print("⚠️ No reports found in dictionary.")
            return []

        # Encode the target images using BioMedCLIP and average them
        img_feats = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            feat = self.encoder.encode_image(img)
            img_feats.append(feat)
            
        # Average the embeddings across all views and re-normalize
        avg_img_feat = torch.mean(torch.cat(img_feats, dim=0), dim=0, keepdim=True)
        avg_img_feat = torch.nn.functional.normalize(avg_img_feat, dim=-1)

        # Encode all report texts using BioMedCLIP (256-token PubMedBERT tokenizer)
        # Process in batches to avoid OOM on large report sets
        BATCH_SIZE = 64
        text_features_list = []
        for i in range(0, len(all_reports), BATCH_SIZE):
            batch = all_reports[i:i + BATCH_SIZE]
            batch_feat = self.encoder.encode_text(batch)
            text_features_list.append(batch_feat)
        text_features = torch.cat(text_features_list, dim=0)

        # Compute cosine similarity
        sims = cosine_similarity(avg_img_feat, text_features)
        
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
    def __init__(self, model_name=config.OLLAMA_MODEL): 
            if model_name.startswith("ollama/"):
                model_name = model_name.split("/", 1)[1]
            
            self.llm = ChatOllama(
                model=model_name, 
                temperature=config.TEMPERATURE
            )

    def generate_report(self, visual_description):
        prompt = (
            "You are a helpful assistant assisting a radiologist.\n"
            "Here are snippets from historical X-ray reports that are visibly similar to the current case:\n\n"
            f"{visual_description}\n\n"
            "Summarize the COMMON findings and the TYPICAL WRITING STYLE used in these examples.\n"
            "Do NOT write a full report. Do NOT diagnose the current patient.\n"
            "Just list:\n"
            "1. Common pathologies mentioned (if any).\n"
            "2. Key phrases or stylistic patterns observed.\n"
            "3. Pertinent negatives (what is typically 'normal' in these cases)."
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
    # Import vision encoder for testing
    from vision import vision_encoder

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
        # 1️⃣ Retrieve top-k similar reports (now using BioMedCLIP)
        retrieval_agent = RetrievalAgent(vision_encoder, k=5)
        top_reports = retrieval_agent.retrieve_top_k([image_path], reports_dict)

        # 2️⃣ Generate structured report using local LLM
        llm_agent = LocalLLMReportAgent()
        visual_desc = "\n\n".join(truncate_report(r, 75) for r in top_reports)
        
        draft_report = llm_agent.generate_report(visual_desc)

        print("\n📝 Draft Radiology Report:\n")
        print(draft_report)
    else:
        print("❌ Could not run pipeline: Missing image or empty reports dictionary.")