# draft.py

import os
import json
import pickle
import hashlib
import torch
import re
import warnings
import numpy as np
from pathlib import Path
from PIL import Image
from torch.nn.functional import cosine_similarity
# CHANGED: Using native ChatOllama for better handling of deepseek outputs
from langchain_community.chat_models import ChatOllama 

import config
from report_labels import DISEASES, label_audit_from_report

REPORT_CONTEXT_WINDOW = min(config.CONTEXT_WINDOW, 8192)
LLM_TIMEOUT_SECONDS = 180

# Suppress specific torch load warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# Load dataset and map image paths to reports (with caching)
# -------------------------------
annotation_path = r"data/iu_xray/annotation.json"
image_base_dir = r"data/iu_xray/images"
BASE = Path(__file__).resolve().parent
cache_file = BASE / "data" / "cache" / "reports_dict.pkl"
report_records_cache_file = BASE / "data" / "cache" / "report_records.pkl"
text_features_cache_file = BASE / "data" / "cache" / "report_text_features.pt"
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


def _build_report_records():
    if not os.path.exists(annotation_path):
        return []
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f).get("train", [])

    seen = set()
    records = []
    for item in annotations:
        report_text = item.get("report", "")
        if not report_text or report_text in seen:
            continue
        seen.add(report_text)
        labels, _ = label_audit_from_report(report_text)
        records.append({
            "report": report_text,
            "labels": labels.astype(np.float32),
        })
    return records


if os.path.exists(report_records_cache_file):
    with open(report_records_cache_file, "rb") as f:
        report_records = pickle.load(f)
else:
    report_records = _build_report_records()
    with open(report_records_cache_file, "wb") as f:
        pickle.dump(report_records, f)


def _report_text_signature(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["report"].encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


REPORT_TEXT_SIGNATURE = _report_text_signature(report_records)

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
    def __init__(self, vision_encoder, k=3, label_weight=0.0):
        self.encoder = vision_encoder
        self.k = k
        self.label_weight = float(label_weight)
        self._report_records = report_records
        self._report_texts = [record["report"] for record in self._report_records]
        self._label_matrix = (
            np.vstack([record["labels"] for record in self._report_records]).astype(np.float32)
            if self._report_records else np.zeros((0, len(DISEASES)), dtype=np.float32)
        )
        self._text_features = None
        print(f"✅ Retrieval Agent initialized (BioMedCLIP). Top-k={self.k}")

    def _ensure_text_features(self):
        if self._text_features is not None:
            return self._text_features
        if not self._report_texts:
            return None

        if text_features_cache_file.exists():
            try:
                payload = torch.load(text_features_cache_file, map_location="cpu")
                if (
                    payload.get("signature") == REPORT_TEXT_SIGNATURE
                    and int(payload.get("count", -1)) == len(self._report_texts)
                ):
                    cached = payload.get("features")
                    if cached is not None:
                        self._text_features = cached.to(self.encoder.device)
                        print(f"✅ Loaded cached retrieval text features ({len(self._report_texts)} reports)")
                        return self._text_features
            except Exception as exc:
                print(f"⚠️ Failed to load cached text features: {exc}")

        batch_size = 64
        text_features_list = []
        for i in range(0, len(self._report_texts), batch_size):
            batch = self._report_texts[i:i + batch_size]
            batch_feat = self.encoder.encode_text(batch)
            text_features_list.append(batch_feat)
        self._text_features = torch.cat(text_features_list, dim=0)

        try:
            torch.save(
                {
                    "signature": REPORT_TEXT_SIGNATURE,
                    "count": len(self._report_texts),
                    "features": self._text_features.detach().cpu(),
                },
                text_features_cache_file,
            )
            print(f"✅ Cached retrieval text features to {text_features_cache_file}")
        except Exception as exc:
            print(f"⚠️ Failed to cache text features: {exc}")

        return self._text_features

    @staticmethod
    def _normalize_similarities(similarities: torch.Tensor) -> np.ndarray:
        sims = similarities.detach().cpu().numpy().astype(np.float32)
        return np.clip((sims + 1.0) * 0.5, 0.0, 1.0)

    def _label_overlap_scores(self, query_label_scores):
        if not query_label_scores or self._label_matrix.size == 0:
            return None
        weights = np.asarray(
            [float(query_label_scores.get(disease, 0.0)) for disease in DISEASES],
            dtype=np.float32,
        )
        weights = np.clip(weights, 0.0, 1.0)
        if float(weights.sum()) <= 0.0:
            return None
        return (self._label_matrix * weights.reshape(1, -1)).sum(axis=1) / float(weights.sum())

    def retrieve_top_k(self, image_paths, reports_dict, query_label_scores=None):
        text_features = self._ensure_text_features()
        if text_features is None or not self._report_texts:
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

        # Compute cosine similarity
        sims = cosine_similarity(avg_img_feat, text_features)
        sim_scores = self._normalize_similarities(sims)
        overlap_scores = self._label_overlap_scores(query_label_scores)
        if overlap_scores is not None and self.label_weight > 0.0:
            blended_scores = (1.0 - self.label_weight) * sim_scores + self.label_weight * overlap_scores
        else:
            blended_scores = sim_scores
        
        # Determine actual k based on available reports
        actual_k = min(self.k, len(self._report_texts))
        if actual_k == 0:
            return []

        topk_idx = np.argsort(-blended_scores)[:actual_k].tolist()
        top_reports = [self._report_texts[i] for i in topk_idx]

        for rank, idx in enumerate(topk_idx, start=1):
            score = float(blended_scores[idx])
            if overlap_scores is not None and self.label_weight > 0.0:
                print(
                    f"[{rank}] Score: {score:.4f} "
                    f"(img={float(sim_scores[idx]):.4f}, label={float(overlap_scores[idx]):.4f})\n"
                    f"{self._report_texts[idx]}\n{'-'*50}"
                )
            else:
                print(f"[{rank}] Score: {score:.4f}\n{self._report_texts[idx]}\n{'-'*50}")
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
                temperature=config.TEMPERATURE,
                num_ctx=REPORT_CONTEXT_WINDOW,
                num_predict=320,
                timeout=LLM_TIMEOUT_SECONDS
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
