# vision.py

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import warnings
import re
from PIL import Image
from torchvision import transforms
import timm
from langchain_community.chat_models import ChatOllama

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
# Model Definitions
# -------------------------------
class VisionEncoderWithProjection(nn.Module):
    def __init__(self, device="cuda", vit_name="vit_base_patch16_224", embed_dim=768, dropout_prob=0.1):
        super().__init__()
        self.device = device
        self.visual_extractor = timm.create_model(
            vit_name, pretrained=True, num_classes=0, global_pool=''
        )
        vit_feat_dim = self.visual_extractor.num_features
        for p in self.visual_extractor.parameters():
            p.requires_grad = False
        self.visual_projection = nn.Sequential(
            nn.Linear(vit_feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(embed_dim, embed_dim)
        )

    def extract_visual_tokens(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            patch_tokens = self.visual_extractor.forward_features(images)
        patch_tokens = patch_tokens[:, 1:, :]
        visual_tokens = self.visual_projection(patch_tokens)
        return visual_tokens

class ProjectionHeads(nn.Module):
    def __init__(self, img_dim=768, txt_dim=384, embed_dim=256):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(img_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(txt_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, img_embeds, txt_embeds=None):
        img_embeds = F.normalize(self.image_proj(img_embeds), dim=-1)
        if txt_embeds is not None:
            txt_embeds = F.normalize(self.text_proj(txt_embeds), dim=-1)
            return img_embeds, txt_embeds
        return img_embeds

# -------------------------------
# Image Preprocessing
# -------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return image_transform(img).unsqueeze(0)

def get_visual_embeddings(image_paths, vision_model, proj_heads):
    vision_model.eval()
    proj_heads.eval()
    
    valid_images = []
    for p in image_paths:
        if os.path.exists(p):
            valid_images.append(preprocess_image(p))
    
    if not valid_images:
        return None

    images = torch.cat(valid_images).to(device)

    with torch.no_grad():
        patch_tokens = vision_model.extract_visual_tokens(images)
        patch_tokens = patch_tokens.flatten(0,1)
        img_embeds = proj_heads.image_proj(patch_tokens)
        img_embeds = img_embeds.mean(dim=0, keepdim=True)
    return img_embeds

def embeddings_to_text(img_embed, top_k=12):
    vec = img_embed.squeeze().cpu().numpy()
    idx = np.argsort(vec)[-top_k:][::-1]
    return ", ".join([f"dim{int(i)}={vec[i]:.2f}" for i in idx])

# -------------------------------
# GLOBAL INITIALIZATION (Fixes ImportError)
# -------------------------------

# 1. Initialize Models Globally
vision_model = VisionEncoderWithProjection(device=device).to(device)
proj_heads = ProjectionHeads(img_dim=768, txt_dim=384, embed_dim=256).to(device)

# 2. Load Weights (Global Scope)
# Adjust paths as needed. Using relative paths assuming execution from project root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_CKPT = os.path.join(BASE_DIR, "model_weights/Vision_Agent/proj_heads.pth")

if os.path.exists(PROJ_CKPT):
    try:
        proj_heads.load_state_dict(torch.load(PROJ_CKPT, map_location=device))
        print(f"✅ Loaded Projection Heads from {PROJ_CKPT}")
    except Exception as e:
        print(f"⚠️ Error loading projection heads: {e}")
else:
    print(f"⚠️ Projection checkpoint not found at {PROJ_CKPT}. Using random weights.")

# Ensure models are in eval mode
vision_model.eval()
proj_heads.eval()

# -------------------------------
# Local LLM Agent
# -------------------------------
class LocalLLMReportAgent:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        self.llm = ChatOllama(model=model_name, temperature=0.1)

    def generate_report(self, visual_description):
        prompt = (
            "You are an expert thoracic radiologist. "
            "Using the following X-ray image features, generate a formal radiology report...\n"
            f"Visual Features: {visual_description}\n"
        )
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return clean_content

# -------------------------------
# Main Execution (Test Only)
# -------------------------------
if __name__ == "__main__":
    # This block now just tests the GLOBALLY defined models
    print("\n--- Running Vision Module Test ---")
    
    test_image = "data/iu_xray/images/CXR688_IM-2256/0.png"
    if os.path.exists(test_image):
        emb = get_visual_embeddings([test_image], vision_model, proj_heads)
        if emb is not None:
            print("Embeddings generated successfully.")
            txt = embeddings_to_text(emb)
            print("Text desc:", txt)
    else:
        print(f"Test image not found at {test_image}")