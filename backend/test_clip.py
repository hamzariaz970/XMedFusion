import os
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP loaded.")

img_path = "data/iu_xray/images/CXR2915_IM-1317/0.png"
if not os.path.exists(img_path):
    print("Image not found!")
    exit(1)

try:
    print(f"Loading {img_path}...")
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    print("Image preprocessed.")
    
    with torch.no_grad():
        features = model.encode_image(image)
    print("Image encoded successfully!")
    print(f"Features shape: {features.shape}")

except Exception as e:
    print(f"Error: {e}")
