# xray_filter.py

import torch
import torch.nn as nn
import os
from PIL import Image
from vision import vision_encoder

class MedicalScanBouncerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3) # 0: X-Ray, 1: CT, 2: Invalid
        )
    def forward(self, x):
        return self.fc(x)

device = vision_encoder.device
filter_model = MedicalScanBouncerHead().to(device)

# Load the NEW multi-class weights
WEIGHTS_PATH = "model_weights/medical_filter_head.pth"

if os.path.exists(WEIGHTS_PATH):
    # weights_only=True handles the security warning from PyTorch
    filter_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    filter_model.eval()
    print("🛡️ Trained Multi-Class Medical Filter Loaded.")
else:
    print("⚠️ Medical Filter weights not found. Please run train_filter.py first.")

# Map model outputs to friendly names
CLASSES = {0: "xray", 1: "ct", 2: "invalid"}

def classify_scan(image_path):
    """
    Returns: (modality, confidence_score)
    modality will be 'xray', 'ct', or 'invalid'
    """
    try:
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            features = vision_encoder.encode_image(img)
            logits = filter_model(features)
            probs = torch.softmax(logits, dim=-1).squeeze()
            
            # Get the highest probability and its index
            confidence, class_idx = torch.max(probs, dim=0)
            predicted_class = CLASSES[class_idx.item()]
            
        print(f"🛡️ [Filter] {os.path.basename(image_path)} -> Modality: {predicted_class.upper()} (Score: {confidence.item():.4f})")
        return predicted_class, confidence.item()
        
    except Exception as e:
        print(f"⚠️ Filter error: {e}")
        return "invalid", 0.0