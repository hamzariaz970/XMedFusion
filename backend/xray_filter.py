import torch
import torch.nn as nn
import os
from PIL import Image
from vision import vision_encoder

class XRayBouncerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

device = vision_encoder.device
filter_model = XRayBouncerHead().to(device)
WEIGHTS_PATH = "model_weights/xray_filter_head.pth"

if os.path.exists(WEIGHTS_PATH):
    filter_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    filter_model.eval()
    print("🛡️ Trained X-Ray Filter Loaded.")
def is_chest_xray(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            features = vision_encoder.encode_image(img)
            confidence = filter_model(features).item()
        
        # Logic: If it's a real X-ray, confidence should be 0.95+. 
        # If it's a random image, it should drop below 0.10.
        print(f"🛡️ [Filter] {os.path.basename(image_path)} -> Score: {confidence:.4f}")
        
        return confidence > 0.90, confidence
    except Exception as e:
        return False, 0.0