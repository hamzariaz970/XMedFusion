import torch
from PIL import Image
import os
import sys

# Add current directory to path to import vision
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vision import vision_encoder, is_medical_scan

def verify_scan(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    print(f"--- Verifying: {image_path} ---")
    try:
        img = Image.open(image_path).convert("RGB")
        is_valid, score, label = is_medical_scan(img, vision_encoder)
        
        print(f"RESULT: {'✅ VALID SCAN' if is_valid else '❌ INVALID IMAGE'}")
        print(f"SCORE: {score:.4f}")
        print(f"DETECTED AS: {label}")
    except Exception as e:
        print(f"❌ Error during verification: {e}")
    print("-" * 30)

if __name__ == "__main__":
    # Test with a few images if provided as args, else look for a default one
    if len(sys.argv) > 1:
        for img in sys.argv[1:]:
            verify_scan(img)
    else:
        # Try to find a sample image in the data directory
        sample_path = "data/iu_xray/images/CXR3655_IM-1817/0.png"
        if os.path.exists(sample_path):
            verify_scan(sample_path)
        else:
            print("Usage: python verify_validation.py path/to/image.png")
            print("Example: python verify_validation.py data/iu_xray/images/CXR3655_IM-1817/0.png")
