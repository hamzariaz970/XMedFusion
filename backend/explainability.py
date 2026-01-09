import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import traceback

# Import your specific model instances and device from vision.py
from vision import vision_model, proj_heads, device, preprocess_image

# -------------------------------
# Grad-CAM for Vision Transformers
# -------------------------------
class ViTGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.eval()
        
        # 1. UNFREEZE everything needed for the forward pass
        # We need both the extractor (timm model) and the projection layers
        for p in self.model.visual_extractor.parameters():
            p.requires_grad = True
        for p in self.model.visual_projection.parameters():
            p.requires_grad = True
            
        self.model.zero_grad()
        
        # 2. MANUAL FORWARD PASS (Bypassing extract_visual_tokens to avoid torch.no_grad)
        # We replicate the logic from vision.py here:
        
        # A. Extract features from the timm backbone
        # We assume the hook is on a layer inside visual_extractor
        patch_tokens = self.model.visual_extractor.forward_features(x)
        
        # B. Slice to remove CLS token or handle specific output format
        # ViT usually outputs [Batch, N, Dim]. Index 1: are patches.
        patch_tokens = patch_tokens[:, 1:, :]
        
        # C. Project to embedding dimension
        output_tokens = self.model.visual_projection(patch_tokens)
        
        # 3. Define Target & Backprop
        # Maximize the sum of activations to find "hot" areas
        score = output_tokens.sum()
        score.backward()
        
        # 4. Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Safety Check
        if gradients is None or activations is None:
            # Re-freeze before crashing
            for p in self.model.visual_extractor.parameters(): p.requires_grad = False
            for p in self.model.visual_projection.parameters(): p.requires_grad = False
            raise RuntimeError("Gradients were not captured. Check if hooks are attached correctly.")

        b, n, d = activations.shape
        
        # Handle ViT token slicing (if CLS token exists in activations)
        # 14x14 = 196 patches. If we have 197, index 0 is CLS.
        if n == 197:
            gradients = gradients[:, 1:, :]
            activations = activations[:, 1:, :]
        
        # Weight activations by average gradient
        weights = gradients.mean(dim=1, keepdim=True)
        cam = (weights * activations).sum(dim=2)

        # ReLU (Clip negative)
        cam = cam.clamp(min=0)
        
        # Reshape 1D patches back to 2D
        h = w = int(np.sqrt(cam.shape[1]))
        cam = cam.reshape(b, h, w)
        
        # 5. CLEANUP: Re-freeze the model parameters
        for p in self.model.visual_extractor.parameters():
            p.requires_grad = False
        for p in self.model.visual_projection.parameters():
            p.requires_grad = False
            
        return cam

# -------------------------------
# Visualization Utils
# -------------------------------
def process_heatmap(cam, input_image_size=(224, 224)):
    """Resizes 14x14 CAM to 224x224 and normalizes it."""
    cam = cam.detach().cpu().numpy()[0] 
    
    # Normalize 0-1
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    # Resize to image size
    cam = cv2.resize(cam, input_image_size)
    return cam

def overlay_heatmap(pil_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlays the heatmap onto the original PIL image."""
    img_np = np.array(pil_image)
    
    # Ensure dimensions match
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Apply Color Map
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    blended = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(blended)

# -------------------------------
# Main Pipeline Function
# -------------------------------
def run_explainability(image_path, output_dir="uploads"):
    """
    Main function called by API.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Target the LayerNorm of the last attention block
    # For 'vit_base_patch16_224', blocks[-1].norm1 is usually the best spot.
    target_layer = vision_model.visual_extractor.blocks[-1].norm1
    
    grad_cam = ViTGradCAM(vision_model, target_layer)

    # 2. Preprocess
    img_tensor = preprocess_image(image_path).to(device)
    img_tensor.requires_grad = True 

    # 3. Compute Heatmap
    cam_tensor = grad_cam(img_tensor)
    
    # 4. Process and Overlay
    heatmap_raw = process_heatmap(cam_tensor)
    
    original_pil = Image.open(image_path).convert("RGB")
    final_image = overlay_heatmap(original_pil, heatmap_raw)

    # 5. Save
    filename = os.path.basename(image_path)
    save_name = f"heatmap_{filename}"
    save_path = os.path.join(output_dir, save_name)
    
    final_image.save(save_path)
    
    return save_path

# -------------------------------
# Test Block
# -------------------------------
if __name__ == "__main__":
    print("\n--- Running ViT Explainability Test ---")
    
    # UPDATE THIS PATH to a real image on your machine for testing
    test_img = "data/iu_xray/images/CXR688_IM-2256/0.png"
    
    if os.path.exists(test_img):
        try:
            saved_path = run_explainability(test_img, output_dir=".")
            print(f"✅ Heatmap generated successfully: {saved_path}")
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Error: {e}")
    else:
        print(f"Test image not found at {test_img}, skipping test.")