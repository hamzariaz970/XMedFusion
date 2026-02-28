import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

class RealWorldFilterDataset(Dataset):
    def __init__(self, xrays_dir, random_dir, num_samples=200):
        self.samples = []
        
        # 1. Load Positive Samples (Real X-rays)
        x_count = 0
        for root, _, files in os.walk(xrays_dir):
            for f in files:
                if f.endswith((".png", ".jpg")) and x_count < num_samples:
                    self.samples.append((os.path.join(root, f), 1.0))
                    x_count += 1
            if x_count >= num_samples: break
            
        # 2. Load Negative Samples (Random images from your new folder)
        r_count = 0
        for root, _, files in os.walk(random_dir):
            for f in files:
                if f.endswith((".png", ".jpg", ".jpeg", ".avif")) and r_count < num_samples:
                    self.samples.append((os.path.join(root, f), 0.0))
                    r_count += 1
            if r_count >= num_samples: break
        
        print(f"📊 Dataset Ready: {x_count} X-rays, {r_count} Random images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                features = vision_encoder.encode_image(img)
            return features.squeeze(0), torch.tensor([label], dtype=torch.float32)
        except Exception as e:
            # Return a zero tensor if an image is corrupted
            return torch.zeros(512), torch.tensor([label], dtype=torch.float32)

def train_bouncer():
    # Update these paths to match your machine exactly
    XRAY_PATH = "data/iu_xray/images"
    RANDOM_PATH = "data/test"
    
    os.makedirs("model_weights", exist_ok=True)
    dataset = RealWorldFilterDataset(XRAY_PATH, RANDOM_PATH, num_samples=200)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = vision_encoder.device
    model = XRayBouncerHead().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4) # Slightly lower learning rate for stability
    criterion = nn.BCELoss()
    
    print("🚀 Training X-Ray Filter with Real-World Negative Samples...")
    for epoch in range(20):
        model.train()
        l_sum = 0
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        print(f"Epoch {epoch+1}/20 - Loss: {l_sum/len(loader):.4f}")
    
    torch.save(model.state_dict(), "model_weights/xray_filter_head.pth")
    print("✅ Balanced filter trained and saved!")

if __name__ == "__main__":
    train_bouncer()