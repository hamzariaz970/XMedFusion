# train_filter.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
            nn.Linear(128, 3) # <--- 3 Classes: 0 (X-Ray), 1 (CT), 2 (Invalid)
        )
    def forward(self, x):
        return self.fc(x)

class MultiClassFilterDataset(Dataset):
    def __init__(self, xray_dir, ct_dir, random_dir, num_samples=200):
        self.samples = []
        
        # Load Class 0: X-Rays
        self._load_samples(xray_dir, 0, num_samples, "X-Rays")
        # Load Class 1: CT Scans
        self._load_samples(ct_dir, 1, num_samples, "CT Scans")
        # Load Class 2: Invalid/Random
        self._load_samples(random_dir, 2, num_samples, "Invalid/Random")

    def _load_samples(self, directory, label, max_samples, name):
        count = 0
        if not os.path.exists(directory):
            print(f"⚠️ Warning: Directory '{directory}' not found! Skipping...")
            return
            
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(root, f), label))
                    count += 1
                if count >= max_samples: break
        print(f"📊 Loaded {count} {name} (Class {label}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                features = vision_encoder.encode_image(img)
            # Use torch.long for CrossEntropyLoss
            return features.squeeze(0), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            return torch.zeros(512), torch.tensor(label, dtype=torch.long)

def train_bouncer():
    # Update these paths to match your machine exactly
    XRAY_PATH = "data/iu_xray/images"
    CT_PATH = "data/ctscan"
    RANDOM_PATH = "data/test"
    
    os.makedirs("model_weights", exist_ok=True)
    dataset = MultiClassFilterDataset(XRAY_PATH, CT_PATH, RANDOM_PATH, num_samples=200)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = vision_encoder.device
    model = MedicalScanBouncerHead().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss() # <--- Changed for multi-class
    
    print("\n🚀 Training Multi-Class Medical Scan Filter...")
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
    
    # Save with a NEW NAME to avoid shape crashes with the old binary model
    torch.save(model.state_dict(), "model_weights/medical_filter_head.pth")
    print("✅ Multi-Class filter trained and saved!")

if __name__ == "__main__":
    train_bouncer()