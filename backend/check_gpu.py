import torch

if torch.cuda.is_available():
    print(f"Success! GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected. Running on CPU.")