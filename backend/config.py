import os
from dotenv import load_dotenv

load_dotenv()

# config.py

# CHANGE THIS ONE LINE to switch models everywhere
OLLAMA_MODEL = "MedAIBase/MedGemma1.5:4b"
OLLAMA_JUDGE_MODEL = "gpt-oss:20b"
OLLAMA_LLAVA_MODEL = "rohithbojja/llava-med-v1.6"

# Global Settings
TEMPERATURE = 0.1
CONTEXT_WINDOW = 65536  # Set to 64k for reliable 16GB VRAM performance
BASE_URL = "http://localhost:11434"

# Required for gated models like MedGemma (https://huggingface.co/google/medgemma-4b-it)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_ENDPOINT_URL = os.environ.get("HF_ENDPOINT_URL", "")

# Knowledge Graph Optimization: Selective Pertinent Negatives
# These findings will always be shown in the graph even if they are 'absent'
CRITICAL_RULE_OUTS = {
    "Pneumothorax",
    "Pleural Effusion",
    "Fracture",
    "Cardiomegaly",
    "Nodule"
}
