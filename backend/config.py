# config.py

# CHANGE THIS ONE LINE to switch models everywhere
OLLAMA_MODEL = "MedAIBase/MedGemma1.5:4b"
OLLAMA_JUDGE_MODEL = "gpt-oss:20b"
OLLAMA_LLAVA_MODEL = "rohithbojja/llava-med-v1.6"

# Global Settings
TEMPERATURE = 0.1
CONTEXT_WINDOW = 65536  # Set to 64k for reliable 16GB VRAM performance
BASE_URL = "http://localhost:11434"

HF_TOKEN = ""