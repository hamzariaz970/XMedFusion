# config.py

# CHANGE THIS ONE LINE to switch models everywhere
# Recommended for RTX 4070ti super (1GB VRAM): "llama3.1" or "deepseek-r1:7b"
OLLAMA_MODEL = "MedAIBase/MedGemma1.5:4b"
OLLAMA_JUDGE_MODEL = "gpt-oss:20b" 

# Global Settings
TEMPERATURE = 0.1
BASE_URL = "http://localhost:11434"