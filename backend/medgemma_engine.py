# medgemma_engine.py
"""
Singleton MedGemma Engine for XMedFusion.
Loads the model ONCE and provides a unified interface for all agents.
"""

from __future__ import annotations
import torch
from pathlib import Path
from PIL import Image
from typing import Union, List, Optional
import threading

# --- Singleton Engine ---
class MedGemmaEngine:
    """
    Singleton class that loads MedGemma once and provides:
    - generate_from_image(image, prompt) -> str
    - generate_text(prompt) -> str
    """
    _instance: Optional["MedGemmaEngine"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        print("🧠 Loading MedGemma via HuggingFace Transformers...")
        
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError(
                "Please install transformers >= 4.50.0:\n"
                "  pip install -U transformers accelerate"
            )
        
        self.model_id = "google/medgemma-1.5-4b-it"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"   Model: {self.model_id}")
        print(f"   Device: {self.device}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Load model with bfloat16 for efficiency
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        print("✅ MedGemma loaded successfully!")
        self._initialized = True
    
    def generate_from_image(
        self, 
        image: Union[str, Path, Image.Image], 
        prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Generate text from an image + prompt.
        Used by Vision Agent and KG Agent.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Expected path or PIL Image, got {type(image)}")
        
        # Build message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
            generation = generation[0][input_len:]
        
        # Decode
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()
    
    def generate_text(
        self, 
        prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Generate text from a text-only prompt.
        Used by Draft Agent and Synthesis Agent.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
            generation = generation[0][input_len:]
        
        # Decode
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()


# --- Global accessor ---
_engine: Optional[MedGemmaEngine] = None

def get_medgemma_engine() -> MedGemmaEngine:
    """Get or create the singleton MedGemma engine."""
    global _engine
    if _engine is None:
        _engine = MedGemmaEngine()
    return _engine


# --- Standalone Test ---
if __name__ == "__main__":
    import os
    
    print("\n=== MedGemma Engine Test ===\n")
    
    # Initialize engine
    engine = get_medgemma_engine()
    
    # Test 1: Text generation
    print("\n--- Test 1: Text Generation ---")
    text_prompt = "What are the typical findings in a normal chest X-ray?"
    result = engine.generate_text(text_prompt)
    print(f"Prompt: {text_prompt}")
    print(f"Response: {result[:500]}...")
    
    # Test 2: Image + text generation
    print("\n--- Test 2: Image + Text Generation ---")
    test_img = "data/iu_xray/images/CXR1_1_IM-0001/1.png"
    if os.path.exists(test_img):
        img_prompt = "Describe this chest X-ray image. What findings do you observe?"
        result = engine.generate_from_image(test_img, img_prompt)
        print(f"Image: {test_img}")
        print(f"Prompt: {img_prompt}")
        print(f"Response: {result}")
    else:
        print(f"Test image not found: {test_img}")
    
    print("\n=== Test Complete ===")
