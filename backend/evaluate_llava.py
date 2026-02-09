import os
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
import base64
import config
import traceback

# CONFIG
ANNOTATIONS_PATH = "data/iu_xray/annotation.json"
IMAGES_ROOT = "data/iu_xray/images"
OUTPUT_FILE = "out/test_generations_llava.json"
MODEL_NAME = "llava-med"  # User must create this model in Ollama
TIMEOUT_SECONDS = 180

async def run_llava_evaluation():
    print(f"🚀 Starting LLaVA-Med Evaluation using model: {MODEL_NAME}")
    
    # Load Data
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)
    test_data = data.get('test', []) if isinstance(data, dict) else [x for x in data if x.get('split') == 'test']
    
    # Load Previous Results
    results_log = []
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results_log = json.load(f)
                processed_ids = {item['id'] for item in results_log}
            print(f"✅ Resuming from {len(processed_ids)} samples.")
        except: pass

    # Initialize Ollama
    llm = ChatOllama(model=MODEL_NAME, temperature=0.2) 

    # Loop
    for idx, ex in enumerate(tqdm(test_data)):
        sample_id = str(idx)
        if sample_id in processed_ids:
            continue

        img_path = os.path.join(IMAGES_ROOT, ex['image_path'][0])
        ground_truth = ex['report']
        
        if not os.path.exists(img_path):
            continue

        try:
            # Encode Image
            with open(img_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Prompt
            # LLaVA-Med specific prompt or generic? 
            # Generic LLaVA prompt often works best if model is aligned.
            prompt = "Describe the findings in this chest x-ray image in detail."
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"},
                ]
            )

            # Generate with Timeout
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(llm.invoke, [message]), 
                    timeout=TIMEOUT_SECONDS
                )
                generated_text = response.content
            except asyncio.TimeoutError:
                print(f"⏰ Timeout on sample {idx}")
                generated_text = "TIMEOUT_ERROR"

            # Save
            entry = {
                "id": sample_id,
                "image": img_path,
                "raw_generated": generated_text,
                "final_formatted": generated_text.lower().strip(), # LLaVA output is direct
                "ground_truth": ground_truth,
                # No Judge here? User can run calculate_metrics.py later? 
                # Or we can add judge score placeholders.
                "judge_score": 0 
            }
            results_log.append(entry)
            
            # Atomic Save
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results_log, f, indent=2)

        except Exception as e:
            print(f"Error on {idx}: {e}")
            traceback.print_exc()
            continue

    print(f"✅ Done. Saved to {OUTPUT_FILE}")
    print(f"To calculate metrics run: python calculate_metrics.py {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(run_llava_evaluation())
