import asyncio
import json
import os
from grok_agent import GroqKGAgent

async def demo_grok_enrichment():
    # 1. Initialize the Groq/Grok Agent
    agent = GroqKGAgent()
    
    # 2. Example Report (In a real scenario, this comes from synthesis.py)
    sample_report = """
    FINDINGS: 
    The heart is mildly enlarged. Small bilateral pleural effusions are present.
    There is no evidence of pneumothorax.
    The lungs show mild bibasilar atelectasis but no focal consolidation.
    
    IMPRESSION: 
    1. Mild cardiomegaly.
    2. Small bilateral pleural effusions.
    3. No acute pneumonia.
    """
    
    print("\n--- [STARTING GROQ/GROK ENRICHMENT] ---")
    print(f"Report to process:\n{sample_report}")
    
    # 3. Extract High-Fidelity KG using Llama-3-70B on Groq
    print("\n[Step 1] Extracting Enriched Knowledge Graph...")
    enriched_kg = agent.extract_enriched_kg(sample_report)
    
    if enriched_kg:
        print("\n✅ Enriched KG Entities:")
        for entity in enriched_kg.get("entities", []):
            print(f"  - {entity[0]} ({entity[1]})")
            
        print("\n✅ Enriched KG Relations:")
        for rel in enriched_kg.get("relations", []):
            try:
                src = enriched_kg['entities'][rel[0]][0]
                tgt = enriched_kg['entities'][rel[1]][0]
                print(f"  - {src} --[{rel[2]}]--> {tgt}")
            except:
                pass
    
    # 4. Refine Interpretability reasoning
    print("\n[Step 2] Generating Expert Explainability Trace...")
    trace = agent.refine_explainability_reasoning(sample_report, enriched_kg)
    
    print("\n✅ AI Interpretability Trace:")
    for i, step in enumerate(trace):
        print(f"  {i+1}. {step}")

    print("\n--- [ENRICHMENT COMPLETE] ---")

if __name__ == "__main__":
    asyncio.run(demo_grok_enrichment())
