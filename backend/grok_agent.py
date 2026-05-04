import os
import json
import re
from openai import OpenAI
import config

class GroqKGAgent:
    """
    Experimental agent using Groq (Llama-3.3-70B) to perform high-fidelity 
    clinical entity extraction and reasoning from radiology reports.
    
    This agent is designed to run in parallel with the local synthesis pipeline
    to provide 'expert-level' graph enrichment.
    """
    
    def __init__(self):
        if not config.GROK_API_KEY:
            print("[GroqKGAgent] WARNING: GROK_API_KEY not found in config.")
        
        self.client = OpenAI(
            api_key=config.GROK_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = config.GROK_MODEL

    def extract_enriched_kg(self, report_text: str) -> dict:
        """
        Uses high-reasoning Llama-3-70B to extract a dense, accurate Knowledge Graph.
        """
        prompt = f"""
        You are an expert Board-Certified Radiologist and Knowledge Engineer.
        
        TASK:
        Extract a structured Knowledge Graph from the following radiology report.
        Focus on identifying specific clinical observations, anatomical locations, 
        and the precise relationship status (present, uncertain, absent).
        
        REPORT:
        {report_text}
        
        OUTPUT FORMAT (Strict JSON):
        {{
          "entities": [
            ["entity_name", "type"],  // type: "Observation", "Anatomy", "UncertainObservation", "AbsentObservation"
            ...
          ],
          "relations": [
            [subject_idx, object_idx, "relation_type"], // relation_type: "located_at", "possible_at", "absent_at", "modify"
            ...
          ],
          "metadata": {{
            "clinical_summary": "One sentence summary",
            "findings_count": 0,
            "reasoning_trace": "Brief explanation of how you linked entities"
          }}
        }}
        
        RULES:
        1. Be extremely precise with anatomy.
        2. Use 'UncertainObservation' for phrases like "may represent", "possible", "equivocal".
        3. Use 'AbsentObservation' for pertinent negatives.
        4. Entity names should be lowercase and clinical (e.g. "small pleural effusion").
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional medical knowledge graph extractor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"[GroqKGAgent] Error extracting KG: {e}")
            return None

    def refine_explainability_reasoning(self, report_text: str, kg_json: dict) -> list[str]:
        """
        Synthesizes a high-level interpretabilty trace that explains 'why' the AI
        generated specific highlights based on the clinical logic.
        """
        prompt = f"""
        Given this radiology report and the corresponding Knowledge Graph, 
        generate a set of 4-5 expert reasoning steps that explain how a vision system
        should prioritize visual highlights for explainability.
        
        REPORT: {report_text}
        KG: {json.dumps(kg_json)}
        
        OUTPUT:
        A list of strings representing the interpretation steps.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in medical AI interpretability."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # Simple line parsing if not returning JSON
            text = response.choices[0].message.content
            steps = [s.strip().lstrip("0123456789. -") for s in text.split("\n") if s.strip()]
            return steps[:6]
        except Exception as e:
            print(f"[GroqKGAgent] Error generating trace: {e}")
            return []

if __name__ == "__main__":
    # Test Block
    test_report = "FINDINGS: Heart size normal. Mild interstitial edema in the lung bases. No pneumothorax. IMPRESSION: Mild pulmonary edema."
    agent = GroqKGAgent()
    result = agent.extract_enriched_kg(test_report)
    print(json.dumps(result, indent=2))
