# Hybrid Phosphorus Parameter Extraction POC
# Combining ChatExtract + Guardrails approaches

import os
import json
import pandas as pd
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

phosphorus_extraction_tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_phosphorus_parameters",
            "description": "Extract phosphorus removal parameters from wastewater treatment research text",
            "parameters": {
                "type": "object",
                "properties": {
                    "influent_P": {
                        "type": "number",
                        "description": "Influent phosphorus concentration in mg/L"
                    },
                    "effluent_P": {
                        "type": "number",
                        "description": "Effluent phosphorus concentration in mg/L"
                    },
                    "removal_efficiency": {
                        "type": "number",
                        "description": "Phosphorus removal efficiency as decimal (0-1)"
                    },
                    "chemical_dose": {
                        "type": "number",
                        "description": "Chemical coagulant dose in mg/L"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Operating temperature in degrees Celsius"
                    },
                    "pH": {
                        "type": "number",
                        "description": "pH level (unitless, typically 0-14)"
                    },
                    "dissolved_O2": {
                        "type": "number",
                        "description": "Dissolved oxygen concentration in mg/L"
                    },
                    "SRT": {
                        "type": "number",
                        "description": "Solids retention time in days"
                    },
                    "HRT": {
                        "type": "number",
                        "description": "Hydraulic retention time in hours"
                    },
                    "material_description": {
                        "type": "string",
                        "description": "Description of the wastewater or treatment system"
                    }
                },
                "required": ["influent_P"]
            }
        }
    }
]

class HybridPhosphorusExtractor:
    """
    Hybrid extractor combining ChatExtract's conversational workflow
    with Guardrails' structured function calling
    """
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.conversation_history = []
        
    def _add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def _call_llm(self, use_tools: bool = False, max_retries: int = 3) -> Dict:
        """Call LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                if use_tools:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        tools=phosphorus_extraction_tools,
                        temperature=self.temperature
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        temperature=self.temperature,
                        max_tokens=50
                    )
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = [{"role": "system", "content": ""}]
    
    # STAGE 1: Binary Classification
    def classify_sentence_relevance(self, sentence: str) -> bool:
        """
        ChatExtract Stage 1: Check if sentence contains phosphorus data
        """
        self.reset_conversation()
        
        prompt = (
            'Answer "Yes" or "No" only. Does the following text contain '
            'phosphorus removal data, phosphorus concentration values, or '
            'phosphorus treatment parameters?\n\n' + sentence
        )
        
        self._add_message("user", prompt)
        response = self._call_llm()
        answer = response.choices[0].message.content.strip().lower()
        self._add_message("assistant", answer)
        
        return "yes" in answer
    
    # STAGE 2: Context Expansion (automatic)
    def build_passage(self, title: str, prev_sentence: str, target_sentence: str) -> str:
        """
        ChatExtract Stage 2: Build expanded context passage
        """
        return f"Title: {title}\n\nPrevious: {prev_sentence}\n\nTarget: {target_sentence}"
    
    # STAGE 3: Multi-value Detection
    def detect_multiple_values(self, passage: str) -> bool:
        """
        ChatExtract Stage 3: Check if passage contains multiple data points
        """
        prompt = (
            'Answer "Yes" or "No" only. Does the following text contain '
            'more than one distinct phosphorus measurement or parameter value?\n\n' + passage
        )
        
        self._add_message("user", prompt)
        response = self._call_llm()
        answer = response.choices[0].message.content.strip().lower()
        self._add_message("assistant", answer)
        
        return "yes" in answer
    
    # STAGE 4: Structured Extraction with Function Calling
    def extract_parameters(self, passage: str) -> Optional[Dict]:
        """
        Guardrails approach: Use function calling for structured extraction
        """
        prompt = (
            "Extract all phosphorus-related parameters from the following text. "
            "If a parameter is not present, do not include it. Be precise with units.\n\n"
            + passage
        )
        
        self._add_message("user", prompt)
        response = self._call_llm(use_tools=True)
        
        message = response.choices[0].message
        self._add_message("assistant", message.content or "")
        
        # Check if function was called
        if message.tool_calls:
            function_args = json.loads(message.tool_calls[0].function.arguments)
            return function_args
        
        return None
    
    # STAGE 5: Verification with Follow-up Questions
    def verify_parameter(self, param_name: str, param_value: any, 
                        passage: str, unit: str = "") -> bool:
        """
        ChatExtract Stage 5: Verify extracted parameter with uncertainty prompts
        """
        value_str = f"{param_value} {unit}".strip()
        
        prompt = (
            f'There is a possibility that the extracted data is incorrect. '
            f'Answer "Yes" or "No" only. Be very strict. '
            f'Is "{value_str}" the value of {param_name} in the following text?\n\n'
            + passage
        )
        
        self._add_message("user", prompt)
        response = self._call_llm()
        answer = response.choices[0].message.content.strip().lower()
        self._add_message("assistant", answer)
        
        return "yes" in answer
    
    # COMPLETE WORKFLOW
    def extract_from_paper(self, paper_text: str, title: str = "") -> List[Dict]:
        """
        Complete extraction workflow combining all stages
        """
        # Split into sentences (simplified - use proper sentence tokenizer in production)
        sentences = [s.strip() + '.' for s in paper_text.split('.') if s.strip()]
        
        extracted_data = []
        
        for i, sentence in enumerate(sentences):
            # Stage 1: Binary Classification
            if not self.classify_sentence_relevance(sentence):
                continue
            
            print(f"✓ Found relevant sentence {i+1}/{len(sentences)}")
            
            # Stage 2: Build passage with context
            prev_sentence = sentences[i-1] if i > 0 else ""
            passage = self.build_passage(title, prev_sentence, sentence)
            
            # Stage 3: Detect single vs multiple values
            is_multi = self.detect_multiple_values(passage)
            
            # Stage 4: Extract parameters using function calling
            params = self.extract_parameters(passage)
            
            if not params:
                continue
            
            # Stage 5: Verify critical parameters
            verified_params = {}
            unit_map = {
                "influent_P": "mg/L",
                "effluent_P": "mg/L",
                "chemical_dose": "mg/L",
                "temperature": "°C",
                "pH": "",
                "dissolved_O2": "mg/L",
                "SRT": "days",
                "HRT": "hours"
            }
            
            for param_name, param_value in params.items():
                if param_name == "material_description":
                    verified_params[param_name] = param_value
                    continue
                
                if isinstance(param_value, (int, float)):
                    unit = unit_map.get(param_name, "")
                    is_valid = self.verify_parameter(param_name, param_value, 
                                                     passage, unit)
                    
                    if is_valid:
                        verified_params[param_name] = param_value
                        verified_params[f"{param_name}_valid"] = True
                    else:
                        verified_params[f"{param_name}_valid"] = False
            
            if verified_params:
                verified_params["sentence_index"] = i
                verified_params["is_multi_value"] = is_multi
                extracted_data.append(verified_params)
                print(f"  → Extracted and verified {len(verified_params)} parameters")
        
        return extracted_data


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example usage of the hybrid extractor
    """
    # Sample paper text (use your actual paper)
    sample_paper = """
    The wastewater treatment plant operated at a temperature of 20°C with a pH of 7.2.
    The influent phosphorus concentration was 8.5 mg/L. After treatment with a chemical
    dose of 18 mg/L alum, the effluent phosphorus was reduced to 1.4 mg/L, achieving
    a removal efficiency of 83%. The system maintained dissolved oxygen at 2.0 mg/L
    with a solids retention time of 14 days.
    """
    
    paper_title = "Phosphorus Removal in Municipal Wastewater Treatment"
    
    # Initialize extractor
    extractor = HybridPhosphorusExtractor(model="gpt-4")
    
    # Extract data
    print("Starting extraction...")
    results = extractor.extract_from_paper(sample_paper, paper_title)
    
    # Display results
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    
    for idx, data in enumerate(results, 1):
        print(f"\nData Point {idx}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # Convert to DataFrame for easy viewing
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("RESULTS AS DATAFRAME")
        print("="*60)
        print(df.to_string())
        
        # Save to CSV
        df.to_csv("phosphorus_extraction_results.csv", index=False)
        print("\n✓ Results saved to phosphorus_extraction_results.csv")


if __name__ == "__main__":
    main()