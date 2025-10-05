# Hybrid Phosphorus Extraction with Model-Aware Function Calling
# Combines ChatExtract workflow + Model-specific function routing

import os
import re
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from typing import Dict, List, Optional, Tuple, Any
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# PHOSPHORUS REMOVAL MODELS (Physical/Chemical Functions)
# ============================================================================

def langmuir_adsorption(Ce: float, qmax: float, KL: float) -> float:
    """
    Langmuir isotherm model for phosphorus adsorption

    Parameters:
    - Ce: Equilibrium concentration (mg/L)
    - qmax: Maximum adsorption capacity (mg/g)
    - KL: Langmuir constant (L/mg)

    Returns:
    - qe: Amount adsorbed at equilibrium (mg/g)
    """
    return (qmax * KL * Ce) / (1 + KL * Ce)


def freundlich_adsorption(Ce: float, KF: float, n: float) -> float:
    """
    Freundlich isotherm model for phosphorus adsorption

    Parameters:
    - Ce: Equilibrium concentration (mg/L)
    - KF: Freundlich constant ((mg/g)(L/mg)^(1/n))
    - n: Adsorption intensity (unitless)

    Returns:
    - qe: Amount adsorbed at equilibrium (mg/g)
    """
    return KF * (Ce ** (1 / n))


def first_order_removal(C0: float, k: float, t: float) -> float:
    """
    First-order kinetics for phosphorus removal

    Parameters:
    - C0: Initial concentration (mg/L)
    - k: First-order rate constant (1/time)
    - t: Time (same unit as k)

    Returns:
    - Ct: Concentration at time t (mg/L)
    """
    return C0 * np.exp(-k * t)


def chemical_precipitation_removal(P_initial: float, Me_dose: float,
                                   molar_ratio: float = 1.5) -> Tuple[float, float]:
    """
    Chemical precipitation model for phosphorus removal

    Parameters:
    - P_initial: Initial P concentration (mg/L as P)
    - Me_dose: Metal salt dose (mg/L as metal, e.g., Al3+ or Fe3+)
    - molar_ratio: Metal:P molar ratio (typically 1.5-2.0 for effective removal)

    Returns:
    - P_removed: Amount of P removed (mg/L)
    - removal_efficiency: Fraction removed (0-1)
    """
    # Convert to molar concentrations (P: 31 g/mol, Al: 27 g/mol, Fe: 56 g/mol)
    P_molar = P_initial / 31
    Me_molar = Me_dose / 27  # Assuming Al, adjust for Fe if needed

    # Calculate theoretical removal based on stoichiometry
    P_precipitated = min(P_molar, Me_molar / molar_ratio) * 31
    removal_eff = P_precipitated / P_initial

    return P_precipitated, removal_eff


def ebpr_model(influent_P: float, VFA: float, SRT: float, temp: float) -> float:
    """
    Enhanced Biological Phosphorus Removal (EBPR) simplified model

    Parameters:
    - influent_P: Influent P concentration (mg/L)
    - VFA: Volatile Fatty Acids (mg COD/L)
    - SRT: Solids Retention Time (days)
    - temp: Temperature (°C)

    Returns:
    - effluent_P: Effluent P concentration (mg/L)
    """
    # Simplified model based on VFA availability and SRT
    removal_eff = min(0.95, 0.5 + 0.3 * (VFA / 100) + 0.15 * min(SRT / 20, 1))
    temp_factor = 1 + 0.02 * (temp - 20)  # Temperature correction
    removal_eff *= temp_factor
    removal_eff = np.clip(removal_eff, 0, 0.98)
    print("EBPR MODEL function is called")
    return influent_P * (1 - removal_eff)


# ============================================================================
# MODEL-SPECIFIC FUNCTION DEFINITIONS FOR LLM
# ============================================================================

model_specific_tools = [
    # Langmuir Adsorption Model
    {
        "type": "function",
        "function": {
            "name": "extract_langmuir_parameters",
            "description": "Extract parameters for Langmuir adsorption isotherm model. Use when paper discusses adsorption equilibrium, maximum capacity, or Langmuir model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Ce": {
                        "type": "number",
                        "description": "Equilibrium phosphorus concentration in mg/L"
                    },
                    "qe": {
                        "type": "number",
                        "description": "Amount of P adsorbed at equilibrium in mg/g"
                    },
                    "qmax": {
                        "type": "number",
                        "description": "Maximum adsorption capacity in mg/g"
                    },
                    "KL": {
                        "type": "number",
                        "description": "Langmuir constant in L/mg"
                    },
                    "R2": {
                        "type": "number",
                        "description": "Coefficient of determination for model fit"
                    },
                    "adsorbent_material": {
                        "type": "string",
                        "description": "Type of adsorbent material used"
                    }
                },
                "required": ["Ce", "qe"]
            }
        }
    },

    # Freundlich Adsorption Model
    {
        "type": "function",
        "function": {
            "name": "extract_freundlich_parameters",
            "description": "Extract parameters for Freundlich adsorption isotherm model. Use when paper discusses Freundlich model, adsorption intensity, or heterogeneous surfaces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Ce": {
                        "type": "number",
                        "description": "Equilibrium phosphorus concentration in mg/L"
                    },
                    "qe": {
                        "type": "number",
                        "description": "Amount of P adsorbed at equilibrium in mg/g"
                    },
                    "KF": {
                        "type": "number",
                        "description": "Freundlich constant in (mg/g)(L/mg)^(1/n)"
                    },
                    "n": {
                        "type": "number",
                        "description": "Adsorption intensity (unitless, typically 1-10)"
                    },
                    "R2": {
                        "type": "number",
                        "description": "Coefficient of determination for model fit"
                    },
                    "adsorbent_material": {
                        "type": "string",
                        "description": "Type of adsorbent material used"
                    }
                },
                "required": ["Ce", "qe"]
            }
        }
    },

    # Kinetics Model
    {
        "type": "function",
        "function": {
            "name": "extract_kinetics_parameters",
            "description": "Extract parameters for phosphorus removal kinetics (first-order, pseudo-first-order, or pseudo-second-order). Use when paper discusses removal rate, reaction kinetics, or time-dependent removal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "C0": {
                        "type": "number",
                        "description": "Initial phosphorus concentration in mg/L"
                    },
                    "Ct": {
                        "type": "number",
                        "description": "Phosphorus concentration at time t in mg/L"
                    },
                    "t": {
                        "type": "number",
                        "description": "Time in minutes or hours"
                    },
                    "k1": {
                        "type": "number",
                        "description": "First-order or pseudo-first-order rate constant in 1/time"
                    },
                    "k2": {
                        "type": "number",
                        "description": "Pseudo-second-order rate constant in g/(mg·time)"
                    },
                    "kinetic_model_type": {
                        "type": "string",
                        "enum": ["first-order", "pseudo-first-order", "pseudo-second-order"],
                        "description": "Type of kinetic model used"
                    },
                    "R2": {
                        "type": "number",
                        "description": "Coefficient of determination for model fit"
                    }
                },
                "required": ["C0", "t"]
            }
        }
    },

    # Chemical Precipitation Model
    {
        "type": "function",
        "function": {
            "name": "extract_chemical_precipitation_parameters",
            "description": "Extract parameters for chemical precipitation-based phosphorus removal. Use when paper discusses coagulation, alum, ferric chloride, lime, or metal salt dosing.",
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
                    "coagulant_type": {
                        "type": "string",
                        "enum": ["alum", "ferric_chloride", "ferric_sulfate", "lime", "PAC", "other"],
                        "description": "Type of chemical coagulant used"
                    },
                    "coagulant_dose": {
                        "type": "number",
                        "description": "Coagulant dose in mg/L"
                    },
                    "pH": {
                        "type": "number",
                        "description": "pH during treatment"
                    },
                    "molar_ratio": {
                        "type": "number",
                        "description": "Metal:Phosphorus molar ratio"
                    },
                    "removal_efficiency": {
                        "type": "number",
                        "description": "Phosphorus removal efficiency as decimal (0-1)"
                    }
                },
                "required": ["influent_P", "coagulant_type", "coagulant_dose"]
            }
        }
    },

    # EBPR Model
    {
        "type": "function",
        "function": {
            "name": "extract_ebpr_parameters",
            "description": "Extract parameters for Enhanced Biological Phosphorus Removal (EBPR). Use when paper discusses biological treatment, PAOs (phosphorus accumulating organisms), anaerobic/aerobic zones, or activated sludge.",
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
                    "VFA": {
                        "type": "number",
                        "description": "Volatile Fatty Acids concentration in mg COD/L"
                    },
                    "SRT": {
                        "type": "number",
                        "description": "Solids Retention Time in days"
                    },
                    "HRT": {
                        "type": "number",
                        "description": "Hydraulic Retention Time in hours"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Operating temperature in °C"
                    },
                    "DO_aerobic": {
                        "type": "number",
                        "description": "Dissolved oxygen in aerobic zone in mg/L"
                    },
                    "anaerobic_time": {
                        "type": "number",
                        "description": "Anaerobic zone retention time in hours"
                    },
                    "PAO_fraction": {
                        "type": "number",
                        "description": "Fraction of PAOs in biomass (0-1)"
                    }
                },
                "required": ["influent_P", "SRT"]
            }
        }
    },

    # General Wastewater Treatment
    {
        "type": "function",
        "function": {
            "name": "extract_general_treatment_parameters",
            "description": "Extract general wastewater treatment parameters when no specific model is mentioned. Use as fallback for general phosphorus removal data.",
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
                    "treatment_type": {
                        "type": "string",
                        "description": "Type of treatment process"
                    },
                    "HRT": {
                        "type": "number",
                        "description": "Hydraulic Retention Time in hours"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Operating temperature in °C"
                    },
                    "pH": {
                        "type": "number",
                        "description": "pH level"
                    }
                },
                "required": ["influent_P"]
            }
        }
    }
]


# ============================================================================
# MODEL-AWARE EXTRACTOR CLASS
# ============================================================================

class ModelAwarePhosphorusExtractor:
    """
    Intelligent extractor that identifies the relevant phosphorus model
    and extracts appropriate parameters using model-specific functions
    """

    def __init__(self, model: str = "gpt-4", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.conversation_history = []

    def _add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def _call_llm(self, use_tools: bool = False, max_retries: int = 3) -> Any:
        """Call LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                if use_tools:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        tools=model_specific_tools,
                        tool_choice="auto",  # Let LLM choose appropriate function
                        temperature=self.temperature
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        temperature=self.temperature,
                        max_tokens=100
                    )
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = [{"role": "system", "content": ""}]

    # STAGE 1: Model Identification
    def identify_phosphorus_model(self, passage: str) -> str:
        """
        Identify which phosphorus removal/adsorption model is discussed
        """
        self.reset_conversation()

        prompt = """Identify the primary phosphorus removal or adsorption model discussed in the following text.
        Choose ONE from: Langmuir, Freundlich, Kinetics, Chemical Precipitation, EBPR, General Treatment, or None.
        Answer with only the model name.

        Text: """ + passage

        self._add_message("user", prompt)
        response = self._call_llm()
        model_type = response.choices[0].message.content.strip()
        self._add_message("assistant", model_type)

        print(f"  → Identified model: {model_type}")
        return model_type

    # STAGE 2: Binary Classification (relevant for phosphorus?)
    def classify_sentence_relevance(self, sentence: str) -> bool:
        """Check if sentence contains phosphorus-related data"""
        prompt = (
                'Answer "Yes" or "No" only. Does the following text contain '
                'phosphorus removal data, adsorption data, or treatment parameters?\n\n'
                + sentence
        )

        self._add_message("user", prompt)
        response = self._call_llm()
        answer = response.choices[0].message.content.strip().lower()
        self._add_message("assistant", answer)

        return "yes" in answer

    # STAGE 3: Model-Aware Extraction with Function Calling
    def extract_model_parameters(self, passage: str, model_type: str) -> Optional[Dict]:
        """
        Extract parameters using model-specific function calling
        """
        prompt = f"""Extract parameters for {model_type} phosphorus model from this text.
        Be precise with units. Only extract values explicitly stated in the text.

        Text: {passage}"""

        self._add_message("user", prompt)
        response = self._call_llm(use_tools=True)

        message = response.choices[0].message

        # Check if function was called
        if message.tool_calls:
            function_name = message.tool_calls[0].function.name
            function_args = json.loads(message.tool_calls[0].function.arguments)

            return {
                "model_type": model_type,
                "function_used": function_name,
                "parameters": function_args
            }

        return None

    # STAGE 4: Verification
    def verify_parameter(self, param_name: str, param_value: Any,
                         passage: str, model_type: str) -> bool:
        """Verify extracted parameter with follow-up question"""

        prompt = f"""There is a possibility the extracted data is incorrect.
        Answer "Yes" or "No" only. Be very strict.

        For the {model_type} model, is "{param_value}" the correct value of {param_name} 
        in the following text?

        Text: {passage}"""

        self._add_message("user", prompt)
        response = self._call_llm()
        answer = response.choices[0].message.content.strip().lower()
        self._add_message("assistant", answer)

        return "yes" in answer

    # Add this function before the class definition
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving decimal numbers.
        """
        placeholder = "<!DECIMAL!>"
        text_protected = re.sub(r'(\d+)\.(\d+)', r'\1' + placeholder + r'\2', text)
        sentences = re.split(r'\.\s+(?=[A-Z])|\.(?=\s*$)', text_protected)
        sentences = [s.replace(placeholder, '.').strip() for s in sentences if s.strip()]
        sentences = [s if s.endswith('.') else s + '.' for s in sentences]
        return sentences


    # COMPLETE WORKFLOW
    def extract_from_paper(self, paper_text: str,
                           verify: bool = True) -> List[Dict]:
        """
        Complete model-aware extraction workflow
        """
        sentences = self.split_into_sentences(paper_text)
        extracted_data = []

        for i, sentence in enumerate(sentences):
            self.reset_conversation()

            # Stage 1: Check relevance

            if not self.classify_sentence_relevance(sentence):
                print("Skipping sentence - " + sentence)
                continue

            print(f"\n✓ Found relevant sentence {i + 1}/{len(sentences)} - " + sentence)

            # Build passage with context
            prev_sentence = sentences[i - 1] if i > 0 else ""
            passage = f"Previous: {prev_sentence}\n\nTarget: {sentence}"

            # Stage 2: Identify model type
            model_type = self.identify_phosphorus_model(passage)

            if "none" in model_type.lower():
                continue

            # Stage 3: Extract parameters using appropriate function
            extraction_result = self.extract_model_parameters(passage, model_type)

            if not extraction_result:
                continue

            params = extraction_result["parameters"]
            print(f"  → Extracted {len(params)} parameters using {extraction_result['function_used']}")

            # Stage 4: Verify critical parameters (optional)
            if verify:
                verified_params = {}
                for param_name, param_value in params.items():
                    if isinstance(param_value, (int, float)):
                        is_valid = self.verify_parameter(param_name, param_value,
                                                         passage, model_type)
                        verified_params[f"{param_name}_verified"] = is_valid
                        if is_valid:
                            verified_params[param_name] = param_value
                    else:
                        verified_params[param_name] = param_value

                params = verified_params

            # Add metadata
            result = {
                "sentence_index": i,
                "model_type": model_type,
                "function_used": extraction_result["function_used"],
                **params
            }

            extracted_data.append(result)

        return extracted_data


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_langmuir_extraction():
    """Example: Extract Langmuir adsorption parameters"""

    paper = """
    Phosphorus adsorption onto modified biochar was investigated. The Langmuir 
    isotherm model provided excellent fit to the experimental data with R² = 0.98.
    The maximum adsorption capacity (qmax) was determined to be 45.2 mg/g, and the
    Langmuir constant (KL) was 0.15 L/mg. At equilibrium concentration of 10 mg/L,
    the amount adsorbed was 8.5 mg/g.
    """

    extractor = ModelAwarePhosphorusExtractor(model="gpt-4")
    results = extractor.extract_from_paper(paper)

    return results


def example_chemical_precipitation():
    """Example: Extract chemical precipitation parameters"""

    paper = """
    Chemical precipitation using alum was tested for phosphorus removal. The influent
    phosphorus concentration was 12.5 mg/L. With an alum dose of 25 mg/L at pH 6.8,
    the effluent phosphorus was reduced to 0.8 mg/L, achieving 93.6% removal efficiency.
    The optimal metal to phosphorus molar ratio was found to be 1.8:1.
    """

    extractor = ModelAwarePhosphorusExtractor(model="gpt-4")
    results = extractor.extract_from_paper(paper)

    return results


def example_ebpr():
    """Example: Extract EBPR parameters"""

    paper = """
    The EBPR system operated at 22°C with an SRT of 15 days. Influent phosphorus 
    was 8.2 mg/L with VFA concentration of 120 mg COD/L. The anaerobic zone retention
    time was 2 hours, followed by aerobic conditions with DO maintained at 2.5 mg/L.
    Effluent phosphorus averaged 0.5 mg/L, demonstrating effective biological removal.
    """

    extractor = ModelAwarePhosphorusExtractor(model="gpt-4")
    results = extractor.extract_from_paper(paper)

    return results


def main():
    """Run all examples and display results"""

    print("=" * 70)
    print("MODEL-AWARE PHOSPHORUS PARAMETER EXTRACTION")
    print("=" * 70)

    examples = [
        ("Langmuir Adsorption", example_langmuir_extraction),
        # ("Chemical Precipitation", example_chemical_precipitation),
        # ("EBPR System", example_ebpr)
    ]

    all_results = []

    for name, example_func in examples:
        print(f"\n\n{'=' * 70}")
        print(f"EXAMPLE: {name}")
        print('=' * 70)

        try:
            results = example_func()
            all_results.extend(results)

            for idx, data in enumerate(results, 1):
                print(f"\nExtracted Data {idx}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"Error in {name}: {e}")

    # Save all results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("model_aware_extraction_results.csv", index=False)
        print(f"\n\n✓ All results saved to model_aware_extraction_results.csv")
        print(f"Total extractions: {len(all_results)}")


if __name__ == "__main__":
    main()