import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Union, Optional
import os


class LLMGenerator:
    MODEL_PATHS = {
        "llama3": "../models/Llama-3",
        "gemma": "../models/gemma",
        "mistral": "../models/mistralai/Mistral"
    }

    def __init__(
            self,
            model_name: str = "llama3",
            device: Optional[torch.device] = None,
            load_in_8bit: bool = False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_name = model_name
        if model_name not in self.MODEL_PATHS:
            raise ValueError(f"Unsupported model: {model_name}, supported models are: {list(self.MODEL_PATHS.keys())}")

        model_path = self.MODEL_PATHS[model_name]
        print(os.getcwd())
        print(f"Loading model: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        if load_in_8bit and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)

        print(f"Model loaded successfully, device: {self.device}")

    def generate(
            self,
            prompt: str,
            max_length: int = 128,
            temperature: float = 0.7,
            top_p: float = 0.9,
            num_return_sequences: int = 1,
            do_sample: bool = True,
    ) -> List[str]:
        # Encode input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        generated_texts = []
        for output in outputs:
            # Remove the input part, keeping only the generated part
            input_length = inputs.input_ids.shape[1]
            generated_output = output[input_length:]
            text = self.tokenizer.decode(generated_output, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def generate_explanations(
            self,
            user_ids: List[int],
            item_ids: List[int],
            user_features: Dict[int, Dict],
            item_features: Dict[int, Dict],
            au_values: List[float],
            eu_values: List[float],
    ) -> List[str]:
        explanations = []

        for i, (user_id, item_id) in enumerate(zip(user_ids, item_ids)):
            user_info = user_features.get(user_id, {})
            item_info = item_features.get(item_id, {})
            au = au_values[i]
            eu = eu_values[i]

            prompt = self._create_explanation_prompt(user_info, item_info, au, eu)
            explanation = self.generate(prompt, max_length=150)[0]
            explanations.append(explanation)

        return explanations

    def _create_explanation_prompt(
            self,
            user_info: Dict,
            item_info: Dict,
            au: float,
            eu: float
    ) -> str:
        uncertainty_type = self._get_uncertainty_type(au, eu)

        if "movie" in item_info:
            prompt = f"""As the explanation generation module of the recommendation system, please generate a natural and informative explanation for the following recommendation:

User Info: {user_info}
Movie Info: {item_info}
Uncertainty Type: {uncertainty_type}

The explanation should state why this movie is recommended to this user and adjust the tone and content according to the uncertainty type.
Explanation:"""
        else:
            prompt = f"""As the explanation generation module of the recommendation system, please generate a natural and informative explanation for the following recommendation:

User Info: {user_info}
Item Info: {item_info}
Uncertainty Type: {uncertainty_type}

The explanation should state why this item is recommended to this user and adjust the tone and content according to the uncertainty type.
Explanation:"""

        return prompt

    def _get_uncertainty_type(self, au: float, eu: float) -> str:
        """Get the uncertainty type"""
        au_threshold = 0.5
        eu_threshold = 0.5

        if au > au_threshold and eu > eu_threshold:
            return "High Aleatoric Uncertainty, High Epistemic Uncertainty (Cold Start Scenario)"
        elif au < au_threshold and eu < eu_threshold:
            return "Low Aleatoric Uncertainty, Low Epistemic Uncertainty (High Confidence Recommendation)"
        elif au < au_threshold and eu > eu_threshold:
            return "Low Aleatoric Uncertainty, High Epistemic Uncertainty (Exploration Needed)"
        else:
            return "High Aleatoric Uncertainty, Low Epistemic Uncertainty (Diversity Needed)"