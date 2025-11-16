import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.special import digamma
import os
from tqdm import tqdm
import time


class LLMRecommender:
    """
    A recommendation model based on large model logits, using a conversational prompt
    and implementing an advanced four-quadrant uncertainty strategy.
    """
    MODEL_PATHS = {
        "llama3": "../models/Llama-3",
        "gemma": "../models/gemma",
        "mistral": "../models/mistralai/Mistral",
        "qwen": "../models/qwen2.5"
    }

    def __init__(
            self,
            model_name: str = "llama3",
            device: Optional[torch.device] = None,
            load_in_8bit: bool = False,
            temperature: float = 1.0,
            du_threshold: float = 1.5,
            mu_threshold: float = 0.05,
            diversity_weight: float = 0.3,
            exploration_ratio: float = 0.2,
            top_k_uncertainty: int = 10
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        model_path = self.MODEL_PATHS.get(model_name, model_name)
        print(f"Loading model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()

        self.temperature = temperature
        self.du_threshold = du_threshold
        self.mu_threshold = mu_threshold
        self.diversity_weight = diversity_weight
        self.exploration_ratio = exploration_ratio
        self.top_k_uncertainty = top_k_uncertainty
        print(f"Model loaded successfully, device: {self.device}")

    def prepare_prompt(self, user_history: List[Tuple[int, Dict[str, str]]],
                       candidate_items: List[Tuple[int, Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Constructs a conversational prompt, displaying movie genres.
        """
        system_prompt = "You are the user. Based on the movies you have recently watched, please select the movies from the candidates that you are most likely to watch next, sorted by interest. You can judge the match based on dimensions like genre or theme."

        def format_item(item_tuple):
            mid, features = item_tuple
            title = features.get('title', 'Unknown Title')
            genres = features.get('genres', 'N/A')
            return f"{title} (ID: {mid}, Genres: {genres})"

        history_str = "\n".join([f"- {format_item(item)}" for item in reversed(user_history[-10:])])
        candidate_str = "\n".join([
            f"{i + 1}. {format_item(item)}" for i, item in enumerate(candidate_items)
        ])

        user_prompt = f"Here are the movies I've recently watched (the latest are at the bottom):\n{history_str}\n\nThe recommendation system has prepared the following candidate movies for me:\n{candidate_str}\n\nBased on my interests, please sort the candidate movies above by their number according to my interest level.\nOutput only the number of the movie you recommend the most, for example: 5\nDo not output any other explanations or text."

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _get_candidate_token_ids(self, candidate_items: List[Tuple[int, Dict[str, str]]]) -> List[int]:
        candidate_token_ids = []
        for i in range(1, len(candidate_items) + 1):
            token_id = self.tokenizer.encode(" " + str(i), add_special_tokens=False)
            if not token_id:
                continue
            candidate_token_ids.append(token_id[0])
        return candidate_token_ids

    def _calculate_evidence(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits) + 1.0

    def compute_data_uncertainty(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_0 = torch.sum(alpha, dim=-1, keepdim=True)
        alpha_np = alpha.cpu().numpy()
        alpha_0_np = alpha_0.cpu().numpy()
        term1 = digamma(alpha_np + 1)
        term2 = digamma(alpha_0_np + 1)
        du_np = -np.sum((alpha_np / alpha_0_np) * (term1 - term2), axis=-1)
        return torch.tensor(du_np, device=alpha.device, dtype=alpha.dtype)

    def compute_model_uncertainty(self, alpha: torch.Tensor) -> torch.Tensor:
        K = alpha.shape[-1]
        sum_alpha = torch.sum(alpha, dim=-1)
        mu = K / sum_alpha
        return mu

    def _apply_hybrid_exploration(self, scores: torch.Tensor, k: int, exploitation_ratio: float = 0.5) -> List[int]:
        num_items = len(scores)

        # 1. Exploitation part: deterministically select Top-N
        n_exploit = int(k * exploitation_ratio)

        # Get the original ranking of all items
        sorted_indices = torch.argsort(scores, descending=True)

        exploit_indices = sorted_indices[:n_exploit].tolist()

        # 2. Exploration part: randomly select from the suboptimal pool
        n_explore = k - n_exploit
        if n_explore > 0:
            # Define an exploration pool, e.g., from n_exploit+1 to 3*k
            pool_start = n_exploit
            pool_end = min(num_items, pool_start + 3 * k)

            exploration_pool = sorted_indices[pool_start:pool_end].tolist()

            # Randomly shuffle the exploration pool
            np.random.shuffle(exploration_pool)

            # Select the required number from the shuffled pool
            explore_indices = exploration_pool[:n_explore]
        else:
            explore_indices = []

        return exploit_indices + explore_indices

    def _blend_with_popularity(self, scores: torch.Tensor, popularity_scores: torch.Tensor,
                               blend_ratio: float = 0.7) -> torch.Tensor:
        """Blend model scores with popularity scores as a safety net strategy"""
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min()) if (
                                                                                             scores.max() - scores.min()) > 0 else scores
        return (1 - blend_ratio) * scores_norm + blend_ratio * popularity_scores

    def _apply_mmr_reranking(self, scores: torch.Tensor, item_ids: List[int],
                             item_similarity_matrix: Dict[int, Dict[int, float]], k: int,
                             diversity_lambda: float = 0.5) -> List[int]:
        """Use the MMR algorithm for diversified reranking, returning the indices of the original list."""
        candidates_scores = scores.clone().tolist()
        num_items = len(item_ids)
        selected_indices, remaining_indices = [], list(range(num_items))

        while len(selected_indices) < min(k, num_items):
            best_score, best_idx = -float('inf'), -1
            for idx in remaining_indices:
                original_score = candidates_scores[idx]
                max_sim = 0.0
                if selected_indices:
                    current_item_id = item_ids[idx]
                    for sel_idx in selected_indices:
                        selected_item_id = item_ids[sel_idx]
                        sim = item_similarity_matrix.get(current_item_id, {}).get(selected_item_id, 0.0)
                        if sim > max_sim: max_sim = sim
                mmr_score = diversity_lambda * original_score - (1 - diversity_lambda) * max_sim
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        return selected_indices

    def _get_recommendation_indices(
            self, scores: torch.Tensor, du: float, mu: float, k: int,
            candidate_item_ids: List[int], item_similarity: Dict[int, Dict[int, float]], item_popularity: torch.Tensor
    ) -> List[int]:
        if du <= self.du_threshold and mu <= self.mu_threshold:
            return torch.topk(scores, k=min(k, len(scores))).indices.tolist()
        elif du <= self.du_threshold and mu > self.mu_threshold:
            print("Strategy: Quadrant II (Low DU, High MU) -> Hybrid Exploration")
            return self._apply_hybrid_exploration(scores, k, exploitation_ratio=0.3)

        elif du > self.du_threshold and mu <= self.mu_threshold:
            return self._apply_mmr_reranking(scores, candidate_item_ids, item_similarity, k, diversity_lambda=0.7)
        else:  # du > self.du_threshold and mu > self.mu_threshold
            final_scores = self._blend_with_popularity(scores, item_popularity, blend_ratio=0.6)
            return torch.topk(final_scores, k=min(k, len(final_scores))).indices.tolist()

    def recommend(
            self, user_history: List[Tuple[int, Dict[str, str]]], candidate_items: List[Tuple[int, Dict[str, str]]],
            k: int = 5,
            item_similarity: Dict[int, Dict[int, float]] = None, item_popularity_map: Dict[int, float] = None
    ) -> Tuple[List[int], np.ndarray, np.ndarray, torch.Tensor]:
        messages = self.prepare_prompt(user_history, candidate_items)
        prompt_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to(self.device)
        candidate_token_ids = self._get_candidate_token_ids(candidate_items)
        if not candidate_token_ids:
            return [], np.array([0.0]), np.array([0.0]), torch.tensor([])
        candidate_token_ids_tensor = torch.tensor(candidate_token_ids, device=self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]
            candidate_logits = last_token_logits[:, candidate_token_ids_tensor]
        alpha = self._calculate_evidence(candidate_logits)
        du = self.compute_data_uncertainty(alpha)
        mu = self.compute_model_uncertainty(alpha)
        scores_tensor = candidate_logits[0].detach().cpu()
        du_val, mu_val = du[0].cpu().numpy(), mu[0].cpu().numpy()

        candidate_item_ids = [item[0] for item in candidate_items]
        popularity_scores_list = [item_popularity_map.get(cid, 0.0) for cid in candidate_item_ids]
        popularity_tensor = torch.tensor(popularity_scores_list, dtype=torch.float32)
        if popularity_tensor.max() > 0:
            popularity_tensor /= popularity_tensor.max()

        strategy_indices = self._get_recommendation_indices(
            scores_tensor, du_val, mu_val, k,
            candidate_item_ids, item_similarity, popularity_tensor
        )
        return strategy_indices, np.array([du_val]), np.array([mu_val]), scores_tensor

    def batch_recommend(
            self, user_histories: List[List[Tuple[int, Dict[str, str]]]],
            candidate_items_list: List[List[Tuple[int, Dict[str, str]]]], k: int = 5,
            batch_size: int = 1, item_similarity: Dict[int, Dict[int, float]] = None,
            item_popularity_map: Dict[int, float] = None
    ) -> List[Tuple[List[int], np.ndarray, np.ndarray, torch.Tensor, float]]:  # Added float to return the time taken
        results = []
        for i in tqdm(range(len(user_histories)), desc="Batch Recommend"):
            hist = user_histories[i]
            cand = candidate_items_list[i]

            # --- Start timer ---
            start_time = time.time()

            try:
                rec, du, mu, scores = self.recommend(
                    hist, cand, k=k,
                    item_similarity=item_similarity,
                    item_popularity_map=item_popularity_map
                )
            except Exception as e:
                print(f"Error processing user index {i}: {e}")
                rec, du, mu, scores = [], np.array([0.0]), np.array([0.0]), torch.tensor([])

            # --- End timer ---
            end_time = time.time()
            duration = end_time - start_time

            # Store the duration and result together
            #            results.append((rec, du, mu, scores, duration))
            results.append((rec, du, mu, scores))

        return results