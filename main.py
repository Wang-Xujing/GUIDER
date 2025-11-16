import argparse
import os
import torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Set, Any

from utils.data_loader import MovieLensDataset
from models.llm_recommender import LLMRecommender
from utils.evaluation import RecommendationEvaluator
import matplotlib.pyplot as plt


def calculate_genre_similarity(movie_features: Dict[int, Dict[str, str]]) -> Dict[int, Dict[int, float]]:
    item_genres = {item_id: set(features.get('genres', '').split('|')) for item_id, features in movie_features.items()}
    similarity_matrix = defaultdict(dict)
    item_ids = list(item_genres.keys())
    for i in tqdm(range(len(item_ids)), desc="Calculating Similarity"):
        for j in range(i, len(item_ids)):
            id1, id2 = item_ids[i], item_ids[j]
            intersection = len(item_genres[id1].intersection(item_genres[id2]))
            union = len(item_genres[id1].union(item_genres[id2]))
            similarity = intersection / union if union > 0 else 0.0
            similarity_matrix[id1][id2] = similarity_matrix[id2][id1] = similarity
    return similarity_matrix


def calculate_hit_rate(actual: list, predicted: list, k: int) -> float:
    return 1.0 if set(actual) & set(predicted[:k]) else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description='LLM Recommender System with Uncertainty Quantification')
    parser.add_argument('--data-path', type=str, default='data/ml-10m', help='Dataset path')
    parser.add_argument('--model', type=str, default='qwen', help='LLM model name')
    parser.add_argument('--top-k', type=int, default=20, help='Evaluation K value')
    parser.add_argument('--max-test-users', type=int, default=1000, help='Maximum number of test users')
    parser.add_argument('--num-neg-samples', type=int, default=29, help='Number of negative samples')
    parser.add_argument('--du-threshold', type=float, default=0, help='DU threshold')
    parser.add_argument('--mu-threshold', type=float, default=0, help='MU threshold')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = MovieLensDataset(data_path=args.data_path)
    test_data_raw = dataset.get_test_data()
    ground_truth_dict = defaultdict(list)
    for user_id, movie_id in test_data_raw:
        ground_truth_dict[user_id].append(movie_id)

    movie_features = dataset.get_item_features()
    item_similarity_matrix = calculate_genre_similarity(movie_features)
    item_popularity = dataset.get_item_popularity()

    all_movie_indices = list(movie_features.keys())
    test_users = list(ground_truth_dict.keys())
    users_to_process = test_users[:min(len(test_users), args.max_test_users)]

    user_histories, candidate_items_list, candidate_indices_list, user_ids_processed = [], [], [], []

    print(f"Preparing data for {len(users_to_process)} users...")
    for user_id in tqdm(users_to_process, desc="Preparing user data"):
        if not ground_truth_dict[user_id]: continue
        user_history_indices = dataset.get_user_history(user_id)
        target_item = ground_truth_dict[user_id][0]
        seen_items = set(user_history_indices) | {target_item}
        unseen_items = [idx for idx in all_movie_indices if idx not in seen_items]
        num_to_sample = min(args.num_neg_samples, len(unseen_items))
        if num_to_sample <= 0: continue
        negative_samples = np.random.choice(unseen_items, num_to_sample, replace=False)
        candidate_indices = np.concatenate(([target_item], negative_samples))
        np.random.shuffle(candidate_indices)

        user_history_tuples = [(idx, movie_features.get(idx, {})) for idx in user_history_indices]
        candidate_items_tuples = [(idx, movie_features.get(idx, {})) for idx in candidate_indices]

        user_ids_processed.append(user_id)
        user_histories.append(user_history_tuples)
        candidate_items_list.append(candidate_items_tuples)
        candidate_indices_list.append(candidate_indices.tolist())

    recommender = LLMRecommender(model_name=args.model, du_threshold=args.du_threshold, mu_threshold=args.mu_threshold,
                                 load_in_8bit=args.load_in_8bit)

    llm_results = recommender.batch_recommend(
        user_histories, candidate_items_list, k=args.top_k,
        item_similarity=item_similarity_matrix, item_popularity_map=item_popularity
    )

    print(f"\n--- Calculating Evaluation Metrics (K={args.top_k}) ---")
    metrics = defaultdict(list)
    for i in range(len(user_ids_processed)):
        rec_indices, _, _ = llm_results[i]
        actual_candidate_ids = candidate_indices_list[i]
        ground_truth = ground_truth_dict[user_ids_processed[i]]
        predicted_ids = [actual_candidate_ids[idx] for idx in rec_indices]

        metrics['precision'].append(RecommendationEvaluator.precision_at_k(ground_truth, predicted_ids, k=args.top_k))
        metrics['recall'].append(RecommendationEvaluator.recall_at_k(ground_truth, predicted_ids, k=args.top_k))
        metrics['ndcg'].append(RecommendationEvaluator.ndcg_at_k(ground_truth, predicted_ids, k=args.top_k))
        metrics['hr'].append(calculate_hit_rate(ground_truth, predicted_ids, k=args.top_k))

    print(f"Model: {args.model}")
    for key, value in metrics.items():
        print(f"  - Average {key.upper()}@{args.top_k}: {np.mean(value):.4f}")

    print("\n--- Uncertainty Analysis (DU/MU) ---")
    all_du = [res[1][0] for res in llm_results if res[1] is not None and res[1].size > 0 and np.isfinite(res[1][0])]
    all_mu = [res[2][0] for res in llm_results if res[2] is not None and res[2].size > 0 and np.isfinite(res[2][0])]
    if all_du and all_mu:
        print(f"  - DU:  Mean = {np.mean(all_du):.4f}, Std Dev = {np.std(all_du):.4f}")
        print(f"  - MU: Mean = {np.mean(all_mu):.4f}, Std Dev = {np.std(all_mu):.4f}")

    print("\n--- Recommendation Samples (First 5 Users) ---")
    for i, user_id in enumerate(user_ids_processed[:5]):
        rec_indices, du_array, mu_array = llm_results[i]
        ground_truth_ids = ground_truth_dict[user_id]
        ground_truth_titles = [movie_features.get(gt_id, {}).get('title', '') for gt_id in ground_truth_ids]

        print(f"\nUser {user_id}:")
        print(f"  - DU: {du_array[0]:.4f}, MU: {mu_array[0]:.4f}")
        print(f"  - Ground Truth: {ground_truth_titles}")
        print(f"  - Top {args.top_k} Recommendation Results:")
        predicted_item_ids = [candidate_indices_list[i][idx] for idx in rec_indices]
        for j, item_id in enumerate(predicted_item_ids[:args.top_k]):
            item_title = movie_features.get(item_id, {}).get('title', 'Unknown Title')
            is_hit = item_id in ground_truth_ids
            print(f"    {j + 1}. {item_title} (ID: {item_id}) {'<- [hit]' if is_hit else ''}")


if __name__ == '__main__':
    main()