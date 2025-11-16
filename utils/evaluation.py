import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Any


class RecommendationEvaluator:
    """Recommendation System Evaluation Tools"""

    @staticmethod
    def precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
        """
        Calculate Precision@K

        Args:
            actual: List of items the user actually interacted with
            predicted: List of items predicted by the recommendation algorithm
            k: Length of the recommendation list

        Returns:
            Precision@K
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        if not actual or not predicted:
            return 0.0

        hits = len(set(actual) & set(predicted))
        return hits / min(k, len(predicted))

    @staticmethod
    def recall_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
        """
        Calculate Recall@K

        Args:
            actual: List of items the user actually interacted with
            predicted: List of items predicted by the recommendation algorithm
            k: Length of the recommendation list

        Returns:
            Recall@K
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        if not actual or not predicted:
            return 0.0

        hits = len(set(actual) & set(predicted))
        return hits / len(actual) if len(actual) > 0 else 0.0

    @staticmethod
    def ndcg_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
        """
        Calculate NDCG@K (Standard Method)

        Args:
            actual: List of items the user actually interacted with
            predicted: List of items predicted by the recommendation algorithm
            k: Length of the recommendation list

        Returns:
            NDCG@K
        """
        if not actual or not predicted:
            return 0.0

        # Only consider the top k predicted items
        if len(predicted) > k:
            predicted = predicted[:k]

        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(predicted):
            if item in actual:  # Binary relevance: 1 if in actual, 0 otherwise
                # i+1 is the rank of the recommended item (starting from 1), then take the logarithm
                dcg += 1.0 / np.log2(i + 2)  # i starts from 0, so use i+2

        # Calculate IDCG (Ideal DCG)
        # The ideal case is that all relevant items are at the top of the recommendation list
        # min(len(actual), k) means at most k items or the actual number of items, whichever is smaller, can be recommended
        idcg = 0.0
        for i in range(min(len(actual), k)):
            idcg += 1.0 / np.log2(i + 2)

        # Avoid division by zero
        if idcg == 0.0:
            return 0.0

        # NDCG = DCG / IDCG
        return dcg / idcg

    @staticmethod
    def map_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
        """
        Calculate MAP@K (Mean Average Precision)

        Args:
            actual: List of items the user actually interacted with
            predicted: List of items predicted by the recommendation algorithm
            k: Length of the recommendation list

        Returns:
            MAP@K
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        if not actual or not predicted:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for i, item in enumerate(predicted):
            if item in actual:
                hits += 1
                sum_precisions += hits / (i + 1)

        return sum_precisions / min(len(actual), k) if min(len(actual), k) > 0 else 0.0

    @staticmethod
    def mrr(actual: List[int], predicted: List[int]) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank)

        Args:
            actual: List of items the user actually interacted with
            predicted: List of items predicted by the recommendation algorithm

        Returns:
            MRR
        """
        if not actual or not predicted:
            return 0.0

        for i, item in enumerate(predicted):
            if item in actual:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def diversity(predicted: List[int], item_features: Dict[int, Any]) -> float:
        """
        Calculate the diversity of the recommendation list

        Args:
            predicted: List of items predicted by the recommendation algorithm
            item_features: Dictionary of item features

        Returns:
            Diversity score (between 0-1, higher means more diverse)
        """
        if not predicted or len(predicted) < 2:
            return 0.0

        # For movie recommendations, a simple calculation method is used here: calculate the proportion of different types of items
        genres_set = set()

        for item_id in predicted:
            if item_id in item_features and 'genres' in item_features[item_id]:
                genres = item_features[item_id]['genres']
                if isinstance(genres, str):
                    genres = genres.split('|')
                for genre in genres:
                    genres_set.add(genre)

        # If there is no genre information, return 0
        if not genres_set:
            return 0.0

        # Calculate diversity
        return len(genres_set) / (len(predicted) * 3)  # 3 is an assumed average number of genres per movie

    @staticmethod
    def coverage(all_predictions: List[List[int]], catalog_size: int) -> float:
        """
        Calculate the coverage of the recommendation system

        Args:
            all_predictions: A list of recommendation lists for all users
            catalog_size: The total size of the item catalog

        Returns:
            Coverage (between 0-1)
        """
        if not all_predictions or catalog_size == 0:
            return 0.0

        # Calculate the set of all recommended items
        recommended_items = set()
        for user_preds in all_predictions:
            recommended_items.update(user_preds)

        # Calculate coverage
        return len(recommended_items) / catalog_size

    @staticmethod
    def evaluate_recommendations(test_data: Dict[int, List[int]],
                                 predicted_data: Dict[int, List[int]],
                                 item_features: Dict[int, Any] = None,
                                 k_values: List[int] = [5, 10, 20],
                                 catalog_size: int = None) -> Dict[str, float]:
        """
        Comprehensively evaluate recommendation results

        Args:
            test_data: Test data, a mapping from user ID to a list of items
            predicted_data: Prediction data, a mapping from user ID to a list of recommended items
            item_features: Item features, used for calculating diversity
            k_values: A list of K values to evaluate
            catalog_size: The total size of the item catalog, used for calculating coverage

        Returns:
            A dictionary containing various metrics
        """
        results = {}

        # Ensure there are common users
        common_users = set(test_data.keys()) & set(predicted_data.keys())
        if not common_users:
            return {"error": "No common users for evaluation"}

        # Calculate various metrics
        all_predictions = []

        for k in k_values:
            precision_sum = 0.0
            recall_sum = 0.0
            ndcg_sum = 0.0
            map_sum = 0.0

            for user_id in common_users:
                actual = test_data[user_id]
                predicted = predicted_data[user_id]

                # Limit the recommendation list length to k
                predicted_at_k = predicted[:k] if len(predicted) > k else predicted

                precision_sum += RecommendationEvaluator.precision_at_k(actual, predicted, k)
                recall_sum += RecommendationEvaluator.recall_at_k(actual, predicted, k)
                ndcg_sum += RecommendationEvaluator.ndcg_at_k(actual, predicted, k)
                map_sum += RecommendationEvaluator.map_at_k(actual, predicted, k)

                # Collect all predictions for coverage calculation
                if k == k_values[-1]:  # Only record at the largest k value
                    all_predictions.append(predicted_at_k)

            n_users = len(common_users)
            results[f'precision@{k}'] = precision_sum / n_users
            results[f'recall@{k}'] = recall_sum / n_users
            results[f'ndcg@{k}'] = ndcg_sum / n_users
            results[f'map@{k}'] = map_sum / n_users

        # Calculate MRR
        mrr_sum = 0.0
        for user_id in common_users:
            actual = test_data[user_id]
            predicted = predicted_data[user_id]
            mrr_sum += RecommendationEvaluator.mrr(actual, predicted)
        results['mrr'] = mrr_sum / len(common_users)

        # Calculate diversity (if item features are provided)
        if item_features is not None:
            diversity_sum = 0.0
            for user_id in common_users:
                predicted = predicted_data[user_id]
                diversity_sum += RecommendationEvaluator.diversity(predicted, item_features)
            results['diversity'] = diversity_sum / len(common_users)

        # Calculate coverage (if catalog size is provided)
        if catalog_size is not None and catalog_size > 0:
            results['coverage'] = RecommendationEvaluator.coverage(all_predictions, catalog_size)

        return results