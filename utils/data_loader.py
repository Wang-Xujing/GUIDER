import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import json
from collections import defaultdict


class MovieLens1Dataset(Dataset):

    def __init__(self, data_path, test_ratio=0.2, min_rating=4.0):
        self.data_path = data_path
        self.min_rating = min_rating
        ratings_file = os.path.join(data_path, 'ratings.dat')
        movies_file = os.path.join(data_path, 'movies.dat')
        users_file = os.path.join(data_path, 'users.dat')

        print(f"Loading ratings data from {ratings_file}...")
        ratings = pd.read_csv(
            ratings_file, sep='::', header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python', encoding='latin-1'
        )
        self.movies = pd.read_csv(
            movies_file, sep='::', header=None,
            names=['movie_id', 'title', 'genres'],
            engine='python', encoding='latin-1'
        )
        self.users = pd.read_csv(
            users_file, sep='::', header=None,
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            engine='python', encoding='latin-1'
        )

        unique_user_ids = ratings['user_id'].unique()
        unique_movie_ids = ratings['movie_id'].unique()
        self.user_id_map = {original: mapped for mapped, original in enumerate(unique_user_ids)}
        self.movie_id_map = {original: mapped for mapped, original in enumerate(unique_movie_ids)}
        self.num_users = len(self.user_id_map)
        self.num_items = len(self.movie_id_map)

        self.ratings_df = ratings.copy()
        self.ratings_df['user_id_mapped'] = self.ratings_df['user_id'].map(self.user_id_map)
        self.ratings_df['movie_id_mapped'] = self.ratings_df['movie_id'].map(self.movie_id_map)
        self.ratings_df.dropna(subset=['user_id_mapped', 'movie_id_mapped'], inplace=True)
        self.ratings_df['user_id_mapped'] = self.ratings_df['user_id_mapped'].astype(int)
        self.ratings_df['movie_id_mapped'] = self.ratings_df['movie_id_mapped'].astype(int)

        positive_ratings = self.ratings_df[self.ratings_df['rating'] >= self.min_rating]
        self.user_item_matrix = defaultdict(list)
        for user_id_mapped, group in positive_ratings.groupby('user_id_mapped'):
            self.user_item_matrix[user_id_mapped] = group['movie_id_mapped'].tolist()

        self.train_data = []
        self.test_data = []
        for user_id, item_ids in self.user_item_matrix.items():
            if len(item_ids) < 2: continue
            np.random.shuffle(item_ids)
            n_test = 1
            test_items = item_ids[:n_test]
            train_items = item_ids[n_test:]
            for item_id in train_items:
                self.train_data.append([user_id, item_id])
            for item_id in test_items:
                self.test_data.append([user_id, item_id])

        self.train_data = np.array(self.train_data)
        self.test_data = np.array(self.test_data)
        print("Dataset loading and processing complete.")

    def get_test_data(self):
        return self.test_data

    def get_user_history(self, user_id_mapped):
        all_hist = self.user_item_matrix.get(user_id_mapped, [])
        test_items_for_user = set(self.test_data[self.test_data[:, 0] == user_id_mapped][:, 1])
        train_hist = [item for item in all_hist if item not in test_items_for_user]
        return train_hist

    def get_item_features(self):
        features = {}
        for _, row in self.movies.iterrows():
            original_id = row['movie_id']
            if original_id in self.movie_id_map:
                mapped_id = self.movie_id_map[original_id]
                features[mapped_id] = {'title': row['title'], 'genres': row['genres']}
        return features

    def get_item_popularity(self):
        """Calculate the popularity of each item based on the number of ratings"""
        if 'movie_id_mapped' not in self.ratings_df.columns:
            return {}
        popularity = self.ratings_df['movie_id_mapped'].value_counts().to_dict()
        return popularity

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]


class MovieLensDataset(Dataset):
    def __init__(self, data_path, test_ratio=0.2, min_rating=4.0):

        self.data_path = data_path
        self.min_rating = min_rating

        # --- File Paths ---
        ratings_file = os.path.join(data_path, 'ratings.dat')
        movies_file = os.path.join(data_path, 'movies.dat')

        # --- Check if files exist ---
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"Ratings data file not found: {ratings_file}")
        if not os.path.exists(movies_file):
            raise FileNotFoundError(f"Movies data file not found: {movies_file}")

        # --- Load Ratings Data ---
        print(f"Loading ratings data from {ratings_file}...")
        ratings = pd.read_csv(
            ratings_file,
            sep='::',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python',
            encoding='utf-8'

        )
        print(f"Loaded {len(ratings)} ratings.")

        # --- Load Movies Data ---
        self.movies = pd.read_csv(
            movies_file,
            sep='::',
            header=None,
            names=['movie_id', 'title', 'genres'],
            engine='python',
            encoding='utf-8'
        )

        unique_user_ids = ratings['user_id'].unique()
        unique_movie_ids = ratings['movie_id'].unique()

        self.user_id_map = {original: mapped for mapped, original in enumerate(unique_user_ids)}
        self.movie_id_map = {original: mapped for mapped, original in enumerate(unique_movie_ids)}
        self.user_id_map_reverse = {mapped: original for original, mapped in self.user_id_map.items()}
        self.movie_id_map_reverse = {mapped: original for original, mapped in self.movie_id_map.items()}

        self.num_users = len(self.user_id_map)
        self.num_items = len(self.movie_id_map)
        print(f"Mapped users: {self.num_users}, Mapped items: {self.num_items}")

        # --- Store and Map the Ratings DataFrame ---
        self.ratings_df = ratings.copy()
        self.ratings_df['user_id_mapped'] = self.ratings_df['user_id'].map(self.user_id_map)
        self.ratings_df['movie_id_mapped'] = self.ratings_df['movie_id'].map(self.movie_id_map)
        self.ratings_df.dropna(subset=['user_id_mapped', 'movie_id_mapped'], inplace=True)
        self.ratings_df['user_id_mapped'] = self.ratings_df['user_id_mapped'].astype(int)
        self.ratings_df['movie_id_mapped'] = self.ratings_df['movie_id_mapped'].astype(int)

        # --- Process Interactions for Train/Test Split ---
        positive_ratings = self.ratings_df[self.ratings_df['rating'] >= self.min_rating]
        self.user_item_matrix = defaultdict(list)
        for user_id_mapped, group in positive_ratings.groupby('user_id_mapped'):
            self.user_item_matrix[user_id_mapped] = group['movie_id_mapped'].tolist()

        self.train_data = []
        self.test_data = []
        for user_id, item_ids in self.user_item_matrix.items():
            if len(item_ids) < 2: continue
            np.random.shuffle(item_ids)
            n_test = 1  # Leave the latest one for each user for testing
            test_items = item_ids[:n_test]
            train_items = item_ids[n_test:]
            for item_id in train_items:
                self.train_data.append([user_id, item_id])
            for item_id in test_items:
                self.test_data.append([user_id, item_id])

        self.train_data = np.array(self.train_data)
        self.test_data = np.array(self.test_data)

        print(f"Dataset loading and processing complete:")
        print(f"  Total ratings loaded: {len(ratings)}")
        print(f"  Training samples: {len(self.train_data)}")
        print(f"  Test samples: {len(self.test_data)}")

    def get_test_data(self):
        return self.test_data

    def get_user_history(self, user_id_mapped):
        all_hist = self.user_item_matrix.get(user_id_mapped, [])
        test_items_for_user = set(self.test_data[self.test_data[:, 0] == user_id_mapped][:, 1])
        train_hist = [item for item in all_hist if item not in test_items_for_user]
        return train_hist

    def get_item_features(self):
        features = {}
        for _, row in self.movies.iterrows():
            original_id = row['movie_id']
            if original_id in self.movie_id_map:
                mapped_id = self.movie_id_map[original_id]
                features[mapped_id] = {
                    'title': row['title'],
                    'genres': row['genres']
                }
        return features

    def get_item_popularity(self):
        if 'movie_id_mapped' not in self.ratings_df.columns:
            return {}
        popularity = self.ratings_df['movie_id_mapped'].value_counts().to_dict()
        return popularity

    def get_user_features(self):
        """
        Get user features. In the ml-10m dataset, there is no users.dat, so an empty dictionary is returned.
        """
        print("Warning: MovieLens-10M dataset does not contain a users.dat file. Returning empty user features.")
        return {}  # Return an empty dictionary to ensure compatibility

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]


class JsonlDataset(Dataset):
    """
    A loader for loading and processing the val.jsonl dataset.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.samples = [json.loads(line) for line in lines]

        self.user_histories = {}
        self.ground_truth = {}
        self.candidates = {}
        self.item_features = {}
        self.user_ids = []

        for sample in self.samples:
            user_id = sample['user_id']
            self.user_ids.append(user_id)
            self.user_histories[user_id] = [item[0] for item in sample['item_list']]
            self.ground_truth[user_id] = sample['target_item'][0]
            self.candidates[user_id] = [candidate[0] for candidate in sample['candidates']]

            all_items_in_sample = sample['item_list'] + sample['candidates'] + [sample['target_item']]
            for item_id, title in all_items_in_sample:
                if item_id not in self.item_features:
                    self.item_features[item_id] = title

    def get_user_ids(self):
        return self.user_ids

    def get_user_history(self, user_id):
        return self.user_histories.get(user_id, [])

    def get_candidates_for_user(self, user_id):
        return self.candidates.get(user_id, [])

    def get_ground_truth_for_user(self, user_id):
        return self.ground_truth.get(user_id)

    def get_all_item_features(self):
        return self.item_features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]