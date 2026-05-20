import numpy as np
import pandas as pd

from . import config


def build_user_positive_item_dict(df):
    positives = df[df["label"] == 1]
    return positives.groupby("user_idx")["movie_idx"].apply(lambda x: set(x.astype(int))).to_dict()


def _candidate_pool(all_movie_indices, blocked):
    return np.array([item for item in all_movie_indices if item not in blocked], dtype=int)


def _replace_movie_context(row, movie_idx, movie_metadata):
    new_row = row.copy()
    metadata = movie_metadata.loc[int(movie_idx)]
    new_row["movie_idx"] = int(movie_idx)
    new_row["movie_id"] = metadata["movie_id"]
    new_row["movie_title"] = metadata["movie_title"]
    new_row["release_year"] = metadata["release_year"]
    new_row["release_year_scaled"] = metadata["release_year_scaled"]
    for col in config.GENRE_COLUMNS:
        new_row[col] = metadata[col]
    return new_row


def sample_train_negatives(train_pos, all_movie_indices, user_pos_dict, num_negatives, movie_metadata, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in train_pos.iterrows():
        pos_row = row.copy()
        pos_row["label"] = 1
        rows.append(pos_row)
        blocked = user_pos_dict.get(int(row["user_idx"]), set())
        pool = _candidate_pool(all_movie_indices, blocked)
        n = min(num_negatives, len(pool))
        sampled = rng.choice(pool, size=n, replace=False)
        for movie_idx in sampled:
            neg_row = _replace_movie_context(row, movie_idx, movie_metadata)
            neg_row["label"] = 0
            rows.append(neg_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def create_eval_candidates(pos_df, all_movie_indices, user_pos_dict, num_negatives, movie_metadata, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for group_id, (_, row) in enumerate(pos_df.iterrows()):
        true_row = row.copy()
        true_row["eval_group_id"] = group_id
        true_row["is_true_item"] = 1
        true_row["label"] = 1
        rows.append(true_row)

        blocked = set(user_pos_dict.get(int(row["user_idx"]), set()))
        blocked.add(int(row["movie_idx"]))
        pool = _candidate_pool(all_movie_indices, blocked)
        n = min(num_negatives, len(pool))
        sampled = rng.choice(pool, size=n, replace=False)
        for movie_idx in sampled:
            neg_row = _replace_movie_context(row, movie_idx, movie_metadata)
            neg_row["eval_group_id"] = group_id
            neg_row["is_true_item"] = 0
            neg_row["label"] = 0
            rows.append(neg_row)
    return pd.DataFrame(rows).reset_index(drop=True)

