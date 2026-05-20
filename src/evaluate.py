import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config
from .datasets import RecommendationDataset


def hit_rate_at_k(ranked_items, true_item, k):
    return float(true_item in ranked_items[:k])


def ndcg_at_k(ranked_items, true_item, k):
    for rank, item in enumerate(ranked_items[:k]):
        if item == true_item:
            return 1.0 / math.log2(rank + 2)
    return 0.0


def recall_at_k(ranked_items, true_item, k):
    return hit_rate_at_k(ranked_items, true_item, k)


def _move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def score_candidates(model, candidates_df, model_variant, batch_size=config.BATCH_SIZE, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = DataLoader(RecommendationDataset(candidates_df, model_variant), batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    scored = candidates_df.copy()
    scored["score"] = scores
    return scored


def evaluate_ranking(model, eval_candidates_df, metadata, model_variant, k=config.TOP_K, return_group_results=True):
    scored = score_candidates(model, eval_candidates_df, model_variant)
    group_rows = []
    for group_id, group in scored.groupby("eval_group_id"):
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        true_rows = ranked[ranked["is_true_item"] == 1]
        if true_rows.empty:
            continue
        true_movie_idx = int(true_rows.iloc[0]["movie_idx"])
        ranked_items = ranked["movie_idx"].astype(int).tolist()
        true_rank = ranked_items.index(true_movie_idx)
        group_rows.append({
            "eval_group_id": group_id,
            "user_idx": int(true_rows.iloc[0]["user_idx"]),
            "true_movie_idx": true_movie_idx,
            "true_movie_title": true_rows.iloc[0].get("movie_title", ""),
            "true_movie_rank": true_rank,
            f"hit_at_{k}": hit_rate_at_k(ranked_items, true_movie_idx, k),
            f"ndcg_at_{k}": ndcg_at_k(ranked_items, true_movie_idx, k),
            f"recall_at_{k}": recall_at_k(ranked_items, true_movie_idx, k),
            "time_of_day": true_rows.iloc[0].get("time_of_day", ""),
            "age_group": true_rows.iloc[0].get("age_group", ""),
            "top_genre": _primary_genre(true_rows.iloc[0]),
        })
    per_group = pd.DataFrame(group_rows)
    metrics = {
        "HR@10": per_group[f"hit_at_{k}"].mean() if not per_group.empty else np.nan,
        "NDCG@10": per_group[f"ndcg_at_{k}"].mean() if not per_group.empty else np.nan,
        "Recall@10": per_group[f"recall_at_{k}"].mean() if not per_group.empty else np.nan,
        "num_eval_groups": len(per_group),
    }
    if return_group_results:
        return metrics, per_group, scored
    return metrics


def _primary_genre(row):
    for genre in config.GENRE_COLUMNS:
        if int(row.get(genre, 0)) == 1:
            return genre
    return "unknown"

