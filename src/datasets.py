import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


class RecommendationDataset(Dataset):
    def __init__(self, dataframe, model_variant="baseline"):
        self.df = dataframe.reset_index(drop=True).copy()
        self.model_variant = model_variant

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {
            "user_idx": torch.tensor(row["user_idx"], dtype=torch.long),
            "movie_idx": torch.tensor(row["movie_idx"], dtype=torch.long),
            "label": torch.tensor(row.get("label", 0), dtype=torch.float32),
        }
        if self.model_variant in {"time", "full"}:
            item.update({
                "hour_idx": torch.tensor(row["hour_idx"], dtype=torch.long),
                "day_of_week_idx": torch.tensor(row["day_of_week_idx"], dtype=torch.long),
                "month_idx": torch.tensor(row["month_idx"], dtype=torch.long),
                "time_of_day_idx": torch.tensor(row["time_of_day_idx"], dtype=torch.long),
                "is_weekend": torch.tensor(row["is_weekend"], dtype=torch.float32),
            })
        if self.model_variant in {"user", "full"}:
            item.update({
                "gender_idx": torch.tensor(row["gender_idx"], dtype=torch.long),
                "occupation_idx": torch.tensor(row["occupation_idx"], dtype=torch.long),
                "age_group_idx": torch.tensor(row["age_group_idx"], dtype=torch.long),
            })
        if self.model_variant in {"movie", "full"}:
            genres = row[config.GENRE_COLUMNS].to_numpy(dtype=np.float32)
            item.update({
                "genre_vector": torch.tensor(genres, dtype=torch.float32),
                "release_year_scaled": torch.tensor(row["release_year_scaled"], dtype=torch.float32),
            })
        return item

