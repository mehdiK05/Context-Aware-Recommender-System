import torch
from torch import nn

from . import config


class ContextAwareNCF(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        num_hours=24,
        num_days=7,
        num_months=12,
        num_time_of_day=4,
        num_genders=2,
        num_occupations=32,
        num_age_groups=6,
        num_genres=19,
        embedding_dim=config.EMBEDDING_DIM,
        use_time_context=False,
        use_user_context=False,
        use_movie_context=False,
    ):
        super().__init__()
        self.use_time_context = use_time_context
        self.use_user_context = use_user_context
        self.use_movie_context = use_movie_context

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        input_dim = embedding_dim * 2

        if use_time_context:
            self.hour_embedding = nn.Embedding(num_hours, embedding_dim // 2)
            self.day_embedding = nn.Embedding(num_days, embedding_dim // 2)
            self.month_embedding = nn.Embedding(num_months, embedding_dim // 2)
            self.time_of_day_embedding = nn.Embedding(num_time_of_day, embedding_dim // 2)
            input_dim += (embedding_dim // 2) * 4 + 1

        if use_user_context:
            self.gender_embedding = nn.Embedding(num_genders, embedding_dim // 2)
            self.occupation_embedding = nn.Embedding(num_occupations, embedding_dim // 2)
            self.age_group_embedding = nn.Embedding(num_age_groups, embedding_dim // 2)
            input_dim += (embedding_dim // 2) * 3

        if use_movie_context:
            self.genre_projection = nn.Linear(num_genres, embedding_dim)
            self.year_projection = nn.Linear(1, embedding_dim // 2)
            input_dim += embedding_dim + (embedding_dim // 2)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, batch):
        features = [
            self.user_embedding(batch["user_idx"]),
            self.movie_embedding(batch["movie_idx"]),
        ]
        if self.use_time_context:
            features.extend([
                self.hour_embedding(batch["hour_idx"]),
                self.day_embedding(batch["day_of_week_idx"]),
                self.month_embedding(batch["month_idx"]),
                self.time_of_day_embedding(batch["time_of_day_idx"]),
                batch["is_weekend"].float().unsqueeze(1),
            ])
        if self.use_user_context:
            features.extend([
                self.gender_embedding(batch["gender_idx"]),
                self.occupation_embedding(batch["occupation_idx"]),
                self.age_group_embedding(batch["age_group_idx"]),
            ])
        if self.use_movie_context:
            features.extend([
                torch.relu(self.genre_projection(batch["genre_vector"].float())),
                torch.relu(self.year_projection(batch["release_year_scaled"].float().unsqueeze(1))),
            ])
        x = torch.cat(features, dim=1)
        return self.mlp(x).squeeze(1)


def variant_flags(model_variant):
    return {
        "use_time_context": model_variant in {"time", "full"},
        "use_user_context": model_variant in {"user", "full"},
        "use_movie_context": model_variant in {"movie", "full"},
    }


def build_model(model_variant, metadata):
    return ContextAwareNCF(
        num_users=metadata["num_users"],
        num_movies=metadata["num_movies"],
        num_hours=metadata["num_hours"],
        num_days=metadata["num_days"],
        num_months=metadata["num_months"],
        num_time_of_day=metadata["num_time_of_day"],
        num_genders=metadata["num_genders"],
        num_occupations=metadata["num_occupations"],
        num_age_groups=metadata["num_age_groups"],
        num_genres=metadata["num_genres"],
        embedding_dim=config.EMBEDDING_DIM,
        **variant_flags(model_variant),
    )

