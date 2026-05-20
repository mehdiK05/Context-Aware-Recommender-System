import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config


def create_time_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    def bucket(hour):
        if 5 <= hour <= 11:
            return "morning"
        if 12 <= hour <= 17:
            return "afternoon"
        if 18 <= hour <= 22:
            return "evening"
        return "night"

    df["time_of_day"] = df["hour"].apply(bucket)
    return df


def create_age_group(df):
    df = df.copy()
    bins = [-np.inf, 17, 24, 34, 44, 54, np.inf]
    labels = ["under_18", "18_24", "25_34", "35_44", "45_54", "55_plus"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels).astype(str)
    return df


def extract_release_year(df):
    df = df.copy()
    title_year = df["movie_title"].astype(str).str.extract(r"\((\d{4})\)\s*$")[0]
    date_year = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["release_year"] = pd.to_numeric(title_year, errors="coerce").fillna(date_year)
    median_year = int(df["release_year"].median()) if df["release_year"].notna().any() else 1995
    df["release_year"] = df["release_year"].fillna(median_year).astype(int)
    scaler = MinMaxScaler()
    df["release_year_scaled"] = scaler.fit_transform(df[["release_year"]]).astype(float)
    return df


def merge_dataset(ratings, users, movies):
    df = ratings.merge(users, on="user_id", how="left").merge(movies, on="movie_id", how="left")
    df = create_time_features(df)
    df = create_age_group(df)
    df = extract_release_year(df)
    df["label"] = (df["rating"] >= config.POSITIVE_RATING_THRESHOLD).astype(int)
    for col in config.GENRE_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)
    return df


def _make_mapping(values):
    ordered = sorted(pd.Series(values).dropna().unique().tolist())
    return {str(value): idx for idx, value in enumerate(ordered)}


def encode_categorical_features(df):
    df = df.copy()
    mappings = {
        "user_id": _make_mapping(df["user_id"]),
        "movie_id": _make_mapping(df["movie_id"]),
        "gender": _make_mapping(df["gender"]),
        "occupation": _make_mapping(df["occupation"]),
        "age_group": _make_mapping(df["age_group"]),
        "time_of_day": _make_mapping(df["time_of_day"]),
        "hour": _make_mapping(df["hour"]),
        "day_of_week": _make_mapping(df["day_of_week"]),
        "month": _make_mapping(df["month"]),
    }
    df["user_idx"] = df["user_id"].astype(str).map(mappings["user_id"]).astype(int)
    df["movie_idx"] = df["movie_id"].astype(str).map(mappings["movie_id"]).astype(int)
    df["gender_idx"] = df["gender"].astype(str).map(mappings["gender"]).astype(int)
    df["occupation_idx"] = df["occupation"].astype(str).map(mappings["occupation"]).astype(int)
    df["age_group_idx"] = df["age_group"].astype(str).map(mappings["age_group"]).astype(int)
    df["time_of_day_idx"] = df["time_of_day"].astype(str).map(mappings["time_of_day"]).astype(int)
    df["hour_idx"] = df["hour"].astype(str).map(mappings["hour"]).astype(int)
    df["day_of_week_idx"] = df["day_of_week"].astype(str).map(mappings["day_of_week"]).astype(int)
    df["month_idx"] = df["month"].astype(str).map(mappings["month"]).astype(int)

    metadata = {
        "num_users": len(mappings["user_id"]),
        "num_movies": len(mappings["movie_id"]),
        "num_hours": len(mappings["hour"]),
        "num_days": len(mappings["day_of_week"]),
        "num_months": len(mappings["month"]),
        "num_time_of_day": len(mappings["time_of_day"]),
        "num_genders": len(mappings["gender"]),
        "num_occupations": len(mappings["occupation"]),
        "num_age_groups": len(mappings["age_group"]),
        "num_genres": len(config.GENRE_COLUMNS),
        "genre_columns": config.GENRE_COLUMNS,
        "mappings": mappings,
    }
    return df, mappings, metadata


def temporal_leave_one_out_split(df):
    positives = df[df["label"] == 1].copy()
    counts = positives.groupby("user_id").size()
    eligible_users = counts[counts >= 3].index
    positives = positives[positives["user_id"].isin(eligible_users)].sort_values(["user_id", "timestamp"])

    train_parts, val_parts, test_parts = [], [], []
    for _, group in positives.groupby("user_id", sort=False):
        # The last two positive interactions are held out. This avoids predicting past behavior from future behavior.
        train_parts.append(group.iloc[:-2])
        val_parts.append(group.iloc[[-2]])
        test_parts.append(group.iloc[[-1]])

    train_pos = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    val_pos = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    test_pos = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_pos, val_pos, test_pos


def build_movie_metadata(df):
    columns = [
        "movie_idx", "movie_id", "movie_title", "release_year", "release_year_scaled",
    ] + config.GENRE_COLUMNS
    return df[columns].drop_duplicates("movie_idx").set_index("movie_idx")


def build_metadata_from_mappings(mappings):
    return {
        "num_users": len(mappings["user_id"]),
        "num_movies": len(mappings["movie_id"]),
        "num_hours": len(mappings["hour"]),
        "num_days": len(mappings["day_of_week"]),
        "num_months": len(mappings["month"]),
        "num_time_of_day": len(mappings["time_of_day"]),
        "num_genders": len(mappings["gender"]),
        "num_occupations": len(mappings["occupation"]),
        "num_age_groups": len(mappings["age_group"]),
        "num_genres": len(config.GENRE_COLUMNS),
        "genre_columns": config.GENRE_COLUMNS,
        "mappings": mappings,
    }
