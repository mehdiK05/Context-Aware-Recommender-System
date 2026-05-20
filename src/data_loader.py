import pandas as pd

from .config import GENRE_COLUMNS
from .utils import check_required_files


def load_ratings(path):
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    return pd.read_csv(path, sep="\t", names=columns, engine="python")


def load_users(path):
    columns = ["user_id", "age", "gender", "occupation", "zip_code"]
    return pd.read_csv(path, sep="|", names=columns, engine="python")


def load_movies(path):
    columns = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL"] + GENRE_COLUMNS
    return pd.read_csv(path, sep="|", names=columns, encoding="latin-1", engine="python")


def load_movielens_100k(raw_dir):
    check_required_files(raw_dir)
    ratings = load_ratings(raw_dir / "u.data")
    users = load_users(raw_dir / "u.user")
    movies = load_movies(raw_dir / "u.item")
    return ratings, users, movies

