from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "ml-100k"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TRAINING_LOGS_DIR = RESULTS_DIR / "training_logs"
MODELS_DIR = PROJECT_ROOT / "saved_models"

RANDOM_SEED = 42
POSITIVE_RATING_THRESHOLD = 4
NUM_NEGATIVES_TRAIN = 4
NUM_NEGATIVES_EVAL = 99
TOP_K = 10
EMBEDDING_DIM = 32
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 4

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

MODEL_SAVE_PATHS = {
    "baseline": MODELS_DIR / "baseline_ncf.pt",
    "time": MODELS_DIR / "ncf_time.pt",
    "user": MODELS_DIR / "ncf_user_context.pt",
    "movie": MODELS_DIR / "ncf_movie_context.pt",
    "full": MODELS_DIR / "ncf_full_context.pt",
}

