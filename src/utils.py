import json
import random
from pathlib import Path

import numpy as np
import torch

from . import config


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df, path):
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def check_required_files(raw_dir=config.RAW_DATA_DIR):
    raw_dir = Path(raw_dir)
    required = ["u.data", "u.user", "u.item"]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing MovieLens 100K file(s): {missing_text}. "
            f"Place u.data, u.user, and u.item under: {raw_dir}"
        )
    return True

