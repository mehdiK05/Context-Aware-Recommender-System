import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .utils import ensure_dir


def _save_or_show(path):
    if path:
        ensure_dir(path.parent)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_rating_distribution(df, path=config.FIGURES_DIR / "rating_distribution.png"):
    df["rating"].value_counts().sort_index().plot(kind="bar", color="#4C78A8")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Rating Distribution")
    _save_or_show(path)


def plot_interactions_by_hour(df, path=config.FIGURES_DIR / "interaction_by_hour.png"):
    df["hour"].value_counts().sort_index().plot(kind="bar", color="#59A14F")
    plt.xlabel("Hour")
    plt.ylabel("Interactions")
    plt.title("Interactions by Hour")
    _save_or_show(path)


def plot_interactions_by_time_of_day(df, path=config.FIGURES_DIR / "interaction_by_time_of_day.png"):
    order = ["morning", "afternoon", "evening", "night"]
    df["time_of_day"].value_counts().reindex(order, fill_value=0).plot(kind="bar", color="#F28E2B")
    plt.xlabel("Time of Day")
    plt.ylabel("Interactions")
    plt.title("Interactions by Time of Day")
    _save_or_show(path)


def plot_model_comparison(metrics_df, metric_name, path=None):
    path = path or config.FIGURES_DIR / f"model_comparison_{metric_name.lower().replace('@', '').replace('/', '_')}.png"
    metrics_df.set_index("model")[metric_name].plot(kind="bar", color="#4C78A8")
    plt.ylabel(metric_name)
    plt.title(f"Model Comparison: {metric_name}")
    _save_or_show(path)


def plot_ablation_results(metrics_df, path=config.FIGURES_DIR / "ablation_ndcg10.png"):
    metrics_df.set_index("model")["NDCG@10"].plot(kind="bar", color="#B07AA1")
    plt.ylabel("NDCG@10")
    plt.title("Ablation Study by NDCG@10")
    _save_or_show(path)


def analyze_performance_by_time_of_day(per_group_eval_results, path=config.FIGURES_DIR / "performance_by_time_of_day.png"):
    summary = per_group_eval_results.groupby("time_of_day")[["hit_at_10", "ndcg_at_10"]].mean().reset_index()
    summary.set_index("time_of_day").plot(kind="bar")
    plt.ylabel("Metric")
    plt.title("Full Model Performance by Time of Day")
    _save_or_show(path)
    return summary


def analyze_performance_by_age_group(per_group_eval_results, path=config.FIGURES_DIR / "performance_by_age_group.png"):
    summary = per_group_eval_results.groupby("age_group")[["hit_at_10", "ndcg_at_10"]].mean().reset_index()
    summary.set_index("age_group").plot(kind="bar")
    plt.ylabel("Metric")
    plt.title("Full Model Performance by Age Group")
    _save_or_show(path)
    return summary


def analyze_performance_by_genre(per_group_eval_results, movies_df=None, path=config.FIGURES_DIR / "performance_by_genre.png"):
    summary = per_group_eval_results.groupby("top_genre")[["hit_at_10", "ndcg_at_10"]].mean().sort_values("ndcg_at_10", ascending=False).reset_index()
    summary.set_index("top_genre").plot(kind="bar", figsize=(10, 4))
    plt.ylabel("Metric")
    plt.title("Full Model Performance by Primary Genre")
    _save_or_show(path)
    return summary


def append_or_replace_metrics(path, new_rows, key_columns=("split", "model")):
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(list(key_columns), keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    return combined
