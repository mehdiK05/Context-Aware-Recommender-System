# Context-Aware Recommender System on MovieLens 100K

This project implements an Advanced Deep Learning recommender system study using MovieLens 100K. It compares a Neural Collaborative Filtering baseline against context-aware variants that add temporal context, user metadata, movie metadata, and all context groups together.

The goal is not only to train models, but to make the experimental design reproducible and interpretable: every model uses the same temporal split, negative sampling strategy, candidate sets, hyperparameters, and ranking metrics.

## Dataset

Use the MovieLens 100K files:

- `u.data`: `user_id`, `movie_id`, `rating`, `timestamp`
- `u.user`: `user_id`, `age`, `gender`, `occupation`, `zip_code`
- `u.item`: movie metadata and 19 genre columns

Place the files manually under:

```bash
data/raw/ml-100k/
```

Expected layout:

```text
data/raw/ml-100k/u.data
data/raw/ml-100k/u.user
data/raw/ml-100k/u.item
```

If these files are missing, the loader raises a clear error explaining where to put them.

MovieLens 100K does not contain real device or session fields. This project therefore uses honest available context: timestamp-derived features, user demographics, movie genres, and release year.

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

In VS Code, select the Python environment where these packages are installed, then open the notebooks from the `notebooks/` folder.

## How to Reproduce

Run the notebooks in order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_preprocessing_and_splitting.ipynb`
3. `notebooks/03_train_baseline_ncf.ipynb`
4. `notebooks/04_train_context_models.ipynb`
5. `notebooks/05_evaluation_and_ablation.ipynb`
6. `notebooks/06_interpretability_analysis.ipynb`

The preprocessing notebook writes processed CSV files and fixed validation/test candidate sets. The training notebooks save model checkpoints. The evaluation notebooks write metrics, ablation tables, per-group interpretability results, and plots.

## Models

- **E1 Baseline NCF**: user embedding + movie embedding.
- **E2 NCF + Time Context**: adds hour, day of week, weekend flag, month, and time of day.
- **E3 NCF + User Context**: adds age group, gender, and occupation.
- **E4 NCF + Movie Context**: adds multi-hot movie genres and release year.
- **E5 Full Context-Aware NCF**: combines time, user, and movie context.

All models are implemented in PyTorch in `src/models.py`.

## Metrics

Ranking is evaluated on one true positive item plus sampled negative candidates per evaluation case.

- **Hit Rate@10**: 1 if the true movie appears in the top 10, otherwise 0.
- **NDCG@10**: `1 / log2(rank + 2)` if the true movie appears in the top 10, otherwise 0. Rank starts at 0.
- **Recall@10**: equivalent to Hit Rate@10 here because each evaluation group has one true positive item.

## Project Structure

```text
data/raw/ml-100k/          Raw MovieLens files placed by the user
data/processed/           Processed interactions, splits, candidates, mappings
notebooks/                Executable VS Code Jupyter workflow
src/                      Reusable Python modules
results/                  Metrics, training logs, plots, interpretability outputs
saved_models/             PyTorch model checkpoints
report_template.md        Report scaffold for final write-up
```

## Outputs

Key generated files:

- `data/processed/interactions_processed.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/val_candidates.csv`
- `data/processed/test_candidates.csv`
- `data/processed/mappings.json`
- `results/metrics.csv`
- `results/ablation_results.csv`
- `results/training_logs/*_history.csv`
- `results/figures/*.png`
- `saved_models/*.pt`

