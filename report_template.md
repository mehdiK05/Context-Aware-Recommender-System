# Project 2: Context-Aware Recommender System on MovieLens 100K

## 1. Abstract

Summarize the project goal, MovieLens 100K dataset, baseline Neural Collaborative Filtering model, context-aware variants, ranking metrics, and the main findings from the ablation study.

## 2. Introduction

Explain recommender systems, collaborative filtering, why Neural Collaborative Filtering is a useful baseline, and why contextual information can improve recommendations. State the research questions:

1. Does adding contextual information improve Neural Collaborative Filtering?
2. Which context type helps most: time, user metadata, or movie metadata?
3. Does the full context-aware model outperform the baseline?
4. Are improvements visible in Hit Rate@10 and NDCG@10?
5. Can we interpret where context helps?

## 3. Problem Formulation

Define users, items, context, and the ranking task.

Given user `u`, movie `i`, and optional context `c`, the model estimates a score `s(u, i, c)`. The recommender ranks candidate unseen movies by this score. The task is implicit-feedback ranking, where ratings greater than or equal to 4 are treated as positive interactions.

## 4. Dataset

Describe MovieLens 100K:

- 100,000 ratings
- user identifiers and demographics
- movie identifiers, titles, release dates, and genres
- Unix timestamps for interactions

Mention the limitation: MovieLens 100K does not include real device or session features. Therefore, this project uses temporal features, user metadata, and movie metadata as context.

## 5. Preprocessing

Explain:

- Unix timestamp conversion
- creation of `hour`, `day_of_week`, `month`, `is_weekend`, and `time_of_day`
- creation of `age_group`
- extraction and scaling of `release_year`
- use of 19 genre columns as a multi-hot vector
- implicit-feedback conversion using `rating >= 4`
- temporal leave-one-out split
- negative sampling for training and evaluation

The temporal split holds out each user's newest positive interaction for test and second-newest positive interaction for validation. This avoids predicting past behavior from future behavior.

## 6. Model Architecture

Describe the five models:

- Baseline NCF
- NCF + Time Context
- NCF + User Context
- NCF + Movie Context
- Full Context-Aware NCF

Text architecture diagram:

```text
user_id -> user embedding ----\
movie_id -> movie embedding ----\
optional time embeddings --------> concatenate -> MLP -> relevance logit
optional user context embeddings /
optional movie context projections /
```

The MLP uses two hidden layers with ReLU activations and dropout, and outputs one logit trained with binary cross entropy.

## 7. Experimental Setup

Include:

- loss function: `BCEWithLogitsLoss`
- optimizer: Adam
- embedding dimension: 32
- batch size: 256
- maximum epochs: 20
- train negatives per positive: 4
- evaluation negatives per positive: 99
- metrics: HR@10, NDCG@10, Recall@10
- fixed random seed: 42

All models use the same train/validation/test split, negative samples, evaluation candidates, and hyperparameters.

## 8. Results

Insert the final metrics table from `results/metrics.csv`.

Include plots:

- `results/figures/model_comparison_hr10.png`
- `results/figures/model_comparison_ndcg10.png`

Discuss how each context-aware model compares with the baseline.

## 9. Ablation Study

Use `results/ablation_results.csv`.

Explain:

- which context improved performance most
- whether the full model improved over baseline
- whether any context group did not help
- how large the deltas were for HR@10, NDCG@10, and Recall@10

## 10. Interpretability Analysis

Discuss:

- performance by time of day
- performance by age group
- performance by genre
- what these patterns suggest about context usefulness

Reference plots:

- `results/figures/performance_by_time_of_day.png`
- `results/figures/performance_by_age_group.png`
- `results/figures/performance_by_genre.png`

## 11. Discussion

Discuss why the results make sense, possible overfitting, the limited size of MovieLens 100K, the absence of true session/device data, negative sampling limitations, and temporal split limitations.

## 12. Conclusion

Answer each research question clearly using the final test metrics and interpretability analysis.

## 13. Reproducibility

Explain:

- install dependencies with `pip install -r requirements.txt`
- place MovieLens 100K files in `data/raw/ml-100k/`
- run notebooks 01 through 06 in order
- random seed and hyperparameters are defined in `src/config.py`
- processed data, models, metrics, and plots are saved under `data/processed/`, `saved_models/`, and `results/`

