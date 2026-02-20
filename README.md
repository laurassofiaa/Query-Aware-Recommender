# Query-Aware Recommender

A query-aware product recommendation system that combines sequential personalization with semantic search. Built for the Instacart dataset.

## Project Status

**Day 1 - Sequential Recommender (In Progress)**

### ✅ Completed Steps

#### Step 3: Sequence-Building Pipeline
- **Data Loading**: Loads and joins `orders.csv` and `order_products__prior.csv` from Instacart dataset
- **Time Ordering**: Sorts interactions chronologically per user by `order_number` and `order_id`
- **Filtering**: 
  - Removes products purchased fewer than `min_item_interactions` times globally
  - Removes users with fewer than `min_user_interactions` interactions
- **ID Remapping**: Maps raw `user_id` and `product_id` to dense integer indices (`user_idx`, `item_idx`) for efficient modeling

**Key Functions:**
- `build_per_interaction_table()` - Joins raw tables and creates per-interaction DataFrame
- `apply_frequency_filters()` - Filters rare items and short user sequences
- `remap_ids()` - Creates dense integer mappings
- `build_sequences_pipeline()` - Runs all steps sequentially

#### Step 4: Time-Based Train/Val/Test Split
- **Per-User Split**: For each user with T ≥ 5 interactions:
  - **Train**: Items 1..(T-2) [first T-2 interactions]
  - **Val**: Item T-1 [second-to-last interaction]
  - **Test**: Item T [last interaction]
- **No Data Leakage**: Ensures temporal ordering - train only uses earlier interactions
- **User Filtering**: Only users with sufficient interactions are included in splits

**Key Functions:**
- `create_time_based_split()` - Creates train/val/test DataFrames with time-based splits

#### Step 5: Baseline Recommendation Methods
- **Global Popularity Baseline**: Non-personalized baseline recommending top-K most frequently purchased items
- **Last-Item Co-occurrence Baseline**: Sequence-aware baseline that recommends items based on "people who bought X also bought Y next"
- Both baselines use only training data (no data leakage)

**Key Functions:**
- `compute_popularity_baseline()` - Computes global top-K items by frequency
- `build_cooccurrence_matrix()` - Builds transition counts (item_a -> item_b)
- `get_cooccurrence_recommendations()` - Gets recommendations based on last item
- `recommend_popularity()` - Convenience function for popularity baseline
- `recommend_cooccurrence()` - Convenience function for co-occurrence baseline

### ⏳ Next Steps

- Step 6: Train sequential model (SASRec/BERT4Rec or custom)
- Step 7: Evaluation protocol (Recall@K, NDCG@K)
- Step 8: Build `topK_rec(user_id)` interface

## Usage

### Setup

```bash
# Install dependencies
uv sync --all-groups

# Or activate virtual environment
source .venv/bin/activate
```

### Build Sequences

```python
from qaware.sequences import SequenceBuilderConfig, build_sequences_pipeline

cfg = SequenceBuilderConfig(
    data_raw_dir=Path("data_raw"),
    min_user_interactions=5,
    min_item_interactions=5,
)

interactions, user_id_to_idx, product_id_to_idx = build_sequences_pipeline(cfg)
```

### Create Train/Val/Test Split

```python
from qaware.sequences import create_time_based_split

train_df, val_df, test_df = create_time_based_split(interactions, cfg)
```

### Get Baseline Recommendations

```python
from qaware.baselines import recommend_popularity, recommend_cooccurrence

# Global popularity baseline (same for all users)
pop_items, pop_scores = recommend_popularity(train_df, K=20)

# Co-occurrence baseline (personalized based on last item)
user_id = 1
cooc_items, cooc_scores = recommend_cooccurrence(
    user_id, 
    train_df, 
    context_df=train_df,  # User's interaction history
    K=20
)
```

### Run Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
Query-Aware-Recommender/
├── data_raw/              # Raw Instacart CSV files (gitignored)
├── data_processed/        # Processed data (gitignored)
├── models/                # Saved model artifacts (gitignored)
├── src/qaware/            # Source code
│   ├── sequences.py       # Sequence building and splitting
│   └── baselines.py        # Baseline recommendation methods
├── tests/                 # Unit tests
│   ├── test_sequences.py
│   └── test_baselines.py
└── test.ipynb             # Exploration notebook (gitignored)
```
