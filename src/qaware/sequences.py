from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


@dataclass(slots=True)
class SequenceBuilderConfig:
    """
    Configuration for building user purchase sequences from the Instacart data.

    This only covers:
    - joining the raw tables into a per-interaction table
    - time-ordering interactions per user
    - filtering infrequent users/items
    - remapping raw IDs to dense integer indices
    """

    data_raw_dir: Path = Path("data_raw")
    data_processed_dir: Path = Path("data_processed")

    # Filtering hyperparameters (can be tuned later)
    min_user_interactions: int = 5
    min_item_interactions: int = 5

    # Split hyperparameters
    min_sequence_length_for_split: int = 5  # Users need at least this many interactions to be split

    # File names expected in data_raw_dir
    orders_filename: str = "orders.csv"
    order_products_prior_filename: str = "order_products__prior.csv"

# creates a dataframe with the orders data
def _load_raw_orders(cfg: SequenceBuilderConfig) -> pd.DataFrame:
    orders_path = cfg.data_raw_dir / cfg.orders_filename
    df = pd.read_csv(
        orders_path,
        usecols=[
            "order_id",
            "user_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
            "eval_set",
        ],
    )
    # We only care about the \"prior\" interactions here; the train/test
    # subsets from the Kaggle competition spec are not needed because
    # we'll define our own time-based splits later.
    df = df[df["eval_set"] == "prior"].copy()
    return df

# creates a dataframe with the order_products__prior data
def _load_raw_order_products_prior(cfg: SequenceBuilderConfig) -> pd.DataFrame:
    opp_path = cfg.data_raw_dir / cfg.order_products_prior_filename
    df = pd.read_csv(
        opp_path,
        usecols=["order_id", "product_id", "add_to_cart_order"],
    )
    return df

# joins the orders and order_products__prior data into a per-interaction table
def build_per_interaction_table(cfg: SequenceBuilderConfig) -> pd.DataFrame:
    """
    Step 3.1/3.2: join raw tables into a per-interaction table and impose
    a clean temporal order per user.

    Output columns:
        user_id, order_id, order_number, order_dow,
        order_hour_of_day, days_since_prior_order, product_id
    """
    orders = _load_raw_orders(cfg)
    opp = _load_raw_order_products_prior(cfg)

    # Inner join to get one row per (user, order, product) interaction.
    interactions = opp.merge(orders, on="order_id", how="inner")

    # Sort by user and time so that each user's interactions are in
    # chronological order. add_to_cart_order gives within-order sequence,
    # which can be useful later, but for now we only care about product_id
    # as an interaction in the global sequence.
    interactions = interactions.sort_values(
        by=["user_id", "order_number", "order_id", "add_to_cart_order"],
    )

    # Keep only the columns we need for the sequence definition.
    interactions = interactions[
        [
            "user_id",
            "order_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
            "product_id",
        ]
    ].reset_index(drop=True)

    return interactions

# filters out very infrequent products and very short-user histories
def apply_frequency_filters(
    interactions: pd.DataFrame, cfg: SequenceBuilderConfig
) -> pd.DataFrame:
    """
    Step 3.3: filter out very infrequent products and very short-user histories.
    """
    # Product filter: drop items that appear fewer than M times globally.
    item_counts = interactions["product_id"].value_counts()
    keep_items = item_counts[item_counts >= cfg.min_item_interactions].index
    filtered = interactions[interactions["product_id"].isin(keep_items)].copy()

    # User filter: drop users with fewer than N remaining interactions.
    user_counts = filtered["user_id"].value_counts()
    keep_users = user_counts[user_counts >= cfg.min_user_interactions].index
    filtered = filtered[filtered["user_id"].isin(keep_users)].copy()

    # Re-sort to be safe after filtering.
    filtered = filtered.sort_values(
        by=["user_id", "order_number", "order_id"],
    ).reset_index(drop=True)

    return filtered

# remaps the user_id and product_id to dense integer indices (uus juttu)
def remap_ids(
    interactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Step 3.4: remap raw user_id and product_id to dense integer indices.

    Returns:
        interactions_remapped:
            Same as input but with extra columns:
                user_idx, item_idx
        user_id_to_idx:
            Dict mapping raw user_id -> user_idx (0..num_users-1)
        product_id_to_idx:
            Dict mapping raw product_id -> item_idx (0..num_items-1)

    Each distinct user_id gets a unique user_idx in 0 .. num_users - 1.
    Each distinct product_id gets a unique item_idx in 0 .. num_items - 1.
    """
    unique_users = interactions["user_id"].unique()
    unique_items = interactions["product_id"].unique()

    user_id_to_idx: Dict[int, int] = {
        int(u): i for i, u in enumerate(unique_users)
    }
    product_id_to_idx: Dict[int, int] = {
        int(p): i for i, p in enumerate(unique_items)
    }

    interactions_remapped = interactions.copy()
    interactions_remapped["user_idx"] = interactions_remapped["user_id"].map(
        user_id_to_idx
    )
    interactions_remapped["item_idx"] = interactions_remapped["product_id"].map(
        product_id_to_idx
    )

    return interactions_remapped, user_id_to_idx, product_id_to_idx


def build_sequences_pipeline(
    cfg: SequenceBuilderConfig,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Convenience function that runs steps 3.1–3.4 in order.

    It does NOT perform any train/val/test splitting or model training –
    those are handled later.

    Returns:
        interactions_remapped, user_id_to_idx, product_id_to_idx
    """
    per_interaction = build_per_interaction_table(cfg)
    filtered = apply_frequency_filters(per_interaction, cfg)
    remapped, user_id_to_idx, product_id_to_idx = remap_ids(filtered)
    return remapped, user_id_to_idx, product_id_to_idx


def create_time_based_split(
    interactions: pd.DataFrame,
    cfg: SequenceBuilderConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 4: Create time-based train/val/test split per user.

    Split rule (for users with T >= min_sequence_length_for_split):
    - Train: items 1..(T-2)  [indices 0 to T-3 in 0-indexed]
    - Val: item T-1          [index T-2 in 0-indexed]
    - Test: item T           [index T-1 in 0-indexed]

    Users with fewer than min_sequence_length_for_split interactions
    are excluded from the split (they won't appear in train/val/test).

    Args:
        interactions: DataFrame with columns including user_id, user_idx, item_idx,
                      and other interaction metadata. Must be sorted by user and time.
        cfg: Configuration object with min_sequence_length_for_split.

    Returns:
        train_df, val_df, test_df: Three DataFrames with the same columns as input,
                                   containing train/val/test interactions respectively.
    """
    # Add a per-user sequence position (0-indexed) for splitting
    interactions = interactions.copy()
    interactions["seq_pos"] = interactions.groupby("user_id").cumcount()

    # Get sequence length per user
    user_seq_lengths = interactions.groupby("user_id").size()

    # Filter to users with sufficient length
    eligible_users = user_seq_lengths[
        user_seq_lengths >= cfg.min_sequence_length_for_split
    ].index

    interactions_eligible = interactions[
        interactions["user_id"].isin(eligible_users)
    ].copy()

    # For each user, determine which interactions go to train/val/test
    def assign_split(row: pd.Series) -> str:
        user_id = row["user_id"]
        seq_pos = row["seq_pos"]
        seq_length = user_seq_lengths[user_id]

        # Train: positions 0 to (seq_length - 3) inclusive
        # Val: position (seq_length - 2)
        # Test: position (seq_length - 1)
        if seq_pos <= seq_length - 3:
            return "train"
        elif seq_pos == seq_length - 2:
            return "val"
        else:  # seq_pos == seq_length - 1
            return "test"

    interactions_eligible["split"] = interactions_eligible.apply(assign_split, axis=1)

    # Split into separate DataFrames
    train_df = interactions_eligible[interactions_eligible["split"] == "train"].drop(
        columns=["seq_pos", "split"]
    )
    val_df = interactions_eligible[interactions_eligible["split"] == "val"].drop(
        columns=["seq_pos", "split"]
    )
    test_df = interactions_eligible[interactions_eligible["split"] == "test"].drop(
        columns=["seq_pos", "split"]
    )

    # Reset indices for clean output
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df

