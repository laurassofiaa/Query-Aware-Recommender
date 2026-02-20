"""Baseline recommendation methods for sequential recommender."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_popularity_baseline(
    train_df: pd.DataFrame, K: int = 20
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Baseline 1: Global popularity baseline.
    
    Returns the top-K most frequently purchased items in the training data.
    This is a non-personalized baseline - same recommendations for all users.
    
    Args:
        train_df: Training interactions DataFrame with 'item_idx' column.
        K: Number of top items to return.
    
    Returns:
        top_k_items: Array of top-K item_idx values (sorted by frequency, descending).
        item_scores: Dict mapping item_idx -> frequency count (for debugging/analysis).
    """
    if "item_idx" not in train_df.columns:
        raise ValueError("train_df must contain 'item_idx' column")
    
    # Count frequency of each item
    item_counts = train_df["item_idx"].value_counts()
    
    # Get top-K items
    top_k_items = item_counts.head(K).index.values
    
    # Create score dict (frequency counts)
    item_scores = item_counts.to_dict()
    
    return top_k_items, item_scores


def build_cooccurrence_matrix(train_df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    """
    Build a co-occurrence matrix for last-item -> next-item transitions.
    
    Counts how often item_b follows item_a in user sequences.
    
    Args:
        train_df: Training interactions DataFrame with 'user_id' and 'item_idx' columns.
                 Must be sorted by user and time.
    
    Returns:
        cooccurrence_counts: Dict mapping (item_a_idx, item_b_idx) -> count
                            of how many times item_b followed item_a.
    """
    if "user_id" not in train_df.columns or "item_idx" not in train_df.columns:
        raise ValueError("train_df must contain 'user_id' and 'item_idx' columns")
    
    cooccurrence_counts: Dict[Tuple[int, int], int] = {}
    
    # Group by user and iterate through their sequences
    for user_id, user_interactions in train_df.groupby("user_id"):
        items = user_interactions["item_idx"].values
        
        # For each consecutive pair (item_a, item_b), increment count
        for i in range(len(items) - 1):
            item_a = int(items[i])
            item_b = int(items[i + 1])
            pair = (item_a, item_b)
            cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1
    
    return cooccurrence_counts


def get_cooccurrence_recommendations(
    last_item_idx: int,
    cooccurrence_matrix: Dict[Tuple[int, int], int],
    K: int = 20,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Baseline 2: Last-item co-occurrence recommendations.
    
    Given a user's last item, recommend items that most often follow it
    in the training data (Markov-like approach).
    
    Args:
        last_item_idx: The item_idx of the user's last interaction.
        cooccurrence_matrix: Dict from build_cooccurrence_matrix().
        K: Number of recommendations to return.
    
    Returns:
        top_k_items: Array of top-K item_idx values (sorted by co-occurrence count).
        item_scores: Dict mapping item_idx -> co-occurrence count.
    
    Note:
        If last_item_idx has no transitions in the matrix, returns empty array.
    """
    # Find all items that follow last_item_idx
    following_items: Dict[int, int] = {}
    
    for (item_a, item_b), count in cooccurrence_matrix.items():
        if item_a == last_item_idx:
            following_items[item_b] = count
    
    if not following_items:
        # No transitions found for this item
        return np.array([], dtype=np.int64), {}
    
    # Sort by count (descending) and take top-K
    sorted_items = sorted(following_items.items(), key=lambda x: x[1], reverse=True)
    top_k_items = np.array([item_idx for item_idx, _ in sorted_items[:K]])
    item_scores = dict(following_items)
    
    return top_k_items, item_scores


def get_user_last_item(
    user_id: int, interactions_df: pd.DataFrame
) -> int | None:
    """
    Get the last item_idx for a given user from their interaction history.
    
    Args:
        user_id: The user's ID.
        interactions_df: DataFrame with 'user_id' and 'item_idx' columns,
                        sorted by user and time.
    
    Returns:
        The item_idx of the user's last interaction, or None if user not found.
    """
    user_interactions = interactions_df[interactions_df["user_id"] == user_id]
    
    if len(user_interactions) == 0:
        return None
    
    # Get the last item (assuming DataFrame is sorted by time)
    last_item = user_interactions["item_idx"].iloc[-1]
    return int(last_item)


def recommend_popularity(
    train_df: pd.DataFrame, K: int = 20
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Convenience function: Get popularity baseline recommendations.
    
    This is the same for all users - just returns global top-K items.
    
    Args:
        train_df: Training interactions DataFrame.
        K: Number of recommendations.
    
    Returns:
        top_k_items, item_scores: Same as compute_popularity_baseline().
    """
    return compute_popularity_baseline(train_df, K)


def recommend_cooccurrence(
    user_id: int,
    train_df: pd.DataFrame,
    context_df: pd.DataFrame | None = None,
    K: int = 20,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Convenience function: Get co-occurrence baseline recommendations for a user.
    
    Args:
        user_id: The user's ID.
        train_df: Training interactions DataFrame (used to build co-occurrence matrix).
        context_df: Optional DataFrame with user's context sequence.
                   If None, uses train_df to find user's last item.
        K: Number of recommendations.
    
    Returns:
        top_k_items, item_scores: Recommendations based on last-item co-occurrence.
    
    Note:
        If user has no history or last item has no transitions, returns empty array.
    """
    # Build co-occurrence matrix from training data
    cooccurrence_matrix = build_cooccurrence_matrix(train_df)
    
    # Get user's last item
    if context_df is not None:
        last_item = get_user_last_item(user_id, context_df)
    else:
        last_item = get_user_last_item(user_id, train_df)
    
    if last_item is None:
        return np.array([], dtype=np.int64), {}
    
    # Get recommendations
    return get_cooccurrence_recommendations(last_item, cooccurrence_matrix, K)
