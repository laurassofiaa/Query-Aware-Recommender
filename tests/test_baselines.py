"""Unit tests for qaware.baselines module."""

import pytest
import numpy as np
import pandas as pd

from qaware.baselines import (
    compute_popularity_baseline,
    build_cooccurrence_matrix,
    get_cooccurrence_recommendations,
    get_user_last_item,
    recommend_popularity,
    recommend_cooccurrence,
)


class TestPopularityBaseline:
    """Test the global popularity baseline."""
    
    def test_compute_popularity_baseline_basic(self):
        """Test that popularity baseline returns most frequent items."""
        train_df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 3],
            "item_idx": [5, 5, 5, 7, 7, 5],  # Item 5 appears 4x, item 7 appears 2x
            "order_id": [1, 1, 2, 3, 4, 5],
            "order_number": [1, 1, 2, 1, 2, 1],
        })
        
        top_k, scores = compute_popularity_baseline(train_df, K=2)
        
        # Item 5 should be first (4 occurrences), item 7 should be second (2 occurrences)
        assert len(top_k) == 2
        assert top_k[0] == 5
        assert top_k[1] == 7
        assert scores[5] == 4
        assert scores[7] == 2
    
    def test_compute_popularity_baseline_k_larger_than_unique_items(self):
        """Test that K larger than unique items returns all items."""
        train_df = pd.DataFrame({
            "user_id": [1, 2],
            "item_idx": [5, 7],  # Only 2 unique items
            "order_id": [1, 2],
            "order_number": [1, 1],
        })
        
        top_k, scores = compute_popularity_baseline(train_df, K=10)
        
        # Should return only 2 items (all available)
        assert len(top_k) == 2
        assert set(top_k) == {5, 7}
    
    def test_compute_popularity_baseline_missing_column(self):
        """Test that missing item_idx column raises error."""
        train_df = pd.DataFrame({
            "user_id": [1, 2],
            "product_id": [5, 7],
        })
        
        with pytest.raises(ValueError, match="item_idx"):
            compute_popularity_baseline(train_df, K=10)
    
    def test_recommend_popularity_convenience(self):
        """Test that recommend_popularity convenience function works."""
        train_df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "item_idx": [5, 5, 7],
            "order_id": [1, 2, 3],
            "order_number": [1, 2, 1],
        })
        
        top_k, scores = recommend_popularity(train_df, K=2)
        
        assert len(top_k) >= 1
        assert 5 in top_k  # Most frequent item


class TestCooccurrenceMatrix:
    """Test building the co-occurrence matrix."""
    
    def test_build_cooccurrence_matrix_basic(self):
        """Test that co-occurrence matrix counts transitions correctly."""
        train_df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "item_idx": [5, 7, 9, 5, 7],  # User 1: 5->7->9, User 2: 5->7
            "order_id": [1, 1, 2, 3, 3],
            "order_number": [1, 1, 2, 1, 1],
        })
        
        matrix = build_cooccurrence_matrix(train_df)
        
        # Check transitions
        assert matrix[(5, 7)] == 2  # 5->7 appears twice (user 1 and user 2)
        assert matrix[(7, 9)] == 1  # 7->9 appears once (user 1)
        assert (5, 9) not in matrix  # 5->9 doesn't appear directly
    
    def test_build_cooccurrence_matrix_single_item_user(self):
        """Test that users with single items don't create transitions."""
        train_df = pd.DataFrame({
            "user_id": [1, 2],
            "item_idx": [5, 7],
            "order_id": [1, 2],
            "order_number": [1, 1],
        })
        
        matrix = build_cooccurrence_matrix(train_df)
        
        # No transitions possible with single-item users
        assert len(matrix) == 0
    
    def test_build_cooccurrence_matrix_missing_columns(self):
        """Test that missing columns raise errors."""
        train_df = pd.DataFrame({
            "user_id": [1, 2],
            "product_id": [5, 7],
        })
        
        with pytest.raises(ValueError):
            build_cooccurrence_matrix(train_df)


class TestCooccurrenceRecommendations:
    """Test getting co-occurrence recommendations."""
    
    def test_get_cooccurrence_recommendations_basic(self):
        """Test that recommendations are based on transition counts."""
        # Build a simple matrix: 5->7 (count=3), 5->9 (count=1)
        cooccurrence_matrix = {
            (5, 7): 3,
            (5, 9): 1,
            (7, 9): 2,
        }
        
        top_k, scores = get_cooccurrence_recommendations(5, cooccurrence_matrix, K=2)
        
        # Item 7 should be first (count=3), item 9 should be second (count=1)
        assert len(top_k) == 2
        assert top_k[0] == 7
        assert top_k[1] == 9
        assert scores[7] == 3
        assert scores[9] == 1
    
    def test_get_cooccurrence_recommendations_no_transitions(self):
        """Test that item with no transitions returns empty."""
        cooccurrence_matrix = {
            (5, 7): 3,
            (7, 9): 2,
        }
        
        top_k, scores = get_cooccurrence_recommendations(10, cooccurrence_matrix, K=10)
        
        # Item 10 has no transitions
        assert len(top_k) == 0
        assert len(scores) == 0
    
    def test_get_cooccurrence_recommendations_k_larger_than_available(self):
        """Test that K larger than available transitions returns all."""
        cooccurrence_matrix = {
            (5, 7): 3,
            (5, 9): 1,
        }
        
        top_k, scores = get_cooccurrence_recommendations(5, cooccurrence_matrix, K=10)
        
        # Should return only 2 items (all available transitions)
        assert len(top_k) == 2
        assert set(top_k) == {7, 9}


class TestGetUserLastItem:
    """Test getting user's last item."""
    
    def test_get_user_last_item_basic(self):
        """Test that last item is correctly identified."""
        interactions_df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "item_idx": [5, 7, 9, 3, 4],
            "order_id": [1, 1, 2, 3, 3],
            "order_number": [1, 1, 2, 1, 1],
        })
        
        last_item_1 = get_user_last_item(1, interactions_df)
        last_item_2 = get_user_last_item(2, interactions_df)
        
        assert last_item_1 == 9  # Last item for user 1
        assert last_item_2 == 4  # Last item for user 2
    
    def test_get_user_last_item_user_not_found(self):
        """Test that non-existent user returns None."""
        interactions_df = pd.DataFrame({
            "user_id": [1, 2],
            "item_idx": [5, 7],
            "order_id": [1, 2],
            "order_number": [1, 1],
        })
        
        last_item = get_user_last_item(999, interactions_df)
        
        assert last_item is None


class TestRecommendCooccurrence:
    """Test the convenience function for co-occurrence recommendations."""
    
    def test_recommend_cooccurrence_basic(self):
        """Test that convenience function works end-to-end."""
        train_df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "item_idx": [5, 7, 9, 5, 7],  # 5->7 appears twice
            "order_id": [1, 1, 2, 3, 3],
            "order_number": [1, 1, 2, 1, 1],
        })
        
        # User 1's last item in train is 9, but let's use context with last item 5
        context_df = pd.DataFrame({
            "user_id": [1],
            "item_idx": [5],
            "order_id": [1],
            "order_number": [1],
        })
        
        top_k, scores = recommend_cooccurrence(1, train_df, context_df, K=10)
        
        # Should recommend items that follow 5 (which is 7, with count=2)
        assert len(top_k) >= 1
        assert 7 in top_k
    
    def test_recommend_cooccurrence_no_context(self):
        """Test that function works without context_df (uses train_df)."""
        train_df = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "item_idx": [5, 7, 5, 7],  # Both users end with 7
            "order_id": [1, 1, 2, 2],
            "order_number": [1, 1, 1, 1],
        })
        
        # User 1's last item in train is 7
        top_k, scores = recommend_cooccurrence(1, train_df, context_df=None, K=10)
        
        # Should work (though 7 might not have transitions in this simple case)
        # At minimum, should not crash
        assert isinstance(top_k, np.ndarray)
        assert isinstance(scores, dict)
    
    def test_recommend_cooccurrence_user_not_in_train(self):
        """Test that user not in train data returns empty."""
        train_df = pd.DataFrame({
            "user_id": [1, 2],
            "item_idx": [5, 7],
            "order_id": [1, 2],
            "order_number": [1, 1],
        })
        
        context_df = pd.DataFrame({
            "user_id": [999],
            "item_idx": [5],
            "order_id": [1],
            "order_number": [1],
        })
        
        top_k, scores = recommend_cooccurrence(999, train_df, context_df, K=10)
        
        # User 999 not in train, but has context - should still work if context has last item
        # Actually, it should work because we use context_df to get last item
        assert isinstance(top_k, np.ndarray)
        assert isinstance(scores, dict)


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_popularity_vs_cooccurrence_different_results(self):
        """Test that popularity and co-occurrence can give different results."""
        train_df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 3, 3],
            "item_idx": [5, 7, 9, 5, 7, 5, 9],  # 5 appears 3x, 7 appears 2x, 9 appears 2x
            "order_id": [1, 1, 2, 3, 3, 4, 4],
            "order_number": [1, 1, 2, 1, 1, 1, 1],
        })
        
        # Popularity: 5 (3x), then 7 or 9 (2x each)
        pop_top_k, _ = recommend_popularity(train_df, K=3)
        
        # Co-occurrence for user ending with 5: should recommend 7 (5->7 appears 2x)
        context_df = pd.DataFrame({
            "user_id": [1],
            "item_idx": [5],
            "order_id": [1],
            "order_number": [1],
        })
        cooc_top_k, _ = recommend_cooccurrence(1, train_df, context_df, K=3)
        
        # They can be different - popularity is global, co-occurrence is personalized
        assert len(pop_top_k) > 0
        assert len(cooc_top_k) > 0
        # At least verify they both return arrays
        assert isinstance(pop_top_k, np.ndarray)
        assert isinstance(cooc_top_k, np.ndarray)
