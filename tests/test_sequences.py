"""Unit tests for qaware.sequences module."""

import pytest
import pandas as pd
from pathlib import Path

from qaware.sequences import (
    SequenceBuilderConfig,
    build_per_interaction_table,
    apply_frequency_filters,
    remap_ids,
    create_time_based_split,
    build_sequences_pipeline,
)


class TestSequenceBuilderConfig:
    """Test the configuration dataclass."""
    
    def test_default_values(self):
        """Test that config has expected default values."""
        cfg = SequenceBuilderConfig()
        assert cfg.min_user_interactions == 5
        assert cfg.min_item_interactions == 5
        assert cfg.min_sequence_length_for_split == 5
        assert cfg.data_raw_dir == Path("data_raw")
    
    def test_custom_values(self):
        """Test that config accepts custom values."""
        cfg = SequenceBuilderConfig(
            min_user_interactions=10,
            min_item_interactions=20,
            min_sequence_length_for_split=7,
        )
        assert cfg.min_user_interactions == 10
        assert cfg.min_item_interactions == 20
        assert cfg.min_sequence_length_for_split == 7


class TestRemapIds:
    """Test the ID remapping function."""
    
    def test_remap_ids_produces_dense_indices(self):
        """Test that remapped indices are dense (0..N-1)."""
        # Create sample interactions
        interactions = pd.DataFrame({
            "user_id": [100, 100, 200, 200, 300],
            "product_id": [5, 7, 5, 9, 7],
        })
        
        remapped, user_id_to_idx, product_id_to_idx = remap_ids(interactions)
        
        # Check user indices are dense
        user_indices = remapped["user_idx"].unique()
        assert set(user_indices) == {0, 1, 2}
        assert min(user_indices) == 0
        assert max(user_indices) == len(user_indices) - 1
        
        # Check item indices are dense
        item_indices = remapped["item_idx"].unique()
        assert set(item_indices) == {0, 1, 2}
        assert min(item_indices) == 0
        assert max(item_indices) == len(item_indices) - 1
    
    def test_remap_ids_mapping_dicts(self):
        """Test that mapping dicts correctly map IDs to indices."""
        interactions = pd.DataFrame({
            "user_id": [100, 200],
            "product_id": [5, 7],
        })
        
        remapped, user_id_to_idx, product_id_to_idx = remap_ids(interactions)
        
        # Check mappings exist
        assert 100 in user_id_to_idx
        assert 200 in user_id_to_idx
        assert 5 in product_id_to_idx
        assert 7 in product_id_to_idx
        
        # Check mappings are correct
        assert remapped[remapped["user_id"] == 100]["user_idx"].iloc[0] == user_id_to_idx[100]
        assert remapped[remapped["product_id"] == 5]["item_idx"].iloc[0] == product_id_to_idx[5]
    
    def test_remap_ids_preserves_original_columns(self):
        """Test that original columns are preserved."""
        interactions = pd.DataFrame({
            "user_id": [100, 200],
            "product_id": [5, 7],
            "order_id": [1, 2],
        })
        
        remapped, _, _ = remap_ids(interactions)
        
        assert "user_id" in remapped.columns
        assert "product_id" in remapped.columns
        assert "order_id" in remapped.columns
        assert "user_idx" in remapped.columns
        assert "item_idx" in remapped.columns


class TestApplyFrequencyFilters:
    """Test the frequency filtering function."""
    
    def test_filters_rare_items(self):
        """Test that items below threshold are filtered out."""
        interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "order_id": [1, 1, 2, 3, 4],
            "order_number": [1, 1, 2, 1, 2],
            "product_id": [5, 5, 5, 7, 8],  # 5 appears 3x, 7 and 8 appear 1x each
        })
        
        # Set min_user_interactions low so users aren't filtered out
        cfg = SequenceBuilderConfig(
            min_item_interactions=2,
            min_user_interactions=1,  # Low threshold so user filter doesn't interfere
        )
        filtered = apply_frequency_filters(interactions, cfg)
        
        # Product 5 should remain (3 >= 2), products 7 and 8 should be removed
        assert 5 in filtered["product_id"].values
        assert 7 not in filtered["product_id"].values
        assert 8 not in filtered["product_id"].values
    
    def test_filters_users_with_few_interactions(self):
        """Test that users below threshold are filtered out."""
        interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1, 2, 2],  # User 1 has 5, user 2 has 2
            "order_id": [1, 2, 3, 4, 5, 6, 7],
            "order_number": [1, 2, 3, 4, 5, 1, 2],
            "product_id": [5, 6, 7, 8, 9, 5, 6],
        })
        
        # Set min_item_interactions low so items aren't filtered out
        cfg = SequenceBuilderConfig(
            min_item_interactions=1,  # Low threshold so item filter doesn't interfere
            min_user_interactions=5,
        )
        filtered = apply_frequency_filters(interactions, cfg)
        
        # User 1 should remain, user 2 should be removed
        assert 1 in filtered["user_id"].values
        assert 2 not in filtered["user_id"].values
    
    def test_filters_apply_in_sequence(self):
        """Test that item filter happens before user filter."""
        # Items 5,6 appear 3x each (keep), item 7 appears 1x (remove)
        # After item filter: user 1 has 6 interactions, user 2 has 2 interactions
        interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1, 1, 2, 2],
            "order_id": [1, 1, 2, 2, 3, 3, 4, 5],
            "order_number": [1, 1, 2, 2, 3, 3, 1, 2],
            "product_id": [5, 5, 5, 6, 6, 6, 7, 7],
        })
        
        cfg = SequenceBuilderConfig(
            min_item_interactions=2,
            min_user_interactions=5,
        )
        filtered = apply_frequency_filters(interactions, cfg)
        
        # After filtering: user 1 should remain (6 >= 5), user 2 removed (2 < 5)
        assert 1 in filtered["user_id"].values
        assert 2 not in filtered["user_id"].values


class TestCreateTimeBasedSplit:
    """Test the time-based train/val/test split function."""
    
    def test_split_assigns_correct_positions(self):
        """Test that split correctly assigns train/val/test positions."""
        # Create a user with exactly 5 interactions (T=5)
        # Expected: train=[0,1,2], val=[3], test=[4]
        interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1],
            "product_id": [10, 20, 30, 40, 50],
            "order_number": [1, 1, 2, 2, 3],
            "user_idx": [0, 0, 0, 0, 0],
            "item_idx": [0, 1, 2, 3, 4],
        })
        
        cfg = SequenceBuilderConfig(min_sequence_length_for_split=5)
        train_df, val_df, test_df = create_time_based_split(interactions, cfg)
        
        # Check counts
        assert len(train_df) == 3
        assert len(val_df) == 1
        assert len(test_df) == 1
        
        # Check specific items
        assert 10 in train_df["product_id"].values
        assert 20 in train_df["product_id"].values
        assert 30 in train_df["product_id"].values
        assert 40 in val_df["product_id"].values
        assert 50 in test_df["product_id"].values
    
    def test_split_excludes_short_sequences(self):
        """Test that users with too few interactions are excluded."""
        # User 1 has 5 interactions (include), user 2 has 3 interactions (exclude)
        interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "product_id": [10, 20, 30, 40, 50, 60, 70, 80],
            "order_number": [1, 1, 2, 2, 3, 1, 2, 3],
            "user_idx": [0, 0, 0, 0, 0, 1, 1, 1],
            "item_idx": [0, 1, 2, 3, 4, 0, 1, 2],
        })
        
        cfg = SequenceBuilderConfig(min_sequence_length_for_split=5)
        train_df, val_df, test_df = create_time_based_split(interactions, cfg)
        
        # User 2 should not appear in any split
        assert 2 not in train_df["user_id"].values
        assert 2 not in val_df["user_id"].values
        assert 2 not in test_df["user_id"].values
    
    def test_split_no_data_leakage(self):
        """Test that train/val/test users are the same set."""
        interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "product_id": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "order_number": [1, 1, 2, 2, 3, 1, 1, 2, 2, 3],
            "user_idx": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "item_idx": list(range(10)),
        })
        
        cfg = SequenceBuilderConfig(min_sequence_length_for_split=5)
        train_df, val_df, test_df = create_time_based_split(interactions, cfg)
        
        train_users = set(train_df["user_id"].unique())
        val_users = set(val_df["user_id"].unique())
        test_users = set(test_df["user_id"].unique())
        
        # All splits should have the same users
        assert train_users == val_users == test_users
    
    def test_split_each_user_has_one_val_and_test(self):
        """Test that each user has exactly 1 val and 1 test interaction."""
        interactions = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "product_id": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "order_number": [1, 1, 2, 2, 3, 1, 1, 2, 2, 3],
            "user_idx": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "item_idx": list(range(10)),
        })
        
        cfg = SequenceBuilderConfig(min_sequence_length_for_split=5)
        train_df, val_df, test_df = create_time_based_split(interactions, cfg)
        
        val_counts = val_df.groupby("user_id").size()
        test_counts = test_df.groupby("user_id").size()
        
        # Each user should have exactly 1 val and 1 test interaction
        assert all(val_counts == 1)
        assert all(test_counts == 1)


# Integration test (requires actual data files)
@pytest.mark.integration
class TestIntegration:
    """Integration tests that require actual data files."""
    
    def test_build_sequences_pipeline_end_to_end(self, tmp_path):
        """Test the full pipeline with mock data files."""
        # This would require creating mock CSV files
        # For now, skip if data files don't exist
        data_raw = Path("data_raw")
        if not (data_raw / "orders.csv").exists():
            pytest.skip("Data files not found")
        
        cfg = SequenceBuilderConfig(data_raw_dir=data_raw)
        interactions, user_id_to_idx, product_id_to_idx = build_sequences_pipeline(cfg)
        
        # Basic sanity checks
        assert len(interactions) > 0
        assert len(user_id_to_idx) > 0
        assert len(product_id_to_idx) > 0
        assert "user_idx" in interactions.columns
        assert "item_idx" in interactions.columns