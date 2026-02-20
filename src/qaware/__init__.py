def main() -> None:
    print("Hello from qaware!")


# Export main functions for easy imports
from qaware.sequences import (
    SequenceBuilderConfig,
    build_sequences_pipeline,
    create_time_based_split,
)

from qaware.baselines import (
    compute_popularity_baseline,
    build_cooccurrence_matrix,
    get_cooccurrence_recommendations,
    recommend_popularity,
    recommend_cooccurrence,
)
