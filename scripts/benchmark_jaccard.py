from pathlib import Path
from itertools import combinations
from time import perf_counter
from datetime import datetime
from uuid import uuid4

import pandas as pd

from sys import path

# Find the main project directory
project_root = Path(__file__).resolve().parents[1]
path.insert(0, str(project_root))

from src.preprocess import create_shingles
from src.jaccard import jaccard_similarity


# Load the same reviews used in the MinHash and LSH experiments
reviews = pd.read_csv(
    project_root / "results" / "jaccard_subset_1000.csv",
    keep_default_na=False
)

# Create 2-word shingles for every review
shingles = {}

for _, row in reviews.iterrows():
    review_id = int(row["review_id"])
    shingles[review_id] = create_shingles(
        row["clean_text"], k=2
    )

# Repeat the exact Jaccard comparison five times
results = []

for trial in range(1, 6):

    start = perf_counter()
    matching_pairs = 0
    comparisons = 0

    for id1, id2 in combinations(shingles.keys(), 2):

        similarity = jaccard_similarity(
            shingles[id1],
            shingles[id2]
        )

        comparisons += 1

        if similarity >= 0.80:
            matching_pairs += 1

    runtime = perf_counter() - start

    results.append({
        "trial": trial,
        "comparisons": comparisons,
        "matching_pairs": matching_pairs,
        "runtime_seconds": runtime
    })

    print(
        f"Trial {trial}: {comparisons} comparisons, "
        f"{matching_pairs} matches, {runtime:.3f}s"
    )

# Save in a new folder so we never overwrite old results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

output_dir = (
    project_root / "results" / "jaccard_benchmarks" /
    f"{timestamp}_{uuid4().hex[:8]}"
)

output_dir.mkdir(parents=True, exist_ok=False)

results_df = pd.DataFrame(results)

results_df.to_csv(
    output_dir / "jaccard_runtime.csv",
    index=False
)

print("\nAverage Jaccard runtime:",
      round(results_df["runtime_seconds"].mean(), 4), "seconds")

print("Results saved to:", output_dir)