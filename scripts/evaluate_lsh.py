import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

# Allow imports from src when running this script.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.preprocess import create_shingles
from src.minhash import create_minhash
from src.lsh import find_candidate_pairs
from src.jaccard import jaccard_similarity

# Use the same reviews and reference matches as the Jaccard baseline.
output_dir = project_root / "results"
subset = pd.read_csv(
    output_dir / "jaccard_subset_1000.csv",
    keep_default_na=False
)
baseline = pd.read_csv(output_dir / "jaccard_pairs_1000.csv")

review_shingles = {}
for _, row in subset.iterrows():
    review_id = int(row["review_id"])
    review_shingles[review_id] = create_shingles(row["clean_text"], k=2)

true_pairs = set()
for _, row in baseline.iterrows():
    pair = tuple(sorted((
        int(row["review_id_1"]),
        int(row["review_id_2"])
    )))
    true_pairs.add(pair)

num_perm = 128
rows_per_band = 8

# Time signature creation, LSH, and exact verification together.
start = perf_counter()

signatures = {}
for review_id, shingles in review_shingles.items():
    signature = create_minhash(shingles, num_perm=num_perm)
    signatures[review_id] = signature.hashvalues

candidates = find_candidate_pairs(signatures, rows_per_band)

verified_rows = []
verified_pairs = set()

for review_a, review_b in sorted(candidates):
    score = jaccard_similarity(
        review_shingles[review_a],
        review_shingles[review_b]
    )
    if score >= 0.80:
        verified_pairs.add((review_a, review_b))
        verified_rows.append({
            "review_id_1": review_a,
            "review_id_2": review_b,
            "similarity": score
        })

elapsed = perf_counter() - start
recovered = verified_pairs & true_pairs

recall = 0.0
if true_pairs:
    recall = len(recovered) / len(true_pairs)

precision = 0.0
if candidates:
    precision = len(candidates & true_pairs) / len(candidates)

summary = {
    "reviews": len(review_shingles),
    "num_perm": num_perm,
    "bands": num_perm // rows_per_band,
    "rows_per_band": rows_per_band,
    "candidate_pairs": len(candidates),
    "verified_pairs": len(verified_pairs),
    "baseline_pairs": len(true_pairs),
    "recovered_pairs": len(recovered),
    "recall": recall,
    "candidate_precision": precision,
    "runtime_seconds": elapsed
}

# Save the measurements and verified matches.
pd.DataFrame([summary]).to_csv(
    output_dir / "lsh_summary_128_8.csv", index=False
)
pd.DataFrame(
    verified_rows,
    columns=["review_id_1", "review_id_2", "similarity"]
).to_csv(output_dir / "lsh_matches_128_8.csv", index=False)

print(pd.DataFrame([summary]).to_string(index=False))
print("\nSaved summary and matches to results/")
