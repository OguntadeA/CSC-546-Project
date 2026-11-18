import pandas as pd
from time import perf_counter

from src.preprocess import create_shingles
from src.minhash import create_minhash
from src.lsh import find_candidate_pairs
from src.jaccard import jaccard_similarity


# Run the complete MinHash and LSH experiment
def run_pipeline(project_root, num_perm=128, rows_per_band=4):

    # Load the same 1,000 reviews used in our Jaccard baseline
    reviews = pd.read_csv(
        project_root / "results" / "jaccard_subset_1000.csv"
    )

    # Load the exact similar review pairs
    exact_pairs = pd.read_csv(
        project_root / "results" / "jaccard_pairs_1000.csv"
    )

    reviews = reviews.set_index("review_id")

    # Turn the reviews into 2-word shingles
    reviews["shingles"] = reviews["clean_text"].apply(
        lambda text: create_shingles(text, k=2)
    )

    # Generate MinHash signatures
    start = perf_counter()

    signatures = {}

    for review_id, row in reviews.iterrows():
        minhash = create_minhash(
            row["shingles"],
            num_perm=num_perm
        )

        # Convert MinHash objects into arrays for LSH
        signatures[review_id] = minhash.hashvalues

    minhash_time = perf_counter() - start

    # Use LSH to find candidate review pairs
    start = perf_counter()

    candidates = find_candidate_pairs(
        signatures,
        rows_per_band=rows_per_band
    )

    lsh_time = perf_counter() - start

    # Check candidates using exact Jaccard
    start = perf_counter()

    verified_pairs = set()

    for id1, id2 in candidates:
        similarity = jaccard_similarity(
            reviews.loc[id1, "shingles"],
            reviews.loc[id2, "shingles"]
        )

        if similarity >= 0.80:
            verified_pairs.add((id1, id2))

    verification_time = perf_counter() - start

    # Get the matching pairs from the Jaccard baseline
    true_pairs = set()

    for _, pair in exact_pairs.iterrows():
        id1 = int(pair["review_id_1"])
        id2 = int(pair["review_id_2"]) 
        true_pairs.add(tuple(sorted((id1, id2))))

    # Compare LSH results to the baseline
    correct_pairs = verified_pairs & true_pairs

    precision = (
        len(correct_pairs) / len(verified_pairs)
        if verified_pairs else 0
    )

    recall = (
        len(correct_pairs) / len(true_pairs)
        if true_pairs else 0
    )

    total_time = minhash_time + lsh_time + verification_time

    # Display the results
    print("Reviews processed:", len(reviews))
    print("Exact similar pairs:", len(true_pairs))
    print("LSH candidates:", len(candidates))
    print("Verified similar pairs:", len(verified_pairs))
    print("Correct pairs:", len(correct_pairs))
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"MinHash runtime: {minhash_time:.3f} seconds")
    print(f"LSH runtime: {lsh_time:.3f} seconds")
    print(f"Total runtime: {total_time:.3f} seconds")

    # Save results for later experiments
    results = pd.DataFrame([{
        "permutations": num_perm,
        "rows_per_band": rows_per_band,
        "candidates": len(candidates),
        "verified_pairs": len(verified_pairs),
        "precision": precision,
        "recall": recall,
        "total_time": total_time
    }])

    output = project_root / "results" / "pipeline_results.csv"
    results.to_csv(output, index=False)

    return results