
import pandas as pd
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from uuid import uuid4

from src.preprocess import create_shingles
from src.minhash import create_minhash
from src.lsh import find_candidate_pairs
from src.jaccard import jaccard_similarity


# Run MinHash and LSH on our Amazon review baseline
def run_pipeline(project_root, num_perm=128, rows_per_band=4):

    project_root = Path(project_root)
    threshold = 0.80

    # Make sure the LSH configuration is valid
    if num_perm <= 0 or rows_per_band <= 0:
        raise ValueError("Permutations and rows must be positive")

    if num_perm % rows_per_band != 0:
        raise ValueError("Permutations must be divisible by rows per band")

    # Load the exact same reviews and matches as our baseline
    reviews = pd.read_csv(
        project_root / "results" / "jaccard_subset_1000.csv",
        keep_default_na=False
    )

    baseline = pd.read_csv(
        project_root / "results" / "jaccard_pairs_1000.csv"
    )

    reviews = reviews.set_index("review_id")

    # Create two-word shingles
    reviews["shingles"] = reviews["clean_text"].apply(
        lambda text: create_shingles(text, k=2)
    )

    # Save the known similar pairs
    true_pairs = set()

    for _, row in baseline.iterrows():
        pair = tuple(sorted((
            int(row["review_id_1"]),
            int(row["review_id_2"])
        )))
        true_pairs.add(pair)

    # Generate MinHash signatures
    start = perf_counter()
    signatures = {}

    for review_id, row in reviews.iterrows():
        minhash = create_minhash(
            row["shingles"],
            num_perm=num_perm
        )
        signatures[review_id] = minhash.hashvalues

    minhash_time = perf_counter() - start

    # Find possible matches with LSH
    start = perf_counter()

    candidates = find_candidate_pairs(
        signatures,
        rows_per_band=rows_per_band
    )

    lsh_time = perf_counter() - start

    # Verify candidates with exact Jaccard
    start = perf_counter()

    verified_pairs = set()
    verified_rows = []

    for id1, id2 in sorted(candidates):
        score = jaccard_similarity(
            reviews.loc[id1, "shingles"],
            reviews.loc[id2, "shingles"]
        )

        if score >= threshold:
            verified_pairs.add((id1, id2))

            verified_rows.append({
                "review_id_1": id1,
                "review_id_2": id2,
                "similarity": score
            })

    verification_time = perf_counter() - start

    # Compare retrieved and verified pairs with ground truth
    correct_pairs = verified_pairs & true_pairs
    candidate_correct = candidates & true_pairs

    candidate_precision = (
        len(candidate_correct) / len(candidates)
        if candidates else 0.0
    )

    candidate_recall = (
        len(candidate_correct) / len(true_pairs)
        if true_pairs else 0.0
    )

    final_precision = (
        len(correct_pairs) / len(verified_pairs)
        if verified_pairs else 0.0
    )

    final_recall = (
        len(correct_pairs) / len(true_pairs)
        if true_pairs else 0.0
    )

    total_time = minhash_time + lsh_time + verification_time

    total_possible_pairs = len(reviews) * (len(reviews) - 1) // 2

    # Make a unique folder for every experiment
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_id = (
        f"{timestamp}_{num_perm}perm_"
        f"{rows_per_band}rows_{uuid4().hex[:8]}"
    )

    output_dir = project_root / "results" / "experiments" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    # Save all measurements for this run
    summary = pd.DataFrame([{
        "run_id": run_id,
        "reviews": len(reviews),
        "permutations": num_perm,
        "bands": num_perm // rows_per_band,
        "rows_per_band": rows_per_band,
        "threshold": threshold,
        "possible_pairs": total_possible_pairs,
        "candidate_pairs": len(candidates),
        "verified_pairs": len(verified_pairs),
        "baseline_pairs": len(true_pairs),
        "recovered_pairs": len(correct_pairs),
        "candidate_precision": candidate_precision,
        "candidate_recall": candidate_recall,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "minhash_time": minhash_time,
        "lsh_time": lsh_time,
        "verification_time": verification_time,
        "total_time": total_time
    }])

    summary.to_csv(output_dir / "summary.csv", index=False)

    pd.DataFrame(
        verified_rows,
        columns=["review_id_1", "review_id_2", "similarity"]
    ).to_csv(output_dir / "verified_pairs.csv", index=False)

    # Display the results
    print("\nMinHash + LSH Results")
    print("Reviews:", len(reviews))
    print("Permutations:", num_perm)
    print("Bands:", num_perm // rows_per_band)
    print("Rows per band:", rows_per_band)
    print("Candidates:", len(candidates))
    print("Verified pairs:", len(verified_pairs))
    print("Ground truth pairs:", len(true_pairs))
    print(f"Candidate precision: {candidate_precision:.3f}")
    print(f"Candidate recall: {candidate_recall:.3f}")
    print(f"Final precision: {final_precision:.3f}")
    print(f"Final recall: {final_recall:.3f}")
    print(f"MinHash time: {minhash_time:.3f}s")
    print(f"LSH time: {lsh_time:.3f}s")
    print(f"Verification time: {verification_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print("Results saved to:", output_dir)

    return summary
