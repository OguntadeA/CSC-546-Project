import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import psutil

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.load_reviews import load_amazon_reviews
from src.preprocess import create_shingles
from src.minhash import create_minhash
from src.lsh import find_candidate_pairs
from src.jaccard import jaccard_similarity


def run_scalability_test(num_reviews, num_perm=128, rows_per_band=8):

    if num_reviews <= 0 or num_perm <= 0 or rows_per_band <= 0:
        raise ValueError("Arguments must be positive")

    if num_perm % rows_per_band != 0:
        raise ValueError("Permutations must be divisible by rows per band")

    # Begin with our 10K experiment.
    # Larger inputs require an additional memory-safety review.
    if num_reviews > 100000:
        raise ValueError(
            "Scalability experiments are currently limited to 100,000 reviews."
        )
    process = psutil.Process(os.getpid())
    memory_samples = []

    def measure_memory(stage):
        ram_mb = process.memory_info().rss / (1024 ** 2)
        memory_samples.append(ram_mb)
        print(f"{stage} memory: {ram_mb:.2f} MB", flush=True)

    print(f"\nStarting scalability test: {num_reviews:,} reviews")

    total_start = perf_counter()

    # Load and clean reviews
    start = perf_counter()
    reviews = load_amazon_reviews(project_root, num_reviews)
    loading_time = perf_counter() - start
    measure_memory("After loading")

    # Generate shingles
    start = perf_counter()

    shingles = {
        review["review_id"]: create_shingles(
            review["clean_text"], k=2
        )
        for review in reviews
    }

    shingling_time = perf_counter() - start
    measure_memory("After shingling")

    # Generate MinHash signatures
    start = perf_counter()

    signatures = {}

    for review_id, review_shingles in shingles.items():
        signature = create_minhash(
            review_shingles,
            num_perm=num_perm
        )
        signatures[review_id] = signature.hashvalues

    minhash_time = perf_counter() - start
    measure_memory("After MinHash")

    # Generate LSH candidate pairs
    start = perf_counter()

    candidates = find_candidate_pairs(
        signatures,
        rows_per_band=rows_per_band
    )

    lsh_time = perf_counter() - start
    measure_memory("After LSH")

    # Verify candidates with exact Jaccard
    start = perf_counter()
    verified_pairs = 0

    for id1, id2 in candidates:
        similarity = jaccard_similarity(
            shingles[id1],
            shingles[id2]
        )

        if similarity >= 0.80:
            verified_pairs += 1

    verification_time = perf_counter() - start
    measure_memory("After verification")

    total_time = perf_counter() - total_start

    possible_pairs = num_reviews * (num_reviews - 1) // 2

    comparison_reduction = (
        (1 - len(candidates) / possible_pairs) * 100
    )

    # These are sampled process memory values, not true peak RAM.
    max_observed_memory = max(memory_samples)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        project_root
        / "results"
        / "scalability"
        / f"{timestamp}_{num_reviews}reviews_{uuid4().hex[:8]}"
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    results = pd.DataFrame([{
        "reviews": num_reviews,
        "permutations": num_perm,
        "rows_per_band": rows_per_band,
        "possible_pairs": possible_pairs,
        "candidate_pairs": len(candidates),
        "verified_pairs": verified_pairs,
        "comparison_reduction_percent": comparison_reduction,
        "loading_time": loading_time,
        "shingling_time": shingling_time,
        "minhash_time": minhash_time,
        "lsh_time": lsh_time,
        "verification_time": verification_time,
        "total_time": total_time,
        "max_observed_rss_mb": max_observed_memory
    }])

    results.to_csv(
        output_dir / "scalability_summary.csv",
        index=False
    )

    print("\nSCALABILITY RESULTS")
    print(f"Reviews: {num_reviews:,}")
    print(f"Possible pairs: {possible_pairs:,}")
    print(f"LSH candidates: {len(candidates):,}")
    print(f"Verified matches: {verified_pairs:,}")
    print(f"Comparison reduction: {comparison_reduction:.4f}%")
    print(f"Loading time: {loading_time:.3f}s")
    print(f"Shingling time: {shingling_time:.3f}s")
    print(f"MinHash time: {minhash_time:.3f}s")
    print(f"LSH time: {lsh_time:.3f}s")
    print(f"Verification time: {verification_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Maximum sampled process RAM: {max_observed_memory:.2f} MB")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reviews", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--rows", type=int, default=8)

    args = parser.parse_args()

    run_scalability_test(
        num_reviews=args.reviews,
        num_perm=args.permutations,
        rows_per_band=args.rows
    )