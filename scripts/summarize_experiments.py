from pathlib import Path
from datetime import datetime
import pandas as pd

# Find our project folder
project_root = Path(__file__).resolve().parents[1]

# Find every saved experiment
experiment_dir = project_root / "results" / "experiments"
# Only include original uninstrumented runtime experiments
# Memory-instrumented runs started after 18:40 
summary_files = sorted(
    file for file in experiment_dir.glob("*/summary.csv")
    if file.parent.name < "20261118_184000"
)
if not summary_files:
    print("No experiment results were found.")
    raise SystemExit

# Combine the results from all saved experiments
# Check each experiment file before combining results
valid_runs = []

for file in summary_files:
    try:
        df = pd.read_csv(file)
        valid_runs.append(df)
        print("Loaded:", file.parent.name)

    except pd.errors.ParserError as error:
        print("\nCSV FORMAT ERROR:")
        print("File:", file)
        print("Problem:", error)

if not valid_runs:
    raise SystemExit("No valid experiment summaries found.")

all_runs = pd.concat(valid_runs, ignore_index=True)

# Group runs by their permutation count and LSH configuration
comparison = (
    all_runs.groupby(
        ["reviews", "permutations", "rows_per_band"],
        as_index=False
    )
    .agg(
        repetitions=("total_time", "count"),
        average_runtime=("total_time", "mean"),
        runtime_std=("total_time", "std"),
        average_candidates=("candidate_pairs", "mean"),
        average_precision=("candidate_precision", "mean"),
        average_recall=("final_recall", "mean")
    )
)

comparison["runtime_std"] = comparison["runtime_std"].fillna(0)

# Create a new folder so we don't overwrite earlier analyses
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_dir = project_root / "results" / "analysis" / timestamp
output_dir.mkdir(parents=True, exist_ok=False)

# Save our combined results
all_runs.to_csv(output_dir / "all_runs.csv", index=False)
comparison.to_csv(output_dir / "comparison.csv", index=False)

import matplotlib.pyplot as plt

# Create labels for each MinHash and LSH configuration
labels = [
    f"{int(row.permutations)} perm / {int(row.rows_per_band)} rows"
    for row in comparison.itertuples()
]

# Graph 1: Average runtime with standard deviation
plt.figure(figsize=(10, 5))

plt.bar(
    labels,
    comparison["average_runtime"],
    yerr=comparison["runtime_std"],
    capsize=5
)

plt.title("MinHash + LSH Runtime Comparison")
plt.ylabel("Average runtime (seconds)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(output_dir / "runtime_comparison.png", dpi=300)
plt.close()


# Graph 2: Candidate precision
plt.figure(figsize=(10, 5))

plt.bar(
    labels,
    comparison["average_precision"] * 100
)

plt.title("LSH Candidate Precision")
plt.ylabel("Candidate precision (%)")
plt.ylim(0, 105)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(output_dir / "candidate_precision.png", dpi=300)
plt.close()


# Graph 3: Recall
plt.figure(figsize=(10, 5))

plt.bar(
    labels,
    comparison["average_recall"] * 100
)

plt.title("LSH Recall by Configuration")
plt.ylabel("Recall (%)")
plt.ylim(0, 105)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(output_dir / "recall_comparison.png", dpi=300)
plt.close()

print("\nThree comparison graphs saved!")

print("Total experiment runs:", len(all_runs))
print("\nComparison results:")
print(comparison.to_string(index=False))
print("\nSaved to:", output_dir)


# SCALABILITY ANALYSIS

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]

scalability_files = sorted(
    (root / "results" / "scalability").glob(
        "*/scalability_summary.csv"
    )
)

if not scalability_files:
    raise SystemExit("No scalability results found.")

# Combine saved measurements
scalability = pd.concat(
    [pd.read_csv(file) for file in scalability_files],
    ignore_index=True
)

# For now, use the latest run at each size.
# single-run measurements
scalability = (
    scalability
    .drop_duplicates(subset=["reviews"], keep="last")
    .sort_values("reviews")
)

# Save the latest analysis in stable locations
figures_dir = root / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

scalability.to_csv(
    root / "results" / "scalability_comparison.csv",
    index=False
)

# Graph runtime, memory, and candidate reduction
graphs = [
    (
        "total_time",
        "Runtime vs. Dataset Size",
        "Runtime (seconds)",
        "scalability_runtime.png"
    ),
    (
        "max_observed_rss_mb",
        "Memory Usage vs. Dataset Size",
        "Maximum Sampled RSS (MB)",
        "scalability_memory.png"
    ),
    (
        "candidate_pairs",
        "LSH Candidates vs. Dataset Size",
        "Candidate Pairs",
        "scalability_candidates.png"
    ),
    (
        "comparison_reduction_percent",
        "Candidate Reduction vs. Dataset Size",
        "Reduction (%)",
        "scalability_reduction.png"
    )
]

for column, title, ylabel, filename in graphs:

    plt.figure(figsize=(9, 5))

    plt.plot(
        scalability["reviews"],
        scalability[column],
        marker="o"
    )

    plt.title(title)
    plt.xlabel("Number of Amazon Reviews")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(figures_dir / filename, dpi=300)
    plt.close()

print("\nFINAL SCALABILITY SUMMARY")
print(
    scalability[
        [
            "reviews",
            "candidate_pairs",
            "verified_pairs",
            "total_time",
            "max_observed_rss_mb",
            "comparison_reduction_percent"
        ]
    ].to_string(index=False)
)

print("\nScalability graphs saved to:", figures_dir)