from pathlib import Path
from datetime import datetime
from uuid import uuid4

import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]

# Find the latest saved analysis
analysis_dir = project_root / "results" / "analysis"
analysis_folders = sorted(
    folder for folder in analysis_dir.iterdir()
    if folder.is_dir() and (folder / "comparison.csv").exists()
)

if not analysis_folders:
    raise SystemExit("No experiment comparison found.")

comparison = pd.read_csv(
    analysis_folders[-1] / "comparison.csv"
)

# Select our preferred MinHash + LSH configuration
selected = comparison[
    (comparison["permutations"] == 128)
    & (comparison["rows_per_band"] == 8)
]

if len(selected) != 1:
    raise SystemExit("Expected exactly one 128/8 configuration.")

lsh_runtime = float(selected.iloc[0]["average_runtime"])

# Load latest exact Jaccard benchmark
jaccard_dir = project_root / "results" / "jaccard_benchmarks"

benchmark_folders = sorted(
    folder for folder in jaccard_dir.iterdir()
    if folder.is_dir() and (folder / "jaccard_runtime.csv").exists()
)

if not benchmark_folders:
    raise SystemExit("No Jaccard benchmark found.")

jaccard = pd.read_csv(
    benchmark_folders[-1] / "jaccard_runtime.csv"
)

jaccard_runtime = jaccard["runtime_seconds"].mean()

# Create a unique output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_dir = (
    project_root / "results" / "method_comparison"
    / f"{timestamp}_{uuid4().hex[:8]}"
)
output_dir.mkdir(parents=True, exist_ok=False)

# Generate the graph
methods = ["Exact Jaccard", "MinHash + LSH"]
times = [jaccard_runtime, lsh_runtime]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, times)

for bar, value in zip(bars, times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.05,
        f"{value:.2f}s",
        ha="center"
    )

plt.ylabel("Average runtime (seconds)")
plt.title("Exact Jaccard vs. MinHash + LSH")
plt.ylim(0, max(times) * 1.2)
plt.tight_layout()

plt.savefig(
    output_dir / "method_runtime_comparison.png",
    dpi=300
)
plt.close()

# Save the numerical comparison
result = pd.DataFrame({
    "method": methods,
    "average_runtime_seconds": times
})

result.to_csv(
    output_dir / "method_runtime_comparison.csv",
    index=False
)

print("\nExact Jaccard:", round(jaccard_runtime, 4), "seconds")
print("MinHash + LSH:", round(lsh_runtime, 4), "seconds")
print("Observed speedup:", round(jaccard_runtime / lsh_runtime, 2), "x")
print("\nResults saved to:", output_dir)