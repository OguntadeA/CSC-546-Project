import tracemalloc
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import pandas as pd

from sys import path

# Find our project folder
project_root = Path(__file__).resolve().parents[1]
path.insert(0, str(project_root))

from src.pipeline import run_pipeline


# Configurations to test
configurations = [
    (64, 4),
    (128, 4),
    (128, 8),
    (128, 16),
    (256, 8)
]

results = []

# Measure memory while running each configuration
for num_perm, rows_per_band in configurations:

    tracemalloc.start()

    run_pipeline(
        project_root,
        num_perm=num_perm,
        rows_per_band=rows_per_band
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results.append({
        "permutations": num_perm,
        "rows_per_band": rows_per_band,
        "peak_memory_mb": peak / (1024 * 1024)
    })

# Save memory benchmark in a unique folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_dir = (
    project_root / "results" / "memory" /
    f"{timestamp}_{uuid4().hex[:8]}"
)
output_dir.mkdir(parents=True, exist_ok=False)

pd.DataFrame(results).to_csv(
    output_dir / "memory_comparison.csv",
    index=False
)

print("\nMemory benchmark results:")
print(pd.DataFrame(results).to_string(index=False))
print("\nSaved to:", output_dir)