import subprocess
import sys
from pathlib import Path

# Find the main project directory
project_root = Path(__file__).resolve().parents[1]

# Configurations we want to compare
# (MinHash permutations, LSH rows per band)
configurations = [
    (64, 4),
    (128, 4),
    (128, 8),
    (128, 16),
    (256, 8)
]

# Repeat each experiment five times
repetitions = 5

for num_perm, rows_per_band in configurations:

    for trial in range(1, repetitions + 1):

        print(
            f"\nRunning {num_perm} permutations, "
            f"{rows_per_band} rows per band "
            f"(Trial {trial}/{repetitions})",
            flush=True
        )

        # Run the same pipeline with this configuration
        subprocess.run(
            [
                sys.executable,
                str(project_root / "run_pipeline.py"),
                "--permutations", str(num_perm),
                "--rows", str(rows_per_band)
            ],
            cwd=project_root,
            check=True
        )

print("\nAll benchmark experiments completed!")
print("Each run was saved separately in results/experiments/")