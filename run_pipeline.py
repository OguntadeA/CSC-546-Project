from pathlib import Path
from src.pipeline import run_pipeline

# Find the main project directory
project_root = Path(__file__).resolve().parent

# Run MinHash + LSH with 128 permutations
# and 4 rows per band
run_pipeline(
    project_root,
    num_perm=128,
    rows_per_band=4
)