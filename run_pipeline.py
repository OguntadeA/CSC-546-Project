
from pathlib import Path
import argparse

from src.pipeline import run_pipeline

# Let us choose experiment settings from the terminal
parser = argparse.ArgumentParser()

parser.add_argument(
    "--permutations",
    type=int,
    default=128
)

parser.add_argument(
    "--rows",
    type=int,
    default=4
)

args = parser.parse_args()

# Find the main project folder
project_root = Path(__file__).resolve().parent

# Run and save this experiment
run_pipeline(
    project_root,
    num_perm=args.permutations,
    rows_per_band=args.rows
)
