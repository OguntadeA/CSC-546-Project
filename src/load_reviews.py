import json
from pathlib import Path

from src.preprocess import clean_text


def load_amazon_reviews(project_root, num_reviews=1000):
    """
    Load a specified number of Amazon reviews
    without reading the entire dataset into memory.
    """

    project_root = Path(project_root)

    dataset_path = (
        project_root / "data" / "All_Beauty (1).jsonl"
    )

    if num_reviews <= 0:
        raise ValueError("Number of reviews must be positive")

    reviews = []

    with open(dataset_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file):

            if not line.strip():
                continue

            review = json.loads(line)

            # Extract the actual review text
            text = clean_text(review.get("text", ""))

            # Skip empty reviews
            if not text:
                continue

            reviews.append({
                "review_id": line_number,
                "clean_text": text
            })

            # Stop when we've reached our target
            if len(reviews) >= num_reviews:
                break

    if len(reviews) < num_reviews:
        raise ValueError(
            f"Requested {num_reviews} reviews, "
            f"but only found {len(reviews)} usable reviews."
        )

    return reviews