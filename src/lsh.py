from collections import defaultdict
from itertools import combinations


def find_candidate_pairs(signatures, rows_per_band):
    """Return unique candidate pairs from a mapping of review IDs to signatures."""
    if rows_per_band <= 0:
        raise ValueError("rows_per_band must be positive")
    if not signatures:
        return set()

    signature_length = len(next(iter(signatures.values())))
    if signature_length == 0 or signature_length % rows_per_band != 0:
        raise ValueError("Signature length must be a positive multiple of rows_per_band")

    buckets = defaultdict(list)

    for review_id, signature in signatures.items():
        if len(signature) != signature_length:
            raise ValueError("All signatures must have the same length")

        # Group reviews by both the band's position and its values.
        for start in range(0, signature_length, rows_per_band):
            band_values = tuple(signature[start:start + rows_per_band])
            buckets[(start, band_values)].append(review_id)

    candidate_pairs = set()

    for review_ids in buckets.values():
        # Sorting gives each pair a consistent order.
        # The set keeps a pair only once if it shares multiple bands.
        for pair in combinations(sorted(review_ids), 2):
            candidate_pairs.add(pair)

    return candidate_pairs