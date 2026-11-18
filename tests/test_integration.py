import unittest

from src.preprocess import clean_text, create_shingles
from src.minhash import create_minhash
from src.jaccard import jaccard_similarity
from src.lsh import find_candidate_pairs

# Test that our different algorithms work together
class TestIntegration(unittest.TestCase):

    def test_identical_reviews(self):
        # Two identical reviews should have similarity 1.0
        text = clean_text("This product works really well")
        shingles = create_shingles(text, k=2)

        # Check exact Jaccard
        self.assertEqual(
            jaccard_similarity(shingles, shingles), 1.0
        )

        # Create MinHash signatures
        minhash1 = create_minhash(shingles, num_perm=128)
        minhash2 = create_minhash(shingles, num_perm=128)

        # Identical reviews should have identical signatures
        self.assertEqual(minhash1.jaccard(minhash2), 1.0)

        # LSH should find the matching pair
        signatures = {
            10: minhash1.hashvalues,
            20: minhash2.hashvalues
        }

        candidates = find_candidate_pairs(
            signatures,
            rows_per_band=8
        )

        self.assertIn((10, 20), candidates)

    def test_different_reviews(self):
        # Completely different reviews should have no shared shingles
        review1 = create_shingles(
            clean_text("This is a really good product")
        )

        review2 = create_shingles(
            clean_text("The weather outside is very cold")
        )

        self.assertEqual(
            jaccard_similarity(review1, review2), 0.0
        )

    def test_invalid_band_size(self):
        # 128 permutations cannot be divided into bands of 3 rows
        shingles = create_shingles("simple product review")
        minhash = create_minhash(shingles, num_perm=128)

        with self.assertRaises(ValueError):
            find_candidate_pairs(
                {1: minhash.hashvalues},
                rows_per_band=3
            )

    def test_empty_lsh_input(self):
        # An empty collection should return no candidate pairs
        candidates = find_candidate_pairs({}, rows_per_band=4)

        self.assertEqual(candidates, set())

if __name__ == "__main__":
    unittest.main()