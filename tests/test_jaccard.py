import unittest

# Import the function we want to test from src/jaccard.py.
from src.jaccard import jaccard_similarity


# Group all tests for the Jaccard function in one class.
class TestJaccardSimilarity(unittest.TestCase):

    def test_identical_sets(self):
        # Identical sets should have a similarity of 1.0.
        set_a = {"a", "b"}
        set_b = {"a", "b"}

        self.assertEqual(jaccard_similarity(set_a, set_b), 1.0)

    def test_no_overlap(self):
        # No shared items means the similarity should be 0.0.
        set_a = {"a", "b"}
        set_b = {"c", "d"}

        self.assertEqual(jaccard_similarity(set_a, set_b), 0.0)

    def test_partial_overlap(self):
        # Intersection: {"b"} → 1 item.
        # Union: {"a", "b", "c"} → 3 items.
        # Expected similarity: 1 / 3.
        set_a = {"a", "b"}
        set_b = {"b", "c"}

        # Use assertAlmostEqual for calculated decimal values.
        self.assertAlmostEqual(
            jaccard_similarity(set_a, set_b), 1 / 3
        )

    def test_exact_threshold(self):
        # These sets share 4 items out of 5 unique items.
        # The similarity should be exactly our 0.80 cutoff.
        set_a = {"a", "b", "c", "d"}
        set_b = {"a", "b", "c", "d", "e"}

        score = jaccard_similarity(set_a, set_b)

        self.assertAlmostEqual(score, 0.80)

        # Confirm that a score at the cutoff qualifies.
        self.assertGreaterEqual(score, 0.80)

    def test_one_empty_set(self):
        # An empty set shares nothing with a nonempty set.
        # Check both input orders.
        self.assertEqual(
            jaccard_similarity(set(), {"a"}), 0.0
        )
        self.assertEqual(
            jaccard_similarity({"a"}, set()), 0.0
        )

    def test_both_empty(self):
        # Our function treats two empty sets as identical.
        # This also checks that it avoids division by zero.
        self.assertEqual(
            jaccard_similarity(set(), set()), 1.0
        )

    def test_symmetry(self):
        # Switching the input order should not change the score.
        set_a = {"a", "b"}
        set_b = {"b", "c", "d"}

        self.assertEqual(
            jaccard_similarity(set_a, set_b),
            jaccard_similarity(set_b, set_a)
        )

    # Run the tests if this file is executed directly.
if __name__ == "__main__":
    unittest.main()