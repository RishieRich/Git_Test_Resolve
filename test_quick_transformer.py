import math
import unittest

from quick_transformer import quick_transform, scaled_dot_product_attention


class AttentionTests(unittest.TestCase):
    def test_attention_preserves_sequence_and_value_dimensions(self) -> None:
        result = scaled_dot_product_attention(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [1.0, 3.0]],
        )

        self.assertEqual((len(result), len(result[0])), (2, 2))
        self.assertGreater(result[0][0], result[1][0])

    def test_quick_transform_produces_normalized_vectors(self) -> None:
        result = quick_transform(
            [[1.0, 0.0, 0.5], [0.2, 0.8, 0.1]],
            layers=2,
        )

        self.assertEqual(len(result), 2)
        for token in result:
            self.assertEqual(len(token), 3)
            self.assertTrue(math.isclose(sum(token), 0.0, abs_tol=1e-9))

    def test_quick_transform_rejects_mismatched_tokens(self) -> None:
        with self.assertRaises(ValueError):
            quick_transform([[1.0, 2.0], [3.0]])


if __name__ == "__main__":
    unittest.main()
