"""Numerical checks and structural coverage for the logical tiled machines."""

from collections import Counter
from dataclasses import replace
import random
import unittest

from pim_foundry.core import ReferenceDevice, Tensor
from pim_foundry.reference import evaluate_matvec, evaluate_relu


def scalar_oracle(x, weights):
    """Column dot products independent of the engine's tile traversal."""
    rows, columns = weights.shape
    return tuple(
        sum(x[row] * weights.data[row * columns + column] for row in range(rows))
        for column in range(columns)
    )


def device(kind="sram", tile_k=3, tile_n=2, units=2, capacity_bytes=1_000_000):
    return ReferenceDevice("target", kind, capacity_bytes, tile_k, tile_n, units)


class MatvecReferenceTests(unittest.TestCase):
    def test_randomized_values_match_independent_oracle_across_tiles(self):
        rng = random.Random(731)
        for _ in range(30):
            k_size, n_size = rng.randint(1, 13), rng.randint(1, 11)
            x = tuple(rng.randint(-19, 19) for _ in range(k_size))
            weights = Tensor(
                "w", (k_size, n_size),
                tuple(rng.randint(-23, 23) for _ in range(k_size * n_size)),
            )
            expected = scalar_oracle(x, weights)
            for kind in ("sram", "hbm"):
                for tile_k, tile_n, units in ((1, 1, 1), (3, 2, 2), (5, 7, 3), (32, 32, 8)):
                    with self.subTest(shape=weights.shape, kind=kind, tile=(tile_k, tile_n, units)):
                        result = evaluate_matvec(x, weights, device(kind, tile_k, tile_n, units))
                        self.assertEqual(result.values, expected)

    def test_tiles_cover_every_weight_exactly_once_with_load_before_compute(self):
        weights = Tensor("w", (7, 5), tuple(range(-17, 18)))
        x = (-3, 4, -5, 6, -7, 8, -9)
        for kind, compute_kind in (("sram", "array_mvm"), ("hbm", "bank_mac")):
            target = device(kind, tile_k=3, tile_n=2, units=2)
            result = evaluate_matvec(x, weights, target)
            self.assertEqual(len(result.events), 18)
            coverage = Counter()
            visited = []
            for index in range(0, len(result.events), 2):
                load, compute = result.events[index:index + 2]
                self.assertEqual(load["kind"], "weight_load")
                self.assertEqual(compute["kind"], compute_kind)
                for key in ("k0", "k1", "n0", "n1", "unit"):
                    self.assertEqual(load[key], compute[key])
                k0, k1, n0, n1 = (load[key] for key in ("k0", "k1", "n0", "n1"))
                self.assertTrue(0 <= k0 < k1 <= 7)
                self.assertTrue(0 <= n0 < n1 <= 5)
                self.assertLessEqual(k1 - k0, target.tile_k)
                self.assertLessEqual(n1 - n0, target.tile_n)
                self.assertEqual(load["unit"], (n0 // target.tile_n) % target.units)
                self.assertTrue(0 <= load["unit"] < target.units)
                self.assertEqual(load["logical_bytes"], 8 * (k1 - k0) * (n1 - n0))
                self.assertEqual(compute["macs"], (k1 - k0) * (n1 - n0))
                coverage.update((k, n) for k in range(k0, k1) for n in range(n0, n1))
                visited.append((n0, k0))
            self.assertEqual(coverage, Counter((k, n) for k in range(7) for n in range(5)))
            self.assertEqual(visited, [(n, k) for n in (0, 2, 4) for k in (0, 3, 6)])
            self.assertEqual(sum(e.get("macs", 0) for e in result.events), 35)
            self.assertEqual(sum(e.get("logical_bytes", 0) for e in result.events), 280)

    def test_exact_intermediates_may_exceed_int64_and_cancel_across_tiles(self):
        maximum = 2**63 - 1
        x = (maximum, maximum, 1)
        weights = Tensor("w", (3, 1), (maximum, -maximum, -7))
        for kind in ("sram", "hbm"):
            for tile_k in (1, 2, 3, 8):
                with self.subTest(kind=kind, tile_k=tile_k):
                    self.assertEqual(
                        evaluate_matvec(x, weights, device(kind, tile_k=tile_k)).values,
                        (-7,),
                    )

    def test_int64_boundary_results_are_preserved(self):
        x = (-(2**63), 2**63 - 1)
        weights = Tensor("identity", (2, 2), (1, 0, 0, 1))
        for kind in ("sram", "hbm"):
            self.assertEqual(evaluate_matvec(x, weights, device(kind)).values, x)

    def test_final_overflow_is_rejected_in_both_directions(self):
        weights = Tensor("w", (1, 1), (2,))
        for kind in ("sram", "hbm"):
            for value in (2**62, -(2**62) - 1):
                with self.subTest(kind=kind, value=value):
                    with self.assertRaisesRegex(ValueError, "signed int64"):
                        evaluate_matvec((value,), weights, device(kind))

    def test_capacity_includes_full_input_output_and_one_clamped_weight_tile(self):
        x = (1, 2, 3)
        weights = Tensor("w", (3, 2), (1, 2, 3, 4, 5, 6))
        for kind in ("sram", "hbm"):
            for tile_k, tile_n, needed in ((100, 100, 88), (2, 1, 56)):
                target = device(kind, tile_k, tile_n, capacity_bytes=needed)
                with self.subTest(kind=kind, tile=(tile_k, tile_n)):
                    self.assertEqual(evaluate_matvec(x, weights, target).values, scalar_oracle(x, weights))
                    with self.assertRaisesRegex(ValueError, "logical bytes"):
                        evaluate_matvec(x, weights, replace(target, capacity_bytes=needed - 1))

    def test_shape_and_initializer_validation(self):
        for weights in (
            Tensor("w", (2,), (1, 2)), Tensor("w", (2, 2)),
            Tensor("w", (3, 1), (1, 2, 3)), None,
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    evaluate_matvec((1, 2), weights, device())

    def test_inputs_remain_unchanged_and_execution_is_deterministic(self):
        x = (3, -2, 4)
        weights = Tensor("w", (3, 2), (1, -2, 3, -4, 5, -6))
        before = (x, weights.shape, weights.data)
        for kind in ("sram", "hbm"):
            first = evaluate_matvec(x, weights, device(kind))
            self.assertEqual(first, evaluate_matvec(x, weights, device(kind)))
            self.assertEqual((x, weights.shape, weights.data), before)


class ReluReferenceTests(unittest.TestCase):
    def test_values_event_and_input_preservation(self):
        x = (-(2**63), -2, 0, 1, 2**63 - 1)
        for kind in ("sram", "hbm"):
            result = evaluate_relu(x, device(kind))
            self.assertEqual(result.values, (0, 0, 0, 1, 2**63 - 1))
            self.assertEqual(result.events, ({"kind": "vector_relu", "elements": 5},))
            self.assertEqual(x, (-(2**63), -2, 0, 1, 2**63 - 1))

    def test_capacity_accounts_for_both_input_and_output(self):
        for kind in ("sram", "hbm"):
            self.assertEqual(evaluate_relu((-2, 3), device(kind, capacity_bytes=32)).values, (0, 3))
            with self.assertRaisesRegex(ValueError, "logical bytes"):
                evaluate_relu((-2, 3), device(kind, capacity_bytes=31))


class ReferenceInputValidationTests(unittest.TestCase):
    def test_non_tuple_empty_boolean_non_integer_and_out_of_range_inputs(self):
        invalid = ((), [], [1], (True,), (False,), (1.0,), ("1",), (None,), (2**63,), (-(2**63) - 1,))
        weights = Tensor("w", (1, 1), (1,))
        for x in invalid:
            for kind in ("sram", "hbm"):
                with self.subTest(x=x, kind=kind):
                    with self.assertRaises(ValueError):
                        evaluate_matvec(x, weights, device(kind))
                    with self.assertRaises(ValueError):
                        evaluate_relu(x, device(kind))

    def test_oversized_standalone_vector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "1000000"):
            evaluate_relu((0,) * 1_000_001, device(capacity_bytes=100_000_000))

    def test_invalid_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ReferenceDevice"):
            evaluate_matvec((1,), Tensor("w", (1, 1), (1,)), None)
        with self.assertRaisesRegex(ValueError, "ReferenceDevice"):
            evaluate_relu((1,), None)


if __name__ == "__main__":
    unittest.main()
