"""Validate emitted memory/instructions independently of native DCC timing."""

import base64
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from pim_foundry import banked
from pim_foundry.typed import Arithmetic, Graph, Node, Tensor, execute


def oracle(x, w):
    m, k, n = len(x), len(w), len(w[0])
    graph = Graph((Tensor("x", (m, k), "float16"),
                   Tensor("w", (k, n), "float16", tuple(value for row in w for value in row)),
                   Tensor("y", (m, n), "float16")),
                  (Node("dot", "matmul", ("x", "w"), "y", Arithmetic("float32")),), ("x",), ("y",))
    flat = execute(graph, {"x": tuple(value for row in x for value in row)})["y"]
    return [list(flat[row * n:(row + 1) * n]) for row in range(m)]


class BankedInterpreterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def compile(self, x=None, w=None, name="artifact"):
        x = x if x is not None else [[-7.0, 2.0], [3.0, -1.0]]
        w = w if w is not None else [[-2.0, 5.0], [3.0, -4.0]]
        artifact = self.root / name
        banked.compile_matmul(x, w, artifact)
        program = json.loads((artifact / "program.json").read_text())
        return artifact, program

    def test_independent_known_integer_values(self):
        artifact, program = self.compile()
        self.assertEqual(banked.interpret_program(program), [[20.0, -43.0], [-9.0, 19.0]])
        result = banked.execute_functional(artifact)
        self.assertEqual(result["outputs"], [[20.0, -43.0], [-9.0, 19.0]])
        self.assertFalse(result["native_arithmetic"])
        self.assertEqual(result["functional_evidence"], "instruction_interpreter_on_emitted_memory_bytes")

    def test_ragged_16_lanes_and_reduction_over_128(self):
        for k in (1, 15, 16, 17, 128, 129, 257):
            with self.subTest(k=k):
                x = [[((row * 7 + index * 3) % 13 - 6) / 8 for index in range(k)] for row in range(3)]
                w = [[((index * 5 + col * 7) % 11 - 5) / 16 for col in range(17)] for index in range(k)]
                artifact, program = self.compile(x, w, "ragged-" + str(k))
                self.assertEqual(banked.interpret_program(program), oracle(x, w))
                self.assertEqual(len(program["outputs"]), 6)
                self.assertEqual([tile["valid_lanes"] for tile in program["outputs"]], [16, 1] * 3)
                for tile in program["outputs"]:
                    self.assertEqual(tile["address"] % 32, 0)
                trace = (artifact / "kernel.trace").read_text().splitlines()
                self.assertEqual(len(trace), 2 * len(program["instructions"]))
                self.assertTrue(all(line.startswith("PIM_BARRIER ") for line in trace[1::2]))

    def test_fp16_rounding_fp32_accumulator_and_half_store(self):
        x = [[1 + 2**-11, 1 + 2**-10, -1.0], [1 + 3 * 2**-11, -1 + 2**-10, 0.0]]
        w = [[1.0, 0.0], [1 + 2**-10, 1 + 2**-10], [1.0, 0.0]]
        _, program = self.compile(x, w)
        self.assertEqual(banked.interpret_program(program), oracle(x, w))
        # Input rounding happens before packing: exactly halfway above 1 rounds to 1.
        input_region = next(region for region in program["memory"] if region["role"] == "input")
        packed = base64.b64decode(input_region["data"])
        self.assertEqual(struct.unpack("<16e", packed[:32]), (1.0,) * 16)

    def test_reduction_order_cancellation(self):
        # FP16 operands produce a large FP32 sum; the unit increment disappears
        # before cancellation. Exact-real accumulation would yield 1 instead.
        x = [[4096.0, 1.0, -4096.0]]
        w = [[4096.0], [1.0], [4096.0]]
        _, program = self.compile(x, w)
        self.assertEqual(banked.interpret_program(program), [[0.0]])
        self.assertEqual(oracle(x, w), [[0.0]])

    def test_fp16_store_overflow_is_not_silent_infinity(self):
        x, w = [[65504.0]], [[2.0]]
        _, program = self.compile(x, w)
        with self.assertRaisesRegex(ValueError, "overflow"):
            banked.interpret_program(program)
        with self.assertRaises(ValueError):
            oracle(x, w)

    def test_half_subnormal_and_signed_zero(self):
        x, w = [[2**-24, -0.0]], [[1.0, 0.0], [0.0, 1.0]]
        _, program = self.compile(x, w)
        self.assertEqual(banked.interpret_program(program), [[2**-24, 0.0]])
        self.assertEqual(banked.interpret_program(program), oracle(x, w))

    def test_memory_bytes_determine_results_without_original_tensors(self):
        _, program = self.compile([[2.0]], [[3.0]])
        weight_region = next(region for region in program["memory"] if region["role"] == "weights")
        weight_region["data"] = base64.b64encode(struct.pack("<16e", 4.0, *([0.0] * 15))).decode()
        self.assertEqual(banked.interpret_program(program), [[8.0]])

    def test_crossbank_instruction_and_unallocated_address_rejected(self):
        _, program = self.compile()
        for mutation in ("crossbank", "unallocated", "unaligned"):
            modified = copy.deepcopy(program)
            instruction = next(item for item in modified["instructions"] if item["op"] == "LOAD16")
            if mutation == "crossbank":
                instruction["bank"] = 1
            elif mutation == "unallocated":
                instruction["address"] = banked.BANK_BYTES - 32
            else:
                instruction["address"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                banked.interpret_program(modified)

    def test_bank_crossing_memory_region_rejected(self):
        _, program = self.compile()
        program["memory"][0]["address"] = banked.BANK_BYTES - 32
        with self.assertRaisesRegex(ValueError, "bank boundary"):
            banked.interpret_program(program)

    def test_memory_overlap_bad_length_base64_and_nonfinite_rejected(self):
        _, program = self.compile()
        cases = []
        modified = copy.deepcopy(program)
        modified["memory"].append(copy.deepcopy(modified["memory"][0]))
        cases.append(modified)
        modified = copy.deepcopy(program)
        modified["memory"][0]["length"] += 32
        cases.append(modified)
        modified = copy.deepcopy(program)
        modified["memory"][0]["data"] = "!!!!"
        cases.append(modified)
        modified = copy.deepcopy(program)
        raw = base64.b64decode(modified["memory"][0]["data"])
        modified["memory"][0]["data"] = base64.b64encode(struct.pack("<16e", *([float("inf")] * 16)) + raw[32:]).decode()
        cases.append(modified)
        for modified in cases:
            with self.assertRaises(ValueError):
                banked.interpret_program(modified)

    def test_writeback_alias_and_duplicate_store_rejected(self):
        _, program = self.compile()
        aliased = copy.deepcopy(program)
        store = next(item for item in aliased["instructions"] if item["op"] == "STORE16")
        store["address"] = next(region["address"] for region in aliased["memory"] if region["role"] == "input")
        with self.assertRaisesRegex(ValueError, "aliases"):
            banked.interpret_program(aliased)
        duplicate = copy.deepcopy(program)
        position = next(index for index, item in enumerate(duplicate["instructions"]) if item["op"] == "STORE16")
        duplicate["instructions"].insert(position + 1, copy.deepcopy(duplicate["instructions"][position]))
        with self.assertRaisesRegex(ValueError, "aliases"):
            banked.interpret_program(duplicate)

    def test_missing_reset_load_and_store_rejected(self):
        _, program = self.compile()
        for op in ("RESET", "LOAD16", "STORE16"):
            modified = copy.deepcopy(program)
            position = next(index for index, item in enumerate(modified["instructions"]) if item["op"] == op)
            modified["instructions"].pop(position)
            with self.subTest(op=op), self.assertRaises(ValueError):
                banked.interpret_program(modified)

    def test_reset_required_when_bank_is_reused(self):
        x, w = [[float(index % 7 + 1)] for index in range(1025)], [[1.0]]
        _, program = self.compile(x, w)
        self.assertEqual(banked.interpret_program(program), x)
        resets = [index for index, item in enumerate(program["instructions"]) if item["op"] == "RESET" and item["bank"] == 0]
        self.assertEqual(len(resets), 2)
        program["instructions"].pop(resets[-1])
        try:
            changed = banked.interpret_program(program)
        except ValueError:
            return
        self.assertNotEqual(changed, x)

    def test_output_mapping_missing_duplicate_and_alias_detected(self):
        _, program = self.compile()
        missing = copy.deepcopy(program)
        missing["outputs"].pop()
        with self.assertRaises(ValueError):
            banked.interpret_program(missing)
        duplicate = copy.deepcopy(program)
        duplicate["outputs"].append(copy.deepcopy(duplicate["outputs"][0]))
        with self.assertRaises(ValueError):
            banked.interpret_program(duplicate)
        aliased = copy.deepcopy(program)
        aliased["outputs"][1]["address"] = aliased["outputs"][0]["address"]
        try:
            actual = banked.interpret_program(aliased)
        except ValueError:
            return
        self.assertNotEqual(actual, [[20.0, -43.0], [-9.0, 19.0]])

    def test_manifest_detects_program_and_trace_tampering(self):
        for name in ("program.json", "kernel.trace"):
            artifact, _ = self.compile(name=name + "-artifact")
            with (artifact / name).open("a") as stream:
                stream.write("\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hash mismatch"):
                banked.execute_functional(artifact)

    def test_wrong_profile_and_manifest_files_rejected(self):
        artifact, program = self.compile()
        program["profile"] = "unrecognized"
        with self.assertRaises(ValueError):
            banked.interpret_program(program)
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["../../foreign"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            banked.execute_functional(artifact)

    def test_no_overwrite_or_creation_on_invalid_matrix(self):
        artifact, _ = self.compile()
        original = (artifact / "program.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "exists"):
            banked.compile_matmul([[1.0]], [[1.0]], artifact)
        self.assertEqual((artifact / "program.json").read_bytes(), original)
        invalid = self.root / "bad"
        for x, w in (([], [[1.0]]), ([[1.0, 2.0], [3.0]], [[1.0]]), ([[float("nan")]], [[1.0]]),
                     ([[1.0]], [[1.0], [2.0]]), ([[True]], [[1.0]])):
            with self.assertRaises(ValueError):
                banked.compile_matmul(x, w, invalid)
            self.assertFalse(invalid.exists())

    def test_gemv_entry_and_deterministic_artifacts(self):
        first, second = self.root / "first", self.root / "second"
        banked.compile_gemv([2.0, -1.0], [[3.0], [4.0]], first)
        banked.compile_gemv([2.0, -1.0], [[3.0], [4.0]], second)
        self.assertEqual(banked.execute_functional(first)["outputs"], [[2.0]])
        for name in ("program.json", "kernel.trace", "manifest.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
