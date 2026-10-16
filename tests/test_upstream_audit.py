"""Record defects in pinned upstream source; passing is not numerical validation.

These assertions intentionally describe the observed problematic behavior. A
future upstream fix must update the reviewed revision, hashes, and this audit.
"""

import copy
import importlib.util
from math import prod
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pim_foundry.backend import PROJECT_ROOT, SOURCE_HASHES, verify_upstream


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def addresses(commands, opcode):
    return [int(command.split()[1], 16) for command in commands
            if command.split()[0] == opcode]


@unittest.skipUnless(
    (PROJECT_ROOT / "upstream/manifest.json").is_file()
    and all((PROJECT_ROOT / "upstream/DCC" / name).is_file() for name in SOURCE_HASHES),
    "optional author DCC generator sources are not installed",
)
class RecordedUpstreamDefectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verify both pinned files before importing either one. Bind the verified
        # dependency explicitly instead of accepting an ambient module/path.
        verify_upstream()
        source = PROJECT_ROOT / "upstream/DCC/src"
        cls.configs = load_source("dcc_audit_configs", source / "generate_configs.py")
        with patch.dict(sys.modules, {"generate_configs": cls.configs}):
            cls.generator = load_source(
                "dcc_audit_gemv", source / "hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py"
            )

    def test_recorded_reduction_defect_above_128_omits_later_chunks(self):
        for local_k in (128, 129, 256, 2048):
            with self.subTest(local_k=local_k):
                tiling = [[1, 1, 1], [1, 1, local_k], [1, 1, 16]]
                commands = self.generator.gen_half_bank_gemv_cms(
                    tiling, 4096, 0, 8192, [0], 0, 0
                )
                self.assertEqual(len(addresses(commands, "PIM_MAC_OP1")), 8)
                self.assertEqual(
                    addresses(commands, "PIM_LD_OP1"),
                    [4096 + 2 * i for i in range(8)],
                )

    def test_recorded_multihead_defect_writes_same_output_address(self):
        tiling = [[1, 1, 2], [1, 1, 16], [1, 1, 16]]
        commands = self.generator.GEMV(tiling, 1024, 0, 2048, 0)
        self.assertEqual(addresses(commands, "PIM_WB_ACC"), [2048, 2048])
        # The return phase nevertheless emits two distinct output addresses.
        self.assertEqual(addresses(commands, "ST")[-2:], [2048, 2080])

    def test_recorded_alignment_defect_exceeds_resources_for_search_results(self):
        configs = self.configs
        configs.current_sol, configs.all_sol = [], []
        configs.generate_search_space_3D([9, 400, 4], (3, 16, 64), {})
        self.assertEqual(len(configs.all_sol), 47)
        overflow = []
        for candidate in configs.all_sol:
            original = [list(axis) for axis in candidate]
            aligned = configs.match_alignment(copy.deepcopy(original))
            if prod(axis[0] for axis in aligned) > 16 or prod(
                axis[1] for axis in aligned
            ) > 64:
                overflow.append((original, aligned))
        self.assertEqual(len(overflow), 15)
        self.assertEqual(overflow[0], (
            [[1, 5, 2], [5, 5, 16], [2, 2, 1]],
            [[1, 8, 2], [8, 8, 16], [2, 2, 1]],
        ))

    def test_recorded_capacity_defect_continues_when_one_matrix_cannot_fit(self):
        generator = self.generator
        tiling = [[1, 1, 1], [1, 1, 4096], [1, 1, 4096]]
        self.assertEqual(sum(generator.total_size(tiling, 1)[1:]), 33570816)
        self.assertEqual(generator.HBM_GS["ba"], 16777216)
        # Observe allocation control flow without generating an oversized trace
        # or implying that the resulting command stream would execute legally.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capacity.trace"
            with patch.object(generator, "GEMV", return_value=[]) as gemv, \
                    patch.object(generator, "gen_yaml"):
                generator.run_GEMV(copy.deepcopy(tiling), str(output))
            self.assertEqual(gemv.call_count, 4)
            self.assertEqual(output.read_text(), "")

    def test_recorded_argument_order_defect_swaps_declared_operand_bases(self):
        generator = self.generator
        tiling = [[1, 1, 1], [1, 1, 16], [1, 1, 16]]
        self.assertEqual(generator.total_size(tiling, 1), (1, 32, 512, 32))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "operands.trace"
            with patch.object(generator, "GEMV", return_value=[]) as gemv, \
                    patch.object(generator, "gen_yaml"):
                generator.run_GEMV(copy.deepcopy(tiling), str(output))
        # run_GEMV declares matrix at 0 and vector at 512; GEMV's signature is
        # (T, Vec_addr, Mat_addr, Ret_addr, itr), but receives these positions.
        self.assertEqual(gemv.call_count, 2)
        for invocation in gemv.call_args_list:
            self.assertEqual(invocation.args, (tiling, 0, 512, 544, 0))

    def test_recorded_iteration_defect_ignores_external_iteration_argument(self):
        tiling = [[1, 1, 1], [1, 1, 16], [1, 1, 16]]
        first = self.generator.GEMV(copy.deepcopy(tiling), 4096, 0, 8192, 0)
        later = self.generator.GEMV(copy.deepcopy(tiling), 4096, 0, 8192, 17)
        self.assertTrue(first)
        self.assertEqual(first, later)


if __name__ == "__main__":
    unittest.main()
