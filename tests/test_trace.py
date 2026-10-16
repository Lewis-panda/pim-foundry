"""Tests for strict trace parsing and observations, without simulator claims."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pim_foundry.trace import (
    ADDRESS_LIMIT_BYTES,
    BANK_STRIDE_BYTES,
    CHANNEL_STRIDE_BYTES,
    OPCODES,
    analyze_trace,
)


class TraceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.trace"

    def analyze(self, content):
        if isinstance(content, str):
            content = content.encode("ascii")
        self.path.write_bytes(content)
        return analyze_trace(self.path)

    def test_mixed_trace_has_deterministic_counts_and_channel_observations(self):
        result = self.analyze(
            "LD 0x20\n"
            "PIM_ACC_RESET 0x01000000\n"
            "PIM_LD_OP1 0x40000002\n"
            "PIM_MAC_OP1 0x40000020\n"
            "PIM_WB_ACC 0x40000040\n"
            "ST 0x40000060\n"
            "PIM_BARRIER 0x3c0000000\n"
        )
        self.assertEqual(result["instruction_count"], 7)
        self.assertEqual(list(result["opcode_counts"]), list(OPCODES))
        self.assertEqual(result["opcode_counts"], dict.fromkeys(OPCODES, 1))
        self.assertEqual(result["host_command_count"], 2)
        self.assertEqual(result["barrier_count"], 1)
        self.assertEqual(result["min_address"], 32)
        self.assertEqual(result["max_address"], 15 * CHANNEL_STRIDE_BYTES)
        self.assertEqual(result["channels_touched"], [0, 1, 15])
        self.assertEqual(result["non_barrier_channels_touched"], [0, 1])
        self.assertEqual(result["unaligned_address_count"], 1)
        self.assertEqual(result["unaligned_address_counts_by_opcode"]["PIM_LD_OP1"], 1)
        self.assertIs(result["observed_only"], True)
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(analyze_trace(self.path), result)

    def test_geometry_matches_pinned_generator(self):
        # 32 B/column * 32 columns/row * 16384 rows/bank.
        self.assertEqual(BANK_STRIDE_BYTES, 32 * 32 * 2**14)
        # 4 banks/group * 4 groups/rank * 2 ranks/pch * 2 pch/channel.
        self.assertEqual(CHANNEL_STRIDE_BYTES, BANK_STRIDE_BYTES * 4 * 4 * 2 * 2)
        self.assertEqual(ADDRESS_LIMIT_BYTES, CHANNEL_STRIDE_BYTES * 16)

    def test_address_boundary_and_unaligned_barrier_are_observations(self):
        result = self.analyze("LD 0x0\nPIM_BARRIER 0x3ffffffff")
        self.assertEqual(result["min_address"], 0)
        self.assertEqual(result["max_address"], ADDRESS_LIMIT_BYTES - 1)
        self.assertEqual(result["channels_touched"], [0, 15])
        self.assertEqual(result["unaligned_address_count"], 1)
        self.assertEqual(result["unaligned_address_counts_by_opcode"]["PIM_BARRIER"], 1)

    def test_whitespace_crlf_and_uppercase_hex_digits(self):
        result = self.analyze(" \tLD\t0xABCD \r\nST 0x20")
        self.assertEqual(result["instruction_count"], 2)
        self.assertEqual(result["max_address"], 0xABCD)
        self.assertEqual(result["opcode_counts"]["PIM_MAC_OP1"], 0)

    def test_repeated_commands_are_counted_individually(self):
        result = self.analyze("PIM_MAC_OP1 0x20\n" * 3)
        self.assertEqual(result["instruction_count"], 3)
        self.assertEqual(result["opcode_counts"]["PIM_MAC_OP1"], 3)
        self.assertEqual(result["host_command_count"], 0)
        self.assertEqual(result["barrier_count"], 0)

    def test_empty_trace_has_line_number(self):
        with self.assertRaisesRegex(ValueError, "line 1: empty trace"):
            self.analyze(b"")

    def test_invalid_lines_have_correct_line_number(self):
        invalid_lines = [
            "",
            "   ",
            "# comment",
            "LD",
            "LD 0x20 extra",
            "LD 0x20 # comment",
            "UNKNOWN 0x20",
            "ld 0x20",
            "LD 32",
            "LD 0X20",
            "LD -0x20",
            "LD +0x20",
            "LD 0x-1",
            "LD 0x",
            "LD 0xg0",
            "LD 0x1_0",
            "LD 0x400000000",
        ]
        for line in invalid_lines:
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, "line 2:"):
                    self.analyze("LD 0x0\n" + line + "\n")

    def test_non_ascii_has_correct_line_number(self):
        with self.assertRaisesRegex(ValueError, "line 2:.*ASCII"):
            self.analyze(b"LD 0x0\nLD 0x\xff\n")

    def test_long_line_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "line 2:.*exceeds"):
            self.analyze("LD 0x0\nLD 0x" + "0" * 10000)

    def test_reader_uses_bounded_reads(self):
        class BoundedReader(io.BytesIO):
            def read(self, *args, **kwargs):
                raise AssertionError("whole-file reads are not allowed")

            def readline(self, size=-1):
                if not 0 < size <= 257:
                    raise AssertionError("line reads must have a fixed small bound")
                return super().readline(size)

        reader = BoundedReader(b"LD 0x20\n" * 10000)
        with patch.object(Path, "open", return_value=reader):
            result = analyze_trace(self.path)
        self.assertEqual(result["instruction_count"], 10000)
        self.assertEqual(result["opcode_counts"]["LD"], 10000)

    def test_missing_file_remains_io_error(self):
        with self.assertRaises(FileNotFoundError):
            analyze_trace(self.path)


if __name__ == "__main__":
    unittest.main()
