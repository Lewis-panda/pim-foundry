"""Shape guards, source pinning, and actual upstream-adapter integration tests.

These tests exercise generated command artifacts.  They do not verify a matrix
product, run a timing simulator, or establish any hardware performance result.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pim_foundry.backend import (
    PROJECT_ROOT,
    REVISION,
    SOURCE_HASHES,
    compile_trace,
    enumerate_candidates,
    verify_upstream,
)
from pim_foundry.model import Gemv, inspect_tiling
from pim_foundry.trace import analyze_trace


class GemvContractTests(unittest.TestCase):
    def test_default_dimensions_preserve_combined_batch_and_heads(self):
        self.assertEqual(Gemv().dimensions, (16, 64, 256))
        self.assertEqual(Gemv(2, 8, 128, 512).dimensions, (16, 128, 512))

    def test_non_integer_nonpositive_and_oversized_dimensions_are_rejected(self):
        for field in ("batch", "nhead", "dhead", "seq_len"):
            for value in (True, False, 0, -1, 1.0, "16", None, 65537):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(Gemv(), **{field: value})

    def test_combined_head_enumeration_bound(self):
        with self.assertRaisesRegex(ValueError, "batch\\*nhead"):
            Gemv(batch=65536, nhead=2)

    def test_malformed_tiling_is_rejected(self):
        malformed = [
            None,
            [],
            [[1, 1, 1]],
            ((1, 1, 1), (1, 1, 64), (1, 1, 256)),
            [[1, 16], [1, 1, 64], [1, 1, 256]],
            [[True, 16, 1], [1, 1, 64], [1, 1, 256]],
            [[1, 16, 0], [1, 1, 64], [1, 1, 256]],
            [[1, 16, -1], [1, 1, 64], [1, 1, 256]],
            [[1, 16, 1.0], [1, 1, 64], [1, 1, 256]],
        ]
        for tiling in malformed:
            with self.subTest(tiling=tiling):
                with self.assertRaisesRegex(ValueError, "tiling"):
                    inspect_tiling(Gemv(), tiling)

    def test_known_unsupported_mappings_are_rejected(self):
        cases = [
            (
                "local heads",
                Gemv(),
                [[1, 1, 16], [1, 1, 64], [1, 1, 256]],
            ),
            (
                "upstream reduction-loop limit",
                Gemv(dhead=256),
                [[1, 16, 1], [1, 1, 256], [1, 1, 256]],
            ),
            (
                "local input extent",
                Gemv(dhead=17),
                [[1, 16, 1], [1, 1, 17], [1, 1, 256]],
            ),
            (
                "local input extent",
                Gemv(dhead=8),
                [[1, 16, 1], [1, 1, 8], [1, 1, 256]],
            ),
            (
                "local output extent",
                Gemv(seq_len=255),
                [[1, 16, 1], [1, 1, 64], [1, 1, 255]],
            ),
            (
                "multi-iteration spilling",
                Gemv(nhead=1, dhead=128, seq_len=65536),
                [[1, 1, 1], [1, 1, 128], [1, 1, 65536]],
            ),
            (
                "exact coverage",
                Gemv(),
                [[1, 16, 1], [1, 1, 64], [1, 1, 128]],
            ),
            (
                "powers of two",
                Gemv(nhead=3),
                [[3, 1, 1], [1, 1, 64], [1, 1, 256]],
            ),
            (
                "fixed 16 groups / 64 cores",
                Gemv(nhead=32),
                [[32, 1, 1], [1, 1, 64], [1, 1, 256]],
            ),
            (
                "fixed 16 groups / 64 cores",
                Gemv(nhead=128),
                [[1, 128, 1], [1, 1, 64], [1, 1, 256]],
            ),
            (
                "250000-command generation budget",
                Gemv(nhead=1024, dhead=128, seq_len=512),
                [[16, 64, 1], [1, 1, 128], [1, 1, 512]],
            ),
        ]
        for expected_reason, workload, tiling in cases:
            with self.subTest(reason=expected_reason):
                result = inspect_tiling(workload, tiling)
                self.assertFalse(result["eligible_for_trace_experiment"])
                self.assertIn(expected_reason, "; ".join(result["rejection_reasons"]))

    def test_exact_single_head_mapping_passes_conservative_guards(self):
        result = inspect_tiling(Gemv(), [[1, 16, 1], [1, 1, 64], [1, 1, 256]])
        self.assertTrue(result["eligible_for_trace_experiment"])
        self.assertEqual(result["rejection_reasons"], [])
        self.assertEqual(result["coverage"], [16, 64, 256])


class SourcePinningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        manifest_target = self.root / "upstream/manifest.json"
        manifest_target.parent.mkdir(parents=True)
        manifest_target.write_text(json.dumps({
            "repository": "synthetic-unit-fixture", "revision": REVISION,
        }))
        fixture_hashes = {}
        for name in SOURCE_HASHES:
            target = self.root / "upstream/DCC" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = ("# Synthetic source-pinning fixture: " + name + "\n").encode()
            target.write_bytes(content)
            fixture_hashes[name] = hashlib.sha256(content).hexdigest()
        # The hash guard is first-party code. Test it without redistributing or
        # importing author code; actual author integration is a separate class.
        expected = patch.dict(SOURCE_HASHES, fixture_hashes, clear=True)
        expected.start()
        self.addCleanup(expected.stop)

    def test_pinned_sources_verify(self):
        result = verify_upstream(self.root)
        self.assertEqual(result["revision"], REVISION)
        self.assertEqual(result["sha256"], SOURCE_HASHES)

    def test_each_source_tamper_is_rejected_before_execution(self):
        for name in SOURCE_HASHES:
            with self.subTest(source=name):
                target = self.root / "upstream/DCC" / name
                original = target.read_bytes()
                target.write_bytes(original + b"\n# changed content\n")
                with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                    verify_upstream(self.root)
                target.write_bytes(original)

    def test_revision_tamper_is_rejected(self):
        target = self.root / "upstream/manifest.json"
        manifest = json.loads(target.read_text())
        manifest["revision"] = "0" * 40
        target.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "revision differs"):
            verify_upstream(self.root)


@unittest.skipUnless(
    (PROJECT_ROOT / "upstream/manifest.json").is_file()
    and all((PROJECT_ROOT / "upstream/DCC" / name).is_file() for name in SOURCE_HASHES),
    "optional author DCC generator sources are not installed",
)
class BackendIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_default_enumeration_is_deterministic_and_has_eligible_candidates(self):
        first = enumerate_candidates(Gemv())
        second = enumerate_candidates(Gemv())
        self.assertEqual(first, second)
        self.assertGreater(first["eligible_count"], 0)
        self.assertGreater(first["candidate_count"], first["eligible_count"])
        self.assertEqual(
            [candidate["id"] for candidate in first["candidates"]],
            list(range(first["candidate_count"])),
        )
        self.assertEqual(
            len({json.dumps(candidate["tiling"]) for candidate in first["candidates"]}),
            first["candidate_count"],
        )
        for candidate in first["candidates"]:
            if candidate["eligible_for_trace_experiment"]:
                self.assertEqual(candidate["coverage"], [16, 64, 256])
                self.assertEqual(candidate["rejection_reasons"], [])

    def test_compile_generates_deterministic_trace_and_report(self):
        first_dir = self.root / "first"
        second_dir = self.root / "second"
        first = compile_trace(Gemv(), first_dir)
        second = compile_trace(Gemv(), second_dir)
        self.assertEqual(first, second)
        self.assertEqual((first_dir / "kernel.trace").read_bytes(),
                         (second_dir / "kernel.trace").read_bytes())
        self.assertEqual((first_dir / "report.json").read_bytes(),
                         (second_dir / "report.json").read_bytes())
        self.assertEqual(json.loads((first_dir / "report.json").read_text()), first)
        self.assertEqual(first["analysis"], analyze_trace(first_dir / "kernel.trace"))
        self.assertEqual(first["analysis"]["instruction_count"],
                         first["selected_candidate"]["predicted_commands"])
        self.assertEqual(first["status"], "trace_generated_not_numerically_verified")
        for claim in ("numerical_correctness", "simulator_execution", "hardware_execution"):
            self.assertIs(first["validation"][claim], False)
        self.assertIs(first["analysis"]["observed_only"], True)
        self.assertEqual(set(path.name for path in first_dir.iterdir()),
                         {"kernel.trace", "kernel.yaml", "report.json", "candidates.json"})

    def test_all_eligible_candidates_in_shape_sweep_match_preflight_counts(self):
        for index, workload in enumerate((Gemv(), Gemv(1, 16, 128, 512))):
            config = enumerate_candidates(workload)
            eligible = [candidate for candidate in config["candidates"]
                        if candidate["eligible_for_trace_experiment"]]
            self.assertTrue(eligible)
            for candidate in eligible:
                with self.subTest(workload=workload, candidate=candidate["id"]):
                    report = compile_trace(
                        workload, self.root / f"shape-{index}-candidate-{candidate['id']}",
                        candidate_id=candidate["id"],
                    )
                    self.assertEqual(report["analysis"]["instruction_count"],
                                     candidate["predicted_commands"])
                    self.assertFalse(report["validation"]["numerical_correctness"])

    def test_no_eligible_candidate_is_an_explicit_failure(self):
        workload = Gemv(1, 8, 64, 256)
        config = enumerate_candidates(workload)
        self.assertGreater(config["candidate_count"], 0)
        self.assertEqual(config["eligible_count"], 0)
        output = self.root / "ineligible"
        with self.assertRaisesRegex(ValueError, "no candidate passes"):
            compile_trace(workload, output)
        self.assertFalse(output.exists())

    def test_existing_output_contents_are_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "kernel.trace"
        sentinel.write_bytes(b"existing user result\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            compile_trace(Gemv(), output)
        self.assertEqual(sentinel.read_bytes(), b"existing user result\n")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_invalid_candidate_id_does_not_create_output(self):
        for identifier in (-1, 10**9, True, 0.0, "0"):
            with self.subTest(identifier=identifier):
                output = self.root / "invalid-id"
                with self.assertRaisesRegex(ValueError, "candidate ID is out of range"):
                    compile_trace(Gemv(), output, candidate_id=identifier)
                self.assertFalse(output.exists())

    def test_rejected_candidate_id_does_not_create_output(self):
        config = enumerate_candidates(Gemv())
        candidate = next(candidate for candidate in config["candidates"]
                         if not candidate["eligible_for_trace_experiment"])
        output = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "candidate rejected"):
            compile_trace(Gemv(), output, candidate_id=candidate["id"])
        self.assertFalse(output.exists())

    def test_yaml_uses_quoted_final_path_and_removes_shared_recorder(self):
        output = self.root / 'output space : # "quoted"'
        compile_trace(Gemv(), output)
        yaml = (output / "kernel.yaml").read_text()
        path_lines = [line for line in yaml.splitlines() if line.startswith("  path:")]
        self.assertEqual(len(path_lines), 1)
        path_scalar = path_lines[0].split(":", 1)[1].strip()
        self.assertTrue(path_scalar.startswith('"'))
        self.assertEqual(json.loads(path_scalar), str(output.resolve() / "kernel.trace"))
        self.assertNotIn(".pim-foundry-build-", yaml)
        self.assertNotIn("plugins:", yaml)
        self.assertNotIn("HBM3TraceRecorder", yaml)
        self.assertNotIn("./temp/log", yaml)
        self.assertIn("  AddrMapper:\n    impl: HBM3-PIM", yaml)


if __name__ == "__main__":
    unittest.main()
