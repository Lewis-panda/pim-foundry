"""Regression gates learned from actual PIMbench paper-artifact failures."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from pim_foundry import application, pimbench_adapter as adapter


def bmp(width=8, height=4, tail=b''):
    stride = (width * 3 + 3) // 4 * 4
    body = bytes((i % 256 for i in range(stride * height)))
    head = b'BM' + struct.pack('<IHHI', 54 + len(body) + len(tail), 0, 0, 54)
    head += struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0, len(body), 0, 0, 0, 0)
    return head + body + tail


def output(marker='Correct!', pixels=32):
    return f'''This file has {pixels*3} bytes of image data, {pixels} pixels
{marker}
Data Copy Stats:
 TOTAL --------- : 96 bytes 0.000001 ms Estimated Runtime 0.000002 mj Estimated Energy
PIM Command Stats:
 TOTAL --------- : 1536 0.003000 0.004000
'''


class PimbenchAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, data):
        path = self.root / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return path

    def test_bmp_24bit_pixel_domain_and_trailing_profile(self):
        clean = adapter.bmp_info(self.write('clean.bmp', bmp()))
        self.assertEqual(clean['pixel_count'], 32)
        self.assertTrue(clean['matches_upstream_pixel_domain'])
        info = adapter.bmp_info(self.write('icc.bmp', bmp(tail=b'profileXYZab')))
        self.assertEqual(info['trailing_bytes'], 12)
        self.assertEqual(info['upstream_pixel_count'], 36)
        self.assertFalse(info['matches_upstream_pixel_domain'])
        observed = adapter.parse_stdout(output(pixels=36), 'histogram', info)
        self.assertEqual(observed['quality']['numerical']['status'], 'passed')
        self.assertEqual(observed['quality']['task']['status'], 'failed')

    def test_bmp_padding_is_not_pixels_even_if_count_divisible_by_three(self):
        info = adapter.bmp_info(self.write('padding.bmp', bmp(width=1, height=3)))
        self.assertEqual(info['row_padding_bytes'], 1)
        result = adapter.parse_stdout(output(pixels=4), 'histogram', info)
        self.assertEqual(result['quality']['task']['status'], 'failed')

    def test_bmp_rejects_truncation_invalid_header_and_unsafe_upstream_reads(self):
        for name, data in [('short', b'BM'), ('badmagic', b'xx' + bmp()[2:]),
                           ('partial', bmp()[:-1]), ('tail', bmp(tail=b'x'))]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter.bmp_info(self.write(name + '.bmp', data))

    def test_clean_histogram_only_claims_reference_agreement(self):
        info = adapter.bmp_info(self.write('clean.bmp', bmp()))
        parsed = adapter.parse_stdout(output(), 'histogram', info)
        self.assertEqual(parsed['quality']['task']['status'], 'not_evaluated')
        self.assertEqual(parsed['statistics']['commands'], 1536)

    def test_success_exit_cannot_replace_verification_marker(self):
        info = adapter.bmp_info(self.write('clean.bmp', bmp()))
        for marker in ['', 'Wrong PIM answer for red', 'Correct!\nAbort', 'Correct!\nPIM-Error: bad']:
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                adapter.parse_stdout(output(marker), 'histogram', info)

    def test_duplicate_stat_blocks_rejected(self):
        info = adapter.bmp_info(self.write('clean.bmp', bmp()))
        with self.assertRaises(ValueError):
            adapter.parse_stdout(output() + output(), 'histogram', info)

    def test_graph_independent_exact_reference(self):
        text = '0 1\n0 2\n1 2\n' + ''.join(f'{i} {i+1}\n' for i in range(2, 11))
        info = adapter.graph_info(self.write('graph.txt', text))
        self.assertEqual(info['independent_triangle_count'], 1)
        passed = adapter.parse_stdout(output('PIM count (1) matches CPU count(1)'), 'triangle-count', info)
        self.assertEqual(passed['quality']['task']['status'], 'passed')
        wrong = adapter.parse_stdout(output('PIM count (2) matches CPU count(2)'), 'triangle-count', info)
        self.assertEqual(wrong['quality']['task']['status'], 'failed')

    def test_graph_rejects_sparse_bitmap_ids_and_too_few_vertices(self):
        for text in ['0 1\n', ''.join(f'{100+i} {101+i}\n' for i in range(11)), '0 0\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                adapter.graph_info(self.write('graph.txt', text))

    def test_knn_file_roles_match_original_parser_not_help_text(self):
        prepared = {'benchmark': 'knn', 'parameters': {'k': 1}}
        runtime = {'binaries': {'knn': {'path': '/trusted/knn.out'}}}
        cmd = adapter.command_for(prepared, runtime, '/cfg', {'queries': '/queries.csv', 'training': '/training.csv'})
        self.assertEqual(cmd[cmd.index('-i') + 1], '/queries.csv')
        self.assertEqual(cmd[cmd.index('-t') + 1], '/training.csv')

    def test_knn_author_reference_does_not_establish_task_accuracy(self):
        info = {'label_leakage_in_original_algorithm': True}
        observed = adapter.parse_stdout(output('KNN was succesfully verified against host result'), 'knn', info)
        self.assertEqual(observed['quality']['numerical']['status'], 'passed')
        self.assertEqual(observed['quality']['task']['status'], 'not_evaluated')
        self.assertIn('label', observed['quality']['task']['reason'])

    def test_csv_requires_bounded_rectangular_numeric_inputs(self):
        for text in ['x,y\n', '1,2\n3\n', '10001,0\n', '1.5,2\n', '']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                adapter.csv_info(self.write('data.csv', text))

    def test_paper_triangle_allocation_rejected_without_launch(self):
        prepared = {'benchmark': 'triangle-count', 'asset_hashes': {'benchmark_definition': 'a' * 64}}
        runtime = {'source_files': {adapter.BENCHMARKS['triangle-count'][0]: 'a' * 64}}
        experiment = {'platform': {'kind': 'pimeval-paper/v1', 'architecture': 'bank', 'ranks': 1, 'profile': 'paper-rank1'},
                      'mapping': {'strategy': 'artifact_native'}, 'variant': {'runtime_descriptor': '/descriptor', 'timeout_seconds': 10}}
        with patch.object(adapter, 'validate_runtime', return_value=runtime), patch.object(adapter.subprocess, 'Popen') as launch:
            inspected = adapter.inspect(prepared, experiment)
        self.assertFalse(inspected['passed']); self.assertIn('16 GiB', inspected['reasons'][0]); launch.assert_not_called()

    def test_runtime_missing_config_hash_rejected(self):
        bindings = {'PIMbench/util.h': 'a'*64, 'libpimeval/src/pimStats.cpp': 'a'*64,
                    **{v[0]: 'a'*64 for v in adapter.BENCHMARKS.values()}}
        descriptor = {'schema': 'pim-foundry.pimbench-runtime.v1', 'source_commit': adapter.SOURCE_COMMIT,
                      'source_root': str(self.root), 'source_files': bindings, 'binaries': {}, 'build': {}}
        with self.assertRaisesRegex(ValueError, 'missing PIMbench source bindings'):
            adapter.validate_runtime(self.write('runtime.json', json.dumps(descriptor)))

    def test_manifest_binds_algorithm_revision_and_asset_formats(self):
        definition = self.write('hist.cpp', '// fixed source\n')
        image = self.write('image.bmp', bmp())
        manifest = {'adapter': adapter.ADAPTER, 'parameters': {'benchmark': 'histogram'},
                    'scenario': adapter.STATELESS, 'quality': adapter.QUALITY,
                    'assets': {'benchmark_definition': application.asset(definition, 'model', 'cpp'),
                               'inputs': application.asset(image, 'inputs', 'bmp')}}
        ready = adapter.prepare(manifest, {'benchmark_definition': definition, 'inputs': image})
        self.assertEqual(ready['asset_hashes']['benchmark_definition'], adapter.sha(definition))
        changed = copy.deepcopy(manifest); changed['assets']['benchmark_definition']['role'] = 'implementation'
        with self.assertRaises(ValueError):
            adapter.prepare(changed, {'benchmark_definition': definition, 'inputs': image})
        changed = copy.deepcopy(manifest); changed['assets']['inputs']['format'] = 'json'
        with self.assertRaises(ValueError):
            adapter.prepare(changed, {'benchmark_definition': definition, 'inputs': image})

    def test_platform_profiles_and_strict_types(self):
        platform = {'kind': 'pimeval-paper/v1', 'architecture': 'bank', 'ranks': 1, 'profile': 'bounded-functional'}
        self.assertEqual(adapter.validate_platform(platform), platform)
        for key, value in [('ranks', True), ('profile', 'paper-unbounded'), ('architecture', 'custom')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.validate_platform({**platform, key: value})


if __name__ == '__main__':
    unittest.main()
