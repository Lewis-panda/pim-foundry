"""Pinned PIMbench application execution with native reference checks.

Original paper applications remain unmodified. Model counters exclude host work;
matching the author's reference is deliberately distinct from task correctness.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import time

from .config import read_json

ADAPTER = 'pimbench-artifact/v1'
SOURCE_COMMIT = '9792b8fc281737e1c4aa8c31b9e68b54225b21c2'
BENCHMARKS = {
    'histogram': ('PIMbench/cpp-histogram/PIM/hist.cpp', 'PIMbench/cpp-histogram/PIM/hist.out'),
    'triangle-count': ('PIMbench/cpp-triangle-count/PIM/tc.cpp', 'PIMbench/cpp-triangle-count/PIM/tc.out'),
    'knn': ('PIMbench/cpp-knn/PIM/knn.cpp', 'PIMbench/cpp-knn/PIM/knn.out'),
}
ARCHITECTURES = {'bitserial': ('BitSerial', 'PIM_DEVICE_BITSIMD_V_AP'),
                 'fulcrum': ('Fulcrum', 'PIM_DEVICE_FULCRUM'),
                 'bank': ('Bank', 'PIM_DEVICE_BANK_LEVEL')}
STATELESS = {'kind': 'stateless_single_batch', 'initial_state': 'cold', 'state_policy': 'reset_each_run'}
QUALITY = {'kind': 'numerical_equivalence', 'reference': 'upstream_cpu_reference', 'atol': 0, 'rtol': 0}
BOUNDED_GEOMETRY = {'num_ranks': 1, 'num_bank_per_rank': 4, 'num_subarray_per_bank': 4,
                    'num_row_per_subarray': 1024, 'num_col_per_subarray': 8192}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _fields(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(label + ' requires exactly ' + ', '.join(sorted(fields)))


def _integer(value, lo, hi, label):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'{label} must be an integer in [{lo}, {hi}]')


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def bmp_info(path):
    """Independent BMP storage audit; do not confuse trailing ICC data with pixels."""
    b = Path(path).read_bytes()
    if len(b) < 54 or b[:2] != b'BM':
        raise ValueError('requires a Windows BMP header')
    size, offset, header = (struct.unpack_from('<I', b, p)[0] for p in (2, 10, 14))
    width, height = struct.unpack_from('<ii', b, 18)
    planes, bits, compression = struct.unpack_from('<HHI', b, 26)
    if header < 40 or offset < 14 + header or offset > 65535 or size != len(b):
        raise ValueError('invalid or unsupported BMP header or 16-bit upstream offset limit')
    if width <= 0 or height == 0 or planes != 1 or bits != 24 or compression != 0:
        raise ValueError('adapter requires uncompressed 24-bit BMP')
    pixels = width * abs(height)
    _integer(pixels, 1, 1_000_000, 'BMP pixels')
    stride = (width * 3 + 3) // 4 * 4
    end = offset + stride * abs(height)
    if end > len(b):
        raise ValueError('truncated BMP pixel storage')
    if (len(b) - offset) % 3:
        raise ValueError('upstream histogram reads beyond its vector when payload is not divisible by three')
    return {'width': width, 'height': height, 'pixel_count': pixels, 'pixel_offset': offset,
            'pixel_storage_bytes': stride * abs(height), 'row_padding_bytes': stride - width * 3,
            'trailing_bytes': len(b) - end, 'upstream_pixel_count': (len(b) - offset) // 3,
            'matches_upstream_pixel_domain': stride == width * 3 and end == len(b)}


def graph_info(path):
    edges = []
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) != 2 or any(not re.fullmatch(r'[0-9]+', x) for x in tokens):
            raise ValueError('triangle-count requires two nonnegative vertex IDs per line')
        u, v = map(int, tokens)
        if u == v:
            raise ValueError('triangle-count self loops are not supported')
        edges.append((u, v))
    vertices = {x for edge in edges for x in edge}
    _integer(len(vertices), 10, 4096, 'graph vertex count')
    _integer(len(edges), 1, 100000, 'graph edge count')
    # The artifact sizes bitmaps from vertex count rather than maximum ID.
    if max(vertices) >= ((len(vertices) + 31) // 32) * 32:
        raise ValueError('upstream bitmap cannot represent these sparse vertex IDs')
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    triangles = sum(1 for u in adj for v in adj[u] if v > u for w in adj[u] & adj[v] if w > v)
    return {'vertices': len(vertices), 'edge_rows': len(edges),
            'unique_undirected_edges': sum(map(len, adj.values())) // 2,
            'independent_triangle_count': triangles, 'reference': 'independent_set_intersection_u_lt_v_lt_w'}


def csv_info(path):
    rows = []
    with Path(path).open(newline='') as stream:
        for row in csv.reader(stream):
            if not row or any(not re.fullmatch(r'-?[0-9]+', x) for x in row):
                raise ValueError('KNN CSV requires exact integer cells without headers')
            values = list(map(int, row))
            if any(abs(x) > 10000 for x in values):
                raise ValueError('bounded KNN values must be in [-10000,10000]')
            rows.append(values)
    if not rows or len({len(r) for r in rows}) != 1 or not 2 <= len(rows[0]) <= 16:
        raise ValueError('KNN CSV must be rectangular with features and a final label column')
    _integer(len(rows), 1, 16384, 'KNN rows')
    return {'rows': len(rows), 'columns': len(rows[0]), 'final_column': 'label_included_in_upstream_distance'}


def prepare(manifest, assets):
    if manifest.get('adapter') != ADAPTER:
        raise ValueError('PIMbench adapter mismatch')
    params = manifest.get('parameters')
    if type(params) is not dict or params.get('benchmark') not in BENCHMARKS:
        raise ValueError('unsupported PIMbench benchmark')
    name = params['benchmark']
    _fields(params, {'benchmark', 'k'} if name == 'knn' else {'benchmark'}, 'PIMbench parameters')
    if manifest.get('scenario') != STATELESS or manifest.get('quality') != QUALITY:
        raise ValueError('PIMbench requires the cold single-run scenario and exact upstream-reference policy')
    names = {'benchmark_definition', 'training', 'queries'} if name == 'knn' else {'benchmark_definition', 'inputs'}
    if set(assets) != names:
        raise ValueError('PIMbench application assets differ from required inputs and benchmark_definition')
    for key, path in assets.items():
        spec = manifest['assets'][key]
        role = 'model' if key == 'benchmark_definition' else 'inputs'
        expected_format = 'cpp' if key == 'benchmark_definition' else ('csv' if name == 'knn' else 'bmp' if name == 'histogram' else 'edge_list')
        if spec.get('role') != role or spec.get('format') != expected_format or spec.get('sha256') != sha(path):
            raise ValueError('invalid asset role/hash: ' + key)
    if name == 'histogram':
        details = bmp_info(assets['inputs'])
    elif name == 'triangle-count':
        details = graph_info(assets['inputs'])
    else:
        train, query = csv_info(assets['training']), csv_info(assets['queries'])
        if train['columns'] != query['columns'] or query['rows'] > 256:
            raise ValueError('KNN columns must match and query rows must not exceed 256')
        _integer(params['k'], 1, min(256, train['rows']), 'k')
        details = {'training': train, 'queries': query, 'label_leakage_in_original_algorithm': True}
    return {'adapter': ADAPTER, 'benchmark': name, 'parameters': dict(params),
            'assets': {key: str(Path(path).resolve()) for key, path in assets.items()},
            'asset_hashes': {key: sha(path) for key, path in assets.items()}, 'input_details': details}


def validate_runtime(path):
    runtime = read_json(Path(path))
    _fields(runtime, {'schema', 'source_commit', 'source_root', 'source_files', 'binaries', 'build'}, 'PIMbench runtime')
    if runtime['schema'] != 'pim-foundry.pimbench-runtime.v1' or runtime['source_commit'] != SOURCE_COMMIT:
        raise ValueError('PIMbench runtime is not the audited IISWC release')
    root = Path(runtime['source_root']).resolve()
    bindings = runtime['source_files']
    minimum = {'PIMbench/util.h', 'libpimeval/src/pimStats.cpp', *(v[0] for v in BENCHMARKS.values()),
               *(f'configs/iiswc/PIMeval_{arch}_Rank1.cfg' for arch, _ in ARCHITECTURES.values())}
    if type(bindings) is not dict or not minimum <= set(bindings):
        raise ValueError('missing PIMbench source bindings')
    for name, expected in bindings.items():
        p = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or not p.resolve().is_relative_to(root)
                or not _hash(expected) or not p.is_file() or sha(p) != expected):
            raise ValueError('source binding mismatch: ' + name)
    if type(runtime['binaries']) is not dict or set(runtime['binaries']) != set(BENCHMARKS):
        raise ValueError('runtime must bind all three audited applications')
    for name, binding in runtime['binaries'].items():
        _fields(binding, {'path', 'sha256'}, 'binary binding')
        p = Path(binding['path'])
        if not p.is_absolute() or not p.is_file() or not _hash(binding['sha256']) or sha(p) != binding['sha256']:
            raise ValueError('binary binding mismatch: ' + name)
    return runtime


def validate_platform(platform):
    _fields(platform, {'kind', 'architecture', 'ranks', 'profile'}, 'PIMbench platform')
    if (platform['kind'] != 'pimeval-paper/v1' or platform['architecture'] not in ARCHITECTURES
            or type(platform['ranks']) is not int or platform['ranks'] != 1
            or platform['profile'] not in ('paper-rank1', 'bounded-functional')):
        raise ValueError('unsupported PIMbench platform')
    return dict(platform)


def config_text(runtime, platform):
    arch, target = ARCHITECTURES[platform['architecture']]
    if platform['profile'] == 'paper-rank1':
        return (Path(runtime['source_root']) / f'configs/iiswc/PIMeval_{arch}_Rank1.cfg').read_text()
    return ''.join(f'{k} = {v}\n' for k, v in BOUNDED_GEOMETRY.items()) + f'simulation_target = {target}\nmax_num_threads = 1\n'


def inspect(prepared, experiment):
    try:
        if experiment.get('mapping') != {'strategy': 'artifact_native'}:
            raise ValueError('PIMbench requires artifact_native mapping')
        platform = validate_platform(experiment.get('platform'))
        variant = experiment.get('variant')
        _fields(variant, {'runtime_descriptor', 'timeout_seconds'}, 'PIMbench variant')
        _integer(variant['timeout_seconds'], 1, 600, 'timeout_seconds')
        runtime = validate_runtime(variant['runtime_descriptor'])
        if runtime['source_files'][BENCHMARKS[prepared['benchmark']][0]] != prepared['asset_hashes']['benchmark_definition']:
            raise ValueError('runtime source differs from benchmark_definition application asset')
        if prepared['benchmark'] == 'triangle-count' and platform['profile'] == 'paper-rank1':
            raise ValueError('paper-rank1 triangle-count allocates 16 GiB of host scratch even for tiny graphs; use bounded-functional profile')
        text = config_text(runtime, platform)
        return {'passed': True, 'execution_mode': 'original_artifact_functional_simulation',
                'lowered_to_common_ir': False, 'platform': platform,
                'config_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'input_details': prepared['input_details'], 'native_reference': 'author CPU reference checks are enabled',
                'resource_scope': 'bounded inputs; paper-rank1 triangle-count blocked before launch'}
    except (ValueError, KeyError, OSError, TypeError) as error:
        return {'passed': False, 'reasons': [str(error)]}


def parse_stdout(text, benchmark, details):
    if re.search(r'Abort|PIM-Error|Wrong|verification failed|does not match|Error:', text):
        raise ValueError('native PIMbench output reports an error or reference mismatch')
    copy = re.findall(r'TOTAL ---------\s*:\s*(\d+) bytes\s+([0-9.]+) ms Estimated Runtime\s+([0-9.]+) mj Estimated Energy', text)
    command = re.findall(r'TOTAL ---------\s*:\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$', text, re.MULTILINE)
    if len(copy) != 1 or len(command) != 1:
        raise ValueError('native copy and command statistics must appear exactly once')
    result = {'copy_bytes': int(copy[0][0]), 'copy_runtime_ms': float(copy[0][1]),
              'copy_energy_mj': float(copy[0][2]), 'commands': int(command[0][0]),
              'command_runtime_ms': float(command[0][1]), 'command_energy_mj': float(command[0][2])}
    if result['commands'] <= 0 or any(not math.isfinite(x) or x < 0 for x in result.values()):
        raise ValueError('invalid native statistics')
    task = {'status': 'not_evaluated', 'reason': 'author-reference agreement does not independently validate application semantics'}
    outputs = {'native_author_reference': 'passed'}
    if benchmark == 'histogram':
        if not re.search(r'^Correct!$', text, re.MULTILINE):
            raise ValueError('histogram author reference success marker is missing')
        counts = re.findall(r'This file has (\d+) bytes of image data, (\d+) pixels', text)
        if len(counts) != 1 or int(counts[0][1]) != details['upstream_pixel_count']:
            raise ValueError('histogram native input domain differs from bound BMP')
        outputs['upstream_pixel_count'] = int(counts[0][1])
        if not details['matches_upstream_pixel_domain']:
            task = {'status': 'failed', 'reason': 'upstream histogram includes BMP row padding or trailing nonpixel bytes',
                    'expected_pixel_count': details['pixel_count'], 'observed_pixel_count': int(counts[0][1]),
                    'row_padding_bytes': details['row_padding_bytes'], 'trailing_bytes': details['trailing_bytes']}
    elif benchmark == 'triangle-count':
        matches = re.findall(r'PIM count \((\d+)\) matches CPU count\((\d+)\)', text)
        if len(matches) != 1 or matches[0][0] != matches[0][1]:
            raise ValueError('triangle-count author reference success marker is missing')
        value = int(matches[0][0]); expected = details['independent_triangle_count']
        outputs['triangle_count'] = value
        task = {'status': 'passed' if value == expected else 'failed', 'metric': 'exact_triangle_count',
                'observed': value, 'expected': expected, 'reference': details['reference']}
    else:
        if 'KNN was succesfully verified against host result' not in text:
            raise ValueError('KNN author reference success marker is missing')
        task['reason'] = 'original PIM and CPU reference both include the target label in distance; no leakage-free classification accuracy claim'
    return {'statistics': result, 'outputs': outputs,
            'quality': {'numerical': {'status': 'passed', 'reference': 'upstream_cpu_reference',
                                      'scope': 'native author reference comparison; final output arrays are not exported'}, 'task': task}}


def command_for(prepared, runtime, config, asset_paths):
    name = prepared['benchmark']
    cmd = [runtime['binaries'][name]['path'], '-c', str(config), '-v', 't']
    if name == 'knn':
        # The original CLI help swaps the two files: actual parser assigns -i
        # to query points and -t to training points.
        cmd += ['-i', str(asset_paths['queries']), '-t', str(asset_paths['training']), '-k', str(prepared['parameters']['k'])]
    else:
        cmd += ['-i', str(asset_paths['inputs'])]
    return cmd


def execute(prepared, experiment, output):
    diagnostics = inspect(prepared, experiment)
    if not diagnostics['passed']:
        raise ValueError('PIMbench preflight failed: ' + '; '.join(diagnostics['reasons']))
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    variant = experiment['variant']; runtime = validate_runtime(variant['runtime_descriptor'])
    descriptor_sha = sha(variant['runtime_descriptor'])
    save(output / 'runtime.json', runtime)
    config = output / 'platform.cfg'; config.write_text(config_text(runtime, experiment['platform']))
    assets = {}
    (output / 'inputs').mkdir()
    for name, path in prepared['assets'].items():
        dst = output / 'inputs' / (name + Path(path).suffix)
        shutil.copyfile(path, dst)
        if sha(dst) != prepared['asset_hashes'][name]:
            raise ValueError('application input changed while snapshotting')
        assets[name] = dst
    cmd = command_for(prepared, runtime, config, assets)
    env = os.environ.copy(); env.update({'LC_ALL': 'C', 'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'})
    env.pop('LD_PRELOAD', None); env.pop('LD_AUDIT', None)
    started = time.monotonic()
    with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(cmd, cwd=output, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            returncode = process.wait(timeout=variant['timeout_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
            save(output / 'command.json', {'argv': cmd, 'cwd': str(output), 'returncode': process.returncode, 'timeout': True})
            raise ValueError('native PIMbench timed out')
    save(output / 'command.json', {'argv': cmd, 'cwd': str(output), 'returncode': returncode,
                                  'elapsed_wall_seconds': time.monotonic() - started,
                                  'wall_time_is_not_modeled_hardware_latency': True,
                                  'environment_overrides': {'LC_ALL': 'C', 'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'}})
    if returncode != 0:
        raise ValueError(f'native PIMbench exited {returncode}; see preserved logs')
    if (validate_runtime(variant['runtime_descriptor']) != runtime or sha(variant['runtime_descriptor']) != descriptor_sha
            or sha(config) != diagnostics['config_sha256']):
        raise ValueError('runtime/config binding changed during execution')
    for name, path in assets.items():
        if sha(path) != prepared['asset_hashes'][name]:
            raise ValueError('native process modified an input asset')
    stdout = (output / 'stdout.log').read_text()
    stderr = (output / 'stderr.log').read_text()
    if stderr.strip():
        raise ValueError('unexpected native stderr; inspect preserved log')
    parsed = parse_stdout(stdout, prepared['benchmark'], prepared['input_details'])
    s = parsed['statistics']
    scope = 'upstream_pim_commands_and_modeled_host_device_copies_excluding_host_application_work'
    metrics = {name: {'value': value, 'unit': unit, 'scope': scope,
                      'aggregation': 'one_application_run_sum_of_printed_command_and_copy_totals',
                      'origin': 'original_PIMeval_IISWC2024_analytical_model_six_decimal_printed_statistics'}
               for name, value, unit in [('latency', round(s['copy_runtime_ms'] + s['command_runtime_ms'], 6), 'ms'),
                                        ('energy', round(s['copy_energy_mj'] + s['command_energy_mj'], 6), 'mJ')]}
    identity = {'source_commit': SOURCE_COMMIT, 'source_files': runtime['source_files'],
                'binary_sha256': runtime['binaries'][prepared['benchmark']]['sha256'],
                'runtime_descriptor_sha256': descriptor_sha, 'config_sha256': sha(config)}
    result = {'execution_mode': 'original_artifact_functional_simulation', 'metrics': metrics,
              'quality': parsed['quality'], 'outputs': parsed['outputs'], 'statistics': s,
              'platform_identity': {**experiment['platform'], 'config_sha256': sha(config)},
              'cost_contract': {'name': 'pimeval-iiswc2024/v1', 'scope': scope,
                                'source_commit': SOURCE_COMMIT, 'config_sha256': sha(config),
                                'modeled_profile': experiment['platform']['profile'],
                                'host_elapsed_time_excluded': True, 'model_default_memory_timing': True,
                                'printed_precision': 'six_decimal_ms_and_mJ', 'hardware_calibrated_here': False},
              'execution_identity': {**identity, 'benchmark': prepared['benchmark'], 'asset_hashes': prepared['asset_hashes']},
              'diagnostics': diagnostics, 'hardware_execution': False,
              'native_functional_execution': True, 'lowered_to_common_ir': False}
    save(output / 'result.json', result)
    return result
