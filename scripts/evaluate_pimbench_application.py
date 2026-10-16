#!/usr/bin/env python3
"""Execute the IISWC release applications, independent direct runs, and replays."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_foundry import application, pimbench_adapter as adapter


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build_runtime(source, out):
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != adapter.SOURCE_COMMIT or subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only'], text=True).strip():
        raise ValueError('requires clean original PIMbench IISWC v1.0.0 checkout')
    logs = out / 'build'; logs.mkdir()
    commands = []
    for subdir in ['libpimeval', *(str(Path(x[1]).parent) for x in adapter.BENCHMARKS.values())]:
        command = ['make', '-C', str(source / subdir), '-j4']
        p = subprocess.run(command, capture_output=True, text=True)
        name = subdir.replace('/', '-')
        (logs / (name + '.stdout.log')).write_text(p.stdout)
        (logs / (name + '.stderr.log')).write_text(p.stderr)
        commands.append({'argv': command, 'returncode': p.returncode})
        if p.returncode:
            raise ValueError('native build failed: ' + subdir)
    tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files'], text=True).splitlines()
    bindings = {name: adapter.sha(source / name) for name in tracked
                if Path(name).suffix in {'.cpp', '.h', '.hpp', '.cfg', '.ini'} or Path(name).name.startswith('Makefile')}
    descriptor = {'schema': 'pim-foundry.pimbench-runtime.v1', 'source_commit': commit,
                  'source_root': str(source), 'source_files': bindings,
                  'binaries': {name: {'path': str(source / paths[1]), 'sha256': adapter.sha(source / paths[1])}
                               for name, paths in adapter.BENCHMARKS.items()},
                  'build': {'commands': commands, 'source_modified': False,
                            'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
                            'flags': 'upstream make defaults: C++17 -Ofast; library -DNDEBUG',
                            'artifact_release': 'v1.0.0 IISWC 2024 submission code'}}
    save(out / 'runtime.json', descriptor)
    return descriptor


def make_bmp(path, width=256, height=256):
    pixels = bytes((x * 13 + y * 7 + channel * 61) % 256
                   for y in range(height) for x in range(width) for channel in range(3))
    header = b'BM' + struct.pack('<IHHI', 54 + len(pixels), 0, 0, 54)
    header += struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    path.write_bytes(header + pixels)


def specs(source, out):
    folder = out / 'specifications'; folder.mkdir()
    make_bmp(folder / 'image.bmp')
    edges = []
    for start in range(0, 1024, 8):
        edges.extend((u, v) for u in range(start, start + 8) for v in range(u + 1, start + 8))
        if start:
            edges.append((start - 1, start))
    (folder / 'graph.txt').write_text(''.join(f'{u} {v}\n' for u, v in edges))
    if adapter.graph_info(folder / 'graph.txt')['independent_triangle_count'] != 128 * 56:
        raise ValueError('graph oracle disagrees with 128 times C(8,3); bridges introduce no triangles')
    (folder / 'training.csv').write_text(''.join(f'{i % 128},{(i * 7) % 97},{i % 4}\n' for i in range(2048)))
    (folder / 'queries.csv').write_text(''.join(f'{(i * 5) % 128},{(i * 11) % 97},{i % 4}\n' for i in range(64)))
    result = {}
    for name in adapter.BENCHMARKS:
        assets = {'benchmark_definition': application.asset(source / adapter.BENCHMARKS[name][0], 'model', 'cpp')}
        if name == 'knn':
            assets.update({key: application.asset(folder / (key + '.csv'), 'inputs', 'csv') for key in ('training', 'queries')})
        else:
            assets['inputs'] = application.asset(folder / ('image.bmp' if name == 'histogram' else 'graph.txt'), 'inputs', 'bmp' if name == 'histogram' else 'edge_list')
        manifest = {'schema': application.APP_SCHEMA, 'id': 'pimbench-' + name, 'adapter': adapter.ADAPTER,
                    'assets': assets, 'parameters': {'benchmark': name, **({'k': 3} if name == 'knn' else {})},
                    'scenario': application.STATELESS, 'quality': adapter.QUALITY}
        save(folder / (name + '.json'), manifest); result[name] = folder / (name + '.json')
    sample = json.loads(result['histogram'].read_text())
    sample['id'] = 'pimbench-original-bmp-counterexample'
    sample['assets']['inputs'] = application.asset(source / 'PIMbench/cpp-histogram/histogram_datafiles/sample1.bmp', 'inputs', 'bmp')
    save(folder / 'histogram-original-bmp.json', sample)
    result['histogram-original-bmp'] = folder / 'histogram-original-bmp.json'
    save(folder / 'input-provenance.json', {
        'image': {'source': 'Foundry deterministic 256x256 BGR pattern, valid uncompressed 24-bit BMP', 'formula': '(13*x+7*y+61*channel)%256'},
        'graph': {'source': 'Foundry deterministic 1024-vertex graph: 128 disjoint K8 cliques joined by 127 bridges', 'expected_triangles': 7168},
        'knn': {'source': 'Foundry deterministic integer training/query fixture; checks original implementation semantics, not accuracy dataset', 'train_points': 2048, 'queries': 64, 'feature_columns': 2, 'label_column': 1},
        'original_bmp': {'source': 'unchanged author release sample1.bmp', 'expected_task_quality': 'failed: trailing ICC color-profile bytes treated as pixels'},
        'paper_scale_reproduction': False})
    return result


def direct_run(manifest_path, experiment, runtime, out):
    out.mkdir()
    manifest, assets = application.load_manifest(manifest_path)
    prepared = adapter.prepare(manifest, assets)
    config = out / 'platform.cfg'; config.write_text(adapter.config_text(runtime, experiment['platform']))
    # Construct the author's command independently of the adapter's launcher.
    command = [str(Path(runtime['source_root']) / adapter.BENCHMARKS[prepared['benchmark']][1]), '-c', str(config), '-v', 't']
    if prepared['benchmark'] == 'knn':
        command += ['-i', str(assets['queries']), '-t', str(assets['training']), '-k', str(manifest['parameters']['k'])]
    else:
        command += ['-i', str(assets['inputs'])]
    env = os.environ.copy(); env.update({'LC_ALL': 'C', 'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'})
    env.pop('LD_PRELOAD', None); env.pop('LD_AUDIT', None)
    p = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True, timeout=300)
    (out / 'stdout.log').write_text(p.stdout); (out / 'stderr.log').write_text(p.stderr)
    save(out / 'command.json', {'argv': command, 'returncode': p.returncode, 'cwd': str(out)})
    if p.returncode or p.stderr.strip():
        raise ValueError('independent direct run failed')
    parsed = adapter.parse_stdout(p.stdout, prepared['benchmark'], prepared['input_details'])
    save(out / 'observations.json', parsed)
    return parsed


def knn_counterexample(source, out, runtime):
    directory = out / 'knn-label-counterexample'; directory.mkdir()
    harness = directory / 'audit.cpp'
    harness.write_text('#define main pimbench_original_main\n#include "' + str(source / adapter.BENCHMARKS['knn'][0]) + '"\n#undef main\n'
        + '''int main(int argc, char **argv) {
  if (argc != 2 || !createDevice(argv[1])) return 1;
  std::vector<std::vector<int>> training{{0,9},{0,100}};
  for (int query_label : {0,100}) {
    std::vector<std::vector<int>> query{{1},{query_label}};
    std::vector<int> prediction(1);
    runKNN(2,1,2,1,training,query,prediction);
    bool author = knn_test(training,query,2,1,prediction);
    std::cout << "AUDIT query_feature=1 query_label=" << query_label << " prediction=" << prediction[0] << " author_reference_pass=" << author << std::endl;
  }
  pimShowStats();
  return 0;
}
''')
    command = ['g++', '-std=c++17', '-O2', '-I' + str(source / 'libpimeval/include'), str(harness),
               '-L' + str(source / 'libpimeval/lib'), '-lpimeval', '-pthread', '-o', str(directory / 'audit.out')]
    built = subprocess.run(command, capture_output=True, text=True)
    (directory / 'build.stdout.log').write_text(built.stdout); (directory / 'build.stderr.log').write_text(built.stderr)
    save(directory / 'build-command.json', {'argv': command, 'returncode': built.returncode, 'source_sha256': adapter.sha(harness)})
    if built.returncode:
        raise ValueError('KNN negative-control build failed')
    records = []
    for arch in adapter.ARCHITECTURES:
        config = directory / (arch + '.cfg')
        config.write_text(adapter.config_text(runtime, {'architecture': arch, 'profile': 'bounded-functional'}))
        command = [str(directory / 'audit.out'), str(config)]
        p = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
        (directory / (arch + '.stdout.log')).write_text(p.stdout); (directory / (arch + '.stderr.log')).write_text(p.stderr)
        import re
        observed = [(int(a), int(b), int(c)) for a, b, c in re.findall(r'AUDIT query_feature=1 query_label=(\d+) prediction=(\d+) author_reference_pass=(\d+)', p.stdout)]
        # With k=1, no voting or nearest-neighbor tie can confound this oracle.
        expected = [0, 0]  # |1-0|=1 < |1-9|=8, regardless of query label.
        found = p.returncode == 0 and observed == [(0, 0, 1), (100, 100, 1)]
        records.append({'architecture': arch, 'argv': command, 'returncode': p.returncode,
                        'native_observations': observed, 'feature_only_L1_reference': expected,
                        'query_label_invariance_passed': False, 'counterexample_reproduced': found,
                        'source_sha256': adapter.sha(source / adapter.BENCHMARKS['knn'][0]),
                        'binary_sha256': adapter.sha(directory / 'audit.out')})
        if not found:
            raise ValueError('negative control did not reproduce expected upstream bug')
    report = {'source_revision': adapter.SOURCE_COMMIT, 'scope': 'unchanged original runKNN and knn_test called by separate audit harness',
              'not_claimed_fixed_in_newer_versions': True, 'task_quality_status': 'failed', 'records': records}
    save(directory / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); parser.add_argument('--source', default=str(ROOT / 'external/pimeval-pimbench'))
    args = parser.parse_args(); source = Path(args.source).resolve(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    runtime = build_runtime(source, out); manifests = specs(source, out)
    rows = []; bundles = []
    for name in adapter.BENCHMARKS:
        for arch in adapter.ARCHITECTURES:
            label = name + '-' + arch
            experiment = {'schema': application.EXPERIMENT_SCHEMA,
                          'platform': {'kind': 'pimeval-paper/v1', 'architecture': arch, 'ranks': 1, 'profile': 'bounded-functional'},
                          'mapping': {'strategy': 'artifact_native'},
                          'variant': {'runtime_descriptor': str(out / 'runtime.json'), 'timeout_seconds': 300}}
            ep = out / 'specifications' / (label + '-experiment.json'); save(ep, experiment)
            result = application.run(manifests[name], ep, out / label)
            direct = direct_run(manifests[name], experiment, runtime, out / (label + '-direct'))
            checks = {key: result[key] == direct[key] for key in ('statistics', 'outputs', 'quality')}
            replay = application.replay(out / label, out / (label + '-replay'))
            if not all(checks.values()) or not replay['passed']:
                raise ValueError('direct/replay parity failed: ' + label)
            bundles += [str(out / label), str(out / (label + '-replay'))]
            rows.append({'benchmark': name, 'architecture': arch, 'profile': 'bounded-functional', 'status': result['status'],
                         'metrics': result['metrics'], 'quality': result['quality'], 'direct_parity': checks, 'replay': replay,
                         'task_sha256': result['task_sha256'], 'platform_sha256': result['platform_sha256']})
            print(label + ': direct parity and frozen replay passed', flush=True)
    experiment['platform'] = {'kind': 'pimeval-paper/v1', 'architecture': 'bitserial', 'ranks': 1, 'profile': 'bounded-functional'}
    ep = out / 'specifications/original-bmp-experiment.json'; save(ep, experiment)
    counter = application.run(manifests['histogram-original-bmp'], ep, out / 'histogram-original-bmp')
    if counter['status'] != 'quality_failed' or counter['quality']['numerical']['status'] != 'passed':
        raise ValueError('original BMP semantic quality rejection did not work')
    rejected = application.compare_results(counter, counter)
    if rejected['same_model_comparable']:
        raise ValueError('bad original BMP was admitted to comparison')
    replay = application.replay(out / 'histogram-original-bmp', out / 'histogram-original-bmp-replay')
    if not replay['passed']:
        raise ValueError('counterexample replay did not preserve failure')
    bundles += [str(out / 'histogram-original-bmp'), str(out / 'histogram-original-bmp-replay')]
    # The very same publication config must reject TC before its unconditional 16 GiB allocation.
    paper_experiment = json.loads(json.dumps(experiment))
    paper_experiment['platform']['profile'] = 'paper-rank1'
    guard_path = out / 'specifications/triangle-paper-resource-guard.json'; save(guard_path, paper_experiment)
    guard = application.inspect(manifests['triangle-count'], guard_path)
    if guard['diagnostics']['passed']:
        raise ValueError('TC paper-profile resource guard failed')
    audit = knn_counterexample(source, out, runtime)
    verified = [application.verify(path) for path in bundles]
    summary = {'schema': 'pim-foundry.pimbench-study.v1', 'paper': 'Architectural Modeling and Benchmarking for Digital DRAM PIM, IISWC 2024',
               'paper_url': 'https://www.cs.virginia.edu/venkat/papers/iiswc2024.pdf', 'source_commit': adapter.SOURCE_COMMIT,
               'source_release': 'v1.0.0 IISWC 2024 submission artifact; newer TACO2026/main revisions not evaluated',
               'source_modified': False, 'paper_scale_reproduction': False, 'native_application_runs': 29,
               'native_audit_harness_runs': 3, 'study_cases': rows, 'original_bmp_counterexample': {'result': counter, 'comparison_rejection': rejected, 'replay': replay},
               'triangle_resource_preflight': guard, 'knn_label_counterexample': audit,
               'bundles_verified': len(verified), 'all_bundle_integrity_checks_passed': all(x['passed'] for x in verified),
               'hardware_performance_claim': False, 'scope': 'functional application integration and scoped original model counters; no full-system speedup'}
    save(out / 'summary.json', summary)
    table = ''.join('<tr><td>' + html.escape(row['benchmark']) + '</td><td>' + row['architecture'] + '</td><td>' + row['status'] + '</td><td>' + str(row['metrics']['latency']['value']) + '</td><td>' + row['quality']['task']['status'] + '</td></tr>' for row in rows)
    (out / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>PIMbench application validation</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.6}td,th{padding:8px;border-bottom:1px solid #ccc}code{word-break:break-all}</style><h1>PIMbench application validation</h1><p>Unmodified IISWC 2024 release ' + adapter.SOURCE_COMMIT + '. Three complete application implementations × three DRAM PIM models; direct runs and frozen replays. Input scales and bounded 16 MiB geometry differ from publication performance experiments.</p><table><tr><th>Application</th><th>Architecture</th><th>Status</th><th>Model commands+copies (ms)</th><th>Task quality</th></tr>' + table + '</table><h2>Independent failures correctly retained</h2><p>Original BMP: author reference says Correct!, but 3,144 ICC profile bytes become 1,048 extra pixels. Application quality fails and comparison is rejected.</p><p>KNN: identical feature query with label 0 versus 100 predicts 0 versus 100; both author checks pass. Feature-only L1 oracle predicts 0 both times. Reproduced on all three models.</p><p>Triangle count: publication Rank1 profile reserves 16 GiB of host scratch regardless of graph size. Preflight rejects that configuration; bounded geometry runs the unchanged code and agrees with independent triangle enumeration.</p><p>Native model counters exclude CPU preprocessing, graph assembly, KNN voting, and reference verification. No end-to-end hardware speedup or publication figure reproduction is claimed. Newer PIMeval versions were not tested.</p><p><a href="summary.json">Machine-readable results and provenance</a></p>')
    save(out / 'manifest.json', {'files': {str(p.relative_to(out)): adapter.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps({'report': str(out / 'report.html'), 'bundles_verified': len(verified), 'native_application_runs': 29}, indent=2))

if __name__ == '__main__':
    main()
