#!/usr/bin/env python3
"""Run original NeuPIMs/NPU-only sources on one shared 512-request length task.

Requires the pinned, locally built source directories from the validation notes.
Uses existing direct executions as an optional audit reference; repeatability is
established by fresh application runs and frozen-engine replay.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_foundry import application, neupims_adapter as n


def descriptor(source, method, output):
    tracked=subprocess.check_output(['git','-C',str(source),'ls-files','-z']).decode().split('\0')
    bindings={rel:n.sha(source/rel) for rel in tracked if rel and (source/rel).is_file()}
    # Main Booksim is a gitlink; record every actually compiled dependency file.
    if method=='neupims':
        sub=source/'extern/booksim'
        for rel in subprocess.check_output(['git','-C',str(sub),'ls-files','-z']).decode().split('\0'):
            if rel and (sub/rel).is_file():bindings['extern/booksim/'+rel]=n.sha(sub/rel)
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=n.COMMITS[method]:raise ValueError('wrong pinned source revision')
    files={name:{'path':str(path),'sha256':n.sha(path)} for name,path in {
        'executable':source/'build/bin/Simulator','library':source/'build/lib/libdramsim3.so',
        'dependencies':source/'build/conaninfo.txt','flags':source/'build/src/CMakeFiles/Simulator.dir/flags.make'}.items()}
    configs={'hardware':'configs/systolic_ws_128x128_dev.json',
             'memory':'configs/memory_configs/'+('neupims' if method=='neupims' else 'dram')+'.json',
             'system':'configs/system_configs/'+('sub-batch-off' if method=='neupims' else 'npu-fusion')+'.json'}
    result={'schema':'pim-foundry.neupims-runtime.v1','method':method,'source_commit':commit,
            'source_root':str(source),'source_files':bindings,'files':files,'config_paths':configs,
            'build':{'compiler':'GCC13.3.0','build_frontend':'Conan1.66.0','source_modified':False,
                     'extra_flags':'-include cstdint -include limits -O3','assertions_enabled':True},'scope':n.SCOPE}
    n.save(output,result); n.runtime(output)
    return {'kind':'neupims-original-system/v1',**{key:json.loads((source/rel).read_text()) for key,rel in configs.items()}}


def prepare(root, out):
    out.mkdir(parents=True,exist_ok=False)
    source=root/'external/neupims'
    trace=source/'request-traces/clb/share-gpt2-bs512-ms7B-tp4-clb-0.csv'
    with trace.open(newline='') as f: rows=list(csv.DictReader(f))
    n.save(out/'lengths.json',[int(row['seq_len']) for row in rows])
    model=source/'configs/model_configs/gpt3-7B.json'
    app={'schema':application.APP_SCHEMA,'id':'neupims-gpt3-7b-sharegpt-512-layer-decode','adapter':n.ADAPTER,
         'assets':{'model_definition':application.asset(model,'model'),'request_lengths':application.asset(out/'lengths.json','inputs')},
         'parameters':{'model':'GPT3-7B','precision_bytes':2},'scenario':n.SCENARIO,
         'quality':{'kind':'not_evaluated','reason':'original artifact simulates shape-only operation timing'}}
    n.save(out/'application.json',app)
    for method,repo in [('neupims',source),('npu-only',source/'baselines/npu-only')]:
        runtime=out/(method+'-runtime.json');platform=descriptor(repo,method,runtime)
        experiment={'schema':application.EXPERIMENT_SCHEMA,'platform':platform,
                    'mapping':{'strategy':'pinned_channel_assignment','trace':str(trace),'sha256':n.sha(trace)},
                    'variant':{'method':method,'runtime_descriptor':str(runtime),'timeout_seconds':180}}
        n.save(out/(method+'-experiment.json'),experiment)
    rr = out/'matched-length-rr.csv'
    with rr.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seq_len','ch_idx'])
        writer.writerows((int(row['seq_len']),i%32) for i,row in enumerate(rows))
    experiment=json.loads((out/'neupims-experiment.json').read_text())
    experiment['mapping']={'strategy':'pinned_channel_assignment','trace':str(rr),'sha256':n.sha(rr)}
    n.save(out/'neupims-rr-experiment.json',experiment)
    return out


def evaluate(out, reference):
    methods=('neupims','npu-only','neupims-rr')
    def run(method):
        return application.run(out/'application.json',out/(method+'-experiment.json'),out/method)
    with ThreadPoolExecutor(max_workers=2) as pool:results=dict(zip(methods,pool.map(run,methods)))
    def replay(method):return application.replay(out/method,out/(method+'-replay'))
    with ThreadPoolExecutor(max_workers=2) as pool:replays=dict(zip(methods,pool.map(replay,methods)))
    if not all(result['passed'] for result in replays.values()):raise ValueError('frozen replay diverged')
    observations={}
    for method,directname in [('neupims','optimized-main'),('npu-only','matched-npu-only')]:
        direct=reference/directname/'stdout.log'
        counters=n.parse_output(direct)
        observed=results[method]['diagnostics']['native_counters']
        if counters!=observed:raise ValueError('direct/application cycle counters differ: '+method)
        observations[method]={'native_counters':observed,'direct_reference_sha256':n.sha(direct),
                              'direct_counter_agreement':True,'replay':replays[method],
                              'bundle_verified':application.verify(out/method)['passed']}
    observations['neupims-rr']={'native_counters':results['neupims-rr']['diagnostics']['native_counters'],
                               'replay':replays['neupims-rr'],
                               'bundle_verified':application.verify(out/'neupims-rr')['passed']}
    comparison=application.compare_results(results['neupims'],results['npu-only'],'latency')
    ablation=application.compare_results(results['neupims'],results['neupims-rr'],'latency')
    manifest,assets=application.load_manifest(out/'application.json')
    prepared=n.prepare(manifest,assets)
    negative_experiment=json.loads((out/'neupims-experiment.json').read_text())
    source=Path(json.loads((out/'neupims-runtime.json').read_text())['source_root'])
    author_rr=source/'request-traces/rr/share-gpt2-bs512-ms7B-tp4-rr-0.csv'
    negative_experiment['mapping']={'strategy':'pinned_channel_assignment','trace':str(author_rr),'sha256':n.sha(author_rr)}
    negative=n.inspect(prepared,negative_experiment)
    if negative['passed']:raise ValueError('different author RR workload was silently accepted')
    summary={'schema':'pim-foundry.neupims-application-study.v1','scope':n.SCOPE,'requests':512,
             'model':'GPT3-7B shape, one modeled layer, TP4','observations':observations,
             'comparison':comparison,'matched_lengths_channel_assignment_ablation':ablation,
             'negative_controls':{'author_default_rr_changes_workload':negative},
             'application_bundles':6,'successful_native_executions_including_direct_references':8,'task_quality_evaluated':False,'full_paper_reproduced':False}
    n.save(out/'summary.json',summary)
    (out/'report.html').write_text('<meta charset="utf-8"><title>NeuPIMs application validation</title><h1>NeuPIMs application validation</h1>'
        '<p>Same 512 request lengths; original one-layer, one-token decode cost models. Numerical and task quality were not evaluated.</p>'
        '<p>Each original source variant was executed through the application interface and frozen-engine replay. Hardware/config/source differences remain explicit.</p>'
        '<pre>'+__import__('html').escape(json.dumps(summary,indent=2))+'</pre>')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--run-prepared',action='store_true')
    args=parser.parse_args();out=args.out.resolve();root=args.project_root.resolve()
    if not args.run_prepared:prepare(root,out)
    if not args.prepare_only:print(json.dumps(evaluate(out,root/'runs/neupims-validation'),indent=2))
