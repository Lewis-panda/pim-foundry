import copy
import json
from pathlib import Path
import tempfile
import unittest

from pim_foundry import neupims_adapter as n


class NeuPIMsTests(unittest.TestCase):
    def test_clock_counter_and_scope_are_not_core_or_stage_counter(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'stdout'
            p.write_text('Client total request cnt: 512\nSimulation Finished\nTotal cycle: 619227\nDRAM total cycles: 619233\nStage A : 87268 cycles\nStage B : 266239 cycles\nStage E : 265724 cycles\n')
            result = n.parse_output(p)
            self.assertEqual(result['latency_ns'], 619233)
            self.assertEqual(sum(row['cycles'] for row in result['stage_cycles']), 619231)
            self.assertEqual(n.parse_output(p, dram_frequency_mhz=500)['latency_ns'], 1238466)

    def test_npu_branch_uses_trace_parser_request_counter(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'stdout'
            p.write_text('parsed 512 lines from file requests.csv\nSimulation Finished\nDRAM total cycles: 5937045\n')
            self.assertEqual(n.parse_output(p)['dram_cycles'],5937045)

    def test_incomplete_or_ambiguous_native_output_fails(self):
        valid = 'Client total request cnt: 512\nSimulation Finished\nDRAM total cycles: 22\n'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'stdout'
            for text in (valid.replace('Simulation Finished', ''), valid.replace('512', '8'),
                         valid + 'DRAM total cycles: 23\n', valid.replace('22', '0')):
                p.write_text(text)
                with self.assertRaises(ValueError): n.parse_output(p)

    def test_application_requires_semantic_lengths_and_unknown_quality(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = {'model_name':'GPT3-7B','model_params_b':7,'model_vocab_size':50304,
                     'model_n_layer':1,'model_n_head':32,'model_n_embd':4096,'n_tp':4,'n_pp':1}
            n.save(d/'model.json', model); n.save(d/'lengths.json', [128]*512)
            assets = {'model_definition':d/'model.json', 'request_lengths':d/'lengths.json'}
            manifest = {'adapter':n.ADAPTER, 'parameters':{'model':'GPT3-7B','precision_bytes':2},
                        'scenario':dict(n.SCENARIO), 'quality':{'kind':'not_evaluated','reason':'timing only'},
                        'assets':{name:{'role':role,'sha256':n.sha(path)} for (name,path),role in zip(assets.items(),('model','inputs'))}}
            result = n.prepare(manifest, assets)
            self.assertEqual(len(result['request_lengths']),512)
            bad = copy.deepcopy(manifest); bad['quality']['kind']='passed'
            with self.assertRaises(ValueError): n.prepare(bad,assets)
            n.save(d/'lengths.json',[128]*8)
            manifest['assets']['request_lengths']['sha256']=n.sha(d/'lengths.json')
            with self.assertRaises(ValueError): n.prepare(manifest,assets)

    def test_runtime_cannot_omit_required_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'runtime.json'
            n.save(p, {'schema':'pim-foundry.neupims-runtime.v1','method':'neupims',
                       'source_commit':n.COMMITS['neupims'],'source_root':d,'source_files':{},
                       'files':{},'config_paths':{},'build':{},'scope':n.SCOPE})
            with self.assertRaisesRegex(ValueError,'required source'):n.runtime(p)

    def _runtime_fixture(self, root):
        source_files = {}
        for rel in ('src/client/Client.cc', 'src/scheduler/Scheduler.cc',
                    'src/Simulator.cc', 'src/Model.cc', 'src/ModelProgram.cc',
                    'CMakeLists.txt', 'conanfile.txt'):
            path = root / rel; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('metadata validation fixture\n')
            source_files[rel] = n.sha(path)
        config_paths = {name: 'configs/' + name + '.json' for name in ('hardware', 'memory', 'system')}
        for name, rel in config_paths.items():
            path = root / rel; path.parent.mkdir(parents=True, exist_ok=True)
            n.save(path, {'pim_config_path': '../timing.ini'} if name == 'memory' else {})
            source_files[rel] = n.sha(path)
        (root / 'timing.ini').write_text('[timing]\nfrequency=1000\n')
        binary = root / 'runtime-file'; binary.write_text('not executable; metadata test only\n')
        return {'schema': 'pim-foundry.neupims-runtime.v1', 'method': 'neupims',
                'source_commit': n.COMMITS['neupims'], 'source_root': str(root),
                'source_files': source_files, 'config_paths': config_paths,
                'files': {name: {'path': str(binary), 'sha256': n.sha(binary)}
                          for name in ('executable', 'library', 'dependencies', 'flags')},
                'build': {}, 'scope': n.SCOPE}

    def test_runtime_requires_transitive_pim_timing_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); descriptor = self._runtime_fixture(root); path = root / 'runtime.json'
            n.save(path, descriptor)
            with self.assertRaisesRegex(ValueError, 'PIM timing config lacks source binding'):
                n.runtime(path)
            descriptor['source_files']['timing.ini'] = n.sha(root / 'timing.ini')
            n.save(path, descriptor)
            n.runtime(path)
            (root / 'timing.ini').write_text('[timing]\nfrequency=500\n')
            with self.assertRaisesRegex(ValueError, 'modified/missing NeuPIMs source'):
                n.runtime(path)

    def test_runtime_rejects_absolute_or_escaping_pim_timing_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); descriptor = self._runtime_fixture(root); path = root / 'runtime.json'
            memory = root / descriptor['config_paths']['memory']
            for timing_path in (str(root / 'timing.ini'), '../../outside.ini'):
                with self.subTest(timing_path=timing_path):
                    n.save(memory, {'pim_config_path': timing_path})
                    descriptor['source_files']['configs/memory.json'] = n.sha(memory)
                    n.save(path, descriptor)
                    with self.assertRaisesRegex(ValueError, 'PIM timing config'):
                        n.runtime(path)


if __name__ == '__main__': unittest.main()
