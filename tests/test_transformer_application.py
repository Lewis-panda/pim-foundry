import importlib.util
import math
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from pim_foundry import transformer_application as tr


@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional NumPy dependency')
class TransformerApplicationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np
        self.config={'n_embd':8,'n_head':2,'n_layer':2,'n_positions':16,'vocab_size':17,'activation_function':'gelu_new','layer_norm_epsilon':1e-5}
        rng=np.random.default_rng(1729)
        self.weights={'transformer.wte.weight':rng.normal(0,.1,(17,8)).astype('float32'),'transformer.wpe.weight':rng.normal(0,.1,(16,8)).astype('float32'),'transformer.ln_f.weight':np.ones(8,dtype='float32'),'transformer.ln_f.bias':np.zeros(8,dtype='float32')}
        for i in range(2):
            p='transformer.h.%d.'%i
            for norm in ('ln_1','ln_2'):
                self.weights[p+norm+'.weight']=np.ones(8,dtype='float32');self.weights[p+norm+'.bias']=np.zeros(8,dtype='float32')
            for name,shape in [('attn.c_attn',(8,24)),('attn.c_proj',(8,8)),('mlp.c_fc',(8,32)),('mlp.c_proj',(32,8))]:
                self.weights[p+name+'.weight']=rng.normal(0,.1,shape).astype('float32');self.weights[p+name+'.bias']=rng.normal(0,.01,shape[-1]).astype('float32')
    def session(self):return tr.GPT2Session(self.config,self.weights)
    def test_cached_decode_matches_full_recompute_and_state_shapes(self):
        np=self.np;cached=self.session();ids=np.array([[1,2,3],[3,4,5]],dtype='int32');cached.forward(ids)
        for token in (np.array([[7],[8]],dtype='int32'),np.array([[6],[9]],dtype='int32')):
            actual=cached.forward(token);ids=np.concatenate([ids,token],axis=1);expected=self.session().forward(ids)
            np.testing.assert_allclose(actual[:,-1],expected[:,-1],rtol=1e-5,atol=1e-6)
        self.assertEqual(cached.state.length,5)
        self.assertEqual(cached.state.tensors[(0,'k')].shape,(2,2,5,4))
    def test_future_tokens_do_not_change_prefix_logits(self):
        np=self.np;a=self.session().forward([[1,2,3,4]]);b=self.session().forward([[1,2,12,13]])
        np.testing.assert_array_equal(a[:,:2],b[:,:2]);self.assertGreater(float(np.abs(a[:,2:]-b[:,2:]).max()),.01)
    def test_batch_matches_independent_requests(self):
        np=self.np;batch=self.session().forward([[1,2,3],[4,5,6]])
        for i,row in enumerate([[1,2,3],[4,5,6]]):np.testing.assert_allclose(batch[i],self.session().forward([row])[0],rtol=1e-5,atol=1e-6)
    def test_reset_prevents_cross_request_cache_contamination(self):
        np=self.np;session=self.session();session.forward([[9,8]]);session.reset();actual=session.forward([[1,2,3]])
        np.testing.assert_array_equal(actual,self.session().forward([[1,2,3]]));self.assertEqual(session.state.length,3)
    def test_generation_is_greedy_and_reset_each_call(self):
        np=self.np;s=self.session();a=s.generate([[1,2]],3);b=s.generate([[1,2]],3)
        np.testing.assert_array_equal(a['token_ids'],a['step_logits'].argmax(-1));np.testing.assert_array_equal(a['token_ids'],b['token_ids']);self.assertEqual(a['cache_length'],4)
    def test_capacity_failure_does_not_mutate_cache(self):
        s=self.session();s.forward([[1]*15]);before=s.state
        with self.assertRaises(ValueError):s.forward([[2,3]])
        self.assertIs(s.state,before)
        with self.assertRaises(ValueError):s.generate([[1]*15],2)
    def test_batch_change_requires_reset(self):
        s=self.session();s.forward([[1,2]])
        with self.assertRaises(ValueError):s.forward([[3],[4]])
    def test_rejects_bad_token_ids(self):
        for ids in ([[1.2]],[[True]],[[-1]],[[17]],[],[1,2]):
            with self.assertRaises(ValueError):self.session().forward(ids)
    def test_nll_target_shift_and_denominator(self):
        np=self.np;logits=np.zeros((2,3,17),dtype='float32');actual=tr.next_token_nll(logits,[[1,2,3],[4,5,6]])
        self.assertAlmostEqual(actual['mean_nll'],math.log(17));self.assertAlmostEqual(actual['perplexity'],17);self.assertEqual(actual['scored_tokens'],4)
    def test_nll_labels_matter(self):
        np=self.np;logits=np.zeros((1,3,17),dtype='float32');logits[0,0,2]=6;logits[0,1,3]=6
        good=tr.next_token_nll(logits,[[1,2,3]]);bad=tr.next_token_nll(logits,[[1,4,5]])
        self.assertGreater(bad['mean_nll']-good['mean_nll'],5.9)
    def platform_fixture(self):
        from pim_foundry.tensor_platform import default_platform
        s=self.session();inputs={'prompt_token_ids':[[1,2]],'evaluation_token_ids':[[1,2,3]]};g=s.generate(inputs['prompt_token_ids'],2)
        ref={'generation_logits':g['step_logits'],'full_recompute_logits':g['step_logits'],'generated_token_ids':g['token_ids']}
        for (i,k),v in s.state.tensors.items():ref['cache_%d_%s'%(i,k)]=v
        s.reset();ref['evaluation_logits']=s.forward(inputs['evaluation_token_ids'])
        prepared={'config':self.config,'weights':self.weights,'inputs':inputs,'reference':ref,'dataset':{},'manifest':{'parameters':{'max_new_tokens':2},'quality':{'atol':1e-5,'rtol':1e-5,'max_nll_delta':1e-5,'reference':'unit-fixture'}}}
        experiment={'platform':default_platform('gpu'),'mapping':{'strategy':'all_to_device','device':'gpu'},'variant':{}}
        return prepared,experiment
    def test_platform_runtime_mutation_between_calls_is_rejected(self):
        from pim_foundry.tensor_executor import execute
        prepared,experiment=self.platform_fixture();calls=[]
        def run(graph,inputs,platform,**kwargs):
            result=execute(graph,inputs);calls.append(1);binding={'runtime':'a' if len(calls)==1 else 'b'}
            result['platform_execution']={'platform_identity':{'config':platform,'runtime_bindings':binding},'runtime_bindings':binding,'modeled_latency_ns':1.,'modeled_energy_pj':None}
            return result
        with tempfile.TemporaryDirectory() as directory,patch('pim_foundry.tensor_platform.run',run),self.assertRaisesRegex(ValueError,'identity changed'):
            tr.execute(prepared,experiment,Path(directory)/'out')
    def test_unknown_platform_energy_is_omitted_not_zero(self):
        from pim_foundry.tensor_executor import execute
        prepared,experiment=self.platform_fixture()
        def run(graph,inputs,platform,**kwargs):
            result=execute(graph,inputs);binding={'runtime':'a'}
            result['platform_execution']={'platform_identity':{'config':platform,'runtime_bindings':binding},'runtime_bindings':binding,'modeled_latency_ns':1.,'modeled_energy_pj':None}
            return result
        with tempfile.TemporaryDirectory() as directory,patch('pim_foundry.tensor_platform.run',run):
            result=tr.execute(prepared,experiment,Path(directory)/'out')
        self.assertNotIn('generation_modeled_energy_pj',result['metrics'])
        self.assertIn('generation_modeled_energy_pj',result['diagnostics']['unknown_metrics'])
        self.assertEqual(result['platform_identity']['runtime_bindings'],{'runtime':'a'})
        self.assertEqual(result['execution_identity']['runtime_bindings'],{'runtime':'a'})
    def test_precision_variant_inserts_real_cast_nodes(self):
        graph=tr.build_graph(self.config,self.weights,1,2)
        changed=tr.apply_fp16_matmul_variant(graph,['layer.0.qkv.matmul'])
        nodes={n['name']:n for n in changed['nodes']};half=nodes['layer.0.qkv.matmul.fp16']
        self.assertEqual(changed['values'][half['output']]['dtype'],'float16')
        self.assertTrue(all(changed['values'][x]['dtype']=='float16' for x in half['inputs']))
        self.assertEqual(nodes['layer.0.qkv.matmul']['op'],'cast')
        self.assertEqual(changed['values']['layer.0.qkv.matmul']['dtype'],'float32')
        self.assertEqual(graph['values']['layer.0.qkv.matmul']['dtype'],'float32')
    def test_precision_variant_rejects_unknown_or_nonlinear_nodes(self):
        graph=tr.build_graph(self.config,self.weights,1,2)
        for names in ([],['missing'],['layer.0.softmax'],['layer.0.qkv.matmul']*2,[[1]]):
            with self.assertRaises(ValueError):tr.apply_fp16_matmul_variant(graph,names)
    def test_nll_rejects_invalid_labels_and_nonfinite_logits(self):
        np=self.np
        for labels in ([[1,-1]],[[1,17]],[[1,1.5]]):
            with self.assertRaises(ValueError):tr.next_token_nll(np.zeros((1,2,17)),labels)
        with self.assertRaises(ValueError):tr.next_token_nll(np.full((1,2,17),np.nan),[[1,2]])
    def test_generation_rejects_fractional_ids(self):
        with self.assertRaises(ValueError):self.session().generate([[1.5,2]],2)
    def test_weight_coverage_and_dtype_are_enforced(self):
        wrong=dict(self.weights);wrong.pop('transformer.h.1.mlp.c_proj.weight')
        with self.assertRaises(ValueError):tr.GPT2Session(self.config,wrong)
        wrong=dict(self.weights);wrong['transformer.wte.weight']=wrong['transformer.wte.weight'].astype('float16')
        with self.assertRaises(ValueError):tr.GPT2Session(self.config,wrong)
    def test_model_config_cannot_silently_change_attention(self):
        for change in ({'n_head':3},{'scale_attn_by_inverse_layer_idx':True},{'activation_function':'relu'},{'add_cross_attention':True}):
            config=dict(self.config);config.update(change)
            with self.assertRaises(ValueError):tr.validate_config(config)

if __name__=='__main__':unittest.main()
