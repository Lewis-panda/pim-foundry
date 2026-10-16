"""Complete GPT-2 inference through the shared tensor IR, with explicit KV state.

NumPy is an optional dependency. Reference inference is never substituted for
application inference: reference arrays are used only after graph execution.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

ADAPTER = 'transformer-graph/v1'
SCENARIO = {'kind': 'autoregressive', 'state_policy': 'reset_each_run',
            'decoding': 'greedy', 'batching': 'equal_length'}


def _np():
    import numpy as np
    return np


def _read(path):
    return json.loads(Path(path).read_text())


def validate_config(config):
    for key in ('n_embd', 'n_head', 'n_layer', 'n_positions', 'vocab_size'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError('GPT-2 config requires positive integer ' + key)
    if config.get('n_inner') is not None and (type(config['n_inner']) is not int or config['n_inner']<=0):
        raise ValueError('n_inner must be a positive integer or null')
    if type(config.get('layer_norm_epsilon',1e-5)) not in (int,float) or not math.isfinite(config.get('layer_norm_epsilon',1e-5)) or config.get('layer_norm_epsilon',1e-5)<=0:
        raise ValueError('layer norm epsilon must be finite and positive')
    if config['n_embd'] % config['n_head']:
        raise ValueError('hidden dimension must divide attention heads')
    if config.get('activation_function', 'gelu_new') not in ('gelu_new', 'gelu_fast', 'gelu_pytorch_tanh'):
        raise ValueError('only the GPT-2 tanh GELU activation contract is supported')
    if config.get('scale_attn_by_inverse_layer_idx', False) or config.get('reorder_and_upcast_attn', False):
        raise ValueError('nonstandard GPT-2 attention scaling is unsupported')
    for key in ('scale_attn_weights','scale_attn_by_inverse_layer_idx','reorder_and_upcast_attn','add_cross_attention','tie_word_embeddings'):
        if key in config and type(config[key]) is not bool:raise ValueError(key+' must be boolean')
    if config.get('tie_word_embeddings',True) is not True:raise ValueError('GPT-2 application requires tied embeddings')
    if config.get('add_cross_attention', False):
        raise ValueError('cross attention is outside this decoder-only application')


def validate_weights(config, weights):
    np = _np(); validate_config(config)
    h, v, p = config['n_embd'], config['vocab_size'], config['n_positions']
    inner = config.get('n_inner') or 4 * h
    expected = {'transformer.wte.weight': (v, h), 'transformer.wpe.weight': (p, h),
                'transformer.ln_f.weight': (h,), 'transformer.ln_f.bias': (h,)}
    for i in range(config['n_layer']):
        prefix = 'transformer.h.%d.' % i
        expected.update({prefix + k: shape for k, shape in {
            'ln_1.weight': (h,), 'ln_1.bias': (h,), 'ln_2.weight': (h,), 'ln_2.bias': (h,),
            'attn.c_attn.weight': (h, 3*h), 'attn.c_attn.bias': (3*h,),
            'attn.c_proj.weight': (h, h), 'attn.c_proj.bias': (h,),
            'mlp.c_fc.weight': (h, inner), 'mlp.c_fc.bias': (inner,),
            'mlp.c_proj.weight': (inner, h), 'mlp.c_proj.bias': (h,)}.items()})
    for name, shape in expected.items():
        if name not in weights or tuple(weights[name].shape) != shape:
            raise ValueError('missing or incompatible weight ' + name)
        if weights[name].dtype != np.float32 or not np.isfinite(weights[name]).all():
            raise ValueError('weights must be finite float32: ' + name)
    if 'lm_head.weight' in weights and (weights['lm_head.weight'].shape != (v,h) or
            not np.array_equal(weights['lm_head.weight'], weights['transformer.wte.weight'])):
        raise ValueError('this GPT-2 contract requires the tied LM head')


def build_graph(config, weights, batch, length, past_length=0):
    """Build concrete rank-N IR. Cache tensors are graph inputs and outputs."""
    np = _np(); validate_config(config)
    if any(type(x) is not int for x in (batch,length,past_length)) or batch < 1 or length < 1 or past_length < 0:
        raise ValueError('batch/length must be positive and past length nonnegative')
    if length + past_length > config['n_positions']:
        raise ValueError('sequence exceeds positional embedding / KV capacity')
    h,n,l,v = (config[k] for k in ('n_embd','n_head','n_layer','vocab_size')); d=h//n; total=length+past_length
    graph={'schema':'pim-foundry.tensor-graph.v2','values':{},'inputs':[], 'outputs':[], 'initializers':{}, 'nodes':[]}
    def value(name,shape,dtype='float32'):
        graph['values'][name]={'shape':list(shape),'dtype':dtype}; return name
    def init(name,array):
        array=np.asarray(array,dtype=np.float32);value(name,array.shape); graph['initializers'][name]=array;return name
    def node(name,op,inputs,shape,attrs=None,dtype='float32'):
        value(name,shape,dtype);graph['nodes'].append({'name':name,'op':op,'inputs':inputs,'output':name,'attrs':attrs or {}});return name
    for name,array in weights.items():
        if name.startswith('transformer.') and not name.endswith(('.attn.bias','.attn.masked_bias')): init(name,array)
    value('token_ids',(batch,length),'int32');value('position_ids',(batch,length),'int32');graph['inputs']=['token_ids','position_ids']
    x=node('embedding.tokens','embedding',['transformer.wte.weight','token_ids'],(batch,length,h))
    pos=node('embedding.positions','embedding',['transformer.wpe.weight','position_ids'],(batch,length,h))
    x=node('embedding.sum','add',[x,pos],(batch,length,h))
    mask=np.where(np.arange(total)[None,:] <= (past_length+np.arange(length))[:,None],0.0,-1e30).astype(np.float32)
    init('attention.mask',mask.reshape(1,1,length,total));init('attention.scale',np.asarray(1/math.sqrt(d) if config.get('scale_attn_weights',True) else 1,dtype=np.float32))
    caches=[]
    def linear(name,inp,prefix,outdim):
        z=node(name+'.matmul','matmul',[inp,prefix+'.weight'],(batch,length,outdim))
        return node(name+'.bias','add',[z,prefix+'.bias'],(batch,length,outdim))
    for i in range(l):
        p='transformer.h.%d.'%i; a='layer.%d.'%i
        norm=node(a+'ln1','layer_norm',[x,p+'ln_1.weight',p+'ln_1.bias'],(batch,length,h),{'epsilon':config.get('layer_norm_epsilon',1e-5)})
        qkv=linear(a+'qkv',norm,p+'attn.c_attn',3*h);parts=[]
        for j,kind in enumerate(('q','k','v')):
            z=node(a+kind+'.slice','slice',[qkv],(batch,length,h),{'starts':[j*h],'ends':[(j+1)*h],'axes':[2]})
            z=node(a+kind+'.reshape','reshape',[z],(batch,length,n,d),{'shape':[batch,length,n,d]})
            z=node(a+kind+'.transpose','transpose',[z],(batch,n,length,d),{'perm':[0,2,1,3]})
            if kind in ('k','v') and past_length:
                cache=a+kind+'.past';value(cache,(batch,n,past_length,d));graph['inputs'].append(cache)
                z=node(a+kind+'.present','concat',[cache,z],(batch,n,total,d),{'axis':2})
            parts.append(z)
            if kind in ('k','v'): caches.append(z)
        q,k,val=parts
        kt=node(a+'kt','transpose',[k],(batch,n,d,total),{'perm':[0,1,3,2]})
        scores=node(a+'qk','matmul',[q,kt],(batch,n,length,total))
        scores=node(a+'scale','mul',[scores,'attention.scale'],(batch,n,length,total))
        scores=node(a+'mask','add',[scores,'attention.mask'],(batch,n,length,total))
        probs=node(a+'softmax','softmax',[scores],(batch,n,length,total),{'axis':-1})
        attn=node(a+'pv','matmul',[probs,val],(batch,n,length,d))
        attn=node(a+'attn.transpose','transpose',[attn],(batch,length,n,d),{'perm':[0,2,1,3]})
        attn=node(a+'attn.reshape','reshape',[attn],(batch,length,h),{'shape':[batch,length,h]})
        attn=linear(a+'attn.proj',attn,p+'attn.c_proj',h)
        x=node(a+'residual1','add',[x,attn],(batch,length,h))
        norm=node(a+'ln2','layer_norm',[x,p+'ln_2.weight',p+'ln_2.bias'],(batch,length,h),{'epsilon':config.get('layer_norm_epsilon',1e-5)})
        inner=config.get('n_inner') or 4*h
        mlp=linear(a+'mlp.fc',norm,p+'mlp.c_fc',inner)
        mlp=node(a+'gelu','gelu',[mlp],(batch,length,inner),{'approximate':'tanh'})
        mlp=linear(a+'mlp.proj',mlp,p+'mlp.c_proj',h)
        x=node(a+'residual2','add',[x,mlp],(batch,length,h))
    x=node('final.norm','layer_norm',[x,'transformer.ln_f.weight','transformer.ln_f.bias'],(batch,length,h),{'epsilon':config.get('layer_norm_epsilon',1e-5)})
    wt=node('lm_head.transpose','transpose',['transformer.wte.weight'],(h,v),{'perm':[1,0]})
    node('logits','matmul',[x,wt],(batch,length,v))
    last=node('selection.last_logits','slice',['logits'],(batch,1,v),{'starts':[length-1],'ends':[length],'axes':[1]})
    last=node('selection.reshape','reshape',[last],(batch,v),{'shape':[batch,v]})
    node('next_token_ids','argmax',[last],(batch,),{'axis':-1,'keepdims':False},dtype='int32')
    graph['outputs']=['logits','next_token_ids']+caches
    return graph


@dataclass
class KVState:
    batch: int
    length: int
    tensors: dict


class GPT2Session:
    """One request batch owns its cache. reset() drops all persistent state."""
    def __init__(self,config,weights,runner=None):
        validate_weights(config,weights); self.config=config; self.weights=weights;self.runner=runner;self.state=None;self.calls=[];self.next_token_ids=None
    def reset(self):
        self.state=None; self.calls=[];self.next_token_ids=None
    def forward(self,token_ids):
        np=_np(); ids=np.asarray(token_ids)
        if ids.ndim!=2 or ids.dtype.kind not in 'iu' or ids.shape[0]<1 or ids.shape[1]<1:
            raise ValueError('token IDs must be a nonempty rank-2 integer batch')
        if (ids<0).any() or (ids>=self.config['vocab_size']).any(): raise ValueError('token ID outside vocabulary')
        b,t=ids.shape;p=self.state.length if self.state else 0
        if self.state and self.state.batch!=b: raise ValueError('batch size changed without cache reset')
        graph=build_graph(self.config,self.weights,b,t,p)
        inputs={'token_ids':ids.astype(np.int32),'position_ids':np.broadcast_to(np.arange(p,p+t,dtype=np.int32),(b,t)).copy()}
        if self.state:
            for i in range(self.config['n_layer']):
                for kind in ('k','v'):inputs['layer.%d.%s.past'%(i,kind)]=self.state.tensors[(i,kind)]
        if self.runner: result=self.runner(graph,inputs)
        else:
            from .tensor_executor import execute
            result=execute(graph,inputs)
        outputs=result['outputs']; caches={}
        for i in range(self.config['n_layer']):
            for kind in ('k','v'):
                key='layer.%d.%s.%s'%(i,kind,'present' if p else 'transpose')
                caches[(i,kind)]=outputs[key]
        self.state=KVState(b,p+t,caches);self.next_token_ids=outputs['next_token_ids']
        self.calls.append({k:v for k,v in result.items() if k not in ('outputs','intermediates')})
        return outputs['logits']
    def generate(self,prompt,max_new_tokens):
        np=_np()
        if type(max_new_tokens)is not int or max_new_tokens<1: raise ValueError('max_new_tokens must be positive integer')
        ids=np.asarray(prompt)
        if ids.ndim!=2 or ids.dtype.kind not in 'iu': raise ValueError('prompt must be rank-2 integer token IDs')
        if ids.shape[1]+max_new_tokens>self.config['n_positions']: raise ValueError('generation exceeds KV capacity')
        self.reset(); logits=self.forward(ids); steps=[];generated=[]
        for i in range(max_new_tokens):
            step=logits[:,-1,:].copy();steps.append(step);token=self.next_token_ids.copy();generated.append(token)
            if i+1<max_new_tokens:logits=self.forward(token[:,None])
        return {'token_ids':np.stack(generated,axis=1),'step_logits':np.stack(steps,axis=1),'cache_length':self.state.length}


def next_token_nll(logits,token_ids):
    np=_np(); logits=np.asarray(logits,dtype=np.float64);ids=np.asarray(token_ids)
    if logits.ndim!=3 or ids.shape!=logits.shape[:2] or ids.shape[1]<2:raise ValueError('NLL requires aligned logits and >=2 input tokens')
    if not np.isfinite(logits).all() or ids.dtype.kind not in 'iu' or (ids<0).any() or (ids>=logits.shape[-1]).any():raise ValueError('NLL requires finite logits and valid integer target IDs')
    z=logits[:,:-1];labels=ids[:,1:];maxima=z.max(axis=-1)
    logsum=maxima+np.log(np.exp(z-maxima[...,None]).sum(axis=-1))
    selected=np.take_along_axis(z,labels[...,None],axis=-1)[...,0]
    nll=float((logsum-selected).mean())
    return {'mean_nll':nll,'perplexity':math.exp(nll),'scored_tokens':int(labels.size),'protocol':'teacher_forcing; next-token shift; first token excluded; no padding'}


def prepare(manifest,assets):
    np=_np(); required={'model','config','inputs','reference','dataset'}
    if set(assets)!=required:raise ValueError('transformer assets require '+','.join(sorted(required)))
    roles={'model':'model','config':'model','inputs':'inputs','reference':'quality','dataset':'quality'}
    if any(manifest['assets'][k]['role']!=v for k,v in roles.items()):raise ValueError('transformer semantic asset roles cannot be hidden')
    if manifest['scenario']!=SCENARIO:raise ValueError('unsupported transformer state/batching/decoding policy')
    params=manifest['parameters']
    if set(params)!={'architecture','max_new_tokens'} or params['architecture']!='gpt2' or type(params['max_new_tokens']) is not int or params['max_new_tokens']<1:
        raise ValueError('expected GPT-2 architecture and positive max_new_tokens')
    quality=manifest['quality']
    if set(quality)!={'kind','reference','atol','rtol','max_nll_delta'} or quality['kind']!='reference_and_next_token_nll':raise ValueError('explicit reference and NLL policy required')
    for k in ('atol','rtol','max_nll_delta'):
        if type(quality[k]) not in (int,float) or not math.isfinite(quality[k]) or quality[k]<0:raise ValueError('invalid quality tolerance')
    cfg=_read(assets['config']); inp=_read(assets['inputs']); dataset=_read(assets['dataset']);validate_config(cfg)
    if set(inp)!={'prompt_token_ids','evaluation_token_ids'}:raise ValueError('prompt and evaluation token IDs required')
    for name in inp:
        ids=np.asarray(inp[name]);
        if ids.ndim!=2 or ids.dtype.kind not in 'iu' or min(ids.shape)<1 or (ids<0).any() or (ids>=cfg['vocab_size']).any():raise ValueError('invalid '+name)
        if ids.shape[1]>cfg['n_positions']:raise ValueError('input exceeds context length')
    if len(inp['evaluation_token_ids'][0])<2:raise ValueError('evaluation requires at least 2 tokens')
    if len(inp['prompt_token_ids'][0])+params['max_new_tokens']>cfg['n_positions']:raise ValueError('generation exceeds context capacity')
    if dataset.get('evaluation_token_sha256')!=hashlib.sha256(json.dumps(inp['evaluation_token_ids'],separators=(',',':')).encode()).hexdigest():raise ValueError('dataset token selection hash mismatch')
    with np.load(assets['model'],allow_pickle=False) as file: weights={name:file[name] for name in file.files}
    validate_weights(cfg,weights)
    with np.load(assets['reference'],allow_pickle=False) as file: reference={name:file[name] for name in file.files}
    b,t=np.asarray(inp['prompt_token_ids']).shape;e_b,e_t=np.asarray(inp['evaluation_token_ids']).shape;n=params['max_new_tokens'];h=cfg['n_head'];d=cfg['n_embd']//h;v=cfg['vocab_size']
    shapes={'generation_logits':(b,n,v),'generated_token_ids':(b,n),'full_recompute_logits':(b,n,v),'evaluation_logits':(e_b,e_t,v)}
    shapes.update({'cache_%d_%s'%(i,k):(b,h,t+n-1,d) for i in range(cfg['n_layer']) for k in ('k','v')})
    if set(reference)!=set(shapes):raise ValueError('reference must cover full logits, greedy tokens, full recompute and every KV cache')
    for key,shape in shapes.items():
        if reference[key].shape!=shape or not np.isfinite(reference[key]).all():raise ValueError('invalid reference array '+key)
    if reference['generated_token_ids'].dtype.kind not in 'iu':raise ValueError('reference tokens must be integers')
    return {'manifest':manifest,'config':cfg,'weights':weights,'inputs':inp,'reference':reference,'dataset':dataset}


def apply_fp16_matmul_variant(graph, selected):
    """Explicit storage casts around selected FP32 matmuls; no implicit precision."""
    if not isinstance(selected,list) or not selected or any(not isinstance(x,str) for x in selected) or len(set(selected))!=len(selected):
        raise ValueError('fp16_matmul_nodes must be a nonempty unique node-name list')
    by_name={n['name']:n for n in graph['nodes']}
    if any(name not in by_name or by_name[name]['op']!='matmul' for name in selected):raise ValueError('FP16 selection must name existing matmul nodes')
    graph={**graph,'values':dict(graph['values']),'nodes':[]}
    for node in by_name.values():
        if node['name'] not in selected:graph['nodes'].append(node);continue
        args=[]
        for i,name in enumerate(node['inputs']):
            if graph['values'][name]['dtype']!='float32':raise ValueError('variant requires float32 source matmuls')
            cast=node['name']+'.fp16.input'+str(i);graph['values'][cast]={**graph['values'][name],'dtype':'float16'}
            graph['nodes'].append({'name':cast,'op':'cast','inputs':[name],'output':cast,'attrs':{'dtype':'float16'}});args.append(cast)
        half=node['name']+'.fp16.value';graph['values'][half]={**graph['values'][node['output']],'dtype':'float16'}
        graph['nodes'].append({**node,'name':node['name']+'.fp16','inputs':args,'output':half})
        graph['nodes'].append({'name':node['name'],'op':'cast','inputs':[half],'output':node['output'],'attrs':{'dtype':'float32'}})
    return graph


def inspect(prepared,experiment):
    variant=experiment['variant']
    if variant:
        if set(variant)!={'fp16_matmul_nodes'}:raise ValueError('only explicit fp16_matmul_nodes variant is supported')
        shape=_np().asarray(prepared['inputs']['prompt_token_ids']).shape
        apply_fp16_matmul_variant(build_graph(prepared['config'],prepared['weights'],*shape),variant['fp16_matmul_nodes'])
    platform=experiment['platform'];mapping=experiment['mapping']
    if platform=={'kind':'tensor-reference-cpu/v1'}:
        if mapping!={'strategy':'reference_cpu'}:raise ValueError('CPU reference execution requires reference_cpu mapping')
        return {'passed':True,'execution':'shared_tensor_ir_cpu','model': 'complete_gpt2_decoder','hardware_performance_claim':False,'state':'explicit_KV_inputs_outputs'}
    from .unified import validate_config as validate_platform
    validate_platform(platform)
    if mapping.get('strategy')=='hybrid_linear':
        if set(mapping)!={'strategy','default_device','fp16_device'} or any(mapping[k] not in {d['name'] for d in platform['devices']} for k in ('default_device','fp16_device')) or not variant:raise ValueError('hybrid_linear requires explicit precision variant and declared devices')
    elif mapping.get('strategy')=='all_to_device':
        if set(mapping)!={'strategy','device'} or mapping['device'] not in {d['name'] for d in platform['devices']}:raise ValueError('all_to_device requires one declared device')
    elif set(mapping)!={'strategy','placement'} or mapping['strategy']!='explicit':raise ValueError('modeled tensor platforms require explicit or all_to_device placement')
    return {'passed':True,'execution':'shared_tensor_ir_platform','hardware_performance_claim':False,'state':'explicit_KV_inputs_outputs'}


def execute(prepared,experiment,output):
    np=_np();output=Path(output);output.mkdir(parents=True,exist_ok=False);inspect(prepared,experiment)
    cpu=experiment['platform']=={'kind':'tensor-reference-cpu/v1'}
    runner=None
    if cpu and experiment['variant']:
        from .tensor_executor import execute as execute_tensor
        runner=lambda graph,inputs:execute_tensor(apply_fp16_matmul_variant(graph,experiment['variant']['fp16_matmul_nodes']),inputs)
    if not cpu:
        from .tensor_platform import run
        call_index=[0]
        def runner(graph,inputs):
            mapping=experiment['mapping']
            if experiment['variant']:graph=apply_fp16_matmul_variant(graph,experiment['variant']['fp16_matmul_nodes'])
            placement=({n['name']:mapping['device'] for n in graph['nodes']} if mapping['strategy']=='all_to_device' else {n['name']:(mapping['fp16_device'] if n['name'].endswith('.fp16') else mapping['default_device']) for n in graph['nodes']} if mapping['strategy']=='hybrid_linear' else mapping['placement'])
            path=output/('platform-call-%03d'%call_index[0]);call_index[0]+=1
            return run(graph,inputs,experiment['platform'],placement=placement,output=path)
    session=GPT2Session(prepared['config'],prepared['weights'],runner)
    generated=session.generate(prepared['inputs']['prompt_token_ids'],prepared['manifest']['parameters']['max_new_tokens'])
    generation_calls=session.calls
    cache_shapes={str(k):list(v.shape) for k,v in session.state.tensors.items()}
    generation_cache=dict(session.state.tensors)
    session.reset(); logits=session.forward(prepared['inputs']['evaluation_token_ids']);evaluation_calls=session.calls
    nll=next_token_nll(logits,prepared['inputs']['evaluation_token_ids']);ref=prepared['reference'];policy=prepared['manifest']['quality']
    def compare(name,actual,expected):
        if actual.shape!=expected.shape:return {'passed':False,'shape_mismatch':[list(actual.shape),list(expected.shape)]}
        delta=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        return {'passed':bool(np.allclose(actual,expected,atol=policy['atol'],rtol=policy['rtol'])),'max_abs_error':float(delta.max())}
    checks={'generation_logits':compare('generation_logits',generated['step_logits'],ref['generation_logits']),
            'evaluation_logits':compare('evaluation_logits',logits,ref['evaluation_logits']),
            'greedy_tokens':{'passed':bool(np.array_equal(generated['token_ids'],ref['generated_token_ids']))}}
    if 'full_recompute_logits' in ref:checks['full_recompute_logits']=compare('full_recompute_logits',generated['step_logits'],ref['full_recompute_logits'])
    for (layer,kind),value in generation_cache.items():
        key='cache_%d_%s'%(layer,kind)
        if key in ref:checks[key]=compare(key,value,ref[key])
    ref_nll=next_token_nll(ref['evaluation_logits'],prepared['inputs']['evaluation_token_ids'])
    delta=abs(nll['mean_nll']-ref_nll['mean_nll']);numeric=all(c['passed'] for c in checks.values());task=numeric and delta<=policy['max_nll_delta']
    np.savez(output/'outputs.npz',generated_token_ids=generated['token_ids'],generation_logits=generated['step_logits'],evaluation_logits=logits)
    evidence={'checks':checks,'generated_token_ids':generated['token_ids'].tolist(),'quality':nll,'reference_quality':ref_nll,'nll_delta':delta,'cache_shapes':cache_shapes,'cache_length_after_generation':generated['cache_length'],'generation_calls':generation_calls,'evaluation_calls':evaluation_calls,'dataset':prepared['dataset']}
    (output/'evidence.json').write_text(json.dumps(evidence,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
    def metric(value,unit,scope,aggregation='total'):return {'value':value,'unit':unit,'scope':scope,'aggregation':aggregation,'origin':'executed_complete_transformer_tensor_graph'}
    metrics={'next_token_nll':metric(nll['mean_nll'],'nats/token','declared_dataset_fixture','mean'), 'perplexity':metric(nll['perplexity'],'ratio','declared_dataset_fixture','exp_mean_nll'),'scored_tokens':metric(nll['scored_tokens'],'token','declared_dataset_fixture'),'generated_tokens':metric(int(generated['token_ids'].size),'token','greedy_generation')}
    platform_identity=experiment['platform'];runtime_bindings={};unknown_metrics=[]
    if not cpu:
        all_calls=generation_calls+evaluation_calls
        platform_identity=all_calls[0]['platform_execution']['platform_identity']
        runtime_bindings=all_calls[0]['platform_execution']['runtime_bindings']
        if any(call['platform_execution']['platform_identity']!=platform_identity or call['platform_execution']['runtime_bindings']!=runtime_bindings for call in all_calls):raise ValueError('platform/runtime identity changed between transformer calls')
        for phase,calls in [('generation',generation_calls),('evaluation',evaluation_calls)]:
            profiles=[call['platform_execution'] for call in calls]
            for key,unit in [('modeled_latency_ns','ns'),('modeled_energy_pj','pJ')]:
                if any(p[key] is None for p in profiles):unknown_metrics.append(phase+'_'+key);continue
                metrics[phase+'_'+key]={'value':sum(p[key] for p in profiles),'unit':unit,'scope':phase+'_cold_serial_graph_calls_including_transfers','aggregation':'sum','origin':'uncalibrated_shared_platform_model'}
        profiles=[call['platform_execution'] for call in generation_calls]
        metrics['modeled_ttft']={'value':profiles[0]['modeled_latency_ns'],'unit':'ns','scope':'cold_batched_prefill_including_greedy_argmax_and_output_transfers','aggregation':'per_batch','origin':'uncalibrated_shared_platform_model'}
        if len(profiles)>1:metrics['modeled_mean_itl']={'value':sum(p['modeled_latency_ns'] for p in profiles[1:])/(len(profiles)-1),'unit':'ns/token','scope':'cold_graph_calls_incremental_decode; KV copied from host between calls','aggregation':'arithmetic_mean_per_batch_step','origin':'uncalibrated_shared_platform_model'}
    return {'execution_mode':'shared_tensor_graph','metrics':metrics,'quality':{'numerical':{'status':'passed' if numeric else 'failed','checks':checks},'task':{'status':'passed' if task else 'failed','metric':'next_token_nll','value':nll['mean_nll'],'reference':ref_nll['mean_nll'],'absolute_delta':delta,'scope':'quality preservation on fixed fixture; no general model-quality claim'}},'platform_identity':platform_identity,'cost_contract':{'kind':'complete_gpt2_graph/v1','precision':'float32_with_explicit_fp16_matmuls' if experiment['variant'] else 'float32','state':'reset_each_run','timing':'not_measured' if cpu else 'model_only','quality_scope':'fixed_dataset_fixture','weights_policy':'cold_reload_each_graph_call','kv_policy':'persistent_host_arrays; explicit_transfer_each_graph_call','scheduling':'serial_dependent_autoregressive_calls','calibration':'not_hardware_calibrated'},'execution_identity':{'model':'gpt2','tensor_ir':'pim-foundry.tensor-graph.v2','numpy_version':np.__version__,'reference':prepared['manifest']['quality']['reference'],'runtime_bindings':runtime_bindings},'diagnostics':{'numerical_checks':checks,'state_shapes':cache_shapes,'unknown_metrics':unknown_metrics,'hardware_performance_claim':False}}
