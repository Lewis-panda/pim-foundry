#!/usr/bin/env python3
"""Complete tiny GPT-2 functional fixture through native HBM and explicit FP16 casts.

Run in the task-local reference environment (NumPy/torch/transformers/msgpack).
The tiny public checkpoint is a functional fixture, not a language-quality benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def download_assets(path):
    import urllib.request
    revision='5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be'
    hashes={'config.json':'77a9e9830c3abeba929f5c61e0e97c398f98b4481ad75516c0be5bf038b340b0','flax_model.msgpack':'3cd4987249615ca870f2c18e657a62f0962aecad0eac790dd89362227310fd30','vocab.json':'03087853bc70c618b66e7c7a43e787d2db4c469416beac9a483e53dad1f72f27','merges.txt':'1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5','tokenizer_config.json':'5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f'}
    path.mkdir(parents=True,exist_ok=True)
    for name,digest in hashes.items():
        target=path/name
        if not target.exists():urllib.request.urlretrieve('https://huggingface.co/sshleifer/tiny-gpt2/resolve/'+revision+'/'+name,target)
        if sha(target)!=digest:raise ValueError('modified tiny model asset '+name)

def prepare_assets(path,source):
    import msgpack
    import numpy as np
    import torch
    import transformers
    from transformers import GPT2Config,GPT2LMHeadModel
    if transformers.__version__!='4.44.2':raise ValueError('reference requires Transformers 4.44.2')
    expected='3cd4987249615ca870f2c18e657a62f0962aecad0eac790dd89362227310fd30'
    if sha(path/'flax_model.msgpack')!=expected:raise ValueError('tiny checkpoint hash mismatch')
    if sha(path/'config.json')!='77a9e9830c3abeba929f5c61e0e97c398f98b4481ad75516c0be5bf038b340b0':raise ValueError('config hash mismatch')
    def ext(code,data):
        if code!=1:raise ValueError('only data-only Flax ndarray extension is allowed')
        shape,dtype,raw=msgpack.unpackb(data,raw=False)
        if dtype!='float32' or not isinstance(shape,list) or any(type(x)is not int or x<0 for x in shape):raise ValueError('unsupported serialized array')
        return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
    tree=msgpack.unpackb((path/'flax_model.msgpack').read_bytes(),raw=False,ext_hook=ext);weights={}
    def walk(tree,prefix=''):
        for key,value in tree.items():
            name=prefix+key
            if isinstance(value,dict):walk(value,name+'.')
            else:
                if name.endswith('.kernel'):name=name[:-7]+'.weight';value=value.T.copy()
                elif name.endswith('.embedding'):name=name[:-10]+'.weight'
                elif name.endswith('.scale'):name=name[:-6]+'.weight'
                weights[name]=value
    walk(tree);np.savez(path/'weights.npz',**weights)
    # Both tokenizers must assign exactly the same IDs before reusing tokenized data.
    if json.loads((path/'vocab.json').read_text())!=json.loads((source/'vocab.json').read_text()):raise ValueError('tokenizer vocabularies differ')
    original=json.loads((source/'inputs.json').read_text());inp={'prompt_token_ids':[original['prompt_token_ids'][0][:4]],'evaluation_token_ids':[original['evaluation_token_ids'][0][:17]]};write(path/'inputs.json',inp)
    dataset=json.loads((source/'dataset.json').read_text());dataset.update({'selection':'first 17 BPE tokens of the already pinned first WikiText test prefix; one independent sequence; 16 targets','evaluation_token_sha256':hashlib.sha256(json.dumps(inp['evaluation_token_ids'],separators=(',',':')).encode()).hexdigest(),'model_checkpoint':'sshleifer/tiny-gpt2','model_revision':'5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be','weights_sha256':expected,'quality_scope':'tiny public checkpoint, functional smoke fixture only; no meaningful pretrained language-quality claim','checkpoint_conversion':'FlaxConv1D stores [out,in] kernel and transposes on call; transpose every .kernel to GPT2 Conv1D [in,out]; embedding/scale renamed weight','reference':{'transformers':transformers.__version__,'torch':torch.__version__,'numpy':np.__version__,'dtype':'float32','device':'cpu','attention':'eager'}});write(path/'dataset.json',dataset)
    cfg=GPT2Config.from_pretrained(path,local_files_only=True);cfg._attn_implementation='eager';torch.set_num_threads(1);model=GPT2LMHeadModel(cfg)
    missing=model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()},strict=False)
    if set(missing.missing_keys)!={'lm_head.weight'} or missing.unexpected_keys:raise ValueError('checkpoint coverage mismatch')
    model.tie_weights();model.eval();ref={};steps=[];tokens=[];full=[];ids=torch.tensor(inp['prompt_token_ids'],dtype=torch.long)
    with torch.no_grad():
        result=model(ids,use_cache=True);allids=ids
        for i in range(2):
            logits=result.logits[:,-1];token=logits.argmax(-1);steps.append(logits.numpy());tokens.append(token.numpy());full.append(model(allids,use_cache=False).logits[:,-1].numpy())
            if i==0:allids=torch.cat([allids,token[:,None]],dim=1);result=model(token[:,None],past_key_values=result.past_key_values,use_cache=True)
        ref['generation_logits']=np.stack(steps,axis=1);ref['generated_token_ids']=np.stack(tokens,axis=1);ref['full_recompute_logits']=np.stack(full,axis=1);ref['evaluation_logits']=model(torch.tensor(inp['evaluation_token_ids']),use_cache=False).logits.numpy()
        for i,pair in enumerate(result.past_key_values):
            for kind,array in zip(('k','v'),pair):ref['cache_%d_%s'%(i,kind)]=array.numpy()
    np.savez(path/'reference.npz',**ref)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--source-assets',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--reuse-reference',action='store_true');parser.add_argument('--download',action='store_true');args=parser.parse_args();assets=args.assets.resolve()
    if args.download:download_assets(assets)
    if not args.reuse_reference:prepare_assets(assets,args.source_assets.resolve())
    from pim_foundry import application as app
    from pim_foundry import transformer_application as tr
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest={'schema':app.APP_SCHEMA,'id':'tiny-gpt2-complete-native-hbm-functional-fixture','adapter':tr.ADAPTER,'assets':{name:app.asset(assets/file,role,fmt) for name,file,role,fmt in [('model','weights.npz','model','npz'),('config','config.json','model','json'),('inputs','inputs.json','inputs','json'),('reference','reference.npz','quality','npz'),('dataset','dataset.json','quality','json')]},'parameters':{'architecture':'gpt2','max_new_tokens':2},'scenario':tr.SCENARIO,'quality':{'kind':'reference_and_next_token_nll','reference':'transformers-4.44.2-cpu-float32-eager','atol':0.002,'rtol':0.002,'max_nll_delta':0.001}}
    from pim_foundry.unified_examples import platform as make_platform
    platform=make_platform(('hbm','gpu'),root=ROOT)
    selected=['layer.%d.%s.matmul'%(i,p) for i in range(2) for p in ('qkv','attn.proj','mlp.fc','mlp.proj')]
    for device in platform['devices']:device['capacity_bytes']=64*1024**2
    experiment={'schema':app.EXPERIMENT_SCHEMA,'platform':platform,'mapping':{'strategy':'hybrid_linear','default_device':'gpu','fp16_device':'hbm'},'variant':{'fp16_matmul_nodes':selected}}
    write(out/'application.json',manifest);write(out/'experiment.json',experiment);result=app.run(out/'application.json',out/'experiment.json',out/'run');replay=app.replay(out/'run',out/'replay')
    baseline_experiment={**experiment,'mapping':{'strategy':'all_to_device','device':'gpu'}};write(out/'baseline-experiment.json',baseline_experiment)
    baseline=app.run(out/'application.json',out/'baseline-experiment.json',out/'baseline');baseline_replay=app.replay(out/'baseline',out/'baseline-replay')
    comparison=app.compare(out/'run',out/'baseline',metric='generation_modeled_latency_ns');write(out/'comparison.json',comparison)
    write(out/'summary.json',{'result':result,'replay':replay,'baseline':baseline,'baseline_replay':baseline_replay,'comparison':comparison,'verified':[app.verify(out/name) for name in ('run','replay','baseline','baseline-replay')],'scope':'complete tiny GPT-2 through native HBM for eight explicit FP16 linear nodes per forward, CPU-functional/analytical GPU for other nodes; numerical/fixture NLL vs FP32 HF; no real hardware or general-quality claim'})
    print(json.dumps({'status':result['status'],'quality':result['quality'],'metrics':result['metrics']},indent=2))
if __name__=='__main__':main()
