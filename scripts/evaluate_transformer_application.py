#!/usr/bin/env python3
"""Pinned pretrained DistilGPT2 / WikiText fixture, independent HF reference, app replay.

Use the task-local CPU reference environment. No model repository code is executed.
Downloads are explicit (--download); all application executions and replays are offline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MODEL_REVISION='2290a62682d06624634c1f46a6ad5be0f47f38aa'
DATASET_REVISION='b08601e04326c79dfdd32d625aee71d232d685c3'
FILES={
 'config.json':'4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a',
 'model.safetensors':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
 'merges.txt':'1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
 'vocab.json':'196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
 'tokenizer.json':'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
 'tokenizer_config.json':'5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
 'wikitext-test.parquet':'5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def prepare_assets(directory,download=False):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import transformers
    from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
    from safetensors.numpy import load_file
    if transformers.__version__!='4.44.2':raise RuntimeError('reference requires transformers==4.44.2')
    directory.mkdir(parents=True,exist_ok=True)
    for name,digest in FILES.items():
        path=directory/name
        if not path.exists() and download:
            import urllib.request
            base=('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'+DATASET_REVISION+'/wikitext-2-raw-v1/test-00000-of-00001.parquet' if name=='wikitext-test.parquet' else 'https://huggingface.co/distilbert/distilgpt2/resolve/'+MODEL_REVISION+'/'+name)
            urllib.request.urlretrieve(base,path)
        if not path.is_file() or sha(path)!=digest:raise ValueError('missing or modified pinned asset '+str(path))
    torch.set_num_threads(4);torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False)
    table=pq.read_table(directory/'wikitext-test.parquet');texts=table.column('text').to_pylist()
    # Fixed prefix including headings/newlines; no test-data-dependent selection.
    text='\n'.join(texts[:32]);tokens=tokenizer.encode(text,add_special_tokens=False)
    if len(tokens)<130:raise ValueError('fixed test prefix unexpectedly too short')
    evaluation=[tokens[:65],tokens[65:130]];prompt=[tokens[:16],tokens[65:81]]
    inputs={'prompt_token_ids':prompt,'evaluation_token_ids':evaluation}
    write(directory/'inputs.json',inputs)
    dataset={'dataset':'Salesforce/wikitext','configuration':'wikitext-2-raw-v1','split':'test','revision':DATASET_REVISION,'source_sha256':FILES['wikitext-test.parquet'],'selection':'join first 32 rows with newline; first 130 BPE tokens; two independent sequences of 65; score 64 targets each','raw_prefix_sha256':hashlib.sha256(text.encode()).hexdigest(),'evaluation_token_sha256':hashlib.sha256(json.dumps(evaluation,separators=(',',':')).encode()).hexdigest(),'tokenizer_model':'distilbert/distilgpt2','tokenizer_revision':MODEL_REVISION,'tokenizer_asset_hashes':{k:v for k,v in FILES.items() if k in ('tokenizer.json','tokenizer_config.json','merges.txt','vocab.json')},'license':'CC-BY-SA-3.0 / GFDL (WikiText source); attribution in dataset upstream','quality_scope':'128 next-token targets from one fixed test prefix; not full WikiText perplexity','model_checkpoint':'distilbert/distilgpt2','model_revision':MODEL_REVISION,'weights_sha256':FILES['model.safetensors'],'reference':{'transformers':transformers.__version__,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','dtype':'float32','attention_implementation':'eager','dropout':'eval'}}
    write(directory/'dataset.json',dataset)
    weights=load_file(str(directory/'model.safetensors'));np.savez(directory/'weights.npz',**weights)
    config=GPT2Config.from_pretrained(str(directory),local_files_only=True);config._attn_implementation='eager'
    model=GPT2LMHeadModel(config)
    buffers=[k for k in weights if k.endswith('.attn.bias')]
    for key in buffers:
        mask=weights[key]
        if not np.array_equal(mask,np.tril(np.ones_like(mask))):raise ValueError('unexpected stored causal mask '+key)
    missing=model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items() if k not in buffers},strict=False)
    model.tie_weights();model.eval()
    if set(missing.missing_keys)-{'lm_head.weight'} or missing.unexpected_keys:raise ValueError('reference checkpoint coverage mismatch '+str(missing))
    refs={};generated=[];step_logits=[];recompute=[];ids=torch.tensor(prompt,dtype=torch.long)
    with torch.no_grad():
        result=model(ids,use_cache=True);all_ids=ids
        for step in range(4):
            logits=result.logits[:,-1,:];step_logits.append(logits.numpy());token=logits.argmax(-1);generated.append(token.numpy())
            recompute.append(model(all_ids,use_cache=False).logits[:,-1,:].numpy())
            if step<3:
                all_ids=torch.cat([all_ids,token[:,None]],dim=1);result=model(token[:,None],past_key_values=result.past_key_values,use_cache=True)
        refs['generation_logits']=np.stack(step_logits,axis=1);refs['generated_token_ids']=np.stack(generated,axis=1);refs['full_recompute_logits']=np.stack(recompute,axis=1)
        for layer,pair in enumerate(result.past_key_values):
            for key,array in zip(('k','v'),pair):refs['cache_%d_%s'%(layer,key)]=array.numpy()
        refs['evaluation_logits']=model(torch.tensor(evaluation,dtype=torch.long),use_cache=False).logits.numpy()
    np.savez(directory/'reference.npz',**refs)
    write(directory/'reference-generation.json',{'reference':dataset['reference'],'max_cache_vs_recompute_error':float(np.abs(refs['generation_logits']-refs['full_recompute_logits']).max()),'generated_text':[tokenizer.decode(row) for row in refs['generated_token_ids'].tolist()],'asset_hashes':{name:sha(directory/name) for name in ['weights.npz','inputs.json','dataset.json','reference.npz']}})
    return dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--out',type=Path);parser.add_argument('--download',action='store_true');parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--reuse-reference',action='store_true');parser.add_argument('--platform',type=Path);parser.add_argument('--device');args=parser.parse_args()
    assets=args.assets.resolve()
    if not args.reuse_reference:prepare_assets(assets,args.download)
    if args.prepare_only:return
    if not args.out:parser.error('--out is required unless --prepare-only')
    from pim_foundry import application as app
    from pim_foundry import transformer_application as tr
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest={'schema':app.APP_SCHEMA,'id':'distilgpt2-wikitext-test-prefix','adapter':tr.ADAPTER,'assets':{name:app.asset(assets/file,role,fmt) for name,file,role,fmt in [('model','weights.npz','model','npz'),('config','config.json','model','json'),('inputs','inputs.json','inputs','json'),('reference','reference.npz','quality','npz'),('dataset','dataset.json','quality','json')]},'parameters':{'architecture':'gpt2','max_new_tokens':4},'scenario':tr.SCENARIO,'quality':{'kind':'reference_and_next_token_nll','reference':'transformers-4.44.2-cpu-float32-eager','atol':0.0003,'rtol':0.0003,'max_nll_delta':0.0001}}
    experiment={'schema':app.EXPERIMENT_SCHEMA,'platform':{'kind':'tensor-reference-cpu/v1'},'mapping':{'strategy':'reference_cpu'},'variant':{}}
    if args.platform:
        platform=json.loads(args.platform.read_text());device=args.device or next(d['name'] for d in platform['devices'] if d['backend']=='analytical')
        experiment={'schema':app.EXPERIMENT_SCHEMA,'platform':platform,'mapping':{'strategy':'all_to_device','device':device},'variant':{}}
    write(out/'application.json',manifest);write(out/'experiment.json',experiment)
    result=app.run(out/'application.json',out/'experiment.json',out/'run');replayed=app.replay(out/'run',out/'replay')
    summary={'application':result,'replay':replayed,'verified':[app.verify(out/name) for name in ('run','replay')],'scope':'complete pretrained six-layer DistilGPT2; two prompts; four generated tokens each; fixed 128-target WikiText fixture; CPU functional execution; no hardware timing claim'}
    write(out/'summary.json',summary);print(json.dumps({'status':result['status'],'metrics':result['metrics'],'quality':result['quality']},indent=2))
if __name__=='__main__':main()
