#!/usr/bin/env python3
"""Render recorded architecture schedules; never invent or re-run observations.

Requires Matplotlib for the static chart. The HTML viewer is self-contained and
uses the same checked-in JSON. No network requests or model execution occur.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'pim-foundry-gallery-mpl'))


def validate(data):
    if data.get('schema') != 'pim-foundry.architecture-gallery.v1' or not data.get('passed'):
        raise ValueError('expected a successful architecture gallery summary')
    scenarios = data['scenarios']
    if len({row['id'] for row in scenarios}) != len(scenarios):
        raise ValueError('duplicate scenario ids')
    for row in scenarios:
        if row['numerical']['status'] != 'passed' or not row['replay']['passed']:
            raise ValueError('unverified scenario')
        events = row['events']
        if sum(e['bytes'] for e in events if e['kind'] == 'transfer') != row['transfer_bytes']:
            raise ValueError('transfer total differs from event evidence')
        resources = {r for e in events for r in e['resources']}
        for resource in resources:
            intervals = sorted((e['start_ns'], e['end_ns']) for e in events
                               if resource in e['resources'] and e['end_ns'] > e['start_ns'])
            if any(a[1] > b[0] + 1e-9 for a, b in zip(intervals, intervals[1:])):
                raise ValueError('resource overlap: ' + resource)
    return scenarios


def chart(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'pim-foundry-architecture-gallery-v1'})
    scenarios = data['scenarios']
    selected = next(row for row in scenarios if row['id'] == 'sram_hbm_gpu')
    navy, muted, teal, background = '#152a43', '#52667b', '#087f8c', '#f4f7fb'
    colors = ['#2563eb', '#087f8c', '#b56316']
    fig = plt.figure(figsize=(16, 10), facecolor=background)
    fig.text(.035, .952, 'PIM FOUNDRY  /  RECORDED RESOURCE SCHEDULE', color=teal, fontsize=13, weight='bold')
    fig.text(.035, .900, 'See where compute overlaps—and where transfers wait.', color=navy, fontsize=25, weight='bold')
    fig.text(.035, .858, 'SRAM-CIM + HBM-PIM + GPU · three requests arriving at t = 0 · shared warm weights', color=muted, fontsize=14)
    lanes = ['compute:sram', 'compute:hbm', 'compute:gpu', 'dma:host', 'dma:sram', 'dma:hbm', 'dma:gpu']
    labels = ['SRAM compute', 'HBM compute', 'GPU compute', 'Host DMA', 'SRAM DMA', 'HBM DMA', 'GPU DMA']
    ax = fig.add_axes((.14, .390, .82, .395), facecolor=background)
    for index, resource in enumerate(lanes):
        for event in selected['events']:
            if resource not in event['resources']:
                continue
            ax.broken_barh([(event['start_ns'], event['end_ns'] - event['start_ns'])],
                           (index - .32, .64), facecolors=colors[event['request']],
                           edgecolors='white', linewidth=.4,
                           hatch='//' if event['kind'] == 'transfer' else None)
    ax.set_yticks(range(len(lanes)), labels, fontsize=12, color=navy)
    ax.invert_yaxis()
    ax.set_xlim(0, selected['modeled_latency_ns'])
    ax.set_xticks(range(0, int(selected['modeled_latency_ns']) + 1, 25))
    ax.set_xlabel('Predicted time (ns) · event start/end values read directly from the recorded schedule', fontsize=12, color=muted, labelpad=12)
    ax.tick_params(length=0, pad=8, labelcolor=muted)
    ax.grid(axis='x', color='#dce5ee', linewidth=.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [Patch(facecolor=c, label=f'Request {i+1}') for i, c in enumerate(colors)]
    legend.append(Patch(facecolor='white', edgecolor=muted, hatch='//', label='Transfer reservation'))
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.14, .839), ncol=4, frameon=False, fontsize=12)
    fig.text(.035, .311, 'A transfer reserves its directed link and BOTH endpoint DMA resources.', fontsize=13, color=navy, weight='bold')
    fig.text(.035, .280, 'The same transfer appears on two DMA rows; these are shared reservations, not two transfers. Link rows are omitted.', fontsize=11, color=muted)
    headers = ['Active placement', 'Stream makespan', 'Serial admission', 'Bytes across links']
    columns = [.035, .43, .61, .80]
    for x, label in zip(columns, headers):
        fig.text(x, .229, label, fontsize=11, color=muted, weight='bold')
    for i, row in enumerate(scenarios):
        vals = [row['label'], f"{row['modeled_latency_ns']:g} ns", f"{row['serial_reference']['modeled_latency_ns']:g} ns", f"{row['transfer_bytes']:,} B"]
        for x, val in zip(columns, vals):
            fig.text(x, .199 - i * .029, val, fontsize=12, color=navy)
    fig.text(.035, .058, 'Same platform and workload; equal synthetic coefficients. One primary profile per placement; verified replay.', fontsize=11, color=muted)
    fig.text(.035, .031, 'Source: validation/architecture-gallery.json · modeled stream, not hardware timing or independent performance samples.', fontsize=10.5, color=muted)
    description = 'Recorded resource events from validation/architecture-gallery.json. Separate analytical scheduling prediction; no hardware measurement.'
    fig.savefig(out / 'architecture-timeline.png', dpi=150, metadata={'Software': 'PIM Foundry', 'Description': description})
    svg = out / 'architecture-timeline.svg'
    fig.savefig(svg, metadata={'Date': None, 'Creator': 'PIM Foundry', 'Description': description})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PIM Foundry · Architecture explorer</title>
<style>
:root{color-scheme:light;--ink:#152a43;--muted:#52667b;--line:#dce5ee;--teal:#087f8c}*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:16px/1.6 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:40px 28px 60px}h1{font-size:clamp(26px,4vw,42px);line-height:1.2;margin:8px 0 16px}h2{font-size:21px;margin:0 0 12px}.eyebrow{color:var(--teal);font-size:13px;font-weight:700;letter-spacing:.08em}.muted{color:var(--muted)}.tabs{display:flex;gap:9px;flex-wrap:wrap;margin:25px 0 18px}button,select{font:inherit;border:1px solid var(--line);border-radius:8px;padding:9px 14px;background:#fff;color:var(--ink)}button{cursor:pointer}button[aria-pressed="true"]{background:var(--ink);color:#fff}button:focus-visible,select:focus-visible{outline:3px solid #60a5fa;outline-offset:3px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat,section{background:white;border:1px solid var(--line);border-radius:13px;padding:20px}.stat strong{display:block;font-size:26px;color:var(--teal)}.stat span{font-size:13px;color:var(--muted)}section{margin-top:18px}.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.node{background:#f4f7fb;padding:12px;border-radius:8px;border-top:4px solid var(--teal)}.node strong,.node small{display:block}.node strong{font-size:15px}.node small{font-size:12px;color:var(--muted)}.device{color:var(--teal);font-weight:700}.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap}.controls label{font-size:14px}.scroll{overflow-x:auto}.timeline{min-width:980px;width:100%;display:block}.event{cursor:pointer}.event:hover,.event:focus{stroke:#111;stroke-width:2;outline:none}.detail{background:#f4f7fb;border-radius:8px;padding:12px 16px;margin-top:12px;min-height:68px;white-space:pre-wrap;font:14px/1.6 ui-monospace,monospace}.legend{display:flex;flex-wrap:wrap;gap:20px;font-size:13px;margin:14px 0}.swatch{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:3px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}th{color:var(--muted)}.foot{font-size:13px;color:var(--muted);overflow-wrap:anywhere}.detail{overflow-wrap:anywhere}code{background:#eef3f7;padding:2px 5px;border-radius:4px}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.two section{min-width:0}@media(max-width:800px){main{padding:24px 15px}.stats{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}.stat strong{font-size:23px}}
</style>
<main><div class="eyebrow">PIM FOUNDRY / EXECUTED EXAMPLE</div>
<h1>Follow the tensors. Inspect the schedule.</h1>
<p class="muted">Choose where six operations run. The input, precision and platform stay fixed; the actual numerical output is checked, then compute and transfers are scheduled under explicit model assumptions.</p>
<div class="tabs" id="tabs" aria-label="Choose an active placement"></div>
<div class="stats" id="stats"></div>
<section><h2>Operation placement</h2><div class="flow" id="flow"></div><p class="foot">Left to right: input INT8 [2,3,4] → output FP32 [2,3,2]. All four device profiles remain declared in every case; only placement changes. Weights upload once to their destinations and remain resident.</p></section>
<section><h2>Compute and transfer reservations</h2><div class="controls"><label>Request <select id="request"><option value="all">All requests</option><option value="0">Request 1</option><option value="1">Request 2</option><option value="2">Request 3</option></select></label><label><input id="links" type="checkbox"> Show directed link rows</label><span class="foot">Select a bar for its recorded event.</span></div><div class="legend"><span><i class="swatch" style="background:#2563eb"></i>Request 1</span><span><i class="swatch" style="background:#087f8c"></i>Request 2</span><span><i class="swatch" style="background:#b56316"></i>Request 3</span></div><div class="scroll" id="timeline"></div><div id="detail" class="detail" aria-live="polite">Select any event to inspect its operation, tensor, bytes, timestamps and reserved resources.</div><p class="foot">One transfer occupies a directed link and both endpoint DMA resources. Its repeated bars represent the same reservation. Compute has separate resources and may overlap DMA; transfers sharing an endpoint serialize. Filtering dims other requests without changing the recorded schedule.</p></section>
<div class="two"><section><h2>Peak modeled device memory</h2><table><thead><tr><th>Device</th><th>Peak live bytes</th><th>Capacity</th></tr></thead><tbody id="memory"></tbody></table><p class="foot">Includes resident copies, live outputs and workspace. These are recorded peaks, not a reconstructed memory timeline. Capacity overflow is rejected; there is no eviction.</p></section><section><h2>Tensor traffic on directed links</h2><table><thead><tr><th>Route</th><th>Bytes</th></tr></thead><tbody id="traffic"></tbody></table><p class="foot">Counts bytes on each link traversal. The first warm weight upload is included; CPU diagnostic arrays are separate from modeled residency.</p></section></div>
<section><h2>What was actually run</h2><p>Each placement evaluates the real tensor graph once on the CPU, checks an independent scalar oracle, verifies a sealed bundle, and replays it. One operation profile predicts three identical requests arriving at t = 0 with at most three admitted at a time.</p><p>All devices use the same illustrative coefficients: 16 MAC/ns, 16 elements/ns, 64 local bytes/ns, 10 ns launch cost and 4 KiB capacity. Each directed link uses 16 bytes/ns plus 5 ns latency. Energy coefficients are zero assumptions and no energy result is claimed.</p><p>The timing is an uncalibrated analytical prediction. This example executes no native simulator or physical accelerator and does not evaluate task quality. Native SRAM and HBM adapters are separate supported interfaces.</p><button id="download">Download the recorded JSON</button><p class="foot" id="source"></p></section></main>
<script type="application/json" id="data">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent),colors=['#2563eb','#087f8c','#b56316'];
let current=data.scenarios.find(x=>x.id==='sram_hbm_gpu')||data.scenarios[0];
const nodes=[['integer_projection','1 · Integer projection','INT8 × INT8 → INT32'],['activation','2 · ReLU','INT32 → INT32'],['to_fp16','3 · Cast','INT32 → FP16'],['half_projection','4 · FP16 projection','FP16 × FP16 → FP16'],['to_fp32','5 · Cast','FP16 → FP32'],['probabilities','6 · Softmax','FP32 → FP32']];
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function fmt(x){return Number(x).toLocaleString('en-US',{maximumFractionDigits:3})}
for(const s of data.scenarios){const b=el('button',s.label);b.dataset.id=s.id;b.addEventListener('click',()=>{current=s;render()});document.getElementById('tabs').append(b)}
function cellRow(values){const tr=el('tr');for(const value of values)tr.append(el('td',String(value)));return tr}
function render(){
 for(const b of document.querySelectorAll('#tabs button'))b.setAttribute('aria-pressed',String(b.dataset.id===current.id));
 const stats=document.getElementById('stats');stats.replaceChildren();
 const err=Math.max(...Object.values(current.numerical.checks).map(x=>x.max_abs_error||0));
 for(const [value,label] of [[fmt(current.modeled_latency_ns)+' ns','Predicted stream makespan'],[fmt(current.transfer_bytes)+' B','Bytes across all links'],[err.toExponential(2),'Maximum output error'],[current.replay.passed?'Verified':'Failed','Sealed-bundle replay']]){const c=el('div',undefined,'stat');c.append(el('strong',value),el('span',label));stats.append(c)}
 const flow=document.getElementById('flow');flow.replaceChildren();for(const [id,title,type] of nodes){const n=el('div',undefined,'node');n.append(el('strong',title),el('span',current.placement[id].toUpperCase(),'device'),el('small',type));flow.append(n)}
 const mem=document.getElementById('memory');mem.replaceChildren();for(const d of data.platform.devices)mem.append(cellRow([d.name.toUpperCase(),fmt(current.peak_device_bytes[d.name])+' B',fmt(d.capacity_bytes)+' B']));
 const traffic=document.getElementById('traffic');traffic.replaceChildren();for(const [route,bytes] of Object.entries(current.transfer_breakdown.by_route))traffic.append(cellRow([route.replace('->',' → '),fmt(bytes)+' B']));
 document.getElementById('detail').textContent='Select an event to inspect its recorded values. Serial admission of the same placement predicts '+fmt(current.serial_reference.modeled_latency_ns)+' ns; this is a separate admission-limit counterfactual.';
 draw();
}
function draw(){
 const ns='http://www.w3.org/2000/svg';function svgEl(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}
 const showLinks=document.getElementById('links').checked,filter=document.getElementById('request').value;
 const resources=[...new Set(current.events.flatMap(e=>e.resources))].filter(r=>showLinks||!r.startsWith('link:')).sort((a,b)=>{const rank=r=>r.startsWith('compute:')?0:r.startsWith('dma:')?1:2;return rank(a)-rank(b)||a.localeCompare(b)});
 const left=178,width=970,row=38,top=44,height=top+resources.length*row+35,end=current.modeled_latency_ns;
 const svg=svgEl('svg',{viewBox:`0 0 1180 ${height}`,class:'timeline',role:'img','aria-label':current.label+' resource reservations'});
 for(let i=0;i<=5;i++){const x=left+i*width/5;svg.append(svgEl('line',{x1:x,x2:x,y1:28,y2:height-26,stroke:'#dce5ee'}),svgEl('text',{x,y:19,fill:'#52667b','font-size':12,'text-anchor':'middle'},fmt(i*end/5)+' ns'))}
 resources.forEach((r,i)=>svg.append(svgEl('text',{x:left-14,y:top+i*row+18,fill:'#152a43','font-size':13,'text-anchor':'end'},r.replaceAll(':',' / '))));
 for(const e of current.events){for(const r of e.resources){const index=resources.indexOf(r);if(index<0)continue;const x=left+e.start_ns/end*width,w=Math.max(.8,(e.end_ns-e.start_ns)/end*width);const rect=svgEl('rect',{x,y:top+index*row,width:w,height:26,rx:3,fill:colors[e.request],opacity:filter==='all'||Number(filter)===e.request?1:.15,class:'event',tabindex:0,role:'button','aria-label':`${e.kind} request ${e.request+1} ${e.node||e.tensor}, ${e.start_ns} to ${e.end_ns} ns`});const detail=JSON.stringify(e,null,2);rect.append(svgEl('title',{},detail));const select=()=>{document.getElementById('detail').textContent='Request '+(e.request+1)+' · '+(e.kind==='transfer'?'Transfer '+e.tensor+' ('+e.bytes+' B)':'Compute '+e.node)+'\n'+(e.kind==='transfer'?'Route: '+e.source.toUpperCase()+' → '+e.destination.toUpperCase():'Device: '+e.device.toUpperCase())+'\nTime: '+fmt(e.start_ns)+'–'+fmt(e.end_ns)+' ns · duration '+fmt(e.end_ns-e.start_ns)+' ns\nReserved resources: '+e.resources.join(', ')};rect.addEventListener('click',select);rect.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select()}});svg.append(rect)}}
 document.getElementById('timeline').replaceChildren(svg);
}
document.getElementById('request').addEventListener('change',draw);document.getElementById('links').addEventListener('change',draw);
document.getElementById('source').textContent='Source: validation/architecture-gallery.json · example SHA-256 '+data.evidence.source_sha256+' · all displayed events come from the recorded summary.';
document.getElementById('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='architecture-gallery.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    validate(data)
    args.out.mkdir(parents=True, exist_ok=True)
    chart(data, args.out)
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (args.out / 'architecture-explorer.html').write_text(PAGE.replace('__DATA__', payload))
    print(json.dumps({'source_sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                      'files': ['architecture-timeline.png', 'architecture-timeline.svg', 'architecture-explorer.html']}))


if __name__ == '__main__':
    main()
