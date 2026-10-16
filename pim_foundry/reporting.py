"""Standalone offline report with model assumptions and an inspectable timeline."""

import html
import json
import re
import statistics


def render_report(suite):
    rows, traces = [], []
    for case in suite["cases"]:
        for run in case["runs"]:
            simulation = run["result"]["simulation"]
            label = case["name"] + " / " + run["label"]
            cells = (case["name"], run["label"],
                     f"{statistics.median(simulation['latency_ns']):,.1f}",
                     f"{simulation['makespan_ns']:,}",
                     f"{simulation['throughput_requests_per_second']:,.2f}",
                     f"{simulation['energy']['total_pj'] / 1e6:,.3f}",
                     f"{simulation['logical_transfer_bytes']:,}")
            rows.append("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in cells) + "</tr>")
            traces.append({"label": label, "timeline": simulation["timeline"],
                           "assumptions": simulation["assumptions"],
                           "requests": simulation["requests"],
                           "peak_reserved_bytes": simulation["peak_reserved_bytes"]})
    payload = json.dumps(traces, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIM Foundry · 異質系統實驗</title>
<style>
:root{color-scheme:light;--ink:#153044;--muted:#536a7a;--line:#d7e0e7;--accent:#176c80}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.65 system-ui,sans-serif}
main{max-width:1360px;margin:38px auto;padding:0 24px 70px}h1{font-size:32px;line-height:1.25}h2{font-size:21px;margin:0 0 12px}
p{max-width:1000px}.eyebrow{letter-spacing:.14em;font-size:12px;color:var(--accent);font-weight:700}.notice{border-left:4px solid #ae6c13;background:#fff6e5;padding:15px 20px}
section{margin-top:24px;padding:22px;background:white;border:1px solid var(--line);border-radius:10px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{text-align:right;padding:10px;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--muted);font-weight:600}a{color:var(--accent)}select,input{font:inherit;padding:5px;max-width:100%}label{display:inline-block;margin:8px 16px 8px 0}.small{font-size:13px;color:var(--muted)}pre{white-space:pre-wrap;word-break:break-word;font-size:13px}svg text{font-family:system-ui,sans-serif}#timeline{overflow:auto;border:1px solid var(--line)}.legend{display:flex;gap:22px;flex-wrap:wrap}.legend span:before{content:'';display:inline-block;width:10px;height:10px;margin-right:6px;background:var(--c)}
</style><main><div class="eyebrow">PIM FOUNDRY / REPRODUCIBLE RESEARCH</div>
<h1>同一份運算，檢查不同裝置分工</h1>
<p>本報告沿著每筆請求的輸入搬移、權重載入、計算與結果回傳，記錄資源占用與預測時間。不同請求可以重疊；同一請求內的節點依序執行。</p>
<p class="notice"><strong>分析模型，尚未校準。</strong>下列數字依設定檔的成本與假設計算，不能當成真實 GPU、NPU 或 CIM／PIM 的效能或能耗。功能輸出已逐項對照整數參考。數值相同不代表效能模型正確。</p>
<p class="small">設定：<strong>CONFIG_NAME</strong> · <a href="config.json">裝置與成本模型</a> · <a href="summary.json">實驗摘要</a> · <a href="manifest.json">檔案雜湊清單</a></p>
<section><h2>模型預測</h2><p class="small">各配置保留相同平台，包括閒置裝置。範圍是 device graph，包含模型內的排隊與搬移；不含應用前後處理、編譯與 host 軟體開銷。暖啟動另排除預載成本。表格按實驗順序排列。</p><div class="scroll"><table><thead><tr><th>工作負載</th><th>配置</th><th>延遲中位數 ns</th><th>總完成時間 ns</th><th>請求／秒</th><th>估計能量 µJ</th><th>搬移 bytes</th></tr></thead><tbody>ROWS</tbody></table></div></section>
<section><h2>請求與資源時間軸</h2><label>選擇實驗 <select id="case"></select></label><label>縮放 <input id="zoom" type="range" min="1" max="12" value="1"></label>
<div class="legend small"><span style="--c:#387a9c">傳輸</span><span style="--c:#4c8b68">計算</span><span style="--c:#b58231">啟動／其他事件</span></div>
<p id="timeline-status" class="small"></p><div id="timeline"></div>
<details><summary>請求完成時間、保留容量與假設</summary><pre id="details"></pre></details></section>
<section><h2>如何檢查這份實驗</h2><p>每個 run 都有完整功能結果、時間軸、評測紀錄和 Chrome Trace。共同比較器會拒絕缺少校準證據的物理效能比較。使用 <code>verify-bundle</code> 檢查檔案一致性，使用 <code>replay</code> 重新計算並比對結果；雜湊一致性不代表量測已經外部稽核。</p></section>
</main><script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),sel=document.getElementById('case'),zoom=document.getElementById('zoom');
data.forEach((item,i)=>{const o=document.createElement('option');o.value=i;o.textContent=item.label;sel.append(o)});
const ns='http://www.w3.org/2000/svg';function el(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n}
function draw(){const d=data[Number(sel.value)||0];if(!d)return;const timeline=d.timeline,events=timeline.slice(0,4000),lanes=[...new Set(events.flatMap(e=>e.resources))].sort();const end=timeline.reduce((m,e)=>Math.max(m,e.end_ns),1);const width=1100*Number(zoom.value),left=210,height=70+lanes.length*40;const svg=el('svg',{width:width+left,height,role:'img','aria-label':d.label+' 資源時間軸'});
for(let i=0;i<=8;i++){let x=left+i*width/8;svg.append(el('line',{x1:x,x2:x,y1:30,y2:height,stroke:'#e1e7eb'}));svg.append(el('text',{x,y:20,fill:'#536a7a','font-size':11},Math.round(i*end/8).toLocaleString()+' ns'))}
lanes.forEach((lane,i)=>svg.append(el('text',{x:8,y:59+i*40,fill:'#153044','font-size':12},lane)));
for(const e of events){for(const resource of e.resources){const row=lanes.indexOf(resource),x=left+e.start_ns/end*width,w=Math.max(.8,(e.end_ns-e.start_ns)/end*width);const color=e.kind.includes('transfer')?'#387a9c':e.kind.includes('compute')?'#4c8b68':'#b58231';const rect=el('rect',{x,y:42+row*40,width:w,height:23,rx:2,fill:color,opacity:.85});rect.append(el('title',{},JSON.stringify(e,null,2)));svg.append(rect)}}
document.getElementById('timeline').replaceChildren(svg);document.getElementById('timeline-status').textContent=timeline.length+' 個事件；'+(timeline.length>4000?'顯示前 4000 個，完整內容見 results／traces。':'游標停在色塊上可查看請求、資料與依賴。');document.getElementById('details').textContent=JSON.stringify({requests:d.requests,peak_reserved_bytes:d.peak_reserved_bytes,assumptions:d.assumptions},null,2)}sel.addEventListener('change',draw);zoom.addEventListener('input',draw);draw();
</script></html>'''
    values = {"ROWS": "".join(rows), "CONFIG_NAME": html.escape(suite["config"]["name"]), "PAYLOAD": payload}
    return re.sub(r"\b(ROWS|CONFIG_NAME|PAYLOAD)\b", lambda match: values[match[0]], page)
