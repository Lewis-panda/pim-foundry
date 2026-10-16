> Paths under `runs/` or `external/` refer to locally generated or separately obtained artifacts, not bundled files. Start with [the public quickstart](getting-started.md); see [validation scope](../VALIDATION.md) for historical observations.

# IANUS：可執行範圍與可信度驗證

本次在 PHSim 跑通兩次相同的 64×64 GEMV 指令模擬，十份非日誌輸出一致；
但這個第三方流程採 LPDDR5，沒有執行矩陣數值，也未確認完整對應 IANUS
的 PAS。原 IANUS 使用 GDDR6 AiM。兩者的名稱相近，不能替代架構與程式
來源的驗證。

查核日期為 2026-10-15。原作者公開的可執行 artifact 尚未定位；ACM supplement
回傳 403，部分作者網站也無法存取，這是搜尋覆蓋的限制，不是「從未公開」
或「結果不實」的證明。本輪沒有新增宣稱重現原 IANUS 的 application adapter。

| 證據 | 實際完成 | 支持範圍 |
|---|---|---|
| 作者原文 | 固定 arXiv:2410.15008v1、PDF hash、逐圖表核對 | 已發表方法與數字的審查；該版明記為會議論文更新版 |
| 公開 artifact 搜尋 | 論文、官方議程、作者機構／實驗室、GitHub、Zenodo、Crossref，保留錯誤與時間 | 目前未定位可認證的作者 artifact，沒有作者原始模擬執行 |
| 第三方 PHSim | 固定 revision、原始碼未修改；原樣 Add 測例一次，衍生 IANUS-layout GEMV 兩次 | 建置、配置及命令路徑可執行；非原論文結果重現 |
| 獨立算術核對 | 8 項算術與顯示精度相容；4 組 Fig.14 分母核對在明示假設下不相容 | 公開數字內部一致性；沒有原始 latency 測量或模型校準 |
| 算術重現 | 封存 script、spec、PDF 後離線重算一致；替換 PDF 被拒絕 | 本次算術審查可重跑；不同於 IANUS 效能重現 |

## 已定位的數字問題與合理結果

- Fig.9 的 3.2× 與九個案例的 **總延遲比** 3.1576 相容；同組資料的
  per-case speedup 算術平均約 13.26、幾何平均約 7.61。必須標明聚合方法。
- Fig.14 的 BERT-B throughput ratio 是 3.1×，而 utilization ratio 是 5.2×。
  若使用 Table 2 的固定峰值 255/184 TFLOPS，以及相同工作負載／聚合權重，
  utilization ratio 應約 4.30×。四組模型都超出一位小數四捨五入所能解釋
  的範圍。不同分母、FLOP 範圍或平均方法仍需原始資料釐清。
- Fig.13 比的是 8 GB PIM 與 4 GB PIM＋4 GB DRAM；PIM throughput 同時
  加倍。這能比較所宣告的整體配置，不能單獨識別排程的效果。
- §6.3 原型使用 NPU 軟體模擬器、真實 AiM 與 FPGA controller；原型的
  功能結果不自動驗證完整硬體的 6.2×／3.2× 模擬加速。
- 「5% 內誤差」是作者對元件模擬器的描述，缺少逐 workload 資料，不能
  當作所有系統預測的信賴區間。dynamic-energy 結果也不能升格為整機實測。

這些發現支持要求原始紀錄與更精確定義，沒有支持造假判定。細節還包括
FFN 權重敘述的 factor-of-two 疑點、修改過的 GPT-2 XL、batch 1 的適用範圍、
多裝置比較與品質驗證資料不足，見下列原文審查。

## 原始資料與重跑入口

- [原文審查與來源位置](ianus-paper-audit.md)
- [Artifact 搜尋、第三方原碼檢查與建置](ianus-artifact-discovery.md)
- 算術核對報告 (local artifact: `../runs/ianus-claim-audit-v1/report.html`)
- 算術重現與變更拒絕檢查 (local artifact: `../runs/ianus-validation/checks.json`)
- 第三方執行與限制 (local artifact: `../runs/ianus-validation/phsim-thirdparty/review.json`)
- 第三方執行命令與退出狀態 (local artifact: `../runs/ianus-validation/phsim-thirdparty/execution-status.json`)
- 第三方重跑檔案核對 (local artifact: `../runs/ianus-validation/phsim-thirdparty/repeat-comparison.json`)

算術審查可在專案根目錄執行：

```sh
python3 scripts/audit_ianus_claims.py --out runs/ianus-arithmetic-new
```

同一份來源快照可在其目錄離線重算：

```sh
python3 audit.py --spec specification.json --paper source.pdf --out NEW_OUTPUT_DIRECTORY
```

已將分母、聚合、固定資源、baseline、統計與證據來源的檢查要求加入
[共用論文驗證規範](paper-validation-protocol.md)。原作者 simulator／compiler、
配置與 GPU baseline 的固定版本、raw timings、校準及品質資料取得後，才能
進行原生整合與完整論文重現。
