> Paths under `runs/` or `external/` refer to locally generated or separately obtained artifacts, not bundled files. Start with [the public quickstart](getting-started.md); see [validation scope](../VALIDATION.md) for historical observations.

# 論文接入的驗證方法

一個 histogram 程式與 CPU reference 都把 BMP 尾端 metadata 當作像素時，兩者
會一致，卻沒有算對圖像的 histogram。因此每篇 paper 接入後，要同時檢查
「框架有沒有改掉作者程式的結果」及「作者程式有沒有完成宣告的任務」。

## 六個獨立問題

| 驗證項目 | 實際檢查 | 可以支持的結論 |
|---|---|---|
| Artifact 身分 | 論文／作者 repo、revision、必要來源、binary、依賴、輸入與設定 hash | 確定執行的是哪個版本 |
| 接入一致性 | 相同輸入與設定，獨立直接執行與 application adapter 的 deterministic outputs／counters／metrics 相同 | 包裝與解析沒有改掉這次實驗 |
| 任務語意 | 獨立 oracle 或已知答案；針對輸入格式、標籤、索引、邊界做反例 | 被檢查輸入上的問題確實算對；不只複製 reference 的錯誤 |
| 實驗公平性 | 相同語意工作負載、batch、序列長度集合、precision、warm/cold 狀態與計量邊界 | 控制住比較中的指定變因 |
| 模型可信度 | 模擬器 vs 獨立硬體量測／原作者可辨識 figure 的誤差，校準與驗證資料分離 | 在所測範圍的模型準確性；僅成功執行不能支持此項 |
| 可重現與適用範圍 | 空白 cache、來源快照重跑、資源界線、完整／縮小的 scenario 標記 | 其他研究者能重做這次實驗，並知道結果能用到哪裡 |

以上六項不能互相代替。例如，native simulator 的 command counts 相同可以
支持接入一致性；沒有生成 tensor 值時，numerical quality 仍維持未評估。

## 對已發表數字保持批判性的規則

數字先記成作者宣稱，附明確版本、圖表、模型、輸入、精度、batch、baseline
與聚合方式；公開發表或 artifact badge 不會自動將它變成已驗證事實。
作者報告、我們實跑、獨立硬體量測、我們的推論須分開。無法取得原始程式時，
可以做 paper arithmetic audit，但不能把重算圖表或第三方重建叫作作者重現。

每次至少檢查下列實際機制：

- **分子／分母與聚合**：由原始 latency 算出 speedup，區分 ratio of sums、
  mean of ratios 與 geometric mean。利用率要與實得 throughput、宣告的 peak
  及時脈相容；已四捨五入的圖表數字用區間核對，不假裝它們是原始資料。
- **控制變因**：固定容量不代表固定算力、PIM bank 數、頻寬、成本或功耗。
  同時增加 PIM 資源和改變排程的比較，不能隔離排程本身的因果效果。
- **baseline 完整度**：核對精度、模型／tokenizer／checkpoint、融合 kernels、
  batch／request stream、計時範圍、warm-up、重複次數、時脈與版本。未提供
  的項目列為未能查核，不預設 baseline 很強或很弱。
- **時序與能源邊界**：排除的 host、DMA、同步、idle/static energy 要明列。
  納入 static energy 後的比較需各自計算 `E_dynamic + P_static × time`，
  不能只因加速就斷言總能源改善。
- **誤差與統計**：區分作者報告的校準誤差、獨立驗證誤差、run-to-run
  變異及信賴區間。元件的「5% 誤差」不能直接套成整合系統每個案例的
  ±5% 界線。非確定性輸出須保存原始樣本，不能擴大 tolerance 直到通過。
- **功能與效能證據**：軟體 NPU 加實體 PIM 的功能原型，沒有自動驗證完整
  硬體的端到端 latency；BF16、剪枝或修改 head 數後須另有任務品質核對。

結論使用「已核對一致」「在明示假設下不一致」「受比較變因混淆」或
「缺少資料而未能查核」等具體狀態。異常數字、缺少 artifact 或無法重現，
本身都不足以證明造假；同樣不能因數字能算回來就認定實驗真實。

本規則已應用於 [IANUS 原文審查](ianus-paper-audit.md) 及
可離線重算的算術核對 (local artifact: `../runs/ianus-claim-audit-v1/report.html`)：Fig.9 的
3.2× 符合延遲總和之比；Fig.14 的利用率分母則需要釐清。兩種結果同時保留。

## 增加 coverage 時如何選 paper

先找能暴露不同問題的 artifact，再考慮新舊與建置成本：

- **PIMeval/PIMbench，IISWC 2024**：數位 DRAM PIM API 與非神經網路應用。
  Histogram、triangle counting、KNN 能檢查資料格式、圖結構與標籤使用，補足
  目前矩陣圖和 transformer timing 的範圍。這輪鎖定作者明示投稿使用的
  [v1.0.0 release](https://github.com/UVA-LavaLab/PIMeval-PIMbench/releases/tag/v1.0.0)。
- **CIMFlow，DAC 2025**：ONNX→CG mapping→OP compilation→SystemC 的原始
  路徑，用 ResNet 的 Conv、residual 與多 core 同步檢查應用接入。實際稽核
  發現發布的 compiler 省略 pooling／classifier，因此量測範圍限定為 backbone。
  [作者完整框架](https://github.com/BUAA-CI-LAB/CIMFlow) 與既有單 kernel SRAM
  adapter 的驗證層次不同。
- **NeuPIMs，ASPLOS 2024**：NPU 與 DRAMsim3-based PIM 的協作排程。
  [作者 artifact](https://github.com/casys-kaist/NeuPIMs) 使用不同於既有
  STARC／AttAcc 的模擬器路線；baseline 分支的實際 workload 與硬體設定須逐項核對。
- **CENT，ASPLOS 2025**：後續可檢查 CXL、多 PIM devices 與 Llama2 workload。
  [作者 artifact](https://github.com/Yufeng98/CENT) 有 figure scripts 與 source
  data；作者估計完整模擬需要 12–24 小時、約 100 GB，quick profile 仍需數小時。
  這是候選，不能把閱讀 source data 當作已重跑。
- **CrossSim**：後續可檢查 analog programming error、ADC、drift 與任務品質。
  [官方工具](https://github.com/sandialabs/cross-sim) 提供 accuracy simulation；
  它明確不建模能耗、面積或速度，接入時應採品質驗證 track，不產生虛構效能值。

PIMeval 另有 [作者列出的 TACO 2026 延伸論文](https://www.cs.virginia.edu/~skadron/)，
DOI `10.1145/3815581`。IISWC release 的觀測只適用於該 revision，不能未經核對
就延伸成對 TACO 版本或最新 main branch 的結論。

## 不同型態的數字如何保存

共同 application result 的每個 metric 都有 `value`、`unit`、`scope`、
`aggregation`、`origin`。Host wall time 是模擬器執行成本；PIM modeled time 是
裝置模型的輸出。含 host 排序的 KNN，若只有 PIM kernel／copy 的模型時間，
就不能把那個數字稱作完整 KNN latency。

相同 task identity 要綁定語意輸入，而不是檔名。例如兩份標成 batch512 的
request traces 若 sequence lengths 不同，不能只依檔名把它們當成相同 workload。
Channel assignment 可以是方法變體；request lengths 的改變則要反映在任務身分。

目前 `app-compare` 接受同 task、platform、cost contract 與 metric 邊界的
條件式模型比較。換 PIM 架構、改 memory 容量或只保留部分系統成本時，應先
定義新的公平比較協議；不能放寬 hash 檢查來製造可比性。

## 失敗也是必要的驗證資料

每份實驗保留成功或失敗狀態與原始證據。已知不安全的資源配置在 preflight
拒絕；無法完成的原始規模要標示 timeout／resource termination。縮小輸入或
平台仍可測試功能與接入，但使用獨立 scenario／platform identity，不能改名
成原論文規模已完成。

`numerical=passed` 可表示與明示 reference 相同；`task=failed` 則表示獨立
語意檢查已發現反例。兩者可以同時出現，此時 application 的整體狀態是
`quality_failed`，效能比較會被拒絕。沒有做任務評測時用 `not_evaluated`，
不把 script exit 0 提升成品質通過。
