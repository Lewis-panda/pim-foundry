# CIM／PIM、GPU、NPU 的共同比較協定草案

> v0.4 已新增分析式事件模擬、矩陣算子、工作負載套件與重現流程；請先看 [使用流程](quickstart.md)、[模擬器契約](simulator.md) 和 [量測驗證](calibration.md)。本文保留 v0.3 的設計與比較規則；其中「目前只有功能模型」描述的是該版本。v0.4 的分析模型仍非硬體量測或週期精確模擬。
HBM-PIM 算完一層後，若下一層交給 GPU 或 NPU，系統仍要搬出結果、調整資料排列、等候傳輸，再執行下一層。只計 PIM 的 MAC 時間，會把這段成本漏掉。因此共同尺度必須固定工作負載、品質要求、請求方式與計時範圍，並讓每個結果說明它來自功能執行、估算、模擬還是硬體量測。

本文件定義 PIM Foundry 的**本地比較協定草案**。已實作的 checker 檢查 schema 與宣告的 metadata 是否相容；它不認證硬體、稽核量測、建立統計顯著性或產生效能排名。這套協定尚未獲得社群或標準組織採認。

## 目前可重現的功能實驗

在專案目錄執行，`--out` 必須指定尚不存在的目錄：

```sh
python3 -m pim_foundry research-demo --devices hbm,gpu --out runs/hbm-gpu-example
python3 -m pim_foundry research-demo --devices hbm,npu --out runs/hbm-npu-example
python3 -m pim_foundry research-demo --devices sram,hbm,gpu,npu --out runs/four-device-example
python3 -m pim_foundry benchmark-check runs/hbm-gpu-example/benchmarks/hbm_only.json runs/hbm-gpu-example/benchmarks/heterogeneous.json
```

`platform_demo` 用相同的微型 `matvec → relu → matvec` 模型與輸入，產生各選定裝置的單獨 placement，以及至少兩裝置時的異質 placement。它用於檢查功能和產物流程，尚非具有代表性的應用 benchmark suite。HBM＋GPU 範例將第一個 `matvec` 放在 HBM，後兩個節點放在 GPU；HBM＋NPU 亦同。四裝置範例保留四種單獨 placement，預設異質 placement 使用 HBM＋GPU，沒有宣稱找出了最佳分工。

所有 profile 共用同一個 Python 整數功能引擎，差異是示意容量、tile、unit 數與事件名稱。它們沒有執行 GPU／NPU kernel 或真實 CIM／PIM 指令，也沒有代表廠商速度。計算依序進行，輸出對照獨立 scalar oracle；int64 值採精確整數中間結果，節點最終結果須落在 int64 範圍。

demo 的互連是完全連接的示意拓樸，各 run 都保留所有選定裝置，即使某些裝置未執行算子。`gpu_only` 等名稱表示 placement，不能解讀為拆除其他裝置後的實體主機。此設定只產生冷啟動的邏輯資料流觀察：輸入／權重從 host 開始、每個消費節點重新串流載入權重、輸出返回 host。

| 產物 | 內容 |
|---|---|
| `experiment.json` | graph、輸入、platform、placement、執行事件、功能驗證與來源識別 |
| `summary.json` | 實驗摘要 |
| `benchmarks/<run_name>.json` | 每個 run 的契約、證據種類、品質、觀察值與雜湊 |
| `comparisons.json` | run 之間的 metadata 相容性、拒絕原因及相容時的觀察值 |

目前可比較的指標只有實際產生的 `logical_transfer_bytes`：沿每條連線計算的邏輯 payload 總量。它不含實體 transaction、頻寬、競爭或 overlap，不能換算成 GPU／PIM 加速比。既有 DCC trace 仍未完成數值驗證；這些 reference 結果不提升它的驗證狀態。

## Checker 已檢查與尚未查核的內容

`benchmark-check LEFT.json RIGHT.json` 的退出碼是：`0` 宣告相容、`1` 宣告不相容、`2` 輸入無效。`0` 只代表通過本地 metadata 規則。

| 檢查項目 | 已實作規則 |
|---|---|
| 工作負載與測量契約 | workload、request stream、measurement protocol 雜湊及契約欄位一致；成本名稱比較不受順序影響 |
| 品質與樣本 | 各自達到宣告門檻、達到 `min_samples`；拒絕非法數字 |
| 指標 | 名稱與固定單位一致；logical bytes 只能是一筆 total，scope 必須為 `logical_dataflow`；吞吐率不可使用 total 統計 |
| 證據 | 證據類別一致；任何 component 為 `functional` 時，不接受物理指標作比較 |
| 校準 | 分析、週期模擬及混合估算的物理指標須宣告 calibration protocol／artifact 雜湊；比較雙方 protocol 相同 |
| 資源軌 | `equal_budget` 須提供相同 budget 雜湊；`whole_system` 不宣稱資源相等 |

Hash 是產物識別碼；checker 不會讀取其指向的檔案來確認內容真實。它也不驗證品質分數、計時範圍是否真的完整、資源是否符合 budget、模型校準是否有效，或硬體是否按描述執行。`numeric_policy` 與 `scenario` 目前是相同文字宣告的檢查，並非可執行政策驗證器。`numeric_policy` 可以命名共同的品質政策，允許雙方使用不同實際精度；各自的實際格式與實作條件須在 system manifest 揭露。

相容時輸出兩邊的 `observations`，以及固定為 false 的 `performance_ranking_produced`、`independently_audited`、`statistical_significance_established`。不同契約時會列出原因而不輸出觀察值對照。現有 `mean`、`median`、`p95`、`p99` 只是描述統計；百分位採 nearest-rank，沒有信賴區間或差異檢定。

## 固定工作與品質要求

需要區分兩種研究問題，並把政策寫入版本化 protocol：

- **固定算術的 kernel 比較**：相同輸入與權重、索引、精度、累加、rounding、overflow／saturation 與容許誤差；這適合驗證不同 mapping 是否保留語意。現有 demo 屬於精確整數功能子集。
- **固定任務品質的模型比較**：相同任務、評估資料與品質門檻，政策可允許不同精度或量化方式；必須保存實際格式、轉換後權重、量化／校準資料與品質結果，並按同一政策驗收。這些內容尚未由本地 checker 自動稽核。

MLPerf Inference 為各 benchmark 定義資料與品質門檻，Closed 要求模型等價、允許量化校準，且 accuracy／performance 使用相同程式。可借鏡這種品質約束；不同精度是否可比，要由明列政策與實測品質決定。[官方 Inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)

workload 身分應連到可重建的規範模型、權重及資料；各後端的合法轉換另留產物與來源。LLM 還須固定 tokenizer、prompt／輸出長度規則、batch、concurrency、arrival trace、seed、KV cache 與 prefill／decode 條件。request stream 應保存真正送入系統的請求，不只記錄總數。

## 情境、初始狀態與成本範圍

SingleStream 的逐筆延遲、Server／Interactive 的負載下延遲要求、Offline 的吞吐回答不同問題。研究採用哪個情境，需固定輸入方式、完成條件、統計方式與最低樣本量；適用情境和門檻應鎖定特定 benchmark 版本。[MLPerf 提交說明](https://docs.mlcommons.org/inference/submission/)

本地 `end_to_end` 契約要求列出 queueing、preprocessing、layout conversion、input transfer、compute、inter-device transfer、synchronization、host fallback、output transfer、postprocessing；`cold` 再要求 model load。`kernel` 範圍至少要列出 compute。某成本為零也應保留範圍聲明，並附理由。此規則是本協定的宣告檢查，沒有量測各階段實際耗時。

`warm` 須說明已常駐的權重、cache／KV 初態、編譯與預熱條件；setup 成本另報。不得讓一邊預先裝入權重、另一邊每次重載，再合併為同一結果。MLPerf 有具體的測量邊界和可排除預處理條款，本協定不將所有啟動成本都宣稱為其要求。[官方測量規則 §§5、7](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)

實體結果須記錄主機、accelerator、記憶體、互連與軟體版本，以及同步完成／讀回的觀測點。能量要區分 array、chip、board 與完整系統；整體能量包含所宣告系統的相關元件，效能與能量取自同一次 run。MLPerf Power 採完整 SUT 的量測原則；若要正式提交，仍需查核適用輪次的完整 power workflow。[官方 Power rules](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc)

## 資源與證據的兩個獨立維度

`whole_system` 比較完整系統在相同 workload／品質／情境下的結果，公開設備數與資源；它不將差異單獨歸因於 PIM。`equal_budget` 研究則明列約束，例如同製程、總面積或功率預算、容量與互連條件，保存預算定義與實際用量；未控制的差異要做敏感度分析。相同 budget hash 只表示宣告相同，實際符合程度需要另外查核。

| 證據類別 | 可以支持的結論與必要限制 |
|---|---|
| `functional` | 所宣告數值語意與邏輯資料流；不能支持物理時間或能量 |
| `analytical` | 指定假設下的估計；需成本來源、校準範圍與誤差 |
| `cycle_simulation` | 指定 simulator 的週期結果；需設定、校準及未建模行為 |
| `hardware_measurement` | 特定硬體／軟體／環境的量測；需原始 logs、邊界、重複試驗與不確定性 |
| `hybrid_estimate` | 不同 component 採不同證據，例如 GPU 實測＋PIM 模擬；需說明組合方式與互連模型 |

混合估算必須列出各 component 的證據，且至少包含兩類。checker 保守地要求比較雙方具有相同的完整 `(component name, evidence)` 對應，因此名稱應代表穩定的系統角色。此條件仍沒有證明 component 範圍或組合模型公平。把獨立 GPU 和 PIM 耗時相加，仍可能漏掉傳輸競爭、排隊和 overlap，不能改標為實機共同執行。

Timeloop 提供分析模型與 mapping 搜尋，官方建議以週期模擬、產品資料或 synthesis 驗證並建立誤差範圍；Accelergy 以 component actions 與估計 plugins 產生能量估計。兩者可成為估計 adapter，但需要對具體 GPU／NPU／CIM 架構驗證其覆蓋，不能從工具名稱推定估計準確。[Timeloop 官方概述](https://timeloop.csail.mit.edu/)、[Accelergy 官方說明](https://accelergy.mit.edu/)

## 從草案走向可信的研究基礎

1. **建立真實 adapter**：匯入模型、配置 layout、lower、transfer、launch、wait、readback；用共用 conformance workload 驗證數值、尾塊、容量、ownership 和同步。公開哪些算子使用 host fallback。
2. **校準及獨立驗證**：將校準 workload 與驗證 workload 分離，保存原始測量及 simulator／模型版本；回報各子系統與完整執行的誤差、適用範圍和敏感度，不只保存 hash。
3. **建立 baseline 與消融**：GPU-only、NPU-only、HBM-PIM-only、手工異質配置及待評估配置；比較是否使用相同平台，並逐項檢查 mapping、layout、權重重用、transfer 與 overlap 的貢獻。
4. **完整實驗 bundle**：固定模型／資料／請求／配置／程式雜湊、seed、依賴與校準檔，保留所有 run、失敗和排除理由；分析腳本提供描述統計、合適的不確定性估計及原始 observations。
5. **公開維護與外部重現**：版本化 schema 與政策、公開變更審查和 conformance 測例，讓外部團隊重現結果並新增不同硬體 adapter。MLPerf 也要求共享 benchmark 實作和可重現結果；其新 benchmark checklist 可作規格完整性的參考。[官方規則](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)、[官方 checklist](https://github.com/mlcommons/inference_policies/blob/master/new_benchmark_checklist.md)

是否成為共同標準取決於外部驗證、實際採用和持續治理。當前交付是能拒絕一部分不一致比較的本地工具；真正的效能結論仍需要符合契約的執行與可查核證據。
