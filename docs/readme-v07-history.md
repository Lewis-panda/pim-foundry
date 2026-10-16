> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

> 本頁保留 v0.7 的歷史說明；現行能力與驗收請見 [目前 README](../README.md)。

# PIM Foundry — Application 與異質 CIM／PIM 研究工具 v0.7

同一個 application 換成 HBM-PIM + GPU 或 NPU 時，不必改寫模型和輸入。**application（任務、資產、scenario、quality）** 與 **experiment（平台、方法、配置）** 分開保存。v0.7 新增三個論文 artifact 入口，以原作者程式的直接執行、框架執行及來源快照重跑交叉核對；獨立語意檢查另外識別作者 reference 也可能漏掉的問題。

- 本輪論文驗證總表 (local artifact: `../runs/paper-validation-v07/report.html`)：PIMbench、CIMFlow、NeuPIMs 的實際結果、計量範圍與品質狀態。
- [驗證方法與候選](paper-validation-protocol.md)：區分接入一致性、任務語意、實驗公平性與模型準確性。
- IANUS 證據稽核 (local artifact: `../runs/ianus-claim-audit-v1/report.html`)：公開圖表的算術、利用率分母與資源對照；這是 paper audit，未執行作者的 IANUS artifact。[原文審查](ianus-paper-audit.md) · [程式搜尋與第三方候選](ianus-artifact-discovery.md)。
- [Application 接入與擴充介面](applications.md)：六種內建 adapter、嚴格資產綁定、共同結果契約、快照重跑。
- [PIMbench 驗證](pimbench-validation.md)：三個應用 × 三種 DRAM PIM 架構；BMP metadata、KNN 標籤洩漏與 triangle-count 資源問題的獨立證據。
- [CIMFlow 驗證](cimflow-application-validation.md)：原生 ResNet18／MobileNetV2 編譯與模擬；相同 ISA 的時序差異與精確重現失敗保留為研究結果。
- [NeuPIMs 驗證](neupims-validation.md)：512-request NPU–PIM 原生模擬，相同有序請求下的通道分配對照，以及不可直接比較的 NPU-only 原始配置。
- 應用部署實驗 (local artifact: `../runs/application-v06/report.html`)：同一 ONNX model 在 HBM+GPU、HBM+NPU 執行；另有 SRAM+HBM+GPU 的明示混合精度圖，含同平台 baseline。
- STARC 論文 application 實驗 (local artifact: `../runs/starc-application-v06/report.html`)：執行 ASPLOS 2026 原始 GPU–PIM transformer 系統模型，Full/STARC 對照、獨立直接執行比對、逐步 trace 與隔離 cache。
- [STARC 來源稽核](starc-source-notes.md)：記憶體錯誤重現與獨立修正版、原始與修正版的結果一致性、GQA 與 clustering 的模型限制。

```sh
cd pim-foundry
.venv-onnx/bin/python -m pim_foundry app-run \
  --application runs/application-v06/specifications/onnx-application.json \
  --experiment runs/application-v06/specifications/onnx-hbm-gpu-native.json \
  --out runs/my-application
.venv-onnx/bin/python -m pim_foundry app-verify runs/my-application
.venv-onnx/bin/python -m pim_foundry app-replay runs/my-application --out runs/my-application-replay
```

ONNX 路徑真正經過共同 typed graph／後端；STARC 路徑保留原始 artifact 的系統模型，沒有宣稱完整 LLM 已移植到共同 IR。本次 STARC 接入驗證使用 128-token 輸出窗口，README 原始 32,000-token 實驗另有可執行規格；完整 figure 與 LongBench 品質尚未重現。所有模擬指標保留單位、計量範圍與品質觀測，不會自動標成硬體效能。

新增的 PIMbench 執行原始 C++ 功能模擬與分析成本模型；CIMFlow 執行原始 ONNX→compiler→ISA→SystemC 流程，但發布的 compiler 省略 pooling／classifier，因此範圍明記為 CNN backbone；NeuPIMs 執行原始 512-request、單層單 token 的 NPU–PIM 時序模型。後兩者沒有產生神經網路數值輸出，任務品質維持未評估。詳細執行範圍與驗收見 [v0.7 紀錄](release-v07.md)。

## 共同原生後端

HBM-PIM 算完後，GPU／NPU 要收到中間資料才能繼續；多個請求也會競爭連線、DMA 和運算單元。PIM Foundry 把工作負載、裝置配置、搬移、排程、數值驗證和評測紀錄接成可重跑的流程。

**v0.5 已接通同一張具明確精度的計算圖：SRAM 指令／記憶體 → 精度轉換 → HBM 指令／記憶體 → GPU／NPU 模型。** 每個後端的輸出實際傳入下一節點，逐節點對照獨立數值參考；搬移、資料整理、同步、輸出與全部裝置閒置成本都留下紀錄。

SRAM 使用 CIMFlow 原生時序模擬器；HBM 使用修復並通過 ASan/UBSan 的 DCC 時序模型與明確定義的 Foundry SIMD16 指令模型。數值由讀取實際 ISA 與記憶體 bytes 的解譯器驗證。GPU／NPU 為明示分析模型；預設硬體係數尚未經實機校準。

```sh
cd pim-foundry
python3 -m pim_foundry native-demo --case mixed --out runs/my-mixed
python3 -m pim_foundry native-verify runs/my-mixed
python3 runs/my-mixed/reproduce.py runs/my-mixed-replayed
```

查看 `runs/my-mixed/report.html`。`--case hbm-gpu`、`--case hbm-npu` 可選其他組合；`native-init` 匯出可修改的圖、輸入、平台與 placement，`native-run` 執行自訂圖。[新流程與建置說明](native-quickstart.md) · [數值契約](numeric-contract.md) · [後端擴充介面](backend-plugin.md) · [v0.5 驗收紀錄](release-v05.md)

本機完整交付範例位於 共同後端實驗報告 (local artifact: `../runs/unified-v05/report.html`)，包含六組配置／baseline 與快照重跑，另有 ASan 和頻寬敏感度檢查。

**2026-10-15 論文實測與後續修復：** 原始 DCC artifact 的 use-after-free 已重現並保留證據；v0.5 另外建立修復版本，加入明示的完成時間與 barrier 語意。原始 artifact 與修復模型保持不同版本及結果，詳見 [論文接入評估](paper-validation-dcc.md)、[runtime 稽核](dcc-runtime-audit.md) 與 [修復補丁](../patches/dcc/README.md)。

## 既有分析式事件模擬流程

Python 3.10+，核心只用標準函式庫。輸出目錄必須尚不存在：

```sh
cd pim-foundry
python3 -m pim_foundry suite --devices hbm,gpu --search --out runs/my-experiment
python3 -m pim_foundry verify-bundle runs/my-experiment
python3 runs/my-experiment/reproduce.py
```

打開 `runs/my-experiment/report.html` 查看各配置的模型預測、逐資源時間軸、請求完成時間與模型假設。每份實驗附上程式來源快照；`reproduce.py` 使用該份快照重算所有結果，不依賴日後修改的程式碼。所有測試：`python3 -m unittest discover -s tests -v`。

若要安裝指令入口，可使用 `python3 -m pip install .`，之後以 `pim-foundry` 取代 `python3 -m pim_foundry`。套件無執行期第三方依賴；安裝流程需要 setuptools。DCC 的固定來源 snapshot 不包含在安裝套件中，其 trace 功能使用原始專案 checkout。

## 既有分析流程的功能與邊界

| 部分 | 已實作與邊界 |
|---|---|
| 工作負載 | GEMV、GEMM、矩陣 MLP、殘差分支；固定 seed／scale，嚴格 JSON 匯入匯出與內容雜湊。這是研究測例套件，尚非完整應用或 LLM benchmark。 |
| 數值 | `matvec`、`matmul`、`relu`、`add`；int64 儲存、精確整數中間值、節點輸出溢位檢查；獨立 scalar oracle。沒有浮點、量化或 analog 噪聲模型。 |
| 裝置與拓樸 | SRAM／HBM／GPU／NPU 的命名裝置、容量、tile、任意有向連線；每裝置可自訂計算與本地記憶體成本。標籤不等於廠商 ISA 支援。 |
| 事件模擬 | 輸入／輸出搬移、逐權重 tile 傳輸與計算、共享連線及半雙工 DMA 競爭、跨請求重疊、固定請求槽與保守容量保留。每筆請求內的節點依序執行。 |
| 初始狀態 | cold 從 host 載入權重；warm 保留完整權重且檢查容量，排除預載時間與能耗，另外報告預載 bytes。 |
| 成本 | 整數 ns 的分析預測；計算與本地記憶體時間取較大者；動態能量加全部所選裝置的基底功耗。範圍是 device graph，排除 host 基底功耗和應用前後處理。 |
| 配置搜尋 | 有上限的完整 placement 列舉、容量失敗原因、模型內的延遲／能量 Pareto 集合；不宣稱硬體最優或實測加速比。 |
| 評測 | 檢查相同工作負載、數值／品質政策、請求、冷暖狀態、測量範圍與證據來源；未附校準證據的物理指標拒絕直接比較。 |
| 模型驗證 | 匯入外部量測、核對原始 log 雜湊與實驗身分、校準／驗證 workload 分離、回報驗證誤差；可將精確匹配且通過驗證的延遲紀錄附上證據。沒有自動參數擬合或獨立硬體稽核。 |
| 產物與重現 | 完整設定、工作負載、事件、評測紀錄、Chrome Trace、離線 HTML、來源快照、檔案 manifest、完整性檢查和數值／時序重播。 |

以上表格描述保留的 int64 `suite`／`simulate` 流程。v0.5 的 `native-*` 流程另使用 INT8／INT32／FP16／FP32 typed graph、原生後端與明確轉型；目前採 cold、單請求序列排程。兩種流程各自保存數值與排程契約，不能混用結果。既有分析式矩陣乘法逐列執行，每列重新載入權重；資料流假設會反映在成本紀錄。

## 修改設定與執行自訂計算圖

```sh
python3 -m pim_foundry init --devices sram,hbm,gpu,npu --out configs/my-platform
python3 -m pim_foundry simulate --config configs/my-platform/platform.json --workload configs/my-platform/residual.json --requests 4 --max-inflight 2 --state warm --search --out runs/my-residual
```

設定檔公開每條連線的頻寬、延遲、共享 resource 與能量，以及每個裝置的吞吐、本地頻寬、啟動成本、動態能量和基底功耗。`--placement` 可讀取 node→device 的 JSON，`--arrivals` 可讀取完整請求抵達時間串流。各次單裝置 baseline 仍保留整個 platform，包括閒置裝置。

預設不同裝置的吞吐／能量成本相同，容量與 tile 是示意值。這是讓假設明確可改的起點；修改 provenance 文字不會自動變成校準模型。

- [完整使用流程](quickstart.md)
- [事件、數學公式、容量與排程限制](simulator.md)
- [外部量測格式與驗證流程](calibration.md)
- [共同比較契約與社群驗收條件](benchmark-protocol.md)
- [跨後端架構設計紀錄](unified-cim.md)

## 公平比較的實際範圍

`benchmark-check LEFT.json RIGHT.json` 回報條件相容性與拒絕原因。它不驗證硬體是否真的依聲明執行；`qualify-latency` 會重新查核提供的驗證資料、原始 log 與精確對應的預測，但仍不代表外部認證。

完整系統與相同資源預算的研究使用不同 comparison track；分析、週期模擬、實測和混合估算保持不同證據種類。模型效能比較的門檻、代表性工作負載、獨立重現與公開維護，仍需研究社群共同驗證。預設報告不提供跨工作負載總分。

`research-demo` 保留舊的微型功能範例。DCC trace 路徑繼續獨立標記為未完成數值驗證；新的功能引擎與分析模擬不替它的地址或指令語意背書。

## 原有 DCC HBM trace 路徑

PIM core 只能處理其所連接 bank 裡的資料，因此把矩陣交給 PIM 前，必須先決定每個 bank 放哪些元素、輸入如何複製、局部結果如何合併。這個專案先把既有 DCC 的 HBM-PIM GEMV 配置搜尋與 trace 產生器接成可重現的入口，再對已知有問題的設定加上檢查。

**DCC 整合支援 trace 產生與獨立的 native simulator 執行入口，尚未完成其數值執行器或硬體驅動。** `compile` 只產生且檢查命令／地址序列；`dcc-simulate` 才呼叫已建置的固定版本 simulator，另存執行報告、原始 log 與檔案雜湊，保留原編譯報告。兩者均不驗證 FP16 答案。已發現的 generator 缺陷見 [原始碼稽核](upstream-audit.md)。

## 立即使用

Python 3.10+，只使用標準函式庫；不需要安裝 NumPy、TVM 或 CUDA。此交付已包含三個固定版本的 upstream 參考檔案，可以離線執行。

```sh
cd pim-foundry
python3 -m pim_foundry doctor
python3 -m pim_foundry enumerate --batch 1 --heads 16 --input 64 --output 256
python3 -m pim_foundry compile --batch 1 --heads 16 --input 64 --output 256 --out runs/my-first-run
python3 -m pim_foundry inspect runs/my-first-run/kernel.trace
python3 -m unittest discover -s tests -v
```

輸出目錄必須尚不存在，避免覆寫舊實驗。`--candidate ID` 可以指定 `enumerate` 列出的配置；預設取第一個通過保守檢查的配置，**沒有宣稱這是最快配置**。

每次編譯留下：

| 檔案 | 內容 |
|---|---|
| `kernel.trace` | 原版 DCC generator 產生的命令與地址 |
| `kernel.yaml` | 原版產生的 simulator 設定；adapter 對 trace 路徑加引號，並移除會共用輸出路徑的 optional recorder |
| `candidates.json` | 配置、資源檢查、排除原因與固定來源版本 |
| `report.json` | 命令統計、來源與 trace SHA-256，以及逐項驗證狀態 |

`kernel.yaml` 對應固定 DCC 修改過的 Ramulator2，不能假設任意 Ramulator2 版本皆相容。下列命令使用本次建置的 CPU simulator；adapter 會在新目錄複製 trace 並改用相對路徑。

```sh
python3 -m pim_foundry dcc-simulate runs/my-first-run --executable runs/paper-dcc-build/build/ramulator2 --library external/dcc-isca26/src/Ramulator2/HBMPIM/libramulator.so --out runs/my-native-run
```

此入口只接受已稽核的固定 HBM YAML 設定，每次最多 60 秒，會檢查載入行數及 17 類 request counters。`completed` 表示程序成功且 counters 相符；報告的 `performance_comparison_eligible` 與 `numerical_correctness_verified` 仍為 false。建置步驟及全部依賴版本保存在 `runs/paper-dcc-build/build-metadata.json`。

該 YAML 保留 upstream `NoTranslation.max_addr=2 GiB`，而 generator 地址範圍可達 16 GiB；本次確認 pinned frontend 不呼叫 Translation，NoTranslation 也不修改地址，故此欄位沒有截斷 trace。早期編譯產物保留當時的「尚未確認」文字；以後續 runtime 稽核為準。

## DCC 路徑已實作的邊界

- 單一 kernel：多 batch／head 的 GEMV 形狀入口。邏輯運算為 `vector[B,H,K] × matrix[B,H,K,N] → result[B,H,N]`；這是介面契約，未證明 upstream 地址序列實現了該契約。
- 單一後端：DCC HBM GEMV trace generator，固定 16 groups、每 group 64 cores；FP16 只表示其地址計算採每元素 2 bytes，沒有浮點結果驗證。
- 先檢查來源雜湊，再用獨立程序執行，避免 upstream 的可變 global state 與 tiling 修改污染其他工作。
- 限制無 padding、group/core 因子為二次方、資源乘積不超額、每個 local partition 只有一個 head、local K 為 16 的倍數且介於 16–128、local N 為 16 的倍數、單次 bank 容量足夠。
- 限制單次 trace 最多 250,000 commands、worker 最多 60 秒。這些是本工具的資源上限。
- 逐行檢查 trace 語法、opcode 和地址範圍；未對齊地址另外統計，不擅自判定硬體一定不接受。

保守檢查可能拒絕硬體原本能處理的 shape。通過檢查也**不等於數值正確、完整硬體合法或具有加速效果**。目前仍有 operand base/layout 語意未釐清，因此所有產物都標記 `trace_generated_not_numerically_verified`。

## 為什麼先接 DCC

截至 2026-10-14 查核，領域已有共同編譯抽象與公開實作，不能視為空白：

| 專案 | 已公開方向 | 對此工作的意義 |
|---|---|---|
| [CINM／Cinnamon](https://github.com/tud-ccc/Cinnamon) | MLIR 上的 CIM／CNM 分層抽象與 UPMEM 路徑 | 未來跨架構 IR 的參考；本次未建置 |
| [DCC](https://github.com/SPIN-Research-Group/DCC) | HBM-PIM／AttAcc 的資料分配與運算排程聯合最佳化 | 與使用者指定的 DRAM／HBM 方向最接近；本次只接入 HBM GEMV 子集 |
| [PIMCOMP](https://github.com/sunxt99/PIMCOMP-NN) | crossbar PIM DNN 的分割、映射與排程 | 架構族不同，本次未接入 |
| [CIMFlow](https://github.com/BUAA-CI-LAB/CIMFlow-Compiler) | CIM 的 ONNX→MLIR→ISA 流程 | v0.7 已接入原始 CNN backbone 編譯／模擬路徑，保留 compiler 省略算子的限制 |

「是否已有像 TVM 一樣成熟的共同框架」需要看多後端可用性、數值測試、runtime、文件與維護，而不能僅憑專案名稱或論文結果判定。這次調查和實作沒有證明全領域不存在這樣的框架。

## 驗證與後續

測試涵蓋共同圖與功能模型的數值、輸入／資源約束、跨裝置路徑與容量，以及 DCC adapter、trace parser、來源雜湊、重複產出一致性與 upstream 已知問題的重現。`test_upstream_audit.py` 通過代表成功重現該來源版本的問題，不能當成 upstream 計算正確的證據。

邁向可共用框架的具體分層與驗收門檻見 [設計與開發路線](design.md)。第一個必要工作是確認每條指令的資料語意，建立數值執行器，修正／回歸驗證地址配置；之後才有依據做效能搜尋與第二個後端。

## 來源與授權狀態

Upstream 固定在 DCC commit `7ffb80e4f33e357177dc8ab148953c0484e1e322`，檔案 URL 與 SHA-256 記錄於 `upstream/manifest.json`。保留其檔案原樣，本專案只增加 adapter，未將 upstream 成果宣稱為新實作。

這份 snapshot 用於本地研究。[DCC 論文 artifact appendix](https://arxiv.org/html/2511.15503v5) 標示 MIT，但此次檢視的 repository 根目錄未提供可核對的 LICENSE 檔；此處沒有替 upstream 新增授權宣告。若要公開發佈包含 upstream 的套件，須先核實相應授權。
