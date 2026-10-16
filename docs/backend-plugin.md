# 後端註冊契約

一個後端必須接收節點的實際輸入值，產生後續節點會消費的輸出值，並回報完成
這個節點所需的時間、能量和額外 workspace。共用 runtime 負責跨裝置搬移、
容量追蹤、逐節點 scalar 核對和保存證據，讓新增模擬器時不用重新寫整條實驗流程。

## 註冊與生命週期

註冊點是 `pim_foundry.unified.register_backend(name, factory)`：

```python
from pim_foundry import unified

unified.register_backend("lab-sram-v1", MySRAMBackend)
result = unified.run(graph, inputs, platform, placement, output_directory)
```

`MySRAMBackend` 是呼叫端自行提供的 class 或 factory。名稱須非空、不可覆蓋既有
註冊；factory 接受一個 `options` dictionary，回傳具備下列介面的物件：

```python
class MySRAMBackend:
    def __init__(self, options):
        ...  # 驗證此後端的設定與必要的 runtime 路徑

    def validate(self, node, tensors):
        ...  # 不支援時拋出 ValueError

    def run(self, node, tensors, values, directory, available_bytes):
        ...  # 編譯／執行／保存證據，回傳下表的 result
```

這是介面示意，省略號須由實際後端實作。Python 呼叫端先顯式匯入／註冊，再載入
平台設定；JSON 不會自動匯入 Python module 或執行 shell。找不到後端、native
binary 缺失、指令驗證失敗或容量不足，都會使執行失敗，沒有隱式 CPU fallback。

`platform.json` 的 device 選擇已註冊的名稱：

```json
{
  "name": "sram0",
  "kind": "sram",
  "backend": "lab-sram-v1",
  "capacity_bytes": 16777216,
  "idle_power_w": 0.01,
  "options": {}
}
```

這是 `devices` 陣列的一個項目。完整 platform 使用 `schema_version: 1`，另有
`links` 陣列；每條有向 link 明確指定 `source`、`destination`、`bytes_per_ns`、
`latency_ns` 和 `energy_pj_per_byte`。`host` 是預留端點，不可作為 device 名稱。

## validate：先確認數值與硬體能力

`node` 是 `typed.Node`，`tensors` 是名稱到 `typed.Tensor` 的映射。
在建立任何節點執行 artifact 前，共用 runtime 會呼叫所有指定後端的 `validate`。
至少要檢查 operation、shape、儲存 dtype、累加精度、overflow／rounding 政策、
是否接受 requantization，以及自身的 instruction／容量限制。

可用 `typed.BackendCapabilities(...).check(node, tensors)` 處理一般 dtype 與
operation 限制，再增加自己的幾何限制。它不會自動替節點插入 cast，也不會
把 unsupported operation 分派給別的裝置。

| 內建後端 | 接受的主要操作與數值政策 |
|---|---|
| `cimflow-sram` | INT8 matvec／matmul → INT32；error-on-overflow；M≤16、K/N≤128 |
| `hbm-simd16` | FP16 matvec／matmul → FP16；K 遞增的非融合 FP32 multiply/add；最多 2,000,000 MACs |
| `analytical` | 通過 typed 檢查的操作；scalar functional execution，另加明示成本模型 |

INT32 SRAM 輸出若要交給 FP16 HBM 節點，必須用圖中的 `cast` 轉換。輸入或輸出
成為 INT8 時需遵守明示的 requantization。完整精度規則見
[數值契約](numeric-contract.md)。

## run：傳入值、輸出值與成本

`values` 是 tensor 名稱到 row-major 扁平數值序列的映射；只讀取節點指定的
inputs。這些值已包含上游後端的結果，也已經過 fabric 的儲存精度序列化／還原。
`directory` 是尚未建立的節點 artifact 目錄，由後端建立並保存指令、memory image、
native logs 與 provenance。`available_bytes` 已扣除該裝置 resident tensors，
包括本節點的邏輯輸出；額外 packed buffers／native scratch 必須計入 workspace。

回傳 dictionary 的基本契約：

| 欄位 | 意義 |
|---|---|
| `values` | 輸出 tensor 的扁平序列，元素數量與儲存 dtype 必須完全相符 |
| `duration_ns` | 節點執行與必要 packing 的模型時間；有限非負數，不可拿 simulator 主機耗時代替 |
| `energy_pj` | 此節點納入的能量；有限非負數，未知則為 `None` |
| `workspace_bytes` | 邏輯 resident tensors 之外，同時需要的額外配置 bytes，非負整數 |
| `evidence` | 清楚標示算術、timing、analytical／原生執行分別來自哪裡 |
| `instruction_lowering_verified` | 是否真的核對 emitted instructions 與 memory bytes 的算術結果 |
| `native_arithmetic` | 原生外部模擬器是否自行計算 tensor 答案 |
| `profile_provenance` | 成本／硬體參數的來源及校準狀態 |

可額外回傳 `native_timing_ns`、`packing_bytes`、`packing_ns`、`native_report` 等
可稽核欄位。當總能量未知但 packing 能量已知，可用
`known_packing_energy_pj` 保留已知部分；共用 runtime 不會因此把總能量標成完整。

後端回傳後，共用 runtime 會檢查輸出長度、精度、數值、時間、能量與 workspace。
每個節點都必須與獨立 typed scalar execution **逐元素一致**，之後才把結果交給
下一個節點。此契約適用目前的確定性數值模型；要接入帶雜訊的類比 CIM，需另行
設計明示的誤差／品質契約，不能把近似值混入現有的 exact gate。

外部程式執行應設定 timeout、保存 return code／stdout／stderr、核對來源與 binary
hash，並把確切 instruction stream 和初始化 bytes 綁定到執行報告。發生錯誤時
拋出例外；共用 runtime 會保存 `failure.json` 與已完成的節點／搬移紀錄。

## 已有原生後端如何實作這份契約

SRAM 後端把 INT8 權重 padding／packing 成 binary image，產生 CIMFlow ISA，
由獨立指令執行器處理 `MEM_CPY`、`CIM_MVM`、`CIM_OUT` 與 INT32 SIMD reduction，
同時執行原生 SystemC timing。HBM 後端產生 Foundry SIMD16 指令與 byte images，
指令執行器做 FP16／FP32 算術，再把 requests 送進有保留 patch provenance 的 DCC。
兩者的 `native_arithmetic` 都是 false；報告並未把指令執行器結果誤稱原生算術。

HBM native projection 可能發出 all-bank 操作，且每個操作後都加 completion
barrier。範例對 PIM completion 採 8 cycles、769 ps/cycle；這是 descriptor
中的明示假設，不能把邏輯單 bank 的指令語意當作硬體單 bank timing。HBM 本身
沒有能量結果；範例平台另提供 synthetic `native_energy_pj_per_cycle`，並將其
來源保留在 report。缺少這項係數時，能量總數保持未知。

原生後端負責本節點 packing／native compute 成本；共用 runtime 負責 fabric、
最終輸出傳回 host 與所有裝置 idle power，避免插件重複加上同一項成本。這份
native graph protocol 是 cold、serial；要引入 overlap 或新的 residency 政策，
需要變更明示 protocol，不能沿用原 protocol 名稱卻改變 scope。

## 重現與比較自訂插件

內建後端的 bundle 保存 Python engine、JSON、指令和 logs；外部 binary 仍依賴
記錄的路徑與身份。`native-verify` 驗證 bundle，`reproduce.py NEWOUT` 重跑內建
後端。`register_backend` 是目前 Python process 的註冊，不會自動寫入 bundle。
自訂插件應保存其來源／版本／相依 runtime，並在自己的 replay entrypoint 顯式
匯入及註冊，再呼叫 `unified.replay`。不要只保存 plugin 名稱便宣稱能獨立重現。

`native-compare --metric latency|energy` 只接受相同 workload、platform／runtime
identity、數值政策及 protocol，並要求數值 gate 通過。自訂 timing 參數應放在
platform `options`，使配置 hash 隨之改變；外部 runtime 的身份也必須保存在
artifact 與有效的比較設定中。對比的結論是指定模型下的 conditional prediction，
不會自動變成硬體性能認證或原論文 benchmark 的重現。
