# 解析式模擬器：資料搬移、資源競爭與成本邊界

一次矩陣運算先把輸入送到指定裝置，再逐塊搬入權重並計算，最後把輸出送回 host；模擬器把這些動作排進時間軸，遇到已被其他請求占用的連線、DMA 或計算資源就延後開始。因此，改變切塊、放置位置、到達時間或成本參數，會透過明確的搬移量與等待時間改變結果。

v0.4 已實作這套確定性排程，以及邏輯容量檢查、功能參考執行、成本設定、放置搜尋和實驗匯出。時間與能量來自設定檔中的解析式成本；預設參數是未校準的合成數值。`evidence` 固定為 `analytical`，`calibrated` 固定為 `false`。功能參考執行比對的是 signed int64 儲存、精確整數中間值的數學契約；此結果不驗證廠商 PIM、GPU 或 NPU 的數值行為。

## 一次請求實際做什麼

`simulate(graph, platform, placement, model, arrivals_ns=(0,), max_inflight=1, initial_state="cold")` 先使用 `research.plan` 檢查放置、傳輸路徑、活躍資料與容量，再建立事件。

同一請求依 graph 中的節點順序執行。每個節點先取得完整輸入，再執行計算；所有節點結束後才進行 graph outputs 的 host readback。分支節點也遵循這個序列順序，目前沒有單一請求內的分支並行或傳輸／計算重疊。

| 運算 | cold 模式的工作分解 |
| --- | --- |
| `matvec`: `x[K] × W[K,N]` | 依 N tile、K tile 順序，載入一塊權重、計算該塊，再處理下一塊。 |
| `matmul`: `A[M,K] × W[K,N]` | 逐列處理 A；每一列都重走 N／K tiles，因此權重共搬入 M 輪。 |
| `relu` | 對輸入的所有元素建立一個計算事件，支援向量與矩陣形狀。 |
| `add` | 取得兩個同形狀輸入，再建立一個逐元素相加事件；`add(x,x)` 的跨裝置輸入搬移會去重。 |

權重 tile 的邊界使用實際剩餘尺寸，沒有把尾端補齊後的資料算進傳輸量。輸入、輸出 activation 仍需完整存放；cold 模式沒有要求整份權重同時位於目標裝置。所有 payload bytes 都採每元素 8 bytes 的邏輯 int64 儲存契約，不代表實體 DRAM transaction bytes。

## 成本設定檔

`platform.json` 的最外層包含 `schema_version`、`name`、`platform`、`model`。`platform.devices` 宣告裝置名稱、種類、容量、`tile_k`、`tile_n`、`units`；`platform.links` 宣告有向連線。`model` 必須精確包含下列四個欄位：

```json
{
  "schema_version": 1,
  "provenance": {
    "kind": "synthetic",
    "description": "Uncalibrated illustrative constants"
  },
  "devices": {
    "hbm": {
      "macs_per_ns": 16,
      "elements_per_ns": 8,
      "local_bytes_per_ns": 32,
      "launch_ns": 20,
      "mac_energy_pj": 1,
      "element_energy_pj": 0.5,
      "local_energy_pj_per_byte": 0.25,
      "idle_power_w": 0.1
    }
  },
  "links": [
    {
      "source": "host",
      "destination": "hbm",
      "bytes_per_ns": 8,
      "latency_ns": 10,
      "energy_pj_per_byte": 2,
      "resource": "hbm<->host"
    },
    {
      "source": "hbm",
      "destination": "host",
      "bytes_per_ns": 8,
      "latency_ns": 10,
      "energy_pj_per_byte": 2,
      "resource": "hbm<->host"
    }
  ]
}
```

這段是單一 hbm 裝置的 `model` 範例；完整設定可由 `init` 產生。裝置名稱與有向連線必須完整對應 `platform`，不可缺少、重複或加入未宣告端點。未知欄位會被拒絕。所有速率須為有限正數；latency、launch、能量與功率可為零，但不得為負、布林值、NaN 或無限大。`provenance.kind` 接受 `synthetic`、`supplied`、`measured`；填入 `measured` 本身不會取得校準資格。

預設 SRAM、HBM、GPU、NPU 的上述成本數值完全相同，都是裝置整體的合成速率。`units` 不會再乘進 throughput。各預設 profile 的容量與 tile 大小不同，因此仍可能產生不同排程；這些差異只反映設定中的資料處理方式，不能用來推論真實裝置的效能排序。

## 每個事件的時間與本地流量

每次跨一條 link 搬移 B bytes，會計入一次 link latency：

```text
transfer_ns = ceil(latency_ns) + ceil(B / bytes_per_ns)
```

跨多個 hop 時，每一 hop 先完成整個 chunk，下一 hop 才能開始。路由採確定性的最少 hop 選擇，沒有依 latency 或頻寬尋找最佳路徑。

線性運算的一個 tile 使用 `Kt` 個輸入元素與 `Nt` 個輸出元素，工作量與本地流量定義如下：

```text
MACs        = Kt * Nt
local_bytes = 8 * (Kt * Nt + Kt + 2 * Nt)
```

三項本地流量分別是權重讀取、輸入讀取、partial output 的讀取與寫回。即使第一個 K tile 尚未有累加結果，也保守計入 partial output 的讀寫。對 `relu` 與 `add`，則使用 `8 * 元素數 * (輸入數 + 1)`；`add(x,x)` 仍算兩次本地讀取。

```text
arithmetic_ns = ceil(MACs / macs_per_ns)
             或 ceil(elements / elements_per_ns)
local_ns      = ceil(local_bytes / local_bytes_per_ns)
compute_ns    = max(arithmetic_ns, local_ns) + launch_charge
```

`launch_charge` 只在每個節點的第一個計算事件加上 `ceil(launch_ns)`，後續 tiles 為零。取最大值代表假設算術與本地資料存取可重疊，較慢的一方限制該 tile；這是簡化的 roofline 成本式。目前 DMA 與計算事件使用不同資源，沒有讓 DMA 流量競爭 `local_bytes_per_ns`。因此，它尚未描述 DMA 與運算共同壓滿同一實體記憶體介面的情況。

## 多請求如何競爭資源

每個裝置有一個 `compute:<device>` 資源。每次傳輸同時占用宣告的 `link:<resource>`、來源 `dma:<source>`、目的地 `dma:<destination>`。host 也有一個 DMA 資源。兩條不同方向或不同端點的連線只要使用相同 `resource`，就必須串行；同一端點的收送也以半雙工方式保守串行。

不同請求可在互不衝突的資源上重疊，例如請求 0 在 hbm 計算時，請求 1 可在 gpu 計算。排程器每次挑出最早可以開始的就緒事件，平手時依請求編號決定，因而相同輸入會得到相同時間軸。

`max_inflight` 使用固定請求槽位：請求 i 必須等待請求 `i - max_inflight` 完成後才能開始，也不得早於自己的 arrival。這項規則可能比任意可用槽位的動態派送更保守。到達時間必須是已排序的非負整數 ns；最多 1,024 個請求、200,000 個事件，時間上限為 signed 64-bit ns。

## cold、warm 與容量

設裝置 d 的單次序列記憶體計畫峰值為 `peak[d]`：

```text
cold reserved[d] = max_inflight * peak[d]
warm reserved[d] = max_inflight * peak[d] + persistent_weight_bytes[d]
```

每個請求槽位都保留完整峰值，而不依事件時間動態共享空閒記憶體。中繼裝置的傳輸 staging 也由原始記憶體計畫檢查。任一裝置超過容量，就拒絕該放置。

cold 模式中，每個請求的權重從 host 開始，每個線性節點重新串流；matmul 每列重載。warm 模式假設每個唯一 `(裝置, 權重 tensor)` 已預載完整權重，計算仍依 tiles 進行，但量測區間內跳過權重傳輸。同一權重在同一裝置供多個節點或請求使用時，只增加一次 persistent 容量。

warm 的 `excluded_setup_logical_bytes` 記錄每個唯一裝置／權重對沿路由預載一次的 link bytes；預載的時間與能量沒有排程或估計。warm 容量另外保留 cold 計畫中的 weight-tile scratch，屬保守重複預留。報告內嵌的 `plan` 仍是 cold 邏輯計畫，實際模式的流量應讀取 simulation 最外層的 `logical_transfer_bytes` 與 `logical_bytes_by_link`。

## 能量與量測範圍

計算動態能量為算術工作量乘上對應每 MAC／每元素成本，再加本地 bytes 的成本；傳輸動態能量為每一 hop 的 payload bytes 乘上 link 成本。另對平台內每個裝置計入：

```text
baseline_energy_pj[d] = idle_power_w[d] * makespan_ns * 1000
```

這個欄位雖名為 `idle_power_w`，實作把它視為持續存在的基線功率：計算中、等待中、請求間隙以及完全未使用的裝置都計費。`only gpu` 之類的 baseline 只改變放置，仍保留同一平台的 hbm 等裝置及其基線功率。

host DMA 的傳輸時間會影響排程，但 host CPU／軟體開銷與 host 基線功率未計入；前後處理、編譯時間也未計入。報告的量測範圍因此為 `device_graph`，不能當成完整受測系統的端到端時間或總能量。不同大小的平台仍應分開說明其資源與成本邊界。

`makespan_ns` 從第一個 arrival 算到最後一個完成；每個請求的 latency 包含其到達後的等待。throughput 是該觀察區間內完成的請求數除以區間長度，包含填管、排空與中間空檔，沒有外推穩態吞吐量。

## 可以手算的合成範例

測試中的一個 GEMV 使用 K=2、N=2，切成四個 1×1 權重 tiles。此處是局部的事件算術檢查，不是預設 suite 的 workload，也不對應任何實體硬體。設定如下：傳輸 8 B/ns、每次 latency 1 ns、計算 1 MAC/ns、本地 8 B/ns、節點 launch 3 ns。

| 動作 | 次數與成本 | 時間 |
| --- | --- | --- |
| 輸入 16 B | `1 + 16/8` | 3 ns |
| 權重 tiles | 四次 `1 + 8/8` | 8 ns |
| 計算 tiles | 每塊本地 32 B，`max(1,32/8)=4`；launch 只算一次 | `4*4+3=19` ns |
| 輸出 16 B | `1 + 16/8` | 3 ns |
| 合計 | 序列執行 | **33 ns** |

若 MAC 為 2 pJ、local byte 為 0.5 pJ、link byte 為 1 pJ，動態能量為 `4*2 + 128*0.5 + 64*1 = 136 pJ`。裝置基線功率 0.001 W 在 33 ns 內增加 33 pJ，合計 169 pJ。warm 模式移除 8 ns 權重搬移，區間時間為 25 ns，但須另保留 32 B 權重；預載成本仍位於區間之外。

這些數值有對應的手算單元測試。測試通過驗證的是模型公式、事件次序與程式結果的一致性；要用於真實裝置預測，仍需外部測量、適用範圍說明與獨立驗證資料。DRAM 指令時序、refresh、cache、bank 衝突、熱與功率狀態、廠商 runtime、硬體執行均未整合。
