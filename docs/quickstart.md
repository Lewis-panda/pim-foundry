# v0.4：執行、檢查與重現實驗

先把同一份 workload 放到 hbm、gpu 或兩者分工的配置上，工具就會檢查功能輸出、排程資料搬移與計算，並把結果存成可離線開啟的報告。預設成本皆為合成參數；結果用來檢查設定與模型內的差異，硬體效能與校準狀態維持未驗證。

以下命令從專案根目錄執行，需要 Python 3.10 以上。核心使用 Python 標準函式庫，不需 GPU、PIM 板卡或第三方 Python 套件。

```sh
cd pim-foundry
python3 -m pim_foundry doctor
```

## 先產生完整實驗包

```sh
python3 -m pim_foundry suite --devices hbm,gpu --out runs/new --search
```

這個命令包含 `gemv`、`gemm`、`mlp`、`residual` 四個 workload，預設四個請求在 0 ns 到達、`max_inflight=2`、cold 模式。每個 workload 會執行各單裝置放置；有適用的多節點 graph 時，也加入分工放置。`--search` 枚舉放置，依模型中的 makespan、再依能量排序，並保留容量或路由不合法的拒絕原因。它沒有取得真實硬體的最優放置證據。

直接以瀏覽器開啟 `runs/new/report.html`。報告內嵌資料、樣式與互動程式，可離線切換 workload／run、縮放時間軸，並將游標停在事件上查看資料；不需要啟動網頁服務。視覺時間軸最多顯示前 4,000 個事件，完整內容保留在 results／traces。每次 `--out` 必須指定不存在的路徑，既有實驗不會被覆寫。

## 建立自己的平台與 workload 設定

```sh
python3 -m pim_foundry init --devices sram,hbm,gpu,npu --out configs/new
```

產生的 `platform.json` 同時包含平台拓樸、邏輯容量與成本模型；另外輸出四個 workload JSON。各 profile 的容量及 tile 大小是示範設定，裝置整體成本速率則採相同合成預設值。設定欄位與假設詳見 [simulator.md](simulator.md)。

只跑一個已匯出的 workload：

```sh
python3 -m pim_foundry simulate \
  --config configs/new/platform.json \
  --workload configs/new/mlp.json \
  --out runs/mlp-new
```

省略 `--placement` 時仍會產生該平台的 baseline 放置。若要只測指定分工，把下列 JSON 存成 `configs/new/mlp-placement.json`：

```json
{"linear1": "hbm", "activation": "gpu", "linear2": "gpu"}
```

```sh
python3 -m pim_foundry simulate \
  --config configs/new/platform.json \
  --workload configs/new/mlp.json \
  --placement configs/new/mlp-placement.json \
  --requests 4 --max-inflight 2 \
  --out runs/mlp-placement-new
```

placement 必須恰好包含 graph 的所有節點，值必須是設定檔中的裝置名稱。這裡仍保留平台中的四個裝置，包含未執行節點裝置的基線功率。若研究問題只包含 hbm 與 gpu，應另以 `--devices hbm,gpu` 建立對應平台。

## 四個內建 workload 與匯出

| 名稱 | scale=1 的計算 | 節點名稱 |
| --- | --- | --- |
| `gemv` | `x[64] × W[64,48]` | `linear` |
| `gemm` | `A[8,32] × W[32,48]` | `linear` |
| `mlp` | `8×32 → 8×48 → ReLU → 8×24` | `linear1`, `activation`, `linear2` |
| `residual` | 相同 `8×32` 輸入經兩個 `32×32` 線性分支，相加後 ReLU | `left_linear`, `right_linear`, `merge`, `activation` |

它們是版本化的合成 graph，沒有訓練權重或應用品質結果。`--scale` 接受 1 到 4，會同比放大各維度；`--seed` 固定輸入與權重產生程序。儲存與驗證採 signed int64、精確整數中間值，不包含量化精度比較。

單獨匯出 workload 的命令是 `workload`：

```sh
python3 -m pim_foundry workload gemm \
  --scale 2 --seed 7 --out configs/new/gemm-scale2.json
```

輸出 JSON 包含 graph、shape、常數與實際 runtime input 值，之後可交給 `simulate --workload`。此命令要求輸出檔案尚不存在，且父目錄已存在。要縮小 suite，可加上 `--workloads gemv,mlp`；`--candidate-limit` 預設 256，限制搜尋中的放置組合數，超出時會明確拒絕。

## 到達時間與 cold／warm

```sh
python3 -m pim_foundry suite \
  --config configs/new/platform.json --workloads gemv,gemm \
  --requests 4 --arrival-interval-ns 1000 --max-inflight 2 \
  --state warm --out runs/warm-new
```

到達時間在這裡為 `[0,1000,2000,3000]` ns。也可將已排序的非負整數陣列存成 JSON，再傳入 `--arrivals 檔案路徑`；它會取代 count／interval 生成的陣列。`max_inflight` 不得大於請求數。單次請求可用 `--requests 1 --max-inflight 1`。

cold 會計入每次請求的權重串流；warm 假設權重已完整常駐各目標裝置，因此需要額外容量，且預載時間與能量不在觀察區間內。warm 報告會列出排除的一次性預載 link bytes。比較兩個結果時應保持 initial state、到達序列、inflight 數及量測範圍一致。

## 實驗包裡有哪些證據

| 檔案或目錄 | 用途 |
| --- | --- |
| `report.html` | 離線互動報告。 |
| `config.json`, `workloads/` | 實際使用的設定、graph、常數與輸入。 |
| `results/` | 功能輸出、驗證結果、完整模擬事件、成本與容量。 |
| `traces/` | Chrome Trace JSON；可載入相容的 trace viewer 檢查事件。 |
| `search/` | 有啟用搜尋時的候選、拒絕原因與模型內 Pareto frontier。 |
| `benchmarks/`, `comparisons/` | 各 metric 的比較契約與相容性檢查。 |
| `summary.json` | workload、run、placement 與彙總指標。 |
| `engine/pim_foundry/` | 產生結果時的 Python 原始碼快照。 |
| `reproduce.py` | 自動使用保存的 engine 重跑，並禁止新增 bytecode 檔案。 |
| `manifest.json` | 實驗包檔案的 SHA-256。 |

例如 `traces/case_000/run_000.json` 包含各事件的 request、資源、依賴、時間及 bytes／MACs 等資訊。Chrome Trace 的 `ts`、`dur` 採格式要求的微秒數值，由內部 ns 除以 1,000 匯出；`results/` 保留整數 ns。link bytes 是每一 hop 的邏輯 payload 總和，不能直接解讀為 DRAM transaction 或實測匯流排流量。

## 驗證檔案與重跑

```sh
python3 -m pim_foundry verify-bundle runs/new
python3 -m pim_foundry replay runs/new
```

`verify-bundle` 檢查清單中的檔案 hash、缺失／額外檔案及路徑。它證明保存的檔案與 manifest 一致，並不驗證測量來源或建立第三方認證。`replay` 重新執行保存的 workloads 與 placements，要求結果與匯出內容完全相同，並回報目前 engine 原始碼 hash 是否與當時一致。

專案之後修改時，可用實驗包內的啟動程式，自動選擇當時的原始碼快照重跑：

```sh
python3 runs/new/reproduce.py
```

它會優先載入保存的 engine，並禁止在實驗包內新增 bytecode，不受目前工作目錄內的新版本影響。若需要明確管理 Python 載入路徑，也可使用下列等價方式；先移到專案外，避免目前工作目錄優先於 `PYTHONPATH`：

```sh
cd /tmp
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=runs/new/engine \
python3 -m pim_foundry replay runs/new
```

`PYTHONDONTWRITEBYTECODE=1` 避免在保存的 engine 內新增 `__pycache__`，否則實驗包會出現未列入 manifest 的檔案。這個快照保存本專案 Python 原始碼；Python runtime 仍來自執行環境，沒有包含作業系統或實體裝置。

產生的 latency、throughput 與能量屬 `device_graph` 範圍：包含模型內搬移、等待與裝置計算，裝置基線功率涵蓋整段觀察區間；host 軟體與 host 基線功率、前後處理、編譯、warm 預載成本位於範圍之外。外部測量與模型驗證使用獨立的 `validate-model` 工作流程；合成參數與通過功能測試本身不會讓報告成為硬體效能證據。
