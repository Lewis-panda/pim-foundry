> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

# Application 接入設計審查

這份文件記錄對 v0.5 的獨立設計審查與實作建議。v0.6 已依此加入 application
層、ONNX frontend 與 STARC adapter；可執行介面以 [application 文件](applications.md)
為準。以下保留當時發現的限制與設計理由。

LLM decode 的第 2 步會讀取第 1 步留下的 KV cache，還會把新 K/V 寫回去；
如果入口只有一張固定形狀、每次從 host 重新載入的矩陣圖，就無法表達這個狀態
延續，也無法正確界定 cache 建置、重排和每 token 延遲。Application 入口必須
先描述「做什麼任務、用什麼輸入、狀態如何延續、怎樣算答對」，再由實驗決定
在哪些裝置上完成它。

## 目前程式確實能做到什麼

依據 `pim_foundry/typed.py`、`unified.py`、`unified_examples.py` 與
`docs/native-quickstart.md`：

| 現有邊界 | 對 application 的具體影響 |
|---|---|
| Tensor 只接受 rank 1/2，具體正整數 shape | 無法直接接受常見 `[batch, heads, sequence, head_dim]` attention tensors；沒有動態長度 |
| 五種操作：matvec、matmul、relu、add、cast | MLP 子圖可接；完整 Transformer 或 ConvNet 尚不能 lowering |
| 一個節點、一個 output，SSA、無 loop/state | 無法表達會跨 decode steps 更新的 KV cache，也沒有 `prefill → decode` 的狀態契約 |
| `unified.run` 要求使用者指定每個 node 的 device | Application 作者仍需知道平台名稱；現有 examples 把 `sram_linear` 等名稱和 placement 寫在一起 |
| native graph 是 cold、serial、single request | 不能把它的總時間稱作 serving throughput、TTFT 或穩態 inter-token latency |
| 所有中間結果要求與既定 scalar 算術逐元素一致 | 能檢查此數值契約的 lowering；無法直接接近似 attention、類比雜訊或不同 fused arithmetic |
| weights 和 inputs 直接置於 JSON/記憶體，reference 有工作量上限 | 適合小型可驗證 graph；不等於可載入大型模型權重的 inference runtime |
| 已有外部 backend registry | 新 application frontend 不應再被塞進 backend registry；兩者分別處理任務和裝置 |

這些是接入能力的邊界，與是否已有社群採用無關。

## 現在應實作的四個獨立物件

1. **Application manifest**：穩定任務名稱、adapter/schema 版本、內容雜湊、
   模型／輸入／前處理資產、任務參數。不得包含 device 名稱或 node placement。
2. **Scenario**：這次處理多少請求／token、哪個 phase、batch 和長度、初始狀態、
   reset/reuse 規則、亂數種子、計量邊界。Scenario 由 runner 檢查 adapter 是否支援。
3. **Quality policy**：oracle/dataset、比較指標、容忍值或最低品質、評估樣本身分。
   Policy 是驗收要求；執行結果另存 observation，不能因 manifest 寫了要求便標成通過。
4. **Experiment binding**：application + scenario + quality + implementation variant +
   platform + mapping policy + requested claims。只有這一層能決定 HBM/SRAM/GPU/NPU。

Application 資產應使用 `{path, sha256, format, role}`。讀檔前檢查 hash；內容身分
不受絕對路徑影響。JSON 不得指定任意 Python import 或 shell command；第三方
adapter 透過 Python 顯式註冊。未知欄位、未知版本和缺失的資產直接拒絕。

建議使用 tagged schemas，避免讓每種 application 填一堆無效欄位：

```json
{
  "schema": "pim-foundry.application.v1",
  "id": "two-layer-mlp",
  "adapter": "typed-graph/v1",
  "assets": {
    "graph": {"path": "graph.json", "sha256": "<64 hex>", "format": "pim-foundry.typed-graph.v1", "role": "model"},
    "inputs": {"path": "inputs.json", "sha256": "<64 hex>", "format": "json", "role": "inputs"}
  },
  "parameters": {},
  "scenario": {
    "kind": "stateless_single_batch",
    "initial_state": "cold",
    "state_policy": "reset_each_run"
  },
  "quality": {
    "kind": "numerical_equivalence",
    "reference": "typed_scalar/v1",
    "atol": 0.0,
    "rtol": 0.0
  }
}
```

上例的 `<64 hex>` 是規格中的佔位符，實際 parser 必須拒絕它。這個例子不包含
trained weights 或分類標籤，因此只能驗證 numerical equivalence，不能回報
模型 accuracy。將來可另加 `classification_dataset/v1` adapter，但現在不要宣稱
已有任意 ONNX／PyTorch model 的支援。

## 兩個第一版 adapter

### typed-graph/v1：真正使用共用 backend 的 application

`prepare` 讀取 graph/inputs，檢查固定形狀與資料，回傳 graph、inputs、scenario
限制及數值品質政策。Experiment 傳入 platform；共用 planner 呼叫每個 backend 的
`validate`，產生逐 node、逐 device 的支援／拒絕理由。

同一份 graph manifest 在 SRAM+GPU 和 HBM+GPU 平台重用，不修改 application
檔案。當 dtype 不相容時明確診斷，例如「INT8 linear 可在 SRAM 執行，HBM
目前只接受 FP16」；不得為了讓配置成功而隱式插入 cast。精度改變屬於新的
implementation variant，必須重做 quality evaluation。

第一版 mapping policy 可以是確定性的 `first_feasible`，由 experiment 指定
device preference，後續按照 device 名稱排序。它只承諾找到符合已檢查條件的
配置，不承諾最優。應記錄搜尋上限及候選順序；greedy 失敗不等於所有配置都失敗。

Preflight 至少檢查操作、dtype、shape、host input 路徑、節點間路徑、output
回 host 路徑，以及可得的容量下界。後端額外 packed workspace 若只能在 compile
時確定，診斷必須明示 `workspace_check: deferred_to_compile`，不能預先聲稱
完整容量驗證通過。最好另讓 backend 提供非執行的 `estimate_workspace`。

### starc-artifact/v1：原始 application artifact 的受控執行

STARC 的公開 artifact 有兩條不同路徑：真實模型／資料集 quality evaluation，
以及基於 AttAcc 的 GPU–PIM system simulator。前者真的輸出任務答案；後者
依模型形狀和 KV row budget 預測 timing/energy。應分別標記。

System simulator adapter 接收 model 名稱與 pinned source 中的模型設定、batch、
prompt length、generated tokens、method 和 KV budget table。硬體 `system`、
GPU 類型／數量、PIM 設定應屬於 experiment target，不能固定在 app manifest。
`method` 和 budget 屬於 implementation variant，不能偷偷改動基準 task。

每一組 variant 使用獨立工作目錄、獨立 cache，保存原始 stdout/stderr、CSV、
trace、參數、source/binary hash。STARC README 指示切換方法時刪除
`ramulator.out`，非 STARC 方法還需移除 clustering overhead 的兩行程式；adapter
應把這些差異固化為受測試的明示變體／patch，而不是使用者手動改 source。

此 adapter 的 execution mode 必須是 `original_artifact_system_model`，不能因為
透過 Foundry 啟動便改稱 `ported_to_foundry_backends`。它能驗證框架的 experiment
介面、資產綁定、cache 隔離、比較條件和 provenance；不能單獨證明共用 IR 已支援
LLM inference。

建議的 state/scenario 描述如下；未有實際 KV tensors 時直接標為 shape-only：

```json
{
  "kind": "llm_prefill_decode_model",
  "batch_size": 16,
  "prompt_tokens": 2048,
  "generated_tokens": 32000,
  "state": {
    "kind": "kv_cache",
    "representation": "shape_and_budget_only",
    "lifetime": "one_sequence",
    "reset": "before_each_run"
  },
  "request_schedule": "fixed_batch",
  "scope": "artifact_declared_system_model"
}
```

這些尺寸來自 repository README 的 E4 命令範例；完整規模與短 smoke run 應有不同
scenario hash，不能把縮短結果標為原論文 workload 已重現。

## 最小 Python 介面

可以以一般 dataclass/Protocol 實作，不需引入大型 framework dependency：

```python
class ApplicationAdapter(Protocol):
    name: str
    version: str

    def prepare(self, manifest, asset_resolver) -> PreparedApplication:
        """Validate and bind assets; no external process or backend execution."""

    def execute(self, prepared, experiment, output) -> ApplicationResult:
        """Delegate to unified.run or a pinned original artifact runner."""
```

`PreparedApplication` 必須帶有 `execution_mode`、capability requirements、
canonical application/scenario/quality 物件及資產 hashes。typed adapter 額外
持有 typed graph/inputs；artifact adapter 持有經驗證的 immutable run specification。
共用 runner 負責建立乾淨目錄、輸出 manifest、執行期限制、結果格式與 claim gate。

外層 app bundle 建議包含 `application.json`、`experiment.json`、`diagnostics.json`、
`result.json` 和 `execution/` 子目錄。`unified.run` 的既有 bundle 放在子目錄，
避免向它已封存的 manifest 加入新檔後破壞既有完整性驗證。外層與內層 manifest
各自有明確的 artifact coverage。

## 比較必須保留兩種 identity

**Task identity** 包含 model/shape 定義、資料集或具體輸入、前處理、scenario、
quality target；代表兩組實驗是否在回答同一個問題。

**Execution identity** 包含 adapter/source/runtime、method、lowered graph、實際
KV-budget trace、platform、mapping、成本假設；代表如何完成任務。Full attention
與 STARC 的 execution identities 理應不同，不能要求 budget 檔案相同才允許
比較；反之，只用相同 model 名稱也不足以保證公平。

既有 `unified.compare` 要求 workload 與 platform hashes 相同，適合它目前的
同圖、同模型 placement 對照。新的 application contrast 應另定
`same_platform`／`equal_budget`／`whole_system`，記錄允許變動的欄位；跨平台比較
要有相應的明示公平條件，不能在舊 compare 中刪掉 hash 檢查就算支援。

| 想提出的結論 | 最少需要的證據 |
|---|---|
| 原始 artifact 可重跑 | 固定來源、參數、資產與 runtime；原始結果；重跑一致性 |
| 共用 backend 正確執行此圖 | typed/ISA 實際輸出、參考核對、完整資料流；external artifact 啟動不算 |
| supplied model 中的延遲比 | 同 task/scenario/cost scope、公平條件及 method 隔離；明示條件和模型來源 |
| 保持任務品質的加速 | 上項加上與相同 model/dataset/method/budget 綁定的 task quality 結果 |
| 真實硬體加速 | 對應裝置上的 measurement、計量範圍、同步與重複統計；simulator cycles 不算 |

Quality observation 至少區分 `passed`、`failed`、`not_evaluated`、`not_applicable`。
缺少 quality 不可填成零誤差，也不可因 script exit 0 就通過。CPU-only STARC
timing experiment 可以回報條件式 model latency ratio，但 quality-preserving
speedup eligibility 應是 false，並附 `task_quality_not_evaluated`。

## 最小驗收測試

1. 同一 app manifest，兩種 platform 和任意改名的 device，能產生合法 placement；
   application bytes/hash 不變，execution hash 改變。
2. 增刪 device 或把 graph dtype 改成後端不支援值，檢查會列出所有受影響的
   node/device 理由；不偷偷改 graph、不執行 subprocess。
3. 缺輸入路徑、缺 output 回傳路徑、資產 hash 不符、unknown schema、未知欄位、
   非有限 metric、錯誤 shape 都在相應階段拒絕，並保存可定位的原因。
4. graph adapter 不接受 stateful/decode scenario；STARC adapter 明示接受模型化
   的 prefill/decode，但不聲稱真的產生 tokens/KV values。
5. 改 prompt length、batch 或原始模型定義使 task/scenario identity 改變；只改
   STARC method/budget 使 execution identity 改變，保留可比較的共同 task identity。
6. 連續執行 STARC → Full → STARC，cache 不相互污染；每種執行與隔離的直接
   upstream 執行相符。原始 patch 若不同，結果保留差異。
7. numerical pass 不會提升為 task-quality pass；timing-only STARC 不會提升為
   quality-preserving/hardware speedup。拒絕把 simulator wall time 當 modeled latency。
8. 外層 bundle 可驗證及重跑，修改 graph、budget、source、binary 或結果任一內容
   都能被發現；相同 task 的不同合法 variant 不會被誤判為完全相同執行。

## 後續擴充應由真實 application 驅動

完整 LLM port 需要多維／動態 shapes、batched matmul、softmax/masking、normalization、
activation/gating、reshape/transpose、gather/scatter、位置編碼和 KV state 更新；
還要 model/tokenizer/dataset frontend、sampling policy、TTFT/ITL 與多請求調度。
Vision frontend 則可能需要 convolution、pooling、image preprocessing 和 label
evaluation。這些不能只加 operator 名稱：每個操作須有數值語意、shape 檢查、
reference、至少一個 executable backend 和能力診斷。

現在先完成上面的 app/experiment contract 和兩個真實 adapter。它們的價值在於
讓未來 frontend、stateful runtime 和新的 paper artifact 都能繼承相同的資產、
scenario、品質及比較邊界，而不必把每篇論文重新包成不可互比的一次性腳本。

## 審查依據

- 本地 `pim_foundry/typed.py`、`unified.py`、`unified_examples.py`、`workloads.py`。
- 本地 `docs/backend-plugin.md`、`docs/native-quickstart.md`。
- [STARC 公開 artifact](https://github.com/EPIC-RPI/STARC)，本地
  `external/starc/README.md` 的 E1–E4 與 cache/clustering 注意事項。
- 本地 STARC `simulator_starc/src/model.py`、`system.py`、`main.py`，確認
  simulator 使用形狀、layer costs 和 KV-budget 情境，非 token-level model execution。
