# Application 入口與實驗契約

同一份 ONNX 模型換成 HBM-PIM + NPU 時，模型檔與輸入不應跟著改寫。
`application` 保存模型、輸入、精度、執行情境與驗收要求；`experiment` 才指定
平台、方法變體與配置策略。兩份檔案經過能力檢查後才執行。

## 已實作的九條入口

| Adapter | 真正執行的內容 | 適用範圍 |
|---|---|---|
| `tensor-graph/v2` | 一般 n-D 張量、明示精度、原生 SRAM／HBM 與分析裝置後端；共同事件排程 | symbolic shape、broadcast、attention／norm／CNN 等算子；多請求冷暖權重與容量檢查 |
| `onnx-tensor/v2` | ONNX 匯入共同 n-D IR，沿同一張量／後端路徑執行 | 明列且經 ReferenceEvaluator 驗證的 CNN／attention 子集；未支援內容直接拒絕 |
| `transformer-graph/v1` | 完整 GPT-2 架構的預訓練權重、prefill／decode、實際 token、KV 與 NLL | DistilGPT-2 完整數值驗證；小型完整 GPT-2 的明示 FP16／原生 HBM 接入 |
| `typed-graph/v1` | 同一張 typed graph 分配到 SRAM/HBM 與 GPU/NPU 後端，實際中間值傳給下一個節點 | 固定形狀、明示 INT8/INT32/FP16/FP32 的冷啟動單批次圖 |
| `onnx-graph/v1` | 讀取 ONNX、lower 到 typed graph、執行後端，最後和原始 ONNX ReferenceEvaluator 核對 | 有界的 MatMul/Gemm/Add/Relu/Cast/Flatten/Identity 子集；限制詳見 ONNX 文件 |
| `starc-artifact/v1` | 執行固定版本 STARC 的完整 transformer 形狀／成本模型與逐 decode step 的原生 PIM trace | 作者的 Mistral-7B 配置、Full 與 STARC；不載入權重或產生 token 值 |
| `pimbench-artifact/v1` | 執行原始 C++ histogram／triangle count／KNN，保留 PIM 功能驗證與分析成本；獨立查核圖像邊界及三角形數 | IISWC 2024 release，bit-serial／Fulcrum／bank-level；明記 bounded-functional 與 paper-rank1 profile |
| `cimflow-artifact/v1` | 原始 ONNX→CG→OP→ISA→SystemC；Conv／residual backbone 的映射與成本 | DAC 2025 author artifact，64 cores、32×32 輸入；省略 pooling／classifier，不宣稱整網路語意或分類準確率 |
| `neupims-artifact/v1` | 原始 NPU–PIM／NPU-only simulator，保留 sequence lengths 與各自 channel assignment | ASPLOS 2024 artifact，512 requests、單層單 token 的成本模型，沒有生成 tensor／token 值 |

新版三條入口回報 `shared_tensor_graph`；v1 graph 回報 `shared_graph_backends`；STARC 回報
`original_artifact_system_model`。接入原始 artifact 已能測試 application、scenario、
cache 隔離與比較契約。完整 Transformer 的共同 IR 驗證由獨立的 `transformer-graph/v1` 路徑提供，不能把該結果移用為 STARC 的任務品質證據。
新增的三個 artifact adapter 同樣保留作者實作；它們共用 application、品質、
計量範圍與重現契約，沒有把其完整應用改寫為共同矩陣 IR。

## 可以直接重跑的例子

本 checkout 的 `.venv-onnx` 已安裝固定版本的可選依賴。新的環境可用
`pip install '.[onnx]'`；基礎套件仍只有標準函式庫依賴。

```sh
.venv-onnx/bin/python -m pim_foundry app-inspect \
  --application runs/application-v06/specifications/onnx-application.json \
  --experiment runs/application-v06/specifications/onnx-hbm-gpu-native.json

.venv-onnx/bin/python -m pim_foundry app-run \
  --application runs/application-v06/specifications/onnx-application.json \
  --experiment runs/application-v06/specifications/onnx-hbm-npu-native.json \
  --out runs/my-onnx-npu

.venv-onnx/bin/python -m pim_foundry app-verify runs/my-onnx-npu
.venv-onnx/bin/python -m pim_foundry app-replay runs/my-onnx-npu --out runs/my-onnx-npu-replay
```

`app-inspect` 不執行模擬器。它對每個節點列出各裝置的支援情形、精度／shape
拒絕原因、host 與節點間路徑、儲存容量下界。確切 live tensors 與後端 packed
workspace 容量仍在執行階段驗證，診斷會明列這個邊界。

## Manifest 與 deployment

Application 有固定的七個欄位：`schema`、`id`、`adapter`、`assets`、
`parameters`、`scenario`、`quality`。所有資產帶 `path`、`sha256`、`format`、
`role`；讀入與快照時均核對雜湊。路徑可相對於 application 檔案，並在結果包
內改寫成可攜的相對路徑。`role=model/inputs/preprocessing/quality` 定義任務，
`implementation` 定義執行版本。內建 adapter 會檢查所需角色，不能用任意標籤
把模型或輸入移出任務身分。

ONNX application 不包含 device 或 node placement：

```python
from pim_foundry.application import asset, APP_SCHEMA, STATELESS

manifest = {
    "schema": APP_SCHEMA,
    "id": "my-model",
    "adapter": "onnx-graph/v1",
    "assets": {
        "model": asset("model.onnx", "model", "onnx"),
        "inputs": asset("inputs.json", "inputs", "json"),
    },
    "parameters": {"input_shapes": {"features": [2, 16]}},
    "scenario": STATELESS,
    "quality": {
        "kind": "numerical_equivalence",
        "reference": "onnx.reference.ReferenceEvaluator",
        "atol": 0.001,
        "rtol": 0.001,
    },
}
```

這裡的形狀與誤差門檻是範例，須依模型指定。浮點規約不同時不能只把門檻放大
直到通過；保存的原始模型、參考值與誤差供研究者判斷。

Experiment 的固定欄位是 `schema=pim-foundry.experiment.v1`、`platform`、
`mapping`、`variant`。Graph 的 `variant` 為空物件；模型實作由 application
的 model asset 綁定。STARC 的 Full/STARC 方法與 KV budget 放在 variant，
兩者可保留共同任務身分。

Graph mapping 可選：

```json
{"strategy":"first_feasible","device_preference":["hbm","gpu"],"max_candidates":4096}
```

此策略按照明示裝置優先順序做有界搜尋，名稱可任意更改，沒有最優性保證。
`max_candidates` 限制嘗試的 node-device assignments 數目。另一種是
`{"strategy":"explicit","placement":{"projection":"hbm","activation":"gpu"}}`。
兩者都不會隱式轉換精度。

## 把「同一個任務」和「不同方法」分開

`task_sha256` 綁定語意資產、模型參數、scenario、品質要求；不受顯示名稱或
路徑搬移影響。`execution_sha256` 另綁 adapter、來源快照、方法、KV budget、
配置與 runtime。STARC 改變 KV budget 是合法的方法差異；改 batch、輸入長度
或輸出長度就改變任務情境。

`app-compare` 先驗證兩個完整結果包，再要求 task、platform、cost contract、
execution mode 以及 metric 的單位／範圍／聚合方法一致。它回報同一模型條件下
的 left/right 比值。不同平台需另行定義資源預算或全系統比較協議；目前這個
入口會拒絕，避免只把裝置名字改掉便自稱公平。

品質觀測分開保存：

- `numerical`：已執行輸入上的數值一致性。
- `task`：資料集／任務品質，未評估則明記 `not_evaluated`。

兩者可以得到不同結果：PIMbench 的原始 BMP 測例與作者 CPU reference 相同，
但把非像素 metadata 算進 histogram，因此為 `numerical=passed`、`task=failed`。
整體狀態會變成 `quality_failed`，比較入口拒絕使用該結果。KNN 原始 reference
也把 label 納入距離；一般原生核對不因此獲得分類品質通過的資格。

數值通過不會提升成 accuracy 通過。STARC 的時間模型沒有重測 LongBench，
仍可產生明示條件下的時間比；`quality_preserving_comparison_eligible` 會是 false。
模擬結果也不會提升成硬體實測。

## 重現與新增 application

外層結果包封存 application、資產、experiment、診斷、指標、engine 快照與
`execution/`。內層原生 graph 或 paper artifact 保留自身輸出與 provenance。
`app-verify` 檢查完整檔案集合與雜湊；這是完整性檢查，並非第三方簽章。
`app-replay` 和 `reproduce.py` 都使用封存 engine；可選套件與原生 runtime 仍須
存在且通過相應綁定檢查。已封存結果不會因新版本程式碼而被覆寫。

新 frontend 可以使用 Python 顯式註冊：

```python
from pim_foundry.application import register_adapter
register_adapter("my-application/v1", MyAdapter())
```

Adapter 提供 `prepare(manifest, assets)`、`inspect(prepared, experiment)`、
`execute(prepared, experiment, output)`。前兩步負責檢查契約；最後一步回傳
有單位／範圍的 metrics、兩種 quality observation、platform identity、cost
contract 與 execution identity。不得把 host 執行模擬器的秒數當成裝置延遲。
JSON 不載入任意 Python 路徑或 shell 命令。第三方 adapter 必須提供固定版本的
註冊 entrypoint 供重現；內建 `reproduce.py` 只自動註冊內建 adapters。

完整 GPT-2 架構的共同 n-D IR、attention／normalization、KV state、預訓練權重、token 與固定資料片段品質檢查已由 `transformer-graph/v1` 提供。一般 v1 無狀態 graph 仍拒絕 decode scenario；新的應用可沿用共同 application／experiment／quality 契約。詳見 [張量契約](tensor-runtime.md) 與 [Transformer 範圍](transformer-application.md)。

相關： [ONNX 支援範圍](onnx-frontend.md)、[STARC 接入](starc-application.md)、
[來源稽核](starc-source-notes.md)、[設計審查](application-design-review.md)。
