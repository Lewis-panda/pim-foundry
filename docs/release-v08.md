> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

# v0.8 軟體交付驗收

完整 application 現在經過共用張量 IR、明示精度轉換、異質後端、資源排程、品質檢查與封存比較。驗收依據是這些路徑實際執行、獨立比對和重播的結果。

機器可核對的驗收總表 (local artifact: `../runs/release-v08/report.html`) · 完整 JSON 與雜湊 (local artifact: `../runs/release-v08/report.json`) · [使用入口](../README.md)

## 已完成的軟體範圍

| 項目 | 實作與驗收 |
|---|---|
| 共用 n-D IR | 明確 dtype、broadcast、符號形狀、SSA、預算、一般 matmul／norm／attention／CNN 算子；獨立 ONNX 與 scalar oracle。 |
| Application 接入 | 九個明示 adapter；一般張量、ONNX、完整 GPT-2 和既有論文 artifact；模型／資料／場景與平台／方法分開綁定。 |
| 完整模型 | DistilGPT-2 的 embedding、六層 decoder、MLP、norm、LM head、IR argmax、prefill、增量 decode、實際 KV 和 token。 |
| 原生後端串接 | SRAM INT8／INT32、HBM FP16／FP32 累加的明確契約；原生指令數值傳給後續節點；完整小型 GPT-2 經 HBM 執行。 |
| 排程與容量 | 同一執行 profile 的 compute、link、DMA 競爭；抵達時間、併發上限、權重冷暖狀態、live tensor／packing 容量。 |
| 公平比較 | 同平台、共同預算、整體配置三種比較；固定任務、精度／品質政策、範圍、單位、彙整方法和來源；重播失敗阻擋合格比較。 |
| 統計與校準 | 確定性重跑、獨立 run bootstrap、明示樣本單位；generic metric calibration 的訓練／held-out 分離與原始資料綁定。 |
| 可安裝交付 | v0.8 wheel、CLI、選用依賴、來源／資產快照；從 checkout 外的安裝目錄實跑混合 CIM、ONNX、完整 HBM Transformer 及重播。 |

完整回歸 **518 項通過，0 跳過**，包含本機原生 SRAM／HBM 檢查。測試紀錄 (local artifact: `../runs/validation-v08.txt`)

交叉審查另修正並回歸驗證：執行但未使用的節點也計入完成時間；抵達時間整體平移不改變 makespan／能量；SRAM K>128 必須用明確跨 tile reduction graph，不能藏入未計費 host 加總；原生 runtime 內容雜湊納入 Transformer 平台與執行身分；未知能量保持未知。

## 實際模型與對照

**DistilGPT-2：**原始公開預訓練權重，六層、768 hidden。相同輸入對照獨立 Transformers 4.44.2 CPU reference：8 個生成 token 一致、12 組 KV 張量通過、logits 最大絕對誤差約 `2.06e-4`。WikiText test 固定片段的 128 個預測目標，NLL 差 `9.84e-7`；此數值只代表該片段的品質保持。執行報告 (local artifact: `../runs/transformer-application-v08-release/run/report.html`) · 平台模型報告 (local artifact: `../runs/transformer-platform-v08-release/run/report.html`)

**完整 tiny GPT-2 → HBM：**指定八個線性節點明示轉為 FP16，經原生 HBM 指令數值路徑；其餘算子用共同功能引擎與 GPU 分析模型。每次 application 執行 32 個原生 tiles，包含 prefill、decode、teacher forcing 與 KV；另有重播及同平台、同精度的 all-GPU baseline。生成 token 一致，固定 16-target fixture NLL 差約 `9.47e-8`。tiny checkpoint 用於功能接入驗收。原生執行報告 (local artifact: `../runs/transformer-native-hbm-v08-release/run/report.html`)

完整比較鏈的條件式 latency 比值為 **HBM 混合／all-GPU = 1.01402948**，即此案例的混合配置在指定模型下約慢 1.4%。品質、重播與同平台比較契約皆通過；不推論實機快慢。比較與證據 (local artifact: `../runs/transformer-comparison-v08-release/report.html`)

**一般應用：**四種平台執行同一批次張量計算；原生 SRAM→cast→HBM→cast→softmax 的每個輸出實際流入下一節點。完整 ONNX CNN 經共同 IR 後，與原始 ONNX 最大絕對誤差 `2.98e-8`。錯誤張量 reference 與錯誤生成 token reference 都確實得到 `quality_failed`。混合平台摘要 (local artifact: `../runs/tensor-platform-v08-release/summary.json`) · ONNX 報告 (local artifact: `../runs/onnx-tensor-v08/application/report.html`)

## 研究結論的證據邊界

- GPU／NPU 成本是明示分析模型；HBM 時序是修復版 DCC 的 request projection，保留完成時間假設。未建立實機校準；本機 GPU driver 不可用。
- Transformer 目前使用等長 batch、host 上的持續 KV，跨 graph 呼叫的 KV／權重搬移計入成本。算子與精度支援表就是當前可執行契約；不宣稱任意模型、任意廠商 ISA 或 continuous batching。
- 實作提供校準與統計規則；測試合成校準資料不會成為真實晶片量測證據。
- CIMFlow 原始實驗的重現失敗持續保留；比較器會阻擋。PIMbench、STARC、NeuPIMs、DCC 的驗證範圍沿用各自報告。
- IANUS 已完成公開資料稽核與第三方候選 smoke。作者 artifact 未取得，沒有將第三方程式或圖表算術冒充作者系統重現。

這些限制被保存為模型與證據契約；軟體驗收不會把它們改寫成硬體認證或社群共識。

## 重跑本版驗收

```sh
.venv-onnx/bin/python scripts/release_acceptance.py --out runs/my-release-audit \
  --tensor-study runs/tensor-platform-v08-release \
  --transformer-study runs/transformer-application-v08-release \
  --transformer-platform-study runs/transformer-platform-v08-release \
  --native-transformer-study runs/transformer-native-hbm-v08-release \
  --onnx-study runs/onnx-tensor-v08 \
  --comparison-study runs/comparison-protocol-v1-final \
  --transformer-comparison runs/transformer-comparison-v08-release/native_hbm_vs_gpu_fp16 \
  --test-log runs/validation-v08.txt \
  --package-check runs/package-v08-final/report.json \
  --wheel dist/pim_foundry-0.8.0-py3-none-any.whl
```

此命令重新驗證既有封存資料、程式／wheel／結果雜湊與各項驗收條件。重新產生應用觀測使用各份 bundle 的 `reproduce.py` 或 [Transformer 文件](transformer-application.md) 與 [平台文件](tensor-platform.md) 中的執行腳本。
