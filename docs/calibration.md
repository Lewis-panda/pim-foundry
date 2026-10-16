# 以外部量測檢查模型誤差

把同一個 workload、裝置配置、placement 與到達時間的預測和實測配成一組，才能判斷模型算出的 latency 差多少。`validate-model` 先核對這些條件及原始 log 的雜湊，再以保留的 validation workloads 檢查誤差；條件不一致的資料會直接拒絕。

此流程不擬合參數。`calibration` split 用來保存調整模型時使用的資料；`validation` split 用來檢查固定後的模型。兩者至少各有兩個不同的 workload SHA-256，且不得交集。所有樣本必須使用相同的模型與 recorded configuration digest。

## 資料夾與 dataset

所有引用的檔案都必須放在 dataset 所在資料夾內，可用子資料夾分類。路徑以 dataset 的父資料夾為基準；絕對路徑、`..`、symlink、重複檔案路徑及重用的 raw-log 內容都會拒絕。Prediction 使用 `experiment` bundle 裡 `results/` 的完整結果檔；每個檔案都包含 functional 與 simulation 結果。

`dataset.json` 的欄位必須恰好如下。下例只是資料組織及門檻的範例，沒有提供硬體實測結果。

```json
{
  "schema_version": 1,
  "description": "固定模型的四組外部量測驗證",
  "max_relative_error": 0.1,
  "samples": [
    {"split": "calibration", "prediction_file": "predictions/calibration_a.json", "measurement_file": "measurements/calibration_a.json"},
    {"split": "calibration", "prediction_file": "predictions/calibration_b.json", "measurement_file": "measurements/calibration_b.json"},
    {"split": "validation", "prediction_file": "predictions/validation_a.json", "measurement_file": "measurements/validation_a.json"},
    {"split": "validation", "prediction_file": "predictions/validation_b.json", "measurement_file": "measurements/validation_b.json"}
  ]
}
```

`max_relative_error` 必須是 0–1 的有限數值，不能使用布林值。Dataset 接受 4–1,024 個樣本；每份 JSON 最多 32 MiB。每個樣本只能有 `split`、`prediction_file`、`measurement_file` 三個欄位。不同樣本可以使用同一 split 內的同一 workload 配合不同 placement，但不能重用檔案或 raw log。

## Measurement 檔案

每份 measurement JSON 必須恰好包含下列欄位。請從對應 prediction 複製識別資訊，從真正的量測流程取得 latency 與原始 log，不要以 prediction 的數值充當實測。

| 欄位 | 內容與檢查 |
|---|---|
| `schema_version` | 整數 `1` |
| `evidence` | 字串 `hardware_measurement`；這是提供者的證據宣告 |
| `workload_sha256` | 與 prediction 頂層相同的完整小寫 SHA-256 |
| `config_sha256` | 與 prediction 頂層相同的完整小寫 SHA-256 |
| `placement` | 與 prediction 的 `simulation.plan.placement` 相同 |
| `arrivals_ns` | 與 `simulation.arrivals_ns` 完全相同的整數陣列 |
| `max_inflight` | 與 `simulation.max_inflight` 相同的正整數 |
| `initial_state` | 與 prediction 相同的 `cold` 或 `warm` |
| `scope` | 字串 `device_graph` |
| `latency_ns` | 每個 request 的實測 latency；有限正數陣列，長度與 arrivals 相同 |
| `provenance` | 恰好包含 `hardware`、`software`、`method`，三者都是非空字串 |
| `raw_log` | 恰好包含 `path` 與 `sha256`；路徑相對 dataset 資料夾，雜湊必須符合檔案實際 bytes |

`device_graph` 的邊界包含等待、輸入搬移、運算、跨裝置搬移與輸出搬移。Cold 包含本次權重載入；warm 排除事先載入的 setup，且必須對應相同的初始狀態。此邊界排除前後處理、編譯時間及 host software overhead；量測方法需要明確使用相同邊界。

## 誤差與輸出

對每組樣本，分別取預測 request latencies 與實測 request latencies 的中位數：

```text
relative_error = abs(predicted_median - measured_median) / measured_median
```

每個 validation 樣本的誤差都必須小於或等於宣告門檻，整份報告才會 `passed: true`。Calibration split 的誤差只做診斷，不影響這個通過條件。報告保存各樣本中位數與誤差、各 split 的平均和最大誤差、所有來源檔案的雜湊，以及驗證規則與整份資料集的雜湊。

```sh
python3 -m pim_foundry validate-model /path/to/dataset.json --out /path/to/new-validation-report.json
```

輸出檔必須尚不存在。Exit code：`0` 表示通過，`1` 表示資料有效但 validation 誤差超標，`2` 表示輸入或證據綁定無效。

## 將驗證結果附到特定 latency record

```sh
python3 -m pim_foundry qualify-latency \
  /path/to/latency-record.json \
  /path/to/prediction-result.json \
  /path/to/dataset.json \
  --out /path/to/new-qualified-latency.json
```

此步驟重新評估 dataset，且要求它通過。Result 檔案的 bytes SHA-256 必須確實出現在 validation split；只出現在 calibration split 的結果無法取得這份證據。Record 的完整 contract 必須與結果產生器一致，還會核對 latency samples、canonical raw-artifact hash、workload 與 configuration digest，以及數值正確性的宣告。

輸出仍標示 `evidence: analytical`，只新增 `calibration_protocol_sha256` 與 `calibration_artifact_sha256`。它只適用於這個已核對的 **median latency、device_graph、特定結果檔案**，不能延伸到其他 workload、不同配置、p95/p99、能耗或吞吐量。後續可將這些 records 交給 `benchmark-check` 檢查比較條件。

## 證據的實際界線

工具會重新計算 prediction 內嵌模型的雜湊，並把 measurement 的 configuration digest 綁到 prediction 已記錄的 digest。它不獨立驗證設定檔對應哪一台實體機器，也不確認 raw-log 內容確實來自硬體；`independently_audited` 永遠是 `false`。

Workload SHA-256 包含輸入值與權重。兩個不同的 SHA-256 可以只差 seed，不能因此推論測試涵蓋不同模型架構。工具也無法證明 validation 資料從未被拿來調參，或量測具備足夠統計代表性。這些事項需要研究者保留實驗設計與審查證據。

`tests/test_calibration.py` 中的 measurements 與 raw logs 都是明確標示的單元測試假資料。測試通過代表輸入檢查、綁定與誤差計算符合規格，不代表任何硬體模型已通過實測驗證。
