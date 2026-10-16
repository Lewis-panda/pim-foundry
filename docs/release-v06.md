> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

# v0.6：Application 入口與論文實驗接入

同一份模型與輸入已能保留不變，改由 experiment 決定 HBM+GPU、HBM+NPU 或
SRAM+HBM+GPU 配置。STARC 的 Full 與 STARC 方法也使用同一 application；
方法及 KV budget 另列為實作變體，獨立保存 cache、來源與原生模擬證據。

## 已完成的接入設計

- 版本化 application／experiment schema；資產內容綁定、scenario 與 quality policy。
- 明確分開 task 與 execution identity。方法改變不等於任務改變；batch、context、
  output length 改變則不能再當成同一情境比較。
- 三個 adapter：typed graph、ONNX graph、STARC 原始 artifact。
- Graph 能力診斷與有界配置搜尋；精度不相容直接拒絕，不隱式量化。
- 帶單位、範圍、聚合方式、來源的 metrics；數值品質、任務品質與硬體證據分開。
- 外層 application bundle、來源快照、完整性檢查與 frozen replay。
- `app-inspect`、`app-run`、`app-verify`、`app-replay`、`app-compare`。

完整入口與擴充方法見 [applications.md](applications.md)。共同 IR 目前仍是有界、
固定 shape 的無狀態 graph；完整 LLM stateful port 尚未完成。STARC 明示使用
原始系統模型，沒有假稱其 Transformer 已由共同 graph 後端執行。

## 真正執行的論文 application

[STARC（ASPLOS 2026）](https://doi.org/10.1145/3779212.3790226) 的作者 artifact
固定在 `63c8aaa7a02114cd4cf21f32e7621fab5b416957`。使用作者 Mistral-7B 形狀
配置、32 layers、H100×8 + bank PIM、batch 16、prompt 2048、generated 128。
原始系統模型執行 prefill 與 127 個 decode steps。

| 方法 | Prefill batch ms | 平均 decode step ms（整批） | 平均 decode step nJ（整批） |
|---|---:|---:|---:|
| Full | 137.06833917286735 | 0.9405744286683656 | 564791586.345942 |
| STARC | 137.06833917286735 | 0.9385560226692454 | 549221514.9294 |

這是指定 artifact 與短窗口下的預測。Full/STARC 的時間比是
`1.0021505439743275`；不能將它延伸成論文完整 workload 或硬體實測的 speedup。
clustering 在最後一步計入，輸出窗口長度會改變攤提。

兩個方法均通過：

- 與獨立直接執行作者程式的 `output.csv`、`ramulator.out` 逐 byte 相同。
- 各自一次 frozen replay，相同 metrics、quality、execution identity。
- 共 508 次原生呼叫的 trace hash、exit code、cycle 與命令計數核對。
- 每次建立空白 cache，沒有跨方法沿用作者 cache 的結果。

原始 controller 的 heap-use-after-free 已由 ASan 重現。獨立安全修正版通過
兩個 native trace 的 ASan/UBSan；原版與修正版的 Full/STARC 128-token 結果
亦逐 byte 相同。有限 sanitizer 檢查不代表硬體時序正確性已驗證。

作者模型宣告 GQA=4，但實際 Transformer 未使用；此次保留其語意並註記。
本次未執行 32,000-token 原始窗口，也未載入 pretrained weights、重新評估
LongBench/RULER/PG-19。Full profile 已匯出，可用相同 adapter 直接執行；
品質保持與硬體加速的 comparison eligibility 維持 false。

見 實驗報告 (local artifact: `../runs/starc-application-v06/report.html`)、
逐步證據稽核 (local artifact: `../runs/starc-application-v06/native-evidence-audit.json`)、
[來源稽核](starc-source-notes.md)。

## 新 application 的互通性驗證

ONNX frontend 實際讀取四個模型檔，使用三種 batch、八個輸入 seed：
96 組輸入、3,584 個輸出與 ONNX ReferenceEvaluator 相符。
FP16 最大絕對誤差為 0；FP32 為 `4.768e-7`。這些是產生權重的介面測例，
沒有當作 trained paper models 或任務 accuracy。

六組 application 部署／baseline 與六次 frozen replay 通過，包括相同 ONNX
application 在 HBM+GPU 與 HBM+NPU，以及明示混合精度圖在 SRAM+HBM+GPU。
不同平台的效能比較被規則拒絕；相同平台下才允許條件式模型比較。

見 應用部署報告 (local artifact: `../runs/application-v06/report.html`)、
ONNX 驗證 (local artifact: `../runs/onnx-frontend-v06/summary.json`)。

## 驗收與套件

- 392 項測試通過，0 skipped；包含可選 ONNX 依賴與真實 SRAM/HBM native checks。
- 17 個 application bundles 完整性檢查通過（含重跑與最後 wheel smoke）。
- wheel 安裝到獨立目錄後，實際執行 ONNX→HBM+GPU 與論文 adapter preflight。
- Wheel：`dist/pim_foundry-0.6.0-py3-none-any.whl`。
- SHA-256：`0e3b9fe58f9ec19e940fe20b3b2a85acce3c8639f46289d3c3f9b92c758a8e73`。

驗收記錄：release-checks-v06.json (local artifact: `../runs/release-checks-v06.json`)。
測試 log：validation-v06.txt (local artifact: `../runs/validation-v06.txt`)。
v0.4/v0.5 的既有封存實驗保留原 engine 與 manifest；沒有改寫舊結果。
