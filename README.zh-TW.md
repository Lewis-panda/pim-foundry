# PIM Foundry

**把同一個應用接到不同 CIM／PIM 與加速器配置，核對實際輸出，並保留每項比較的依據。**

[English](README.md) · [開始使用](docs/getting-started.md) · [架構](docs/architecture.md) · [限制](docs/limitations.md) · [驗證紀錄](VALIDATION.md)

放在 HBM 的矩陣運算會讀取前一節點真正產生的張量，結果再傳給下一節點；另外的資源模型計算運算、搬移、競爭與記憶體容量。更換平台或節點配置時，應用、輸入與品質要求仍有明確定義。

PIM Foundry 是用於 compute-in-memory（CIM）、processing-in-memory（PIM）及異質系統的研究框架，提供共用應用契約、可執行張量圖、模擬器接入與比較規範。目前持續開發中；介面有版本標示，1.0 前仍可能調整。

## 直接執行

需要 Python 3.10+ 與 CPU。安裝完成後，以下範例不需要網路、模型下載、GPU 或外部模擬器。

```sh
git clone https://github.com/Lewis-panda/pim-foundry.git
cd pim-foundry
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[onnx]'
python examples/quickstart.py --out runs/quickstart
```

範例計算實際張量、核對獨立數值參考、評估明示的 HBM／GPU 分析配置，並完成結果封存驗證與重播。報告位於 `runs/quickstart/application/report.html`，總表位於 `runs/quickstart/summary.json`。延遲與能量是**使用範例係數的模型預測**，沒有在實體 HBM-PIM 或 GPU 上執行。每次請使用尚未存在的輸出目錄。

[使用說明](docs/getting-started.md)介紹如何檢查結果及更換應用或配置；只需要 NumPy 張量執行時，可改裝 `.[tensor]`。

## 目前提供的功能

| 項目 | 實作範圍 |
|---|---|
| 共用應用入口 | 綁定模型、輸入、場景、參考輸出與品質政策；平台與配置另行定義 |
| 張量執行 | n-D 張量、明示精度、符號維度、broadcast、attention、normalization、CNN 與顯式狀態輸入／輸出 |
| ONNX 匯入 | 文件列明的算子子集，可與原始 ONNX 模型核對數值 |
| 完整 GPT-2 應用 | 真實權重、embedding、所有 decoder layers、LM head、prefill、增量 decode、KV cache、greedy tokens 與限定範圍的 NLL 檢查 |
| SRAM／HBM 接入 | INT8／INT32 CIMFlow 指令解譯；FP16 HBM 指令解譯與分開標記的 DCC 時序投影 |
| 異質排程 | Compute、單向 links、端點 DMA、張量生命週期、workspace、容量、請求抵達與權重冷暖狀態 |
| 可重現實驗 | 資產及原始碼快照、檔案雜湊、數值檢查、報告、驗證與重播 |
| 比較規範 | 同平台、相同預算、整體配置三種比較；資源揭露、品質、計量範圍、重複觀測與校準證據 |

**裝置名稱本身不能證明實機執行。** 目前 GPU／NPU 使用分析成本與 CPU 數值計算。原生模擬器路徑需要另外取得及建置上游 runtime；尚未建立跨裝置的實機校準或通用準確度保證。

## 接入自己的研究

自行撰寫的圖使用 `tensor-graph/v2`；支援的 ONNX 模型使用 `onnx-tensor/v2`；已實作的 GPT-2 架構使用 `transformer-graph/v1`。Application 決定工作負載與品質檢查，experiment 決定裝置、連線、節點配置與成本假設。

- [張量與數值契約](docs/tensor-runtime.md)、[ONNX 支援表](docs/tensor-onnx.md)
- [Transformer 狀態與評估](docs/transformer-application.md)
- [裝置執行與排程](docs/tensor-platform.md)
- [應用 adapter](docs/applications.md)、[後端擴充介面](docs/backend-plugin.md)

不支援的算子、精度、路徑或容量會明確失敗。精度轉換必須出現在圖中；擴充後端需要顯式 Python 註冊，JSON 不會任意載入程式。

比較時先固定應用與計量範圍，再選擇同平台、共同資源預算或整體配置。數值正確、任務品質、重播、論文重現及硬體校準分別記錄；檔案完整性不等於量測經過外部認證，未知能量也不會算成零。見[比較規範](docs/comparison.md)及[研究證據](docs/research-evidence.md)。

## 論文與驗證範圍

目前包含 PIMbench、CIMFlow、NeuPIMs、STARC、DCC 的部分應用或模擬器接入，以及 IANUS 的公開資料與數字審查。IANUS 審查時未取得可認證的作者執行 artifact。單一應用或元件跑通，不足以證明完整論文效能結果已重現。

此公開 repository 提供第一方程式、測試、配置及文件；沒有重新散布上游模擬器原始碼、下載的模型／資料集或歷史實驗封存包。[研究證據](docs/research-evidence.md)列出來源與已知問題；[VALIDATION.md](VALIDATION.md)區分公開 checkout 驗證與歷史本地驗收。

## 參與開發

```sh
python -m pip install -e '.[onnx]'
python -m unittest discover -s tests -v
```

需要選用依賴或外部 runtime 的測試，在環境不具備條件時可能跳過；請保留實際測試結果中的區別。回報問題與提交修改請參考 [Contributing](CONTRIBUTING.md)，並使用 [GitHub issues](https://github.com/Lewis-panda/pim-foundry/issues) 或 pull requests。

第一方程式採 [MIT 授權](LICENSE)。引用的第三方程式、論文、模型及資料集保留各自授權。
