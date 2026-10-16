> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

# v0.7 論文 application 驗證

作者程式和 CPU reference 可以共同犯錯：本輪 PIMbench histogram 把 BMP
尾端色彩描述資料當成像素，作者 reference 仍然通過。v0.7 保留原始執行，
加入獨立語意檢查並回報 `quality_failed`，讓效能比較能拒絕這種結果。

本輪新增 `pimbench-artifact/v1`、`cimflow-artifact/v1`、
`neupims-artifact/v1`。三者沿用既有 application／experiment／quality／
來源快照契約，執行固定版本的作者 artifact，沒有把完整論文應用改寫成
共同 typed IR。既有 typed graph、ONNX 和 STARC 路徑仍保留。

## 實際執行與結論

| Artifact | 執行範圍 | 實際查核 |
|---|---|---|
| PIMeval/PIMbench，IISWC 2024 | Histogram、triangle count、KNN；bit-serial、Fulcrum、bank-level 三種模型，明示縮小的幾何配置 | 九組框架結果與獨立直接執行一致；三角形數均為 7,168，通過獨立 oracle；一般案例與原始 BMP 反例均完成來源快照重跑 |
| CIMFlow，DAC 2025 | 原始 ResNet18 ONNX→CG→MLIR operator compiler→ISA→SystemC，batch 8、64 cores，作者 dp／baseline1 | dp 的直接核對與一次重跑通過；baseline1 的 ISA bytes 一致，但原生時序有差異，未通過精確接入一致性驗收；保存 compiler 省略 pooling／classifier 的實際範圍 |
| NeuPIMs，ASPLOS 2024 | 原始 512-request、單層單 token decode timing；NeuPIMs、NPU-only，另加相同請求長度的 RR 分配對照 | 原始兩條路徑與獨立直接執行的 counters 一致；三個變體皆完成來源快照重跑；錯誤替換為作者不同長度的 RR 輸入會在執行前被拒絕 |

各結果包、品質狀態、計量範圍與詳細研究報告集中在
本輪索引 (local artifact: `../runs/paper-validation-v07/report.html`)。索引計數包含 replay，
不能當作不同工作負載數量。共有 **31 份完整性核對通過的結果包**，其中
兩份是 BMP 品質失敗及其 replay；另有兩份 CIMFlow 研究總結未通過精確重現。

### PIMbench：作者 reference 與任務語意分開驗證

- 原始 `sample1.bmp` 有 273,280 個像素，作者程式卻計入 274,328 個；差額
  1,048 來自尾端 ICC 資料。框架保留 numerical reference 通過與 task 失敗，
  並拒絕比較，來源快照重跑得到同樣反例。
- KNN 的 PIM 路徑和 CPU verifier 都把最後一欄 label 加入 Manhattan 距離。
  獨立 harness 呼叫未修改的作者函式，在三種架構都重現「只改 query label，
  預測就從 0 變 100」；feature-only 答案兩次都應為 0。一般 KNN 核對只能
  證明與作者 reference 一致，不能當作分類品質通過。
- 原始 Rank1 triangle-count 配置導致來源程式配置 16 GiB host scratch。
  實際嘗試曾收到 SIGKILL，沒有把它說成已確證的 kernel OOM。新的 preflight
  拒絕這個已知配置；完整演算法在明示的 bounded-functional 平台完成。

九組一般案例、原圖反例與 replay 共 29 次原生 application 呼叫，另有三次
KNN 原函式 harness。20 個 application 結果包中包含兩份相同 BMP 反例的
`quality_failed` 結果。這些失敗是驗收資料，不計作任務正確性通過。
所有觀測只適用於固定的 IISWC release，沒有推論最新 main／TACO 2026 版本
也有同樣問題。詳見 [完整證據](pimbench-validation.md)。

### CIMFlow：保留編譯器實際執行的範圍

ResNet18 的首次獨立 dp 與 baseline1 分別得到 3.560054 ms、4.874690 ms；能量分別為
17,082,780,117.398392 pJ、22,679,911,357.88909 pJ。單位與範圍來自作者
SystemC report，代表一次 batch 8 的 compiled backbone 模型成本。

baseline1 在框架執行得到 4.875508 ms；輸入 ISA 與獨立執行逐 byte 相同，
時間仍相差 0.000818 ms（約 0.017%），因此研究流程拒絕把它列為精確核對通過，
沒有調大容許誤差或發布合格的 baseline／dp 加速比。後續同 ISA 重複執行的
觀測與可能機制記在該研究報告；dp 曾通過的配對只適用於已觀測執行。

baseline1 的來源快照 replay 又得到 4.874690 ms，因此回報 `passed=false`：
task、execution、platform、cost 與 ISA 身分相同，但 metrics 不同。
額外 MobileNetV2 完整編譯／模擬也完成；直接執行為 17.443438 ms、框架為
17.443910 ms，ISA bytes 一致，精確核對同樣失敗。它沒有另做 application
replay，不能計作 replay 通過。這些觀測顯示乾淨執行環境下的時序不穩定，
沒有證明相同命令每次都隨機變動；確切根因仍需控制實驗。

發布的 compiler 省略 pooling／classifier，沒有執行完整 ONNX 的數值語意；
report 的 op count 又包含 compiler metadata 中的 FC 計數。因此入口只匯出
已明示範圍的 latency／energy，不據此宣稱整網路 accuracy、TOPS 或 TOPS/W。
原始 ONNX 為 32×32 輸入、100 類，沒有改稱 224×224 ImageNet 實驗。
詳見 [來源、建置與驗證](cimflow-application-validation.md)。

### NeuPIMs：固定 workload 再比較 mapping

作者附的 CLB／RR traces 都叫 batch512，但 sequence-length 集合不同。
本輪將 512 個有序長度保存為語意資產，再獨立指定 channel assignment：

| 實驗 | DRAM cycles，1 GHz | 允許的解讀 |
|---|---:|---|
| 作者 CLB，NeuPIMs main | 619,233 | 原始 trace 的單層單 token batch 模型時間 |
| 相同請求長度，重新分配 RR | 650,229 | 與 CLB 同 task／platform／cost，可做條件式 mapping 比較 |
| 相同請求長度，NPU-only 分支 | 5,937,045 | 分支、硬體與成本契約不同；共同比較入口拒絕直接算加速比 |

在這份固定 trace，CLB/RR 為 0.952330640435908，亦即 CLB 模型延遲低
4.7669%。這是單一 workload 的確定性模擬結果，沒有外推成所有請求分布的
平均改善，也沒有量測實機速度或 LLM 品質。八次成功原生執行包括兩次獨立
直接執行、三次 application 執行及三次來源快照重跑。詳見
[完整來源與控制變因](neupims-validation.md)。

## 交付驗收

- 完整測試 **420 項通過，0 項跳過**；啟用原生 HBM、SRAM 與 ONNX 路徑。
  完整輸出 (local artifact: `../runs/validation-v07.txt`)。
- `pim_foundry-0.7.0-py3-none-any.whl` 已在隔離目錄安裝，離開原始碼目錄
  執行原始 PIMbench triangle count；得到獨立 oracle 確認的 7,168，且結果包
  完整性通過。安裝驗收 (local artifact: `../runs/package-check-v07.json`)。
- 實驗來源快照、模型／輸入雜湊、binary／依賴／設定綁定與原始 logs 保留；
  runtime 是外部依賴，wheel 沒有包入各作者的大型 compiler／simulator。
- `app-compare` 持續要求相同 task、platform、cost contract 與 metric 範圍；
  未評估的 task quality 不會提升成品質保持或硬體效能通過。
- 獨立 review 後補強 NeuPIMs DRAM timing INI 的必要綁定檢查與兩項測試。
  已執行的實驗原先就有保存該檔案的 hash；封存結果沒有因入口補強而改寫。

套件 SHA-256、程式檔案 hash、測試 log hash、31 份結果包及索引連結核對
保存在 機器可讀驗收紀錄 (local artifact: `../runs/release-checks-v07.json`)。

這輪建立的是原生接入一致性、獨立反例與控制變因的證據。完整論文 figure
sweep、模型對實機的誤差、LLM／分類資料集品質仍各有獨立驗收要求。
[驗證方法及後續 CENT／CrossSim 候選](paper-validation-protocol.md) 明列
已執行項目與尚未執行項目。
