> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

# SRAM／HBM 與 GPU／NPU 共用研究框架：契約、現況與驗收

> v0.4 已新增分析式事件模擬、矩陣算子、工作負載套件與重現流程；請先看 [使用流程](quickstart.md)、[模擬器契約](simulator.md) 和 [量測驗證](calibration.md)。本文保留 v0.3 的設計與比較規則；其中「目前只有功能模型」描述的是該版本。v0.4 的分析模型仍非硬體量測或週期精確模擬。
同一個 `y[n] = Σk x[k] × W[k,n]`，若交給 SRAM CIM，需要安排權重分塊進入陣列、輸入送到本地記憶體，以及部分和累加；若交給 HBM PIM，則要安排權重所在 bank、輸入廣播與結果讀回。當下一個算子移到另一個裝置，還必須把前一個輸出搬到它能讀取的位置。因此框架要共同描述「算什麼、資料在哪裡、誰何時能讀寫」，再依具體硬體產生指令。

目標涵蓋兩種使用方式：同一份模型可選 SRAM、HBM、GPU 或 NPU 目標；同一次模型執行也能分配工作給不同裝置。後者需要額外的互連、資料生命週期、同步及並行排程。兩種方式都必須以數值與資源契約驗收。

## v0.3 現在能做什麼

| 項目 | 已實作 | 實際邊界 |
|---|---|---|
| 運算表示 | `Graph`、`Tensor`、`Node`；常數權重 `matvec` 與向量 `relu` | 固定形狀、一／二維張量；尚無一般模型匯入器 |
| 功能參考 | 分塊計算、尾塊、跨 K 分塊累加，對照獨立 scalar oracle | int64 儲存、精確整數中間結果；每個節點的最終結果必須可存入 int64 |
| 目標配置 | SRAM／HBM／GPU／NPU 邏輯資源的容量、tile 大小與 unit 數 | 共用整數功能引擎與示意資源；`kind` 改變事件名稱，沒有廠商 ISA lowering |
| 跨裝置執行 | 每節點 placement、有向互連路徑、activation 複製、結果回 host | 依序執行；尚無兩個裝置同時運算或傳輸重疊 |
| 資源規劃 | live tensor、輸入暫存、權重 tile、路徑中繼暫存的邏輯容量檢查 | 完整輸入與輸出向量須容納；不計 Python 物件與無界中間整數的實體大小 |
| 搜尋與紀錄 | 列舉合法 placement、拒絕原因、逐連線搬移量、來源與 workload 雜湊 | 排序目標只有 logical transfer bytes，沒有延遲或能耗預測 |
| 比較契約 | 工作負載、品質、成本、證據種類、統計與校準聲明的相容性檢查 | 只檢查 metadata；沒有讀取／稽核被雜湊的外部證據，也沒有效能認證 |

目前 demo 用同一個 `matvec → relu → matvec` 模型，對所選裝置執行單裝置及跨裝置 placement，檢查輸出是否符合相同數學契約。`--devices hbm,gpu` 或 `hbm,npu` 可選異質組合，預設仍為 `sram,hbm`。這驗證共用表示與功能參考路徑；不能據此宣稱真實硬體已被支援。程式設定 reference tensor／工作量與搜尋候選數上限，以控制本地實驗資源。評測產物與公平比較的實際邊界見 [共同評測契約](benchmark-protocol.md)。

每次呼叫視為冷啟動：輸入與常數從 host 開始，權重在每個消費節點重新分塊載入。activation 的功能複製依規劃路徑執行；權重計算從 initializer 取 tile 並留下事件，其路徑與容量由 planner 記帳。沒有權重常駐快取、activation 複本快取或實體匯流排交易模型。最少 hop 的路徑也沒有被解讀為最低延遲。

原有 DCC trace 路徑保持獨立，狀態仍是 `trace_generated_not_numerically_verified`。新 reference 的正確性不替 DCC 的地址、累加或浮點語意背書。

## 要建立的共同層

下圖是目標架構；v0.3 只實作上述功能子集，尚未提供圖中的完整編譯與 runtime 層。GPU／NPU adapter 也需遵循同樣契約，目前僅有邏輯 profile。

```text
 [模型／手寫 kernel／研究 workload]
                  |
 [運算與數值契約：shape、dtype、reduction、誤差]
                  |
 [layout／placement／effect／dependency IR]
                  |
 [分塊、資料搬移、buffer lifetime、裝置分配與排程]
          |                    |                    |
 [功能 reference]      [SRAM target adapter]  [HBM target adapter]
          |                    |                    |
          |             [各自 ISA／SDK／simulator]
          +--------------------+--------------------+
                               |
 [一致性驗證、可重現實驗、經校準成本模型、比較與失敗診斷]
```

共同契約至少需要五部分：

1. **Operator**：確切索引、shape、broadcast／reduction 語意與合法輸入域；不支援的運算要回報原因。
2. **Numeric**：儲存格式、乘法／累加精度、累加順序、rounding、overflow／saturation 與量化尺度。允許近似時明列誤差與驗證方法。
3. **Layout**：邏輯元素到實體位置的對應、packing、padding、alignment 與重排；shape 相同不表示地址排列相同。
4. **Placement**：tensor／tile 所在 memory space、可讀取的 compute unit、複本與部分和所有者；跨位置使用必須有合法 transfer。
5. **Effects 與依賴**：讀、寫、累加、配置／釋放、同步完成條件。寫後讀與 buffer 重用須有可檢查的先後關係。

架構模型應把 memory technology、compute primitive、數值能力與連接方式分開。`SRAM` 或 `HBM` 標籤無法決定是否支援浮點、向量非線性運算、跨 bank 通訊或 analog 計算。長期模型應描述 memory spaces、compute units、links、共享資源與能力約束；目前 `ReferenceDevice` 的簡單欄位是起點。

數位目標按其實際整數或浮點規則驗證；analog 目標還需要 ADC／DAC、量化、飽和、噪聲／變異與校準條件。固定噪聲 seed 只能重現指定模型，不能證明真實晶片的誤差分布。目前的精確整數 oracle 用於抓取分塊與資料流錯誤，尚未實作這些硬體數值模型。

## Adapter 與一致性驗收

建議 adapter 介面依序提供 `capabilities`、`validate`、`lower`、`execute`、`collect`；成本估計另以帶校準資訊的可選介面提供。這是待實作介面，並非目前已有的 plugin API。每個 adapter 必須回報版本、支援範圍、產物格式及驗證層級，並保留 unsupported 的具體原因。

接入驗收必須讓共同 workload 經該後端指令或 SDK 執行後讀回數值；只有產生設定檔或 trace 時，狀態維持「未驗證數值」。共用測例涵蓋非整除 tile、容量邊界、不同 layout、負值、溢位、累加、分支 fan-out、無可達路徑及資料搬移。任何 host fallback 必須明示、計費並納入端到端結果。

真正的異質並行 runtime 還要檢查 event／wait 語意、buffer ownership、跨裝置可見性、transfer 與 compute 的資源競爭；先證明序列和並行執行符合相同數值契約，再評估重疊能否減少時間。

## 研究結果如何可重現與可比較

目前 demo 已保存完整 graph／input、platform、placement、workload SHA-256 與三個 reference 模組來源雜湊。後續實驗 bundle 應再固定 generator seed、依賴與 simulator commit、adapter／硬體設定雜湊、數值契約、cold／warm 狀態、原始輸出與失敗紀錄。另記錄環境及輸入產生方式，讓第三方能重建相同測例。

共同 baseline 至少包含 SRAM-only、HBM-only、手工異質配置、固定配置與待評估搜尋策略；加入 CPU／host baseline 時使用相同工作量和數值要求。消融須分開測試 placement、layout、權重重用、資料搬移與 pipeline 的貢獻。LLM 類 workload 還要控制 batch、sequence length、KV cache 成長及 prefill／decode 比例。

`logical_transfer_bytes` 是目前 reference 契約下跨各 link 的 payload 總量；它沒有描述 burst、padding、協定、頻寬、競爭或重疊。不能從此數字、事件名稱或命令數宣稱延遲與能耗改善。未來成本模型必須對獨立 workload 校準／驗證，報告誤差與適用域；實測做重複試驗與區間估計，決定性模擬則報告參數敏感度和模型誤差。

## 與現有工作的連接點

| 來源 | 可沿用的工作與限制 |
|---|---|
| [DCC](https://github.com/SPIN-Research-Group/DCC) | 公開資料分配與運算排程工具；此專案只接入固定版本的 HBM GEMV trace 子集，數值與 simulator 整合仍須完成。 |
| [CIMFlow](https://github.com/BUAA-CI-LAB/CIMFlow) | 公開數位 CIM 的 MLIR compiler 與 SystemC simulator，是 SRAM adapter 候選；目前沒有整合或建置。其 [MVM 契約](https://www.cimflow.org/docs/Abstraction/Instruction-Design/Instruction-Set/Matrix-Vector-Operation) 包含 CIM 權重、本地輸入與輸出累加，接入時須逐項驗證。 |
| [CINM／Cinnamon](https://github.com/tud-ccc/Cinnamon) | 可評估重用其多層 IR 與 lowering；[論文](https://esim-project.eu/files/Images/people/chair-cc/publications/2504_Khan_CINM_ASPLOS.pdf) 評估 UPMEM 和 memristor CIM，提供跨裝置成本模型的擴充位置；不是本專案已完成 SRAM／HBM 支援的證據。 |
| [HPIM v3](https://arxiv.org/html/2509.12993v3) | 研究同一系統中的 SRAM-PIM／HBM-PIM 分工、獨立指令流及同步；SRAM subsystem 另含 tensor／vector units，KV 資料仍需從 HBM 搬入。採模擬評估；本次未找到可確認的 HPIM compiler 公開發布連結。 |

來源查核日期：2026-10-14。這些工作可提供既有實作與對照，跨專案重用仍要查核實際介面、授權、版本及數值語意。是否採 MLIR／CINM，應由已驗證的後端需求與可重用程度決定。

## 讓其他研究者願意使用的增量價值

框架是否成為廣泛依賴無法預先保證。可驗收的價值是：加入新硬體時能沿用 workload、正確性測例、實驗紀錄與比較流程；調整 mapping 時能定位哪個 tensor、地址、累加或同步步驟出錯；跨論文比較時能指出不同數值與成本假設。這些需要可執行契約與檢查器持續維護。

| 階段 | 交付與驗收門檻 |
|---|---|
| 現在：功能基礎 | 共用 graph 在四類邏輯配置及跨裝置 placement 下與 scalar oracle 一致；比較器檢查聲明相容性，清楚區分邏輯驗證和廠商驗證。 |
| 下一步：具體後端 | 一個 SRAM、一個 HBM adapter 讀回結果，通過共用 conformance suite；已知不支援處明確拒絕。 |
| 異質共同執行 | 同一模型在兩者執行，有可追蹤 transfer／wait／readback，通過資料生命週期與並行正確性測試。 |
| 可信的研究比較 | 固定實驗 bundle、完整端到端成本、校準誤差、必要 baseline／ablation；外部使用者可重現。 |
| 可持續生態 | 版本化 schema、相容性測試、乾淨環境範例，以及由外部貢獻者新增 adapter 的實際成功案例。 |

目前優先建立能驗證結果、搬移與假設的共同基礎；之後的吞吐、能耗與易用性主張，都以相應實驗證據驗收。
