> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

# v0.5：共用數值圖與兩個原生 CIM 時序後端

SRAM 算出 INT32 中間結果後，圖中的 cast 節點把它轉為 FP16，HBM 後端接收這批實際 bytes 再計算，GPU／NPU 模型接續 ReLU 與加法。這條流程已執行並逐節點驗證，不再只是在同一份報告裡並列彼此獨立的 trace。

交付日期：2026-10-15。此版本完成可執行的共同研究介面、兩個指令後端與異質模擬流程；它的硬體係數、完成延遲及能量假設都有來源標示，沒有宣稱已取得實機校準或社群認證。

| 原先的實作缺口 | v0.5 交付 | 驗證 |
|---|---|---|
| 只有 int64 功能圖，FP16 trace 未連接 | 共同 typed IR：INT8／INT32／FP16／FP32，matvec／matmul／ReLU／add／cast，明確累加、溢位、rounding 與 requantization | 25 項精度／圖契約測試；浮點逐步 rounding 與整數 overflow 反例 |
| 編譯出的配置沒有數值驗證 | SRAM 原生 ISA byte interpreter；HBM SIMD16 byte interpreter，讀取實際指令、padding、地址與記憶體內容 | 指令結果對照獨立 scalar reference；K=129／257、N=17、多列與 bank reuse 等邊界 |
| DCC controller 的 use-after-free | 可重放補丁、來源 hashes、Release 與 ASan/UBSan runtime；read/write/refresh/PIM 完成追蹤與 barrier | 40 次原生回歸，包含實際論文尺寸；write/PIM/refresh 延遲改動的因果測試 |
| 只有 HBM 外部後端 | CIMFlow SRAM 編譯與原生時序後端，含 CIM programming、K-tile reduction 與輸出記憶體 | 作者 fixture 通過；24 次 ISA 驗證、48 次直接／wrapper 原生執行一致 |
| SRAM／HBM／GPU／NPU 未在共同數值流程執行 | backend registry、能力檢查、明確 placement、真實中間資料傳遞、容量／存活期間與搬移記錄 | mixed、HBM+GPU、HBM+NPU 三種圖；相同平台配置下的 GPU／NPU baseline |
| 缺少完整操作與重現入口 | native-init/run/demo/verify/compare、HTML 報告、原始 log、完整 manifest、來源快照、reproduce.py | 六組實驗全部數值相符且快照重跑一致；ASan 與 Release 混合圖結果一致 |

## 可直接開啟的交付

- 六組實驗與 baseline 報告 (local artifact: `../runs/unified-v05/report.html`)
- SRAM→GPU cast→HBM→NPU→GPU 混合圖 (local artifact: `../runs/unified-v05/mixed/report.html`)
- HBM＋GPU (local artifact: `../runs/unified-v05/hbm-gpu/report.html`)、HBM＋NPU (local artifact: `../runs/unified-v05/hbm-npu/report.html`)
- 完整驗收 JSON (local artifact: `../runs/unified-v05/summary.json`)
- 原生 SRAM build 與版本 (local artifact: `../runs/cimflow-build-final/provenance.json`)
- 修復 HBM runtime descriptor (local artifact: `../runs/dcc-patched-final/runtime.json`)、原生回歸紀錄 (local artifact: `../runs/dcc-patched-final/regressions.json`)
- 全部測試紀錄 (local artifact: `../runs/validation-v05.txt`)

實驗包的 Python 程式快照會固定重跑邏輯；原生 binary、library 與 descriptor 另以 hash 綁定，仍需保留本機 runtime。套件位於 `dist/pim_foundry-0.5.0-py3-none-any.whl`，不把 C++ simulator 或第三方原始碼偷偷包進 wheel；完整 checkout 的 build scripts 會記錄相應依賴與版本。

## 同一把尺如何落到程式

工作負載身分包含完整圖、輸入資料、精度與每個節點的算術政策。平台身分包含裝置、容量、成本、連線，以及原生 binary／runtime descriptor 的 hash。每次實驗都保留輸入、權重、跨裝置搬移、padding／packing、計算、同步、輸出與所有所選裝置的閒置成本。

baseline 保留相同 platform，僅改變 node placement；因此仍計入其他裝置的閒置功率。精度、工作負載、平台成本或 protocol 不同，`native-compare` 會拒絕當成同模型比較。驗收另跑了頻寬加倍的敏感度案例，確定平台差異被拒絕，沒有包裝成同條件加速比。

同模型比較輸出的是指定模型假設下的比值。要把它解讀為真實硬體效能，仍需外部量測；既有 calibration 流程保留量測身分、raw logs、資料分離及驗證誤差。程式可以強制記錄這些條件，不能代替量測或替社群宣告標準已成立。

## 後端語意

**SRAM**：CIMFlow 原生時序程式不算權重數值；Foundry 對其 dense signed INT8 ISA 子集實作了 byte interpreter。它讀取實際 emit 的指令、input image 與 tile layout，和獨立矩陣公式比較。原生時序包含 CIM programming 與 K reduction，外部搬移和資料整理由共同 runtime 計入。

**HBM**：修復了原版執行期錯誤，但沒有假稱修復就證明原始 generator 的 FP16 layout。新增的是明確命名的 `foundry-hbm-simd16-fp16-fp32/v1`：FP16 inputs/output、逐步 FP32 multiply/add、16 個輸出 lanes，使用自己的可驗證配置與 byte interpreter。時序由修復 DCC request model 投影；每條依賴指令後都插入 channel barrier。來源會觸發 all-bank activation，不能解讀成 Samsung 實體 bank-selective ISA 的精確模擬。缺少硬體依據的 PIM 完成時間必須顯式指定，本例為 8 cycles。

**GPU／NPU**：功能數值遵循同一份 typed contract，時間／能量採設定檔內的分析係數。這是使用者可選的異質模擬組合；本機沒有執行 GPU／NPU 硬體 benchmark。

SRAM profile 與 HBM completion／energy 係數都是明示研究模型。新流程目前採 cold、單請求序列排程；舊分析流程仍提供多請求及資源競爭。數值語意、資料放置與模擬層級差異不會因共用入口而消失，報告會保留它們。

## 使用

```sh
python3 -m pim_foundry native-demo --case mixed --out runs/my-mixed
python3 -m pim_foundry native-verify runs/my-mixed
python3 runs/my-mixed/reproduce.py runs/my-mixed-replayed
```

自訂模型與後端請看 [快速開始](native-quickstart.md)、[數值契約](numeric-contract.md)、[backend plugin API](backend-plugin.md)。完整驗收可由 `python3 scripts/evaluate_unified.py --out runs/new-validation` 重跑，輸出目錄必須尚不存在。
