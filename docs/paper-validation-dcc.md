> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

# 用 ISCA 2026 DCC 實測 PIM Foundry

版本說明：本頁保留對**未修復 DCC artifact** 的第一次評估。後續 v0.5 已增加修復 runtime、typed IR、SRAM／HBM 指令數值解譯與共同圖執行，見 [v0.5 驗收](release-v05.md)。後續修復不能追溯改寫本頁原始實驗的資格，也不等同重新證明論文圖表。

把作者測試集的一個 GEMV 交給 Foundry，現在能列舉配置、擋下已知風險、產生原版 trace，實際執行作者的 HBM-PIM simulator，再留下可核對的輸入與輸出。**這次證明它可以作為有限範圍的實驗工作台；還不足以充當通用 CIM 編譯器或可信的效能評測標準。**

評估日期：2026-10-15。採用 [DCC：ISCA 2026 官方議程 Session 10C](https://www.iscaconf.org/isca2026/program/)；[公開論文 v5](https://arxiv.org/html/2511.15503v5) 更新於 2026-08-28，[artifact](https://github.com/SPIN-Research-Group/DCC) 固定至 `7ffb80e4f33e357177dc8ab148953c0484e1e322`。這是近期正式發表且可取得原始碼的 HBM-PIM 論文；本次查核不是全領域文獻普查，也不主張它是所有 CIM 分支中最新的一篇。

## 實際測到什麼

| 驗收項目 | 結果 | 證據及範圍 |
|---|---|---|
| 原始 simulator 接入 | 通過 | CPU Release build 成功，使用作者 C++ Ramulator；沒有把 Foundry 人工成本模型冒充論文模型。 |
| 原始 generator 接入一致性 | 通過 | 27 個配置分別從完整 checkout 和 Foundry snapshot 產生 trace，逐 byte 相同。 |
| 包裝前後行為一致性 | 通過 | 每配置一次直接呼叫、兩次 adapter 呼叫，共 81 次；所有 stdout、stderr、18 個統計欄位完全相同。 |
| 執行紀錄完整性 | 通過 | 保留原始 trace、YAML、版本、binary/library hashes、原始 log、拒絕原因與 Python 來源快照；686 個檔案雜湊驗證通過。 |
| 失敗紀錄與資格限制 | 通過 | 實際 ASan 版本經 adapter 執行回報失敗並保存堆疊；成功執行也不會取得效能比較資格。 |
| 論文工作負載覆蓋 | 有限 | 12 種形狀中只有 8 種找到通過保守檢查的配置；26,324 個候選只接受 27 個。拒絕數包含本工具限制，不能解讀成其餘全部是硬體錯誤。 |
| 原論文 baseline | 未通過本工具檢查 | 本次 8 種 `gemv_single` 的 baseline 都含多個 local heads；已知 writeback 地址別名問題尚未修正。沒有重現原 baseline 加速比。 |
| simulator 執行期正確性 | 發現錯誤 | 同一來源及輸入的 ASan build 確認 controller 的 heap-use-after-free。 |
| GEMV 數值正確性 | 未驗證 | trace 只有 opcode/address，native simulator 沒有執行 FP16 tensor 算術。 |
| SRAM／HBM／GPU／NPU 公平比較 | 未完成 | 本次只有 HBM-PIM 子路徑；沒有通用圖至 DCC FP16/layout 的 lowering、SRAM backend 或實體 GPU baseline。 |

81 次執行是固定的接入一致性檢查，並非 81 個獨立效能樣本；相同 deterministic simulator 的重跑不建立統計信賴區間，也不證明模型正確。全部結果的 `performance_comparison_eligible=false`、`numerical_correctness_verified=false`、`hardware_execution=false`。

## 測試設計與覆蓋

前八個形狀是固定版本 `src/test_configs.py` 的全部 `gemv_single` 集合。後四個是預先選定的 `gemv_inference` 探針，涵蓋非二次方 head 數及較長 reduction；沒有依 cycles 好壞挑選配置。每個形狀都執行所有通過現有 guards 的候選。

每個候選經以下流程：

1. 使用固定 generator 列舉候選，保存所有拒絕原因。
2. Foundry 產生 trace；另外從完整原始 checkout 獨立呼叫相同 upstream generator，核對 trace hash。
3. 對兩條路徑使用相同 HBM 設定；只調整相對 trace 路徑、移除共用 recorder，保持私有輸出目錄。
4. 原始執行檔直接跑一次，再經 adapter 跑兩次。每次上限 60 秒，逐項比對 request counters 和完整原始 log。
5. 使用另外建置的 ASan/UBSan 版本執行既有 smoke trace，確認執行期錯誤；不修補或覆寫原始程式。

維度定義為 `vector[B,H,K] × matrix[B,H,K,N] → result[B,H,N]`；這是預期邏輯運算，尚未由 native trace 的數值結果證實。

| B | H | K | N | 原始候選 | 通過 guards／實際測試 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 128 | 512 | 1,373 | 4 |
| 1 | 64 | 128 | 4,096 | 2,552 | 2 |
| 1 | 32 | 128 | 512 | 1,081 | 4 |
| 1 | 32 | 128 | 4,096 | 2,184 | 4 |
| 4 | 64 | 128 | 512 | 2,205 | 3 |
| 4 | 64 | 128 | 4,096 | 3,509 | 0 |
| 4 | 32 | 128 | 512 | 1,789 | 4 |
| 4 | 32 | 128 | 4,096 | 3,054 | 0 |
| 1 | 8 | 128 | 5,120 | 1,391 | 5 |
| 1 | 11 | 128 | 6,656 | 1,710 | 0 |
| 1 | 42 | 128 | 6,656 | 2,433 | 0 |
| 1 | 8 | 5,120 | 384 | 3,043 | 1 |
| **合計** | | | | **26,324** | **27** |

guards 包含 local heads=1、local K 的限制、完整覆蓋、group/core 二次方因子及資源上限，也包含本工具每 trace 最多 250,000 commands 的資源預算。覆蓋不足是本框架目前的實際限制；不能因成功的子集重跑一致，就宣稱支援整個論文 workload。

完整逐候選結果：summary.json (local artifact: `../runs/dcc-isca26-study/summary.json`)。例如 `(1,32,128,512)` 四個候選輸出 4,342、2,854、2,586、3,065 memory-system cycles；它們是帶有下述正確性問題的原始觀察，**沒有據此選出最快配置或計算加速比**。

## 為何不能採用這些 cycles 做排名

controller 在第 166 行移除 list 裡的請求後，第 176 行仍讀取該請求的 command。相同 6,704-command smoke trace 在 Release build 兩次回報 660 cycles；獨立 ASan/UBSan build 則以 exit 1 停在 `heap-use-after-free`。來源副本的 133 個 tracked files 雜湊全部與原版相同，沒有修補來源。這是本次版本、工具鏈與輸入下的直接執行證據；它沒有量化 cycles 偏差，也不能據此推論整篇論文所有結果都錯誤。

- 原始 sanitizer 完整堆疊 (local artifact: `../runs/paper-dcc-build/asan/demo.stderr.log`)
- sanitizer 版本、命令、hash 與結果 (local artifact: `../runs/paper-dcc-build/asan/simulation-report.json`)
- 經 Foundry 執行 sanitizer 的失敗報告 (local artifact: `../runs/paper-dcc-adapter-asan/report.json`)
- [完整 runtime 原始碼稽核](dcc-runtime-audit.md)

除此之外，原始模型只替一般 read 保留完成佇列；最後的 write/PIM 命令發出後即移除請求。barrier 也沒有等待 pending read completions，部分 PIM 操作沒有明確的 timing constraints。這些機制限制了 cycles 的意義，需要分別確認完成點與時序假設。

原有 generator 的 operand base、lane 與 multihead 地址問題也未因此消失。Foundry 的 int64 scalar oracle 驗證的是自身功能引擎，不能替另外一條 FP16 trace 路徑提供數值背書。

## 對這套工具的判斷

**已有用的部分：** 一個尺寸可以批次展開配置，每個結果都能追到輸入、來源及執行檔；程序失敗會保留證據，已知不可信結果不會自動成為效能排名。本次確實用這條流程接上近期正式論文 artifact，並核對了接入前後的一致性。

**目前不好用的部分：** 接入仍需人工閱讀來源、建置 C++ runtime、確認統計語意。這次新增 253 行 native adapter、169 行 study runner、CLI 入口及 331 行測試；原有核心不需重構，但這也顯示它還不是填一份 backend 描述就能支援新論文。這些行數僅描述實作成本，沒有使用者研究足以宣稱學習成本低。

**最大缺口：** 共用圖目前沒有浮點／量化政策與實際 bank layout lowering，DCC adapter 尚是獨立的外部實驗路徑。另一方面，實驗身分與紀錄規則可以重用，但只有單一 native backend 的子集測試，尚不能證明跨 SRAM/HBM/GPU/NPU 的公平性。

下一個驗收門檻應是修正並回歸檢查 native runtime、定義完整結果完成時間、為 DCC 指令建立可與 FP16 reference 對照的功能模型，再恢復原 baseline 與未覆蓋形狀。完成後才適合接第二個 SRAM backend，檢查共同介面究竟保留了多少硬體差異。這次沒有把這些後續工作算成已完成。

## 如何重跑與查證

建置使用 GCC 13.3.0、CMake 3.28.3、Ninja 1.11.1，最多 4 個編譯工作；Release 約 45 秒，另建 ASan/UBSan 約 130 秒。僅建 CPU simulator 子集，未執行 upstream 安裝腳本；與作者完整 artifact 環境不同。全部依賴 commits、命令與建置紀錄 (local artifact: `../runs/paper-dcc-build/README.md`)

在專案根目錄使用尚不存在的輸出路徑：

```sh
python3 scripts/evaluate_dcc.py \
  --checkout external/dcc-isca26 \
  --executable runs/paper-dcc-build/build/ramulator2 \
  --library external/dcc-isca26/src/Ramulator2/HBMPIM/libramulator.so \
  --out runs/dcc-study-repeat
```

固定此次 Python 程式快照、重用保存的 native binary/library：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 runs/dcc-isca26-study/engine/scripts/evaluate_dcc.py \
  --checkout external/dcc-isca26 \
  --executable runs/dcc-isca26-study/runtime/ramulator2 \
  --library runs/dcc-isca26-study/runtime/libramulator.so \
  --out runs/dcc-frozen-repeat
```

快照仍需要本機 Python、Linux shared libraries 與上述固定 upstream checkout；不是包含作業系統的獨立容器。`manifest.json` 驗證本地完整性，不代表第三方認證。

新增的 17 項 native adapter 測試與既有回歸合計 **236 項通過**。合成 subprocess 僅測介面成功／失敗／逾時、輸入竄改及檔案保存；真實 native 執行證據來自上述 study 和 ASan 報告。測試紀錄 (local artifact: `../runs/dcc-adapter-test-results.json`)、686 檔案完整性核對 (local artifact: `../runs/dcc-study-integrity-check.json`)
