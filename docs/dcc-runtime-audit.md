> Paths under `runs/` or `external/` refer to locally generated or separately obtained artifacts, not bundled files. Start with [the public quickstart](getting-started.md); see [validation scope](../VALIDATION.md) for historical observations.

# DCC HBM-PIM 執行期稽核

本頁保留固定 upstream 原版的稽核。後續修復版與完整驗證見 [補丁紀錄](../patches/dcc/README.md)；v0.5 使用獨立命名的模型及輸出，未覆寫原始 artifact。

PIM 請求發出最後一條命令後，controller 就把它從佇列移除；模擬器只替一般讀取保留完成時間。因此，即使成功輸出 `memory_system_cycles`，也還不能把它當成 GEMV 結果可用的延遲。以下固定版本還有 iterator 失效問題，必須先處理，才適合用於效能排序。

查核日期：2026-10-15。本文只稽核來源、核對小型執行證據與選定工作負載；沒有修改 DCC 來源，也沒有重現整篇論文的圖表或加速比。

**固定版本與論文身分**

- 本地 checkout：`external/dcc-isca26`；revision [`7ffb80e4f33e357177dc8ab148953c0484e1e322`](https://github.com/SPIN-Research-Group/DCC/commit/7ffb80e4f33e357177dc8ab148953c0484e1e322)，commit 時間為 2026-08-03 06:57:49 UTC。下列來源連結全部固定至此 revision。
- [ISCA 2026 官方議程](https://www.iscaconf.org/isca2026/program/)在 Session 10C 列出 DCC，足以確認它是 ISCA 2026 論文；不能只依 arXiv 的首次上傳年份稱為「2025 年論文」。
- [arXiv 版本歷史](https://arxiv.org/abs/2511.15503)顯示 v1 為 2025-11-19，最新 v5 為 2026-08-28；論文 v5 比固定的程式 commit 晚，本文不假設兩者逐項一致。
- [論文 §5](https://arxiv.org/html/2511.15503v5#S5)描述修改 AttAcc/Ramulator 2.0、模擬 A100 與五個 PIM HBM 裝置的評估。這是作者的模擬設定；本次僅執行其中 HBM-PIM trace 路徑。

**請求實際如何執行、何時結束**

以下 `src/...` 路徑以 DCC repository 為起點；模擬器位於 `src/Ramulator2/HBMPIM/`。

| 查核項目 | 來源中的具體動作 | 可支持的解讀與限制 |
| --- | --- | --- |
| iterator 失效 | `src/Ramulator2/HBMPIM/src/dram_controller/impl/hbm3_pim_controller.cpp` 第 161、166 行刪除 `req_it` 指向的 list 節點，第 176 行卻讀取 `req_it->command`；這一行也在 `request_found` 保護之外。[來源][controller-issue] | **靜態與執行期均確認**：同一 demo 的 AddressSanitizer 執行回報第 176 行 `heap-use-after-free`，被釋放的位置為第 166 行，經 `request.h:94` 的 list erase。沒有可發出請求時也可能解參考無效 iterator，此另一條路徑僅靜態確認。相同 release binary 重跑相同 cycles 無法排除此問題。 |
| 終止條件 | `src/Ramulator2/HBMPIM/src/main.cpp` 第 108 行要求 frontend 完成且 memory system 無 pending；controller 第 202–204 行包含 active、priority、read、write、PIM buffers 與讀取完成佇列。[主迴圈][main-loop]、[佇列][controller-pending] | 確實會排空這些軟體佇列；不是只讀完 trace 就立刻結束。 |
| 完成時間 | controller 第 154–161、183–189 行只有 `Read` 被加入 `pending`，時間為 `m_clk + m_read_latency`；Write 與 PIM 在最後命令發出時即刪除。[來源][controller-issue] | 結束點涵蓋已追蹤的讀取完成，沒有追蹤最後一筆 PIM 運算／寫回的獨立完成尾端。稱為「此模型排空請求與讀取完成佇列所用 cycles」較準確。 |
| barrier | controller 第 56–62 行只檢查 active、PIM、read、write buffers；沒有檢查 `pending` 讀取完成佇列或 priority buffer。[來源][controller-barrier] | `PIM_BARRIER` 不能單憑名稱視為完整資料完成 fence；早先讀取仍可能尚未到 depart 時間。 |
| 指令 timing | `src/Ramulator2/HBMPIM/src/dram/impl/HBM3-PIM.cpp` 第 489–511 行替 LD/ADD/MUL/MAC 的 OP1/2/3 設定序列化間隔；WB_ACC、WB_RES、ACC_RESET、RELU 沒有出現在 timing constraint 清單。WB 等另有 row-open prerequisite。[timing][custom-timing]、[prerequisite][custom-prereq] | 這些操作仍會經過 controller 發射與部分 prerequisite，不能說完全免費；但來源沒有給它們明確的運算／寫回完成延遲，可能影響候選配置的相對 cycles。未量化影響大小。 |

**cycles 與計數的單位**

`src/Ramulator2/HBMPIM/src/memory_system/impl/PIM_DRAM_system.cpp` 第 56 行把 `m_clk` 登記為 `memory_system_cycles`；第 194–199 行每個 memory-system tick 加一，所有 channel controller 在同一 tick 更新，因此不是各 channel cycles 的總和。[來源][memory-clock]

`HBM3_5.2Gbps` 預設表最後一欄雖列出 1300，初始化在 `HBM3-PIM.cpp` 第 363–364 行以 `int(1E6 / (5200 / 4))` 覆寫 `tCK_ps`，實際為 **769 ps = 0.769 ns**。`src/predictor.py` 第 20–21 行也以 `cycles × 0.769 / 10^6` 換算毫秒。[覆寫位置][tck]、[上游換算][predictor-time] 這只確定此程式版本的單位；不解決上面的未定義行為或完成時間缺漏，也不證明硬體 GEMV 延遲。

計數在 `PIM_DRAM_system.cpp` 第 78–188 行於 request 被 controller 接受後增加；barrier 分支不計數，LD_OP1/2/3 與 MAC_OP1/2/3 各自合併。對目前 GEMV trace，各 request counter 的總和應等於「trace 行數減 barrier 行數」。它們不計算內部插入的 ACT/PRE 數，也不表示 scalar MAC 或 FLOPs。[來源][request-counts]

本地 release 執行的 demo log (local artifact: `../runs/paper-dcc-build/demo.stdout.log`)載入 6,704 行並輸出 660 cycles；request counts 合計 6,656，差額 48 對應 barrier。這是「流程能執行」的證據，不能因輸出數字或重跑一致就建立效能可信度。

獨立的 ASan/UBSan build 使用相同 133 個 tracked source files、`-fsanitize=address,undefined -fno-omit-frame-pointer`，沒有來源 patch。相同 demo 在 0.466 秒內以 exit 1 結束，沒有觸及 60 秒 timeout；stderr (local artifact: `../runs/paper-dcc-build/asan/demo.stderr.log`)保留失效讀取、釋放與配置的完整 stack。執行報告 (local artifact: `../runs/paper-dcc-build/asan/simulation-report.json`)保留命令、環境設定、trace/config/binary hashes；建置紀錄 (local artifact: `../runs/paper-dcc-build/asan/build-metadata.json`)保留來源 hash 與編譯參數。原始 release 執行檔與 library hashes 未變動。這確認缺陷會被本次輸入觸發，仍不能單憑 sanitizer 決定它使 cycles 偏高、偏低或偏移多少。

**沒有數值與 lane 正確性證據**

trace frontend 的一筆資料只有 opcode 與 address；parser 要求兩個欄位。`HBM3-PIM.cpp` 第 267–270 行發出指令時，只呼叫 timing/state 更新，沒有 FP16 operand、累加器數值或輸出 tensor 的算術執行。[frontend][trace-frontend]、[issue][dram-issue] 因此這條執行路徑不能驗證 GEMV 數值、FP16 誤差或 packing/layout 正確性。

地址仍影響 channel、bank、row 與 timing，但粒度需要區分：`hbm3_pim_linear_mappers.cpp` 第 37–38 行以 prefetch 8、channel width 32 bits 算出 32-byte transaction；第 64–73 行先右移 5 bits，再拆出 column/row/bank 等欄位。[mapper][address-map] generator 第 134 行的 OP1 load 位移是 `pos_i × 2` bytes，例如同一個 32-byte 區段內的 0、2、4，都映射到相同的 DRAM timing column。[generator][generator-op1] 來源沒有對這些低位元執行 FP16 lane 取值；無法從這個 timing 模型判定其 packed layout 是否實現正確計算。

**採用論文 artifact 的實際形狀，保留 baseline 邊界**

`src/test_configs.py` 第 33 行的 `gemv_single` 是以下笛卡兒積，共八種形狀：`batch ∈ {1,4}`、`heads ∈ {64,32}`、`K=128`、`N ∈ {512,4096}`。第 35–38 行另列 inference 形狀；先用下列較小的兩種即可檢查接入流程。[來源][paper-shapes]

| `(batch,heads,K,N)` | 上游原始代表配置數 | Foundry 現有 guards 接受數 | 接受配置的預測 trace 行數範圍 |
| --- | ---: | ---: | ---: |
| `(1,32,128,512)` | 1,081 | 4 | 22,576–31,792 |
| `(1,64,128,512)` | 1,373 | 4 | 32,816–61,488 |

這些數字由固定版本 `generate_search_space_3D` 在記憶體內枚舉，再套用 [Foundry guards](../pim_foundry/model.py#L32) 得到；是 trace 大小與範圍檢查，尚非 timing 結果。例 `(1,32,128,512)` 可接受 `[[1,32,1],[1,2,64],[16,1,32]]`；每一列依序是該軸的 `[groups,cores,local_extent]`。

上游 3D baseline 條件見 `src/predictor.py` 第 132 行；這兩個形狀對應 `[[16,1,heads/16],[1,4,32],[1,16,32]]`，local heads 為 2 或 4。[baseline][baseline] generator 的 writeback 沒有加入 head 位移，已有獨立稽核記錄，因此 Foundry 要求 local heads=1，會拒絕這兩個 baseline。[writeback][generator-writeback] 在此 guard 範圍內比較候選配置，不能稱為重現論文 baseline 的加速比。

此外，`src/run_kernels.py` 第 147 行在與 GPU 模型比較前把 batch 乘 5，對應五個 HBM cubes；第 179–185 行的 GPU time 來自 `get_time_and_energy`。單一 trace 的 cycles 不能直接對照完整五裝置工作負載，更不能當成實測 GPU 比較。[來源][gpu-comparison]

可採用的下一步是保留 pin、trace 與原始輸出，先解決並回歸驗證 iterator 問題，再以相同 trace 比較直接呼叫和 Foundry 呼叫是否一致。若進一步宣稱效能改善，還需要定義結果完成點、補足或證明 timing 假設、加入可信 baseline 與數值/layout 驗證。本次沒有替這些未完成條件背書。

[controller-issue]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/dram_controller/impl/hbm3_pim_controller.cpp#L138-L198
[controller-pending]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/dram_controller/impl/hbm3_pim_controller.cpp#L202-L204
[controller-barrier]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/dram_controller/impl/hbm3_pim_controller.cpp#L53-L62
[main-loop]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/main.cpp#L99-L109
[custom-timing]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/dram/impl/HBM3-PIM.cpp#L489-L511
[custom-prereq]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/dram/impl/HBM3-PIM.cpp#L634-L651
[memory-clock]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/memory_system/impl/PIM_DRAM_system.cpp#L194-L215
[tck]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/dram/impl/HBM3-PIM.cpp#L356-L364
[predictor-time]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/predictor.py#L20-L21
[request-counts]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/memory_system/impl/PIM_DRAM_system.cpp#L78-L188
[trace-frontend]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/frontend/impl/memory_trace/pim_loadstore_trace.cpp#L16-L19
[dram-issue]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/dram/impl/HBM3-PIM.cpp#L267-L270
[address-map]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/Ramulator2/HBMPIM/src/addr_mapper/impl/hbm3_pim_linear_mappers.cpp#L34-L73
[generator-op1]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py#L132-L144
[paper-shapes]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/test_configs.py#L33-L38
[baseline]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/predictor.py#L124-L141
[generator-writeback]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py#L147-L151
[gpu-comparison]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/run_kernels.py#L130-L185
