# DCC HBM GEMV 原始碼稽核

local K 從 128 增至 256 時，這份 generator 仍只送出前 8 組 OP1 load/MAC：下一個 chunk 的起點是 128，終點卻是 `ceil(256/16)=16`，迴圈因此為空。這種可重現的缺漏，使目前產出的 trace 無法作為數值正確性的證據。

本頁原始稽核對象固定為 DCC revision `7ffb80e4f33e357177dc8ab148953c0484e1e322` 的兩個 Python 檔。測試先呼叫 `pim_foundry.backend.verify_upstream()` 比對 revision 與兩檔 SHA-256，再匯入原始碼；該階段沒有執行 upstream 安裝腳本、完整編譯器、模擬器或硬體。2026-10-15 已另建置並實跑 CPU simulator，且以 AddressSanitizer 確認執行期錯誤，見 [runtime 稽核](dcc-runtime-audit.md) 與 [論文接入評估](paper-validation-dcc.md)；以下 generator 問題仍保留。

## 已確認的來源行為

| 問題 | 最小重現與直接原因 | 本地 adapter 的處理 |
|---|---|---|
| reduction chunk 遺漏 | 單 bank、單 output block，local K=128、129、256、2048 都只有 8 個 MAC。chunk 起點與終點使用不同單位。[來源][reduction] | local K 僅接受 16 至 128 的 16 倍數。 |
| local head 輸出重疊 | local heads=2、K=N=16，兩次 writeback 都寫 2048；return phase 卻取 2048 與 2080。writeback 地址未加入 head 位移。[來源][writeback] | local heads 必須為 1。 |
| alignment 超額 | 搜尋維度 `[9,400,4]` 得到 47 個代表配置，其中 15 個在各因子獨立向上取二次方後超過資源。例：bank 因子 `5×5×2` 變為 `8×8×2=128`，上限為 64。[來源][alignment] | 要求原始 group/core 因子為二次方、維度完全覆蓋，且總量不超過 16 groups / 每組 64 cores。 |
| 容量不足仍繼續 | 一個 local 4096×4096 matrix 加 vector/result 要 33,570,816 bytes，bank 只有 16,777,216 bytes。容量迴圈超過 head 數後沒有拒絕；以 mock 攔截後仍觀察到 4 次 GEMV 呼叫。[來源][capacity] | 超過單 bank 容量即拒絕，未支援 upstream 分段路徑。 |
| operand base 參數顛倒 | local 16×16 時，呼叫端宣告 matrix=0、vector=512，卻依 `GEMV(T, Vec_addr, Mat_addr, ...)` 傳入 0、512。[簽名][signature]、[呼叫][capacity] | 保留 upstream 原始行為，報告列為未解決；未宣稱數值正確。 |
| 外部 iteration 無效 | 同一 tiling 傳入 `itr=0` 或 `itr=17`，命令完全相同；參數被 local head 迴圈覆蓋。[來源][iteration] | 目前 guard 排除需要多段迭代的配置。 |

## 尚未驗證的硬體語意

K=128 時 OP1 load 地址為 `base+0,2,…,14`，都在同一個 32-byte 區間；是否代表重讀，取決於 backend 如何解碼低位元。matrix offset 的 `pos_i*K + pos_o*16` 也需要明確的 packed layout 才能判定。這兩項只記錄觀察，不據此斷言硬體結果錯誤。

guard 通過表示避開上述部分來源風險；地址範圍、命令數一致與語法檢查也都不構成硬體合法性或數值驗證。trace 沒有 tensor values 或運算結果，因此不提供誤差、延遲、能耗、加速比或最優排程結論。

## 重現

在專案根目錄執行：

```sh
python3 -m unittest discover -s tests -p 'test_upstream_audit.py' -v
```

六個 `test_recorded_*_defect_*` 測試通過，表示**固定版本的已知問題仍可重現**。容量與參數順序測試以 mock 觀察控制流程，不生成大型 trace。未來若修正 upstream，必須一併更新來源 pin、hash 與這份稽核，不能把這些測試當作 GEMV 正確性測試。

[reduction]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py#L127-L144
[writeback]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py#L147-L151
[alignment]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/generate_configs.py#L147-L163
[capacity]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py#L227-L259
[signature]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py#L155
[iteration]: https://github.com/SPIN-Research-Group/DCC/blob/7ffb80e4f33e357177dc8ab148953c0484e1e322/src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py#L198-L202
