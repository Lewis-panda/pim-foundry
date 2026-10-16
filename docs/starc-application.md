> Paths under `runs/` or `external/` refer to locally generated or separately obtained artifacts, not bundled files. Start with [the public quickstart](getting-started.md); see [validation scope](../VALIDATION.md) for historical observations.

# 用 Application 入口執行 STARC

Full attention 和 STARC 在同一個模型、batch 與 context 上選不同的 KV 存取方式；
若前一次的 simulator cache 留在目錄中，下一次可能直接讀到前一個方法的結果。
這個 adapter 每次建立新的工作目錄與空白 cache，將 method、KV budget、來源與
runtime 綁定在 experiment，讓同一個 application 可以比較兩種實作。

論文是 [STARC，ASPLOS 2026](https://doi.org/10.1145/3779212.3790226)；
[作者 artifact](https://github.com/EPIC-RPI/STARC) 固定在
`63c8aaa7a02114cd4cf21f32e7621fab5b416957`。這個入口執行作者的 CPU system
simulator：GPU 使用作者分析模型，HBM-PIM 真正呼叫其 Ramulator 模擬器。
它的模式是 `original_artifact_system_model`，不代表 Transformer 已 lowering
到 Foundry 的共用 tensor backend。

已執行的 v0.6 接入報告 (local artifact: `../runs/starc-application-v06/report.html`) 使用 prompt 2048、
generated tokens 128、batch 16。Full／STARC 各自完成 127 個 native calls，再各
重跑一次，共 **508 個 calls**；每個 call 的 trace hash、退出狀態和全部原生命令
統計皆與當次 cache 核對。兩種方法的 CSV 與 cache 均逐 byte 等於獨立執行作者
程式的結果。Full 與 STARC 的平均 decode batch-token 模型時間分別為
**0.9405744286683656 ms** 和 **0.9385560226692454 ms**，條件式時間比約
**1.00215×**。此短視窗的差異很小，不能據此推論論文長 context 的效果。

## 一次完成實驗、原始結果核對與重跑

在已建置 CPU runtime 的專案內執行：

```sh
python3 scripts/evaluate_starc_application.py \
  --runtime runs/starc-build/runtime-safe.json \
  --out runs/starc-application-new \
  --prompt-tokens 2048 --generated-tokens 128 --batch-size 16 \
  --direct-full runs/starc-safe-full-128 \
  --direct-starc runs/starc-safe-starc-128
```

最後兩個參數指定已獨立直接執行作者程式的目錄；腳本逐 byte 核對 `output.csv`
與 `ramulator.out`。若沒有直接執行的結果，可省略這兩個參數；報告便明示
`direct_parity.tested=false`。每個方法另使用封存的 Python engine 重跑，不能把
省略的直接核對標成通過。

輸出的 `inputs/application.json` 只包含內容綁定的模型定義與請求情境。
`inputs/full.json`／`starc.json` 各自指定平台、runtime、method 和 budget。
也能單獨使用共用入口：

```sh
python3 -m pim_foundry app-inspect \
  --application runs/starc-application-new/inputs/application.json \
  --experiment runs/starc-application-new/inputs/starc.json
python3 -m pim_foundry app-run \
  --application runs/starc-application-new/inputs/application.json \
  --experiment runs/starc-application-new/inputs/starc.json --out runs/starc-custom
```

`app-inspect` 不執行模擬，會檢查型別、model asset、runtime/source hashes、平台、
mapping 與 budget 完整性。若 budget 不足，adapter 拒絕作者程式原本會執行的
最後一筆重複延伸，避免把不完整輸入隱藏起來。

## 模型與 metric 範圍

目前 adapter 支援作者 `Mistral-7B` 定義：32 layers、hidden size 4096、32 heads、
head dimension 128。manifest 的 `model_definition` 必須是內容相符的原始
`src/config.py`，並透過 AST 讀取表格，不匯入或執行 asset 中的 Python 程式。
實際執行前再核對 runtime 使用同一份定義。

作者設定表宣告 `gqa_size=4`，但 `Transformer` 未使用它，CSV 將 GQA 欄位寫為
0。因此本次保存並驗證的是作者公開的 shape model，不能稱為忠實的 Mistral
GQA 數值推論。完整 source audit 見 [STARC 來源審查](starc-source-notes.md)。

| 指標 | 原始 CSV | 單位與界線 |
|---|---|---|
| `prefill_latency` | `s_time` | ms；整個 fixed batch 的 prefill 模型時間 |
| `decode_latency` | `g_time (ms)` | ms；`generated_tokens - 1` 個 decode steps 的平均，每一步處理整個 batch |
| `decode_energy` | `g_energy (nJ)` | nJ；同樣的平均 decode batch-token 範圍，來自作者能量表與命令流量模型 |

generation 指標不包含 prefill。未除以 batch size，也不把模擬器主機耗時轉成
裝置延遲。`host_wall_seconds_not_device_latency` 僅保存在 process metadata。

## 確實保存哪些證據

每個 application bundle 的 `execution/artifact/` 包含原始 Python sources、
當次 executable/shared library、budget、`output.csv`、`ramulator.out`、完整
stdout/stderr。`ramulator2/calls/` 另保存每次 native call 的 YAML、gzip trace、
trace hash、exit code 和原始 simulator stdout/stderr，先保存再交回作者 wrapper
繼續執行，因此 upstream 的 trace 刪除不會讓證據消失。

Runtime descriptor 綁定原始 Git revision、來源檔案、binary/library hashes、
controller safety patch、Python interpreter 及 package 版本。執行前後核對
安裝套件內容 digest；清除 `PYTHONPATH`、`PYTHONHOME`、`LD_PRELOAD` 和
`LD_AUDIT`，使用私有目錄內的 simulator library。bundle 不依賴 symlink。
較大的 timeout 由 experiment 明示，最多 86400 秒，預設 120 秒；timeout 會
終止整個 child process group 並保留已有紀錄。

`runtime-safe.json` 的 patch 修復實際由 ASan 重現的 controller lifetime 錯誤；
它保留作者的 timing/completion 語意。原始版與安全版的 128-token Full/STARC
CSV/cache parity，以及有限個 trace 的 ASan/UBSan 結果，由建置報告分開記錄。
Sanitizer 通過的範圍不能延伸為硬體時序已校準。

## 可比較和不可宣稱的結果

相同 semantic task、scenario、平台與成本契約下，Full/STARC 的條件式模型時間
比可以計算；method/budget 不同會改 execution identity，不會改共同 task identity。
`numerical.status` 與 `task.status` 保持 `not_evaluated`，因為此 CPU 路徑沒有
執行 pretrained weights、LongBench、RULER 或 PG-19。任務品質保持與硬體加速
的 eligibility 因而都不會自動通過。

128-token window 用來檢查接入、cache 隔離、完整 provenance、直接原始結果
一致性與 replay；它不等於完成論文 README E4 的 32000-token 工作量。
研究腳本另外輸出 `application-full-paper-profile.json`，記錄 E4 的
2048 prompt tokens、32000 generated tokens、batch 16，並標明是否真的執行。
要使用完整情境，不必修改 adapter：

```sh
python3 scripts/evaluate_starc_application.py \
  --runtime runs/starc-build/runtime-safe.json --out runs/starc-full-profile \
  --prompt-tokens 2048 --generated-tokens 32000 --batch-size 16 --timeout 86400
```

這個完整命令包含 Full、STARC 及各自 frozen replay，會比短視窗耗費更多時間與
磁碟；是否完成以實際輸出與 summary 為準。
