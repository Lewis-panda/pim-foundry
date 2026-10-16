# IANUS 論文證據審查

IANUS 把 NPU 讀取權重與 AiM 執行矩陣向量乘法放在同一組記憶體；這兩種動作會爭用記憶體，所以驗證必須同時追蹤排程、搬移與記憶體容量，才能判斷加速來自哪一項改動。

本審查使用作者公開的 [arXiv:2410.15008v1](https://arxiv.org/abs/2410.15008v1)，2024-10-19 上傳，首頁明確標示為 ASPLOS 2024 論文的 **updated version**。[會議 DOI](https://doi.org/10.1145/3620666.3651324) 為 10.1145/3620666.3651324。以下頁碼均為這份 16 頁 PDF 的 **1 起算實體頁碼**；沒有假定它與會議出版檔逐字相同。

保存的原文、文字擷取與結構化審查分別在 `runs/ianus-validation/ianus-paper.pdf`、`ianus-paper.txt`、`paper-audit.json`。數字來自作者論文；本次完成的是原文交叉核對及算術檢查，沒有執行作者的 IANUS 模擬器，也沒有量測 IANUS 實機。

## 可確認的實驗條件

| 項目 | 作者條件 | 原文位置 |
|---|---|---|
| NPU | 4 cores、700 MHz；每 core 128×64 PE、每 PE 4 MAC，約 46 TFLOPS/core、184 TFLOPS total | p.8，Tables 1–2 |
| NPU scratchpad | 每 core activation 12 MB、weight 4 MB；全系統 48 MB + 16 MB | p.8，Tables 1–2 |
| PIM | GDDR6-AiM；4 chips、8 channels、每 channel 16 banks；每 bank 1 PU、32 GFLOPS；1 GHz | pp.8–9，Table 1、§6.1 |
| 記憶體 | 8 GB；外部 256 GB/s；PIM 內部合計 4096 GB/s；row/global buffer 為 2 KB | p.8，Tables 1–2 |
| GPU | A100-SXM 80 GB；表列 1155 MHz、255 TFLOPS、2039 GB/s；PyTorch 2.0、CUDA 11.8、Hugging Face/Megatron-LM、CUDA Event 計時 | pp.8–9，Table 2、§6.1 |
| DFX | 4 FPGA；200 MHz、1.64 TFLOPS、32 GB HBM2、1840 GB/s | pp.8–9，Table 2、§6.1 |
| 精度與負載 | BF16；batch 1；主 GPT 圖使用 input 128/256/512、output 1/8/64/512；DFX 比較另使用 input 32/64/128、output 1/16/256 | pp.9–10，§6.1、Figs.8–9 |
| 模型 | GPT-2 M/L/XL/2.5B；其中 XL 的 head 數由 25 改為 24，Table 3 給出 d=1536、head dim=64、48 blocks | p.9，Table 3、§6.1 |
| 較大模型 | 6.7B/13B/30B 分別使用 2/4/8 台 IANUS，與單台 A100 比較；假定 PCIe 5.0 ×16 互連 | pp.12–13，Table 4、Figs.17–18、§7.1 |

主 headline 的 6.2× 對應 Fig.8 的 GPT-2 XL 平均延遲欄位；同一圖其餘模型為 11.3×、7.6×、4.3×。報告應保留模型與負載範圍，不把 6.2× 當成任何 GPT 或任何批次大小的固定收益。

## 算得回來的數字

基礎資源表在常見 FLOP 計數下相容：`128×64×4 MAC×2 FLOP×0.7 GHz×4 cores = 183.5008 TFLOPS`，四捨五入為 184。外部頻寬 `16 Gb/s ×16 pins×8 channels÷8 = 256 GB/s`。PIM 峰值 `8×16×32 GFLOPS = 4.096 TFLOPS`。§5.3 的 head dimension 64 在 1024-element row 上只使用 `64/1024=6.25%`，比例正確。這些是規格算術一致性，不能證明實際時序正確。

Fig.9 的九組 **顯示值** 可以重算作者的 DFX 比較：

| input, output | DFX ms | NPU-MEM ms | IANUS ms |
|---|---:|---:|---:|
| 32,1 | 227 | 18 | 18 |
| 32,16 | 330 | 247 | 73 |
| 32,256 | 1981 | 3970 | 989 |
| 64,1 | 447 | 18 | 18 |
| 64,16 | 550 | 246 | 72 |
| 64,256 | 2201 | 3972 | 990 |
| 128,1 | 887 | 18 | 18 |
| 128,16 | 991 | 249 | 73 |
| 128,256 | 2642 | 3983 | 997 |

`sum(DFX)/sum(IANUS)=3.1576`，可四捨五入為 3.2×；`sum(NPU-MEM)/sum(DFX)=1.24035`，與 24% slowdown 相容。`887/18=49.2778`，與 (128,1) 的 49.3× 相容。**這裡的 3.2× 符合延遲總和的比值**；相同顯示值的逐 case speedup 算術平均約 13.26、幾何平均約 7.61。平均方式必須成為結果格式的顯式欄位。

Fig.8 顯示的 Avg 延遲分別為 M 2024/179 ms、L 2950/388 ms、XL 3962/637 ms、2.5B 4418/1020 ms（GPU/IANUS）；比值與 11.3/7.6/6.2/4.3× 相容。這些顯示值已四捨五入，不能拿它們聲稱重建了未公開原始資料。

## 需要釐清的具體問題

1. **Fig.14 的 utilization 與 Table 2 的峰值分母不相容。** p.11 的 BERT-B throughput ratio 為 3.1×，utilization ratio 為 5.2×。若 utilization 是相同 FLOP 定義下的實得 throughput/peak，且兩圖採相同負載與平均方式，Table 2 的 255/184 TFLOPS 要求 utilization ratio 為 `3.1×255/184≈4.30`。即使把顯示值視為一位小數的四捨五入區間，仍不相交。L、1.3B、3.9B 也呈相同方向差異。可能原因包括使用不同 GPU peak、不同 FLOP 範圍或不同統計方法；公開文字沒有交代到足以決定是哪一種。這是可定位的**分母／定義缺口**，不能由此斷言延遲造假。
2. **Fig.10 的 FFN 權重敘述有 factor-of-two 疑點。** p.10 把 QKV 與 attention output 兩組 FC 的延遲相加：568+322=890 ms，127+88=215 ms，這部分正確。在標準 expansion-4 GPT 結構假設下，前者權重為 `3d²+d²=4d²`；兩層 FFN 為 `4d²+4d²=8d²`，比例應為 2，而同段文字稱為 4。同篇 p.3 Fig.2(a) 也顯示 FFN/attention FC FLOP 份額為 63.11%/31.51%，約為 2。這支持檢查敘述或權重計數定義，尚不能推斷模擬器內實作錯誤。
3. **Fig.13 同容量比較同時改變 PIM 運算資源。** p.11 的 unified 是 8 GB PIM；partitioned 是 4 GB PIM + 4 GB 普通 DRAM。作者明說前者 PIM throughput 加倍。圖中的收益同時包含容量使用方式、PIM 數量、參數是否可完整複製、排程，不能全部歸因於 unified memory 或 PAS 本身。應另做固定 PIM PU/內部頻寬的比較，並分開報告固定容量與固定成本的問題。
4. **BF16/batch 1 的結果不足以推廣到現代服務負載。** 小 batch 的 matrix-vector 對 PIM 有利；更大的 batch、continuous batching、長 context、GQA/MQA 和不同精度會改變資料重用與資源限制。作者選 batch 1 有明示理由，但沒有提供這些情境的結果。GPU 端未給出可鎖定的程式 revision、實際 fusion/attention kernel、warm-up/repetition/時脈控制與誤差分布，無法只憑框架名稱確認 baseline 最佳化程度。
5. **更大模型的比較加入更多裝置。** p.12 Fig.17 的 2/4/8 台 IANUS 同時增加記憶體、NPU 與 PIM，對照單張 A100。這是作者定義的容量配置比較；不能直接讀成同裝置數或同功耗預算的優劣。p.13 的性能/TDP 使用每台 IANUS 120 W 的估計及 A100 400 W，沒有實測整機功率；它也不等於已量測的 TCO。

## 模擬、原型、品質與能源的證據邊界

§6.1（pp.8–9）稱 NPU/PIM 模擬器分別對其實機有 5% 內誤差。本篇未列 calibration workload、逐項誤差、樣本數、平均／最壞值定義、原始測量或整合後 PCU/NoC/共享記憶體的端到端 calibration 表。應記為 **作者報告的元件校準界線**，不能直接用作所有 IANUS 預測的 ±5% 信賴區間。

§6.3（p.12，Fig.16）具體說明原型是 **NPU 軟體模擬器 + FPGA 上 PCU/記憶體控制器 + 實際 AiM**；NPU RTL 太大，沒有放進 FPGA。它提供功能整合證據，沒有把前文模擬的 6.2×/3.2× 轉成完整硬體實測。作者報告 WikiText-2 perplexity：117M/M/L/XL 為 30.92/22.60/19.39/17.48；但沒有在此列出可逐項配對的 full-precision 數字、checkpoint、tokenizer、資料 split、context/stride 與 scoring 腳本。此審查記為 **reported quality**，沒有當作本框架已重現的 quality pass。

§6.1/§6.2（pp.9–10）能源包含 NPU core、PIM 與普通 DRAM operation 的 dynamic energy；PIM compute power 採普通 DRAM read 的 3 倍假設，static energy 明確排除。Fig.11 的收益因此不是 wall-plug/J-per-request 實測。作者推測加入 static energy 可能更有利；方向須取決於兩系統 static power 與執行時間，現有數據不足以保證改善。正確追加式應是各自 `E_total=E_dynamic+P_static×latency`，再比較。

## 後續接入框架的驗收條件

原始程式可取得時，需要先鎖定 compiler/simulator/driver revision、模型與輸入資產，再保留逐層指令、NPU/PIM memory conflict 與 DMA trace；用相同輸入做直接執行、框架執行及重跑。品質軌須保存真正 logits 或 perplexity 計算，而不是從 latency simulator 推論正確性。性能軌至少應補固定資源 ablation、GPU baseline 的可重現設定、mean/tail latency 與重複量測，以及 component calibration 與整合 calibration 的分開紀錄。

在原始工具或硬體尚未可用時，可以執行算術一致性與條件化獨立模型，但結果必須標示為 paper audit / independent reconstruction。**本審查沒有提出造假結論；已確認的差異、作者假設與未提供資料各自保留。**
