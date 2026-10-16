# 共用 SRAM／HBM 圖執行：從一次完整資料流開始

`mixed` 範例先把 `2×7` INT8 輸入送到 SRAM，與 `7×19` 權重相乘，
再把產生的 INT32 中間值明確轉成 FP16，送到 HBM 與 `19×11` 權重相乘。
後面的 NPU ReLU 與 GPU residual add 讀取前一個後端真正回傳的值。
每次搬移都序列化／還原 tensor bytes，並記錄 payload hash、時間與能量；
每個節點的結果另與獨立 scalar reference 核對。

公開 checkout 不附原生模擬器、上游原始碼或歷史 runtime。請先完成下方[建置前置條件](#建置原生-runtimes)，並確認範例所指向的兩個 runtime 路徑存在，才執行以下指令。若要先跑不依賴外部模擬器的完整例子，使用[一般 quickstart](getting-started.md)。

```sh
python3 -m pim_foundry native-demo --case mixed --out runs/my-mixed
```

開啟 `runs/my-mixed/report.html` 查看節點、數值檢查、搬移和成本；
`result.json` 保留完整機器可讀結果。輸出目錄必須尚未存在。

## 改選 HBM-PIM + GPU／NPU

```sh
python3 -m pim_foundry native-demo --case hbm-gpu --out runs/my-hbm-gpu
python3 -m pim_foundry native-demo --case hbm-npu --out runs/my-hbm-npu
```

這兩個範例以 `2×17` FP16 輸入乘上 `17×19` 權重，再執行 ReLU。
HBM 節點有獨立的 SIMD16 指令執行器與原生 DCC timing projection；GPU／NPU
節點目前明確指定 `analytical` 後端，使用 scalar functional execution 與提供的
成本係數。選到無法執行的原生後端會失敗，不會自動換成 analytical 後端。

要同時啟用修補後 HBM runtime 的 AddressSanitizer／UBSan：

```sh
python3 -m pim_foundry native-demo --case mixed --asan --out runs/my-mixed-asan
```

## 編輯自己的 typed graph 與配置

先匯出一份完整、可直接修改的輸入：

```sh
python3 -m pim_foundry native-init --case mixed --out runs/my-inputs
```

四份 JSON 各自負責一件事：

| 檔案 | 要修改的內容 |
|---|---|
| `graph.json` | tensor 名稱、真實 shape、dtype、常數權重、節點與逐節點算術政策 |
| `inputs.json` | graph input 的 row-major 扁平數值陣列 |
| `platform.json` | 後端、容量、原生 runtime 路徑、packing、fabric、idle 與成本參數 |
| `placement.json` | 每個節點要在哪一個 device 執行 |

例如匯出的 `placement.json` 為：

```json
{
  "sram_linear": "sram",
  "convert": "gpu",
  "hbm_linear": "hbm",
  "activation": "npu",
  "residual": "gpu"
}
```

將 `activation` 改成 `gpu`，就會在同一份平台上測試另一種配置；保留其他 JSON
不變，後續比較才是在同一 workload、精度與平台假設下比較 placement。
節點可以移動到接受其 shape、dtype 與運算的後端；驗證器會拒絕不支援的指定。

執行修改後的四份 JSON：

```sh
python3 -m pim_foundry native-run \
  --graph runs/my-inputs/graph.json \
  --inputs runs/my-inputs/inputs.json \
  --config runs/my-inputs/platform.json \
  --placement runs/my-inputs/placement.json \
  --out runs/my-custom-run
```

`graph.json` 的 schema 是 `pim-foundry.typed-graph.v1`，layout 是 `row_major`。
SRAM linear 節點接受 INT8 輸入／權重，使用 INT32 累加與輸出；HBM linear 節點
接受 FP16 輸入／權重，以 K 遞增順序執行 FP32 乘法捨入、FP32 加法捨入，最後
存成 FP16。兩者之間的 `cast` 必須出現在圖裡。轉回 INT8 時還需明確的
requantization 參數；詳見 [數值契約](numeric-contract.md)。

## 驗證、重跑與比較

```sh
python3 -m pim_foundry native-verify runs/my-mixed
python3 runs/my-mixed/reproduce.py runs/my-mixed-replayed
python3 -m pim_foundry native-compare \
  runs/my-mixed/result.json runs/my-mixed-replayed/result.json --metric latency
python3 -m pim_foundry native-compare \
  runs/my-mixed/result.json runs/my-mixed-replayed/result.json --metric energy
```

`native-verify` 檢查 bundle 的完整檔案涵蓋與 hashes。`reproduce.py NEWOUT` 使用
bundle 中凍結的 Python engine，再次執行外部原生模擬器，核對數值輸出、時間、
能量、容量與搬移量。外部 binary 沒有包進 bundle，必須仍在記錄的路徑，且符合
保存的 runtime identity；搬到另一台機器時需另建可稽核的新實驗。

`native-compare` 的兩個位置參數是 **`result.json` 檔案**。它要求相同 workload、
平台／runtime identity、數值契約與執行 protocol，並要求兩邊數值驗證通過。
`--metric energy` 另外要求成本完整；未知能量不當作零。回傳的
`same_model_comparable` 與 `modeled_ratio` 表示在相同假設下比較模型預測，
`hardware_performance_comparison_eligible` 仍為 false。

## 時間與能量實際涵蓋哪些動作

新入口採 **cold、serial typed graph**：第一次使用的輸入與權重從 host 搬入；
每個節點完成後才開始下一個節點；最後把 graph outputs 搬回 host。總成本包含：

- fabric 上的 input／weight／interdevice／output 搬移；
- 後端 packing、padding、解包與原生計算；
- HBM 指令間明確的 completion barriers；
- 所有配置裝置在整條圖執行時間內的 idle power。

總成本不含 compiler 的主機執行時間、應用程式前處理和 host idle power。
這個入口沒有請求間 overlap；原有 `suite`／`simulate` 的分析模擬器仍提供
多請求／資源重疊實驗，見 [分析模擬器](simulator.md)。

SRAM 的實際 CIMFlow 模擬器與 HBM 的修補版 DCC 都負責 timing，算術由各自
獨立的指令執行器產生，再逐節點核對 typed reference。CIMFlow upstream 沒有
CIM 權重與輸出的完整算術實作，DCC 原生 trace simulator 也不計算 tensor 值。
報告因此分開保存 instruction lowering、native timing 與 analytical 成本的證據。

HBM 的 Foundry SIMD16 指令先投影成 DCC PIM requests，每個指令後都有 barrier。
原生 requests 可能作用在該 channel 的全部 64 banks；logical bank 編號不代表
原生硬體只啟動單一 bank。範例 runtime 對每種 PIM 命令採明示的 **8-cycle
completion** 假設，clock 為 **769 ps/cycle**；completion 並未補齊 upstream
所有 issue-spacing 規則。這些是保存於 descriptor 的模型選擇，並非論文量測值。

範例 SRAM 幾何／功率、GPU／NPU 係數、fabric、packing、idle 與 HBM 每 cycle
能量係數也都是公開的 synthetic 參數。結果可用於核對編譯、資料流和探索配置，
尚未取得硬體校準或社群 benchmark 認證，也不等同於重現原論文整體加速比。

## 建置原生 runtimes

這是選用流程，需要 Linux C++ 建置環境、Git、CMake、Ninja，以及各上游專案的依賴；公開套件不會自動安裝它們。

- SRAM：[CIMFlow 建置說明](cimflow-backend.md#reproduce)與 [build_cimflow.py](../scripts/build_cimflow.py)。此 helper 會在需要時下載固定版本的作者 simulator／submodules，檢查來源並建置；不是離線 quickstart。
- HBM：[修補版 DCC 的來源與建置契約](../patches/dcc/README.md#rebuild-and-regressions)及 [build_dcc_patched.py](../scripts/build_dcc_patched.py)。先另行取得 [DCC 作者原碼](https://github.com/SPIN-Research-Group/DCC/tree/7ffb80e4f33e357177dc8ab148953c0484e1e322)，固定在該 revision，再準備下表的乾淨 Git dependency checkouts。DCC helper 不會替你下載這些來源。

| dependency 目錄名稱 | builder 要求的 revision |
|---|---|
| `yaml-cpp` | `0579ae3d976091d7d664aa9d2527e0d0cff25763` |
| `spdlog` | `ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36` |
| `argparse` | `997da9255618311d1fcb0135ce86022729d1f1cb` |

以下 DCC 指令假設這些目錄已置於 `external/dcc-dependencies/`，上游原碼已置於 `external/dcc-isca26/`。尚未取得它們時不能直接執行。使用新的建置／紀錄目錄；`8` cycles 是明示的模型假設。

```sh
python3 scripts/build_cimflow.py --record runs/cimflow-rebuild
python3 scripts/build_dcc_patched.py \
  --upstream external/dcc-isca26 \
  --source external/dcc-foundry-patched \
  --dependencies external/dcc-dependencies \
  --output runs/dcc-patched-final --pim-completion-cycles 8
```

CIMFlow helper 可取得固定版本與 submodules；DCC helper 使用已取得並固定版本的
upstream 與 dependency checkout，拒絕版本／來源不一致。DCC helper 會保留 patch、
編譯和 sanitizer regression 證據。若改用其他 `--output` 目錄，把 `platform.json`
中 HBM 的 `runtime_descriptor` 改成該建置所產生的 `runtime.json` 絕對路徑。

SRAM 範例指向 `external/cimflow-simulator/build/cim-simulator`；HBM 範例指向
`runs/dcc-patched-final/runtime.json`。範例中的外部 runtime 路徑以專案根目錄建立。
使用安裝後的套件時，以 `native-init`／`native-demo` 的 `--project-root` 明確指定
含有上述外部 artifact 的專案根目錄。若 runtime 建置於其他位置，先用 `native-init` 匯出配置，修改 `platform.json` 的 `executable`／`runtime_descriptor` 為實際絕對路徑，再用 `native-run` 執行；`native-demo` 使用上述預設路徑。

後端細節與擴充方式見 [SRAM 後端](cimflow-backend.md)及
[後端註冊契約](backend-plugin.md)。
