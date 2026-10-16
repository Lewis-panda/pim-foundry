> Historical local development record. This is not the current installation guide or a claim that its local artifacts are distributed. See [the current guide](getting-started.md), [current limitations](limitations.md), and [public validation scope](../VALIDATION.md).

# 從目前工作台走向共用 PIM 框架

權重已分到各 bank 後，若某個 kernel 又要求跨 bank 的元素一起計算，就需要重新搬資料或在 host 合併局部結果；只最佳化 bank 裡的 MAC 次數會漏掉這些成本。共同介面必須同時描述運算、資料放置與搬移，後端才能做可驗證的選擇。

## 已有實作

v0.2 的共同 CIM／PIM 設計見 [unified-cim.md](unified-cim.md)。新增了以精確整數為契約的功能參考圖、SRAM／HBM 邏輯資源模型、顯式搬移與配置搜尋，可執行單後端和異質配置。這個功能 reference 與下列 DCC trace 路徑的驗證狀態分開。

```text
Gemv shape contract
  → pinned DCC configuration enumeration
  → conservative shape/resource guards
  → isolated DCC HBM GEMV trace generator
  → syntax/address inspection + provenance report
```

DCC 路徑目前沒有 tensor value frontend、已驗證的 IR lowering、schedule 最佳化器或 runtime。`model.py` 是此路徑的形狀與配置契約。新增的 `core.py` 提供另一套有限的功能圖，尚未接入 DCC 或 CIMFlow 的數值執行。第一個符合條件的 DCC candidate 只是預設實驗點，不能用命令較少推論速度較快。

來源 snapshot 不修改；後端的動態程式載入只針對固定雜湊的已檢查來源。每次工作在獨立程序執行，防止 global search state 與 list mutation 污染下一次結果。輸出帶來源與 trace 雜湊，方便追查差異。

## 後續驗收門檻

| 階段 | 具體實作 | 必須拿出的證據 |
|---|---|---|
| 1. 功能正確的單後端 | 查核 DCC simulator 的 LD/ST、低位地址、MAC 和 writeback 語意；明確定義 packed layout；修正 base/stride/head/chunk 問題；載入真實數值 | 隨機與邊界資料對照獨立 CPU reference；說明 FP16 累加與 rounding 規則；每個修正有最小反例 |
| 2. 可重用的 kernel IR | 先支援 GEMV、reduction、elementwise；明確 dtype、shape、layout、placement 和 effect；不支援的 op 明確報錯 | 同一輸入經 reference 與 PIM 路徑語意一致；非法 layout、alias 與容量案例會被拒絕 |
| 3. 第二個 backend | 選 AttAcc 或 UPMEM，依實際指令能力實作 lower、transfer、launch、wait 和 readback | 共用 kernel tests 能在兩條路徑執行；不能只把 target 名稱換掉 |
| 4. 有效的成本模型與搜尋 | 分開計算 host rearrangement、搬移、PIM compute、同步與 reduction；區分初次權重載入與重複使用 | 對獨立 shape 測試集驗證模型誤差；比較手工 baseline、固定 schedule、只算 compute 的 ablation；記錄完整端到端成本 |
| 5. 可持續使用的發布 | 穩定版本契約、依賴鎖定、CI、失敗診斷、真實 SDK／硬體 adapter 和授權檔 | 乾淨環境可重現、支援矩陣清楚、無默默 fallback、至少兩個後端有持續測試 |

這些階段不附缺乏人力／硬體依據的完成日期。能在本機產生一個 trace，不足以推算做成成熟框架的工期。

## 研究評估的最低要求

1. 固定來源版本、完整 shape、精度、layout、warm/cold 狀態、模擬器與硬體參數。
2. 先證明數值語意，再比較效能；trace command 數不等於 MAC 數或週期。
3. 與相同工作負載的 host baseline、原版 DCC、手工／固定配置比較。若是模擬數據，明示 simulator 假設與 host 成本來源。
4. 將資料搬移納入端到端成本，並做去掉搬移成本、固定 layout、固定 schedule 的 ablation。
5. 預測模型的訓練資料與評估 shape 分離；不可把同一 trace 衍生的樣本散放兩邊後宣稱可泛化。
6. 對實測的隨機變動做重複試驗與區間估計；對決定性的 simulator，回報模型誤差和參數敏感度，避免把重跑相同模型當成實機統計證據。

## 整合選擇

目前依賴 DCC 是為了沿用其具體 HBM trace 路徑。若後續需要廣泛前端與跨 CIM/CNM dialect，可先評估 [CINM](https://github.com/tud-ccc/Cinnamon) 的實際覆蓋再決定接入 MLIR，避免僅因 framework 名稱就重建整套 IR。這份 adapter 目前沒有接 TVM、MLIR、ONNX 或 PyTorch。
