# 網格策略快速啟動指南

本文檔提供所有交易所的網格策略快速啟動指令。

## 目錄
- [環境變量配置](#環境變量配置)
- [Backpack 現貨網格](#backpack-現貨網格)
- [Aster 永續網格](#aster-永續網格)
- [Paradex 永續網格](#paradex-永續網格)
- [Lighter 永續網格](#lighter-永續網格)
- [進階配置](#進階配置)

---

## 環境變量配置

### Backpack 交易所
```bash
export BACKPACK_KEY="your_api_key"
export BACKPACK_SECRET="your_secret_key"
```

### Aster 交易所
```bash
export ASTER_API_KEY="your_api_key"
export ASTER_SECRET_KEY="your_secret_key"
```

### Paradex 交易所
```bash
export PARADEX_PRIVATE_KEY="your_starknet_private_key"
export PARADEX_ACCOUNT_ADDRESS="your_starknet_account_address"
```

### Lighter 交易所
```bash
export LIGHTER_PRIVATE_KEY="your_private_key"
export LIGHTER_ACCOUNT_INDEX="your_account_index"
# 或者通過地址自動查找
export LIGHTER_ADDRESS="your_wallet_address"
```

---

## Backpack 現貨網格

### 基本使用（自動價格範圍）
```bash
python run.py --exchange backpack \
  --symbol SOL_USDC \
  --strategy grid \
  --auto-price \
  --grid-num 10 \
  --quantity 0.1
```

### 手動設定價格範圍
```bash
python run.py --exchange backpack \
  --symbol SOL_USDC \
  --strategy grid \
  --grid-lower 140 \
  --grid-upper 160 \
  --grid-num 10 \
  --quantity 0.1
```

### 等比網格
```bash
python run.py --exchange backpack \
  --symbol SOL_USDC \
  --strategy grid \
  --auto-price \
  --grid-num 10 \
  --grid-mode geometric \
  --quantity 0.1
```

---

## Aster 永續網格

### 中性網格（自動價格）
```bash
python run.py --exchange aster \
  --symbol BTCUSDT \
  --strategy perp_grid \
  --auto-price \
  --grid-num 10 \
  --grid-type neutral \
  --max-position 0.5
```

### 做多網格（手動價格）
```bash
python run.py --exchange aster \
  --symbol BTCUSDT \
  --strategy perp_grid \
  --grid-lower 45000 \
  --grid-upper 50000 \
  --grid-num 10 \
  --grid-type long \
  --max-position 1.0
```

### 做空網格
```bash
python run.py --exchange aster \
  --symbol BTCUSDT \
  --strategy perp_grid \
  --grid-lower 50000 \
  --grid-upper 55000 \
  --grid-num 10 \
  --grid-type short \
  --max-position 1.0
```

### 帶止損止盈的網格
```bash
python run.py --exchange aster \
  --symbol ETHUSDT \
  --strategy perp_grid \
  --auto-price \
  --grid-num 15 \
  --grid-type neutral \
  --stop-loss 100 \
  --take-profit 200 \
  --max-position 2.0
```

---

## Paradex 永續網格

### 中性網格（自動價格）
```bash
python run.py --exchange paradex \
  --symbol BTC-USD-PERP \
  --strategy perp_grid \
  --auto-price \
  --grid-num 10 \
  --grid-type neutral \
  --max-position 0.5
```

### 做多網格
```bash
python run.py --exchange paradex \
  --symbol BTC-USD-PERP \
  --strategy perp_grid \
  --grid-lower 45000 \
  --grid-upper 50000 \
  --grid-num 10 \
  --grid-type long \
  --max-position 1.0
```

### 做空網格
```bash
python run.py --exchange paradex \
  --symbol ETH-USD-PERP \
  --strategy perp_grid \
  --grid-lower 2500 \
  --grid-upper 3000 \
  --grid-num 10 \
  --grid-type short \
  --max-position 2.0
```

### 帶止損止盈
```bash
python run.py --exchange paradex \
  --symbol BTC-USD-PERP \
  --strategy perp_grid \
  --auto-price \
  --grid-num 15 \
  --stop-loss 150 \
  --take-profit 300 \
  --max-position 1.0
```

---

## Lighter 永續網格

### 中性網格（自動價格）
```bash
python run.py --exchange lighter \
  --symbol BTCUSDT \
  --strategy perp_grid \
  --auto-price \
  --grid-num 10 \
  --grid-type neutral \
  --max-position 0.5
```

### 做多網格
```bash
python run.py --exchange lighter \
  --symbol BTCUSDT \
  --strategy perp_grid \
  --grid-lower 45000 \
  --grid-upper 50000 \
  --grid-num 10 \
  --grid-type long \
  --max-position 1.0
```

### 做空網格
```bash
python run.py --exchange lighter \
  --symbol ETHUSDT \
  --strategy perp_grid \
  --grid-lower 2500 \
  --grid-upper 3000 \
  --grid-num 10 \
  --grid-type short \
  --max-position 2.0
```

---

## 進階配置

### 網格參數說明

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--grid-num` | 網格數量 | 10 |
| `--grid-mode` | 網格模式（arithmetic/geometric） | arithmetic |
| `--grid-type` | 網格類型（neutral/long/short） | neutral |
| `--auto-price` | 自動設置價格範圍 | false |
| `--price-range` | 自動模式下的價格範圍百分比 | 5.0 |
| `--grid-lower` | 網格下限價格 | - |
| `--grid-upper` | 網格上限價格 | - |

### 永續合約參數說明

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--max-position` | 最大持倉量（絕對值） | 1.0 |
| `--target-position` | 目標持倉量 | 0.0 |
| `--position-threshold` | 倉位調整觸發值 | 0.1 |
| `--inventory-skew` | 庫存偏移係數（0-1） | 0.0 |
| `--stop-loss` | 止損觸發值（報價資產） | - |
| `--take-profit` | 止盈觸發值（報價資產） | - |

### 運行時間設置

```bash
# 運行 1 小時，每 30 秒更新一次
python run.py --exchange aster \
  --symbol BTCUSDT \
  --strategy perp_grid \
  --auto-price \
  --duration 3600 \
  --interval 30
```

### 數據庫記錄

```bash
# 啟用數據庫寫入
python run.py --exchange aster \
  --symbol BTCUSDT \
  --strategy perp_grid \
  --auto-price \
  --enable-db
```

---

## 網格類型說明

### 中性網格（neutral）
- 在當前價格下方掛開多單
- 在當前價格上方掛開空單
- 適合震盪市場
- 風險：雙向持倉

### 做多網格（long）
- 僅在價格下跌時開多
- 價格上漲時平多
- 適合看多但波動的市場
- 風險：單向多頭持倉

### 做空網格（short）
- 僅在價格上漲時開空
- 價格下跌時平空
- 適合看空但波動的市場
- 風險：單向空頭持倉

---

## 風險提示

1. **網格策略風險**
   - 單邊行情可能導致持倉累積
   - 建議設置止損止盈
   - 建議控制最大持倉量

2. **永續合約風險**
   - 槓桿風險
   - 資金費率
   - 強平風險

3. **交易所差異**
   - 不同交易所的交易規則可能不同
   - 建議小額測試
   - 注意最小訂單量限制

---

## 交易所差異說明

### 批量下單支持
- **Backpack**: ✅ 支持批量下單（`execute_order_batch`）
- **Aster**: ❌ 不支持批量下單（自動使用逐個下單）
- **Paradex**: ❌ 不支持批量下單（自動使用逐個下單）
- **Lighter**: ❌ 不支持批量下單（自動使用逐個下單）

**注意**: 策略會自動檢測交易所是否支持批量下單，如果不支持會自動切換到逐個下單模式，不影響使用。

### 訂單初始化時間
由於不同交易所的批量下單支持不同，網格初始化時間會有差異：
- Backpack: 快速（批量下單）
- 其他交易所: 稍慢（逐個下單，網格數量越多越慢）

建議：
- 網格數量不要設置過多（建議 10-20 個）
- 對於不支持批量下單的交易所，可以適當減少網格數量以加快初始化

---

## 常見問題

### Q: 如何選擇網格數量？
A: 建議根據預期波動範圍決定，一般 10-20 個網格較為合適。網格過多會導致單網格利潤過小。對於不支持批量下單的交易所（Aster、Paradex、Lighter），網格數量過多會導致初始化時間過長。

### Q: 如何設置價格範圍？
A:
- 使用 `--auto-price` 自動設置（適合新手）
- 根據技術分析手動設置支撐位和壓力位
- 建議價格範圍不要超過當前價格的 ±10%

### Q: 中性網格會一直增加持倉嗎？
A: 是的，在單邊行情中可能累積持倉。建議：
- 設置 `--max-position` 限制最大持倉
- 設置 `--stop-loss` 止損
- 使用做多或做空網格代替中性網格

### Q: 如何查看運行狀態？
A: 策略會定期輸出統計信息，包括：
- 網格成交次數
- 當前持倉
- 未實現盈虧
- 網格利潤

---

## 技術支持

如有問題或建議，請提交 Issue 到項目 GitHub 倉庫。
