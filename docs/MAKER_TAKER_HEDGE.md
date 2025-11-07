# Maker-Taker 對沖策略說明

## 策略概述

Maker-Taker 對沖策略僅在買一/賣一掛單，當掛單成交後立即以市價單對沖，保持接近零持倉。此策略適合追求穩定收益並希望降低方向性風險的用戶。

## 核心特點

- **單層掛單**：僅在買一/賣一價格掛單，提高資金利用率
- **即時對沖**：Maker 訂單成交後立即以市價單對沖
- **零持倉管理**：保持倉位接近目標值，降低方向性風險
- **支持雙市場**：同時支持現貨和永續合約市場
- **延遲價差收益**：從 Maker 成交到 Taker 對沖的時間差中獲利

## 交易所相容性

| 交易所 | 現貨對沖 | 永續對沖 | 特殊處理 |
|--------|---------|---------|----------|
| Backpack | ✅ | ✅ | 自動啟用 `autoLendRedeem` |
| Aster | ❌ | ✅ | 移除 `postOnly` (永續) |
| Paradex | ❌ | ✅ | JWT 自動刷新 |

## 啟動示例

### 現貨市場對沖

```bash
# Backpack 現貨對沖
python run.py --exchange backpack --symbol SOL_USDC --spread 0.01 --strategy maker_hedge --duration 3600 --interval 30
```

### 永續合約對沖

```bash
# Backpack 永續對沖
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.01 --quantity 0.1 --strategy maker_hedge --target-position 0 --max-position 5 --position-threshold 2 --duration 3600 --interval 8

# Aster 永續對沖
python run.py --exchange aster --market-type perp --symbol SOLUSDT --spread 0.01 --quantity 0.1 --strategy maker_hedge --target-position 0 --max-position 5 --position-threshold 2 --duration 3600 --interval 15

# Paradex 永續對沖
python run.py --exchange paradex --market-type perp --symbol BTC-USD-PERP --spread 0.01 --quantity 0.001 --strategy maker_hedge --target-position 0 --max-position 1 --position-threshold 0.1 --duration 3600 --interval 8
```

## 執行流程示例

### 永續合約對沖流程

```
[INFO] 初始化 Maker-Taker 對沖策略 (永續合約僅掛買一/賣一)
[INFO] 對沖參考倉位初始化為 0.00000000
[INFO] 買單已掛出: 價格 239.45, 數量 0.1
[INFO] 賣單已掛出: 價格 239.65, 數量 0.1
[INFO] 處理Maker成交：Ask 0.1@239.65
[INFO] 偵測到 Maker 成交，準備以市價對沖 0.10000000 Bid
[INFO] 提交市價對沖訂單: Bid 0.10000000 (第 1 次嘗試) [reduceOnly=True]
[INFO] 市價對沖訂單已提交: def456
[INFO] 市價對沖已完成，倉位回到參考水位
```

### 現貨市場對沖流程

```
[INFO] 初始化 Maker-Taker 對沖策略 (現貨僅掛買一/賣一)
[INFO] 對沖參考倉位初始化為 0.00000000 SOL
[INFO] 買單已掛出: 價格 239.45, 數量 0.1
[INFO] 賣單已掛出: 價格 239.65, 數量 0.1
[INFO] 處理Maker成交：Bid 0.1@239.45
[INFO] 偵測到 Maker 成交，準備以市價對沖 0.10000000 Ask
[INFO] 提交市價對沖訂單: Ask 0.10000000 (第 1 次嘗試)
[INFO] 市價對沖訂單已提交: abc123
[INFO] 市價對沖已完成，倉位回到參考水位
```

## 對沖機制詳解

### 對沖觸發條件

1. **Maker 訂單成交**：檢測到買單或賣單成交
2. **計算對沖數量**：累積需要對沖的數量（含殘量）
3. **提交市價單**：立即以市價單對沖
4. **驗證結果**：確認倉位回到目標水位

### 殘量處理

由於交易所精度限制，對沖數量可能無法完全匹配成交量：

```python
# 示例
Maker 成交: 0.123456 SOL
對沖精度: 0.01 SOL (最小下單量)
實際對沖: 0.12 SOL
殘量累積: 0.003456 SOL → 下次一併處理
```

### 永續合約特殊處理

永續合約對沖使用 `reduceOnly=True` 參數：
- ✅ 確保不會意外開新倉位
- ✅ 只減少現有持倉
- ✅ 提高風控安全性

## 策略對比

| 特性 | 標準做市 | Maker-Taker 對沖 |
|------|----------|-----------------|
| 訂單層數 | 多層 (可配置) | 單層 (買一/賣一) |
| 持倉風險 | 累積持倉 | 即時對沖，接近零持倉 |
| 資金占用率 | 較高 | 較低 |
| 適合場景 | 震盪行情 | 提高交易量 |
| 盈利模式 | 價差 + 持倉升值 | 延遲價差 |
| 手續費成本 | 較低 (純 Maker) | 較高 (Maker + Taker) |

## 參數說明

### 基本參數
- `--strategy maker_hedge`: 指定使用 Maker-Taker 對沖策略
- `--symbol`: 交易對
- `--spread`: 價差百分比
- `--quantity`: 訂單數量
- `--interval`: 更新間隔（建議 5-15 秒）

### 永續合約參數
- `--target-position`: 目標持倉量（通常設為 0）
- `--max-position`: 最大持倉量（風控上限）
- `--position-threshold`: 倉位調整閾值

## 常見問題

### Q1: 為什麼對沖後還有小額殘量？

**A**: 由於交易所的精度限制，對沖數量可能無法完全匹配成交量。策略會記錄這些殘量（通常 < 最小下單量），並在下次成交時一併處理。

### Q2: 對沖失敗會怎樣？

**A**: 若市價對沖訂單提交失敗（如餘額不足、API 錯誤），策略會：
1. 保留完整對沖目標量至下次嘗試
2. 在日誌中記錄錯誤訊息
3. 繼續運行，等待下次對沖機會

### Q3: 對沖策略能用於高波動行情嗎？

**A**: 可以，但需注意：
- 高波動可能導致對沖滑點增加
- 建議縮短 `interval` (如 5-10 秒) 以更快反應市場變化
- 設置合理的 `max-position` 作為風控上限

### Q4: 對沖策略的手續費成本如何？

**A**: 對沖策略會產生：
- **Maker 費用**：掛單成交費率
- **Taker 費用**：市價單手續費
- **淨成本**：取決於交易所的費率結構

建議在手續費較低或 Maker 返佣較高的交易所使用。

### Q5: 為什麼建議使用較短的更新間隔？

**A**: 較短的更新間隔（5-15 秒）可以：
- 更快檢測 Maker 訂單成交
- 減少對沖延遲，降低價格波動風險
- 及時調整掛單價格，保持競爭力

## 風險提示

1. **滑點風險**：市價對沖可能產生滑點，特別是在流動性較差或高波動時
2. **手續費成本**：頻繁對沖會產生 Taker 手續費，需評估盈利空間
3. **對沖延遲**：從檢測成交到完成對沖存在時間差，價格可能波動
4. **殘量累積**：長期運行可能累積較多殘量，需定期手動平倉
5. **API 限制**：過於頻繁的請求可能觸發 API 限流