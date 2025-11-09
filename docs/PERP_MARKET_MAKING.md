# 永續合約做市策略說明

## 策略概述

永續合約做市策略針對永續合約市場設計，通過智能倉位管理和風險中性機制，在獲取做市收益的同時控制方向性風險。

## 核心功能

- **智能倉位管理**：根據目標倉位自動調整訂單和平倉
- **風險中性機制**：通過 `inventory_skew` 參數自動調整報價，將倉位推向目標值
- **止損止盈**：支持設置未實現盈虧的自動平倉閾值
- **分層風控**：倉位閾值、最大倉位多重保護
- **增強日誌**：詳細的市場狀態、倉位和盈虧追蹤

## 啟動示例

```bash
# Backpack 永續做市
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 3600 --interval 10

# Aster 永續做市
python run.py --exchange aster --market-type perp --symbol SOLUSDT --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 3600 --interval 10

# Paradex 永續做市
python run.py --exchange paradex --market-type perp --symbol BTC-USD-PERP --spread 0.01 --quantity 0.001 --max-orders 2 --target-position 0 --max-position 1 --position-threshold 0.1 --inventory-skew 0 --stop-loss -10 --take-profit 20 --duration 3600 --interval 10
```

## 倉位管理邏輯

### 倉位調整示例

| 當前倉位 | 目標倉位 | 閾值 | 最大倉位 | 執行動作 |
|---------|---------|------|---------|---------|
| 0.1 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 無操作（在目標範圍內） |
| 0.25 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 減倉 0.05 SOL（只平掉超出閾值線的部分） |
| 0.5 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 風控平倉 0.1 SOL（降到最大倉位限制內） |

### 增強日誌輸出範例

```
=== 市場狀態 ===
盤口: Bid 239.379 | Ask 239.447 | 價差 0.068 (0.028%)
中間價: 239.408
持倉: 空頭 6.000 SOL | 目標: 1.0 | 上限: 1.0

=== 價格計算 ===
原始掛單: 買 238.800 | 賣 239.996
偏移計算: 淨持倉 -6.000 | 偏移係數 0.00 | 偏移量 0.0000
調整後掛單: 買 238.800 | 賣 239.996

=== 本次執行總結 ===
成交: 買入 0.200 SOL | 賣出 0.150 SOL
本次盈虧: 2.4500 USDT (手續費: 0.1200)
累計盈虧: 15.2300 USDT | 未實現: -1.8900 USDT
活躍訂單: 買 238.800 | 賣 239.996 | 價差 1.196 (0.50%)
```

## 參數詳解

### 倉位參數
- `--target-position`: **目標持倉量**（絕對值）。策略會在持倉超過此目標時進行減倉
- `--max-position`: **最大持倉量**。倉位的硬性上限，超出後會立即強制平倉
- `--position-threshold`: **倉位調整閾值**。當 `當前持倉 > target_position + threshold` 時觸發減倉

### 風險中性參數
- `--inventory-skew`: **風險中性係數** (0-1)。根據淨倉位自動調整報價：
  - 持有多單時：報價下移，吸引賣單成交
  - 持有空單時：報價上移，吸引買單成交
  - 目標：持續將淨倉位推向 `0`，降低方向性風險

### 止損止盈參數
- `--stop-loss`: **未實現止損閾值**。以**負值**輸入（例如 `-25`），代表允許的最大未實現虧損金額
- `--take-profit`: **未實現止盈閾值**。當倉位未實現利潤達到設定值時，自動鎖定收益

> 止損/止盈閾值以報價資產（如 USDC、USDT）為單位。僅當持倉存在且未實現盈虧超過設定值時才會觸發。

## 止損止盈觸發流程

假設您對 SOL_USDC_PERP 設置 `stop_loss=-25`、`take_profit=50`：

### 止損觸發流程

1. 當前持有 0.8 SOL 多頭倉位，未實現虧損擴大到 **-27 USDC**
2. 策略偵測到虧損超過 25 USDC 閾值
3. 立即**取消所有未成交掛單**
4. 以市價賣出平倉
5. 日誌提示「止損觸發，已以市價平倉」
6. 策略重新計算報價並繼續掛單

### 止盈觸發流程

1. 持倉未實現利潤達到 **52 USDC**
2. 策略偵測到利潤超過 50 USDC 閾值
3. 自動鎖定利潤並平倉
4. 策略持續運作，等待下一次機會

## 交易所特殊處理

| 交易所 | 特殊處理 | 說明 |
|--------|---------|------|
| Backpack | `autoLendRedeem=true` | 自動借貸贖回 |
| Aster | 移除 `postOnly` | 永續合約不支持 postOnly |
| Paradex | JWT 自動刷新 | 保持認證狀態 |

## 風險提示

1. **資金費率**：長期持倉會產生資金費率成本
2. **爆倉風險**：確保保證金充足，設置合理的最大倉位
3. **滑點風險**：市價平倉可能產生滑點，特別是在流動性較差的市場
4. **監控重要性**：建議定期檢查倉位和盈虧狀態

## 最佳實踐

1. **新手建議**：
   - 使用較小的 `max-position` (如 0.5-1.0)
   - 設置保守的止損閾值
   - 使用較高的 `inventory-skew` (0.3-0.5)

2. **進階用戶**：
   - 根據市場波動率調整 `spread`
   - 使用較短的 `interval` (5-10秒) 以更快反應市場
   - 結合市場深度調整訂單數量

3. **監控指標**：
   - 未實現盈虧變化
   - 倉位偏離目標的程度
   - 成交率和手續費佔比
   - 資金費率累積成本