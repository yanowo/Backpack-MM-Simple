# 現貨做市策略說明

## 策略概述

現貨做市策略通過在買賣盤口同時掛出多層訂單，從買賣價差中獲利。程序會根據市場波動自動調整訂單，並可選擇性啟用資產重平衡功能。

> **支援交易所**: 目前僅 Backpack 支持現貨做市。其他交易所（Aster、Paradex、Lighter、APEX、StandX）為純永續合約交易所，請使用永續合約做市策略。

## 核心功能

- **多層訂單管理**：支持在買賣盤口掛出多層訂單（可配置）
- **智能價差管理**：根據市場波動自動調整買賣價差
- **自動重平衡**：維持資產配置比例，降低單邊風險
- **實時監控**：通過 WebSocket 實時更新訂單狀態

## 啟動示例

### 基本現貨做市

```bash
# Backpack 現貨做市
python run.py --exchange backpack --symbol SOL_USDC --spread 0.01 --max-orders 3 --duration 3600 --interval 60
```

### 開啟重平衡功能

```bash
# 標準重平設置
python run.py --exchange backpack --symbol SOL_USDC --spread 0.2 --enable-rebalance

# 自定義重平設置
python run.py --exchange backpack --symbol SOL_USDC --spread 0.2 --enable-rebalance --base-asset-target 25 --rebalance-threshold 12

# 高風險環境設置
python run.py --exchange backpack --symbol SOL_USDC --spread 0.3 --enable-rebalance --base-asset-target 20 --rebalance-threshold 10
```

### 長時間運行

```bash
python run.py --exchange backpack --symbol SOL_USDC --spread 0.1 --duration 86400 --interval 120 --enable-rebalance --base-asset-target 30
```

## 重平衡功能詳解

### 什麼是重平衡？

重平衡功能用於維持資產配置的平衡，避免因市場波動導致的資產比例失衡。例如，如果您設置基礎資產目標比例為30%，當基礎資產比例因價格波動偏離目標過多時，程序會自動執行交易來恢復平衡。

### 重平設置參數説明

1. **重平功能開關**: 控制是否啟用自動重平衡
2. **基礎資產目標比例**: 基礎資產應佔總資產的百分比 (0-100%)
3. **重平觸發閾值**: 當實際比例偏離目標比例超過此閾值時觸發重平衡

### 重平設置建議

根據不同的市場環境，建議使用不同的重平設置：以 SOL 為例

#### 高波動率市場 (波動率 > 5%)
```bash
--base-asset-target 20 --rebalance-threshold 10
```
- 基礎資產比例: 20-25% (降低風險暴露)
- 重平閾值: 10-12% (更頻繁重平衡)

#### 中等波動率市場 (波動率 2-5%)
```bash
--base-asset-target 30 --rebalance-threshold 15
```
- 基礎資產比例: 25-35% (標準配置)
- 重平閾值: 12-18% (適中頻率)

#### 低波動率市場 (波動率 < 2%)
```bash
--base-asset-target 35 --rebalance-threshold 20
```
- 基礎資產比例: 30-40% (可承受更高暴露)
- 重平閾值: 15-25% (較少重平衡)

### 重平計算示例

假設您設置：
- 總資產: 1000 USDC
- 基礎資產目標比例: 30% (SOL)
- 重平觸發閾值: 15%

**相當於:**
- 基礎資產 SOL 價值: 300 USDC (30%)
- 報價資產 USDC 價值: 700 USDC (70%)

**觸發重平衡的情況:**
- 當 SOL 價值 > 450 USDC (偏差 > 150 USDC = 15%) → 賣出 SOL
- 當 SOL 價值 < 150 USDC (偏差 > 150 USDC = 15%) → 買入 SOL
- 在 150-450 USDC 範圍內不會觸發重平衡

## 參數說明

### 基本參數
- `--symbol`: 交易對 (例如: SOL_USDC)
- `--spread`: 價差百分比 (例如: 0.5)
- `--quantity`: 訂單數量 (可選)
- `--max-orders`: 每側最大訂單數量 (默認: 3)
- `--duration`: 運行時間（秒）(默認: 3600)
- `--interval`: 更新間隔（秒）(默認: 60)

### 重平衡參數
- `--enable-rebalance`: 開啟重平功能
- `--disable-rebalance`: 關閉重平功能
- `--base-asset-target`: 基礎資產目標比例 (0-100，默認: 30)
- `--rebalance-threshold`: 重平觸發閾值 (>0，默認: 15)

## 注意事項

### 風險提示
- 交易涉及風險，請謹慎使用
- 建議先在小資金上測試策略效果
- 定期檢查交易統計以評估策略表現

### 重平功能注意事項
- **手續費成本**: 重平衡會產生交易手續費，過於頻繁的重平衡可能影響整體收益
- **閾值設置**: 過低的閾值可能導致頻繁重平衡；過高的閾值可能無法及時控制風險
- **市場環境**: 根據市場波動率調整重平參數，高波動率時建議使用更保守的設置
- **資金效率**: 確保有足夠的可用餘額支持重平衡操作
- **監控建議**: 定期檢查重平衡執行情況和效果，根據需要調整參數

### 最佳實踐建議

1. **新手用户**: 建議從默認設置開始 (30% 基礎資產，15% 閾值)
2. **保守策略**: 使用較低的基礎資產比例 (20-25%) 和較低的閾值 (10-12%)
3. **激進策略**: 可以使用較高的基礎資產比例 (35-40%) 和較高的閾值 (20-25%)
4. **測試驗證**: 先在小資金上測試不同的重平設置，找到最適合的參數組合
