# 網格交易策略使用説明

本項目現已整合網格交易策略，支持現貨和永續合約兩種市場類型。

## 功能特點

### 現貨網格策略 (GridStrategy)
- 在價格區間內設置多個網格價格點位
- 在每個點位掛限價單
- 買單成交後，在上一個網格點位掛賣單
- 賣單成交後，在下一個網格點位掛買單
- 通過價格波動賺取網格利潤

### 永續合約網格策略 (PerpGridStrategy)
- 支持三種網格類型：
  - **neutral (中性網格)**: 在當前價格下方開多，上方開空，雙向網格
  - **long (做多網格)**: 只在下方開多，適合看漲行情
  - **short (做空網格)**: 只在上方開空，適合看跌行情
- 開倉後自動在對應網格點位掛平倉單
- 支持止損和止盈
- 持倉風險管理

## 使用方法

### 1. 現貨網格策略

#### 基本用法（自動價格範圍）
```bash
python run.py --symbol SOL_USDC --strategy grid --auto-price --grid-num 10 --quantity 0.1
```

#### 指定價格範圍
```bash
python run.py --symbol SOL_USDC --strategy grid \
  --grid-lower 140 \
  --grid-upper 160 \
  --grid-num 20 \
  --quantity 0.1 \
  --duration 7200
```

#### 等比網格（適合波動大的市場）
```bash
python run.py --symbol SOL_USDC --strategy grid \
  --auto-price \
  --price-range 10 \
  --grid-num 15 \
  --grid-mode geometric \
  --quantity 0.1
```

### 2. 永續合約網格策略

#### 中性網格（雙向開倉）
```bash
python run.py --symbol SOL_USDC --strategy perp_grid \
  --grid-type neutral \
  --auto-price \
  --price-range 5 \
  --grid-num 10 \
  --quantity 0.1 \
  --max-position 2.0
```

#### 做多網格（單向做多）
```bash
python run.py --symbol SOL_USDC --strategy perp_grid \
  --grid-type long \
  --grid-lower 140 \
  --grid-upper 150 \
  --grid-num 15 \
  --quantity 0.1 \
  --max-position 2.0
```

#### 做空網格（單向做空）
```bash
python run.py --symbol SOL_USDC --strategy perp_grid \
  --grid-type short \
  --grid-lower 150 \
  --grid-upper 160 \
  --grid-num 15 \
  --quantity 0.1 \
  --max-position 2.0
```

#### 帶止損止盈的網格
```bash
python run.py --symbol SOL_USDC --strategy perp_grid \
  --grid-type neutral \
  --auto-price \
  --grid-num 10 \
  --quantity 0.1 \
  --max-position 2.0 \
  --stop-loss 50 \
  --take-profit 100
```

## 參數説明

### 網格通用參數

| 參數 | 説明 | 默認值 | 示例 |
|------|------|--------|------|
| `--strategy` | 策略類型 | `standard` | `grid` 或 `perp_grid` |
| `--grid-upper` | 網格上限價格 | - | `160` |
| `--grid-lower` | 網格下限價格 | - | `140` |
| `--grid-num` | 網格數量 | `10` | `20` |
| `--auto-price` | 自動設置價格範圍 | `False` | - |
| `--price-range` | 自動模式下的價格範圍百分比 | `5.0` | `10.0` |
| `--grid-mode` | 網格模式 | `arithmetic` | `geometric` |
| `--quantity` | 每格訂單數量 | 自動計算 | `0.1` |

### 永續合約專用參數

| 參數 | 説明 | 默認值 | 示例 |
|------|------|--------|------|
| `--grid-type` | 網格類型 | `neutral` | `long` 或 `short` |
| `--max-position` | 最大持倉量 | `1.0` | `2.0` |
| `--stop-loss` | 止損閾值（以報價資產計） | - | `50` |
| `--take-profit` | 止盈閾值（以報價資產計） | - | `100` |

## 網格模式説明

### 等差網格 (arithmetic)
- 價格間隔相等
- 適合價格波動較小的市場
- 計算方式: `step = (upper - lower) / (num - 1)`

**示例**: 下限140，上限160，10個網格
```
價格點位: 140, 142.22, 144.44, 146.67, 148.89, 151.11, 153.33, 155.56, 157.78, 160
```

### 等比網格 (geometric)
- 價格比例相等
- 適合價格波動較大的市場
- 計算方式: `ratio = (upper / lower) ^ (1 / (num - 1))`

**示例**: 下限140，上限160，10個網格
```
價格點位: 140, 141.56, 143.15, 144.77, 146.42, 148.11, 149.82, 151.57, 153.35, 155.16
```

## 風險提示

1. **價格區間設置**:
   - 價格區間過窄: 容易觸及上下限，網格失效
   - 價格區間過寬: 單次利潤小，資金利用率低

2. **網格數量**:
   - 網格過少: 利潤空間小
   - 網格過多: 單格利潤低，手續費佔比高

3. **資金管理**:
   - 確保有足夠的基礎資產和報價資產
   - 永續合約注意槓桿風險
   - 建議設置合理的止損止盈

4. **市場選擇**:
   - 適合震盪行情
   - 不適合單邊趨勢市場
   - 注意市場流動性

## 監控與統計

運行時會顯示以下統計信息：

### 現貨網格
- 網格數量和價格範圍
- 買單/賣單成交次數
- 網格利潤
- 活躍訂單數

### 永續合約網格
- 網格類型和持倉情況
- 開多/開空次數
- 網格利潤
- 未實現盈虧
- 止損止盈狀態

## 完整示例

### 1. 保守型現貨網格（小波動）
```bash
python run.py --symbol SOL_USDC --strategy grid \
  --grid-lower 145 \
  --grid-upper 155 \
  --grid-num 20 \
  --grid-mode arithmetic \
  --quantity 0.05 \
  --duration 14400 \
  --interval 120
```

### 2. 激進型現貨網格（大波動）
```bash
python run.py --symbol SOL_USDC --strategy grid \
  --auto-price \
  --price-range 15 \
  --grid-num 30 \
  --grid-mode geometric \
  --quantity 0.1 \
  --duration 7200
```

### 3. 永續合約中性網格
```bash
python run.py --symbol SOL_USDC --strategy perp_grid \
  --grid-type neutral \
  --auto-price \
  --price-range 8 \
  --grid-num 15 \
  --quantity 0.08 \
  --max-position 2.0 \
  --stop-loss 80 \
  --take-profit 150 \
  --duration 10800
```

### 4. 永續合約做多網格（看漲）
```bash
python run.py --symbol SOL_USDC --strategy perp_grid \
  --grid-type long \
  --grid-lower 140 \
  --grid-upper 150 \
  --grid-num 20 \
  --grid-mode arithmetic \
  --quantity 0.1 \
  --max-position 3.0 \
  --take-profit 200
```

## 常見問題

### Q1: 如何選擇網格數量？
A: 根據預期波動和手續費率：
- 手續費率高：選擇較少網格（5-10個）
- 波動大：選擇較多網格（15-30個）
- 波動小：選擇中等網格（10-15個）

### Q2: 等差網格和等比網格如何選擇？
A:
- 等差網格：適合價格相對穩定的幣種（如穩定幣對）
- 等比網格：適合價格波動大的幣種（如主流幣、山寨幣）

### Q3: 永續合約網格類型如何選擇？
A:
- neutral: 適合震盪市場，雙向賺取價差
- long: 看漲時使用，逢低做多
- short: 看跌時使用，逢高做空

### Q4: 止損止盈如何設置？
A: 建議設置：
- 止損：網格總價值的 5-10%
- 止盈：網格總價值的 10-20%
- 根據市場波動調整

### Q5: 網格運行中價格突破網格怎麼辦？
A:
- 現貨網格：會停止在該方向掛單，等待價格迴歸
- 永續網格：觸及最大持倉限制後停止開倉
- 建議手動調整網格範圍或重啓策略

## 技術架構

網格策略完全基於現有的做市交易框架：

- **GridStrategy**: 繼承自 `MarketMaker`
- **PerpGridStrategy**: 繼承自 `PerpetualMarketMaker`
- 複用了所有基礎設施：
  - WebSocket 實時數據流
  - 訂單管理系統
  - 數據庫統計
  - 風險控制

## 更新日誌

- 2025-11-07: 整合網格交易策略到做市套利項目
  - 添加現貨網格策略
  - 添加永續合約網格策略
  - 支持等差和等比兩種網格模式
  - 支持自動價格範圍設置
  - 永續網格支持三種類型（neutral/long/short）

## 相關文檔

- [主項目 README](README.md)
- [永續合約做市策略](docs/perp_market_maker.md)
- [API 文檔](docs/api.md)
