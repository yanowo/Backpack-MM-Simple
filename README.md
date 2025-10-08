# Backpack Exchange 做市交易程序

這是一個針對 Backpack Exchange 的加密貨幣做市交易程序。該程序提供自動化做市功能，通過維持買賣價差賺取利潤。

Backpack 註冊連結：[https://backpack.exchange/refer/yan](https://backpack.exchange/refer/yan)

Twitter：[Yan Practice ⭕散修](https://x.com/practice_y11)

**使用本程式運行 MM 策略 可獲得 10~30% 自返傭 (每週由官方自動發放)**

## 功能特點

- **多交易所架構**：支援 Backpack、未來可擴展至其他交易所
- **自動化做市策略**：智能價差管理和訂單調整
- **永續合約做市**：倉位風險管理與風險中性機制
- **智能重平衡系統**：自動維持資產配置比例
- **增強日誌系統**：詳細的市場狀態和策略追蹤
- **WebSocket 實時連接**：即時市場數據和訂單更新
- **命令行界面**：靈活的參數配置和策略執行
- **可選資料庫紀錄**：根據需求啟用或停用資料庫寫入以兼顧效能

## 項目結構

```
lemon_trader/
│
├── api/                  # API相關模塊
│   ├── __init__.py
│   ├── auth.py           # API認證和簽名相關
│   ├── base_client.py    # 抽象基礎客户端 (支持繼承開發接入任意交易所)
│   ├── aster_client.py   # Aster Exchange 客户端
│   └── bp_client.py      # Backpack Exchange 客户端
│
├── websocket/            # WebSocket模塊
│   ├── __init__.py
│   └── client.py         # WebSocket客户端
│
├── database/             # 數據庫模塊
│   ├── __init__.py
│   └── db.py             # 數據庫操作
│
├── strategies/           # 策略模塊
│   ├── __init__.py
│   ├── market_maker.py   # 做市策略
│   └── perp_market_maker.py   # 合約做市策略
│
├── utils/                # 工具模塊
│   ├── __init__.py
│   └── helpers.py        # 輔助函數
│
├── cli/                  # 命令行界面
│   ├── __init__.py
│   └── commands.py       # 命令行命令
│
├── config.py             # 配置文件
├── logger.py             # 日誌配置
├── run.py                # 統一入口文件
└── README.md             # 説明文檔
```

## 環境要求

- Python 3.8 或更高版本
- 所需第三方庫：
  - nacl (用於API簽名)
  - requests
  - websocket-client
  - numpy
  - python-dotenv

## 安裝

1. 克隆或下載此代碼庫:

```bash
git clone https://github.com/yanowo/Backpack-MM-Simple.git
cd Backpack-MM-Simple
```

2. 安裝依賴:

```bash
pip install -r requirements.txt
```

3. 設置環境變數:

複製 `.env.example` 為 `.env` 並添加:

```
# Backpack Exchange
BACKPACK_KEY=your_backpack_api_key
BACKPACK_SECRET=your_backpack_secret_key
BACKPACK_PROXY_WEBSOCKET=http://user:pass@host:port/ 或者 http://host:port (若不需要則留空或移除)
BASE_URL=https://api.backpack.work

# Aster Exchange
ASTER_API_KEY=your_aster_api_key
ASTER_SECRET_KEY=your_aster_secret_key

# Optional Features
# ENABLE_DATABASE=1  # 啟用資料庫寫入 (預設0關閉)
```
## 使用方法

### 統一入口 (推薦)

```bash

# 啟動命令行界面 (推薦)
python run.py --cli  

# 直接運行做市策略
python run.py --exchange backpack --symbol SOL_USDC --spread 0.5 --max-orders 3 --duration 3600 --interval 60

# 直接運行 BackPack 永續做市
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 999999999 --interval 10

# 直接運行 Aster 永續做市
python run.py --exchange aster --market-type perp --symbol SOLUSDT --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 999999999 --interval 10

```

### 命令行參數

#### 基本參數
- `--api-key`: API 密鑰 (可選，默認使用環境變數)
- `--secret-key`: API 密鑰 (可選，默認使用環境變數)
- `--exchange`: 交易所選擇 (默認: backpack)
- `--ws-proxy`: Websocket 代理 (可選，默認使用環境變數)
- `--cli`: 啟動命令行界面
- `--enable-db`: 啟用資料庫寫入 (預設關閉)
- `--disable-db`: 停用資料庫寫入 (覆寫環境變數設定)

#### 做市參數
- `--symbol`: 交易對 (例如: SOL_USDC)
- `--spread`: 價差百分比 (例如: 0.5)
- `--quantity`: 訂單數量 (可選)
- `--max-orders`: 每側最大訂單數量 (默認: 3)
- `--duration`: 運行時間（秒）(默認: 3600)
- `--interval`: 更新間隔（秒）(默認: 60)
- `--market-type`: 市場類型 (`spot` 或 `perp`)
- `--target-position`: 永續合約目標淨倉位 (僅 `perp` 模式)
- `--max-position`: 永續合約最大允許淨倉 (僅 `perp` 模式)
- `--position-threshold`: 永續倉位調整觸發值 (僅 `perp` 模式)
- `--inventory-skew`: 永續做市報價偏移係數 (0-1，僅 `perp` 模式)
- `--stop-loss`: 以報價資產計價的未實現止損閾值 (僅 `perp` 模式)
- `--take-profit`: 以報價資產計價的未實現止盈閾值 (僅 `perp` 模式)

#### 重平設置參數
- `--enable-rebalance`: 開啟重平功能
- `--disable-rebalance`: 關閉重平功能
- `--base-asset-target`: 基礎資產目標比例 (0-100，默認: 30)
- `--rebalance-threshold`: 重平觸發閾值 (>0，默認: 15)

### 資料庫寫入選項

- 預設情況下，程式僅在記憶體中追蹤交易統計，不會寫入 SQLite 資料庫，以降低 I/O 負擔。
- 透過設定環境變數 `ENABLE_DATABASE=1` 或在啟動命令時加上 `--enable-db` 可啟用資料庫寫入功能。
- 若要臨時停用資料庫，可使用 `--disable-db` 覆寫設定。
- 當資料庫功能關閉時，相關的歷史統計/報表選單會顯示為停用狀態。

### 永續合約做市


#### 倉位管理邏輯優化

| 當前倉位 | 目標倉位 | 閾值 | 最大倉位 | 執行動作 |
|---------|---------|------|---------|---------|
| 0.1 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 無操作（在目標範圍內） |
| 0.25 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 減倉 0.05 SOL（只平掉超出閾值線的部分） |
| 0.5 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 風控平倉 0.1 SOL（降到最大倉位限制內） |

#### 增強日誌輸出範例

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

#### 啟動永續做市範例

```bash
# BackPack 永續做市
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 999999999 --interval 10
```

```bash
# Aster 永續做市
python run.py --exchange aster --market-type perp --symbol SOLUSDT --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 999999999 --interval 10
```

#### 永續合約參數詳解

- `target_position`：**目標持倉量**（絕對值）。設置您希望維持的庫存大小，策略會在持倉**超過**此目標時進行減倉，而非主動開倉達到目標。
- `max_position`：**最大持倉量**。倉位的硬性上限，超出後會立即強制平倉，是最高優先級的風控機制。
- `position_threshold`：**倉位調整閾值**。當 `當前持倉 > target_position + threshold` 時觸發減倉操作。
- `inventory_skew`：**風險中性係數** (0-1)。根據淨倉位自動調整報價：
  - 持有多單時：報價下移，吸引賣單成交
  - 持有空單時：報價上移，吸引買單成交
  - 目標：持續將淨倉位推向 `0`，降低方向性風險
- `stop_loss`：**未實現止損閾值**。請以**負值**輸入（例如 `-25`），代表允許的最大未實現虧損金額；當倉位的未實現虧損達到設定值時，策略會自動取消掛單並以市價平倉，防止虧損擴大；平倉成功後策略會立即恢復掛單。
- `take_profit`：**未實現止盈閾值**。當倉位未實現利潤達到設定值時，自動鎖定收益並平倉退出；平倉成功後策略會持續運行，等待下一次機會。

> ℹ️ 止損/止盈閾值以報價資產（如 USDC、USDT）為單位。僅當持倉存在且未實現盈虧超過設定值時才會觸發。

##### 止損/止盈觸發流程示例

假設您對 SOL_USDC_PERP 設置 `stop_loss=-25`、`take_profit=50`：

1. 當前持有 0.8 SOL 多頭倉位，未實現虧損擴大到 **-27 USDC**。
2. 策略偵測到虧損超過 25 USDC 閾值，立即**取消所有未成交掛單**並以市價賣出平倉。
3. 平倉成功後，日誌會提示「止損觸發，已以市價平倉」，接著策略重新計算報價並繼續掛單，整個流程無需人工介入。
4. 若之後行情好轉並達成 `take_profit=50` 的設定，流程相同，策略會自動鎖定利潤並持續運作。

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

## 運行示例

### 基本做市示例

```bash
python run.py --exchange backpack --symbol SOL_USDC --spread 0.2 --max-orders 5
```

### 開啟重平功能示例

```bash
# 標準重平設置
python run.py --exchange backpack --symbol SOL_USDC --spread 0.2 --enable-rebalance

# 自定義重平設置
python run.py --exchange backpack --symbol SOL_USDC --spread 0.2 --enable-rebalance --base-asset-target 25 --rebalance-threshold 12

# 高風險環境設置
python run.py --exchange backpack --symbol SOL_USDC --spread 0.3 --enable-rebalance --base-asset-target 20 --rebalance-threshold 10
```

### 關閉重平功能示例

```bash
python run.py --exchange backpack --symbol SOL_USDC --spread 0.2 --disable-rebalance
```

### 長時間運行示例

```bash
python run.py --exchange backpack --symbol SOL_USDC --spread 0.1 --duration 86400 --interval 120 --enable-rebalance --base-asset-target 30
```

### 完整參數示例

```bash
python run.py --exchange backpack --symbol SOL_USDC --spread 0.3 --quantity 0.5 --max-orders 3 --duration 7200 --interval 60 --enable-rebalance --base-asset-target 25 --rebalance-threshold 15
``` 

## 重平功能管理

### 通過 CLI 界面管理

```bash
python run.py --cli
```

在 CLI 界面中選擇：
- `5 - 執行做市策略`: 設置完整的做市和重平參數
- `8 - 重平設置管理`: 查看重平設置説明和測試配置

### 交互式配置

CLI 界面提供交互式重平配置，包括：
- 詳細的參數説明
- 模擬計算示例
- 智能參數建議
- 配置驗證

## 注意事項

### 一般注意事項
- 交易涉及風險，請謹慎使用
- 建議先在小資金上測試策略效果
- 定期檢查交易統計以評估策略表現

### 重平功能注意事項
- **手續費成本**: 重平衡會產生交易手續費，過於頻繁的重平衡可能影響整體收益
- **閾值設置**: 過低的閾值可能導致頻繁重平衡；過高的閾值可能無法及時控制風險
- **市場環境**: 根據市場波動率調整重平參數，高波動率時建議使用更保守的設置
- **資金效率**: 確保有足夠的可用餘額或抵押品支持重平衡操作
- **監控建議**: 定期檢查重平衡執行情況和效果，根據需要調整參數

### 最佳實踐建議

1. **新手用户**: 建議從默認設置開始 (30% 基礎資產，15% 閾值)
2. **保守策略**: 使用較低的基礎資產比例 (20-25%) 和較低的閾值 (10-12%)
3. **激進策略**: 可以使用較高的基礎資產比例 (35-40%) 和較高的閾值 (20-25%)
4. **測試驗證**: 先在小資金上測試不同的重平設置，找到最適合的參數組合

## 技術架構

程式採用模組化設計，支援多交易所擴展：

- **Base Client 架構**：抽象基礎類別，統一不同交易所的 API 介面
- **精確倉位管理**：只平掉超出閾值的部分，避免過度平倉風控
- **分層日誌系統**：市場狀態、策略決策、價格計算、執行結果四層資訊
- **相容性設計**：支援多種 API 回應格式，強化錯誤處理機制