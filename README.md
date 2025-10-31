# 加密貨幣做市交易程序

這是一個支援對沖與做市策略的加密貨幣交易架構，具備多交易所自助接入與自訂策略擴充能力。系統內建自動化做市功能，可透過維持買賣價差獲取穩定收益。目前已支援 Backpack、Aster 和 Paradex 等交易所。

Backpack 註冊連結：[https://backpack.exchange/refer/yan](https://backpack.exchange/refer/yan)

Asterdex 註冊連結：[https://www.asterdex.com/referral/1a7b6E](https://www.asterdex.com/referral/1a7b6E)

Paradex 註冊連結：[https://app.paradex.trade/r/yanowo](https://app.paradex.trade/r/yanowo)

Twitter：[Yan Practice ⭕散修](https://x.com/practice_y11)

**使用本程式運行 MM 策略 可獲得 10~30% 自返傭 (每週由官方自動發放)**

## 功能特點

- **Web 控制台**：直觀的圖形化界面，實時監控交易狀態和策略表現
- **多交易所架構**：支援 Backpack、Aster、Paradex，可擴展至其他交易所
- **Paradex 整合**：完整支援 Paradex 永續合約交易，包含 JWT 自動更新機制
- **自動化做市策略**：智能價差管理和訂單調整
- **Maker-Taker 對沖策略**：僅在買一/賣一掛單並於成交後以市價單即刻對沖，支援現貨與永續市場
- **永續合約做市**：倉位風險管理與風險中性機制
- **智能重平衡系統**：自動維持資產配置比例
- **JWT 自動更新**：Paradex JWT token 自動刷新，避免認證過期
- **增強日誌系統**：詳細的市場狀態和策略追蹤
- **WebSocket 實時連接**：即時市場數據和訂單更新（支持策略 WebSocket 和 Web 控制台 WebSocket）
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
│   ├── bp_client.py      # Backpack Exchange 客户端
│   ├── aster_client.py   # Aster Exchange 客户端
│   └── paradex_client.py # Paradex Exchange 客户端 (含 JWT 認證)
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
│   ├── perp_market_maker.py   # 合約做市策略
│   └── maker_taker_hedge.py   # 合約對沖策略
│
├── utils/                # 工具模塊
│   ├── __init__.py
│   └── helpers.py        # 輔助函數
│
├── cli/                  # 命令行界面
│   ├── __init__.py
│   └── commands.py       # 命令行命令
│
├── web/                  # Web 控制台界面
│   ├── __init__.py
│   ├── server.py         # Flask Web 服務器
│   ├── templates/        # HTML 模板
│   │   └── index.html
│   └── static/           # 靜態資源 (CSS, JS)
│
├── config.py             # 配置文件
├── logger.py             # 日誌配置
├── run.py                # 統一入口文件
└── README.md             # 説明文檔
```

## 環境要求

- Python 3.8 或更高版本
- 所需第三方庫：
  - PyNaCl
  - requests
  - websocket-client
  - numpy
  - python-dotenv
  - starknet-py
  - flask
  - flask-socketio
  - python-socketio

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
BASE_URL=https://api.backpack.work

# WS 代理格式 http://user:pass@host:port/ 或者 http://host:port (若不需要則留空或移除)
BACKPACK_PROXY_WEBSOCKET=


# Aster Exchange
ASTER_API_KEY=your_aster_api_key
ASTER_SECRET_KEY=your_aster_secret_key

# Paradex Exchange (使用 StarkNet 賬户認證)
PARADEX_PRIVATE_KEY=your_starknet_private_key
PARADEX_ACCOUNT_ADDRESS=your_starknet_account_address
PARADEX_BASE_URL=https://api.prod.paradex.trade/v1

# WS 代理格式 http://user:pass@host:port/ 或者 http://host:port (若不需要則留空或移除)
PARADEX_PROXY_WEBSOCKET=

# Optional Features
# ENABLE_DATABASE=1  # 啟用資料庫寫入 (預設0關閉)
```
## 使用方法

### Web 控制台 (推薦)

程序提供了直觀的 Web 控制台界面，方便可視化管理和監控交易策略。

![Web 控制台界面](dashboard.png)

#### 啟動 Web 服務器

```bash
# 啟動 Web 服務器（默認端口 5000）
python run.py --web
```

#### 訪問控制台

啟動後，在瀏覽器中訪問：
```
http://localhost:5000
```

#### Web 界面功能

- **實時監控**：查看交易統計、餘額、盈虧等實時數據（每5秒更新）
- **策略管理**：啟動/停止做市策略，支持多種策略類型
- **參數配置**：
  - 交易所選擇（Backpack、Aster、Paradex）
  - 市場類型（現貨 / 永續合約）
  - 策略類型（標準做市 / Maker-Taker 對沖）
  - 交易對、價差、訂單數量等
  - 永續合約參數（目標倉位、最大倉位、止損止盈等）
  - 現貨重平衡參數
- **數據展示**：
  - 當前價格和余額（只顯示報價資產 USDT/USDC/USD）
  - 交易統計（買賣筆數、成交量、手續費）
  - 盈虧分析（已實現/未實現盈虧、累計盈虧、磨損率）
  - 運行時間統計

#### 使用示例

1. 啟動 Web 服務器
2. 在瀏覽器打開控制台
3. 配置環境變量（API Key 需提前在 .env 文件中設置）
4. 選擇交易所和交易對
5. 設置策略參數
6. 點擊"啟動機器人"開始交易
7. 實時查看交易狀態和統計數據
8. 需要停止時點擊"停止機器人"

#### 注意事項

- Web 服務器需要持續運行以監控策略
- API 密鑰通過環境變量讀取，不會在 Web 界面中顯示
- 支持多個瀏覽器標籤頁同時查看（通過 WebSocket 同步）
- 停止策略後統計數據會保留，方便查看最終結果

---

### 啟動命令行界面

```bash
python run.py --cli  
```


### 快速啟動方式

```bash
# BackPack 現貨做市
python run.py --exchange backpack --symbol SOL_USDC --spread 0.01 --max-orders 3 --duration 3600 --interval 60

# BackPack Maker-Taker 現貨對沖
python run.py --exchange backpack --symbol SOL_USDC --spread 0.01 --strategy maker_hedge --duration 3600 --interval 30

# BackPack 永續做市
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 3600 --interval 10

# BackPack Maker-Taker 永續對沖
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.01 --quantity 0.1 --strategy maker_hedge --target-position 0 --max-position 5 --position-threshold 2 --duration 3600 --interval 8

# Aster 永續做市
python run.py --exchange aster --market-type perp --symbol SOLUSDT --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 3600 --interval 10

# Aster Maker-Taker 永續對沖
python run.py --exchange aster --market-type perp --symbol SOLUSDT --spread 0.01 --quantity 0.1 --strategy maker_hedge --target-position 0 --max-position 5 --position-threshold 2 --duration 3600 --interval 15

# Paradex 永續做市
python run.py --exchange paradex --market-type perp --symbol BTC-USD-PERP --spread 0.01 --quantity 0.001 --max-orders 2 --target-position 0 --max-position 1 --position-threshold 0.1 --inventory-skew 0 --stop-loss -10 --take-profit 20 --duration 3600 --interval 10

# Paradex Maker-Taker 對沖
python run.py --exchange paradex --market-type perp --symbol BTC-USD-PERP --spread 0.01 --quantity 0.001 --strategy maker_hedge --target-position 0 --max-position 1 --position-threshold 0.1 --duration 3600 --interval 8

```

### 命令行參數

#### 基本參數
- `--api-key`: API 密鑰 (可選，默認使用環境變數)
- `--secret-key`: API 密鑰 (可選，默認使用環境變數；Paradex 使用` Paradex 帳戶`私鑰)
- `--exchange`: 交易所選擇 (支援: `backpack`, `aster`, `paradex`，默認: `backpack`)
- `--ws-proxy`: Websocket 代理 (可選，默認使用環境變數)
- `--cli`: 啟動命令行界面
- `--enable-db`: 啟用資料庫寫入 (預設關閉)
- `--disable-db`: 停用資料庫寫入 (覆寫環境變數設定)

#### 做市參數
- `--symbol`: 交易對 (例如: SOL_USDC)
- `--spread`: 價差百分比 (例如: 0.5)
- `--quantity`: 訂單數量 (可選)
- `--max-orders`: 每側最大訂單數量 (默認: 3)
- `--strategy`: 策略選擇 (`standard` 或 `maker_hedge`，可在現貨與永續中使用)
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
___
### Maker-Taker 對沖策略



#### 啟動永續合約對沖範例

```bash
# 以 BackPack 為例
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.01 --quantity 0.1 --strategy maker_hedge --target-position 0 --max-position 5 --position-threshold 2 --duration 86400 --interval 8
```


**執行情況：**
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

**優勢：**
- 永續合約支持雙向開倉，對沖更靈活
- `reduceOnly` 確保不會意外開新倉位


#### 常見問題

**Q1: 為什麼對沖後還有小額殘量？**

A: 由於交易所的精度限制，對沖數量可能無法完全匹配成交量。策略會記錄這些殘量（通常 < 最小下單量），並在下次成交時一併處理。

**Q2: 對沖失敗會怎樣？**

A: 若市價對沖訂單提交失敗（如餘額不足、API 錯誤），策略會保留完整對沖目標量至下次嘗試。同時會在日誌中記錄錯誤訊息，便於追蹤問題。


**Q3: 對沖策略能用於高波動行情嗎？**

A: 可以，但需注意：
- 高波動可能導致對沖滑點增加
- 建議縮短 `interval` (如 5-10 秒) 以更快反應市場變化


**Q5: Maker-Taker 策略與標準做市的區別？**

| 特性 | 標準做市 | Maker-Taker 對沖 |
|------|----------|-----------------|
| 訂單層數 | 多層 (可配置) | 單層 (買一/賣一) |
| 持倉風險 | 累積持倉 | 即時對沖，接近零持倉 |
| 資金占用率 | 較高 | 較低 |
| 適合場景 | 震盪行情 | 提高交易量 |
| 盈利模式 | 價差 + 持倉升值 | 延遲價差 |

#### 交易所相容性

| 交易所 | 永續對沖 | 特殊處理 |
|--------|---------|----------|
| Backpack | ✓ | 自動啟用 `autoLendRedeem` |
| Aster | ✓ | 移除 `postOnly` (永續) |
| Paradex | ✓ | JWT 自動刷新 |
___
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
# 以 BackPack 為例
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.01 --quantity 0.1 --max-orders 2 --target-position 0 --max-position 5 --position-threshold 2 --inventory-skew 0 --stop-loss -1 --take-profit 5 --duration 999999999 --interval 10
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

___

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

___

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