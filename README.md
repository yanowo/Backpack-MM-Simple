# Backpack Exchange 做市交易程序

這是一個針對 Backpack Exchange 的加密貨幣做市交易程序。該程序提供自動化做市功能，通過維持買賣價差賺取利潤。

Backpack 註冊連結：[https://backpack.exchange/refer/yan](https://backpack.exchange/refer/yan)

Twitter：[Yan Practice ⭕散修](https://x.com/practice_y11)

**使用本程式運行 MM 策略 可獲得 10~30% 返佣**

## 功能特點

- 自動化做市策略
- 基礎價差設置
- **智能重平衡倉位系統**
- **自定義資產配置比例**
- **永續合約做市與倉位風險管理**
- 詳細的交易統計
- WebSocket 實時數據連接
- 命令行界面
- 交互式面板

## 項目結構

```
lemon_trader/
│
├── api/                  # API相關模塊
│   ├── __init__.py
│   ├── auth.py           # API認證和簽名相關
│   └── client.py         # API請求客戶端
│
├── websocket/            # WebSocket模塊
│   ├── __init__.py
│   └── client.py         # WebSocket客戶端
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
├── panel/                # 交互式面板
│   ├── __init__.py
│   └── interactive_panel.py  # 交互式面板實現
│
├── config.py             # 配置文件
├── logger.py             # 日誌配置
├── run.py                # 統一入口文件
└── README.md             # 說明文檔
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

創建 `.env` 文件並添加:

```
API_KEY=your_api_key
SECRET_KEY=your_secret_key
PROXY_WEBSOCKET=http://user:pass@host:port/ 或者 http://host:port (若不需要則留空或移除)
```

## 使用方法

### 統一入口 (推薦)

```bash
# 啟動交互式面板
python run.py --panel

# 啟動命令行界面 (推薦)
python run.py --cli  

# 直接運行做市策略
python run.py --symbol SOL_USDC --spread 0.5 --max-orders 3 --duration 3600 --interval 60
```

### 命令行參數

#### 基本參數
- `--api-key`: API 密鑰 (可選，默認使用環境變數)
- `--secret-key`: API 密鑰 (可選，默認使用環境變數)
- `--ws-proxy`: Websocket 代理 (可選，默認使用環境變數)
- `--cli`: 啟動命令行界面
- `--panel`: 啟動交互式面板

				

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

#### 重平設置參數
- `--enable-rebalance`: 開啟重平功能
- `--disable-rebalance`: 關閉重平功能
- `--base-asset-target`: 基礎資產目標比例 (0-100，默認: 30)
- `--rebalance-threshold`: 重平觸發閾值 (>0，默認: 15)

### 永續合約做市

程式現已支援永續合約做市。永續模式會自動追蹤淨倉位，並透過 Reduce-Only 訂單進行風險控管。

```bash
# 啟動永續做市，維持零淨倉
python run.py --symbol SOL_PERP --spread 0.3 --market-type perp --target-position 0 --max-position 2
```

主要特性：

- `target_position`：設置**持倉量**。這是一個**絕對值**，代表您希望持有的庫存大小（例如 1.5 SOL），無論多空。 策略不會主動開倉達到此目標，而是在持倉**超過**此目標時進行減倉。
- `max_position`：**最大持倉量**。這是倉位的硬性上限，超出後會立即強制平倉至上限以內，是最高優先級的風控。
- `position_threshold`：觸發倉位調整的最小偏離量。當 `(當前持倉 - 設置持倉)` 的差額大於此值，會觸發減倉。
- `inventory_skew`：**風險中性工具**。此參數會根據您當前的**淨倉位**（`net position`）來調整掛單價格。
      - 若您持有多單，報價會自動下移，吸引他人成交您的賣單。
      - 若您持有空單，報價會自動上移，吸引他人成交您的買單。
      - 其核心目標是持續將您的**淨倉位**推向 `0`，以最大限度地降低方向性風險。

## 重平衡功能詳解

### 什麼是重平衡？

重平衡功能用於維持資產配置的平衡，避免因市場波動導致的資產比例失衡。例如，如果您設置基礎資產目標比例為30%，當基礎資產比例因價格波動偏離目標過多時，程序會自動執行交易來恢復平衡。

### 重平設置參數說明

1. **重平功能開關**: 控制是否啟用自動重平衡
2. **基礎資產目標比例**: 基礎資產應佔總資產的百分比 (0-100%)
3. **重平觸發閾值**: 當實際比例偏離目標比例超過此閾值時觸發重平衡

### 重平設置建議

根據不同的市場環境，建議使用不同的重平設置：

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
- 基礎資產目標比例: 30%
- 重平觸發閾值: 15%

**理想配置:**
- 基礎資產價值: 300 USDC (30%)
- 報價資產價值: 700 USDC (70%)

**觸發重平衡的情況:**
- 當基礎資產價值 > 450 USDC (偏差 > 150 USDC = 15%) → 賣出基礎資產
- 當基礎資產價值 < 150 USDC (偏差 > 150 USDC = 15%) → 買入基礎資產
- 在 150-450 USDC 範圍內不會觸發重平衡

## 設定保存

通過面板模式修改的設定會自動保存到 `settings/panel_settings.json` 文件中，下次啟動時會自動加載。
**注意：Panel 暫時不支援設定重平**

## 運行示例

### 基本做市示例

```bash
python run.py --symbol SOL_USDC --spread 0.2 --max-orders 5
```

### 開啟重平功能示例

```bash
# 標準重平設置
python run.py --symbol SOL_USDC --spread 0.2 --enable-rebalance

# 自定義重平設置
python run.py --symbol SOL_USDC --spread 0.2 --enable-rebalance --base-asset-target 25 --rebalance-threshold 12

# 高風險環境設置
python run.py --symbol SOL_USDC --spread 0.3 --enable-rebalance --base-asset-target 20 --rebalance-threshold 10
```

### 關閉重平功能示例

```bash
python run.py --symbol SOL_USDC --spread 0.2 --disable-rebalance
```

### 長時間運行示例

```bash
python run.py --symbol SOL_USDC --spread 0.1 --duration 86400 --interval 120 --enable-rebalance --base-asset-target 30
```

### 完整參數示例

```bash
python run.py --symbol SOL_USDC --spread 0.3 --quantity 0.5 --max-orders 3 --duration 7200 --interval 60 --enable-rebalance --base-asset-target 25 --rebalance-threshold 15
``` 

## 重平功能管理

### 通過 CLI 界面管理

```bash
python run.py --cli
```

在 CLI 界面中選擇：
- `5 - 執行做市策略`: 設置完整的做市和重平參數
- `8 - 重平設置管理`: 查看重平設置說明和測試配置

### 交互式配置

CLI 界面提供交互式重平配置，包括：
- 詳細的參數說明
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

1. **新手用戶**: 建議從默認設置開始 (30% 基礎資產，15% 閾值)
2. **保守策略**: 使用較低的基礎資產比例 (20-25%) 和較低的閾值 (10-12%)
3. **激進策略**: 可以使用較高的基礎資產比例 (35-40%) 和較高的閾值 (20-25%)
4. **測試驗證**: 先在小資金上測試不同的重平設置，找到最適合的參數組合