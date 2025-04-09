# Backpack Exchange 做市交易程序

這是一個針對 Backpack Exchange 的加密貨幣做市交易程序。該程序提供自動化做市功能，通過維持買賣價差賺取利潤。

Backpack 註冊連結：[https://backpack.exchange/refer/yan](https://backpack.exchange/refer/yan)

Twitter：[Yan Practice ⭕散修](https://x.com/practice_y11)

## 功能特點

- 自動化做市策略
- 基礎價差設置
- 自動重平衡倉位
- 詳細的交易統計
- WebSocket 實時數據連接
- 命令行界面

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
│   └── market_maker.py   # 做市策略
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
├── main.py               # 主執行文件
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
```

## 使用方法

### 統一入口 (推薦)

```bash
# 啟動交互式面板
python run.py --panel

# 啟動命令行界面
python run.py --cli  

# 直接運行做市策略
python run.py --symbol SOL_USDC --spread 0.1
```

### 命令行界面

啟動命令行界面:

```bash
python main.py --cli
```

### 直接執行做市策略

```bash
python main.py --symbol SOL_USDC --spread 0.5 --max-orders 3 --duration 3600 --interval 60
```

### 命令行參數

- `--api-key`: API 密鑰 (可選，默認使用環境變數)
- `--secret-key`: API 密鑰 (可選，默認使用環境變數)
- `--ws-proxy`: Websocket 代理 (可選，默認使用環境變數)
- `--cli`: 啟動命令行界面
- `--panel`: 啟動交互式面板

### 做市參數

- `--symbol`: 交易對 (例如: SOL_USDC)
- `--spread`: 價差百分比 (例如: 0.5)
- `--quantity`: 訂單數量 (可選)
- `--max-orders`: 每側最大訂單數量 (默認: 3)
- `--duration`: 運行時間（秒）(默認: 3600)
- `--interval`: 更新間隔（秒）(默認: 60)

## 設定保存

通過面板模式修改的設定會自動保存到 `settings/panel_settings.json` 文件中，下次啟動時會自動加載。

## 運行示例

### 基本做市示例

```bash
python run.py --symbol SOL_USDC --spread 0.2 --max-orders 5
```

### 長時間運行示例

```bash
python run.py --symbol SOL_USDC --spread 0.1 --duration 86400 --interval 120
```

### 完整參數示例

```bash
python run.py --symbol SOL_USDC --spread 0.3 --quantity 0.5 --max-orders 3 --duration 7200 --interval 60
``` 

## 注意事項

- 交易涉及風險，請謹慎使用
- 建議先在小資金上測試策略效果
- 定期檢查交易統計以評估策略表現
