# 交互式命令面板模組

這個模組提供了一個交互式的命令面板，用於控制和監控做市策略。

## 功能特點

- 直觀的圖形界面顯示市場資料
- 命令行操作控制策略
- 實時數據更新
- 設定保存和加載
- 跨平台鍵盤輸入處理

## 文件說明

- `interactive_panel.py`: 核心面板界面和功能
- `key_handler.py`: 跨平台鍵盤輸入處理
- `panel_main.py`: 面板獨立運行入口
- `settings.py`: 設定的保存和加載

## 使用方法

### 直接運行

```bash
python panel/panel_main.py
```

### 透過統一入口運行

```bash
python run.py --panel
```

### 命令行參數

```bash
python panel/panel_main.py --api-key YOUR_API_KEY --secret-key YOUR_SECRET_KEY --symbol SOL_USDC
```

## 設定文件

設定自動保存在 `settings/panel_settings.json` 文件中，修改設定後會自動保存。

## 鍵盤操作

- 按 `:` 或 `/` 進入命令模式
- 命令模式下按 `Enter` 執行命令，按 `ESC` 取消
- 按 `q` 退出程序

## 可用命令

### 基本命令
- `help`: 顯示幫助信息
- `symbols`: 列出可用交易對
- `start <symbol>`: 啟動指定交易對的做市策略
  - 現貨示例: `start SOL_USDC`
  - 永續示例: `start SOL_USDC_PERP` (需先設置 market_type perp)
- `stop`: 停止當前策略
- `params`: 顯示當前策略參數
- `set spread <值>`: 設置價差百分比
- `set max_orders <值>`: 設置每側最大訂單數
- `set quantity <值>`: 設置訂單數量
- `set interval <值>`: 設置更新間隔（秒）
- `status`: 顯示當前狀態
- `balance`: 查詢餘額
- `orders`: 顯示活躍訂單
- `cancel`: 取消所有訂單
- `clear`: 清除日誌
- `exit`/`quit`: 退出程序

### 參數設置命令

#### 基本參數
- `set base_spread <值>`: 設置價差百分比 (例: `set base_spread 0.2` = 0.2%)
- `set order_quantity <值>`: 設置訂單數量 (例: `set order_quantity 0.5` 或 `set order_quantity auto`)
- `set max_orders <值>`: 設置每側最大訂單數 (例: `set max_orders 5`)
- `set interval <值>`: 設置更新間隔（秒）
- `set market_type <值>`: 設置市場類型
  - `set market_type spot`: 現貨模式
  - `set market_type perp`: 永續合約模式

#### 永續合約專用參數 (需先設置 `market_type perp`)
- `set target_position <值>`: 目標淨倉位 (例: `set target_position 0.0` = 中性倉位)
- `set max_position <值>`: 最大倉位限制 (例: `set max_position 1.0`)
- `set position_threshold <值>`: 倉位調整觸發值 (例: `set position_threshold 0.1`)
- `set inventory_skew <值>`: 報價偏移係數 (例: `set inventory_skew 0.25`)

## 使用流程

### 現貨交易
1. `symbols` - 查看可用現貨交易對
2. `set base_spread 0.1` - 設置價差
3. `set order_quantity 0.5` - 設置訂單數量
4. `start SOL_USDC` - 啟動現貨做市

### 永續合約交易
1. `set market_type perp` - 切換到永續合約模式
2. `symbols` - 查看可用永續合約交易對
3. `set target_position 0.0` - 設置目標中性倉位
4. `set max_position 1.0` - 設置最大倉位限制
5. `start SOL_USDC_PERP` - 啟動永續合約做市 