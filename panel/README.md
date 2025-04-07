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

- `help`: 顯示幫助信息
- `symbols`: 列出可用交易對
- `start <symbol>`: 啟動指定交易對的做市策略
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