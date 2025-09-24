# 交互式命令面板模组

这个模组提供了一个交互式的命令面板，用于控制和监控做市策略。

## 功能特点

- 直观的图形界面显示市场资料
- 命令行操作控制策略
- 实时数据更新
- 设定保存和加载
- 跨平台键盘输入处理

## 文件说明

- `interactive_panel.py`: 核心面板界面和功能
- `key_handler.py`: 跨平台键盘输入处理
- `panel_main.py`: 面板独立运行入口
- `settings.py`: 设定的保存和加载

## 使用方法

### 直接运行

```bash
python panel/panel_main.py
```

### 透过统一入口运行

```bash
python run.py --panel
```

### 命令行参数

```bash
python panel/panel_main.py --api-key YOUR_API_KEY --secret-key YOUR_SECRET_KEY --symbol SOL_USDC
```

## 设定文件

设定自动保存在 `settings/panel_settings.json` 文件中，修改设定后会自动保存。

## 键盘操作

- 按 `:` 或 `/` 进入命令模式
- 命令模式下按 `Enter` 执行命令，按 `ESC` 取消
- 按 `q` 退出程序

## 可用命令

### 基本命令
- `help`: 显示帮助信息
- `symbols`: 列出可用交易对
- `start <symbol>`: 启动指定交易对的做市策略
  - 现货示例: `start SOL_USDC`
  - 永续示例: `start SOL_USDC_PERP` (需先设置 market_type perp)
- `stop`: 停止当前策略
- `params`: 显示当前策略参数
- `set spread <值>`: 设置价差百分比
- `set max_orders <值>`: 设置每侧最大订单数
- `set quantity <值>`: 设置订单数量
- `set interval <值>`: 设置更新间隔（秒）
- `status`: 显示当前状态
- `balance`: 查询余额
- `orders`: 显示活跃订单
- `cancel`: 取消所有订单
- `clear`: 清除日志
- `exit`/`quit`: 退出程序

### 参数设置命令

#### 基本参数
- `set base_spread <值>`: 设置价差百分比 (例: `set base_spread 0.2` = 0.2%)
- `set order_quantity <值>`: 设置订单数量 (例: `set order_quantity 0.5` 或 `set order_quantity auto`)
- `set max_orders <值>`: 设置每侧最大订单数 (例: `set max_orders 5`)
- `set interval <值>`: 设置更新间隔（秒）
- `set market_type <值>`: 设置市场类型
  - `set market_type spot`: 现货模式
  - `set market_type perp`: 永续合约模式

#### 永续合约专用参数 (需先设置 `market_type perp`)
- `set target_position <值>`: 目标净仓位 (例: `set target_position 0.0` = 中性仓位)
- `set max_position <值>`: 最大仓位限制 (例: `set max_position 1.0`)
- `set position_threshold <值>`: 仓位调整触发值 (例: `set position_threshold 0.1`)
- `set inventory_skew <值>`: 报价偏移系数 (例: `set inventory_skew 0.25`)

## 使用流程

### 现货交易
1. `symbols` - 查看可用现货交易对
2. `set base_spread 0.1` - 设置价差
3. `set order_quantity 0.5` - 设置订单数量
4. `start SOL_USDC` - 启动现货做市

### 永续合约交易
1. `set market_type perp` - 切换到永续合约模式
2. `symbols` - 查看可用永续合约交易对
3. `set target_position 0.0` - 设置目标中性仓位
4. `set max_position 1.0` - 设置最大仓位限制
5. `start SOL_USDC_PERP` - 启动永续合约做市 