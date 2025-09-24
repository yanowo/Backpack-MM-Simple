# Backpack Exchange 做市交易程序

针对加密货币做市交易的通用架构。该程序提供自动化做市功能，通过维持买卖价差赚取利润。

Twitter：[0xYuCry](https://x.com/0xYuCry)

## 功能特点

- **多交易所架构**：支援 Backpack、未来可扩展至其他交易所
- **自动化做市策略**：智能价差管理和订单调整
- **永续合约做市**：仓位风险管理与风险中性机制
- **智能重平衡系统**：自动维持资产配置比例
- **增强日志系统**：详细的市场状态和策略追踪
- **WebSocket 实时连接**：即时市场数据和订单更新
- **命令行界面**：灵活的参数配置和策略执行
- **交互式面板**：用户友好的操作介面

*2025.09.24 更新：新增多 Aster 交易所支持。*

*2025.09.22 更新：新增多交易所架构和仓位管理优化，增强日志系统提供更清晰的市场状态追踪。*

## 项目结构

```
lemon_trader/
│
├── api/                  # API相关模块
│   ├── __init__.py
│   ├── auth.py           # API认证和签名相关
│   ├── base_client.py    # 抽象基础客户端 (支持继承开发接入任意交易所)
│   ├── bp_client.py      # Backpack Exchange 客户端
│   ├── aster_client.py   # aster 交易所客户端
│   └── websea_client.py  # websea 交易所客户端
│
├── websocket/            # WebSocket模块
│   ├── __init__.py
│   └── client.py         # WebSocket客户端
│
├── database/             # 数据库模块
│   ├── __init__.py
│   └── db.py             # 数据库操作
│
├── strategies/           # 策略模块
│   ├── __init__.py
│   ├── market_maker.py   # Backpack 永续做市策略
│   └── perp_market_maker.py   # 永续合约做市策略
│
├── utils/                # 工具模块
│   ├── __init__.py
│   └── helpers.py        # 辅助函数
│
├── cli/                  # 命令行界面
│   ├── __init__.py
│   └── commands.py       # 命令行命令
│
├── panel/                # 交互式面板
│   ├── __init__.py
│   └── interactive_panel.py  # 交互式面板实现
│
├── config.py             # 配置文件
├── logger.py             # 日志配置
├── run.py                # 统一入口文件
└── README.md             # 说明文档
```

## 环境要求

- Python 3.8 或更高版本
- 所需第三方库：
  - nacl (用于API签名)
  - requests
  - websocket-client
  - numpy
  - python-dotenv

## 安装

1. 克隆或下载此代码库:

```bash
git clone https://github.com/SoYuCry/MarketMakerForCrypto.git
cd MarketMakerForCrypto
```

2. 安装依赖:

```bash
pip install -r requirements.txt
```

3. 设置环境变数:

复制 `.env.example` 为 `.env` 并添加:

```
# Backpack Exchange
BACKPACK_KEY=your_backpack_api_key
BACKPACK_SECRET=your_backpack_secret_key
BACKPACK_PROXY_WEBSOCKET=
BASE_URL=https://api.backpack.work

# Websea Exchange
WEBSEA_TOKEN=your_websea_token
WEBSEA_SECRET=your_websea_secret
WEBSEA_BASE_URL=https://coapi.websea.com

# Aster Exchange
ASTER_API_KEY=your_aster_api_key
ASTER_SECRET_KEY=your_aster_secret_key
```

## 使用方法

#### BackPack 永续做市
```bash
python run.py --exchange backpack --market-type perp --symbol SOL_USDC_PERP --spread 0.05 \
  --quantity 0.1 --max-orders 2 --target-position 0 --max-position 0.5 \
  --position-threshold 0.4 --inventory-skew 0 --duration 999999999 --interval 10
```

#### Aster 永续做市
```bash
python run.py --exchange aster --market-type perp --symbol SOLUSDT --spread 0.05 \
  --quantity 0.1 --max-orders 2 --target-position 0 --max-position 0.5 \
  --position-threshold 0.4 --inventory-skew 0 --duration 999999999 --interval 10
```

### 命令行参数

#### 基本参数
- `--exchange`：交易所选择（目前支持 backpack, aster, websea）
- `--symbol`：交易对（仅永续，例如：SOL_USDC_PERP）
- `--spread`：价差百分比（如：0.05,表示 0.05%）
- `--quantity`：单笔订单数量（可选）
- `--max-orders`：每侧最大挂单数（默认：3）
- `--duration`：运行时间（秒，默认：3600，永久填 999999999999）
- `--interval`：挂单更新间隔（秒，5-15秒为宜）
- `--market-type`：市场类型（`spot` 或 `perp`）
- `--target-position`：永续合约目标净仓位（通常为 0）
- `--max-position`：永续合约最大允许净仓
- `--position-threshold`：永续仓位调整触发值
- `--inventory-skew`：永续做市报价偏移系数（暂时为 0，功能等待更新）



#### 永续合约参数详解

- `target_position`：**目标持仓量**（绝对值）。设置您希望维持的库存大小，策略会在持仓**超过**此目标时进行减仓，而非主动开仓达到目标。
- `max_position`：**最大持仓量**。仓位的硬性上限，超出后会立即强制平仓，是最高优先级的风控机制。
- `position_threshold`：**仓位调整阈值**。当 `当前持仓 > target_position + threshold` 时触发减仓操作。
- `inventory_skew`：**风险中性系数** (0-1)。根据净仓位自动调整报价：
  - 持有多单时：报价下移，吸引卖单成交
  - 持有空单时：报价上移，吸引买单成交
  - 目标：持续将净仓位推向 `0`，降低方向性风险


### 永续合约做市


#### 仓位管理逻辑优化

| 当前仓位 | 目标仓位 | 阈值 | 最大仓位 | 执行动作 |
|---------|---------|------|---------|---------|
| 0.1 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 无操作（在目标范围内） |
| 0.25 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 减仓 0.05 SOL（只平掉超出阈值线的部分） |
| 0.5 SOL | 0 SOL | 0.2 SOL | 0.4 SOL | 风控平仓 0.1 SOL（降到最大仓位限制内） |

#### 增强日志输出范例

```
=== 市场状态 ===
盘口: Bid 239.379 | Ask 239.447 | 价差 0.068 (0.028%)
中间价: 239.408
持仓: 空头 6.000 SOL | 目标: 1.0 | 上限: 1.0

=== 价格计算 ===
原始挂单: 买 238.800 | 卖 239.996
偏移计算: 净持仓 -6.000 | 偏移系数 0.00 | 偏移量 0.0000
调整后挂单: 买 238.800 | 卖 239.996

=== 本次执行总结 ===
成交: 买入 0.200 SOL | 卖出 0.150 SOL
本次盈亏: 2.4500 USDT (手续费: 0.1200)
累计盈亏: 15.2300 USDT | 未实现: -1.8900 USDT
活跃订单: 买 238.800 | 卖 239.996 | 价差 1.196 (0.50%)
```

## 注意事项

### 一般注意事项
- 交易涉及风险，请谨慎使用
- 建议先在小资金上测试策略效果
- 定期检查交易统计以评估策略表现