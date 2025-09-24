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

*2025.09.23 更新：新增多 Aster 交易所支持*
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

```bash

### 永续做市运行示例

#### Backpack 永续做市
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
- `--api-key`: API 密钥 (可选，默认使用环境变数)
- `--secret-key`: API 密钥 (可选，默认使用环境变数)
- `--exchange`: 交易所选择 (默认: backpack)
- `--ws-proxy`: Websocket 代理 (可选，默认使用环境变数)
- `--cli`: 启动命令行界面
- `--panel`: 启动交互式面板

#### 做市参数
- `--symbol`: 交易对 (仅永续，例如: SOL_USDC_PERP)
- `--spread`: 价差百分比 (例如: 0.5)
- `--quantity`: 订单数量 (可选)
- `--max-orders`: 每侧最大订单数量 (默认: 3)
- `--duration`: 运行时间（秒）(默认: 3600)
- `--interval`: 更新间隔（秒）(默认: 60)
- `--market-type`: 市场类型 (`spot` 或 `perp`)
- `--target-position`: 永续合约目标净仓位 (仅 `perp` 模式)
- `--max-position`: 永续合约最大允许净仓 (仅 `perp` 模式)
- `--position-threshold`: 永续仓位调整触发值 (仅 `perp` 模式)
- `--inventory-skew`: 永续做市报价偏移系数 (0-1，仅 `perp` 模式)

#### 重平设置参数
- `--enable-rebalance`: 开启重平功能
- `--disable-rebalance`: 关闭重平功能
- `--base-asset-target`: 基础资产目标比例 (0-100，默认: 30)
- `--rebalance-threshold`: 重平触发阈值 (>0，默认: 15)

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


#### 永续合约参数详解

- `target_position`：**目标持仓量**（绝对值）。设置您希望维持的库存大小，策略会在持仓**超过**此目标时进行减仓，而非主动开仓达到目标。
- `max_position`：**最大持仓量**。仓位的硬性上限，超出后会立即强制平仓，是最高优先级的风控机制。
- `position_threshold`：**仓位调整阈值**。当 `当前持仓 > target_position + threshold` 时触发减仓操作。
- `inventory_skew`：**风险中性系数** (0-1)。根据净仓位自动调整报价：
  - 持有多单时：报价下移，吸引卖单成交
  - 持有空单时：报价上移，吸引买单成交
  - 目标：持续将净仓位推向 `0`，降低方向性风险


## 注意事项

### 一般注意事项
- 交易涉及风险，请谨慎使用
- 建议先在小资金上测试策略效果
- 定期检查交易统计以评估策略表现

### 重平功能注意事项
- **手续费成本**: 重平衡会产生交易手续费，过于频繁的重平衡可能影响整体收益
- **阈值设置**: 过低的阈值可能导致频繁重平衡；过高的阈值可能无法及时控制风险
- **市场环境**: 根据市场波动率调整重平参数，高波动率时建议使用更保守的设置
- **资金效率**: 确保有足够的可用余额或抵押品支持重平衡操作
- **监控建议**: 定期检查重平衡执行情况和效果，根据需要调整参数

### 最佳实践建议

1. **新手用户**: 建议从默认设置开始 (30% 基础资产，15% 阈值)
2. **保守策略**: 使用较低的基础资产比例 (20-25%) 和较低的阈值 (10-12%)
3. **激进策略**: 可以使用较高的基础资产比例 (35-40%) 和较高的阈值 (20-25%)
4. **测试验证**: 先在小资金上测试不同的重平设置，找到最适合的参数组合

## 技术架构

程式采用模组化设计，支援多交易所扩展：

- **Base Client 架构**：抽象基础类别，统一不同交易所的 API 介面
- **精确仓位管理**：只平掉超出阈值的部分，避免过度平仓风控
- **分层日志系统**：市场状态、策略决策、价格计算、执行结果四层资讯
- **相容性设计**：支援多种 API 回应格式，强化错误处理机制