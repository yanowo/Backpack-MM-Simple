# 磨损计算功能实现说明

## 功能概述

在 TriHedge 策略中集成了实时磨损计算功能，基于仓位均价和数量直接计算，无需通过订单匹配。

## 实现的功能

### 1. 磨损计算时机

- **Enter 阶段结束**：当 accumulating -> holding 时，计算并记录 Enter 阶段的磨损
- **Exit 阶段结束**：当 exiting -> done 时，计算并记录 Exit 阶段的磨损
- **Round 完成**：所有 session 完成后，计算本轮总结并保存

### 2. 磨损计算方式

#### Enter 阶段（建仓）
- **价差计算**：
  - 主账户：买入（Bid），使用仓位均价
  - 对冲账户：卖出（Ask），使用仓位均价
  - 价差 = |对冲均价 - 主账户均价| × 数量
- **手续费计算**：
  - 主账户：maker 费率 0.002%
  - 对冲账户：taker 费率 0.02%
  - 手续费 = 主账户交易额 × 0.002% + 对冲账户交易额 × 0.02%

#### Exit 阶段（平仓）
- **价差计算**：
  - 主账户：卖出（Ask），使用当前市场价格
  - 对冲账户：买入（Bid），使用仓位均价
  - 价差 = |主账户卖出价 - 对冲均价| × 数量
- **手续费计算**：
  - 主账户：maker 费率 0.002%
  - 对冲账户：taker 费率 0.02%

### 3. 数据保存格式

保存到 `tri_hedge_round_stats.json` 文件，格式如下：

```json
[
  {
    "round": 1,
    "enter": {
      "time": {
        "primary_FIL": 1244.0,
        "hedge1_ICP": 1350.0,
        "hedge2_STRK": 1100.0
      },
      "slippage": {
        "primary_FIL": 0.1408,
        "hedge1_ICP": 0.1523,
        "hedge2_STRK": 0.0987
      },
      "fee": {
        "primary_FIL": 0.0753,
        "hedge1_ICP": 0.0812,
        "hedge2_STRK": 0.0654
      },
      "wear": {
        "primary_FIL": 0.2161,
        "hedge1_ICP": 0.2335,
        "hedge2_STRK": 0.1641
      }
    },
    "exit": {
      "time": {
        "primary_FIL": 1553.7,
        "hedge1_ICP": 1620.0,
        "hedge2_STRK": 1480.0
      },
      "slippage": {
        "primary_FIL": 0.1014,
        "hedge1_ICP": 0.1123,
        "hedge2_STRK": 0.0876
      },
      "fee": {
        "primary_FIL": 0.0714,
        "hedge1_ICP": 0.0789,
        "hedge2_STRK": 0.0621
      },
      "wear": {
        "primary_FIL": 0.1728,
        "hedge1_ICP": 0.1912,
        "hedge2_STRK": 0.1497
      }
    },
    "summary": {
      "round_times": {
        "enter": 1350.0,
        "exit": 1620.0,
        "total": 2970.0
      },
      "wears": {
        "enter": 0.6137,
        "exit": 0.5137,
        "total": 1.1274
      },
      "slippage": {
        "enter": 0.3918,
        "exit": 0.3013,
        "total": 0.6931
      },
      "fee": {
        "enter": 0.2219,
        "exit": 0.2124,
        "total": 0.4343
      },
      "totaltime": 2970.0
    }
  }
]
```

### 4. Telegram 通知

#### Enter 阶段通知
当每个标的完成 Enter 阶段时，发送：
```
Enter Phase Complete: primary_FIL
Slippage: 0.140800
Fee: 0.075300
Total Wear: 0.216100
Duration: 1244.0s
```

#### Exit 阶段通知
当每个标的完成 Exit 阶段时，发送：
```
Exit Phase Complete: primary_FIL
Slippage: 0.101400
Fee: 0.071400
Total Wear: 0.172800
Duration: 1553.7s
```

#### Round 总结通知
当所有 session 完成时，发送完整的 Round 总结，包含：
- 每个标的的 Enter/Exit 时间和磨损
- 最大时间（max(所有标的的时间)）
- 总磨损、总价差、总手续费

## 关键实现细节

### 1. 仓位和均价获取

```python
def _get_position_with_price(self, account_idx: int, symbol: str) -> Tuple[float, float]:
    """获取仓位数量和均价"""
    qty = self._get_cached_position(account_idx, symbol)
    price = self._position_price_cache.get(account_idx, {}).get(symbol, 0.0)
    if price <= 0:
        # 如果缓存中没有均价，刷新一次
        self._refresh_position(account_idx, symbol)
        price = self._position_price_cache.get(account_idx, {}).get(symbol, 0.0)
    return qty, price
```

### 2. 时间统计

- **Enter 时间**：从 `enter_start_time` 到 `enter_end_time`
- **Exit 时间**：从 `exit_start_time` 到 `exit_end_time`
- **总时间**：max(所有标的的 Enter 时间) + max(所有标的的 Exit 时间)

### 3. 三个账户并行计算

三个账户同时作为主账户处理不同的标的，磨损计算会：
- 分别计算每个账户-标的组合的磨损
- 在 Round 总结中汇总所有标的的磨损
- 使用 `max()` 计算总时间（因为三个账户并行执行）

## 使用说明

### 配置文件要求

确保配置文件包含 Telegram 配置：
```json
{
  "telegram_bot_token": "your_bot_token",
  "telegram_chat_id": "your_chat_id"
}
```

### 输出文件

- **JSON 文件**：`tri_hedge_round_stats.json`
  - 每次 Round 完成后追加保存
  - 包含所有历史 Round 的统计数据

### 日志输出

策略运行时会输出磨损计算日志：
```
[ENTER WEAR] primary_FIL: slippage=0.140800, fee=0.075300, total=0.216100, duration=1244.0s
[EXIT WEAR] primary_FIL: slippage=0.101400, fee=0.071400, total=0.172800, duration=1553.7s
[ROUND SUMMARY] Round 1: total_wear=1.127400, total_time=2970.0s
```

## 注意事项

1. **手续费费率**：当前使用固定费率（maker 0.002%, taker 0.02%），如需调整，修改 `_maker_fee_rate` 和 `_taker_fee_rate`
2. **仓位均价**：依赖 API 返回的 `entryPrice` 或 `avgEntryPrice` 字段
3. **价差计算**：基于仓位均价，可能与实际订单执行价格有细微差异
4. **时间统计**：使用 `max()` 计算总时间，因为三个账户并行执行

