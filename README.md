# Tri Hedge Hold 策略（Lighter-only）

`tri_hedge_hold_strategy_lighter.py` 实现了一个三账户协同的做市/持仓策略，主账号负责累积多头仓位，两个对冲账号即时抵消敞口，最终在持有一段时间后分批平仓。策略核心关注 Lighter 的积分激励：在交易量、持仓时长、标的流动性和 Maker/Taker 比例之间寻找最划算的组合，从而在尽量小的磨损下获取更多积分。

## 周期流程概览

1. **启动周期**：为当前标的挑选一个主账户（Maker）和两个对冲账户（Hedger），加载该标的的市场限制和下单计划。
2. **累积仓位**：主账户按 `slice_notional` 逐笔挂买单，直到达到 `target_notional` 对应的目标仓位；每个切片成交都会触发 hedger 反向做空，尽量把整体净敞口控在接近零的位置。
3. **持有仓位**：建仓完成后，主账户保持多头头寸 `hold_minutes`，期间只跟踪仓位和对冲情况。
4. **缓慢平仓**：持仓时间结束，主账户按切片 reduce-only 卖出，同时 hedger 逐步回补空单，直到剩余仓位低于交易所最小下单量为止。
5. **轮换**：本轮完成后切换到下一个标的，并把主账户顺延到下一位账号，持续循环。

> 由于 Lighter 的最小成交金额 10u，末尾零星仓位可能无法完全清零。一旦某次切片部分成交、hedger 下单量不足最小金额，也会短暂出现 ≤11 USDC 的净敞口，待下一个切片填补后会立即修正。

## 策略特性

- **三账户协同**：主账户负责 maker 下单，两位 hedger 负责同步对冲，`random_split_range` 控制对冲数量在两者间的随机分配。
- **动态切片**：根据目标名义金额和实时参考价自动计算下单数量，确保每笔交易都满足 Lighter 的最小报价与基准数量要求。
- **周期轮换**：主账户和标的按顺序轮转，均衡不同账户的负载。
- **保证金风控**：当任意账户收到 “not enough margin / collateral” 保证金不足拒单时，风控会立刻触发 failsafe，强制平掉当前标的后终止策略，防止单腿风险。

## 环境要求

1. 三个可用的 Lighter 账户（一个 Maker、两个 Hedger），配置中至少提供 `api_private_key` 与 `account_index` 或 `account_address`。
2. 已部署的 Lighter signer 动态库（默认搜索 `api/signers/`、`Signer/Lighter/`，也可以用 `signer_lib_dir` 指向自定义路径）。
3. 完成 `requirements.txt` 依赖安装，并能访问配置中的 `base_url`。

## 配置文件

策略使用 JSON（如 `settings/tri_hedge_strategy.json`）加载运行参数：

```jsonc
{
  "base_url": "https://mainnet.zklighter.elliot.ai",
  "hold_minutes": 21,
  "entry_price_offset_bps": 0,
  "exit_price_offset_bps": 0,
  "slice_delay_seconds": 3,
  "slice_delay_jitter_seconds": 4,
  "slice_fill_timeout": 8,
  "order_poll_interval": 2,
  "pause_between_symbols": 15,
  "random_split_range": [0.45, 0.55],
  "primary_time_in_force": "GTC",
  "default_target_notional": 5000,
  "default_slice_count": 50,
  "coinlist": [
    {"symbol": "BTC", "target_notional": 800, "slice_notional": 150}
  ],
  "accounts": [
    {"label": "Maker", "api_private_key": "...", "account_address": "..."},
    {"label": "Hedge-A", "api_private_key": "...", "account_address": "..."},
    {"label": "Hedge-B", "api_private_key": "...", "account_address": "..."}
  ]
}
```

### 主要参数说明

- `hold_minutes`：每个标的建仓后持有多头仓位的时间（分钟）。
- `entry_price_offset_bps` / `exit_price_offset_bps`：建仓/平仓时在买一卖一附近加减的基点偏移，控制挂单与盘口距离。
- `slice_delay_seconds` / `slice_delay_jitter_seconds`：切片下单之间的基础间隔与随机抖动，避免频率过高。
- `slice_fill_timeout`：等待单笔切片成交的最长时间，超时会尝试取消并重发下一笔。
- `order_poll_interval`：轮询仓位变化以判断成交的时间间隔。
- `pause_between_symbols`：完成一轮后在切换下一个标的前的休眠时间。
- `random_split_range`：对冲数量分配给两个 hedger 的比例区间，例如 `[0.45, 0.55]` 表示第一位分得 45%~55%。
- `primary_time_in_force`：主账户下单的 TIF（如 `GTC`、`IOC`、`FOK`、`PO`）。
- `default_target_notional`：若单个标的未指定 `target_notional`，使用此默认名义金额（USDC 计价）。
- `default_slice_count`：默认把目标名义金额等分成多少个切片，用于推导 `slice_notional`。
- `coinlist`：目标标的列表，可自定义 `target_notional`、`slice_notional`、`slice_count`、`hold_minutes` 等；字段名称也可写作 `symbols` 或 `coins`。
- `accounts`：三组账户配置，可选字段还包括 `api_key_index`、`base_url`、`chain_id`、`signer_lib_dir` 等。

> 设置 `run_once: true` 可在跑完整个标的列表后退出。

## 启动方式

指定配置路径或设置 `TRI_HEDGE_CONFIG` 后执行：

```bash
python run.py \
  --strategy tri_hedge \
  --exchange lighter \
  --strategy-config settings/tri_hedge_strategy.json
```

日志默认写入 `market_maker.log`，包含周期起止、成交统计、hedge 分配以及风控触发情况。

## 积分与磨损考量

> 规则并未公开，但根据 2025/11 多次实盘复盘，总结出以下经验：

- **持仓时间阈值**：秒开秒平基本没有积分，而持仓 20 分钟与 1 小时差别不大，说明积分在 15~20 分钟左右达到“饱和”。`hold_minutes` 建议设在该区间即可。
- **优先小币种**：流动性好的 BTC/ETH 约 50 万 USDC 才能换 1 积分，STRK 等小币只需 5 万 USDC。同样磨损下优先选择低流动性的标的。
- **Maker/Taker 比例**：交易所希望你提供流动性，过多市价单会被视为“刷量”并惩罚。所以策略让一个账户纯 Maker 开多，两个账户用小额随机拆分的市价空单对冲，保证 Maker 成交始终占主导。
- **多笔小额随机成交**：`random_split_range` 和切片机制制造更多自然成交，减少两个对冲账户数量完全一致的“女巫”特征，也能降低滑点。


## 风险与监控

- **保证金风控**：出现保证金不足拒单时，策略会自动触发 failsafe，强制平掉当前标的并停止，需补充保证金或降低目标仓位后再重启。
- **仓位快照**：日志里的 `Positions[symbol]: ...` 记录可快速检查三账户的净仓位，确认是否达成预期。
- **单账户测试**：上线前可用 `manual_margin_test.py` 单独验证账户的保证金头寸，确保下单不会立即触发 failsafe。
- **账户轮换**：三个账户可以轮流担任 Maker，保持更健康的 Maker/Taker 比例，同时还能错开单一账户的资金压力。

## 常见问题

- **配置缺失**：JSON 中缺少必填字段会抛出 `StrategyConfigError`，根据提示补齐即可。
- **Signer nonce 错误**：策略会自动刷新一次 nonce 再重试；若仍失败，多半是账户索引或 API Key 不匹配，需要重新检查配置。
- **最小成交量限制**：若日志频繁提示数量低于最小下单量，可增大 `slice_notional` 或减少 `default_slice_count`，让单笔切片金额 ≥ 10~11 USDC。

请勿将私钥、账户索引等敏感信息提交到版本库；运行时务必引用本地受控的配置文件。
