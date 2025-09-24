"""
永续合约做市策略模块。

此模块在现货做市策略的基础上扩展，提供专为永续合约设计的
开仓、平仓与仓位风险管理功能。
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# 全局函数导入已移除，现在使用客户端方法
from logger import setup_logger
from strategies.market_maker import MarketMaker, format_balance
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("perp_market_maker")


class PerpetualMarketMaker(MarketMaker):
    """专为永续合约设计的做市策略。"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        target_position: float = 0.0,
        max_position: float = 1.0,
        position_threshold: float = 0.1,
        inventory_skew: float = 0.0,
        leverage: float = 1.0,
        ws_proxy: Optional[str] = None,
        exchange: str = 'backpack',
        exchange_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        初始化永续合约做市策略。

        Args:
            api_key (str): API金钥。
            secret_key (str): API私钥。
            symbol (str): 交易对。
            target_position (float): 目标持仓量 (绝对值)，策略会将仓位大小维持在此数值附近。
            max_position (float): 最大允许持仓量（绝对值）。
            position_threshold (float): 触发仓位调整的阈值。
            inventory_skew (float): 库存偏移，影响报价的不对称性，以将净仓位推向0。
            leverage (float): 杠杆倍数。
            ws_proxy (Optional[str]): WebSocket代理地址。
        """
        kwargs.setdefault("enable_rebalance", False)
        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            ws_proxy=ws_proxy,
            exchange=exchange,
            exchange_config=exchange_config,
            **kwargs,
        )

        # 核心修改：target_position 现在是绝对持仓量目标
        self.target_position = abs(target_position)
        self.max_position = max(abs(max_position), self.min_order_size)
        self.position_threshold = max(position_threshold, self.min_order_size)
        self.inventory_skew = max(0.0, min(1.0, inventory_skew))
        self.leverage = max(1.0, leverage)

        self.position_state: Dict[str, Any] = {
            "net": 0.0,
            "avg_entry": 0.0,
            "direction": "FLAT",
            "unrealized": 0.0,
        }

        # 添加总成交量统计（以报价资产计价）
        self.total_volume_quote = 0.0
        self.session_total_volume_quote = 0.0

        logger.info(
            "初始化永续合约做市: %s | 目标持仓量: %s | 最大持仓量: %s | 触发阈值: %s",
            symbol,
            format_balance(self.target_position),
            format_balance(self.max_position),
            format_balance(self.position_threshold),
        )
        self._update_position_state()

    def on_ws_message(self, stream, data):
        """处理WebSocket消息回调 - 添加总成交量统计"""
        # 先调用父类的消息处理
        super().on_ws_message(stream, data)
        
        # 添加总成交量统计
        if stream.startswith("account.orderUpdate."):
            event_type = data.get('e')
            
            if event_type == 'orderFill':
                try:
                    quantity = float(data.get('l', '0'))  # 成交数量
                    price = float(data.get('L', '0'))     # 成交价格
                    
                    # 计算成交额（以报价资产计价）
                    trade_volume = quantity * price
                    
                    # 更新总成交量统计
                    self.total_volume_quote += trade_volume
                    self.session_total_volume_quote += trade_volume
                    
                    logger.debug(f"更新总成交量: +{trade_volume:.2f} {self.quote_asset}, 累计: {self.total_volume_quote:.2f}")
                    
                except Exception as e:
                    logger.error(f"更新总成交量统计时出错: {e}")

    # ------------------------------------------------------------------
    # 基础资讯与工具方法 (此处函数未变动)
    # ------------------------------------------------------------------
    def _fetch_positions(self) -> List[Dict[str, Any]]:
        """从交易所获取仓位列表，统一处理错误与日志。"""
        try:
            result = self.client.get_positions(self.symbol)

            if isinstance(result, dict) and "error" in result:
                error_msg = result["error"]
                if any(code in error_msg for code in ("404", "RESOURCE_NOT_FOUND")):
                    logger.debug("未找到 %s 的仓位记录(404)", self.symbol)
                    return []
                logger.error("查询仓位失败: %s", error_msg)
                return []

            if not isinstance(result, list):
                logger.warning("仓位API返回格式异常: %s", type(result))
                return []

            return result

        except Exception as exc:
            logger.error("获取永续仓位时发生错误: %s", exc)
            logger.warning("使用本地统计作为备用仓位信息")
            return []

    def _aggregate_positions(self, positions: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        if not positions:
            return 0.0, {}

        net_quantity = 0.0
        total_abs_quantity = 0.0
        long_quantity = 0.0
        short_quantity = 0.0
        entry_weight = 0.0
        entry_qty = 0.0
        mark_weight = 0.0
        mark_qty = 0.0
        unrealized = 0.0
        leverage = None

        for pos in positions:
            qty_raw = pos.get("netQuantity", 0)
            try:
                qty = float(qty_raw)
            except (TypeError, ValueError):
                continue

            position_side = (pos.get("positionSide") or pos.get("side") or "").upper()
            if position_side == "LONG":
                net_quantity += abs(qty)
                long_quantity += abs(qty)
            elif position_side == "SHORT":
                net_quantity -= abs(qty)
                short_quantity += abs(qty)
            else:
                net_quantity += qty
                if qty > 0:
                    long_quantity += qty
                elif qty < 0:
                    short_quantity += abs(qty)

            abs_qty = abs(qty)
            if abs_qty == 0:
                continue

            entry_price = pos.get("entryPrice") or pos.get("entry_price")
            mark_price = pos.get("markPrice") or pos.get("mark_price")
            pnl_value = pos.get("pnlUnrealized") if pos.get("pnlUnrealized") is not None else pos.get("unrealizedPnl")

            try:
                if entry_price is not None:
                    entry_weight += float(entry_price) * abs_qty
                    entry_qty += abs_qty
            except (TypeError, ValueError):
                pass

            try:
                if mark_price is not None:
                    mark_weight += float(mark_price) * abs_qty
                    mark_qty += abs_qty
            except (TypeError, ValueError):
                pass

            try:
                if pnl_value is not None:
                    unrealized += float(pnl_value)
            except (TypeError, ValueError):
                pass

            total_abs_quantity += abs_qty
            if leverage is None and pos.get("leverage") is not None:
                leverage = pos.get("leverage")

        avg_entry = entry_weight / entry_qty if entry_qty else 0.0
        avg_mark = mark_weight / mark_qty if mark_qty else None

        aggregated = {
            "symbol": self.symbol,
            "netQuantity": net_quantity,
            "longQuantity": long_quantity,
            "shortQuantity": short_quantity,
            "entryPrice": avg_entry,
            "markPrice": avg_mark,
            "pnlUnrealized": unrealized,
            "unrealizedPnl": unrealized,
            "leverage": leverage,
            "raw": positions,
        }

        return net_quantity, aggregated

    def _get_position_snapshot(self) -> Tuple[float, float, Dict[str, Any]]:
        positions = self._fetch_positions()
        net_quantity, aggregated = self._aggregate_positions(positions)

        long_qty = float(aggregated.get("longQuantity", 0.0)) if aggregated else 0.0
        short_qty = float(aggregated.get("shortQuantity", 0.0)) if aggregated else 0.0
        max_exposure = max(abs(net_quantity), abs(long_qty), abs(short_qty))

        return net_quantity, max_exposure, aggregated

    def get_net_position(self) -> float:
        net_quantity, _, _ = self._get_position_snapshot()
        logger.debug("從API聚合 %s 永續倉位: %s", self.symbol, net_quantity)
        return net_quantity

    def _get_actual_position_info(self) -> Dict[str, Any]:
        _, _, aggregated = self._get_position_snapshot()
        return aggregated

    def _calculate_average_short_entry(self) -> float:
        """计算目前空头仓位的平均开仓价格。"""
        if not self.sell_trades:
            return 0.0

        sell_queue: List[Tuple[float, float]] = self.sell_trades.copy()
        for _, buy_quantity in self.buy_trades:
            remaining_buy = buy_quantity
            while remaining_buy > 0 and sell_queue:
                sell_price, sell_quantity = sell_queue[0]
                matched = min(remaining_buy, sell_quantity)
                remaining_buy -= matched
                if matched >= sell_quantity:
                    sell_queue.pop(0)
                else:
                    sell_queue[0] = (sell_price, sell_quantity - matched)

        unmatched_quantity = sum(quantity for _, quantity in sell_queue)
        if unmatched_quantity <= 0:
            return 0.0

        unmatched_notional = sum(price * quantity for price, quantity in sell_queue)
        return unmatched_notional / unmatched_quantity

    def _update_position_state(self) -> None:
        """更新仓位相关统计。"""
        net = self.get_net_position()  # 现在这会从API获取实际仓位
        current_price = self.get_current_price()
        direction = "FLAT"
        avg_entry = 0.0
        unrealized = 0.0

        # 尝试从API获取更准确的信息
        position_info = self._get_actual_position_info()
        
        if position_info:
            # 使用API返回的精确信息
            avg_entry = float(position_info.get("entryPrice", 0))
            unrealized = float(position_info.get("pnlUnrealized", 0))
        else:
            # 使用本地计算作为备用
            if net > 0:
                avg_entry = self._calculate_average_buy_cost()
                if current_price:
                    unrealized = (current_price - avg_entry) * net
            elif net < 0:
                avg_entry = self._calculate_average_short_entry()
                if current_price:
                    unrealized = (avg_entry - current_price) * abs(net)

        if net > 0:
            direction = "LONG"
        elif net < 0:
            direction = "SHORT"

        self.position_state = {
            "net": net,
            "avg_entry": avg_entry,
            "direction": direction,
            "unrealized": unrealized,
            "target": self.target_position,
            "max_position": self.max_position,
            "threshold": self.position_threshold,
            "inventory_skew": self.inventory_skew,
            "leverage": self.leverage,
            "timestamp": datetime.utcnow().isoformat(),
            "current_price": current_price or 0.0,
        }

    def get_position_state(self) -> Dict[str, Any]:
        """取得仓位资讯快照。"""
        self._update_position_state()
        return self.position_state

    def estimate_profit(self):
        """覆盖父类方法，添加总成交量统计显示"""
        # 先调用父类的方法
        super().estimate_profit()
        
        # 然后添加总成交量统计显示
        logger.info(f"\n---永续合约总成交量统计---")
        logger.info(f"累计总成交量: {self.total_volume_quote:.2f} {self.quote_asset}")
        logger.info(f"本次执行总成交量: {self.session_total_volume_quote:.2f} {self.quote_asset}")

    def run(self, duration_seconds=3600, interval_seconds=60):
        """执行永续合约做市策略"""
        logger.info(f"开始运行永续合约做市策略: {self.symbol}")
        
        # 重置本次执行的总成交量统计
        self.session_total_volume_quote = 0.0
        
        # 调用父类的 run 方法
        super().run(duration_seconds, interval_seconds)

    # ------------------------------------------------------------------
    # 下单相关 (此处函数未变动)
    # ------------------------------------------------------------------
    def open_position(
        self,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "Limit",
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        client_id: Optional[str] = None,
    ) -> Dict:
        """提交开仓或平仓订单。"""

        normalized_order_type = order_type.capitalize()
        if normalized_order_type not in {"Limit", "Market"}:
            raise ValueError("order_type 仅支持 'Limit' 或 'Market'")

        qty = round_to_precision(abs(quantity), self.base_precision)
        if qty < self.min_order_size:
            logger.warning(
                "下单数量 %s 低于最小单位 %s，取消下单",
                format_balance(qty),
                format_balance(self.min_order_size),
            )
            return {"error": "quantity_too_small"}

        order_details: Dict[str, object] = {
            "orderType": normalized_order_type,
            "quantity": str(qty),
            "side": side,
            "symbol": self.symbol,
            "reduceOnly": reduce_only,
        }

        if normalized_order_type == "Limit":
            if price is None:
                raise ValueError("Limit 订单需要提供价格")
            price_value = round_to_tick_size(price, self.tick_size)
            order_details["price"] = str(price_value)
            order_details["timeInForce"] = time_in_force.upper()
        else:
            # 使用当前深度推估价格方便记录
            bid_price, ask_price = self.get_market_depth()
            reference_price = ask_price if side == "Bid" else bid_price
            if reference_price:
                logger.debug(
                    "使用市场价格 %.8f 作为 %s 市价订单参考",
                    reference_price,
                    side,
                )

        if client_id:
            order_details["clientId"] = str(client_id)

        logger.info(
            "提交永续合约订单: %s %s %s | reduceOnly=%s | 类型=%s",
            side,
            format_balance(qty),
            self.symbol,
            reduce_only,
            normalized_order_type,
        )

        result = self.client.execute_order(order_details)
        if isinstance(result, dict) and "error" in result:
            logger.error(f"永续合约订单失败: {result['error']}, 订单信息：{order_details}")
        else:
            self.orders_placed += 1
            logger.info("永续合约订单提交成功: %s", result.get("id", "unknown"))

        return result

    def open_long(
        self,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "Limit",
        reduce_only: bool = False,
        **kwargs,
    ) -> Dict:
        """开启或增加多头仓位。"""
        return self.open_position(
            side="Bid",
            quantity=quantity,
            price=price,
            order_type=order_type,
            reduce_only=reduce_only,
            **kwargs,
        )

    def open_short(
        self,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "Limit",
        reduce_only: bool = False,
        **kwargs,
    ) -> Dict:
        """开启或增加空头仓位。"""
        return self.open_position(
            side="Ask",
            quantity=quantity,
            price=price,
            order_type=order_type,
            reduce_only=reduce_only,
            **kwargs,
        )

    def close_position(
        self,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        order_type: str = "Market",
        side: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> bool:
        """平仓操作。"""
        net = self.get_net_position()  # 使用API获取实际仓位
        if math.isclose(net, 0.0, abs_tol=self.min_order_size / 10):
            logger.info("实际仓位为零，无需平仓")
            return False

        # 根据实际仓位确定平仓方向
        if net > 0:  # 多头仓位，需要卖出平仓
            direction = "long"
            order_side = "Ask"
        else:  # 空头仓位，需要买入平仓
            direction = "short"
            order_side = "Bid"

        # 如果手动指定了side，则使用指定的
        if side is not None:
            direction = side
            order_side = "Ask" if direction == "long" else "Bid"

        long_qty = float(position_info.get("longQuantity", 0.0)) if position_info else 0.0
        short_qty = float(position_info.get("shortQuantity", 0.0)) if position_info else 0.0
        qty_available = short_qty if order_side == "Bid" else long_qty
        if qty_available <= 0:
            qty_available = abs(net)

        qty = qty_available if quantity is None else min(abs(quantity), qty_available)
        qty = round_to_precision(qty, self.base_precision)
        if qty < self.min_order_size:
            logger.info("平仓数量 %s 低于最小单位，忽略", format_balance(qty))
            return False

        logger.info(
            "执行平仓: 实际仓位=%s, 平仓数量=%s, 方向=%s",
            format_balance(net),
            format_balance(qty),
            direction
        )

        result = self.open_position(
            side=order_side,
            quantity=qty,
            price=price,
            order_type=order_type,
            reduce_only=True,
            client_id=client_id,
        )

        if isinstance(result, dict) and "error" in result:
            logger.error(
                "平仓失败: %s | side=%s qty=%s price=%s type=%s reduceOnly=%s clientId=%s result=%s",
                result.get("error"),
                order_side,
                format_balance(qty),
                price if price is not None else "Market",
                order_type,
                True,
                client_id or "",
                result,
            )
            return False

        logger.info("平仓完成，数量 %s", format_balance(qty))
        self._update_position_state()
        return True

    # ------------------------------------------------------------------
    # 仓位管理 (核心修改)
    # ------------------------------------------------------------------
    def need_rebalance(self) -> bool:
        """判断是否需要仓位调整 (仅减仓)。"""
        net = self.get_net_position()
        current_size = abs(net)

        if current_size > self.max_position:
            logger.warning(
                "持仓量 %s 超过最大允许 %s，准备执行风控平仓",
                format_balance(current_size),
                format_balance(self.max_position),
            )
            return True

        # 检查是否超过目标 + 阈值（而非仅仅偏离目标）
        threshold_line = self.target_position + self.position_threshold
        if current_size > threshold_line:
            logger.info(
                "持仓量 %s 超过阈值线 %s (目标 %s + 阈值 %s)，触发减仓调整",
                format_balance(current_size),
                format_balance(threshold_line),
                format_balance(self.target_position),
                format_balance(self.position_threshold)
            )
            return True

        return False

    def manage_positions(self) -> bool:
        """根据持仓量状态主动调整，此函数只负责减仓以控制风险。"""
        net = self.get_net_position()
        current_size = abs(net)
        desired_size = self.target_position

        logger.debug(
            "仓位管理: 实际持仓量=%s, 目标持仓量=%s",
            format_balance(current_size),
            format_balance(desired_size)
        )

        # 1. 风控：检查是否超过最大持仓量
        if current_size > self.max_position:
            short_qty = float(position_info.get("shortQuantity", 0.0)) if position_info else 0.0
            long_qty = float(position_info.get("longQuantity", 0.0)) if position_info else 0.0
            # 以實際方向的倉位作為可用數量
            qty_exposure = short_qty if net < 0 else long_qty
            excess = current_size - self.max_position
            logger.warning(
                "持仓量 %s 超过最大允许 %s，执行紧急平仓 %s",
                format_balance(current_size),
                format_balance(self.max_position),
                format_balance(excess)
            )
            close_amount = excess if excess > 0 else qty_exposure
            return self.close_position(quantity=close_amount, order_type="Market")

        # 2. 减仓：检查持仓量是否超过目标 + 阈值
        # 只有当实际持仓超过 (目标持仓 + 阈值) 时才减仓
        threshold_line = desired_size + self.position_threshold
        if current_size > threshold_line:
            # 只平掉超出阈值线的部分，而不是全部
            qty_to_close = current_size - threshold_line
            logger.info(
                "持仓量 %s 超过阈值线 %s (目标 %s + 阈值 %s)，执行减仓 %s",
                format_balance(current_size),
                format_balance(threshold_line),
                format_balance(desired_size),
                format_balance(self.position_threshold),
                format_balance(qty_to_close)
            )
            # close_position 会根据净仓位自动判断平仓方向
            return self.close_position(quantity=qty_to_close, order_type="Market")

        logger.debug("持仓量在目标范围内，无需主动管理。")
        return False

    def rebalance_position(self) -> None:
        """覆写现货逻辑，改为永续仓位管理。"""
        logger.info("执行永续仓位管理")
        acted = self.manage_positions()
        if not acted:
            logger.info("仓位已在安全范围内，无需调整")

    # ------------------------------------------------------------------
    # 报价调整 (核心修改)
    # ------------------------------------------------------------------
    def calculate_prices(self):  # type: ignore[override]
        """计算买卖订单价格，并根据净仓位进行偏移以控制方向风险。"""
        buy_prices, sell_prices = super().calculate_prices()
        if not buy_prices or not sell_prices:
            return buy_prices, sell_prices

        net = self.get_net_position()
        current_price = self.get_current_price()
        
        # 获取盘口信息
        orderbook = self.client.get_order_book(self.symbol)
        best_bid = best_ask = None
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            if orderbook['bids']:
                best_bid = float(orderbook['bids'][0][0])
            if orderbook['asks']:
                best_ask = float(orderbook['asks'][0][0])
        
        # 输出盘口和持仓信息
        logger.info("=== 市场状态 ===")
        if best_bid and best_ask:
            spread = best_ask - best_bid
            spread_pct = (spread / current_price * 100) if current_price else 0
            logger.info(f"盘口: Bid {best_bid:.3f} | Ask {best_ask:.3f} | 价差 {spread:.3f} ({spread_pct:.3f}%)")
        if current_price:
            logger.info(f"中间价: {current_price:.3f}")
        
        # 输出持仓信息
        direction = "空头" if net < 0 else "多头" if net > 0 else "无仓位"
        logger.info(f"持仓: {direction} {abs(net):.3f} SOL | 目标: {self.target_position:.1f} | 上限: {self.max_position:.1f}")

        # 如果没有库存偏移系数或没有仓位，则不进行调整
        if self.inventory_skew <= 0 or abs(net) < self.min_order_size:
            logger.info(f"原始挂单: 买 {buy_prices[0]:.3f} | 卖 {sell_prices[0]:.3f} (无偏移)")
            return buy_prices, sell_prices

        if self.max_position <= 0:
            return buy_prices, sell_prices

        # 核心偏移逻辑：根据净仓位(net)调整报价，目标是将净仓位推向0 (Delta中性)
        # 偏离量就是净仓位本身
        deviation = net
        skew_ratio = max(-1.0, min(1.0, deviation / self.max_position))

        if not current_price:
            return buy_prices, sell_prices

        # 如果是多头 (net > 0)，skew_offset为正；如果是空头 (net < 0)，skew_offset为负
        skew_offset = current_price * self.inventory_skew * skew_ratio

        # 调整价格以鼓励反向交易，使净仓位回归0
        # 如果是多头 (net > 0)，降低买卖价以鼓励市场吃掉我们的卖单，同时降低我们买入的意愿
        # 如果是空头 (net < 0)，提高买卖价以鼓励市场吃掉我们的买单，同时降低我们卖出的意愿
        adjusted_buys = [
            round_to_tick_size(price - skew_offset, self.tick_size)
            for price in buy_prices
        ]
        adjusted_sells = [
            round_to_tick_size(price - skew_offset, self.tick_size)
            for price in sell_prices
        ]

        # 输出价格调整详情
        logger.info("=== 价格计算 ===")
        logger.info(f"原始挂单: 买 {buy_prices[0]:.3f} | 卖 {sell_prices[0]:.3f}")
        logger.info(f"偏移计算: 净持仓 {net:.3f} | 偏移系数 {self.inventory_skew:.2f} | 偏移量 {skew_offset:.4f}")
        logger.info(f"调整后挂单: 买 {adjusted_buys[0]:.3f} | 卖 {adjusted_sells[0]:.3f}")

        # 风控：确保调整后买卖价没有交叉
        if adjusted_buys[0] >= adjusted_sells[0]:
            logger.warning("报价调整后买卖价交叉或价差过小，恢复原始报价。买: %s, 卖: %s", adjusted_buys[0], adjusted_sells[0])
            return buy_prices, sell_prices

        return adjusted_buys, adjusted_sells

    # ------------------------------------------------------------------
    # 其他辅助方法 (核心修改)
    # ------------------------------------------------------------------
    def set_target_position(self, target: float, threshold: Optional[float] = None) -> None:
        """更新目标持仓量 (绝对值) 及触发阈值。"""
        self.target_position = abs(target)
        if threshold is not None:
            self.position_threshold = max(threshold, self.min_order_size)
        logger.info(
            "更新目标持仓量: %s (阈值: %s)",
            format_balance(self.target_position),
            format_balance(self.position_threshold),
        )
        self._update_position_state()

    def set_max_position(self, max_position: float) -> None:
        """更新最大允许仓位。"""
        self.max_position = max(abs(max_position), self.min_order_size)
        logger.info(
            "更新最大仓位限制: %s",
            format_balance(self.max_position),
        )
        self._update_position_state()
