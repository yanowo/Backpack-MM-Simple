"""
永續合約做市策略模塊。

此模塊在現貨做市策略的基礎上擴展，提供專為永續合約設計的
開倉、平倉與倉位風險管理功能。
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from api.client import execute_order
from logger import setup_logger
from strategies.market_maker import MarketMaker, format_balance
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("perp_market_maker")


class PerpetualMarketMaker(MarketMaker):
    """專為永續合約設計的做市策略。"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        target_position: float = 0.0,
        max_position: float = 1.0,
        position_threshold: float = 0.1,
        inventory_skew: float = 0.25,
        leverage: float = 1.0,
        ws_proxy: Optional[str] = None,
        **kwargs,
    ) -> None:
        """初始化永續合約做市策略。"""

        kwargs.setdefault("enable_rebalance", False)
        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            ws_proxy=ws_proxy,
            **kwargs,
        )

        self.target_position = target_position
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

        logger.info(
            "初始化永續合約做市: %s | 目標倉位: %s | 最大倉位: %s | 觸發閾值: %s",
            symbol,
            format_balance(self.target_position),
            format_balance(self.max_position),
            format_balance(self.position_threshold),
        )
        self._update_position_state()

    # ------------------------------------------------------------------
    # 基礎資訊與工具方法
    # ------------------------------------------------------------------
    def get_net_position(self) -> float:
        """取得目前的淨倉位。"""
        return self.total_bought - self.total_sold

    def _calculate_average_short_entry(self) -> float:
        """計算目前空頭倉位的平均開倉價格。"""
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
        """更新倉位相關統計。"""
        net = self.get_net_position()
        current_price = self.get_current_price()
        direction = "FLAT"
        avg_entry = 0.0
        unrealized = 0.0

        if net > 0:
            direction = "LONG"
            avg_entry = self._calculate_average_buy_cost()
            if current_price:
                unrealized = (current_price - avg_entry) * net
        elif net < 0:
            direction = "SHORT"
            avg_entry = self._calculate_average_short_entry()
            if current_price:
                unrealized = (avg_entry - current_price) * abs(net)

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
        """取得倉位資訊快照。"""
        self._update_position_state()
        return self.position_state

    # ------------------------------------------------------------------
    # 下單相關
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
        """提交開倉或平倉訂單。"""

        normalized_order_type = order_type.capitalize()
        if normalized_order_type not in {"Limit", "Market"}:
            raise ValueError("order_type 僅支援 'Limit' 或 'Market'")

        qty = round_to_precision(abs(quantity), self.base_precision)
        if qty < self.min_order_size:
            logger.warning(
                "下單數量 %s 低於最小單位 %s，取消下單",
                format_balance(qty),
                format_balance(self.min_order_size),
            )
            return {"error": "quantity_too_small"}

        order_details: Dict[str, object] = {
            "orderType": normalized_order_type,
            "quantity": str(qty),
            "side": side,
            "symbol": self.symbol,
            "timeInForce": time_in_force if normalized_order_type == "Limit" else "IOC",
            "reduceOnly": reduce_only,
        }

        if normalized_order_type == "Limit":
            if price is None:
                raise ValueError("Limit 訂單需要提供價格")
            price_value = round_to_tick_size(price, self.tick_size)
            order_details["price"] = str(price_value)
        else:
            # 使用當前深度推估價格方便記錄
            bid_price, ask_price = self.get_market_depth()
            reference_price = ask_price if side == "Bid" else bid_price
            if reference_price:
                logger.debug(
                    "使用市場價格 %.8f 作為 %s 市價訂單參考",
                    reference_price,
                    side,
                )

        if client_id:
            order_details["clientId"] = str(client_id)

        logger.info(
            "提交永續合約訂單: %s %s %s | reduceOnly=%s | 類型=%s",
            side,
            format_balance(qty),
            self.symbol,
            reduce_only,
            normalized_order_type,
        )

        result = execute_order(self.api_key, self.secret_key, order_details)
        if isinstance(result, dict) and "error" in result:
            logger.error("永續合約訂單失敗: %s", result["error"])
        else:
            self.orders_placed += 1
            logger.info("永續合約訂單提交成功: %s", result.get("id", "unknown"))

        return result

    def open_long(
        self,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "Limit",
        reduce_only: bool = False,
        **kwargs,
    ) -> Dict:
        """開啟或增加多頭倉位。"""
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
        """開啟或增加空頭倉位。"""
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
        """平倉操作。"""
        net = self.get_net_position()
        if math.isclose(net, 0.0, abs_tol=self.min_order_size / 10):
            logger.info("倉位已經為零，無需平倉")
            return False

        direction = side
        if direction is None:
            direction = "long" if net > 0 else "short"

        qty_available = abs(net)
        qty = qty_available if quantity is None else min(abs(quantity), qty_available)
        qty = round_to_precision(qty, self.base_precision)
        if qty < self.min_order_size:
            logger.info("平倉數量 %s 低於最小單位，忽略", format_balance(qty))
            return False

        order_side = "Ask" if direction == "long" else "Bid"
        result = self.open_position(
            side=order_side,
            quantity=qty,
            price=price,
            order_type=order_type,
            reduce_only=True,
            client_id=client_id,
        )

        if isinstance(result, dict) and "error" in result:
            logger.error("平倉失敗: %s", result["error"])
            return False

        logger.info("平倉完成，數量 %s", format_balance(qty))
        self._update_position_state()
        return True

    # ------------------------------------------------------------------
    # 倉位管理
    # ------------------------------------------------------------------
    def need_rebalance(self) -> bool:
        """判斷是否需要倉位調整。"""
        net = self.get_net_position()
        deviation = abs(net - self.target_position)

        if abs(net) > self.max_position:
            logger.warning(
                "淨倉位 %s 超過最大允許 %s，準備執行風控平倉",
                format_balance(net),
                format_balance(self.max_position),
            )
            return True

        if deviation >= self.position_threshold:
            logger.info(
                "倉位偏離目標 %.8f，觸發調整", deviation
            )
            return True

        return False

    def manage_positions(self) -> bool:
        """根據倉位狀態主動調整。"""
        net = self.get_net_position()
        desired = self.target_position
        deviation = net - desired

        if abs(net) > self.max_position:
            excess = abs(net) - self.max_position
            logger.info(
                "淨倉位超限 %s，執行緊急平倉",
                format_balance(excess),
            )
            return self.close_position(quantity=excess, order_type="Market")

        if deviation > self.position_threshold:
            if net >= 0:
                qty = min(deviation, net)
                return self.close_position(quantity=qty, side="long")
            max_extra = max(0.0, self.max_position - abs(net))
            qty = min(deviation, max_extra)
            if qty >= self.min_order_size:
                self.open_short(qty, order_type="Market")
                self._update_position_state()
                return True
            return False

        if deviation < -self.position_threshold:
            if net <= 0:
                qty = min(abs(deviation), abs(net))
                return self.close_position(quantity=qty, side="short")
            max_extra = max(0.0, self.max_position - net)
            qty = min(abs(deviation), max_extra)
            if qty >= self.min_order_size:
                self.open_long(qty, order_type="Market")
                self._update_position_state()
                return True
            return False

        # 若目標倉位不為零且目前倉位低於目標，主動開倉
        net_abs = abs(net)
        target_abs = abs(desired)
        if target_abs >= self.min_order_size and net_abs + self.min_order_size <= self.max_position:
            if net_abs < target_abs and abs(deviation) >= self.position_threshold:
                qty_to_open = min(target_abs - net_abs, self.max_position - net_abs)
                if qty_to_open >= self.min_order_size:
                    if desired > 0:
                        self.open_long(qty_to_open, order_type="Market")
                    elif desired < 0:
                        self.open_short(qty_to_open, order_type="Market")
                    self._update_position_state()
                    return True

        return False

    def rebalance_position(self) -> None:
        """覆寫現貨邏輯，改為永續倉位管理。"""
        logger.info("執行永續倉位管理")
        acted = self.manage_positions()
        if not acted:
            logger.info("倉位已在安全範圍內，無需調整")

    # ------------------------------------------------------------------
    # 報價調整
    # ------------------------------------------------------------------
    def calculate_prices(self):  # type: ignore[override]
        buy_prices, sell_prices = super().calculate_prices()
        if not buy_prices or not sell_prices:
            return buy_prices, sell_prices

        net = self.get_net_position()
        desired = self.target_position
        deviation = net - desired

        if self.max_position <= 0:
            return buy_prices, sell_prices

        skew_ratio = max(-1.0, min(1.0, deviation / self.max_position))
        if abs(skew_ratio) < 1e-6 or self.inventory_skew <= 0:
            return buy_prices, sell_prices

        current_price = self.get_current_price()
        if not current_price:
            return buy_prices, sell_prices

        skew_offset = current_price * self.inventory_skew * skew_ratio
        adjusted_buys = [
            round_to_tick_size(price - skew_offset, self.tick_size)
            for price in buy_prices
        ]
        adjusted_sells = [
            round_to_tick_size(price + skew_offset, self.tick_size)
            for price in sell_prices
        ]

        logger.debug(
            "倉位偏移 %.6f，調整買價 %s -> %s，賣價 %s -> %s",
            deviation,
            buy_prices[0],
            adjusted_buys[0],
            sell_prices[0],
            adjusted_sells[0],
        )

        return adjusted_buys, adjusted_sells

    # ------------------------------------------------------------------
    # 其他輔助方法
    # ------------------------------------------------------------------
    def set_target_position(self, target: float, threshold: Optional[float] = None) -> None:
        """更新目標淨倉位及觸發閾值。"""
        self.target_position = target
        if threshold is not None:
            self.position_threshold = max(threshold, self.min_order_size)
        logger.info(
            "更新目標倉位: %s (閾值: %s)",
            format_balance(self.target_position),
            format_balance(self.position_threshold),
        )
        self._update_position_state()

    def set_max_position(self, max_position: float) -> None:
        """更新最大允許倉位。"""
        self.max_position = max(abs(max_position), self.min_order_size)
        logger.info(
            "更新最大倉位限制: %s",
            format_balance(self.max_position),
        )
        self._update_position_state()
