"""Maker掛單 + Taker對沖策略模組。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from logger import setup_logger
from strategies.market_maker import MarketMaker, format_balance
from strategies.perp_market_maker import PerpetualMarketMaker
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("maker_taker_hedge")


class _MakerTakerHedgeMixin:
    """封裝 Maker 掛單 + Taker 對沖的核心實作。"""

    def __init__(self, *args: Any, hedge_label: str = "現貨", **kwargs: Any) -> None:
        kwargs.pop("max_orders", None)
        kwargs.pop("enable_rebalance", None)
        kwargs.pop("base_asset_target_percentage", None)
        kwargs.pop("rebalance_threshold", None)

        kwargs["max_orders"] = 1
        kwargs["enable_rebalance"] = False

        self._hedge_label = hedge_label
        super().__init__(*args, **kwargs)

        self.max_orders = 1
        logger.info("初始化 Maker-Taker 對沖策略 (%s)", self._hedge_label)

    # ------------------------------------------------------------------
    # 下單與倉位管理
    # ------------------------------------------------------------------
    def place_limit_orders(self) -> None:
        """僅在買一/賣一位置掛出Post-Only訂單。"""

        self.check_ws_connection()
        self.cancel_existing_orders()

        bid_price, ask_price = self.get_market_depth()
        if bid_price is None or ask_price is None:
            logger.warning("無法取得買一/賣一價格，跳過本輪掛單")
            return

        buy_price = round_to_tick_size(bid_price, self.tick_size)
        sell_price = round_to_tick_size(ask_price, self.tick_size)

        if sell_price <= buy_price:
            sell_price = round_to_tick_size(buy_price + self.tick_size, self.tick_size)
            if sell_price <= buy_price:
                logger.warning("價差過窄無法安全掛單，跳過本輪")
                return

        buy_qty, sell_qty = self._determine_order_sizes(buy_price, ask_price)
        if buy_qty is None or sell_qty is None:
            logger.warning("無法計算掛單數量，跳過本輪")
            return

        self.active_buy_orders = []
        self.active_sell_orders = []

        if buy_qty >= self.min_order_size:
            buy_order = self._build_limit_order(
                side="Bid",
                price=buy_price,
                quantity=buy_qty,
            )
            result = self.client.execute_order(buy_order)
            if isinstance(result, dict) and "error" in result:
                logger.error(f"買單掛單失敗: {result['error']}")
            else:
                logger.info(
                    "買單已掛出: 價格 %s, 數量 %s",
                    format_balance(buy_price),
                    format_balance(buy_qty),
                )
                self.active_buy_orders.append(result)
                self.orders_placed += 1

        if sell_qty >= self.min_order_size:
            sell_order = self._build_limit_order(
                side="Ask",
                price=sell_price,
                quantity=sell_qty,
            )
            result = self.client.execute_order(sell_order)
            if isinstance(result, dict) and "error" in result:
                logger.error(f"賣單掛單失敗: {result['error']}")
            else:
                logger.info(
                    "賣單已掛出: 價格 %s, 數量 %s",
                    format_balance(sell_price),
                    format_balance(sell_qty),
                )
                self.active_sell_orders.append(result)
                self.orders_placed += 1

    def _determine_order_sizes(self, buy_price: float, ask_price: float) -> Tuple[Optional[float], Optional[float]]:
        """根據餘額決定單筆買/賣單量。"""

        if self.order_quantity is not None:
            quantity = max(
                self.min_order_size,
                round_to_precision(self.order_quantity, self.base_precision),
            )
            return quantity, quantity

        base_available, base_total = self.get_asset_balance(self.base_asset)
        quote_available, quote_total = self.get_asset_balance(self.quote_asset)

        reference_price = ask_price if ask_price else buy_price
        if reference_price <= 0:
            return None, None

        allocation = 0.05  # 使用總資金的5%
        quote_budget = quote_total * allocation
        base_budget = base_total * allocation

        if quote_budget <= 0 or base_budget <= 0:
            logger.warning("餘額不足，無法掛出Maker訂單")
            return None, None

        buy_qty = round_to_precision(quote_budget / reference_price, self.base_precision)
        sell_qty = round_to_precision(base_budget, self.base_precision)

        buy_qty = max(self.min_order_size, buy_qty)
        sell_qty = max(self.min_order_size, sell_qty)

        if quote_available < buy_qty * reference_price:
            logger.info(
                "可用報價資產不足 (%.8f)，將依賴自動贖回",
                quote_available,
            )
        if base_available < sell_qty:
            logger.info(
                "可用基礎資產不足 (%.8f)，將依賴自動贖回",
                base_available,
            )

        return buy_qty, sell_qty

    # ------------------------------------------------------------------
    # 成交後置處理
    # ------------------------------------------------------------------
    def _after_fill_processed(self, fill_info: Dict[str, Any]) -> None:
        """Maker成交後立即以市價對沖。"""

        super()._after_fill_processed(fill_info)

        if not fill_info.get("maker"):
            return

        side = fill_info.get("side")
        quantity = float(fill_info.get("quantity", 0) or 0)
        if not side or quantity <= 0:
            return

        hedge_side = "Ask" if side == "Bid" else "Bid"
        hedge_qty = round_to_precision(quantity, self.base_precision)
        if hedge_qty < self.min_order_size:
            logger.warning(
                "成交數量 %.8f 低於最小下單量，跳過對沖", quantity
            )
            return

        logger.info(
            "偵測到 Maker 成交，準備以市價對沖 %s %s",
            format_balance(hedge_qty),
            hedge_side,
        )
        self._execute_taker_hedge(hedge_side, hedge_qty)

    def _execute_taker_hedge(self, side: str, quantity: float) -> None:
        """提交市價單完成對沖。"""

        order = {
            "orderType": "Market",
            "quantity": str(round_to_precision(quantity, self.base_precision)),
            "side": side,
            "symbol": self.symbol,
        }

        if getattr(self, "exchange", "backpack") == "backpack":
            order["timeInForce"] = "IOC"
            order["autoLendRedeem"] = True
            order["autoLend"] = True

        if isinstance(self, PerpetualMarketMaker):
            order["reduceOnly"] = True

        logger.info("提交市價對沖訂單: %s %s", side, format_balance(quantity))
        result = self.client.execute_order(order)
        if isinstance(result, dict) and "error" in result:
            logger.error(f"市價對沖失敗: {result['error']}")
        else:
            logger.info("市價對沖訂單已提交: %s", result.get("id", "未知ID"))

    def _build_limit_order(self, side: str, price: float, quantity: float) -> Dict[str, str]:
        """依交易所特性構建單向限價訂單負載。"""

        order = {
            "orderType": "Limit",
            "price": str(round_to_tick_size(price, self.tick_size)),
            "quantity": str(round_to_precision(quantity, self.base_precision)),
            "side": side,
            "symbol": self.symbol,
            "timeInForce": "GTC",
        }

        if getattr(self, "exchange", "backpack") == "backpack":
            order["postOnly"] = True
            order["autoLendRedeem"] = True
            order["autoLend"] = True

        return order


class _SpotMakerTakerHedgeStrategy(_MakerTakerHedgeMixin, MarketMaker):
    """現貨 Maker 掛單 + Taker 對沖實作。"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        base_spread_percentage: float = 0.0,
        order_quantity: Optional[float] = None,
        ws_proxy: Optional[str] = None,
        exchange: str = "backpack",
        exchange_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            base_spread_percentage=base_spread_percentage,
            order_quantity=order_quantity,
            ws_proxy=ws_proxy,
            exchange=exchange,
            exchange_config=exchange_config,
            hedge_label="現貨僅掛買一/賣一",
            **kwargs,
        )


class _PerpMakerTakerHedgeStrategy(_MakerTakerHedgeMixin, PerpetualMarketMaker):
    """永續合約 Maker 掛單 + Taker 對沖實作。"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        base_spread_percentage: float = 0.0,
        order_quantity: Optional[float] = None,
        target_position: float = 0.0,
        max_position: float = 1.0,
        position_threshold: float = 0.1,
        inventory_skew: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        ws_proxy: Optional[str] = None,
        exchange: str = "backpack",
        exchange_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            base_spread_percentage=base_spread_percentage,
            order_quantity=order_quantity,
            target_position=target_position,
            max_position=max_position,
            position_threshold=position_threshold,
            inventory_skew=inventory_skew,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ws_proxy=ws_proxy,
            exchange=exchange,
            exchange_config=exchange_config,
            hedge_label="永續合約僅掛買一/賣一",
            **kwargs,
        )


class MakerTakerHedgeStrategy:
    """根據市場類型返回對應的 Maker-Taker 對沖策略實例。"""

    def __new__(cls, *args: Any, market_type: str = "spot", **kwargs: Any):
        market = (market_type or "spot").lower()
        if market == "perp":
            return _PerpMakerTakerHedgeStrategy(*args, **kwargs)
        return _SpotMakerTakerHedgeStrategy(*args, **kwargs)

