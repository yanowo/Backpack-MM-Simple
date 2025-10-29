"""Maker掛單 + Taker對沖策略模組。"""
from __future__ import annotations

import time
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
        self._hedge_residuals: Dict[str, float] = {"Bid": 0.0, "Ask": 0.0}
        self._hedge_position_reference: float = 0.0
        self._hedge_poll_attempts = 6
        self._hedge_poll_interval = 0.5
        self._hedge_flat_tolerance = 1e-8

        super().__init__(*args, **kwargs)

        self.max_orders = 1

        self._hedge_flat_tolerance = max(getattr(self, "min_order_size", 0.0) / 10, 1e-8)
        self._initialize_hedge_reference_position()

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
        """所有成交後立即以市價對沖。"""

        super()._after_fill_processed(fill_info)

        # 獲取成交詳情
        side = fill_info.get("side")
        quantity = float(fill_info.get("quantity", 0) or 0)
        price = float(fill_info.get("price", 0) or 0)
        
        if not side or quantity <= 0:
            logger.warning("成交資訊不完整，跳過對沖")
            return
            
        logger.info(f"處理Maker成交：{side} {quantity}@{price}")
            
        # 先更新當前倉位引用
        current_position = self._fetch_current_position_reference()
        if current_position is not None:
            self._hedge_position_reference = 0.0  # 將目標倉位設為0
            logger.info(f"當前倉位：{current_position}")
            
            # 根據實際倉位計算對沖量
            if abs(current_position) > self._hedge_flat_tolerance:
                hedge_side = "Ask" if current_position > 0 else "Bid"
                hedge_qty = round_to_precision(abs(current_position), self.base_precision)
                
                if hedge_qty >= self.min_order_size:
                    logger.info(f"根據實際倉位執行對沖：{hedge_side} {hedge_qty}")
                    
                    # 提交對沖訂單
                    order = {
                        "orderType": "Market",
                        "quantity": str(hedge_qty),
                        "side": hedge_side,
                        "symbol": self.symbol,
                        "reduceOnly": True  # 確保是對沖訂單
                    }
                    
                    # 執行對沖訂單
                    result = self.client.execute_order(order)
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"對沖訂單執行失敗: {result['error']}")
                    else:
                        logger.info(f"對沖訂單已提交: {result.get('id', 'unknown')}")
                else:
                    logger.info(f"當前倉位 {current_position} 小於最小下單量，暫不對沖")
            else:
                logger.info("當前倉位接近0，無需對沖")
        else:
            logger.error("無法獲取當前倉位，對沖失敗")

        hedge_side = "Ask" if side == "Bid" else "Bid"

        previous_residual = self._hedge_residuals.get(hedge_side, 0.0)
        target_quantity = quantity + previous_residual
        if target_quantity <= 0:
            logger.debug("對沖目標數量 <= 0，跳過")
            return

        hedge_qty = round_to_precision(target_quantity, self.base_precision)

        if hedge_qty < self.min_order_size:
            self._hedge_residuals[hedge_side] = target_quantity
            logger.info(
                "對沖目標 %.8f 低於最小下單量，累積至下次 (殘留 %.8f)",
                target_quantity,
                target_quantity,
            )
            return

        logger.info(
            "偵測到 Maker 成交，準備以市價對沖 %s %s",
            format_balance(hedge_qty),
            hedge_side,
        )
        residual_delta = self._execute_taker_hedge(hedge_side, hedge_qty)

        if residual_delta is None:
            # 對沖提交失敗，保留完整目標量
            self._hedge_residuals[hedge_side] = target_quantity
            return

        self._hedge_residuals["Bid"] = 0.0
        self._hedge_residuals["Ask"] = 0.0

        if abs(residual_delta) <= self._hedge_flat_tolerance:
            logger.info("市價對沖已完成，倉位回到參考水位")
            return

        residual_side = "Ask" if residual_delta > 0 else "Bid"
        residual_amount = round_to_precision(abs(residual_delta), self.base_precision)
        self._hedge_residuals[residual_side] = residual_amount
        logger.warning(
            "市價對沖後仍剩餘倉位 %.8f，記錄為後續殘量 (方向=%s)",
            residual_amount,
            residual_side,
        )

    def _execute_taker_hedge(self, side: str, quantity: float) -> Optional[float]:
        """提交市價單完成對沖，並回傳剩餘倉位差值。"""

        attempt_side = side
        remaining_quantity = round_to_precision(quantity, self.base_precision)
        last_delta: Optional[float] = None

        current_delta = self._calculate_position_delta()
        if current_delta is not None:
            if abs(current_delta) <= self._hedge_flat_tolerance:
                latest_position = self._fetch_current_position_reference()
                if latest_position is not None:
                    self._hedge_position_reference = latest_position
                logger.info("目前倉位已接近參考水位，無需對沖")
                return 0.0

            attempt_side = "Ask" if current_delta > 0 else "Bid"
            remaining_quantity = round_to_precision(abs(current_delta), self.base_precision)
            logger.debug(
                "以實際倉位差 %.8f 重新設定對沖方向為 %s", current_delta, attempt_side
            )

        for attempt in range(1, 4):
            if remaining_quantity < self.min_order_size:
                logger.info(
                    "剩餘對沖量 %.8f 低於最小下單量，停止提交", remaining_quantity
                )
                break

            order = {
                "orderType": "Market",
                "quantity": str(remaining_quantity),
                "side": attempt_side,
                "symbol": self.symbol,
            }

            if getattr(self, "exchange", "backpack") == "backpack":
                order["timeInForce"] = "IOC"
                order["autoLendRedeem"] = True
                order["autoLend"] = True

            if isinstance(self, PerpetualMarketMaker):
                order["reduceOnly"] = True

            logger.info(
                "提交市價對沖訂單: %s %s (第 %d 次嘗試)",
                attempt_side,
                format_balance(remaining_quantity),
                attempt,
            )
            result = self.client.execute_order(order)
            if isinstance(result, dict) and "error" in result:
                logger.error(f"市價對沖失敗: {result['error']}")
                return None

            logger.info("市價對沖訂單已提交: %s", result.get("id", "未知ID"))

            last_delta = self._poll_position_delta()
            if last_delta is None:
                logger.warning("無法從API/WS獲取最新倉位，保留殘量待下次對沖")
                return None

            if abs(last_delta) <= self._hedge_flat_tolerance:
                current_position = self._fetch_current_position_reference()
                if current_position is not None:
                    self._hedge_position_reference = current_position
                return 0.0

            attempt_side = "Ask" if last_delta > 0 else "Bid"
            remaining_quantity = round_to_precision(abs(last_delta), self.base_precision)
            logger.warning(
                "市價對沖後仍有倉位差 %.8f，將再次以市價 %s 對沖",
                abs(last_delta),
                attempt_side,
            )

        return last_delta

    def _poll_position_delta(self) -> Optional[float]:
        """輪詢API/WS獲取最新倉位差值。"""

        delta: Optional[float] = None
        for _ in range(self._hedge_poll_attempts):
            time.sleep(self._hedge_poll_interval)
            delta = self._calculate_position_delta()
            if delta is None:
                continue
            if abs(delta) <= self._hedge_flat_tolerance:
                break
        return delta

    def _initialize_hedge_reference_position(self) -> None:
        """初始化倉位參考水位。"""

        reference = self._fetch_current_position_reference()
        if reference is None:
            reference = 0.0
        self._hedge_position_reference = reference
        logger.info("對沖參考倉位初始化為 %.8f", reference)

    def _calculate_position_delta(self) -> Optional[float]:
        """計算當前倉位相對參考水位的差值。"""

        current = self._fetch_current_position_reference()
        if current is None:
            return None
        return current - self._hedge_position_reference

    def _fetch_current_position_reference(self) -> Optional[float]:
        """透過API或WS獲取當前倉位指標。"""

        try:
            if isinstance(self, PerpetualMarketMaker):
                # 強制重新獲取倉位信息
                for attempt in range(3):  # 最多重試3次
                    positions = self.client.get_positions(self.symbol)
                    
                    if isinstance(positions, dict) and "error" in positions:
                        error_msg = positions.get("error", "")
                        if "404" in str(error_msg) or "RESOURCE_NOT_FOUND" in str(error_msg):
                            logger.info("無倉位記錄，當前倉位為0")
                            return 0.0
                        logger.error(f"獲取倉位失敗 (嘗試 {attempt + 1}/3): {error_msg}")
                        if attempt < 2:  # 如果不是最後一次嘗試
                            time.sleep(0.5)  # 等待500ms後重試
                            continue
                        return None

                    if isinstance(positions, list):
                        if not positions:
                            logger.info("無倉位記錄，當前倉位為0")
                            return 0.0
                            
                        position = positions[0]
                        # 嘗試所有可能的字段名
                        for field in ["netQuantity", "size", "position_size", "amount"]:
                            if field in position:
                                net_value = float(position[field] or 0)
                                logger.info(f"當前倉位 (from {field}): {net_value}")
                                return net_value
                                
                        logger.warning(f"倉位信息中找不到數量字段: {position}")
                        return 0.0
                    
                    logger.error(f"意外的API響應格式: {positions}")
                    if attempt < 2:
                        time.sleep(0.5)
                        continue
                    return None

                logger.error("多次嘗試後仍無法獲取有效倉位信息")
                return None

            _, total = self.get_asset_balance(self.base_asset)
            return float(total or 0.0)
            
        except Exception as exc:
            logger.error("獲取倉位資訊時發生錯誤: %s", exc)
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return None

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

