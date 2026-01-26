"""Maker掛單 + Taker對沖策略模組。"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from logger import setup_logger
from strategies.market_maker import MarketMaker, format_balance
from strategies.perp_market_maker import PerpetualMarketMaker
from utils.helpers import round_to_precision, round_to_tick_size, format_quantity

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
        self._hedge_flat_tolerance = 1e-8
        
        # 本地倉位追蹤（基於成交推算，減少 API 請求）
        self._local_position: Optional[float] = None
        self._local_position_synced: bool = False
        self._position_sync_interval: float = 60.0  # 每 60 秒強制同步一次
        self._last_position_sync_ts: float = 0.0

        self._request_intervals: Dict[str, float] = {
            "limit": 0.35,
            "market": 0.45,
            "position": 1.0,
        }
        self._last_request_ts: Dict[str, float] = {key: 0.0 for key in self._request_intervals}
        self._rate_limit_retries = 4
        self._rate_limit_backoff = 0.6
        self._rate_limit_max_backoff = 5.0

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
            result = self._submit_order(buy_order, slot="limit")
            if not result.success:
                logger.error(f"買單掛單失敗: {result.error_message}")
            else:
                logger.info(
                    "買單已掛出: 價格 %s, 數量 %s",
                    format_balance(buy_price),
                    format_balance(buy_qty),
                )
                if result.data:
                    self.active_buy_orders.append(result.data)
                self.orders_placed += 1

        if sell_qty >= self.min_order_size:
            sell_order = self._build_limit_order(
                side="Ask",
                price=sell_price,
                quantity=sell_qty,
            )
            result = self._submit_order(sell_order, slot="limit")
            if not result.success:
                logger.error(f"賣單掛單失敗: {result.error_message}")
            else:
                logger.info(
                    "賣單已掛出: 價格 %s, 數量 %s",
                    format_balance(sell_price),
                    format_balance(sell_qty),
                )
                if result.data:
                    self.active_sell_orders.append(result.data)
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

        def _to_bool(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if value in (None, "", "None"):
                return None
            try:
                return str(value).lower() in {"true", "1", "yes"}
            except Exception:
                return None

        # 獲取成交詳情
        side = fill_info.get("side")
        quantity = float(fill_info.get("quantity", 0) or 0)
        price = float(fill_info.get("price", 0) or 0)
        maker_flag = None
        for key in ("is_maker", "maker", "isMaker", "m"):
            if key in fill_info:
                maker_flag = fill_info.get(key)
                break
        is_maker = _to_bool(maker_flag)
        if is_maker is False:
            logger.debug("忽略 Taker 成交事件，無需對沖")
            return
        
        if not side or quantity <= 0:
            logger.warning("成交資訊不完整，跳過對沖")
            return
            
        logger.info(f"處理Maker成交：{side} {quantity}@{price}")
        
        # 先根據 Maker 成交更新本地倉位追蹤
        # Maker Bid（買入）成交 = 倉位增加，Maker Ask（賣出）成交 = 倉位減少
        self._update_local_position_from_fill(side, quantity)
        
        # 使用更新後的本地追蹤倉位
        current_position = self._get_tracked_position()
        if current_position is None:
            logger.warning("本地倉位未初始化，嘗試同步 API")
            current_position = self._sync_position_from_api()
            if current_position is None:
                logger.error("無法獲取當前倉位，對沖失敗")
                return

        logger.info(f"當前追蹤倉位：{current_position}")

        net_delta = current_position - self._hedge_position_reference
        if abs(net_delta) <= self._hedge_flat_tolerance:
            logger.info("倉位已在參考水位附近，跳過對沖")
            self._hedge_residuals["Bid"] = 0.0
            self._hedge_residuals["Ask"] = 0.0
            return

        # 根據實際倉位差來決定對沖方向和數量
        # 倉位 > 參考 = 需要賣出(Ask)，倉位 < 參考 = 需要買入(Bid)
        hedge_side = "Ask" if net_delta > 0 else "Bid"
        hedge_qty = round_to_precision(abs(net_delta), self.base_precision)

        if hedge_qty < self.min_order_size:
            # 對沖量太小，累積到下次
            self._hedge_residuals[hedge_side] = abs(net_delta)
            logger.info(
                "對沖目標 %.8f 低於最小下單量，累積至下次 (殘留 %.8f)",
                abs(net_delta),
                abs(net_delta),
            )
            return

        logger.info(
            "偵測到倉位偏移 %.8f，準備以市價對沖 %s %s",
            net_delta,
            format_balance(hedge_qty),
            hedge_side,
        )
        residual_delta = self._execute_taker_hedge(hedge_side, hedge_qty, current_position=current_position)

        if residual_delta is None:
            # 對沖提交失敗，保留完整目標量
            self._hedge_residuals[hedge_side] = hedge_qty
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

    def _execute_taker_hedge(
        self,
        side: str,
        quantity: float,
        *,
        current_position: Optional[float] = None,
    ) -> Optional[float]:
        """提交市價單完成對沖，並回傳剩餘倉位差值。"""

        attempt_side = side
        remaining_quantity = round_to_precision(quantity, self.base_precision)
        last_delta: Optional[float] = None

        current_delta = self._calculate_position_delta(current_position=current_position)
        if current_delta is not None:
            if abs(current_delta) <= self._hedge_flat_tolerance:
                latest_position = current_position if current_position is not None else self._fetch_current_position_reference()
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
                "quantity": format_quantity(remaining_quantity, self.base_precision),
                "side": attempt_side,
                "symbol": self.symbol,
            }

            if getattr(self, "exchange", "backpack") == "backpack":
                order["timeInForce"] = "IOC"
                order["autoLendRedeem"] = True
                order["autoLend"] = True

            if isinstance(self, PerpetualMarketMaker):
                # reduceOnly 需要確保對沖數量不超過實際倉位
                actual_position = current_position if current_position is not None else self._get_tracked_position()
                if actual_position is not None:
                    max_reduce = abs(actual_position)
                    if remaining_quantity > max_reduce:
                        logger.warning(
                            "對沖數量 %.8f 超過當前倉位 %.8f，調整為最大可平倉量",
                            remaining_quantity,
                            max_reduce,
                        )
                        remaining_quantity = round_to_precision(max_reduce, self.base_precision)
                        order["quantity"] = str(remaining_quantity)
                order["reduceOnly"] = True

            logger.info(
                "提交市價對沖訂單: %s %s (第 %d 次嘗試)",
                attempt_side,
                format_balance(remaining_quantity),
                attempt,
            )
            result = self._submit_order(order, slot="market")
            if not result.success:
                logger.error(f"市價對沖失敗: {result.error_message}")
                # 對沖失敗，強制同步 API 校正本地追蹤
                self._sync_position_from_api()
                return None

            order_result = result.data
            logger.info("市價對沖訂單已提交: %s", order_result.order_id if order_result else "未知ID")

            # 計算預期倉位變化：Ask(賣出)=-quantity, Bid(買入)=+quantity
            expected_change = -remaining_quantity if attempt_side == "Ask" else remaining_quantity
            last_delta = self._poll_position_delta(expected_change=expected_change)
            if last_delta is None:
                logger.warning("無法確認倉位變化，保留殘量待下次對沖")
                return None

            if abs(last_delta) <= self._hedge_flat_tolerance:
                # 對沖成功，更新參考倉位（使用本地追蹤，避免額外 API 請求）
                if self._local_position is not None:
                    self._hedge_position_reference = self._local_position
                logger.info("對沖成功，倉位已回到參考水位")
                return 0.0

            attempt_side = "Ask" if last_delta > 0 else "Bid"
            remaining_quantity = round_to_precision(abs(last_delta), self.base_precision)
            logger.warning(
                "市價對沖後仍有倉位差 %.8f，將再次以市價 %s 對沖",
                abs(last_delta),
                attempt_side,
            )

        return last_delta

    def _poll_position_delta(self, expected_change: float = 0.0) -> Optional[float]:
        """等待並確認倉位變化，對沖後從 API 驗證實際倉位。
        
        Args:
            expected_change: 預期的倉位變化量（正=買入，負=賣出）
        """
        # 等待訂單處理時間
        time.sleep(1.0)
        
        # 對沖後必須從 API 確認實際倉位，避免追蹤偏差
        # 這是唯一需要請求 API 的關鍵時刻
        actual_position = self._sync_position_from_api()
        if actual_position is None:
            logger.warning("無法從 API 確認倉位，使用本地預估")
            # 備用：使用本地追蹤 + 預期變化
            if self._local_position is not None:
                self._local_position += expected_change
                actual_position = self._local_position
            else:
                return None
        
        delta = actual_position - self._hedge_position_reference
        logger.debug("倉位差值: %.8f (實際 %.8f, 參考 %.8f)", delta, actual_position, self._hedge_position_reference)
        return delta

    def _initialize_hedge_reference_position(self) -> None:
        """初始化倉位參考水位和本地追蹤。"""

        reference = self._fetch_current_position_reference()
        if reference is None:
            reference = 0.0
        self._hedge_position_reference = reference
        # 同時初始化本地追蹤
        self._local_position = reference
        self._local_position_synced = True
        self._last_position_sync_ts = time.monotonic()
        logger.info("對沖參考倉位初始化為 %.8f（本地追蹤已同步）", reference)

    def _get_tracked_position(self) -> Optional[float]:
        """獲取本地追蹤的倉位，必要時自動同步。"""
        now = time.monotonic()
        
        # 檢查是否需要強制同步（超過 60 秒）
        if self._local_position_synced and (now - self._last_position_sync_ts) > self._position_sync_interval:
            logger.info("定期同步：重新從 API 獲取倉位")
            self._sync_position_from_api()
        
        return self._local_position
    
    def _sync_position_from_api(self) -> Optional[float]:
        """從 API 同步倉位到本地追蹤。"""
        position = self._fetch_current_position_reference()
        if position is not None:
            self._local_position = position
            self._local_position_synced = True
            self._last_position_sync_ts = time.monotonic()
            logger.debug("倉位已從 API 同步: %.8f", position)
        return position
    
    def _update_local_position_from_fill(self, side: str, quantity: float) -> None:
        """根據成交更新本地追蹤倉位。
        
        Args:
            side: 成交方向 ("Bid" = 買入, "Ask" = 賣出)
            quantity: 成交數量
        """
        if self._local_position is None:
            return
            
        if side == "Bid":
            self._local_position += quantity
        elif side == "Ask":
            self._local_position -= quantity
        
        logger.debug("成交更新本地倉位: %s %.8f -> 當前 %.8f", side, quantity, self._local_position)

    def _calculate_position_delta(
        self,
        *,
        current_position: Optional[float] = None,
    ) -> Optional[float]:
        """計算當前倉位相對參考水位的差值（優先使用本地追蹤）。"""

        if current_position is not None:
            current = current_position
        else:
            # 優先使用本地追蹤
            current = self._get_tracked_position()
            if current is None:
                # 回退到 API
                current = self._sync_position_from_api()
        if current is None:
            return None
        return current - self._hedge_position_reference

    def _fetch_current_position_reference(self, force_refresh: bool = False) -> Optional[float]:
        """透過 API 獲取當前倉位指標。
        
        Args:
            force_refresh: 已棄用，保留供相容性
        """
        try:
            if isinstance(self, PerpetualMarketMaker):
                positions_response = self._request_positions()
                
                if not positions_response.success:
                    error_msg = positions_response.error_message or ""
                    if "404" in str(error_msg) or "RESOURCE_NOT_FOUND" in str(error_msg):
                        logger.debug("無倉位記錄，當前倉位為0")
                        return 0.0
                    logger.error(f"獲取倉位失敗: {error_msg}")
                    return None

                positions = positions_response.data
                if isinstance(positions, list):
                    if not positions:
                        logger.debug("無倉位記錄，當前倉位為0")
                        return 0.0
                        
                    position = positions[0]
                    # 支援 PositionInfo dataclass
                    if hasattr(position, 'size'):
                        size_value = float(position.size or 0)
                        # 根據 side 決定正負號：LONG=正, SHORT=負
                        if hasattr(position, 'side'):
                            if position.side == "SHORT":
                                size_value = -size_value
                            # LONG 或 FLAT 保持原值
                        return size_value
                    # 後備：字典格式
                    for field in ["netQuantity", "size", "position_size", "amount"]:
                        if field in position:
                            return float(position[field] or 0)
                            
                    logger.warning(f"倉位信息中找不到數量字段: {position}")
                    return 0.0
                
                logger.error(f"意外的API響應格式: {positions}")
                return None

            _, total = self.get_asset_balance(self.base_asset)
            return float(total or 0.0)
            
        except Exception as exc:
            logger.error("獲取倉位資訊時發生錯誤: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 節流與重試工具
    # ------------------------------------------------------------------
    def _respect_request_interval(self, slot: str) -> None:
        interval = self._request_intervals.get(slot)
        if not interval:
            return
        last_ts = self._last_request_ts.get(slot, 0.0)
        now = time.monotonic()
        wait_for = interval - (now - last_ts)
        if wait_for > 0:
            time.sleep(wait_for)
        self._last_request_ts[slot] = time.monotonic()

    def _detect_rate_limit(self, payload: Any) -> Optional[str]:
        if payload is None:
            return None
        message = None
        if isinstance(payload, dict):
            for key in ("error", "message", "detail"):
                value = payload.get(key)
                if value:
                    message = str(value)
                    break
        else:
            message = str(payload)

        if not message:
            return None

        lowered = message.lower()
        keywords = ("too many", "rate limit", "429", "request limit")
        if any(keyword in lowered for keyword in keywords):
            return message
        return None

    def _request_with_backoff(self, slot: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        backoff = self._rate_limit_backoff
        result: Any = None
        for attempt in range(1, self._rate_limit_retries + 1):
            self._respect_request_interval(slot)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - log for visibility
                message = self._detect_rate_limit(exc)
                if not message or attempt == self._rate_limit_retries:
                    raise
                logger.warning(
                    "API %s 請求觸發限制 (%s)，將在 %.2fs 後重試 (%d/%d)",
                    slot,
                    message,
                    backoff,
                    attempt,
                    self._rate_limit_retries,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.6, self._rate_limit_max_backoff)
                continue

            message = self._detect_rate_limit(result)
            if not message:
                return result

            if attempt == self._rate_limit_retries:
                logger.error("API %s 持續遭遇請求限制: %s", slot, message)
                return result

            logger.warning(
                "API %s 請求觸發限制 (%s)，將在 %.2fs 後重試 (%d/%d)",
                slot,
                message,
                backoff,
                attempt,
                self._rate_limit_retries,
            )
            time.sleep(backoff)
            backoff = min(backoff * 1.6, self._rate_limit_max_backoff)

        return result

    def _submit_order(self, order: Dict[str, Any], slot: str) -> Any:
        return self._request_with_backoff(slot, self.client.execute_order, order)

    def _request_positions(self) -> Any:
        return self._request_with_backoff("position", self.client.get_positions, self.symbol)

    def _build_limit_order(self, side: str, price: float, quantity: float) -> Dict[str, str]:
        """依交易所特性構建單向限價訂單負載。"""

        order = {
            "orderType": "Limit",
            "price": str(round_to_tick_size(price, self.tick_size)),
            "quantity": format_quantity(round_to_precision(quantity, self.base_precision), self.base_precision),
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

