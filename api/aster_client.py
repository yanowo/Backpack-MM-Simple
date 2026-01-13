"""Aster exchange REST client implementation."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional, Set
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

import requests

from .base_client import (
    BaseExchangeClient,
    ApiResponse,
    OrderResult,
    OrderInfo,
    BalanceInfo,
    CollateralInfo,
    PositionInfo,
    MarketInfo,
    TickerInfo,
    OrderBookInfo,
    OrderBookLevel,
    KlineInfo,
    TradeInfo,
    CancelResult,
    BatchOrderResult,
)
from .proxy_utils import get_proxy_config
from logger import setup_logger

logger = setup_logger("api.aster_client")



class AsterClient(BaseExchangeClient):
    """REST client for the Aster perpetual futures API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.base_url = config.get("base_url", "https://fapi.asterdex.com")
        self.recv_window = int(config.get("recv_window", 5000))
        self.timeout = float(config.get("timeout", 10))
        self.max_retries = int(config.get("max_retries", 3))
        self.session = requests.Session()

        # 從環境變量讀取代理配置
        proxies = get_proxy_config()
        if proxies:
            self.session.proxies.update(proxies)
            logger.info(f"Aster 客户端已配置代理: {proxies}")

        self._symbol_cache: Dict[str, str] = {}
        self._market_info_cache: Dict[str, Dict[str, Any]] = {}

    def get_exchange_name(self) -> str:
        return "Aster"

    async def connect(self) -> None:
        logger.info("Aster 客户端已連接")

    async def disconnect(self) -> None:
        self.session.close()
        logger.info("Aster 客户端已斷開連接")

    def _current_timestamp(self) -> int:
        return int(time.time() * 1000)

    def _bool_to_lower(self, value: Any) -> str:
        if isinstance(value, str):
            return value.lower()
        return "true" if value else "false"

    def _normalize_order_fields(self, order: Dict[str, Any]) -> Dict[str, Any]:
        if "orderId" in order and "id" not in order:
            order["id"] = str(order["orderId"])
        if "clientOrderId" in order:
            order.setdefault("clientId", order["clientOrderId"])
        side = order.get("side")
        if side:
            normalized = side.upper()
            if normalized == "BUY":
                order["side"] = "Bid"
            elif normalized == "SELL":
                order["side"] = "Ask"
        if "origQty" in order and "quantity" not in order:
            order["quantity"] = order["origQty"]
        if "price" in order:
            order["price"] = order["price"]
        return order

    def _lookup_key(self, symbol: str) -> str:
        """Generate a case-insensitive lookup key for exchange symbols."""
        return symbol.upper()

    def _ensure_symbol_cache(self) -> None:
        """Lazy-load the symbol cache from exchange info."""
        if self._symbol_cache and self._market_info_cache:
            return

        response = self.get_markets()
        if not response.success:
            logger.error("獲取交易對列表失敗: %s", response.error_message)
            self._symbol_cache = {}
            self._market_info_cache = {}
            return

        # response.data 是 List[MarketInfo]，response.raw 是原始字典
        raw_data = response.raw or {}
        symbols = raw_data.get("symbols", []) if isinstance(raw_data, dict) else []
        cache: Dict[str, str] = {}
        market_cache: Dict[str, Dict[str, Any]] = {}
        for item in symbols:
            actual_symbol = item.get("symbol")
            if not actual_symbol:
                continue
            cache[self._lookup_key(actual_symbol)] = actual_symbol
            market_cache[actual_symbol] = item

        self._symbol_cache = cache
        self._market_info_cache = market_cache

    def _resolve_symbol(self, symbol: Optional[str]) -> Optional[str]:
        """Resolve user provided symbol aliases to Aster native symbols."""
        if not symbol:
            return None

        self._ensure_symbol_cache()
        if not self._symbol_cache:
            return None

        sanitized = symbol.strip().upper()
        return self._symbol_cache.get(self._lookup_key(sanitized))

    def _decimal_to_str(self, value: Decimal) -> str:
        """Format Decimal without scientific notation and trim trailing zeros."""
        normalized = value.normalize() if value != 0 else Decimal("0")
        text = format(normalized, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    def _find_symbol_suggestions(self, symbol: str, limit: int = 5) -> List[str]:
        """Suggest possible symbols when lookup fails."""
        self._ensure_symbol_cache()
        if not self._market_info_cache:
            return []

        sanitized = symbol.strip().upper().replace("-", "_")
        token = sanitized.replace("_", "")
        candidates: List[str] = []
        seen: Set[str] = set()

        # Fuzzy match by substring
        if token:
            for actual in self._market_info_cache.keys():
                if token in actual.upper() and actual not in seen:
                    candidates.append(actual)
                    seen.add(actual)
                    if len(candidates) >= limit:
                        return candidates

        parts = [p for p in sanitized.split("_") if p]
        if parts:
            base = parts[0]
            for actual, info in self._market_info_cache.items():
                base_asset = (info.get("baseAsset") or "").upper()
                if base_asset == base and actual not in seen:
                    candidates.append(actual)
                    seen.add(actual)
                    if len(candidates) >= limit:
                        break

        return candidates[:limit]

    def _unknown_symbol_error(self, symbol: str) -> Dict[str, Any]:
        suggestions = self._find_symbol_suggestions(symbol)
        message = f"無法解析交易對: {symbol}"
        if suggestions:
            message += f"。可能的交易對: {', '.join(suggestions)}"
            logger.error(message)
            return {"error": message, "status_code": 400, "details": {"candidates": suggestions}}
        logger.error(message)
        return {"error": message, "status_code": 400}

    def _sign_payload(self, payload: Dict[str, Any], secret_key: str) -> str:
        query_string = urlencode(payload, doseq=True)
        return hmac.new(secret_key.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()

    def make_request(
        self,
        method: str,
        endpoint: str,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        instruction: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 3,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        payload: Dict[str, Any] = {}
        if params:
            payload.update({k: v for k, v in params.items() if v is not None})
        if data:
            payload.update({k: v for k, v in data.items() if v is not None})

        signed = bool(instruction)
        headers = {}
        if api_key:
            headers["X-MBX-APIKEY"] = api_key

        if signed:
            if not secret_key:
                return {"error": "缺少簽名所需的密鑰"}
            payload.setdefault("timestamp", self._current_timestamp())
            if self.recv_window:
                payload.setdefault("recvWindow", self.recv_window)
            signature = self._sign_payload(payload, secret_key)
            payload["signature"] = signature

        method_upper = method.upper()
        retry_total = retry_count or self.max_retries

        for attempt in range(retry_total):
            try:
                if method_upper in {"GET", "DELETE"}:
                    response = self.session.request(
                        method_upper,
                        url,
                        params=payload,
                        timeout=self.timeout,
                        headers=headers,
                    )
                else:
                    response = self.session.request(
                        method_upper,
                        url,
                        params=None,
                        data=payload,
                        timeout=self.timeout,
                        headers=headers,
                    )
                if 200 <= response.status_code < 300:
                    return response.json() if response.text else {}
                if response.status_code == 429:
                    wait_time = min(1 * (2 ** attempt), 8)
                    logger.warning("Aster API 達到速率限制，等待 %.1f 秒後重試", wait_time)
                    time.sleep(wait_time)
                    continue
                try:
                    error_body = response.json()
                    message = error_body.get("msg") or error_body.get("message") or str(error_body)
                except ValueError:
                    message = response.text or f"HTTP {response.status_code}"
                    error_body = {"msg": message}
                if attempt < retry_total - 1 and response.status_code >= 500:
                    time.sleep(1)
                    continue
                return {"error": message, "status_code": response.status_code, "details": error_body}
            except requests.RequestException as exc:
                if attempt < retry_total - 1:
                    logger.warning("Aster API 請求異常 (%s)，重試中...", exc)
                    time.sleep(1)
                    continue
                return {"error": f"請求失敗: {exc}"}
        return {"error": "達到最大重試次數"}

    def get_deposit_address(self, blockchain: str) -> ApiResponse:
        """Aster Futures 不支持此功能"""
        return ApiResponse.error("Aster Futures 不支持通過此API獲取充值地址")

    def get_balance(self) -> ApiResponse:
        """獲取賬户餘額
        
        Returns:
            ApiResponse with data: List[BalanceInfo]
        """
        raw = self.make_request(
            "GET",
            "/fapi/v2/balance",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        balances = []
        for item in raw:
            asset = item.get("asset")
            if not asset:
                continue
            available = self.safe_decimal(item.get("availableBalance"), Decimal("0"))
            total = self.safe_decimal(item.get("balance"), Decimal("0"))
            locked = max(total - available, Decimal("0"))
            balances.append(BalanceInfo(
                asset=asset,
                available=available,
                locked=locked,
                total=total,
                raw=item
            ))
        
        return ApiResponse.ok(balances, raw=raw)

    def get_collateral(self, subaccount_id: Optional[str] = None) -> ApiResponse:
        """獲取抵押品信息
        
        Returns:
            ApiResponse with data: List[CollateralInfo]
        """
        raw = self.make_request(
            "GET",
            "/fapi/v4/account",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            retry_count=self.max_retries,
        )
        
        if isinstance(raw, dict) and "error" in raw and "assets" not in raw:
            return self._parse_raw_to_error(raw)
        
        collaterals = []
        for item in raw.get("assets", []):
            asset = item.get("asset")
            if not asset:
                continue
            collaterals.append(CollateralInfo(
                asset=asset,
                total_collateral=self.safe_decimal(item.get("marginBalance"), Decimal("0")),
                free_collateral=self.safe_decimal(item.get("availableBalance"), Decimal("0")),
                account_value=self.safe_decimal(item.get("walletBalance")),
                unrealized_pnl=self.safe_decimal(item.get("unrealizedProfit")),
                raw=item
            ))
        
        return ApiResponse.ok(collaterals, raw=raw)

    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        """執行訂單
        
        Returns:
            ApiResponse with data: OrderResult
        """
        symbol = order_details.get("symbol")
        if not symbol:
            return ApiResponse.error("缺少交易對")
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            msg = f"無法解析交易對: {symbol}"
            if suggestions:
                msg += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(msg)

        side = order_details.get("side")
        if not side:
            return ApiResponse.error("缺少買賣方向")
        if side.lower() in {"bid", "buy"}:
            normalized_side = "BUY"
        elif side.lower() in {"ask", "sell"}:
            normalized_side = "SELL"
        else:
            return ApiResponse.error(f"不支持的方向: {side}")

        order_type = order_details.get("orderType") or order_details.get("type")
        if not order_type:
            return ApiResponse.error("缺少訂單類型")
        normalized_type = order_type.upper()

        payload: Dict[str, Any] = {"symbol": resolved_symbol, "side": normalized_side, "type": normalized_type}

        # 處理 postOnly（僅掛單模式）
        post_only = order_details.get("postOnly", False)
        time_in_force = order_details.get("timeInForce")
        
        if normalized_type == "LIMIT":
            if post_only:
                payload["timeInForce"] = "GTX"
            else:
                payload["timeInForce"] = (time_in_force or "GTC").upper()
        elif time_in_force:
            payload["timeInForce"] = time_in_force.upper()

        quantity = order_details.get("quantity") or order_details.get("size")
        if quantity is not None:
            payload["quantity"] = str(quantity)

        quote_quantity = order_details.get("quoteQuantity")
        if quote_quantity is not None:
            payload["quoteOrderQty"] = str(quote_quantity)

        price = order_details.get("price")
        if price is not None and normalized_type != "MARKET":
            payload["price"] = str(price)

        stop_price = order_details.get("stopPrice")
        if stop_price is not None:
            payload["stopPrice"] = str(stop_price)

        for key in ["reduceOnly", "closePosition", "priceProtect"]:
            if key in order_details:
                payload[key] = self._bool_to_lower(order_details[key])

        if "positionSide" in order_details:
            payload["positionSide"] = order_details["positionSide"].upper()
        if "clientId" in order_details:
            payload["newClientOrderId"] = order_details["clientId"]
        if "workingType" in order_details:
            payload["workingType"] = order_details["workingType"]
        if "newOrderRespType" in order_details:
            payload["newOrderRespType"] = order_details["newOrderRespType"]
        else:
            payload["newOrderRespType"] = "RESULT"

        raw = self.make_request(
            "POST",
            "/fapi/v1/order",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params=payload,
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return ApiResponse(
                success=False,
                data=OrderResult(success=False, error_message=error.error_message, raw=raw),
                error_message=error.error_message,
                raw=raw
            )
        
        order_result = OrderResult(
            success=True,
            order_id=str(raw.get("orderId", "")),
            client_order_id=raw.get("clientOrderId"),
            symbol=raw.get("symbol"),
            side="Bid" if raw.get("side", "").upper() == "BUY" else "Ask",
            order_type=raw.get("type"),
            size=self.safe_decimal(raw.get("origQty")),
            price=self.safe_decimal(raw.get("price")),
            filled_size=self.safe_decimal(raw.get("executedQty")),
            status=raw.get("status"),
            created_at=self.safe_int(raw.get("updateTime")),
            raw=raw
        )
        
        return ApiResponse.ok(order_result, raw=raw)

    def execute_order_batch(self, orders_details: List[Dict[str, Any]]) -> ApiResponse:
        """批量執行訂單

        Aster 批量下單限制：每批最多 5 個訂單

        Args:
            orders_details: 訂單詳情列表

        Returns:
            ApiResponse with data: BatchOrderResult
        """
        if not orders_details:
            return ApiResponse.error("訂單列表為空")

        # Aster 限制每批最多 5 個訂單
        if len(orders_details) > 5:
            logger.warning("Aster 批量下單限制為 5 個訂單，當前 %d 個，將拆分為多批", len(orders_details))

        # 將訂單拆分為多批（每批最多 5 個）
        batch_size = 5
        all_results: List[OrderResult] = []
        all_errors: List[str] = []

        for batch_start in range(0, len(orders_details), batch_size):
            batch = orders_details[batch_start:batch_start + batch_size]

            # 構建批量訂單請求
            batch_orders = []

            for order_details in batch:
                symbol = order_details.get("symbol")
                if not symbol:
                    logger.warning("跳過無效訂單: 缺少交易對")
                    all_errors.append("缺少交易對")
                    continue

                resolved_symbol = self._resolve_symbol(symbol)
                if not resolved_symbol:
                    logger.warning("跳過無效訂單: 未知交易對 %s", symbol)
                    all_errors.append(f"未知交易對: {symbol}")
                    continue

                side = order_details.get("side")
                if not side:
                    logger.warning("跳過無效訂單: 缺少買賣方向")
                    all_errors.append("缺少買賣方向")
                    continue

                # 標準化方向
                if side.lower() in {"bid", "buy"}:
                    normalized_side = "BUY"
                elif side.lower() in {"ask", "sell"}:
                    normalized_side = "SELL"
                else:
                    logger.warning("跳過無效訂單: 不支持的方向 %s", side)
                    all_errors.append(f"不支持的方向: {side}")
                    continue

                order_type = order_details.get("orderType") or order_details.get("type")
                if not order_type:
                    logger.warning("跳過無效訂單: 缺少訂單類型")
                    all_errors.append("缺少訂單類型")
                    continue

                normalized_type = order_type.upper()

                # 構建單個訂單
                order_payload = {
                    "symbol": resolved_symbol,
                    "side": normalized_side,
                    "type": normalized_type
                }

                # 處理 postOnly（僅掛單模式）
                post_only = order_details.get("postOnly", False)
                time_in_force = order_details.get("timeInForce")
                
                if normalized_type == "LIMIT":
                    if post_only:
                        order_payload["timeInForce"] = "GTX"
                    else:
                        order_payload["timeInForce"] = (time_in_force or "GTC").upper()
                elif time_in_force:
                    order_payload["timeInForce"] = time_in_force.upper()

                # 添加數量
                quantity = order_details.get("quantity") or order_details.get("size")
                if quantity is not None:
                    order_payload["quantity"] = str(quantity)
                else:
                    logger.warning("跳過無效訂單: 缺少數量")
                    all_errors.append("缺少數量")
                    continue

                # 添加價格（限價單）
                price = order_details.get("price")
                if price is not None and normalized_type != "MARKET":
                    order_payload["price"] = str(price)

                # 添加止損價格
                stop_price = order_details.get("stopPrice")
                if stop_price is not None:
                    order_payload["stopPrice"] = str(stop_price)

                # 添加可選參數
                for key in ["reduceOnly", "closePosition", "priceProtect"]:
                    if key in order_details:
                        order_payload[key] = self._bool_to_lower(order_details[key])

                if "positionSide" in order_details:
                    order_payload["positionSide"] = order_details["positionSide"].upper()

                if "clientId" in order_details:
                    order_payload["newClientOrderId"] = order_details["clientId"]

                if "workingType" in order_details:
                    order_payload["workingType"] = order_details["workingType"]

                batch_orders.append(order_payload)

            # 如果這批沒有有效訂單，跳過
            if not batch_orders:
                continue

            # 發送批量請求
            logger.info("發送批量訂單請求: %d 個訂單", len(batch_orders))

            # Aster/Binance Futures API 要求 batchOrders 作為 JSON 字符串
            raw = self.make_request(
                "POST",
                "/fapi/v1/batchOrders",
                api_key=self.api_key,
                secret_key=self.secret_key,
                instruction=True,
                data={"batchOrders": json.dumps(batch_orders)},
                retry_count=self.max_retries
            )

            # 處理響應
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        # 檢查是否是錯誤
                        if "code" in item and "msg" in item:
                            all_errors.append(str(item.get("msg", "Unknown error")))
                            logger.warning("批量下單部分失敗: %s", item.get("msg"))
                        else:
                            # 成功的訂單
                            all_results.append(OrderResult(
                                success=True,
                                order_id=str(item.get("orderId", "")),
                                client_order_id=item.get("clientOrderId"),
                                symbol=item.get("symbol"),
                                side="Bid" if item.get("side", "").upper() == "BUY" else "Ask",
                                order_type=item.get("type"),
                                size=self.safe_decimal(item.get("origQty")),
                                price=self.safe_decimal(item.get("price")),
                                status=item.get("status"),
                                raw=item
                            ))
                logger.info("批量下單完成: %d 成功, %d 失敗", len(all_results), len(all_errors))
            elif isinstance(raw, dict) and "error" in raw:
                logger.error("批量下單失敗: %s", raw["error"])
                all_errors.append(str(raw["error"]))

        batch_result = BatchOrderResult(
            success=len(all_results) > 0,
            orders=all_results,
            failed_count=len(all_errors),
            errors=all_errors,
            raw=None
        )
        
        return ApiResponse.ok(batch_result) if all_results else ApiResponse.error("批量下單全部失敗", raw={"errors": all_errors})

    def get_open_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取未成交訂單
        
        Returns:
            ApiResponse with data: List[OrderInfo]
        """
        params: Dict[str, Any] = {}
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                suggestions = self._find_symbol_suggestions(symbol)
                msg = f"無法解析交易對: {symbol}"
                if suggestions:
                    msg += f"。可能的交易對: {', '.join(suggestions)}"
                return ApiResponse.error(msg)
            params["symbol"] = resolved_symbol
        
        raw = self.make_request(
            "GET",
            "/fapi/v1/openOrders",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        orders = []
        for item in raw:
            orders.append(OrderInfo(
                order_id=str(item.get("orderId", "")),
                symbol=item.get("symbol"),
                side="Bid" if item.get("side", "").upper() == "BUY" else "Ask",
                order_type=item.get("type"),
                size=self.safe_decimal(item.get("origQty"), Decimal("0")),
                price=self.safe_decimal(item.get("price")),
                status=item.get("status"),
                filled_size=self.safe_decimal(item.get("executedQty"), Decimal("0")),
                remaining_size=self.safe_decimal(item.get("origQty"), Decimal("0")) - self.safe_decimal(item.get("executedQty"), Decimal("0")),
                client_order_id=item.get("clientOrderId"),
                created_at=self.safe_int(item.get("time")),
                time_in_force=item.get("timeInForce"),
                reduce_only=item.get("reduceOnly", False),
                raw=item
            ))
        
        return ApiResponse.ok(orders, raw=raw)

    def cancel_all_orders(self, symbol: str) -> ApiResponse:
        """取消指定交易對的所有訂單
        
        Returns:
            ApiResponse with data: CancelResult
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            msg = f"無法解析交易對: {symbol}"
            if suggestions:
                msg += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(msg)
        
        raw = self.make_request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params={"symbol": resolved_symbol},
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return ApiResponse(
                success=False,
                data=CancelResult(success=False, error_message=error.error_message, raw=raw),
                error_message=error.error_message,
                raw=raw
            )
        
        cancelled_count = len(raw) if isinstance(raw, list) else 1
        return ApiResponse.ok(CancelResult(success=True, cancelled_count=cancelled_count, raw=raw), raw=raw)

    def cancel_order(self, order_id: str, symbol: str) -> ApiResponse:
        """取消指定訂單
        
        Returns:
            ApiResponse with data: CancelResult
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            msg = f"無法解析交易對: {symbol}"
            if suggestions:
                msg += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(msg)
        
        raw = self.make_request(
            "DELETE",
            "/fapi/v1/order",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params={"symbol": resolved_symbol, "orderId": order_id},
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return ApiResponse(
                success=False,
                data=CancelResult(success=False, order_id=order_id, error_message=error.error_message, raw=raw),
                error_message=error.error_message,
                raw=raw
            )
        
        return ApiResponse.ok(CancelResult(success=True, order_id=order_id, cancelled_count=1, raw=raw), raw=raw)

    def get_ticker(self, symbol: str) -> ApiResponse:
        """獲取行情信息
        
        Returns:
            ApiResponse with data: TickerInfo
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            msg = f"無法解析交易對: {symbol}"
            if suggestions:
                msg += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(msg)
        
        raw = self.make_request(
            "GET",
            "/fapi/v1/ticker/24hr",
            params={"symbol": resolved_symbol},
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        ticker = TickerInfo(
            symbol=resolved_symbol,
            last_price=self.safe_decimal(raw.get("lastPrice") or raw.get("close") or raw.get("price")),
            bid_price=self.safe_decimal(raw.get("bidPrice")),
            ask_price=self.safe_decimal(raw.get("askPrice")),
            bid_size=self.safe_decimal(raw.get("bidQty")),
            ask_size=self.safe_decimal(raw.get("askQty")),
            volume_24h=self.safe_decimal(raw.get("volume")),
            turnover_24h=self.safe_decimal(raw.get("quoteVolume")),
            high_24h=self.safe_decimal(raw.get("highPrice")),
            low_24h=self.safe_decimal(raw.get("lowPrice")),
            change_percent_24h=self.safe_decimal(raw.get("priceChangePercent")),
            raw=raw
        )
        
        return ApiResponse.ok(ticker, raw=raw)

    def get_markets(self) -> ApiResponse:
        """獲取所有交易對信息
        
        Returns:
            ApiResponse with data: List[MarketInfo]
        """
        raw = self.make_request(
            "GET",
            "/fapi/v1/exchangeInfo",
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        markets = []
        for item in raw.get("symbols", []):
            filters = {f.get("filterType"): f for f in item.get("filters", [])}
            price_filter = filters.get("PRICE_FILTER", {})
            lot_size_filter = filters.get("LOT_SIZE", {})
            
            markets.append(MarketInfo(
                symbol=item.get("symbol"),
                base_asset=item.get("baseAsset"),
                quote_asset=item.get("quoteAsset"),
                market_type=item.get("contractType", "PERP"),
                status=item.get("status"),
                min_order_size=self.safe_decimal(lot_size_filter.get("minQty"), Decimal("0")),
                tick_size=self.safe_decimal(price_filter.get("tickSize"), Decimal("0.00000001")),
                base_precision=item.get("quantityPrecision", item.get("baseAssetPrecision", 6)),
                quote_precision=item.get("pricePrecision", item.get("quotePrecision", 6)),
                raw=item
            ))
        
        return ApiResponse.ok(markets, raw=raw)

    def get_order_book(self, symbol: str, limit: int = 20) -> ApiResponse:
        """獲取訂單簿
        
        Returns:
            ApiResponse with data: OrderBookInfo
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            msg = f"無法解析交易對: {symbol}"
            if suggestions:
                msg += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(msg)
        
        raw = self.make_request(
            "GET",
            "/fapi/v1/depth",
            params={"symbol": resolved_symbol, "limit": limit},
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        bids = []
        for level in raw.get("bids", []):
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                bids.append(OrderBookLevel(
                    price=self.safe_decimal(level[0], Decimal("0")),
                    quantity=self.safe_decimal(level[1], Decimal("0"))
                ))
        
        asks = []
        for level in raw.get("asks", []):
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                asks.append(OrderBookLevel(
                    price=self.safe_decimal(level[0], Decimal("0")),
                    quantity=self.safe_decimal(level[1], Decimal("0"))
                ))
        
        # 排序
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        order_book = OrderBookInfo(
            symbol=resolved_symbol,
            bids=bids,
            asks=asks,
            timestamp=self.safe_int(raw.get("T")),
            sequence=self.safe_int(raw.get("lastUpdateId")),
            raw=raw
        )
        
        return ApiResponse.ok(order_book, raw=raw)

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100) -> ApiResponse:
        """獲取成交歷史
        
        Returns:
            ApiResponse with data: List[TradeInfo]
        """
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                suggestions = self._find_symbol_suggestions(symbol)
                msg = f"無法解析交易對: {symbol}"
                if suggestions:
                    msg += f"。可能的交易對: {', '.join(suggestions)}"
                return ApiResponse.error(msg)
            params["symbol"] = resolved_symbol
        
        raw = self.make_request(
            "GET",
            "/fapi/v1/userTrades",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        trades = []
        for item in raw:
            trades.append(TradeInfo(
                trade_id=str(item.get("id", "")),
                order_id=str(item.get("orderId", "")),
                symbol=item.get("symbol"),
                side="Bid" if item.get("side", "").upper() == "BUY" else "Ask",
                size=self.safe_decimal(item.get("qty"), Decimal("0")),
                price=self.safe_decimal(item.get("price"), Decimal("0")),
                fee=self.safe_decimal(item.get("commission")),
                fee_asset=item.get("commissionAsset"),
                timestamp=self.safe_int(item.get("time")),
                is_maker=item.get("maker"),
                raw=item
            ))
        
        return ApiResponse.ok(trades, raw=raw)

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> ApiResponse:
        """獲取K線數據
        
        Returns:
            ApiResponse with data: List[KlineInfo]
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            msg = f"無法解析交易對: {symbol}"
            if suggestions:
                msg += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(msg)
        
        params = {"symbol": resolved_symbol, "interval": interval, "limit": limit}
        raw = self.make_request(
            "GET",
            "/fapi/v1/klines",
            params=params,
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        klines = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 6:
                klines.append(KlineInfo(
                    open_time=self.safe_int(item[0], 0),
                    close_time=self.safe_int(item[6], 0) if len(item) > 6 else self.safe_int(item[0], 0),
                    open_price=self.safe_decimal(item[1], Decimal("0")),
                    high_price=self.safe_decimal(item[2], Decimal("0")),
                    low_price=self.safe_decimal(item[3], Decimal("0")),
                    close_price=self.safe_decimal(item[4], Decimal("0")),
                    volume=self.safe_decimal(item[5], Decimal("0")),
                    quote_volume=self.safe_decimal(item[7]) if len(item) > 7 else None,
                    trades_count=self.safe_int(item[8]) if len(item) > 8 else None,
                    raw=item
                ))
        
        return ApiResponse.ok(klines, raw=raw)

    def get_market_limits(self, symbol: str) -> ApiResponse:
        """獲取市場限制信息
        
        Returns:
            ApiResponse with data: MarketInfo
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            msg = f"無法解析交易對: {symbol}"
            if suggestions:
                msg += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(msg)

        self._ensure_symbol_cache()
        symbol_info = self._market_info_cache.get(resolved_symbol)
        if not symbol_info:
            return ApiResponse.error(f"交易所返回的資料中找不到交易對 {resolved_symbol}")
        
        filters = {f.get("filterType"): f for f in symbol_info.get("filters", [])}
        price_filter = filters.get("PRICE_FILTER", {})
        lot_size_filter = filters.get("LOT_SIZE", {})
        
        market_info = MarketInfo(
            symbol=resolved_symbol,
            base_asset=symbol_info.get("baseAsset"),
            quote_asset=symbol_info.get("quoteAsset"),
            market_type=symbol_info.get("contractType", "PERP"),
            status=symbol_info.get("status"),
            min_order_size=self.safe_decimal(lot_size_filter.get("minQty"), Decimal("0")),
            tick_size=self.safe_decimal(price_filter.get("tickSize"), Decimal("0.00000001")),
            base_precision=symbol_info.get("quantityPrecision", symbol_info.get("baseAssetPrecision", 6)),
            quote_precision=symbol_info.get("pricePrecision", symbol_info.get("quotePrecision", 6)),
            raw=symbol_info
        )
        
        return ApiResponse.ok(market_info, raw=symbol_info)

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取持倉信息
        
        Returns:
            ApiResponse with data: List[PositionInfo]
        """
        params: Dict[str, Any] = {}
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                suggestions = self._find_symbol_suggestions(symbol)
                msg = f"無法解析交易對: {symbol}"
                if suggestions:
                    msg += f"。可能的交易對: {', '.join(suggestions)}"
                return ApiResponse.error(msg)
            params["symbol"] = resolved_symbol
        
        raw = self.make_request(
            "GET",
            "/fapi/v2/positionRisk",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        positions = []
        for item in raw:
            raw_amt = item.get("positionAmt", "0") or "0"
            try:
                pos_dec = Decimal(str(raw_amt))
            except (InvalidOperation, TypeError):
                pos_dec = Decimal("0")

            if pos_dec > 0:
                side = "LONG"
            elif pos_dec < 0:
                side = "SHORT"
            else:
                side = "FLAT"

            positions.append(PositionInfo(
                symbol=item.get("symbol"),
                side=side,
                size=abs(pos_dec),
                entry_price=self.safe_decimal(item.get("entryPrice")),
                mark_price=self.safe_decimal(item.get("markPrice")),
                liquidation_price=self.safe_decimal(item.get("liquidationPrice")),
                unrealized_pnl=self.safe_decimal(item.get("unRealizedProfit")),
                leverage=self.safe_decimal(item.get("leverage")),
                margin_mode="CROSS" if item.get("marginType") == "cross" else "ISOLATED",
                raw=item
            ))
        
        return ApiResponse.ok(positions, raw=raw)