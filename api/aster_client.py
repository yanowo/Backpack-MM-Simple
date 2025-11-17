"""Aster exchange REST client implementation."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional, Set
from decimal import Decimal, InvalidOperation
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

import requests

from .base_client import BaseExchangeClient
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

        # 代理配置
        http_proxy = config.get("http_proxy")
        https_proxy = config.get("https_proxy")
        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
                # 如果没有单独设置 https_proxy，HTTPS 也使用 http_proxy
                if not https_proxy:
                    proxies['https'] = http_proxy
            if https_proxy:
                proxies['https'] = https_proxy
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

        info = self.get_markets()
        if isinstance(info, dict) and info.get("error"):
            logger.error("獲取交易對列表失敗: %s", info["error"])
            self._symbol_cache = {}
            self._market_info_cache = {}
            return

        symbols = info.get("symbols", []) if isinstance(info, dict) else []
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

    def get_deposit_address(self, blockchain: str) -> Dict[str, Any]:
        return {"error": "Aster Futures 不支持通過此API獲取充值地址"}

    def get_balance(self) -> Dict[str, Any]:
        result = self.make_request(
            "GET",
            "/fapi/v2/balance",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result
        balances: Dict[str, Dict[str, Any]] = {}
        for item in result:
            asset = item.get("asset")
            if not asset:
                continue
            available = float(item.get("availableBalance", 0))
            total = float(item.get("balance", 0))
            locked = max(total - available, 0.0)
            balances[asset] = {
                "available": available,
                "locked": locked,
                "total": total,
                "raw": item,
            }
        return balances

    def get_collateral(self, subaccount_id: Optional[str] = None) -> Dict[str, Any]:
        result = self.make_request(
            "GET",
            "/fapi/v4/account",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result and "assets" not in result:
            return result
        assets_payload = []
        for item in result.get("assets", []):
            symbol = item.get("asset")
            if not symbol:
                continue
            assets_payload.append(
                {
                    "symbol": symbol,
                    "totalQuantity": item.get("marginBalance", "0"),
                    "availableQuantity": item.get("availableBalance", "0"),
                    "walletBalance": item.get("walletBalance", "0"),
                    "unrealizedPnl": item.get("unrealizedProfit", "0"),
                }
            )
        return {"assets": assets_payload, "raw": result}

    def execute_order(self, order_details: Dict[str, Any]) -> Dict[str, Any]:
        symbol = order_details.get("symbol")
        if not symbol:
            return {"error": "缺少交易對", "status_code": 400}
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)

        side = order_details.get("side")
        if not side:
            return {"error": "缺少買賣方向"}
        if side.lower() in {"bid", "buy"}:
            normalized_side = "BUY"
        elif side.lower() in {"ask", "sell"}:
            normalized_side = "SELL"
        else:
            return {"error": f"不支持的方向: {side}"}

        order_type = order_details.get("orderType") or order_details.get("type")
        if not order_type:
            return {"error": "缺少訂單類型"}
        normalized_type = order_type.upper()

        payload: Dict[str, Any] = {"symbol": resolved_symbol, "side": normalized_side, "type": normalized_type}

        # 處理 postOnly（僅掛單模式）
        post_only = order_details.get("postOnly", False)
        time_in_force = order_details.get("timeInForce")
        
        if normalized_type == "LIMIT":
            # postOnly 模式使用 GTX（Good Till Crossing）- 無法成為掛單方就撤銷
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

        result = self.make_request(
            "POST",
            "/fapi/v1/order",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params=payload,
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result
        return self._normalize_order_fields(result)

    def execute_order_batch(self, orders_details: List[Dict[str, Any]]) -> Any:
        """批量執行訂單

        Aster 批量下單限制：每批最多 5 個訂單

        Args:
            orders_details: 訂單詳情列表

        Returns:
            成功訂單列表或錯誤信息
        """
        if not orders_details:
            return {"error": "訂單列表為空"}

        # Aster 限制每批最多 5 個訂單
        if len(orders_details) > 5:
            logger.warning("Aster 批量下單限制為 5 個訂單，當前 %d 個，將拆分為多批", len(orders_details))

        # 將訂單拆分為多批（每批最多 5 個）
        batch_size = 5
        all_results = []
        all_errors = []

        for batch_start in range(0, len(orders_details), batch_size):
            batch = orders_details[batch_start:batch_start + batch_size]

            # 構建批量訂單請求
            batch_orders = []

            for order_details in batch:
                symbol = order_details.get("symbol")
                if not symbol:
                    logger.warning("跳過無效訂單: 缺少交易對")
                    all_errors.append({"error": "缺少交易對", "order": order_details})
                    continue

                resolved_symbol = self._resolve_symbol(symbol)
                if not resolved_symbol:
                    logger.warning("跳過無效訂單: 未知交易對 %s", symbol)
                    all_errors.append({"error": f"未知交易對: {symbol}", "order": order_details})
                    continue

                side = order_details.get("side")
                if not side:
                    logger.warning("跳過無效訂單: 缺少買賣方向")
                    all_errors.append({"error": "缺少買賣方向", "order": order_details})
                    continue

                # 標準化方向
                if side.lower() in {"bid", "buy"}:
                    normalized_side = "BUY"
                elif side.lower() in {"ask", "sell"}:
                    normalized_side = "SELL"
                else:
                    logger.warning("跳過無效訂單: 不支持的方向 %s", side)
                    all_errors.append({"error": f"不支持的方向: {side}", "order": order_details})
                    continue

                order_type = order_details.get("orderType") or order_details.get("type")
                if not order_type:
                    logger.warning("跳過無效訂單: 缺少訂單類型")
                    all_errors.append({"error": "缺少訂單類型", "order": order_details})
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
                    # postOnly 模式使用 GTX（Good Till Crossing）
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
                    all_errors.append({"error": "缺少數量", "order": order_details})
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
            result = self.make_request(
                "POST",
                "/fapi/v1/batchOrders",
                api_key=self.api_key,
                secret_key=self.secret_key,
                instruction=True,
                data={"batchOrders": json.dumps(batch_orders)},
                retry_count=self.max_retries
            )

            # 處理響應
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # 檢查是否是錯誤
                        if "code" in item and "msg" in item:
                            all_errors.append(item)
                            logger.warning("批量下單部分失敗: %s", item.get("msg"))
                        else:
                            # 成功的訂單
                            normalized_order = self._normalize_order_fields(dict(item))
                            all_results.append(normalized_order)
                logger.info("批量下單完成: %d 成功, %d 失敗", len(all_results), len(all_errors))
            elif isinstance(result, dict) and "error" in result:
                # 整個批次失敗
                logger.error("批量下單失敗: %s", result["error"])
                all_errors.append({"error": result["error"], "batch": batch_orders})

        # 返回結果
        if all_results:
            # 有成功的訂單
            if all_errors:
                # 部分成功
                return {
                    "orders": all_results,
                    "errors": all_errors,
                    "partial_success": True
                }
            else:
                # 全部成功
                return all_results
        else:
            # 全部失敗
            return {"error": "批量下單全部失敗", "errors": all_errors}

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                return self._unknown_symbol_error(symbol)
            params["symbol"] = resolved_symbol
        result = self.make_request(
            "GET",
            "/fapi/v1/openOrders",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result
        normalized: List[Dict[str, Any]] = []
        for item in result:
            normalized.append(self._normalize_order_fields(dict(item)))
        return normalized

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """取消指定交易對的所有訂單"""
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)
        result = self.make_request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params={"symbol": resolved_symbol},
            retry_count=self.max_retries,
        )
        if isinstance(result, list):
            return [self._normalize_order_fields(dict(item)) for item in result]
        return result

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)
        result = self.make_request(
            "DELETE",
            "/fapi/v1/order",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params={"symbol": resolved_symbol, "orderId": order_id},
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result
        return self._normalize_order_fields(result)

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)
        result = self.make_request(
            "GET",
            "/fapi/v1/ticker/24hr",
            params={"symbol": resolved_symbol},
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result
        if isinstance(result, dict) and "lastPrice" not in result:
            result["lastPrice"] = result.get("close", result.get("price"))
        return result

    def get_markets(self) -> Dict[str, Any]:
        return self.make_request(
            "GET",
            "/fapi/v1/exchangeInfo",
            retry_count=self.max_retries,
        )

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)
        result = self.make_request(
            "GET",
            "/fapi/v1/depth",
            params={"symbol": resolved_symbol, "limit": limit},
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result
        bids = result.get("bids", [])
        asks = result.get("asks", [])
        try:
            bids = sorted(bids, key=lambda level: float(level[0]))
        except (ValueError, TypeError):
            pass
        result["bids"] = bids
        result["asks"] = asks
        return result

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100) -> Any:
        params = {"limit": limit}
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                return self._unknown_symbol_error(symbol)
            params["symbol"] = resolved_symbol
        return self.make_request(
            "GET",
            "/fapi/v1/userTrades",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> Any:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)
        params = {"symbol": resolved_symbol, "interval": interval, "limit": limit}
        return self.make_request(
            "GET",
            "/fapi/v1/klines",
            params=params,
            retry_count=self.max_retries,
        )

    def get_market_limits(self, symbol: str) -> Optional[Dict[str, Any]]:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            self._unknown_symbol_error(symbol)
            return None

        self._ensure_symbol_cache()
        symbol_info = self._market_info_cache.get(resolved_symbol)
        if not symbol_info:
            logger.error("交易所返回的資料中找不到交易對 %s", resolved_symbol)
            return None
        filters = {f.get("filterType"): f for f in symbol_info.get("filters", [])}
        price_filter = filters.get("PRICE_FILTER", {})
        lot_size_filter = filters.get("LOT_SIZE", {})
        return {
            "symbol": resolved_symbol,
            "base_asset": symbol_info.get("baseAsset"),
            "quote_asset": symbol_info.get("quoteAsset"),
            "market_type": symbol_info.get("contractType", "PERP"),
            "status": symbol_info.get("status"),
            "min_order_size": lot_size_filter.get("minQty", "0"),
            "tick_size": price_filter.get("tickSize", "0.00000001"),
            "base_precision": symbol_info.get("quantityPrecision", symbol_info.get("baseAssetPrecision", 6)),
            "quote_precision": symbol_info.get("pricePrecision", symbol_info.get("quotePrecision", 6)),
        }

    def get_positions(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                return self._unknown_symbol_error(symbol)
            params["symbol"] = resolved_symbol
        result = self.make_request(
            "GET",
            "/fapi/v2/positionRisk",
            api_key=self.api_key,
            secret_key=self.secret_key,
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result
        normalized: List[Dict[str, Any]] = []
        for item in result:
            raw_amt = item.get("positionAmt", "0") or "0"
            try:
                pos_dec = Decimal(str(raw_amt))
            except (InvalidOperation, TypeError):
                pos_dec = Decimal("0")
            pos_amt = float(pos_dec)

            if pos_amt > 0:
                mapped_side = "LONG"
            elif pos_amt < 0:
                mapped_side = "SHORT"
            else:
                mapped_side = "FLAT"

            long_dec = pos_dec if pos_dec > 0 else Decimal("0")
            short_dec = -pos_dec if pos_dec < 0 else Decimal("0")
            mark_price = item.get("markPrice")
            entry_price = item.get("entryPrice")
            unrealized = item.get("unRealizedProfit")
            position_side = item.get("positionSide") or mapped_side

            normalized.append(
                {
                    "symbol": item.get("symbol"),
                    "side": mapped_side,
                    "positionSide": position_side,
                    "netQuantity": self._decimal_to_str(pos_dec),
                    "longQuantity": self._decimal_to_str(long_dec),
                    "shortQuantity": self._decimal_to_str(short_dec),
                    "size": self._decimal_to_str(abs(pos_dec)),
                    "entryPrice": entry_price,
                    "markPrice": mark_price,
                    "pnlUnrealized": unrealized,
                    "unrealizedPnl": unrealized,
                    "leverage": item.get("leverage"),
                    "raw": item,
                }
            )
        return normalized