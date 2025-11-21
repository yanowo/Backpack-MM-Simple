"""APEX Omni exchange REST client implementation."""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

import requests

from .base_client import BaseExchangeClient
from .proxy_utils import get_proxy_config
from logger import setup_logger

logger = setup_logger("api.apex_client")


class ApexClient(BaseExchangeClient):
    """REST client for the APEX Omni perpetual futures API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.passphrase = config.get("passphrase", "")
        self.base_url = config.get("base_url", "https://omni.apex.exchange/api")
        self.timeout = float(config.get("timeout", 10))
        self.max_retries = int(config.get("max_retries", 3))
        self.session = requests.Session()

        # 從環境變量讀取代理配置
        proxies = get_proxy_config()
        if proxies:
            self.session.proxies.update(proxies)
            logger.info(f"APEX 客户端已配置代理: {proxies}")

        self._symbol_cache: Dict[str, str] = {}
        self._market_info_cache: Dict[str, Dict[str, Any]] = {}

    def get_exchange_name(self) -> str:
        return "APEX"

    async def connect(self) -> None:
        logger.info("APEX 客户端已連接")

    async def disconnect(self) -> None:
        self.session.close()
        logger.info("APEX 客户端已斷開連接")

    def _current_timestamp(self) -> int:
        return int(time.time() * 1000)

    def _iso_timestamp(self) -> str:
        """Generate ISO 8601 timestamp for APEX API."""
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _normalize_order_fields(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize order fields to standard format."""
        if "id" in order and "order_id" not in order:
            order["order_id"] = order["id"]
        if "orderId" in order and "id" not in order:
            order["id"] = str(order["orderId"])

        side = order.get("side")
        if side:
            normalized = side.upper()
            if normalized == "BUY":
                order["side"] = "Bid"
            elif normalized == "SELL":
                order["side"] = "Ask"

        if "size" in order and "quantity" not in order:
            order["quantity"] = order["size"]

        return order

    def _lookup_key(self, symbol: str) -> str:
        """Generate a case-insensitive lookup key for exchange symbols."""
        return symbol.upper().replace("_", "-")

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

        # APEX returns perpetualContract array in configs
        contracts = info.get("data", {}).get("perpetualContract", []) if isinstance(info, dict) else []
        cache: Dict[str, str] = {}
        market_cache: Dict[str, Dict[str, Any]] = {}

        for item in contracts:
            actual_symbol = item.get("symbol")
            if not actual_symbol:
                continue
            cache[self._lookup_key(actual_symbol)] = actual_symbol
            market_cache[actual_symbol] = item

        self._symbol_cache = cache
        self._market_info_cache = market_cache

    def _resolve_symbol(self, symbol: Optional[str]) -> Optional[str]:
        """Resolve user provided symbol aliases to APEX native symbols."""
        if not symbol:
            return None

        self._ensure_symbol_cache()
        if not self._symbol_cache:
            return None

        # Try direct lookup first
        sanitized = symbol.strip().upper().replace("_", "-")
        resolved = self._symbol_cache.get(self._lookup_key(sanitized))
        if resolved:
            return resolved

        # Try without separator
        no_sep = sanitized.replace("-", "")
        for key, value in self._symbol_cache.items():
            if key.replace("-", "") == no_sep:
                return value

        return None

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

        sanitized = symbol.strip().upper().replace("_", "-")
        token = sanitized.replace("-", "")
        candidates: List[str] = []
        seen: Set[str] = set()

        # Fuzzy match by substring
        if token:
            for actual in self._market_info_cache.keys():
                if token in actual.upper().replace("-", "") and actual not in seen:
                    candidates.append(actual)
                    seen.add(actual)
                    if len(candidates) >= limit:
                        return candidates

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

    def _sign_request(self, request_path: str, method: str, timestamp: str, data: Dict[str, Any] = None) -> str:
        """Generate APEX API signature.

        Args:
            request_path: API endpoint path (e.g., /v3/account)
            method: HTTP method (GET, POST, DELETE)
            timestamp: ISO 8601 timestamp
            data: Request parameters

        Returns:
            Base64 encoded HMAC-SHA256 signature
        """
        if data:
            # Sort parameters alphabetically
            sorted_items = sorted(data.items(), key=lambda x: x[0])
            data_string = '&'.join(
                f'{key}={value}' for key, value in sorted_items if value is not None
            )
        else:
            data_string = ''

        # Build message: timestamp + method + path + data
        if method.upper() == 'GET':
            message = timestamp + method.upper() + request_path
        else:
            message = timestamp + method.upper() + request_path + data_string

        # Create HMAC-SHA256 signature with base64 encoded secret
        secret_bytes = base64.standard_b64encode(self.secret_key.encode('utf-8'))
        hashed = hmac.new(
            secret_bytes,
            msg=message.encode('utf-8'),
            digestmod=hashlib.sha256
        )

        return base64.standard_b64encode(hashed.digest()).decode()

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
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        if signed:
            if not self.api_key or not self.secret_key:
                return {"error": "缺少 API Key 或 Secret Key"}

            timestamp = self._iso_timestamp()
            signature = self._sign_request(endpoint, method, timestamp, payload if method.upper() != 'GET' else None)

            headers.update({
                'APEX-SIGNATURE': signature,
                'APEX-TIMESTAMP': timestamp,
                'APEX-API-KEY': self.api_key,
                'APEX-PASSPHRASE': self.passphrase or ''
            })

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
                    logger.warning("APEX API 達到速率限制，等待 %.1f 秒後重試", wait_time)
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
                    logger.warning("APEX API 請求異常 (%s)，重試中...", exc)
                    time.sleep(1)
                    continue
                return {"error": f"請求失敗: {exc}"}

        return {"error": "達到最大重試次數"}

    def get_deposit_address(self, blockchain: str) -> Dict[str, Any]:
        return {"error": "請使用 APEX 網頁界面獲取充值地址"}

    def get_balance(self) -> Dict[str, Any]:
        result = self.make_request(
            "GET",
            "/v3/account-balance",
            instruction=True,
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result

        # Parse APEX balance response
        data = result.get("data", {})
        balances: Dict[str, Dict[str, Any]] = {}

        # APEX uses totalEquityValue and availableBalance
        total_equity = float(data.get("totalEquityValue", 0))
        available = float(data.get("availableBalance", 0))
        locked = max(total_equity - available, 0.0)

        # Primary settlement currency
        balances["USDT"] = {
            "available": available,
            "locked": locked,
            "total": total_equity,
            "asset": "USDT",
            "raw": data,
        }

        return balances

    def get_collateral(self, subaccount_id: Optional[str] = None) -> Dict[str, Any]:
        result = self.make_request(
            "GET",
            "/v3/account",
            instruction=True,
            retry_count=self.max_retries,
        )
        if isinstance(result, dict) and "error" in result:
            return result

        data = result.get("data", {})

        return {
            "totalCollateral": data.get("totalEquityValue", "0"),
            "availableCollateral": data.get("availableBalance", "0"),
            "initialMargin": data.get("initialMargin", "0"),
            "maintenanceMargin": data.get("maintenanceMargin", "0"),
            "raw": data
        }

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

        # Map order types to APEX format
        type_mapping = {
            "LIMIT": "LIMIT",
            "MARKET": "MARKET",
            "STOP_LIMIT": "STOP_LIMIT",
            "STOP_MARKET": "STOP_MARKET",
        }
        normalized_type = type_mapping.get(order_type.upper(), order_type.upper())

        payload: Dict[str, Any] = {
            "symbol": resolved_symbol,
            "side": normalized_side,
            "type": normalized_type,
        }

        # Time in force
        post_only = order_details.get("postOnly", False)
        time_in_force = order_details.get("timeInForce")

        if post_only:
            payload["timeInForce"] = "POST_ONLY"
        elif time_in_force:
            tif_mapping = {
                "GTC": "GOOD_TIL_CANCEL",
                "FOK": "FILL_OR_KILL",
                "IOC": "IMMEDIATE_OR_CANCEL",
            }
            payload["timeInForce"] = tif_mapping.get(time_in_force.upper(), time_in_force.upper())
        else:
            payload["timeInForce"] = "GOOD_TIL_CANCEL"

        # Quantity
        quantity = order_details.get("quantity") or order_details.get("size")
        if quantity is not None:
            payload["size"] = str(quantity)

        # Price
        price = order_details.get("price")
        if price is not None:
            payload["price"] = str(price)

        # Client order ID
        if "clientId" in order_details:
            payload["clientOrderId"] = order_details["clientId"]

        # Reduce only
        if order_details.get("reduceOnly"):
            payload["reduceOnly"] = "true"

        result = self.make_request(
            "POST",
            "/v3/order",
            instruction=True,
            data=payload,
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return result

        return self._normalize_order_fields(result.get("data", result))

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                return self._unknown_symbol_error(symbol)
            params["symbol"] = resolved_symbol

        result = self.make_request(
            "GET",
            "/v3/open-orders",
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return result

        orders = result.get("data", {}).get("orders", [])
        normalized: List[Dict[str, Any]] = []
        for item in orders:
            normalized.append(self._normalize_order_fields(dict(item)))
        return normalized

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol."""
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)

        result = self.make_request(
            "POST",
            "/v3/delete-open-orders",
            instruction=True,
            data={"symbol": resolved_symbol},
            retry_count=self.max_retries,
        )
        return result

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)

        result = self.make_request(
            "POST",
            "/v3/delete-order",
            instruction=True,
            data={"id": order_id},
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return result

        return self._normalize_order_fields(result.get("data", result))

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)

        result = self.make_request(
            "GET",
            "/v3/ticker",
            params={"symbol": resolved_symbol},
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return result

        data = result.get("data", result)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        # Normalize to standard format
        if "lastPrice" not in data and "price" in data:
            data["lastPrice"] = data["price"]

        return data

    def get_markets(self) -> Dict[str, Any]:
        return self.make_request(
            "GET",
            "/v3/configs",
            retry_count=self.max_retries,
        )

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)

        result = self.make_request(
            "GET",
            "/v3/depth",
            params={"symbol": resolved_symbol, "limit": limit},
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return result

        data = result.get("data", result)
        bids = data.get("b", data.get("bids", []))
        asks = data.get("a", data.get("asks", []))

        # Sort bids descending, asks ascending
        try:
            bids = sorted(bids, key=lambda level: float(level[0]), reverse=True)
            asks = sorted(asks, key=lambda level: float(level[0]))
        except (ValueError, TypeError, IndexError):
            pass

        return {"bids": bids, "asks": asks}

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100) -> Any:
        params = {"limit": limit}
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                return self._unknown_symbol_error(symbol)
            params["symbol"] = resolved_symbol

        return self.make_request(
            "GET",
            "/v3/fills",
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> Any:
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            return self._unknown_symbol_error(symbol)

        # Map interval to APEX format
        interval_mapping = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "4h": "240",
            "1d": "D",
        }
        apex_interval = interval_mapping.get(interval, interval)

        params = {
            "symbol": resolved_symbol,
            "interval": apex_interval,
            "limit": limit
        }

        return self.make_request(
            "GET",
            "/v3/klines",
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

        # Parse symbol (e.g., BTC-USDT -> base=BTC, quote=USDT)
        parts = resolved_symbol.split("-")
        base_asset = parts[0] if len(parts) > 0 else resolved_symbol
        quote_asset = parts[1] if len(parts) > 1 else "USDT"

        return {
            "symbol": resolved_symbol,
            "base_asset": base_asset,
            "quote_asset": quote_asset,
            "market_type": "PERP",
            "status": "TRADING",
            "min_order_size": symbol_info.get("minOrderSize", "0.001"),
            "tick_size": symbol_info.get("tickSize", "0.1"),
            "base_precision": int(symbol_info.get("stepSize", "0.001").count('0') if '.' in str(symbol_info.get("stepSize", "0.001")) else 3),
            "quote_precision": int(symbol_info.get("tickSize", "0.1").count('0') if '.' in str(symbol_info.get("tickSize", "0.1")) else 1),
        }

    def get_positions(self, symbol: Optional[str] = None) -> Any:
        result = self.make_request(
            "GET",
            "/v3/account",
            instruction=True,
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return result

        data = result.get("data", {})
        positions_raw = data.get("openPositions", [])

        normalized: List[Dict[str, Any]] = []
        for item in positions_raw:
            item_symbol = item.get("symbol", "")

            # Filter by symbol if specified
            if symbol:
                resolved = self._resolve_symbol(symbol)
                if resolved and item_symbol != resolved:
                    continue

            raw_size = item.get("size", "0") or "0"
            try:
                pos_dec = Decimal(str(raw_size))
            except (InvalidOperation, TypeError):
                pos_dec = Decimal("0")

            side_str = item.get("side", "").upper()
            if side_str == "LONG" or pos_dec > 0:
                mapped_side = "LONG"
            elif side_str == "SHORT" or pos_dec < 0:
                mapped_side = "SHORT"
            else:
                mapped_side = "FLAT"

            long_dec = abs(pos_dec) if mapped_side == "LONG" else Decimal("0")
            short_dec = abs(pos_dec) if mapped_side == "SHORT" else Decimal("0")

            entry_price = item.get("entryPrice")
            unrealized = item.get("unrealizedPnl", item.get("unrealizedProfit"))

            normalized.append({
                "symbol": item_symbol,
                "side": mapped_side,
                "positionSide": mapped_side,
                "netQuantity": self._decimal_to_str(pos_dec if mapped_side == "LONG" else -abs(pos_dec)),
                "longQuantity": self._decimal_to_str(long_dec),
                "shortQuantity": self._decimal_to_str(short_dec),
                "size": self._decimal_to_str(abs(pos_dec)),
                "entryPrice": entry_price,
                "pnlUnrealized": unrealized,
                "unrealizedPnl": unrealized,
                "raw": item,
            })

        return normalized
