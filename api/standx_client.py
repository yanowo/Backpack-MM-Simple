"""StandX Perps REST client implementation."""
from __future__ import annotations

import base64
import json
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

import nacl.signing
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
)
from .proxy_utils import get_proxy_config
from logger import setup_logger

logger = setup_logger("api.standx_client")


class StandxClient(BaseExchangeClient):
    """REST client for StandX Perps API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_token = (
            config.get("api_token")
            or config.get("api_key")
            or config.get("jwt")
            or config.get("token")
        )
        self.signing_key = (
            config.get("signing_key")
            or config.get("secret_key")
        )
        self.base_url = (config.get("base_url") or "https://perps.standx.com").rstrip("/")
        self.timeout = float(config.get("timeout", 10))
        self.max_retries = int(config.get("max_retries", 3))
        self.session_id = config.get("session_id") or str(uuid.uuid4())
        self.default_symbol = config.get("default_symbol")
        self.margin_asset = config.get("margin_asset") or "DUSD"
        self.session = requests.Session()

        proxies = get_proxy_config()
        if proxies:
            self.session.proxies.update(proxies)
            logger.info("StandX 客户端已配置代理: %s", proxies)

        self._symbol_cache: Dict[str, Dict[str, Any]] = {}
        self._signing_key_cache: Optional[bytes] = None

        self._base58_alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        self._base58_index = {char: idx for idx, char in enumerate(self._base58_alphabet)}

    def get_exchange_name(self) -> str:
        return "StandX"

    async def connect(self) -> None:
        logger.info("StandX 客户端已連接")

    async def disconnect(self) -> None:
        self.session.close()
        logger.info("StandX 客户端已斷開連接")

    # ==================== Low-level helpers ====================

    def _normalize_symbol_key(self, symbol: str) -> str:
        return symbol.strip().upper().replace("_", "-").replace("/", "-")

    def _ensure_symbol_cache(self) -> None:
        if self._symbol_cache:
            return

        response = self.get_markets()
        if not response.success:
            logger.warning("獲取 StandX 交易對列表失敗: %s", response.error_message)
            return

        markets = response.raw or {}
        if isinstance(markets, list):
            items = markets
        elif isinstance(markets, dict):
            items = markets.get("data") or markets.get("result") or markets.get("symbols") or []
        else:
            items = []
        for item in items:
            symbol = item.get("symbol")
            if not symbol:
                continue
            key = self._normalize_symbol_key(symbol)
            self._symbol_cache[key] = item

    def _resolve_symbol(self, symbol: Optional[str]) -> Optional[str]:
        if not symbol:
            return None
        key = self._normalize_symbol_key(symbol)
        self._ensure_symbol_cache()
        if key in self._symbol_cache:
            return self._symbol_cache[key].get("symbol") or symbol
        return symbol

    def _parse_decimal(self, value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    def _parse_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_timestamp_ms(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            if value.isdigit():
                return int(value)
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                return None
        return None

    def _format_decimal(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            dec = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None
        normalized = dec.normalize() if dec != 0 else Decimal("0")
        text = format(normalized, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    def _base58_decode(self, text: str) -> Optional[bytes]:
        if not text:
            return None
        num = 0
        for char in text:
            if char not in self._base58_index:
                return None
            num = num * 58 + self._base58_index[char]

        # Convert integer to bytes
        if num == 0:
            decoded = b""
        else:
            decoded = num.to_bytes((num.bit_length() + 7) // 8, "big")

        # Add leading zero bytes for each leading '1'
        padding = len(text) - len(text.lstrip("1"))
        return b"\x00" * padding + decoded

    def _decode_signing_key(self) -> Optional[bytes]:
        if self._signing_key_cache:
            return self._signing_key_cache
        if not self.signing_key:
            return None

        key_text = str(self.signing_key).strip()
        if not key_text:
            return None

        # Try base58 first if all characters match base58 alphabet
        decoded: Optional[bytes] = None
        if all(char in self._base58_alphabet for char in key_text):
            decoded = self._base58_decode(key_text)
            if decoded and len(decoded) not in (32, 64):
                decoded = None

        if not decoded:
            try:
                decoded = base64.b64decode(key_text)
            except Exception as exc:
                logger.error("StandX 簽名密鑰解碼失敗: %s", exc)
                return None

        if not decoded or len(decoded) < 32:
            logger.error("StandX 簽名密鑰長度不足，無法簽名")
            return None

        self._signing_key_cache = decoded
        return decoded

    def _sign_message(self, message: str) -> Optional[str]:
        decoded = self._decode_signing_key()
        if not decoded:
            return None

        seed = decoded[:32]
        try:
            signing_key = nacl.signing.SigningKey(seed)
            signature = signing_key.sign(message.encode("utf-8")).signature
            return base64.b64encode(signature).decode("utf-8")
        except Exception as exc:
            logger.error("StandX 簽名失敗: %s", exc)
            return None

    def _extract_error_message(self, payload: Any) -> Optional[str]:
        if isinstance(payload, dict):
            if payload.get("code") not in (None, 0):
                return payload.get("message") or payload.get("msg") or str(payload.get("code"))
            if payload.get("success") is False:
                return payload.get("message") or payload.get("msg") or "Request failed"
            if payload.get("status") in {"error", "failed"}:
                return payload.get("message") or payload.get("msg") or "Request failed"
        return None

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
        method_upper = method.upper()
        params = {k: v for k, v in (params or {}).items() if v is not None}
        data = {k: v for k, v in (data or {}).items() if v is not None}

        is_private = bool(instruction)
        token = api_key or self.api_token
        signing_key = secret_key or self.signing_key

        headers: Dict[str, str] = {
            "Accept": "application/json",
        }
        if method_upper in {"POST", "PUT", "DELETE"}:
            headers["Content-Type"] = "application/json"

        if is_private:
            if not token:
                return {"error": "缺少 StandX API Token (JWT)"}
            headers["Authorization"] = f"Bearer {token}"

        payload_str = ""
        if method_upper in {"POST", "PUT", "DELETE"}:
            payload_str = json.dumps(data, separators=(",", ":"), ensure_ascii=False) if data else ""

        should_sign = is_private and bool(signing_key)
        if is_private and method_upper in {"POST", "PUT", "DELETE"} and not signing_key:
            return {"error": "StandX 私有請求缺少簽名密鑰"}

        if should_sign:
            request_id = uuid.uuid4().hex
            timestamp = str(int(time.time() * 1000))
            message = f"v1,{request_id},{timestamp},{payload_str}"
            signature = self._sign_message(message)
            if not signature:
                return {"error": "StandX 請求簽名失敗"}
            headers.update({
                "x-request-sign-version": "v1",
                "x-request-id": request_id,
                "x-request-timestamp": timestamp,
                "x-request-signature": signature,
            })

        if endpoint in {"/api/new_order", "/api/cancel_order", "/api/cancel_orders"} and self.session_id:
            headers["x-session-id"] = self.session_id

        retry_total = retry_count or self.max_retries
        for attempt in range(retry_total):
            try:
                if method_upper == "GET":
                    response = self.session.request(
                        method_upper,
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout,
                    )
                else:
                    response = self.session.request(
                        method_upper,
                        url,
                        params=params if params else None,
                        data=payload_str if payload_str else None,
                        headers=headers,
                        timeout=self.timeout,
                    )
            except Exception as exc:
                logger.warning("StandX 請求失敗 (%s/%s): %s", attempt + 1, retry_total, exc)
                time.sleep(min(1 * (2 ** attempt), 8))
                continue

            if 200 <= response.status_code < 300:
                try:
                    payload = response.json() if response.text else {}
                except ValueError:
                    return {"error": "StandX 響應解析失敗", "status_code": response.status_code}

                error_message = self._extract_error_message(payload)
                if error_message:
                    return {"error": error_message, "status_code": response.status_code, "raw": payload}
                return payload

            if response.status_code == 429:
                wait_time = min(1 * (2 ** attempt), 8)
                logger.warning("StandX API 達到速率限制，等待 %.1f 秒後重試", wait_time)
                time.sleep(wait_time)
                continue

            try:
                error_payload = response.json()
                message = error_payload.get("message") or error_payload.get("msg") or str(error_payload)
            except ValueError:
                error_payload = response.text
                message = response.text or f"HTTP {response.status_code}"
            return {
                "error": f"{message}",
                "status_code": response.status_code,
                "raw": error_payload,
            }

        return {"error": "StandX 請求失敗，已超過重試次數"}

    # ==================== Standardized methods ====================
    def get_balance(self) -> ApiResponse:
        raw = self.make_request("GET", "/api/query_balance", instruction=True)
        error = self._check_raw_error(raw)
        if error:
            return error

        data = None
        if isinstance(raw, dict):
            data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        if not isinstance(data, dict):
            return ApiResponse.error("StandX balance response malformed", raw=raw)

        asset = data.get("asset") or self.margin_asset
        total = self._parse_decimal(data.get("balance") or data.get("total"))
        available = self._parse_decimal(data.get("cross_available") or data.get("available"))
        locked = None
        if total is not None and available is not None:
            locked = total - available

        balance = BalanceInfo(
            asset=str(asset),
            available=available or Decimal("0"),
            locked=locked or Decimal("0"),
            total=total or (available or Decimal("0")),
            raw=data,
        )
        return ApiResponse.ok([balance], raw=raw)

    def get_collateral(self, subaccount_id: Optional[str] = None) -> ApiResponse:
        raw = self.make_request("GET", "/api/query_balance", instruction=True)
        error = self._check_raw_error(raw)
        if error:
            return error

        data = None
        if isinstance(raw, dict):
            data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        if not isinstance(data, dict):
            return ApiResponse.error("StandX balance response malformed", raw=raw)

        asset = data.get("asset") or self.margin_asset
        total = self._parse_decimal(data.get("balance") or data.get("total"))
        available = self._parse_decimal(data.get("cross_available") or data.get("available"))

        collateral = CollateralInfo(
            asset=str(asset),
            total_collateral=total or Decimal("0"),
            free_collateral=available or Decimal("0"),
            initial_margin=self._parse_decimal(data.get("cross_margin")),
            maintenance_margin=self._parse_decimal(data.get("isolated_margin")),
            account_value=self._parse_decimal(data.get("equity")),
            unrealized_pnl=self._parse_decimal(data.get("upnl")),
            raw=data,
        )
        return ApiResponse.ok([collateral], raw=raw)

    def get_markets(self) -> ApiResponse:
        params: Dict[str, Any] = {}
        if self.default_symbol:
            params["symbol"] = self.default_symbol
        raw = self.make_request("GET", "/api/query_symbol_info", params=params or None)
        error = self._check_raw_error(raw)
        if error:
            return error

        items = None
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get("data") or raw.get("result") or raw.get("symbols")
        if not isinstance(items, list):
            return ApiResponse.error("StandX symbol info response malformed", raw=raw)

        markets: List[MarketInfo] = []
        for item in items:
            symbol = item.get("symbol")
            if not symbol:
                continue
            base_asset = item.get("base_asset") or item.get("baseAsset")
            quote_asset = item.get("quote_asset") or item.get("quoteAsset")
            price_decimals = self._parse_int(item.get("price_tick_decimals") or item.get("price_precision"))
            qty_decimals = self._parse_int(item.get("qty_tick_decimals") or item.get("quantity_precision"))
            tick_size = Decimal("1") / (Decimal(10) ** price_decimals) if price_decimals is not None else Decimal("0.00000001")
            step_size = Decimal("1") / (Decimal(10) ** qty_decimals) if qty_decimals is not None else None

            market = MarketInfo(
                symbol=symbol,
                base_asset=base_asset or "",
                quote_asset=quote_asset or "",
                market_type="PERP",
                status="TRADING" if item.get("enabled", True) else "OFFLINE",
                min_order_size=self._parse_decimal(item.get("min_order_qty")) or Decimal("0"),
                max_order_size=self._parse_decimal(item.get("max_order_qty")),
                tick_size=tick_size,
                step_size=step_size,
                base_precision=qty_decimals or 8,
                quote_precision=price_decimals or 8,
                min_notional=self._parse_decimal(item.get("min_notional")),
                maker_fee=self._parse_decimal(item.get("maker_fee")),
                taker_fee=self._parse_decimal(item.get("taker_fee")),
                raw=item,
            )
            markets.append(market)
            key = self._normalize_symbol_key(symbol)
            self._symbol_cache[key] = item

        return ApiResponse.ok(markets, raw=raw)

    def get_market_limits(self, symbol: str) -> ApiResponse:
        resolved_symbol = self._resolve_symbol(symbol)
        raw = self.make_request("GET", "/api/query_symbol_info", params={"symbol": resolved_symbol})
        error = self._check_raw_error(raw)
        if error:
            return error

        items: Optional[List[Dict[str, Any]]] = None
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            candidate = raw.get("data") or raw.get("result") or raw.get("symbols")
            if isinstance(candidate, list):
                items = candidate
            elif raw.get("symbol"):
                items = [raw]
        if not isinstance(items, list) or not items:
            return ApiResponse.error(f"StandX 未找到交易對資訊: {symbol}", raw=raw)

        item = items[0]
        base_asset = item.get("base_asset") or item.get("baseAsset")
        quote_asset = item.get("quote_asset") or item.get("quoteAsset")
        price_decimals = self._parse_int(item.get("price_tick_decimals") or item.get("price_precision"))
        qty_decimals = self._parse_int(item.get("qty_tick_decimals") or item.get("quantity_precision"))
        tick_size = Decimal("1") / (Decimal(10) ** price_decimals) if price_decimals is not None else Decimal("0.00000001")
        step_size = Decimal("1") / (Decimal(10) ** qty_decimals) if qty_decimals is not None else None

        market = MarketInfo(
            symbol=item.get("symbol") or symbol,
            base_asset=base_asset or "",
            quote_asset=quote_asset or "",
            market_type="PERP",
            status="TRADING" if item.get("enabled", True) else "OFFLINE",
            min_order_size=self._parse_decimal(item.get("min_order_qty")) or Decimal("0"),
            max_order_size=self._parse_decimal(item.get("max_order_qty")),
            tick_size=tick_size,
            step_size=step_size,
            base_precision=qty_decimals or 8,
            quote_precision=price_decimals or 8,
            min_notional=self._parse_decimal(item.get("min_notional")),
            maker_fee=self._parse_decimal(item.get("maker_fee")),
            taker_fee=self._parse_decimal(item.get("taker_fee")),
            raw=item,
        )
        return ApiResponse.ok(market, raw=raw)

    def get_ticker(self, symbol: str) -> ApiResponse:
        resolved_symbol = self._resolve_symbol(symbol)
        raw = self.make_request("GET", "/api/query_symbol_price", params={"symbol": resolved_symbol})
        error = self._check_raw_error(raw)
        if error:
            return error

        data = None
        if isinstance(raw, dict):
            data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        if not isinstance(data, dict):
            return ApiResponse.error("StandX ticker response malformed", raw=raw)

        bid_price = self._parse_decimal(data.get("spread_bid") or data.get("bid_price") or data.get("bid"))
        ask_price = self._parse_decimal(data.get("spread_ask") or data.get("ask_price") or data.get("ask"))
        spread = data.get("spread")
        if isinstance(spread, (list, tuple)) and len(spread) >= 2:
            if bid_price is None:
                bid_price = self._parse_decimal(spread[0])
            if ask_price is None:
                ask_price = self._parse_decimal(spread[1])
        last_price = self._parse_decimal(data.get("last_price") or data.get("price"))
        if last_price is None and bid_price and ask_price:
            last_price = (bid_price + ask_price) / 2

        ticker = TickerInfo(
            symbol=data.get("symbol") or symbol,
            last_price=last_price,
            bid_price=bid_price,
            ask_price=ask_price,
            mark_price=self._parse_decimal(data.get("mark_price")),
            index_price=self._parse_decimal(data.get("index_price")),
            timestamp=self._parse_timestamp_ms(data.get("timestamp") or data.get("time")),
            raw=data,
        )
        return ApiResponse.ok(ticker, raw=raw)

    def get_order_book(self, symbol: str, limit: int = 20) -> ApiResponse:
        resolved_symbol = self._resolve_symbol(symbol)
        raw = self.make_request("GET", "/api/query_depth_book", params={"symbol": resolved_symbol})
        error = self._check_raw_error(raw)
        if error:
            return error

        data = None
        if isinstance(raw, dict):
            data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        if not isinstance(data, dict):
            return ApiResponse.error("StandX order book response malformed", raw=raw)

        bids_raw = data.get("bids") or []
        asks_raw = data.get("asks") or []
        bids: List[OrderBookLevel] = []
        asks: List[OrderBookLevel] = []

        for bid in bids_raw[:limit]:
            try:
                price = Decimal(str(bid[0]))
                qty = Decimal(str(bid[1]))
                bids.append(OrderBookLevel(price=price, quantity=qty))
            except Exception:
                continue

        for ask in asks_raw[:limit]:
            try:
                price = Decimal(str(ask[0]))
                qty = Decimal(str(ask[1]))
                asks.append(OrderBookLevel(price=price, quantity=qty))
            except Exception:
                continue

        if bids:
            bids.sort(key=lambda level: level.price, reverse=True)
        if asks:
            asks.sort(key=lambda level: level.price)

        order_book = OrderBookInfo(
            symbol=data.get("symbol") or symbol,
            bids=bids,
            asks=asks,
            timestamp=self._parse_timestamp_ms(data.get("timestamp") or data.get("time")),
            raw=data,
        )
        return ApiResponse.ok(order_book, raw=raw)

    def get_open_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        params: Dict[str, Any] = {}
        resolved_symbol = self._resolve_symbol(symbol) if symbol else None
        if resolved_symbol:
            params["symbol"] = resolved_symbol

        raw = self.make_request("GET", "/api/query_open_orders", params=params or None, instruction=True)
        error = self._check_raw_error(raw)
        if error:
            return error

        data = raw.get("data") if isinstance(raw, dict) else None
        orders_raw = None
        if isinstance(raw, list):
            orders_raw = raw
        elif isinstance(data, dict):
            orders_raw = data.get("result")
        if orders_raw is None and isinstance(raw, dict):
            orders_raw = raw.get("result")

        if not isinstance(orders_raw, list):
            return ApiResponse.error("StandX open orders response malformed", raw=raw)

        orders: List[OrderInfo] = []
        for item in orders_raw:
            qty = self._parse_decimal(item.get("qty"))
            filled = self._parse_decimal(item.get("filled_qty"))
            if qty is None:
                qty = Decimal("0")
            if filled is None:
                filled = Decimal("0")
            side_raw = str(item.get("side") or "").lower()
            side = "BUY" if side_raw in ("buy", "bid", "long") else "SELL"
            status_raw = str(item.get("status") or "").upper()
            status_map = {
                "NEW": "NEW",
                "PARTIALLY_FILLED": "PARTIALLY_FILLED",
                "FILLED": "FILLED",
                "CANCELED": "CANCELLED",
                "CANCELLED": "CANCELLED",
                "REJECTED": "REJECTED",
                "EXPIRED": "CANCELLED",
            }
            orders.append(
                OrderInfo(
                    order_id=str(item.get("order_id") or item.get("id") or ""),
                    symbol=item.get("symbol") or symbol or "",
                    side=side,
                    order_type=str(item.get("order_type") or item.get("type") or "LIMIT").upper(),
                    size=qty,
                    price=self._parse_decimal(item.get("price")),
                    status=status_map.get(status_raw, status_raw or "NEW"),
                    filled_size=filled,
                    remaining_size=qty - filled,
                    client_order_id=str(item.get("cl_ord_id") or item.get("client_id") or "") or None,
                    created_at=self._parse_timestamp_ms(item.get("created_at")),
                    updated_at=self._parse_timestamp_ms(item.get("updated_at")),
                    time_in_force=str(item.get("time_in_force") or "").upper() or None,
                    post_only=str(item.get("time_in_force") or "").lower() == "alo",
                    reduce_only=bool(item.get("reduce_only", False)),
                    raw=item,
                )
            )

        return ApiResponse.ok(orders, raw=raw)

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> ApiResponse:
        payload: Dict[str, Any] = {}
        if order_id and str(order_id).isdigit():
            payload["order_id"] = int(order_id)
        else:
            payload["cl_ord_id"] = str(order_id)
        if symbol:
            payload["symbol"] = self._resolve_symbol(symbol)

        raw = self.make_request("POST", "/api/cancel_order", data=payload, instruction=True)
        error = self._check_raw_error(raw)
        if error:
            return error

        data = raw.get("data") if isinstance(raw, dict) else None
        success = bool(data) if data is not None else True
        cancel_result = CancelResult(
            success=success,
            order_id=str(order_id),
            cancelled_count=1 if success else 0,
            raw=raw,
        )
        return ApiResponse.ok(cancel_result, raw=raw)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        open_orders = self.get_open_orders(symbol)
        if not open_orders.success:
            return open_orders

        orders = open_orders.data or []
        order_ids: List[int] = []
        cl_ord_ids: List[str] = []
        for order in orders:
            oid = getattr(order, "order_id", None)
            cid = getattr(order, "client_order_id", None)
            if oid and str(oid).isdigit():
                order_ids.append(int(oid))
            elif cid:
                cl_ord_ids.append(str(cid))

        if not order_ids and not cl_ord_ids:
            return ApiResponse.ok(CancelResult(success=True, cancelled_count=0))

        payload: Dict[str, Any] = {}
        if order_ids:
            payload["order_id_list"] = order_ids
        if cl_ord_ids:
            payload["cl_ord_id_list"] = cl_ord_ids

        raw = self.make_request("POST", "/api/cancel_orders", data=payload, instruction=True)
        error = self._check_raw_error(raw)
        if error:
            return error

        data = raw.get("data") if isinstance(raw, dict) else None
        cancelled_count = len(order_ids) + len(cl_ord_ids)
        if isinstance(data, dict):
            success_list = data.get("success") or []
            cancelled_count = len(success_list) if isinstance(success_list, list) else cancelled_count

        cancel_result = CancelResult(
            success=True,
            cancelled_count=cancelled_count,
            raw=raw,
        )
        return ApiResponse.ok(cancel_result, raw=raw)

    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        symbol = self._resolve_symbol(order_details.get("symbol"))
        side_raw = str(order_details.get("side") or "").lower()
        side = "buy" if side_raw in ("buy", "bid", "long") else "sell"

        order_type_raw = (
            order_details.get("orderType")
            or order_details.get("order_type")
            or order_details.get("type")
        )
        order_type = str(order_type_raw or "limit").lower()
        if order_type not in {"limit", "market"}:
            order_type = "limit"

        qty = order_details.get("quantity") or order_details.get("qty") or order_details.get("size")
        price = order_details.get("price")
        time_in_force = order_details.get("timeInForce") or order_details.get("time_in_force")
        post_only = bool(order_details.get("postOnly") or order_details.get("post_only"))
        reduce_only = bool(order_details.get("reduceOnly") or order_details.get("reduce_only"))
        client_id = order_details.get("clientId") or order_details.get("client_id")

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "qty": self._format_decimal(qty),
            "reduce_only": bool(reduce_only),
        }

        if order_type == "limit":
            payload["price"] = self._format_decimal(price)

        if not time_in_force:
            time_in_force = "ioc" if order_type == "market" else "gtc"
        payload["time_in_force"] = str(time_in_force).lower()
        if post_only and payload.get("time_in_force") in (None, "", "gtc"):
            payload["time_in_force"] = "alo"
        if reduce_only:
            payload["reduce_only"] = True
        if client_id:
            payload["cl_ord_id"] = str(client_id)

        raw = self.make_request("POST", "/api/new_order", data=payload, instruction=True)
        error = self._check_raw_error(raw)
        if error:
            return error

        data = None
        if isinstance(raw, dict):
            data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        if not isinstance(data, dict):
            return ApiResponse.error("StandX 下單回傳格式異常", raw=raw)

        resolved_client_id = (
            data.get("cl_ord_id")
            or data.get("client_id")
            or data.get("clientId")
            or (str(client_id) if client_id else None)
        )
        order_id = data.get("order_id") or data.get("id") or data.get("orderId") or ""
        if not order_id and resolved_client_id:
            order_id = resolved_client_id

        result = OrderResult(
            success=True,
            order_id=str(order_id),
            client_order_id=str(resolved_client_id) if resolved_client_id else None,
            symbol=data.get("symbol") or symbol,
            side="BUY" if str(data.get("side") or "").lower() == "buy" else "SELL",
            order_type=str(data.get("order_type") or data.get("type") or "").upper(),
            size=self._parse_decimal(data.get("qty")),
            price=self._parse_decimal(data.get("price")),
            filled_size=self._parse_decimal(data.get("filled_qty")),
            status=str(data.get("status") or "").upper() or None,
            created_at=self._parse_timestamp_ms(data.get("created_at")),
            raw=data,
        )
        return ApiResponse.ok(result, raw=raw)

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        params: Dict[str, Any] = {}
        resolved_symbol = self._resolve_symbol(symbol) if symbol else None
        if resolved_symbol:
            params["symbol"] = resolved_symbol

        raw = self.make_request("GET", "/api/query_positions", params=params or None, instruction=True)
        error = self._check_raw_error(raw)
        if error:
            return error

        positions_raw = None
        if isinstance(raw, list):
            positions_raw = raw
        elif isinstance(raw, dict):
            data = raw.get("data")
            if isinstance(data, dict):
                positions_raw = data.get("result")
            if positions_raw is None:
                positions_raw = raw.get("result")

        if not isinstance(positions_raw, list):
            return ApiResponse.error("StandX positions response malformed", raw=raw)

        positions: List[PositionInfo] = []
        for item in positions_raw:
            qty = self._parse_decimal(item.get("qty") or item.get("size") or 0) or Decimal("0")
            side = "FLAT"
            if qty > 0:
                side = "LONG"
            elif qty < 0:
                side = "SHORT"

            entry_price = self._parse_decimal(item.get("entry_price"))
            if entry_price is None:
                entry_value = self._parse_decimal(item.get("entry_value"))
                if entry_value is not None and qty != 0:
                    entry_price = entry_value / abs(qty)

            mark_price = self._parse_decimal(item.get("mark_price"))

            positions.append(
                PositionInfo(
                    symbol=item.get("symbol") or symbol or "",
                    side=side,
                    size=abs(qty),
                    entry_price=entry_price,
                    mark_price=mark_price,
                    liquidation_price=self._parse_decimal(item.get("liquidation_price")),
                    unrealized_pnl=self._parse_decimal(item.get("upnl")),
                    realized_pnl=self._parse_decimal(item.get("realized_pnl")),
                    margin=self._parse_decimal(item.get("holding_margin")),
                    leverage=self._parse_decimal(item.get("leverage")),
                    raw=item,
                )
            )

        return ApiResponse.ok(positions, raw=raw)

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100) -> ApiResponse:
        params: Dict[str, Any] = {"page_size": limit}
        resolved_symbol = self._resolve_symbol(symbol) if symbol else None
        if resolved_symbol:
            params["symbol"] = resolved_symbol

        raw = self.make_request("GET", "/api/query_trades", params=params, instruction=True)
        error = self._check_raw_error(raw)
        if error:
            return error

        trades_raw = None
        if isinstance(raw, list):
            trades_raw = raw
        elif isinstance(raw, dict):
            data = raw.get("data")
            if isinstance(data, dict):
                trades_raw = data.get("result")
            if trades_raw is None:
                trades_raw = raw.get("result")

        if not isinstance(trades_raw, list):
            return ApiResponse.error("StandX trades response malformed", raw=raw)

        trades: List[TradeInfo] = []
        for item in trades_raw:
            side_raw = str(item.get("side") or "").lower()
            side = "BUY" if side_raw in ("buy", "bid", "long") else "SELL"
            trades.append(
                TradeInfo(
                    trade_id=str(item.get("id") or item.get("trade_id") or ""),
                    order_id=str(item.get("order_id") or item.get("cl_ord_id") or ""),
                    symbol=item.get("symbol") or symbol or "",
                    side=side,
                    size=self._parse_decimal(item.get("qty")) or Decimal("0"),
                    price=self._parse_decimal(item.get("price")) or Decimal("0"),
                    fee=self._parse_decimal(item.get("fee_qty")),
                    fee_asset=item.get("fee_asset"),
                    timestamp=self._parse_timestamp_ms(item.get("created_at")),
                    is_maker=None,
                    raw=item,
                )
            )

        return ApiResponse.ok(trades, raw=raw)

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> ApiResponse:
        resolved_symbol = self._resolve_symbol(symbol)
        interval_map = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "1d": "1D",
            "1w": "1W",
            "1M": "1M",
        }
        resolution = interval_map.get(interval, interval)
        seconds_map = {
            "1": 60,
            "3": 180,
            "5": 300,
            "15": 900,
            "30": 1800,
            "60": 3600,
            "120": 7200,
            "240": 14400,
            "1D": 86400,
            "1W": 604800,
        }
        step_seconds = seconds_map.get(resolution, 3600)
        to_ts = int(time.time())
        from_ts = max(0, to_ts - step_seconds * max(limit, 1))

        params = {
            "symbol": resolved_symbol,
            "resolution": resolution,
            "from": from_ts,
            "to": to_ts,
        }
        raw = self.make_request("GET", "/api/kline/history", params=params)
        error = self._check_raw_error(raw)
        if error:
            return error

        data = raw.get("data") if isinstance(raw, dict) else None
        if not isinstance(data, dict):
            return ApiResponse.error("StandX kline response malformed", raw=raw)

        if data.get("s") not in (None, "ok"):
            return ApiResponse.error("StandX kline response error", raw=raw)

        times = data.get("t") or []
        opens = data.get("o") or []
        highs = data.get("h") or []
        lows = data.get("l") or []
        closes = data.get("c") or []
        volumes = data.get("v") or []

        klines: List[KlineInfo] = []
        for idx in range(min(len(times), len(opens), len(highs), len(lows), len(closes), len(volumes))):
            open_time = self._parse_int(times[idx])
            if open_time is None:
                continue
            close_time = open_time + step_seconds
            klines.append(
                KlineInfo(
                    open_time=open_time,
                    close_time=close_time,
                    open_price=self._parse_decimal(opens[idx]) or Decimal("0"),
                    high_price=self._parse_decimal(highs[idx]) or Decimal("0"),
                    low_price=self._parse_decimal(lows[idx]) or Decimal("0"),
                    close_price=self._parse_decimal(closes[idx]) or Decimal("0"),
                    volume=self._parse_decimal(volumes[idx]) or Decimal("0"),
                    raw={
                        "t": times[idx],
                        "o": opens[idx],
                        "h": highs[idx],
                        "l": lows[idx],
                        "c": closes[idx],
                        "v": volumes[idx],
                    },
                )
            )

        return ApiResponse.ok(klines, raw=raw)
