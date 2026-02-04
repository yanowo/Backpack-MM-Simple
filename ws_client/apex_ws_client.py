"""
Apex 交易所 WebSocket 客戶端
"""
from __future__ import annotations

import json
import time
import os
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import APEX_WS_URL
from api.apex_client import ApexClient
from logger import setup_logger

from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSTickerData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSFillData,
)

logger = setup_logger("apex_ws")


class ApexWebSocket(BaseWebSocketClient):
    """Apex WebSocket 客戶端（Public WS）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        symbol: str = "BTCUSDT",
        enable_private: bool = False,
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("APEX_API_KEY")
        self.secret_key = secret_key or os.getenv("APEX_SECRET_KEY")
        self.passphrase = passphrase or os.getenv("APEX_PASSPHRASE")
        self.enable_private = enable_private
        self._private_channel = "ws_zk_accounts_v3"
        self._private_notify_channel = "ws_notify_v1"
        self._private_ws_base = "wss://quote.omni.apex.exchange/realtime_private"
        self._client_cache: Dict[str, ApexClient] = {}

        config = WSConnectionConfig(
            ws_url=ws_url or APEX_WS_URL,
            api_key=self.api_key,
            secret_key=self.secret_key,
            proxy=proxy,
            auto_reconnect=auto_reconnect,
            reconnect_delay=1.0,
            max_reconnect_delay=1800.0,
            max_reconnect_attempts=2,
            heartbeat_interval=30,
            ping_interval=30,
            ping_timeout=10,
        )

        super().__init__(config=config, symbol=symbol, on_message_callback=on_message_callback)

    # ==================== 抽象方法實現 ====================

    def get_exchange_name(self) -> str:
        return "Apex"

    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        if not self.enable_private or not self.api_key or not self.secret_key:
            return None
        timestamp = str(int(time.time() * 1000))
        client = self._get_rest_client()
        signature = client._sign_request("/ws/accounts", "GET", timestamp, {}, include_data=False)
        login_payload = {
            "type": "login",
            "topics": [self._private_notify_channel, self._private_channel],
            "httpMethod": "GET",
            "requestPath": "/ws/accounts",
            "apiKey": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            "passphrase": self.passphrase or "",
        }
        return {"op": "login", "args": [json.dumps(login_payload)]}

    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        return {"op": "subscribe", "args": [channel]}

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        return {"op": "unsubscribe", "args": [channel]}

    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict) and payload.get("op") in {"ping", "pong", "auth", "login"}:
            return payload.get("op"), payload

        if isinstance(payload, dict) and payload.get("topic") and ("data" in payload or "contents" in payload):
            topic = payload.get("topic")
            if topic in {self._private_channel, self._private_notify_channel}:
                # 保留完整 payload，方便私有資料解析（訂單/成交可能深層嵌套）
                return "private.accounts", payload
            return topic, payload.get("data") if "data" in payload else payload.get("contents")
        if isinstance(payload, dict) and payload.get("channel") and ("data" in payload or "contents" in payload):
            channel = payload.get("channel")
            return channel, payload.get("data") if "data" in payload else payload.get("contents")
        return None

    def _get_ticker_channel(self) -> str:
        # instrumentInfo.{frequency}.{symbol}
        return f"instrumentInfo.H.{self.symbol}"

    def _get_depth_channel(self) -> str:
        # orderBook{limit}.{frequency}.{symbol}
        return f"orderBook25.H.{self.symbol}"

    def _get_order_update_channel(self) -> str:
        return "private" if self.enable_private else ""

    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        if data is None:
            return None

        # data 可能是 dict 或 list
        if isinstance(data, list) and data:
            record = data[0]
        else:
            record = data if isinstance(data, dict) else {}

        bid_price = self._extract_decimal(record, ["bid", "bidPrice", "bp", "bestBidPrice"])
        ask_price = self._extract_decimal(record, ["ask", "askPrice", "ap", "bestAskPrice"])
        last_price = self._extract_decimal(record, ["last", "lastPrice", "lp", "markPrice", "indexPrice"])

        if last_price is None and bid_price and ask_price:
            last_price = (bid_price + ask_price) / 2

        return WSTickerData(
            symbol=self.symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            last_price=last_price,
            source="ws",
        )

    def _handle_depth_message(self, data: Any) -> Optional[WSOrderBookData]:
        bids, asks = self._extract_orderbook_levels(data, bid_keys=("b", "bids"), ask_keys=("a", "asks"))
        if not bids and not asks:
            return None

        return WSOrderBookData(
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            source="ws",
        )

    def _handle_order_update_message(self, data: Any) -> Optional[WSOrderUpdateData]:
        if not isinstance(data, dict):
            return None
        if "data" in data and isinstance(data.get("data"), dict):
            data = data.get("data")  # 公共/私有回傳 data
        if "contents" in data and isinstance(data.get("contents"), dict):
            data = data.get("contents")  # 私有頻道 delta 結構
        orders = (
            data.get("orders")
            or data.get("order")
            or data.get("orderUpdates")
            or data.get("orderList")
            or []
        )
        if isinstance(orders, dict):
            orders = [orders]
        if not orders:
            return None

        record = orders[0]
        order_id = str(record.get("orderId") or record.get("id") or "")
        side = "BUY" if str(record.get("side", "")).lower() in {"buy", "bid"} else "SELL"
        status_raw = str(record.get("status") or "").upper()
        status_map = {
            "NEW": "NEW",
            "PARTIALLY_FILLED": "PARTIALLY_FILLED",
            "FILLED": "FILLED",
            "CANCELED": "CANCELLED",
            "CANCELLED": "CANCELLED",
            "EXPIRED": "CANCELLED",
        }

        return WSOrderUpdateData(
            symbol=record.get("symbol") or self.symbol,
            order_id=order_id,
            side=side,
            order_type=str(record.get("type") or record.get("orderType") or "LIMIT").upper(),
            status=status_map.get(status_raw, status_raw or "NEW"),
            price=self._to_decimal(record.get("price")),
            quantity=self._to_decimal(record.get("size") or record.get("quantity")),
            filled_quantity=self._to_decimal(record.get("filledSize") or record.get("filledQty")),
            remaining_quantity=self._to_decimal(record.get("remainingSize") or record.get("remainingQty")),
            timestamp=self._to_int(record.get("updatedTime") or record.get("timestamp")),
            source="ws",
        )

    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        if not isinstance(data, dict):
            return None
        if "data" in data and isinstance(data.get("data"), dict):
            data = data.get("data")  # 公共/私有回傳 data
        if "contents" in data and isinstance(data.get("contents"), dict):
            data = data.get("contents")  # 私有頻道 delta 結構
        fills = (
            data.get("fills")
            or data.get("trades")
            or data.get("trade")
            or data.get("fill")
            or data.get("tradeList")
            or []
        )
        if isinstance(fills, dict):
            fills = [fills]
        if not fills:
            return None

        record = fills[0]
        side = "BUY" if str(record.get("side", "")).lower() in {"buy", "bid"} else "SELL"
        price = self._to_decimal(record.get("price"))
        qty = self._to_decimal(record.get("size") or record.get("quantity"))
        if price <= 0 or qty <= 0:
            return None

        return WSFillData(
            symbol=record.get("symbol") or self.symbol,
            fill_id=str(record.get("tradeId") or record.get("id") or ""),
            order_id=str(record.get("orderId") or ""),
            side=side,
            price=price,
            quantity=qty,
            fee=self._to_decimal(record.get("fee") or record.get("commission")),
            fee_asset=record.get("feeAsset") or record.get("commissionAsset"),
            is_maker=bool(record.get("maker", True)),
            timestamp=self._to_int(record.get("timestamp") or record.get("time")),
            source="ws",
        )

    def _get_rest_client(self) -> ApexClient:
        cache_key = "apex_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = ApexClient({
                "api_key": self.api_key,
                "secret_key": self.secret_key,
                "passphrase": self.passphrase,
            })
        return self._client_cache[cache_key]

    def _handle_ping(self, message: Any) -> Optional[Dict[str, Any]]:
        if isinstance(message, dict) and message.get("op") == "ping":
            return {"op": "pong", "args": message.get("args", [])}
        return None

    def subscribe_order_updates(self) -> bool:
        if self.enable_private:
            return True
        return super().subscribe_order_updates()

    def connect(self):
        if self.enable_private:
            self.config.ws_url = self._build_private_ws_url()
        super().connect()

    def _on_open(self, ws_app):
        super()._on_open(ws_app)
        if not self.enable_private:
            return
        auth_message = self._create_auth_message()
        if auth_message and self.ws:
            self.ws.send(json.dumps(auth_message))
            self.ws.send(json.dumps({"op": "subscribe", "args": [self._private_channel]}))
            self.ws.send(json.dumps({"op": "subscribe", "args": [self._private_notify_channel]}))

    def _build_private_ws_url(self) -> str:
        timestamp = int(time.time() * 1000)
        return f"{self._private_ws_base}?v=2&timestamp={timestamp}"

    def _to_decimal(self, value: Any) -> Decimal:
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")

    def _to_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    # ==================== 內部工具 ====================

    def _extract_decimal(self, record: Dict[str, Any], keys: List[str]) -> Optional[Decimal]:
        for key in keys:
            if key in record and record[key] is not None:
                try:
                    return Decimal(str(record[key]))
                except Exception:
                    return None
        return None
