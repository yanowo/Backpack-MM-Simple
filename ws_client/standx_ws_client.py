"""
StandX 交易所 WebSocket 客戶端
"""
from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import STANDX_API_TOKEN, STANDX_WS_URL
from api.standx_client import StandxClient
from logger import setup_logger

from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSTickerData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSFillData,
)

logger = setup_logger("standx_ws")


class StandxWebSocket(BaseWebSocketClient):
    """StandX WebSocket 客戶端（Market Stream + 可選私有頻道）"""

    def __init__(
        self,
        api_token: Optional[str] = None,
        symbol: str = "BTC-PERP",
        enable_private: bool = False,
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self.api_token = api_token or STANDX_API_TOKEN or os.getenv("STANDX_API_TOKEN")
        self.enable_private = enable_private
        self._auth_sent = False
        self._client_cache: Dict[str, StandxClient] = {}

        config = WSConnectionConfig(
            ws_url=ws_url or STANDX_WS_URL,
            api_key=self.api_token,
            secret_key=None,
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
        return "StandX"

    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        if not self.enable_private or not self.api_token:
            return None
        return {
            "auth": {
                "token": self.api_token,
                "streams": [{"channel": "order"}, {"channel": "trade"}],
            }
        }

    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        if is_private:
            return {"subscribe": {"channel": channel}}
        parts = channel.split(".", 1)
        if len(parts) == 2:
            stream, symbol = parts
            return {"subscribe": {"channel": stream, "symbol": symbol}}
        return {"subscribe": {"channel": channel, "symbol": self.symbol}}

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        parts = channel.split(".", 1)
        if len(parts) == 2:
            stream, symbol = parts
            return {"unsubscribe": {"channel": stream, "symbol": symbol}}
        return {"unsubscribe": {"channel": channel}}

    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict) and payload.get("channel") and "data" in payload:
            channel = payload.get("channel")
            symbol = payload.get("symbol")
            if channel == "trade":
                return "order.trade", payload.get("data")
            if symbol:
                return f"{channel}.{symbol}", payload.get("data")
            return channel, payload.get("data")
        return None

    def _get_ticker_channel(self) -> str:
        return f"price.{self.symbol}"

    def _get_depth_channel(self) -> str:
        return f"depth_book.{self.symbol}"

    def _get_order_update_channel(self) -> str:
        return "order" if self.enable_private else ""

    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        if not isinstance(data, dict):
            return None

        bid_price = self._to_decimal(data.get("spread_bid") or data.get("bid"))
        ask_price = self._to_decimal(data.get("spread_ask") or data.get("ask"))
        last_price = self._to_decimal(data.get("price"))
        if last_price is None and bid_price and ask_price:
            last_price = (bid_price + ask_price) / 2

        return WSTickerData(
            symbol=self.symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            last_price=last_price,
            timestamp=self._to_int(data.get("timestamp")),
            source="ws",
        )

    def _handle_depth_message(self, data: Any) -> Optional[WSOrderBookData]:
        bids, asks = self._extract_orderbook_levels(data, bid_keys=("bids", "b"), ask_keys=("asks", "a"))
        if not bids and not asks:
            return None

        timestamp = None
        if isinstance(data, dict):
            timestamp = self._to_int(data.get("timestamp"))

        return WSOrderBookData(
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            source="ws",
        )

    def _handle_order_update_message(self, data: Any) -> Optional[WSOrderUpdateData]:
        if not isinstance(data, dict):
            return None

        side_raw = str(data.get("side") or "").lower()
        side = "BUY" if side_raw in {"buy", "bid", "long"} else "SELL"
        status_raw = str(data.get("status") or "").upper()
        status_map = {
            "NEW": "NEW",
            "PARTIALLY_FILLED": "PARTIALLY_FILLED",
            "FILLED": "FILLED",
            "CANCELED": "CANCELLED",
            "CANCELLED": "CANCELLED",
            "REJECTED": "REJECTED",
            "EXPIRED": "CANCELLED",
        }

        qty = self._to_decimal(data.get("qty"))
        filled = self._to_decimal(data.get("fill_qty") or data.get("filled_qty"))
        if qty is None or filled is None:
            remaining = None
        else:
            remaining = qty - filled

        return WSOrderUpdateData(
            symbol=data.get("symbol") or self.symbol,
            order_id=str(data.get("id") or data.get("order_id") or ""),
            side=side,
            order_type=str(data.get("order_type") or data.get("type") or "LIMIT").upper(),
            status=status_map.get(status_raw, status_raw or "NEW"),
            price=self._to_decimal(data.get("price")),
            quantity=qty,
            filled_quantity=filled,
            remaining_quantity=remaining,
            timestamp=self._to_int(data.get("timestamp")),
            source="ws",
        )

    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        if not isinstance(data, dict):
            return None

        if not (data.get("trade_id") or data.get("id")):
            return None

        side_raw = str(data.get("side") or "").lower()
        side = "BUY" if side_raw in {"buy", "bid", "long"} else "SELL"

        return WSFillData(
            symbol=data.get("symbol") or self.symbol,
            fill_id=str(data.get("trade_id") or data.get("id") or ""),
            order_id=str(data.get("order_id") or data.get("cl_ord_id") or ""),
            side=side,
            price=self._to_decimal(data.get("price")) or Decimal("0"),
            quantity=self._to_decimal(data.get("qty")) or Decimal("0"),
            fee=self._to_decimal(data.get("fee_qty")) or Decimal("0"),
            fee_asset=data.get("fee_asset"),
            is_maker=None,
            timestamp=self._to_int(data.get("timestamp")),
            source="ws",
        )

    def _get_rest_client(self) -> StandxClient:
        cache_key = "standx_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = StandxClient({
                "api_token": self.api_token,
            })
        return self._client_cache[cache_key]

    def _handle_ping(self, message: Any) -> Optional[Dict[str, Any]]:
        if isinstance(message, dict) and "ping" in message:
            return {"pong": message.get("ping")}
        return None

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except Exception:
            return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def subscribe_order_updates(self) -> bool:
        if not self.enable_private:
            return False
        if not self.connected or not self.ws:
            return False
        auth_message = self._create_auth_message()
        if not auth_message:
            return False
        try:
            self.ws.send(json.dumps(auth_message))
            self._auth_sent = True
            if "order" not in self.subscriptions:
                self.subscriptions.append("order")
            if "trade" not in self.subscriptions:
                self.subscriptions.append("trade")
            return True
        except Exception as exc:
            logger.error("StandX 私有頻道訂閱失敗: %s", exc)
            return False
