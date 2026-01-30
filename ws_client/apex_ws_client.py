"""
Apex 交易所 WebSocket 客戶端
"""
from __future__ import annotations

import json
import time
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
        symbol: str = "BTCUSDT",
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
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
        # Public WS 不需要認證
        return None

    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        return {"op": "subscribe", "args": [channel]}

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        return {"op": "unsubscribe", "args": [channel]}

    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict) and payload.get("topic") and "data" in payload:
            return payload.get("topic"), payload.get("data")
        return None

    def _get_ticker_channel(self) -> str:
        # instrumentInfo.{frequency}.{symbol}
        return f"instrumentInfo.H.{self.symbol}"

    def _get_depth_channel(self) -> str:
        # orderBook{limit}.{frequency}.{symbol}
        return f"orderBook25.H.{self.symbol}"

    def _get_order_update_channel(self) -> str:
        return ""

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
        if not isinstance(data, dict):
            return None

        bids: List[Tuple[Decimal, Decimal]] = []
        asks: List[Tuple[Decimal, Decimal]] = []

        for bid in data.get("b", []) or data.get("bids", []):
            try:
                price = Decimal(str(bid[0]))
                qty = Decimal(str(bid[1]))
                bids.append((price, qty))
            except Exception:
                continue

        for ask in data.get("a", []) or data.get("asks", []):
            try:
                price = Decimal(str(ask[0]))
                qty = Decimal(str(ask[1]))
                asks.append((price, qty))
            except Exception:
                continue

        return WSOrderBookData(
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            source="ws",
        )

    def _handle_order_update_message(self, data: Any) -> Optional[WSOrderUpdateData]:
        return None

    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        return None

    def _get_rest_client(self) -> ApexClient:
        cache_key = "apex_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = ApexClient({})
        return self._client_cache[cache_key]

    def _handle_ping(self, message: Any) -> Optional[Dict[str, Any]]:
        if isinstance(message, dict) and message.get("op") == "ping":
            return {"op": "pong", "args": message.get("args", [])}
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
