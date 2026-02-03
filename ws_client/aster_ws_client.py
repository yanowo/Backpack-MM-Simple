"""
Aster 交易所 WebSocket 客戶端
"""
from __future__ import annotations

import json
import time
import threading
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import ASTER_API_KEY, ASTER_SECRET_KEY, ASTER_WS_URL
from api.aster_client import AsterClient
from logger import setup_logger

from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSTickerData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSFillData,
)

logger = setup_logger("aster_ws")


class AsterWebSocket(BaseWebSocketClient):
    """Aster WebSocket 客戶端（基於 Binance Futures 風格）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        symbol: str = "BTCUSDT",
        enable_private: bool = False,
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or ASTER_API_KEY
        self.secret_key = secret_key or ASTER_SECRET_KEY
        self._client_cache: Dict[str, AsterClient] = {}
        self.enable_private = enable_private
        self._listen_key: Optional[str] = None
        self._listen_key_lock = threading.Lock()
        self._listen_key_thread: Optional[threading.Thread] = None
        self._listen_key_keepalive_interval = 30 * 60

        config = WSConnectionConfig(
            ws_url=ws_url or ASTER_WS_URL,
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
        return "Aster"

    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        # Aster 私有頻道使用 listenKey
        return None

    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        return {
            "method": "SUBSCRIBE",
            "params": [channel],
            "id": int(time.time() * 1000),
        }

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        return {
            "method": "UNSUBSCRIBE",
            "params": [channel],
            "id": int(time.time() * 1000),
        }

    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict) and payload.get("e") == "ORDER_TRADE_UPDATE":
            return "private.orders", payload

        if isinstance(payload, dict) and "stream" in payload and "data" in payload:
            return payload["stream"], payload["data"]
        return None

    def _get_ticker_channel(self) -> str:
        # bookTicker 提供最佳買賣價
        return f"{self.symbol.lower()}@bookTicker"

    def _get_depth_channel(self) -> str:
        # 深度增量更新（100ms）
        return f"{self.symbol.lower()}@depth@100ms"

    def _get_order_update_channel(self) -> str:
        return "private" if self.enable_private else ""

    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        if not isinstance(data, dict):
            return None

        bid_price = None
        ask_price = None
        last_price = None

        for key in ("b", "bidPrice", "bid"):
            if key in data:
                try:
                    bid_price = Decimal(str(data[key]))
                except Exception:
                    pass
                break

        for key in ("a", "askPrice", "ask"):
            if key in data:
                try:
                    ask_price = Decimal(str(data[key]))
                except Exception:
                    pass
                break

        for key in ("lastPrice", "last", "c"):
            if key in data:
                try:
                    last_price = Decimal(str(data[key]))
                except Exception:
                    pass
                break

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
        if not isinstance(data, dict):
            return None
        order = data.get("o")
        if not isinstance(order, dict):
            return None

        side = "BUY" if str(order.get("S", "")).upper() == "BUY" else "SELL"
        status_raw = str(order.get("X", "")).upper()
        status_map = {
            "NEW": "NEW",
            "PARTIALLY_FILLED": "PARTIALLY_FILLED",
            "FILLED": "FILLED",
            "CANCELED": "CANCELLED",
            "CANCELLED": "CANCELLED",
            "EXPIRED": "CANCELLED",
        }

        return WSOrderUpdateData(
            symbol=order.get("s") or self.symbol,
            order_id=str(order.get("i") or ""),
            side=side,
            order_type=str(order.get("o") or "LIMIT").upper(),
            status=status_map.get(status_raw, status_raw or "NEW"),
            price=self._to_decimal(order.get("p")),
            quantity=self._to_decimal(order.get("q")),
            filled_quantity=self._to_decimal(order.get("z")),
            remaining_quantity=self._to_decimal(order.get("q")) - self._to_decimal(order.get("z")),
            timestamp=self._to_int(order.get("T") or data.get("T")),
            source="ws",
        )

    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        if not isinstance(data, dict):
            return None
        order = data.get("o")
        if not isinstance(order, dict):
            return None
        last_qty = self._to_decimal(order.get("l"))
        if last_qty <= Decimal("0"):
            return None

        side = "BUY" if str(order.get("S", "")).upper() == "BUY" else "SELL"
        return WSFillData(
            symbol=order.get("s") or self.symbol,
            fill_id=str(order.get("t") or ""),
            order_id=str(order.get("i") or ""),
            side=side,
            price=self._to_decimal(order.get("L")),
            quantity=last_qty,
            fee=self._to_decimal(order.get("n")),
            fee_asset=order.get("N"),
            is_maker=bool(order.get("m", True)),
            timestamp=self._to_int(order.get("T") or data.get("T")),
            source="ws",
        )

    def _get_rest_client(self) -> AsterClient:
        cache_key = "aster_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = AsterClient({
                "api_key": self.api_key,
                "secret_key": self.secret_key,
            })
        return self._client_cache[cache_key]

    def connect(self):
        if self.enable_private:
            self._refresh_listen_key()
            if self._listen_key:
                self.config.ws_url = f"wss://fstream.asterdex.com/ws/{self._listen_key}"
        super().connect()

    def subscribe_order_updates(self) -> bool:
        if self.enable_private:
            return True
        return super().subscribe_order_updates()

    def _refresh_listen_key(self) -> None:
        if not self.api_key:
            return
        client = self._get_rest_client()
        raw = client.make_request(
            "POST",
            "/fapi/v1/listenKey",
            api_key=self.api_key,
            instruction=False,
        )
        if isinstance(raw, dict) and raw.get("listenKey"):
            with self._listen_key_lock:
                self._listen_key = raw.get("listenKey")
        if self._listen_key and (self._listen_key_thread is None or not self._listen_key_thread.is_alive()):
            self._listen_key_thread = threading.Thread(target=self._keepalive_listen_key, daemon=True)
            self._listen_key_thread.start()

    def _keepalive_listen_key(self) -> None:
        while self.running and self.enable_private:
            time.sleep(self._listen_key_keepalive_interval)
            with self._listen_key_lock:
                listen_key = self._listen_key
            if not listen_key:
                continue
            client = self._get_rest_client()
            client.make_request(
                "PUT",
                "/fapi/v1/listenKey",
                api_key=self.api_key,
                instruction=False,
                params={"listenKey": listen_key},
            )

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
