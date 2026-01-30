"""
Aster 交易所 WebSocket 客戶端
"""
from __future__ import annotations

import json
import time
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
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or ASTER_API_KEY
        self.secret_key = secret_key or ASTER_SECRET_KEY
        self._client_cache: Dict[str, AsterClient] = {}

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
        # Aster 公共頻道不需要認證；私有頻道需 listenKey，暫不實作
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
        # 需要 listenKey 才能訂閱，先禁用
        return ""

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
        return None

    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        return None

    def _get_rest_client(self) -> AsterClient:
        cache_key = "aster_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = AsterClient({
                "api_key": self.api_key,
                "secret_key": self.secret_key,
            })
        return self._client_cache[cache_key]
