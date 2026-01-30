"""
Lighter 交易所 WebSocket 客戶端
"""
from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import LIGHTER_WS_URL
from api.lighter_client import LighterClient
from logger import setup_logger

from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSTickerData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSFillData,
)

logger = setup_logger("lighter_ws")


class LighterWebSocket(BaseWebSocketClient):
    """Lighter WebSocket 客戶端"""

    def __init__(
        self,
        symbol: str = "BTC",
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self._client_cache: Dict[str, LighterClient] = {}
        self._market_id: Optional[int] = None

        config = WSConnectionConfig(
            ws_url=ws_url or LIGHTER_WS_URL,
            api_key=None,
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
        return "Lighter"

    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        return None

    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        # Lighter 以 market_id 訂閱 order_book
        market_id = self._resolve_market_id()
        if market_id is None:
            logger.error("無法取得 market_id，無法訂閱 Lighter 訂單簿")
            return {}
        return {
            "type": "subscribe",
            "channels": [
                {
                    "name": channel,
                    "symbols": [market_id],
                }
            ],
        }

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        market_id = self._resolve_market_id()
        if market_id is None:
            return {}
        return {
            "type": "unsubscribe",
            "channels": [
                {
                    "name": channel,
                    "symbols": [market_id],
                }
            ],
        }

    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict) and payload.get("type"):
            return payload.get("type"), payload
        return None

    def _get_ticker_channel(self) -> str:
        # Lighter 未穩定提供 ticker channel，僅使用 order_book
        return ""

    def _get_depth_channel(self) -> str:
        return "order_book"

    def _get_order_update_channel(self) -> str:
        return ""

    def subscribe_ticker(self) -> bool:
        return False

    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        return None

    def _handle_depth_message(self, data: Any) -> Optional[WSOrderBookData]:
        if not isinstance(data, dict):
            return None

        bids: List[Tuple[Decimal, Decimal]] = []
        asks: List[Tuple[Decimal, Decimal]] = []

        for bid in data.get("bids", []):
            try:
                price = Decimal(str(bid.get("price")))
                qty = Decimal(str(bid.get("size")))
                bids.append((price, qty))
            except Exception:
                continue

        for ask in data.get("asks", []):
            try:
                price = Decimal(str(ask.get("price")))
                qty = Decimal(str(ask.get("size")))
                asks.append((price, qty))
            except Exception:
                continue

        # 更新最佳買賣價，避免缺少 ticker 造成價格不刷新
        if bids:
            self.bid_price = float(bids[0][0])
        if asks:
            self.ask_price = float(asks[0][0])
        if self.bid_price and self.ask_price:
            self.last_price = (self.bid_price + self.ask_price) / 2
            self.add_price_to_history(self.last_price)

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

    def _get_rest_client(self) -> LighterClient:
        cache_key = "lighter_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = LighterClient({
                "api_key": "",
                "secret_key": "",
            })
        return self._client_cache[cache_key]

    # ==================== 內部輔助 ====================

    def _resolve_market_id(self) -> Optional[int]:
        if self._market_id is not None:
            return self._market_id

        try:
            client = self._get_rest_client()
            market = client._lookup_market(self.symbol)
            if market and market.get("market_id") is not None:
                self._market_id = int(market.get("market_id"))
                return self._market_id
        except Exception as exc:
            logger.debug(f"獲取 Lighter market_id 失敗: {exc}")
        return None
