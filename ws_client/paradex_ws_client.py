"""
Paradex 交易所 WebSocket 客戶端
"""
from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import PARADEX_ACCOUNT_ADDRESS, PARADEX_PRIVATE_KEY, PARADEX_WS_URL
from api.paradex_client import ParadexClient
from logger import setup_logger

from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSTickerData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSFillData,
)

logger = setup_logger("paradex_ws")


class ParadexWebSocket(BaseWebSocketClient):
    """Paradex WebSocket 客戶端（JSON-RPC 2.0）"""

    def __init__(
        self,
        account_address: Optional[str] = None,
        private_key: Optional[str] = None,
        symbol: str = "BTC-USD-PERP",
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self.account_address = account_address or PARADEX_ACCOUNT_ADDRESS
        self.private_key = private_key or PARADEX_PRIVATE_KEY
        self._client_cache: Dict[str, ParadexClient] = {}
        self._price_tick: Optional[str] = None

        config = WSConnectionConfig(
            ws_url=ws_url or PARADEX_WS_URL,
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
        return "Paradex"

    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        # 如需私有頻道（下單/成交推播），需 JWT 認證；此處僅實作公共行情
        return None

    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        return {
            "id": int(time.time() * 1000),
            "jsonrpc": "2.0",
            "method": "subscribe",
            "params": {"channel": channel},
        }

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        return {
            "id": int(time.time() * 1000),
            "jsonrpc": "2.0",
            "method": "unsubscribe",
            "params": {"channel": channel},
        }

    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict):
            if payload.get("method") == "subscription":
                params = payload.get("params", {})
                return params.get("channel"), params.get("data")
        return None

    def _get_ticker_channel(self) -> str:
        # Best Bid/Offer
        return f"bbo.{self.symbol}"

    def _get_depth_channel(self) -> str:
        # order_book.{market_symbol}.{price_tick}.{price_tick_num_levels}.{refresh_rate}
        price_tick = self._resolve_price_tick()
        return f"order_book.{self.symbol}.{price_tick}.20.100"

    def _get_order_update_channel(self) -> str:
        # 私有頻道未實作
        return ""

    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        if not isinstance(data, dict):
            return None

        bid_price = None
        ask_price = None
        last_price = None

        if "bid" in data:
            try:
                bid_price = Decimal(str(data["bid"]))
            except Exception:
                pass
        if "ask" in data:
            try:
                ask_price = Decimal(str(data["ask"]))
            except Exception:
                pass

        if "last_traded_price" in data:
            try:
                last_price = Decimal(str(data["last_traded_price"]))
            except Exception:
                pass

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

    def _get_rest_client(self) -> ParadexClient:
        cache_key = "paradex_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = ParadexClient({
                "account_address": self.account_address,
                "private_key": self.private_key,
            })
        return self._client_cache[cache_key]

    # ==================== 內部輔助 ====================

    def _resolve_price_tick(self) -> str:
        if self._price_tick:
            return self._price_tick

        try:
            client = self._get_rest_client()
            markets = client.get_markets()
            if markets.success and markets.data:
                for market in markets.data:
                    if getattr(market, "symbol", None) == self.symbol:
                        tick_size = getattr(market, "tick_size", None)
                        if tick_size:
                            self._price_tick = str(Decimal(str(tick_size)).normalize())
                            return self._price_tick
        except Exception as exc:
            logger.debug(f"獲取 Paradex tick size 失敗: {exc}")

        # fallback
        self._price_tick = "1"
        return self._price_tick
