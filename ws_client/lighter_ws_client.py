"""
Lighter 交易所 WebSocket 客戶端
"""
from __future__ import annotations

import json
import time
from decimal import Decimal
import os
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
        enable_private: bool = False,
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self._client_cache: Dict[str, LighterClient] = {}
        self._market_id: Optional[int] = None
        self.enable_private = enable_private
        self._account_index: Optional[int] = None
        self._private_channels: List[str] = []

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
        # Lighter 私有頻道在 subscribe 時攜帶 auth，不需要單獨 auth 消息
        return None

    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        if is_private:
            account_index = self._resolve_account_index()
            token = self._get_auth_token()
            if account_index is None or not token:
                logger.error("無法取得 account_index 或 auth token，無法訂閱 Lighter 私有頻道")
                return {}
            channel_name = channel
            if "/" not in channel_name:
                if channel_name == "account_orders":
                    market_id = self._resolve_market_id()
                    if market_id is None:
                        logger.error("無法取得 market_id，無法訂閱 Lighter account_orders")
                        return {}
                    channel_name = f"account_orders/{market_id}/{account_index}"
                elif channel_name == "account_all_trades":
                    channel_name = f"account_all_trades/{account_index}"
                elif channel_name == "account_all_orders":
                    channel_name = f"account_all_orders/{account_index}"
                else:
                    channel_name = f"{channel_name}/{account_index}"
            return {
                "type": "subscribe",
                "channel": channel_name,
                "auth": token,
            }

        market_id = self._resolve_market_id()
        if market_id is None:
            logger.error("無法取得 market_id，無法訂閱 Lighter 訂單簿")
            return {}
        channel_name = channel if "/" in channel else f"{channel}/{market_id}"
        return {
            "type": "subscribe",
            "channel": channel_name,
        }

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        market_id = self._resolve_market_id()
        if market_id is None:
            return {}
        channel_name = channel if "/" in channel else f"{channel}/{market_id}"
        return {
            "type": "unsubscribe",
            "channel": channel_name,
        }

    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        if payload.get("type") in {"auth", "subscribed"}:
            return payload.get("type"), payload

        channel = payload.get("channel") or payload.get("type")
        if channel:
            channel_base = str(channel).split(":", 1)[0]
            if channel_base in {"account_orders", "orders", "account_all_orders"}:
                return "private.orders", payload
            if channel_base in {"account_all_trades", "account_trades", "fills", "trades"}:
                return "private.fills", payload
            if channel_base in {"order_book", "book"}:
                return "order_book", payload
            return channel, payload
        return None

    def _get_ticker_channel(self) -> str:
        # Lighter 未穩定提供 ticker channel，僅使用 order_book
        return ""

    def _get_depth_channel(self) -> str:
        return "order_book"

    def _get_order_update_channel(self) -> str:
        return "private" if self.enable_private else ""

    def subscribe_ticker(self) -> bool:
        return False

    def subscribe_order_updates(self) -> bool:
        if not self.enable_private:
            return False
        if not self.connected:
            return False
        self._subscribe_private_channels()
        return True

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
        if not isinstance(data, dict):
            return None

        records = data.get("orders") or data.get("data") or []
        if isinstance(records, dict):
            flattened: List[Dict[str, Any]] = []
            for value in records.values():
                if isinstance(value, list):
                    flattened.extend([v for v in value if isinstance(v, dict)])
                elif isinstance(value, dict):
                    flattened.append(value)
            records = flattened
        if not records:
            return None

        client = self._get_rest_client()
        record = records[0]
        normalized = client._normalize_order_record(record, self.symbol)

        order_id = str(normalized.get("order_id") or normalized.get("id") or "")
        side = normalized.get("side") or normalized.get("direction")
        side = "BUY" if str(side).lower() in {"bid", "buy", "long"} else "SELL"

        status = normalized.get("status") or normalized.get("state") or "NEW"
        status_map = {
            "open": "NEW",
            "new": "NEW",
            "filled": "FILLED",
            "partial": "PARTIALLY_FILLED",
            "partially_filled": "PARTIALLY_FILLED",
            "canceled": "CANCELLED",
            "cancelled": "CANCELLED",
        }
        status = status_map.get(str(status).lower(), str(status).upper())

        price = normalized.get("price")
        qty = normalized.get("quantity") or normalized.get("original_quantity")
        filled = normalized.get("filled_quantity")
        remaining = normalized.get("remaining_quantity")

        return WSOrderUpdateData(
            symbol=self.symbol,
            order_id=order_id,
            side=side,
            order_type=str(normalized.get("order_type") or "LIMIT").upper(),
            status=status,
            price=Decimal(str(price)) if price is not None else None,
            quantity=Decimal(str(qty)) if qty is not None else None,
            filled_quantity=Decimal(str(filled)) if filled is not None else None,
            remaining_quantity=Decimal(str(remaining)) if remaining is not None else None,
            timestamp=self._safe_timestamp(normalized.get("timestamp") or normalized.get("created_at")),
            source="ws",
        )

    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        if not isinstance(data, dict):
            return None

        records = data.get("fills") or data.get("trades") or []
        if isinstance(records, dict):
            flattened: List[Dict[str, Any]] = []
            for value in records.values():
                if isinstance(value, list):
                    flattened.extend([v for v in value if isinstance(v, dict)])
                elif isinstance(value, dict):
                    flattened.append(value)
            records = flattened
        if not records:
            candidate = data.get("data")
            if isinstance(candidate, list) and candidate:
                sample = candidate[0]
                if isinstance(sample, dict):
                    has_trade_fields = any(
                        key in sample for key in ("trade_id", "tradeId", "is_maker_ask", "makerIsAsk")
                    )
                    looks_like_trade = ("price" in sample and "size" in sample and "order_id" not in sample)
                    if has_trade_fields or looks_like_trade:
                        records = candidate
        if not records:
            return None
        if isinstance(records, dict):
            records = [records]
        if not records:
            return None

        client = self._get_rest_client()
        record = records[0]
        normalized = client._normalize_trade_record(record)

        side = normalized.get("side")
        side = "BUY" if str(side).lower() in {"bid", "buy", "long"} else "SELL"

        price = normalized.get("price")
        qty = normalized.get("size") or normalized.get("quantity")
        if price is None or qty is None:
            return None

        return WSFillData(
            symbol=self.symbol,
            fill_id=str(normalized.get("trade_id") or normalized.get("id") or ""),
            order_id=str(normalized.get("order_id") or ""),
            side=side,
            price=Decimal(str(price)),
            quantity=Decimal(str(qty)),
            fee=Decimal(str(normalized.get("fee") or 0)),
            fee_asset=normalized.get("fee_asset"),
            is_maker=bool(normalized.get("is_maker", True)),
            timestamp=self._safe_timestamp(normalized.get("timestamp")),
            source="ws",
        )

    def _get_rest_client(self) -> LighterClient:
        cache_key = "lighter_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = LighterClient({
                "api_private_key": os.getenv("LIGHTER_PRIVATE_KEY"),
                "account_index": os.getenv("LIGHTER_ACCOUNT_INDEX"),
                "api_key_index": os.getenv("LIGHTER_API_KEY_INDEX") or 0,
                "signer_lib_dir": os.getenv("LIGHTER_SIGNER_LIB_DIR"),
                "base_url": os.getenv("LIGHTER_BASE_URL") or "https://mainnet.zklighter.elliot.ai",
            })
        return self._client_cache[cache_key]

    def _on_open(self, ws_app):
        super()._on_open(ws_app)
        if not self.enable_private:
            return
        self._subscribe_private_channels()

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

    def _resolve_account_index(self) -> Optional[int]:
        if self._account_index is not None:
            return self._account_index
        client = self._get_rest_client()
        account_index = client.account_index
        if account_index is None:
            return None
        self._account_index = int(account_index)
        return self._account_index

    def _get_auth_token(self) -> Optional[str]:
        client = self._get_rest_client()
        return client._get_auth_token()

    def _subscribe_private_channels(self) -> None:
        if not self.enable_private or not self.ws:
            return
        if not self._private_channels:
            self._private_channels = ["account_orders", "account_all_trades"]
        for channel in self._private_channels:
            message = self._create_subscribe_message(channel, is_private=True)
            if message:
                self.ws.send(json.dumps(message))

    def _safe_timestamp(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None
