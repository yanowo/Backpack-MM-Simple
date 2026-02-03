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
        enable_private: bool = False,
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self.account_address = account_address or PARADEX_ACCOUNT_ADDRESS
        self.private_key = private_key or PARADEX_PRIVATE_KEY
        self.enable_private = enable_private
        self._auth_request_id: Optional[int] = None
        self._auth_completed: bool = not enable_private
        self._private_channels: List[str] = []
        self._client_cache: Dict[str, ParadexClient] = {}
        self._price_tick: Optional[str] = None
        self._request_id: int = int(time.time() * 1000)

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
        if not self.enable_private:
            return None
        try:
            client = self._get_rest_client()
            client._ensure_jwt_valid()
            token = getattr(client, "_jwt_token", None)
            if not token:
                logger.error("Paradex JWT token 生成失敗，無法訂閱私有頻道")
                return None
        except Exception as exc:
            logger.error("Paradex 生成 JWT token 失敗: %s", exc)
            return None

        self._auth_request_id = self._next_request_id()
        return {
            "id": self._auth_request_id,
            "jsonrpc": "2.0",
            "method": "auth",
            "params": {"bearer": token},
        }

    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        return {
            "id": self._next_request_id(),
            "jsonrpc": "2.0",
            "method": "subscribe",
            "params": {"channel": channel},
        }

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        return {
            "id": self._next_request_id(),
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
                channel = params.get("channel")
                data = params.get("data")
                if isinstance(channel, str):
                    if channel.startswith("orders."):
                        return "private.orders", data
                    if channel.startswith("fills."):
                        return "private.fills", data
                return channel, data
        return None

    def _get_ticker_channel(self) -> str:
        # Best Bid/Offer
        return f"bbo.{self.symbol}"

    def _get_depth_channel(self) -> str:
        # order_book.{market_symbol}.{price_tick}.{price_tick_num_levels}.{refresh_rate}
        price_tick = self._resolve_price_tick()
        return f"order_book.{self.symbol}.{price_tick}.20.100"

    def _get_order_update_channel(self) -> str:
        return "private"

    def subscribe_order_updates(self) -> bool:
        """Paradex 私有訂單需先完成 auth，再訂閱 orders/fills，避免無效的 'private' 訂閱。"""
        if not self.enable_private:
            return False
        if not self.connected:
            return False
        if self._auth_completed:
            self._subscribe_private_channels()
        return True

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
        if data is None:
            return None
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return None

        order_id = str(data.get("id") or data.get("order_id") or data.get("orderId") or "")
        if not order_id:
            return None

        side_raw = str(data.get("side") or "").upper()
        side = "BUY" if side_raw in {"BUY", "BID"} else "SELL"

        status_raw = str(data.get("status") or data.get("state") or "").upper()
        status_map = {
            "NEW": "NEW",
            "OPEN": "NEW",
            "PARTIAL": "PARTIALLY_FILLED",
            "PARTIALLY_FILLED": "PARTIALLY_FILLED",
            "FILLED": "FILLED",
            "CANCELLED": "CANCELLED",
            "CANCELED": "CANCELLED",
            "REJECTED": "CANCELLED",
        }
        status = status_map.get(status_raw, status_raw or "NEW")

        price = self._to_decimal(data.get("price"))
        quantity = self._to_decimal(data.get("size") or data.get("quantity"))
        filled_quantity = self._to_decimal(data.get("filled_size") or data.get("filled") or data.get("filled_quantity"))
        remaining_quantity = self._to_decimal(data.get("remaining_size") or data.get("remaining") or data.get("remaining_quantity"))

        if remaining_quantity is None and quantity is not None and filled_quantity is not None:
            remaining_quantity = max(quantity - filled_quantity, Decimal("0"))

        return WSOrderUpdateData(
            symbol=data.get("market") or self.symbol,
            order_id=order_id,
            side=side,
            order_type=str(data.get("type") or data.get("orderType") or "LIMIT").upper(),
            status=status,
            price=price,
            quantity=quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            timestamp=self._as_ts(data.get("timestamp") or data.get("created_at") or data.get("updated_at")),
            source="ws",
        )

    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        if data is None:
            return None
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return None

        fill_id = str(data.get("id") or data.get("fill_id") or data.get("trade_id") or "")
        order_id = str(data.get("order_id") or data.get("orderId") or "")
        if not order_id:
            return None

        side_raw = str(data.get("side") or "").upper()
        side = "BUY" if side_raw in {"BUY", "BID"} else "SELL"

        price = self._to_decimal(data.get("price")) or Decimal("0")
        quantity = self._to_decimal(data.get("size") or data.get("quantity")) or Decimal("0")
        fee = self._to_decimal(data.get("fee") or data.get("fee_amount")) or Decimal("0")
        fee_asset = data.get("fee_asset") or data.get("feeAsset")

        liquidity = data.get("liquidity") or data.get("liquidity_type") or data.get("liquidityType")
        maker_flag = self._parse_maker_flag(data.get("is_maker"))
        if maker_flag is None:
            maker_flag = self._parse_maker_flag(data.get("maker"))
        if maker_flag is None:
            maker_flag = self._parse_maker_flag(liquidity)
        is_maker = bool(maker_flag) if maker_flag is not None else False

        return WSFillData(
            symbol=data.get("market") or self.symbol,
            fill_id=fill_id or f"{order_id}-{self._as_ts(data.get('timestamp'))}",
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            fee=fee,
            fee_asset=fee_asset,
            is_maker=is_maker,
            timestamp=self._as_ts(data.get("timestamp") or data.get("created_at")),
            source="ws",
        )

    def _get_rest_client(self) -> ParadexClient:
        cache_key = "paradex_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = ParadexClient({
                "account_address": self.account_address,
                "private_key": self.private_key,
            })
        return self._client_cache[cache_key]

    # ==================== 內部輔助 ====================

    @staticmethod
    def _parse_maker_flag(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        try:
            text = str(value).strip().lower()
        except Exception:
            return None
        if text in {"true", "1", "yes", "maker", "m"}:
            return True
        if text in {"false", "0", "no", "taker", "t"}:
            return False
        return None

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

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

    # ==================== 私有頻道支持 ====================

    def _on_open(self, ws_app):
        super()._on_open(ws_app)
        if not self.enable_private:
            return
        auth_message = self._create_auth_message()
        if auth_message and self.ws:
            self.ws.send(json.dumps(auth_message))
        else:
            logger.error("Paradex 私有頻道認證消息建立失敗")

    def _on_message(self, ws_app, message: str):
        if self.enable_private:
            try:
                payload = json.loads(message)
                if isinstance(payload, dict) and payload.get("id") == self._auth_request_id:
                    if payload.get("error"):
                        self._on_auth_failure(str(payload.get("error")))
                    else:
                        self._auth_completed = True
                        self._on_auth_success()
                        self._subscribe_private_channels()
                    return
            except Exception:
                pass
        super()._on_message(ws_app, message)

    def _subscribe_private_channels(self):
        if not self._auth_completed:
            return
        if not self._private_channels:
            self._private_channels = [
                f"orders.{self.symbol}",
                f"fills.{self.symbol}",
            ]
        for channel in self._private_channels:
            self._subscribe(channel, is_private=True)

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except Exception:
            return None

    @staticmethod
    def _as_ts(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return int(value)
            return int(Decimal(str(value)))
        except Exception:
            return None
