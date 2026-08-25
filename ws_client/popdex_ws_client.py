"""POPDEX Futures WebSocket client.

POPDEX exposes market and wallet-scoped topics on one unauthenticated public
WebSocket endpoint.  Wallet topics are scoped by ``walletId``; trading writes
remain locally signed Agent-Key transactions handled by :class:`PopdexClient`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from decimal import Decimal
from typing import Any, Callable, Dict, Optional, Tuple

from api.popdex_client import PopdexClient

from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSFillData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSTickerData,
)


class PopdexWebSocket(BaseWebSocketClient):
    """POPDEX market data and wallet event stream."""

    MAINNET_WS_URL = "wss://ws.popdex.xyz/v1/ws/public"
    TESTNET_WS_URL = "wss://testnet-ws.popdex.xyz/v1/ws/public"
    PRIVATE_TOPICS = ("order", "fill", "position", "account")

    def __init__(
        self,
        account_address: Optional[str] = None,
        symbol: str = "BTCUSDT",
        enable_private: bool = True,
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
        rest_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rest_config = dict(rest_config or {})
        self.account_address = str(
            account_address
            or self.rest_config.get("account_address")
            or self.rest_config.get("wallet_address")
            or ""
        ).strip()
        self.enable_private = bool(enable_private and self.account_address)
        self._rest_client_cache: Optional[PopdexClient] = None
        self._last_application_ping = 0.0
        self._awaiting_application_pong = False
        self.account_update: Optional[Dict[str, Any]] = None
        self.position_updates: Dict[str, Dict[str, Any]] = {}

        normalized_symbol = PopdexClient._symbol_key(symbol)
        # An explicitly selected network must not inherit a stale WS endpoint
        # from the opposite network in the process environment.
        has_explicit_network = bool(self.rest_config.get("network"))
        configured_url = (
            ws_url
            or self.rest_config.get("ws_url")
            or (None if has_explicit_network else os.getenv("POPDEX_WS_URL"))
            or self._default_ws_url()
        )
        config = WSConnectionConfig(
            ws_url=str(configured_url),
            proxy=proxy,
            auto_reconnect=auto_reconnect,
            reconnect_delay=1.0,
            max_reconnect_delay=1800.0,
            max_reconnect_attempts=2,
            heartbeat_interval=30,
            # POPDEX requires application-level text "ping", not a WebSocket
            # control-frame ping.  Disable the base control-frame heartbeat.
            ping_interval=0,
            ping_timeout=None,
        )
        super().__init__(
            config=config,
            symbol=normalized_symbol,
            on_message_callback=on_message_callback,
        )

    def _default_ws_url(self) -> str:
        network = str(self.rest_config.get("network") or "").lower()
        base_url = str(self.rest_config.get("base_url") or "").lower()
        if network in {"test", "testnet"} or "testnet" in base_url:
            return self.TESTNET_WS_URL
        return self.MAINNET_WS_URL

    def get_exchange_name(self) -> str:
        return "POPDEX"

    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        # The official public endpoint serves both market and wallet-scoped
        # topics without an authentication handshake.
        return None

    def _channel_arg(self, channel: str) -> Dict[str, Any]:
        if channel in self.PRIVATE_TOPICS:
            return {"walletId": self.account_address, "topic": channel}

        topic, separator, channel_symbol = channel.partition(".")
        return {
            "category": "Futures",
            "topic": topic,
            "symbol": channel_symbol if separator else self.symbol,
        }

    def _create_subscribe_message(
        self, channel: str, is_private: bool = False
    ) -> Dict[str, Any]:
        return {"op": "subscribe", "args": [self._channel_arg(channel)]}

    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        return {"op": "unsubscribe", "args": [self._channel_arg(channel)]}

    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        if raw_message == "pong":
            self.last_heartbeat = time.time()
            self._awaiting_application_pong = False
            return None

        try:
            payload = json.loads(raw_message)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        # Subscription acknowledgements contain event/arg but no data.
        if "data" not in payload:
            if payload.get("event") in {"error", "fail", "failed"}:
                self._logger.error("POPDEX WebSocket subscription error: %s", payload)
            return None

        arg = payload.get("arg") or {}
        if not isinstance(arg, dict):
            return None
        topic = str(arg.get("topic") or "")
        if not topic:
            return None

        data = payload.get("data")
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("_wsAction", payload.get("action"))
            data.setdefault("_wsTs", payload.get("ts"))
            data.setdefault("_wsBlockNumber", payload.get("blockNumber"))

        if topic in self.PRIVATE_TOPICS:
            stream = topic
        else:
            stream = f"{topic}.{arg.get('symbol') or self.symbol}"

        if stream == "account" and isinstance(data, dict):
            self.account_update = data
        elif stream == "position" and isinstance(data, dict):
            key = "%s:%s" % (
                data.get("symbol") or "default",
                data.get("positionSide") or "net",
            )
            self.position_updates[key] = data
        return stream, data

    def _on_message(self, ws_app, message: str):
        if message == "pong":
            self.last_heartbeat = time.time()
            self._awaiting_application_pong = False
            return

        # POPDEX wallet channels can batch multiple records in data.  The base
        # callback contract handles one record at a time, so fan the batch out
        # without dropping fills or order updates.
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            for item in payload["data"]:
                single = dict(payload)
                single["data"] = item
                super()._on_message(ws_app, json.dumps(single))
            return

        # Let the base class update ticker/book state and invoke the strategy
        # callback, keeping all shared dispatch behavior in one place.
        super()._on_message(ws_app, message)

    def _get_ticker_channel(self) -> str:
        return f"ticker.{self.symbol}"

    def _get_depth_channel(self) -> str:
        # books5 is a complete snapshot on every push, avoiding incremental
        # book sequence recovery after reconnects.
        return f"books5.{self.symbol}"

    def _get_order_update_channel(self) -> str:
        return "order" if self.enable_private else ""

    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        if not isinstance(data, dict):
            return None
        bid = self._to_decimal(data.get("bid1Price"))
        ask = self._to_decimal(data.get("ask1Price"))
        last = self._to_decimal(data.get("lastPrice"))
        if last is None and bid is not None and ask is not None:
            last = (bid + ask) / Decimal("2")
        if bid is None and ask is None and last is None:
            return None
        return WSTickerData(
            symbol=str(data.get("symbol") or self.symbol),
            bid_price=bid,
            ask_price=ask,
            last_price=last,
            timestamp=self._to_int(data.get("updatedTime") or data.get("_wsTs")),
            source="ws",
        )

    def _handle_depth_message(self, data: Any) -> Optional[WSOrderBookData]:
        bids, asks = self._extract_orderbook_levels(
            data, bid_keys=("b", "bids"), ask_keys=("a", "asks")
        )
        if not bids and not asks:
            return None

        # books5 messages are snapshots, not deltas.
        self.orderbook = {"bids": [], "asks": []}
        bids.sort(key=lambda level: level[0], reverse=True)
        asks.sort(key=lambda level: level[0])
        timestamp = self._to_int(data.get("ts") or data.get("_wsTs")) if isinstance(data, dict) else None
        return WSOrderBookData(
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            source="ws",
        )

    def _handle_order_update_message(self, data: Any) -> Optional[WSOrderUpdateData]:
        if not isinstance(data, dict) or not (data.get("orderId") or data.get("clientOid")):
            return None
        side_raw = str(data.get("side") or "").upper()
        status_raw = str(data.get("status") or "NEW").replace("_", "").upper()
        status = {
            "CREATED": "NEW",
            "OPEN": "NEW",
            "PARTIALLYFILLED": "PARTIALLY_FILLED",
            "FILLED": "FILLED",
            "CANCELED": "CANCELLED",
            "CANCELLED": "CANCELLED",
            "REJECTED": "REJECTED",
            "EXPIRED": "CANCELLED",
        }.get(status_raw, status_raw)
        quantity = self._to_decimal(data.get("qty") or data.get("quantity"))
        filled = self._to_decimal(data.get("filledQty"))
        remaining = self._to_decimal(data.get("remainingQty"))
        if remaining is None and quantity is not None and filled is not None:
            remaining = quantity - filled
        return WSOrderUpdateData(
            symbol=str(data.get("symbol") or self.symbol),
            order_id=str(data.get("orderId") or data.get("clientOid") or ""),
            side="BUY" if side_raw in {"BUY", "BID"} else "SELL",
            order_type=str(data.get("type") or data.get("orderType") or "LIMIT").upper(),
            status=status,
            price=self._to_decimal(data.get("price") or data.get("avgDealPrice")),
            quantity=quantity,
            filled_quantity=filled,
            remaining_quantity=remaining,
            timestamp=self._to_int(data.get("updatedAt") or data.get("createdAt") or data.get("_wsTs")),
            source="ws",
        )

    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        if not isinstance(data, dict) or not data.get("execId"):
            return None
        price = self._to_decimal(data.get("execPrice"))
        quantity = self._to_decimal(data.get("execQty"))
        if price is None or quantity is None or quantity <= 0:
            return None

        fee = Decimal("0")
        fee_asset = None
        details = data.get("feeDetail") or []
        if isinstance(details, list):
            for item in details:
                if not isinstance(item, dict):
                    continue
                fee += self._to_decimal(item.get("fee")) or Decimal("0")
                fee_asset = fee_asset or item.get("feeToken")
        if data.get("fee") is not None:
            fee = self._to_decimal(data.get("fee")) or fee
        fee_asset = data.get("feeToken") or data.get("feeAsset") or fee_asset

        scope = str(data.get("tradeScope") or "").upper()
        is_maker = True if scope == "MAKER" else False if scope == "TAKER" else None
        timestamp = self._to_int(
            data.get("updatedAt") or data.get("createdAt") or data.get("_wsTs")
        )
        self._remember_fill(str(data.get("execId")), timestamp or 0)
        side_raw = str(data.get("side") or "").upper()
        return WSFillData(
            symbol=str(data.get("symbol") or self.symbol),
            fill_id=str(data.get("execId")),
            order_id=str(data.get("orderId") or data.get("clientOid") or ""),
            side="BUY" if side_raw in {"BUY", "BID"} else "SELL",
            price=price,
            quantity=quantity,
            fee=fee,
            fee_asset=str(fee_asset) if fee_asset else None,
            is_maker=is_maker,
            timestamp=timestamp,
            source="ws",
        )

    def _get_rest_client(self) -> PopdexClient:
        if self._rest_client_cache is None:
            config = dict(self.rest_config)
            config.setdefault("account_address", self.account_address)
            self._rest_client_cache = PopdexClient(config)
        return self._rest_client_cache

    def _on_open(self, ws_app):
        self._logger.info("POPDEX WebSocket connected")
        self.connected = True
        self.reconnect_attempts = 0
        self.reconnecting = False
        self.last_heartbeat = time.time()
        self._last_application_ping = 0.0
        self._awaiting_application_pong = False
        self._stop_api_fallback()

        # books5 immediately provides a full snapshot, so no blocking REST
        # bootstrap is required in the WebSocket callback thread.
        self.subscribe_ticker()
        self.subscribe_depth()
        if self.enable_private:
            self.subscribe_order_updates()

    def _subscribe(self, channel: str, is_private: bool = False) -> bool:
        # MarketMaker also verifies subscriptions during startup.  Keep that
        # check idempotent so each topic consumes only one POPDEX subscription.
        if channel in self.subscriptions:
            return True
        return super()._subscribe(channel, is_private=is_private)

    def subscribe_order_updates(self) -> bool:
        if not self.enable_private:
            self._logger.info("POPDEX wallet address unavailable; wallet topics disabled")
            return False
        if not self.connected or not self.ws:
            return False

        missing = [topic for topic in self.PRIVATE_TOPICS if topic not in self.subscriptions]
        if not missing:
            return True
        message = {
            "op": "subscribe",
            "args": [self._channel_arg(topic) for topic in missing],
        }
        try:
            self.ws.send(json.dumps(message))
            self.subscriptions.extend(missing)
            self._logger.info("POPDEX wallet topics subscribed: %s", ", ".join(missing))
            return True
        except Exception as exc:
            self._logger.error("POPDEX wallet topic subscription failed: %s", exc)
            return False

    def _heartbeat_check(self):
        """Send POPDEX's required application text heartbeat."""
        while self.running:
            now = time.time()
            if self.connected and self.ws:
                if (
                    self._awaiting_application_pong
                    and now - self._last_application_ping >= self.heartbeat_interval
                ):
                    self._logger.warning("POPDEX application pong timed out; reconnecting")
                    self._awaiting_application_pong = False
                    threading.Thread(target=self._trigger_reconnect, daemon=True).start()
                elif now - self._last_application_ping >= self.heartbeat_interval:
                    try:
                        self.ws.send("ping")
                        self._last_application_ping = now
                        self._awaiting_application_pong = True
                    except Exception as exc:
                        self._logger.warning("POPDEX application ping failed: %s", exc)
                        threading.Thread(target=self._trigger_reconnect, daemon=True).start()
            time.sleep(1)

    def _remember_fill(self, fill_id: str, timestamp: int) -> None:
        if not fill_id or fill_id in self._seen_fill_id_set:
            self._last_fill_timestamp = max(self._last_fill_timestamp, int(timestamp or 0))
            return
        if len(self._seen_fill_ids) == self._seen_fill_ids.maxlen:
            oldest = self._seen_fill_ids[0]
            self._seen_fill_id_set.discard(oldest)
        self._seen_fill_ids.append(fill_id)
        self._seen_fill_id_set.add(fill_id)
        self._last_fill_timestamp = max(self._last_fill_timestamp, int(timestamp or 0))

    def _api_fallback_loop(self):
        """Keep market data and fills alive with REST while WS reconnects."""
        client = self._get_rest_client()
        while self.running and self.api_fallback_active:
            try:
                book_response = client.get_order_book(self.symbol, 50)
                if book_response.success and book_response.data:
                    book = book_response.data
                    self.orderbook = {
                        "bids": [[float(level.price), float(level.quantity)] for level in book.bids],
                        "asks": [[float(level.price), float(level.quantity)] for level in book.asks],
                    }
                    self.last_depth_update = time.time()

                ticker_response = client.get_ticker(self.symbol)
                if ticker_response.success and ticker_response.data:
                    ticker = ticker_response.data
                    self.bid_price = float(ticker.bid_price) if ticker.bid_price is not None else self.bid_price
                    self.ask_price = float(ticker.ask_price) if ticker.ask_price is not None else self.ask_price
                    self.last_price = float(ticker.last_price) if ticker.last_price is not None else self.last_price
                    self.last_ticker_update = time.time()
                    if self.last_price is not None:
                        self.add_price_to_history(self.last_price)

                if self.enable_private:
                    fills_response = client.get_fill_history(self.symbol, limit=100)
                    if fills_response.success:
                        fills = sorted(
                            fills_response.data or [],
                            key=lambda item: int(item.timestamp or 0),
                        )
                        if not self._fallback_bootstrapped and not self._seen_fill_id_set:
                            for fill in fills:
                                self._remember_fill(str(fill.trade_id or ""), int(fill.timestamp or 0))
                        else:
                            for fill in fills:
                                fill_id = str(fill.trade_id or "")
                                if not fill_id or fill_id in self._seen_fill_id_set:
                                    continue
                                if self.on_message_callback and isinstance(fill.raw, dict):
                                    self.on_message_callback("fill", fill.raw)
                                self._remember_fill(fill_id, int(fill.timestamp or 0))
                        self._fallback_bootstrapped = True
            except Exception as exc:
                self._logger.error("POPDEX REST fallback failed: %s", exc)
            time.sleep(self.api_poll_interval)

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        if value is None or value == "":
            return None
        try:
            return Decimal(str(value))
        except Exception:
            return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
