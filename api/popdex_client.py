"""POPDEX perpetual REST and on-chain order client.

Market/account reads use POPDEX REST. Order mutations are ABI-encoded calls to
the fixed POPDEX Order precompile and are signed locally with an authorized
secp256k1 Agent Key.
"""
from __future__ import annotations

import threading
import time
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_account import Account
from eth_utils import is_address, keccak, to_checksum_address

from .base_client import (
    ApiResponse,
    BalanceInfo,
    BaseExchangeClient,
    CancelResult,
    MarketInfo,
    OrderBookInfo,
    OrderBookLevel,
    OrderInfo,
    OrderResult,
    PositionInfo,
    TickerInfo,
    TradeInfo,
)
from .proxy_utils import get_proxy_config
from logger import setup_logger


logger = setup_logger("api.popdex_client")


class PopdexClient(BaseExchangeClient):
    """Client for POPDEX Futures (perpetuals)."""

    ORDER_CONTRACT = "0x0000000000000000000000000000000000001000"
    ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
    CATEGORY_FUTURES = 2
    SCALE = Decimal(10) ** 18

    PLACE_ORDER_SIGNATURE = (
        "placeOrder(address,bytes32,uint16,bytes32,uint256,uint256,uint256,address,uint256)"
    )
    CANCEL_ORDER_SIGNATURE = "cancelOrder(address,uint128,bytes32)"
    CANCEL_ALL_SIGNATURE = (
        "cancelAllOrders(address,uint16,(bool,uint8),(bool,uint8),(bool,bool))"
    )
    ORDER_CREATE_EVENT = keccak(
        text="OrderCreate(address,uint16,uint128,bytes32,int256,int256,int256,uint8,bool,uint32)"
    ).hex()
    ORDER_CANCEL_EVENT = keccak(
        text="OrderCancel(address,uint128,bytes32,bool,uint32)"
    ).hex()
    ORDER_ALL_CANCEL_EVENT = keccak(
        text="OrderAllCancel(address,uint16,(bool,uint8),(bool,uint8),(bool,bool),bool,uint32)"
    ).hex()
    ORDER_ERROR_MESSAGES = {
        15002: "Hedge mode order requires position_side",
        15003: "position_side is not allowed in one-way position mode",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.account_address = str(
            config.get("account_address") or config.get("wallet_address") or ""
        ).strip()
        self.agent_private_key = str(
            config.get("agent_private_key")
            or config.get("private_key")
            or config.get("secret_key")
            or ""
        ).strip()
        self.base_url = str(config.get("base_url") or "https://api.popdex.xyz").rstrip("/")
        self.rpc_url = str(
            config.get("rpc_url") or f"{self.base_url}/api/v1/web3/rpc"
        )
        self.chain_id = int(config.get("chain_id") or self._default_chain_id())
        self.order_contract = to_checksum_address(
            str(config.get("order_contract") or self.ORDER_CONTRACT)
        )
        self.timeout = float(config.get("timeout", 10))
        self.max_retries = int(config.get("max_retries", 3))
        self.receipt_timeout = float(config.get("receipt_timeout", 8))
        self.receipt_poll_interval = float(config.get("receipt_poll_interval", 0.2))
        self.wait_for_receipt = bool(config.get("wait_for_receipt", True))
        configured_position_mode = str(config.get("position_mode") or "").strip()
        self._position_mode = self._normalize_position_mode(configured_position_mode)
        self._position_mode_checked = bool(self._position_mode)
        configured_slippage = self._decimal(config.get("market_slippage"))
        self.market_slippage = (
            configured_slippage if configured_slippage is not None else Decimal("0.01")
        )
        self.session = requests.Session()
        proxies = get_proxy_config()
        if proxies:
            self.session.proxies.update(proxies)

        self._market_cache: Dict[str, Dict[str, Any]] = {}
        self._last_nonce = 0
        self._nonce_lock = threading.Lock()
        self._rpc_id = 0
        self._rpc_lock = threading.Lock()

    def _default_chain_id(self) -> int:
        return 34952 if "testnet" in self.base_url.lower() else 2184

    def get_exchange_name(self) -> str:
        return "POPDEX"

    async def connect(self) -> None:
        logger.info("POPDEX client connected")

    async def disconnect(self) -> None:
        self.session.close()
        logger.info("POPDEX client disconnected")

    # ------------------------- transport/helpers -------------------------
    def make_request(
        self,
        method: str,
        endpoint: str,
        api_key=None,
        secret_key=None,
        instruction=None,
        params=None,
        data=None,
        retry_count: int = 3,
    ) -> Dict[str, Any]:
        url = endpoint if str(endpoint).startswith("http") else f"{self.base_url}{endpoint}"
        retry_total = max(1, int(retry_count or self.max_retries))
        method_upper = method.upper()

        for attempt in range(retry_total):
            try:
                response = self.session.request(
                    method_upper,
                    url,
                    params=params,
                    json=data if method_upper not in {"GET", "DELETE"} else None,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    payload = response.json() if response.content else {}
                except ValueError:
                    payload = {"message": response.text or f"HTTP {response.status_code}"}

                if 200 <= response.status_code < 300:
                    return payload
                if (response.status_code == 429 or response.status_code >= 500) and attempt + 1 < retry_total:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                message = (
                    payload.get("msg")
                    or payload.get("message")
                    or f"HTTP {response.status_code}"
                ) if isinstance(payload, dict) else f"HTTP {response.status_code}"
                return {
                    "error": str(message),
                    "status_code": response.status_code,
                    "details": payload,
                }
            except requests.RequestException as exc:
                if attempt + 1 < retry_total:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                return {"error": f"POPDEX request failed: {exc}"}
        return {"error": "POPDEX maximum retry count reached"}

    def _api_error(self, raw: Any) -> Optional[ApiResponse]:
        if not isinstance(raw, dict):
            return ApiResponse.error("POPDEX response malformed", raw=raw)
        if raw.get("error") is not None:
            return ApiResponse.error(
                str(raw.get("error")),
                code=str(raw.get("status_code") or "") or None,
                raw=raw,
            )
        code = raw.get("code")
        if code is not None and str(code) != "200":
            return ApiResponse.error(
                str(raw.get("msg") or "POPDEX API error"),
                code=str(code),
                raw=raw,
            )
        return None

    @staticmethod
    def _decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
        if value in (None, ""):
            return default
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return default

    @staticmethod
    def _int(value: Any, default: Optional[int] = None) -> Optional[int]:
        if value in (None, ""):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _timestamp_ms(value: Any) -> Optional[int]:
        parsed = PopdexClient._int(value)
        if parsed is None:
            return None
        return parsed * 1000 if parsed < 10_000_000_000 else parsed

    @staticmethod
    def _symbol_key(symbol: str) -> str:
        key = str(symbol or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")
        for suffix in ("PERPETUAL", "PERP"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key

    def _ensure_markets(self) -> None:
        if self._market_cache:
            return
        response = self.get_markets()
        if not response.success:
            logger.warning("Unable to load POPDEX markets: %s", response.error_message)

    def _resolve_symbol(self, symbol: Optional[str]) -> Optional[str]:
        if not symbol:
            return None
        self._ensure_markets()
        key = self._symbol_key(symbol)
        item = self._market_cache.get(key)
        if item:
            return str(item.get("symbol") or key)
        return key

    def _private_path(self, suffix: str) -> Optional[str]:
        if not self.account_address:
            return None
        return f"/api/v1/account/{self.account_address}{suffix}"

    @staticmethod
    def _normalize_position_mode(value: Any) -> Optional[str]:
        normalized = str(value or "").strip().replace("_", "").replace("-", "").upper()
        if normalized in {"ONEWAY", "HEDGE"}:
            return normalized
        return None

    def _get_position_mode(self) -> Optional[str]:
        """Load the account mode once so orderParams[7] matches POPDEX settings."""
        if self._position_mode_checked:
            return self._position_mode
        self._position_mode_checked = True
        path = self._private_path("/settings")
        if not path:
            return None
        raw = self.make_request("GET", path)
        error = self._api_error(raw)
        data = raw.get("data") if isinstance(raw, dict) else None
        if error or not isinstance(data, dict):
            logger.warning(
                "Unable to load POPDEX position mode; defaulting order position_side to None: %s",
                error.error_message if error else "malformed account settings",
            )
            return None
        self._position_mode = self._normalize_position_mode(data.get("positionMode"))
        if not self._position_mode:
            logger.warning(
                "Unknown POPDEX position mode %r; defaulting order position_side to None",
                data.get("positionMode"),
            )
        else:
            logger.info("POPDEX account position mode: %s", self._position_mode)
        return self._position_mode

    def _apply_account_position_mode(self, order_details: Dict[str, Any]) -> Dict[str, Any]:
        """Return an order copy with a position side valid for One-way/Hedge mode."""
        prepared = dict(order_details)
        mode = self._get_position_mode()
        if mode == "ONEWAY" or mode is None:
            # POPDEX error 15003: One-way accounts must use PositionSide.None.
            prepared["positionSide"] = "None"
            return prepared

        explicit = str(
            prepared.get("positionSide") or prepared.get("position_side") or ""
        ).upper()
        if explicit in {"LONG", "SHORT", "1", "2"}:
            return prepared

        side = str(prepared.get("side") or "").upper()
        reduce_only = bool(prepared.get("reduceOnly") or prepared.get("reduce_only"))
        if reduce_only:
            prepared["positionSide"] = "Short" if side in {"BUY", "BID", "LONG"} else "Long"
        else:
            prepared["positionSide"] = "Long" if side in {"BUY", "BID", "LONG"} else "Short"
        return prepared

    def _require_account(self) -> Optional[ApiResponse]:
        if not self.account_address or not is_address(self.account_address):
            return ApiResponse.error("POPDEX_ACCOUNT_ADDRESS is missing or invalid")
        return None

    def _require_signer(self) -> Optional[ApiResponse]:
        account_error = self._require_account()
        if account_error:
            return account_error
        if not self.agent_private_key:
            return ApiResponse.error("POPDEX_AGENT_PRIVATE_KEY is missing")
        try:
            Account.from_key(self.agent_private_key)
        except Exception as exc:
            return ApiResponse.error(f"POPDEX Agent Key is invalid: {exc}")
        return None

    def _next_nonce(self) -> int:
        # POPDEX Agent transactions use a millisecond timestamp nonce. Keep it
        # monotonic so concurrent strategy workers never reuse a nonce.
        with self._nonce_lock:
            nonce = max(int(time.time() * 1000), self._last_nonce + 1)
            self._last_nonce = nonce
            return nonce

    def _rpc(self, method: str, params: Sequence[Any]) -> Dict[str, Any]:
        with self._rpc_lock:
            self._rpc_id += 1
            rpc_id = self._rpc_id
        try:
            response = self.session.post(
                self.rpc_url,
                json={"jsonrpc": "2.0", "method": method, "params": list(params), "id": rpc_id},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"error": f"POPDEX RPC request failed: {exc}"}
        if not 200 <= response.status_code < 300:
            return {"error": payload, "status_code": response.status_code}
        if payload.get("error") is not None:
            error = payload["error"]
            return {
                "error": error.get("message") if isinstance(error, dict) else str(error),
                "details": error,
            }
        return payload

    @staticmethod
    def _encode_call(signature: str, types: Sequence[str], values: Sequence[Any]) -> bytes:
        return keccak(text=signature)[:4] + abi_encode(list(types), list(values))

    def _send_contract_call(self, call_data: bytes) -> Tuple[Optional[str], Optional[str], Any]:
        signer_error = self._require_signer()
        if signer_error:
            return None, signer_error.error_message, signer_error.raw
        transaction = {
            "chainId": self.chain_id,
            "nonce": self._next_nonce(),
            "gasPrice": 0,
            "gas": 1_000_000,
            "to": self.order_contract,
            "value": 0,
            "data": call_data,
        }
        try:
            signed = Account.sign_transaction(transaction, self.agent_private_key)
            raw_tx = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
            raw_hex = "0x" + bytes(raw_tx).hex()
        except Exception as exc:
            return None, f"Unable to sign POPDEX transaction: {exc}", transaction

        raw = self._rpc("eth_sendRawTransaction", [raw_hex])
        if raw.get("error") is not None:
            return None, str(raw.get("error")), raw
        tx_hash = raw.get("result")
        if not tx_hash:
            return None, "POPDEX RPC did not return a transaction hash", raw
        return str(tx_hash), None, raw

    def _wait_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        if not self.wait_for_receipt or self.receipt_timeout <= 0:
            return None
        deadline = time.monotonic() + self.receipt_timeout
        while time.monotonic() < deadline:
            raw = self._rpc("eth_getTransactionReceipt", [tx_hash])
            receipt = raw.get("result") if isinstance(raw, dict) else None
            if isinstance(receipt, dict):
                return receipt
            time.sleep(self.receipt_poll_interval)
        return None

    @staticmethod
    def _receipt_succeeded(receipt: Optional[Dict[str, Any]]) -> bool:
        if not receipt:
            return True
        status = receipt.get("status")
        return status in (None, "0x1", "1", 1, True)

    def _order_from_receipt(self, receipt: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        if not receipt:
            return None, None
        expected_topic = "0x" + self.ORDER_CREATE_EVENT.removeprefix("0x")
        for log in receipt.get("logs") or []:
            topics = log.get("topics") or []
            if not topics or str(topics[0]).lower() != expected_topic.lower():
                continue
            try:
                decoded = abi_decode(
                    ["uint128", "bytes32", "int256", "int256", "int256", "uint8", "bool", "uint32"],
                    bytes.fromhex(str(log.get("data") or "0x")[2:]),
                )
                order_id = str(decoded[0])
                client_id = bytes(decoded[1]).rstrip(b"\x00").decode("utf-8", errors="ignore") or None
                if not bool(decoded[6]) or int(decoded[7]) != 0:
                    code = int(decoded[7])
                    description = self.ORDER_ERROR_MESSAGES.get(code)
                    suffix = f": {description}" if description else ""
                    return None, f"POPDEX OrderCreate rejected with code {code}{suffix}"
                return order_id, client_id
            except Exception:
                continue
        return None, None

    @staticmethod
    def _event_business_error(
        receipt: Optional[Dict[str, Any]],
        event_topic: str,
        types: Sequence[str],
        succeeded_index: int,
        code_index: int,
        event_name: str,
    ) -> Optional[str]:
        if not receipt:
            return None
        expected_topic = "0x" + event_topic.removeprefix("0x")
        for log in receipt.get("logs") or []:
            topics = log.get("topics") or []
            if not topics or str(topics[0]).lower() != expected_topic.lower():
                continue
            try:
                values = abi_decode(
                    list(types), bytes.fromhex(str(log.get("data") or "0x")[2:])
                )
                succeeded = bool(values[succeeded_index])
                code = int(values[code_index])
                if not succeeded or code != 0:
                    return f"POPDEX {event_name} rejected with code {code}"
            except Exception as exc:
                logger.warning("Unable to decode POPDEX %s event: %s", event_name, exc)
            return None
        return None

    @classmethod
    def _to_fixed_18(cls, value: Any) -> int:
        parsed = cls._decimal(value)
        if parsed is None or parsed < 0:
            raise ValueError(f"Invalid non-negative decimal value: {value}")
        return int((parsed * cls.SCALE).to_integral_value(rounding=ROUND_DOWN))

    @staticmethod
    def _client_id_bytes(value: Any) -> Tuple[str, bytes]:
        client_id = str(value or f"pdx{int(time.time() * 1_000_000)}")
        encoded = client_id.encode("utf-8")
        if len(encoded) > 32:
            client_id = "pdx" + keccak(encoded)[:14].hex()
            encoded = client_id.encode("ascii")
        return client_id, encoded.ljust(32, b"\x00")

    @staticmethod
    def _position_side(order_details: Dict[str, Any], side: int, reduce_only: bool) -> int:
        explicit = str(order_details.get("positionSide") or order_details.get("position_side") or "").upper()
        if explicit in {"LONG", "1"}:
            return 1
        if explicit in {"SHORT", "2"}:
            return 2
        if explicit in {"NONE", "NET", "0"}:
            return 0
        # PositionSide.None is required for One-way accounts. Hedge-mode
        # inference is applied by _apply_account_position_mode before packing.
        return 0

    @classmethod
    def _pack_order_params(cls, order_details: Dict[str, Any]) -> bytes:
        order_type_raw = str(
            order_details.get("orderType") or order_details.get("order_type") or "Limit"
        ).upper()
        order_type = {"LIMIT": 0, "MARKET": 1, "PLAN": 2, "TPSL": 3}.get(order_type_raw)
        if order_type is None:
            raise ValueError(f"Unsupported POPDEX order type: {order_type_raw}")

        side_raw = str(order_details.get("side") or "").upper()
        if side_raw in {"BUY", "BID", "LONG"}:
            side = 0
        elif side_raw in {"SELL", "ASK", "SHORT"}:
            side = 1
        else:
            raise ValueError(f"Unsupported POPDEX order side: {side_raw}")

        default_tif = "IOC" if order_type == 1 else "GTC"
        tif_raw = str(order_details.get("timeInForce") or order_details.get("time_in_force") or default_tif).upper()
        post_only = bool(order_details.get("postOnly") or order_details.get("post_only"))
        tif = 4 if post_only else {"DEFAULT": 0, "GTC": 1, "IOC": 2, "FOK": 3, "POSTONLY": 4, "POST_ONLY": 4}.get(tif_raw)
        if tif is None:
            raise ValueError(f"Unsupported POPDEX time in force: {tif_raw}")

        reduce_only = bool(order_details.get("reduceOnly") or order_details.get("reduce_only"))
        packed = bytearray(32)
        packed[0] = cls.CATEGORY_FUTURES
        packed[1] = order_type
        packed[2] = side
        packed[3] = tif
        packed[4] = int(order_details.get("marketUnit") or order_details.get("market_unit") or 0)
        packed[5] = int(order_details.get("bbo") or 0)
        packed[6] = 1 if reduce_only else 0
        packed[7] = cls._position_side(order_details, side, reduce_only)
        packed[8] = int(order_details.get("marginMode") or order_details.get("margin_mode") or 0)
        packed[9] = int(order_details.get("stpMode") or order_details.get("stp_mode") or 0)
        return bytes(packed)

    # ----------------------------- market reads -----------------------------
    def get_markets(self) -> ApiResponse:
        raw = self.make_request(
            "GET", "/api/v1/config/symbols", params={"category": "Futures"}
        )
        error = self._api_error(raw)
        if error:
            return error
        items = raw.get("data") or []
        if not isinstance(items, list):
            return ApiResponse.error("POPDEX symbols response malformed", raw=raw)

        markets: List[MarketInfo] = []
        for item in items:
            symbol = str(item.get("symbol") or "")
            if not symbol:
                continue
            market = MarketInfo(
                symbol=symbol,
                base_asset=str(item.get("baseToken") or ""),
                quote_asset=str(item.get("quoteToken") or item.get("settlementToken") or ""),
                market_type="PERP",
                status=str(item.get("status") or "").upper(),
                min_order_size=self._decimal(item.get("minQty"), Decimal("0")) or Decimal("0"),
                max_order_size=self._decimal(item.get("maxQty")),
                tick_size=self._decimal(item.get("tickSize"), Decimal("0.00000001")) or Decimal("0.00000001"),
                step_size=self._decimal(item.get("lotSize")),
                base_precision=self._int(item.get("baseTokenPrecision"), 8) or 0,
                quote_precision=self._int(item.get("pricePrecision"), 8) or 0,
                min_notional=self._decimal(item.get("minNotional")),
                raw=item,
            )
            markets.append(market)
            self._market_cache[self._symbol_key(symbol)] = item
        return ApiResponse.ok(markets, raw=raw)

    def get_market_limits(self, symbol: str) -> ApiResponse:
        self._ensure_markets()
        item = self._market_cache.get(self._symbol_key(symbol))
        if not item:
            return ApiResponse.error(f"POPDEX market not found: {symbol}")
        market = MarketInfo(
            symbol=str(item.get("symbol") or symbol),
            base_asset=str(item.get("baseToken") or ""),
            quote_asset=str(item.get("quoteToken") or item.get("settlementToken") or ""),
            market_type="PERP",
            status=str(item.get("status") or "").upper(),
            min_order_size=self._decimal(item.get("minQty"), Decimal("0")) or Decimal("0"),
            max_order_size=self._decimal(item.get("maxQty")),
            tick_size=self._decimal(item.get("tickSize"), Decimal("0.00000001")) or Decimal("0.00000001"),
            step_size=self._decimal(item.get("lotSize")),
            base_precision=self._int(item.get("baseTokenPrecision"), 8) or 0,
            quote_precision=self._int(item.get("pricePrecision"), 8) or 0,
            min_notional=self._decimal(item.get("minNotional")),
            raw=item,
        )
        return ApiResponse.ok(market, raw=item)

    def get_ticker(self, symbol: str) -> ApiResponse:
        resolved = self._resolve_symbol(symbol)
        raw = self.make_request(
            "GET",
            "/api/v1/public/market/tickers",
            params={"category": "Futures", "symbol": resolved},
        )
        error = self._api_error(raw)
        if error:
            return error
        items = raw.get("data") or []
        if not isinstance(items, list) or not items:
            return ApiResponse.error(f"POPDEX ticker not found: {symbol}", raw=raw)
        item = items[0]
        ticker = TickerInfo(
            symbol=str(item.get("symbol") or resolved),
            last_price=self._decimal(item.get("lastPrice")),
            bid_price=self._decimal(item.get("bid1Price")),
            ask_price=self._decimal(item.get("ask1Price")),
            bid_size=self._decimal(item.get("bid1Size")),
            ask_size=self._decimal(item.get("ask1Size")),
            mark_price=self._decimal(item.get("markPrice")),
            index_price=self._decimal(item.get("indexPrice")),
            volume_24h=self._decimal(item.get("volume24h")),
            turnover_24h=self._decimal(item.get("turnover24h")),
            high_24h=self._decimal(item.get("highPrice24h")),
            low_24h=self._decimal(item.get("lowPrice24h")),
            change_percent_24h=self._decimal(item.get("price24hPcnt")),
            open_interest=self._decimal(item.get("openInterest")),
            funding_rate=self._decimal(item.get("fundingRate")),
            timestamp=self._timestamp_ms(item.get("updatedTime")),
            raw=item,
        )
        return ApiResponse.ok(ticker, raw=raw)

    def get_order_book(self, symbol: str, limit: int = 20) -> ApiResponse:
        resolved = self._resolve_symbol(symbol)
        levels = max(1, min(int(limit), 500))
        raw = self.make_request(
            "GET",
            "/api/v1/public/market/orderbook",
            params={"category": "Futures", "symbol": resolved, "levels": levels},
        )
        error = self._api_error(raw)
        if error:
            return error
        data = raw.get("data") or {}

        def parse_levels(values: Any) -> List[OrderBookLevel]:
            parsed: List[OrderBookLevel] = []
            for value in values or []:
                try:
                    parsed.append(
                        OrderBookLevel(price=Decimal(str(value[0])), quantity=Decimal(str(value[1])))
                    )
                except (IndexError, InvalidOperation, TypeError, ValueError):
                    continue
            return parsed

        bids = parse_levels(data.get("bids"))
        asks = parse_levels(data.get("asks"))
        bids.sort(key=lambda level: level.price, reverse=True)
        asks.sort(key=lambda level: level.price)
        book = OrderBookInfo(
            symbol=str(resolved),
            bids=bids,
            asks=asks,
            timestamp=self._timestamp_ms(raw.get("updatedTime")),
            sequence=self._int(raw.get("updatedBlock")),
            raw=raw,
        )
        return ApiResponse.ok(book, raw=raw)

    # ----------------------------- account reads -----------------------------
    def _get_overview(self) -> Tuple[Optional[Dict[str, Any]], Optional[ApiResponse], Any]:
        account_error = self._require_account()
        if account_error:
            return None, account_error, None
        raw = self.make_request("GET", self._private_path("/overview"))
        error = self._api_error(raw)
        if error:
            return None, error, raw
        data = raw.get("data")
        if not isinstance(data, dict):
            return None, ApiResponse.error("POPDEX account overview malformed", raw=raw), raw
        return data, None, raw

    def get_balance(self) -> ApiResponse:
        data, error, raw = self._get_overview()
        if error:
            return error
        balances: List[BalanceInfo] = []
        for item in data.get("balances") or []:
            asset = str(item.get("token") or item.get("asset") or item.get("symbol") or "")
            if not asset:
                continue
            total = self._decimal(item.get("balance") or item.get("equity"), Decimal("0")) or Decimal("0")
            available = self._decimal(item.get("available"), total) or Decimal("0")
            frozen = self._decimal(item.get("frozen"), total - available) or Decimal("0")
            balances.append(
                BalanceInfo(asset=asset, available=available, locked=frozen, total=total, raw=item)
            )
        return ApiResponse.ok(balances, raw=raw)

    def get_collateral(self, subaccount_id: Optional[str] = None) -> ApiResponse:
        # POPDEX overview balances already include collateral/equity. Returning
        # them a second time would make MarketMaker.get_total_balance double count.
        account_error = self._require_account()
        if account_error:
            return account_error
        return ApiResponse.ok([])

    def get_open_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        account_error = self._require_account()
        if account_error:
            return account_error
        params: Dict[str, Any] = {"category": "Futures"}
        if symbol:
            params["symbol"] = self._resolve_symbol(symbol)
        raw = self.make_request("GET", self._private_path("/orders"), params=params)
        error = self._api_error(raw)
        if error:
            return error
        items = raw.get("data") or []
        if isinstance(items, dict):
            items = items.get("list") or items.get("rows") or []
        if not isinstance(items, list):
            return ApiResponse.error("POPDEX open orders response malformed", raw=raw)

        orders: List[OrderInfo] = []
        for item in items:
            qty = self._decimal(item.get("qty"), Decimal("0")) or Decimal("0")
            filled = self._decimal(item.get("filledQty"), Decimal("0")) or Decimal("0")
            remaining = self._decimal(item.get("remainingQty"), qty - filled) or Decimal("0")
            side_raw = str(item.get("side") or "").upper()
            status_raw = str(item.get("status") or "NEW").replace("_", "").upper()
            status = {
                "PARTIALLYFILLED": "PARTIALLY_FILLED",
                "CANCELED": "CANCELLED",
                "CANCELLED": "CANCELLED",
            }.get(status_raw, status_raw)
            tif = str(item.get("timeInForce") or "").upper() or None
            orders.append(
                OrderInfo(
                    order_id=str(item.get("orderId") or item.get("id") or ""),
                    client_order_id=str(item.get("clientOid") or item.get("clientOrderId") or "") or None,
                    symbol=str(item.get("symbol") or symbol or ""),
                    side="BUY" if side_raw in {"BUY", "BID"} else "SELL",
                    order_type=str(item.get("type") or item.get("orderType") or "LIMIT").upper(),
                    size=qty,
                    price=self._decimal(item.get("price")),
                    status=status,
                    filled_size=filled,
                    remaining_size=remaining,
                    created_at=self._timestamp_ms(item.get("createdAt")),
                    updated_at=self._timestamp_ms(item.get("updatedAt")),
                    time_in_force=tif,
                    post_only=tif in {"POSTONLY", "POST_ONLY"},
                    reduce_only=bool(item.get("reduceOnly", False)),
                    raw=item,
                )
            )
        return ApiResponse.ok(orders, raw=raw)

    def get_order_by_client_id(self, client_order_id: str) -> ApiResponse:
        """Resolve the on-chain order ID when only a client OID is available."""
        account_error = self._require_account()
        if account_error:
            return account_error
        encoded_id = quote(str(client_order_id), safe="")
        raw = self.make_request(
            "GET", self._private_path(f"/orders/client-oid/{encoded_id}")
        )
        error = self._api_error(raw)
        if error:
            return error
        data = raw.get("data")
        if not isinstance(data, dict):
            return ApiResponse.error("POPDEX client-order lookup response malformed", raw=raw)
        return ApiResponse.ok(data, raw=raw)

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        account_error = self._require_account()
        if account_error:
            return account_error
        params = {"symbol": self._resolve_symbol(symbol)} if symbol else None
        raw = self.make_request("GET", self._private_path("/positions"), params=params)
        error = self._api_error(raw)
        if error:
            return error
        items = raw.get("data") or []
        if not isinstance(items, list):
            return ApiResponse.error("POPDEX positions response malformed", raw=raw)
        positions: List[PositionInfo] = []
        for item in items:
            qty = self._decimal(item.get("holdQty"), Decimal("0")) or Decimal("0")
            side_raw = str(item.get("positionSide") or "").upper()
            if side_raw not in {"LONG", "SHORT"}:
                side_raw = "LONG" if qty > 0 else "SHORT" if qty < 0 else "FLAT"
            positions.append(
                PositionInfo(
                    symbol=str(item.get("symbol") or symbol or ""),
                    side=side_raw,
                    size=abs(qty),
                    entry_price=self._decimal(item.get("avgOpenPrice")),
                    mark_price=self._decimal(item.get("markPrice")),
                    liquidation_price=self._decimal(item.get("liqPrice")),
                    unrealized_pnl=self._decimal(item.get("unPnl")),
                    realized_pnl=self._decimal(item.get("realizedPnl") or item.get("netProfit")),
                    margin=self._decimal(item.get("initialMargin")),
                    leverage=self._decimal(item.get("symbolLeverage")),
                    margin_mode=str(item.get("marginMode") or "").upper() or None,
                    raw=item,
                )
            )
        return ApiResponse.ok(positions, raw=raw)

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100) -> ApiResponse:
        account_error = self._require_account()
        if account_error:
            return account_error
        params: Dict[str, Any] = {"category": "Futures", "limit": max(1, min(int(limit), 100))}
        if symbol:
            params["symbol"] = self._resolve_symbol(symbol)
        raw = self.make_request("GET", self._private_path("/trade/fills"), params=params)
        error = self._api_error(raw)
        if error:
            return error
        items = raw.get("data") or []
        if isinstance(items, dict):
            items = items.get("list") or items.get("rows") or []
        if not isinstance(items, list):
            return ApiResponse.error("POPDEX fills response malformed", raw=raw)
        fills: List[TradeInfo] = []
        for item in items:
            side_raw = str(item.get("side") or "").upper()
            role = str(
                item.get("tradeScope") or item.get("role") or item.get("liquidity") or ""
            ).upper()
            fee_detail = item.get("feeDetail") or []
            fee_item = fee_detail[0] if isinstance(fee_detail, list) and fee_detail else {}
            fills.append(
                TradeInfo(
                    trade_id=str(item.get("execId") or item.get("tradeId") or item.get("fillId") or item.get("id") or ""),
                    order_id=str(item.get("orderId") or "") or None,
                    symbol=str(item.get("symbol") or symbol or ""),
                    side="BUY" if side_raw in {"BUY", "BID"} else "SELL",
                    size=self._decimal(item.get("execQty") or item.get("qty") or item.get("size"), Decimal("0")) or Decimal("0"),
                    price=self._decimal(item.get("execPrice") or item.get("price"), Decimal("0")) or Decimal("0"),
                    fee=self._decimal(item.get("fee") or fee_item.get("fee")),
                    fee_asset=str(item.get("feeToken") or item.get("feeAsset") or fee_item.get("feeToken") or "") or None,
                    timestamp=self._timestamp_ms(item.get("updatedAt") or item.get("createdAt") or item.get("timestamp")),
                    is_maker=True if role == "MAKER" else False if role == "TAKER" else None,
                    raw=item,
                )
            )
        return ApiResponse.ok(fills, raw=raw)

    # ----------------------------- order writes -----------------------------
    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        signer_error = self._require_signer()
        if signer_error:
            return signer_error
        symbol = self._resolve_symbol(order_details.get("symbol"))
        self._ensure_markets()
        market = self._market_cache.get(self._symbol_key(symbol or ""))
        if not market:
            return ApiResponse.error(f"POPDEX market not found: {order_details.get('symbol')}")

        quantity = order_details.get("quantity") or order_details.get("qty") or order_details.get("size")
        order_type = str(order_details.get("orderType") or order_details.get("order_type") or "Limit")
        price = order_details.get("price")
        if order_type.upper() == "MARKET" and price in (None, ""):
            price = 0
        if quantity in (None, "") or price in (None, ""):
            return ApiResponse.error("POPDEX order requires quantity and a price for limit orders")
        slippage = order_details.get("slippage")
        if slippage is None:
            slippage = self.market_slippage if order_type.upper() == "MARKET" else 0

        try:
            prepared_order = self._apply_account_position_mode(order_details)
            params = self._pack_order_params(prepared_order)
            client_id, client_id_bytes = self._client_id_bytes(
                order_details.get("clientId")
                or order_details.get("clientOrderId")
                or order_details.get("client_order_id")
            )
            call_data = self._encode_call(
                self.PLACE_ORDER_SIGNATURE,
                ["address", "bytes32", "uint16", "bytes32", "uint256", "uint256", "uint256", "address", "uint256"],
                [
                    to_checksum_address(self.account_address),
                    client_id_bytes,
                    int(market["symbolId"]),
                    params,
                    self._to_fixed_18(price),
                    self._to_fixed_18(quantity),
                    self._to_fixed_18(slippage),
                    to_checksum_address(str(order_details.get("builder") or self.ZERO_ADDRESS)),
                    self._to_fixed_18(order_details.get("builderFeeRate") or 0),
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            return ApiResponse.error(f"Invalid POPDEX order: {exc}")

        tx_hash, send_error, raw = self._send_contract_call(call_data)
        if send_error:
            return ApiResponse.error(send_error, raw=raw)
        receipt = self._wait_receipt(tx_hash)
        order_id, event_error = self._order_from_receipt(receipt)
        if event_error:
            return ApiResponse.error(event_error, raw={"txHash": tx_hash, "receipt": receipt})
        if not self._receipt_succeeded(receipt):
            return ApiResponse.error("POPDEX order transaction reverted", raw={"txHash": tx_hash, "receipt": receipt})

        lookup_data: Optional[Dict[str, Any]] = None
        if not order_id and self.wait_for_receipt:
            lookup = self.get_order_by_client_id(client_id)
            if lookup.success and isinstance(lookup.data, dict):
                lookup_data = lookup.data
                if lookup_data.get("orderId") is not None:
                    order_id = str(lookup_data["orderId"])

        packed_side = params[2]
        result = OrderResult(
            success=True,
            order_id=order_id or client_id,
            client_order_id=client_id,
            symbol=str(symbol),
            side="BUY" if packed_side == 0 else "SELL",
            order_type=order_type.upper(),
            size=self._decimal(quantity),
            price=self._decimal(price),
            filled_size=Decimal("0"),
            status=str(lookup_data.get("status")).upper() if lookup_data and lookup_data.get("status") else "NEW" if receipt else "SUBMITTED",
            created_at=int(time.time() * 1000),
            raw={"txHash": tx_hash, "receipt": receipt, "order": lookup_data},
        )
        return ApiResponse.ok(result, raw=result.raw)

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> ApiResponse:
        signer_error = self._require_signer()
        if signer_error:
            return signer_error
        try:
            numeric_id = int(order_id) if str(order_id).isdigit() else 0
            _, client_id_bytes = self._client_id_bytes(None if numeric_id else order_id)
            if numeric_id:
                client_id_bytes = bytes(32)
            call_data = self._encode_call(
                self.CANCEL_ORDER_SIGNATURE,
                ["address", "uint128", "bytes32"],
                [to_checksum_address(self.account_address), numeric_id, client_id_bytes],
            )
        except (TypeError, ValueError) as exc:
            return ApiResponse.error(f"Invalid POPDEX cancel request: {exc}")

        tx_hash, send_error, raw = self._send_contract_call(call_data)
        if send_error:
            return ApiResponse.error(send_error, raw=raw)
        receipt = self._wait_receipt(tx_hash)
        if not self._receipt_succeeded(receipt):
            return ApiResponse.error("POPDEX cancel transaction reverted", raw={"txHash": tx_hash, "receipt": receipt})
        event_error = self._event_business_error(
            receipt,
            self.ORDER_CANCEL_EVENT,
            ["uint128", "bytes32", "bool", "uint32"],
            2,
            3,
            "OrderCancel",
        )
        if event_error:
            return ApiResponse.error(event_error, raw={"txHash": tx_hash, "receipt": receipt})
        result = CancelResult(
            success=True,
            order_id=str(order_id),
            cancelled_count=1,
            raw={"txHash": tx_hash, "receipt": receipt},
        )
        return ApiResponse.ok(result, raw=result.raw)

    def cancel_all_orders(self, symbol: str) -> ApiResponse:
        signer_error = self._require_signer()
        if signer_error:
            return signer_error
        resolved = self._resolve_symbol(symbol)
        market = self._market_cache.get(self._symbol_key(resolved or ""))
        if not market:
            return ApiResponse.error(f"POPDEX market not found: {symbol}")
        call_data = self._encode_call(
            self.CANCEL_ALL_SIGNATURE,
            ["address", "uint16", "(bool,uint8)", "(bool,uint8)", "(bool,bool)"],
            [
                to_checksum_address(self.account_address),
                int(market["symbolId"]),
                (True, self.CATEGORY_FUTURES),
                (False, 0),
                (False, False),
            ],
        )
        tx_hash, send_error, raw = self._send_contract_call(call_data)
        if send_error:
            return ApiResponse.error(send_error, raw=raw)
        receipt = self._wait_receipt(tx_hash)
        if not self._receipt_succeeded(receipt):
            return ApiResponse.error("POPDEX cancel-all transaction reverted", raw={"txHash": tx_hash, "receipt": receipt})
        event_error = self._event_business_error(
            receipt,
            self.ORDER_ALL_CANCEL_EVENT,
            ["uint16", "(bool,uint8)", "(bool,uint8)", "(bool,bool)", "bool", "uint32"],
            4,
            5,
            "OrderAllCancel",
        )
        if event_error:
            return ApiResponse.error(event_error, raw={"txHash": tx_hash, "receipt": receipt})
        result = CancelResult(
            success=True,
            cancelled_count=0,
            raw={"txHash": tx_hash, "receipt": receipt},
        )
        return ApiResponse.ok(result, raw=result.raw)
