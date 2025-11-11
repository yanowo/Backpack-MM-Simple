"""HTTP-only Lighter exchange client (no official SDK dependency)."""
from __future__ import annotations

import ctypes
import itertools
import json
import os
import platform
import time
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

from .base_client import BaseExchangeClient
from logger import setup_logger

logger = setup_logger("api.lighter_client")

DEFAULT_HTTP_TIMEOUT = 10.0

DEFAULT_SYMBOL_OVERRIDES: Dict[str, Dict[str, Any]] = {
}

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SIGNER_SEARCH_PATHS = [
    os.path.join(_MODULE_DIR, "signers"),
    os.path.join(os.path.dirname(_MODULE_DIR), "external", "lighter-python", "lighter", "signers"),
    os.path.join(os.path.dirname(_MODULE_DIR), "Signer", "Lighter"),
]
_SIGNER_FILENAMES = {
    ("windows", "amd64"): "signer-amd64.dll",
    ("windows", "x86_64"): "signer-amd64.dll",
    ("linux", "x86_64"): "signer-amd64.so",
    ("linux", "amd64"): "signer-amd64.so",
    ("darwin", "arm64"): "signer-arm64.dylib",
}


class StrOrErr(ctypes.Structure):
    _fields_ = [("str", ctypes.c_char_p), ("err", ctypes.c_char_p)]


class SimpleSignerError(Exception):
    """Raised when the native signer cannot be initialised or used."""


class SimpleSignerClient:
    """Thin wrapper around Lighter's native signer shared library."""

    TX_TYPE_CREATE_ORDER = 14
    TX_TYPE_CANCEL_ORDER = 15

    ORDER_TYPE_LIMIT = 0
    ORDER_TYPE_MARKET = 1
    ORDER_TYPE_STOP_LOSS = 2
    ORDER_TYPE_STOP_LOSS_LIMIT = 3
    ORDER_TYPE_TAKE_PROFIT = 4
    ORDER_TYPE_TAKE_PROFIT_LIMIT = 5

    ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL = 0
    ORDER_TIME_IN_FORCE_GOOD_TILL_TIME = 1
    ORDER_TIME_IN_FORCE_POST_ONLY = 2

    NIL_TRIGGER_PRICE = 0
    DEFAULT_28_DAY_ORDER_EXPIRY = -1
    DEFAULT_IOC_EXPIRY = 0
    DEFAULT_10_MIN_AUTH_EXPIRY = -1
    MINUTE = 60

    def __init__(
        self,
        base_url: str,
        private_key: str,
        account_index: int,
        api_key_index: int = 0,
        *,
        session: Optional[requests.Session] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        signer_dir: Optional[str] = None,
        chain_id: Optional[int] = None,
    ) -> None:
        if not private_key:
            raise SimpleSignerError("API private key is required for signer initialisation")

        self.base_url = base_url.rstrip("/")
        self.account_index = int(account_index)
        self.api_key_index = int(api_key_index or 0)
        self.timeout = timeout or DEFAULT_HTTP_TIMEOUT
        self.verify_ssl = verify_ssl
        self._nonce: Optional[int] = None
        self.session = session or requests.Session()
        self.private_key = self._sanitize_private_key(private_key)
        self.chain_id = int(chain_id) if chain_id is not None else (304 if "mainnet" in self.base_url else 300)

        self.signer = self._load_library(signer_dir)
        self._configure_library()
        self._create_client()

    # ---- initialisation helpers -------------------------------------------------
    def _sanitize_private_key(self, key: str) -> str:
        cleaned = key.strip()
        cleaned = cleaned[2:] if cleaned.startswith("0x") else cleaned
        if len(cleaned) not in (64, 80):
            raise SimpleSignerError(
                "API private key must be 32 or 40 bytes expressed as hex (64 or 80 characters)"
            )
        try:
            int(cleaned, 16)
        except ValueError as exc:
            raise SimpleSignerError("API private key contains non-hex characters") from exc
        return cleaned

    def _load_library(self, signer_dir: Optional[str]) -> ctypes.CDLL:
        system = platform.system().lower()
        arch = platform.machine().lower()
        filename = _SIGNER_FILENAMES.get((system, arch))
        if not filename:
            raise SimpleSignerError(f"Unsupported platform/architecture: {system}/{arch}")

        search_paths: List[str] = []
        if signer_dir:
            search_paths.append(signer_dir)
        search_paths.extend(_DEFAULT_SIGNER_SEARCH_PATHS)

        for candidate_dir in search_paths:
            if not candidate_dir:
                continue
            candidate = os.path.join(candidate_dir, filename)
            if os.path.isfile(candidate):
                return ctypes.CDLL(candidate)
        raise SimpleSignerError(
            f"Unable to locate signer library '{filename}'. "
            "Set `signer_lib_dir` in config or place the library under api/signers/."
        )

    def _configure_library(self) -> None:
        self.signer.CreateClient.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
        ]
        self.signer.CreateClient.restype = ctypes.c_char_p

        self.signer.CheckClient.argtypes = [ctypes.c_int, ctypes.c_longlong]
        self.signer.CheckClient.restype = ctypes.c_char_p

        self.signer.SwitchAPIKey.argtypes = [ctypes.c_int]
        self.signer.SwitchAPIKey.restype = ctypes.c_char_p

        self.signer.CreateAuthToken.argtypes = [ctypes.c_longlong]
        self.signer.CreateAuthToken.restype = StrOrErr

        self.signer.SignCreateOrder.argtypes = [
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        self.signer.SignCreateOrder.restype = StrOrErr

        self.signer.SignCancelOrder.argtypes = [
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        self.signer.SignCancelOrder.restype = StrOrErr

    def _create_client(self) -> None:
        err_ptr = self.signer.CreateClient(
            self.base_url.encode("utf-8"),
            self.private_key.encode("utf-8"),
            ctypes.c_int(self.chain_id),
            ctypes.c_int(self.api_key_index),
            ctypes.c_longlong(self.account_index),
        )
        if err_ptr:
            raise SimpleSignerError(err_ptr.decode("utf-8"))

    # ---- signer primitives ------------------------------------------------------
    def _decode_str_or_err(self, result: StrOrErr) -> Tuple[Optional[str], Optional[str]]:
        payload = result.str.decode("utf-8") if result.str else None
        error = result.err.decode("utf-8") if result.err else None
        return payload, error

    def _fetch_nonce(self) -> int:
        url = f"{self.base_url}/api/v1/nextNonce"
        params = {
            "account_index": self.account_index,
            "api_key_index": self.api_key_index,
        }
        try:
            response = self.session.get(url, params=params, timeout=self.timeout, verify=self.verify_ssl)
        except requests.RequestException as exc:
            raise SimpleSignerError(f"Failed to fetch nonce: {exc}") from exc

        if response.status_code != 200:
            raise SimpleSignerError(f"Nonce request failed with status {response.status_code}: {response.text}")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SimpleSignerError(f"Invalid nonce response: {exc}") from exc

        nonce_value = payload.get("nonce")
        if nonce_value is None:
            raise SimpleSignerError(f"Nonce missing in response: {payload}")
        self._nonce = int(nonce_value) - 1
        return self._nonce

    def _next_nonce(self) -> int:
        if self._nonce is None:
            self._fetch_nonce()
        assert self._nonce is not None
        self._nonce += 1
        return self._nonce

    def _send_tx(self, tx_type: int, tx_info: str, price_protection: bool = True) -> Dict[str, Any]:
        if not tx_info:
            raise SimpleSignerError("Signer returned empty tx_info payload")

        url = f"{self.base_url}/api/v1/sendTx"
        files = {
            "tx_type": (None, str(int(tx_type))),
            "tx_info": (None, tx_info),
            "price_protection": (None, "true" if price_protection else "false"),
        }
        try:
            response = self.session.post(url, files=files, timeout=self.timeout, verify=self.verify_ssl)
        except requests.RequestException as exc:
            raise SimpleSignerError(f"Failed to submit transaction: {exc}") from exc

        try:
            payload = response.json() if response.text else {}
        except json.JSONDecodeError as exc:
            raise SimpleSignerError(f"Failed to decode sendTx response: {exc}") from exc

        if response.status_code != 200:
            message = payload.get("message") or response.text
            raise SimpleSignerError(f"Transaction rejected ({response.status_code}): {message}")
        return payload

    def _send_tx_batch(self, tx_list: List[Tuple[int, str]], price_protection: bool = True) -> Dict[str, Any]:
        """發送批量交易

        Args:
            tx_list: List of (tx_type, tx_info) tuples
            price_protection: Whether to enable price protection

        Returns:
            Response payload from the server
        """
        if not tx_list:
            raise SimpleSignerError("Empty transaction list")

        url = f"{self.base_url}/api/v1/sendTxBatch"

        # 根據官方示例：tx_types 和 tx_infos 應該是 JSON 數組字符串
        # 參考：https://github.com/elliottech/lighter-python/blob/main/examples/send_tx_batch.py
        tx_types = []
        tx_infos = []

        for tx_type, tx_info in tx_list:
            if not tx_info:
                raise SimpleSignerError("Empty tx_info in batch")
            tx_types.append(int(tx_type))
            tx_infos.append(tx_info)

        # 使用 multipart/form-data 格式，參數為 JSON 字符串
        files = {
            "tx_types": (None, json.dumps(tx_types)),
            "tx_infos": (None, json.dumps(tx_infos)),
            "price_protection": (None, "true" if price_protection else "false"),
        }

        logger.debug("Batch request: %d transactions, tx_types: %s", len(tx_list), tx_types)

        try:
            response = self.session.post(url, files=files, timeout=self.timeout * 2, verify=self.verify_ssl)
        except requests.RequestException as exc:
            raise SimpleSignerError(f"Failed to submit batch transaction: {exc}") from exc

        try:
            payload = response.json() if response.text else {}
        except json.JSONDecodeError as exc:
            raise SimpleSignerError(f"Failed to decode sendTxBatch response: {exc}") from exc

        if response.status_code != 200:
            message = payload.get("message") or response.text
            raise SimpleSignerError(f"Batch transaction rejected ({response.status_code}): {message}")
        return payload

    # ---- public API -------------------------------------------------------------
    def check_client(self) -> Optional[str]:
        result = self.signer.CheckClient(ctypes.c_int(self.api_key_index), ctypes.c_longlong(self.account_index))
        return result.decode("utf-8") if result else None

    def create_auth_token_with_expiry(self, deadline: int = DEFAULT_10_MIN_AUTH_EXPIRY) -> Tuple[Optional[str], Optional[str]]:
        actual_deadline = deadline
        if deadline == self.DEFAULT_10_MIN_AUTH_EXPIRY:
            actual_deadline = int(time.time() + 10 * self.MINUTE)
        payload, error = self._decode_str_or_err(self.signer.CreateAuthToken(ctypes.c_longlong(actual_deadline)))
        return payload, error

    def create_order(
        self,
        *,
        market_index: int,
        client_order_index: int,
        base_amount: int,
        price: int,
        is_ask: bool,
        order_type: int,
        time_in_force: int,
        reduce_only: bool = False,
        trigger_price: int = NIL_TRIGGER_PRICE,
        order_expiry: int = DEFAULT_28_DAY_ORDER_EXPIRY,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
        last_payload: Optional[Dict[str, Any]] = None
        for attempt in range(2):
            nonce = self._next_nonce()
            payload, error = self._decode_str_or_err(
                self.signer.SignCreateOrder(
                    ctypes.c_int(market_index),
                    ctypes.c_longlong(client_order_index),
                    ctypes.c_longlong(base_amount),
                    ctypes.c_int(price),
                    ctypes.c_int(int(is_ask)),
                    ctypes.c_int(order_type),
                    ctypes.c_int(time_in_force),
                    ctypes.c_int(int(reduce_only)),
                    ctypes.c_int(trigger_price),
                    ctypes.c_longlong(order_expiry),
                    ctypes.c_longlong(nonce),
                )
            )
            if error:
                return None, None, error
            try:
                last_payload = json.loads(payload) if payload else None
            except json.JSONDecodeError:
                last_payload = {"raw": payload}

            try:
                response = self._send_tx(self.TX_TYPE_CREATE_ORDER, payload or "")
                return last_payload, response, None
            except SimpleSignerError as exc:
                message = str(exc)
                if "invalid nonce" in message.lower() and attempt == 0:
                    self._fetch_nonce()
                    time.sleep(0.25)
                    continue
                return last_payload, None, message
        return last_payload, None, "Unable to submit order after nonce retries"

    def cancel_order(
        self,
        *,
        market_index: int,
        order_index: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
        last_payload: Optional[Dict[str, Any]] = None
        for attempt in range(2):
            nonce = self._next_nonce()
            payload, error = self._decode_str_or_err(
                self.signer.SignCancelOrder(
                    ctypes.c_int(market_index),
                    ctypes.c_longlong(order_index),
                    ctypes.c_longlong(nonce),
                )
            )
            if error:
                return None, None, error
            try:
                last_payload = json.loads(payload) if payload else None
            except json.JSONDecodeError:
                last_payload = {"order_index": order_index, "raw": payload}

            try:
                response = self._send_tx(self.TX_TYPE_CANCEL_ORDER, payload or "")
                return last_payload, response, None
            except SimpleSignerError as exc:
                message = str(exc)
                if "invalid nonce" in message.lower() and attempt == 0:
                    self._fetch_nonce()
                    time.sleep(0.25)
                    continue
                return last_payload, None, message
        return last_payload, None, "Unable to cancel order after nonce retries"

    def create_order_batch(
        self,
        orders: List[Dict[str, Any]],
    ) -> Tuple[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]], Optional[str]]:
        """批量創建訂單

        Args:
            orders: List of order dictionaries, each containing:
                - market_index: int
                - client_order_index: int
                - base_amount: int
                - price: int
                - is_ask: bool
                - order_type: int
                - time_in_force: int
                - reduce_only: bool (optional, default False)
                - trigger_price: int (optional, default NIL_TRIGGER_PRICE)
                - order_expiry: int (optional, default DEFAULT_28_DAY_ORDER_EXPIRY)

        Returns:
            Tuple of (list of payloads, response, error message)
        """
        if not orders:
            return [], None, "Empty order list"

        tx_list: List[Tuple[int, str]] = []
        payloads: List[Optional[Dict[str, Any]]] = []

        for attempt in range(2):
            tx_list.clear()
            payloads.clear()
            all_signed = True

            for order in orders:
                nonce = self._next_nonce()

                # 提取訂單參數
                market_index = order.get("market_index")
                client_order_index = order.get("client_order_index")
                base_amount = order.get("base_amount")
                price = order.get("price")
                is_ask = order.get("is_ask")
                order_type = order.get("order_type")
                time_in_force = order.get("time_in_force")
                reduce_only = order.get("reduce_only", False)
                trigger_price = order.get("trigger_price", self.NIL_TRIGGER_PRICE)
                order_expiry = order.get("order_expiry", self.DEFAULT_28_DAY_ORDER_EXPIRY)

                # 簽名訂單
                payload, error = self._decode_str_or_err(
                    self.signer.SignCreateOrder(
                        ctypes.c_int(market_index),
                        ctypes.c_longlong(client_order_index),
                        ctypes.c_longlong(base_amount),
                        ctypes.c_int(price),
                        ctypes.c_int(int(is_ask)),
                        ctypes.c_int(order_type),
                        ctypes.c_int(time_in_force),
                        ctypes.c_int(int(reduce_only)),
                        ctypes.c_int(trigger_price),
                        ctypes.c_longlong(order_expiry),
                        ctypes.c_longlong(nonce),
                    )
                )

                if error:
                    return payloads, None, error

                try:
                    parsed_payload = json.loads(payload) if payload else None
                except json.JSONDecodeError:
                    parsed_payload = {"raw": payload}

                payloads.append(parsed_payload)
                tx_list.append((self.TX_TYPE_CREATE_ORDER, payload or ""))

            # 發送批量交易
            try:
                response = self._send_tx_batch(tx_list)
                return payloads, response, None
            except SimpleSignerError as exc:
                message = str(exc)
                if "invalid nonce" in message.lower() and attempt == 0:
                    self._fetch_nonce()
                    time.sleep(0.25)
                    continue
                return payloads, None, message

        return payloads, None, "Unable to submit batch orders after nonce retries"


def _compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}

def _get_lihgter_account_index(address):
    # 通過錢包地址查找主賬户
    from eth_utils import to_checksum_address
    import requests

    # 轉換為 EIP-55 校驗格式
    checksum_address = to_checksum_address(address.lower())
    url = 'https://mainnet.zklighter.elliot.ai/api/v1/account?by=l1_address&value='
    full_url = url + checksum_address

    res = requests.get(full_url)
    data = res.json()

    # 提取 account_index
    if 'accounts' in data:
        account_index = data['accounts'][0]['account_index']
        return int(account_index)
    else:
        raise ValueError(f"Account not found for address: {address}")
    
class LighterClient(BaseExchangeClient):
    """HTTP-based Lighter exchange adapter compatible with the strategy layer."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url: str = config.get("base_url").rstrip("/")
        self.verify_ssl: bool = bool(config.get("verify_ssl", True))
        self.timeout: float = float(config.get("timeout", DEFAULT_HTTP_TIMEOUT) or DEFAULT_HTTP_TIMEOUT)
        self.session = requests.Session()
        self.session.verify = self.verify_ssl
        self.session.headers.update(
            {
                "User-Agent": "backpack-mm-lighter/1.0",
                "Accept": "application/json",
            }
        )

        overrides = dict(DEFAULT_SYMBOL_OVERRIDES)
        overrides.update(config.get("symbol_overrides", {}) or {})
        self._raw_overrides: Dict[str, Dict[str, Any]] = overrides
        self._market_cache: Dict[str, Dict[str, Any]] = {}
        self._alias_map: Dict[str, str] = {}
        self._market_id_map: Dict[int, Dict[str, Any]] = {}
        self._allow_fee_rate_inference: bool = bool(config.get("allow_fee_rate_inference", False))
        self._market_fetch_retries: int = max(1, int(config.get("market_fetch_retries", 3)))

        self.account_index: Optional[int] = self._as_int(
            config.get("account_index")
            or config.get("accountIndex")
            or config.get("account_id")
            or config.get("accountId")
        )
        self.api_key_index: int = self._as_int(config.get("api_key_index") or config.get("apiKeyIndex") or 0, default=0)
        self.private_key: Optional[str] = (
            config.get("api_private_key")
            or config.get("private_key")
            or config.get("api_key")
        )
        self.signer_dir: Optional[str] = config.get("signer_lib_dir")
        self.chain_id: Optional[int] = self._as_int(config.get("chain_id"))

        self.auth_token_ttl: int = max(self._as_int(config.get("auth_token_ttl"), default=600) or 600, 120)

        self._signer: Optional[SimpleSignerClient] = None
        self._auth_token: Optional[str] = None
        self._auth_expiry: float = 0.0
        self._client_order_counter = itertools.count(int(time.time() * 1000) % 1_000_000_000)

    # ---- BaseExchangeClient lifecycle ------------------------------------------
    async def connect(self) -> None:
        self._ensure_market_cache()
        signer = self._ensure_signer_client()
        if signer:
            logger.info("Lighter signer initialised for account %s", self.account_index)
        else:
            logger.info("Lighter HTTP client ready (trading disabled: missing signer credentials)")

    async def disconnect(self) -> None:
        self._signer = None
        try:
            self.session.close()
        except Exception:
            pass

    def get_exchange_name(self) -> str:
        return "Lighter"

    # ---- HTTP helpers ----------------------------------------------------------
    def make_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        merged_headers = dict(self.session.headers)
        if headers:
            merged_headers.update(headers)

        query = _compact_dict(params or {})
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=query or None,
                data=data,
                json=json_data,
                headers=merged_headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            return {"error": str(exc)}

        if response.status_code >= 400:
            text = response.text.strip()
            return {"error": f"HTTP {response.status_code}: {text or response.reason}"}

        if not response.text:
            return {}

        try:
            return response.json()
        except json.JSONDecodeError:
            return {"error": "Failed to decode JSON response", "raw": response.text}

    # ---- Signer helpers --------------------------------------------------------
    def _ensure_signer_client(self) -> Optional[SimpleSignerClient]:
        if self._signer is not None:
            return self._signer

        if not self.private_key or self.account_index is None:
            return None

        try:
            signer = SimpleSignerClient(
                base_url=self.base_url,
                private_key=self.private_key,
                account_index=int(self.account_index),
                api_key_index=self.api_key_index,
                session=self.session,
                timeout=self.timeout,
                verify_ssl=self.verify_ssl,
                signer_dir=self.signer_dir,
                chain_id=self.chain_id,
            )
        except SimpleSignerError as exc:
            logger.error("Failed to initialise Lighter signer: %s", exc)
            return None

        mismatch = signer.check_client()
        if mismatch:
            logger.error("Signer key verification failed: %s", mismatch)
            return None

        self._signer = signer
        self._auth_token = None
        self._auth_expiry = 0.0
        return signer

    def _get_auth_token(self) -> Optional[str]:
        signer = self._ensure_signer_client()
        if not signer:
            return None

        now = time.time()
        if self._auth_token and now < self._auth_expiry - 5:
            return self._auth_token

        deadline = int(now + self.auth_token_ttl)
        token, error = signer.create_auth_token_with_expiry(deadline)
        if error or not token:
            logger.error("Failed to generate Lighter auth token: %s", error or "unknown error")
            self._auth_token = None
            return None

        self._auth_token = token
        self._auth_expiry = deadline
        return token

    # ---- Account helpers -------------------------------------------------------
    def _fetch_account_details(self) -> Dict[str, Any]:
        if self.account_index is None:
            return {"error": "Account index is not configured"}

        payload = self.make_request(
            "GET",
            "/api/v1/account",
            params={"by": "index", "value": str(self.account_index)},
        )
        if isinstance(payload, dict) and "error" in payload:
            return payload
        if isinstance(payload, dict):
            accounts = payload.get("accounts")
            if isinstance(accounts, list) and accounts:
                primary = accounts[0]
                if isinstance(primary, dict):
                    return primary
        return payload

    # ---- Utility conversions ---------------------------------------------------
    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        decimal_value = self._safe_decimal(value)
        if decimal_value is None:
            return None
        return float(decimal_value)

    def _as_int(self, value: Any, *, default: Optional[int] = None) -> Optional[int]:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _as_bool(self, value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value in (None, "", "NaN"):
            return None
        if isinstance(value, (int, float, Decimal)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "y", "t"):
                return True
            if lowered in ("false", "0", "no", "n", "f"):
                return False
        return None

    def _scale_to_int(self, value: Any, precision: int) -> Optional[int]:
        decimal_value = self._safe_decimal(value)
        if decimal_value is None:
            return None
        factor = Decimal(10) ** precision
        return int((decimal_value * factor).to_integral_value(rounding=ROUND_DOWN))

    def _next_client_order_index(self) -> int:
        return next(self._client_order_counter)

    # ---- Market metadata -------------------------------------------------------
    def _normalize_symbol_key(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    def _infer_tick_size(self, decimals: Optional[int]) -> str:
        if decimals is None or decimals <= 0:
            return "1"
        return "0." + "0" * (decimals - 1) + "1"

    def _apply_override(
        self,
        base_entry: Dict[str, Any],
        override: Optional[Dict[str, Any]],
        alias_symbol: str,
    ) -> Dict[str, Any]:
        entry = dict(base_entry)
        entry["symbol"] = alias_symbol
        if override:
            for field in (
                "base_asset",
                "quote_asset",
                "market_type",
                "status",
                "min_order_size",
                "tick_size",
                "base_precision",
                "quote_precision",
                "market_id",
            ):
                if override.get(field) is not None:
                    entry[field] = override[field]
        return entry

    def _build_market_entry(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = item.get("symbol")
        if not symbol:
            return None

        key = self._normalize_symbol_key(symbol)
        override = self._raw_overrides.get(symbol) or self._raw_overrides.get(key)

        base_asset = (
            (override.get("base_asset") if override else None)
            or item.get("base_asset")
            or symbol
        )
        quote_asset = (
            (override.get("quote_asset") if override else None)
            or item.get("quote_asset")
            or "USDT"
        )
        market_type = (override.get("market_type") if override else None) or item.get("market_type") or "PERP"
        status = (override.get("status") if override else None) or item.get("status") or "TRADING"

        base_precision = (
            override.get("base_precision")
            if override and override.get("base_precision") is not None
            else item.get("size_decimals") or item.get("supported_size_decimals") or 3
        )
        quote_precision = (
            override.get("quote_precision")
            if override and override.get("quote_precision") is not None
            else item.get("price_decimals") or item.get("supported_price_decimals") or 3
        )

        min_order_size = (
            override.get("min_order_size")
            if override and override.get("min_order_size") is not None
            else item.get("min_base_amount") or item.get("min_order_size") or "0"
        )

        tick_size = (
            override.get("tick_size")
            if override and override.get("tick_size") is not None
            else item.get("tick_size") or self._infer_tick_size(item.get("supported_price_decimals"))
        )

        market_id = item.get("market_id") or item.get("id")
        last_price = item.get("last_trade_price") or item.get("lastPrice")

        try:
            base_precision = int(base_precision)
        except (TypeError, ValueError):
            base_precision = 3

        try:
            quote_precision = int(quote_precision)
        except (TypeError, ValueError):
            quote_precision = 3

        if market_id is not None:
            try:
                market_id = int(market_id)
            except (TypeError, ValueError):
                pass

        return {
            "symbol": symbol,
            "base_asset": base_asset,
            "quote_asset": quote_asset,
            "market_type": market_type,
            "status": status,
            "min_order_size": str(min_order_size),
            "tick_size": str(tick_size),
            "base_precision": base_precision,
            "quote_precision": quote_precision,
            "market_id": market_id,
            "last_price": self._safe_float(last_price),
        }

    def _fetch_markets(self) -> List[Dict[str, Any]]:
        last_error: Optional[str] = None
        for attempt in range(1, self._market_fetch_retries + 1):
            payload = self.make_request("GET", "/api/v1/orderBookDetails")
            if isinstance(payload, dict) and "error" in payload:
                last_error = payload["error"]
                logger.error(
                    "Failed to fetch Lighter markets (attempt %d/%d): %s",
                    attempt,
                    self._market_fetch_retries,
                    last_error,
                )
                if attempt < self._market_fetch_retries:
                    time.sleep(min(1.0, 0.2 * attempt))
                continue

            details = payload.get("order_book_details") if isinstance(payload, dict) else None
            if not isinstance(details, list):
                logger.error(
                    "Unexpected market metadata format (attempt %d/%d): %s",
                    attempt,
                    self._market_fetch_retries,
                    type(details).__name__,
                )
                if attempt < self._market_fetch_retries:
                    time.sleep(min(1.0, 0.2 * attempt))
                    continue
                return []

            return [entry for entry in details if isinstance(entry, dict)]

        if last_error:
            logger.error("Failed to fetch Lighter markets after retries: %s", last_error)
        return []

    def _ensure_market_cache(self) -> None:
        if self._market_cache:
            return

        items = self._fetch_markets()
        if not items:
            return

        cache: Dict[str, Dict[str, Any]] = {}
        alias_map: Dict[str, str] = {}
        id_map: Dict[int, Dict[str, Any]] = {}

        for item in items:
            entry = self._build_market_entry(item)
            if not entry:
                continue
            key = self._normalize_symbol_key(entry["symbol"])
            cache[key] = entry
            market_id = entry.get("market_id")
            try:
                market_id_int = int(market_id) if market_id is not None else None
            except (TypeError, ValueError):
                market_id_int = None
            if market_id_int is not None:
                id_map[market_id_int] = entry

        for alias, override in self._raw_overrides.items():
            alias_key = self._normalize_symbol_key(alias)
            target_symbol = override.get("symbol") or alias
            target_key = self._normalize_symbol_key(target_symbol)
            alias_map[alias_key] = target_key

            target_entry = cache.get(target_key)
            if not target_entry:
                continue
            alias_entry = self._apply_override(target_entry, override, alias)
            cache[alias_key] = alias_entry
            market_id = alias_entry.get("market_id")
            try:
                market_id_int = int(market_id) if market_id is not None else None
            except (TypeError, ValueError):
                market_id_int = None
            if market_id_int is not None:
                id_map[market_id_int] = alias_entry

        self._market_cache = cache
        self._alias_map = alias_map
        self._market_id_map = id_map

    def _lookup_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._ensure_market_cache()
        key = self._normalize_symbol_key(symbol)

        entry = self._market_cache.get(key)
        if entry:
            if entry["symbol"] != symbol:
                entry = dict(entry)
                entry["symbol"] = symbol
            return entry

        target_key = self._alias_map.get(key)
        if target_key and target_key in self._market_cache:
            base_entry = self._market_cache[target_key]
            override = self._raw_overrides.get(symbol) or self._raw_overrides.get(key) or {}
            return self._apply_override(base_entry, override, symbol)
        return None

    def _lookup_market_by_id(self, market_id: Any) -> Optional[Dict[str, Any]]:
        self._ensure_market_cache()
        try:
            market_id_int = int(market_id)
        except (TypeError, ValueError):
            return None

        entry = self._market_id_map.get(market_id_int)
        if entry:
            return entry

        for candidate in self._market_cache.values():
            try:
                candidate_id = int(candidate.get("market_id"))
            except (TypeError, ValueError):
                continue
            if candidate_id == market_id_int:
                return candidate
        return None

    # ---- Public REST wrappers ---------------------------------------------------
    def get_markets(self) -> List[Dict[str, Any]]:
        self._ensure_market_cache()
        return [dict(value) for value in self._market_cache.values()]

    def get_market_limits(self, symbol: str) -> Optional[Dict[str, Any]]:
        market = self._lookup_market(symbol)
        if market:
            return dict(market)

        markets_info = self.get_markets()
        if isinstance(markets_info, list):
            normalized_symbol = self._normalize_symbol_key(symbol)
            for market_info in markets_info:
                if not isinstance(market_info, dict):
                    continue

                entry_symbol = market_info.get("symbol") or market_info.get("symbolName")
                if not entry_symbol:
                    continue

                if entry_symbol == symbol or self._normalize_symbol_key(entry_symbol) == normalized_symbol:
                    base_asset = (
                        market_info.get("base_asset")
                        or market_info.get("baseSymbol")
                        or market_info.get("baseToken")
                        or market_info.get("baseAsset")
                        or symbol.split("/")[0]
                    )
                    quote_asset = (
                        market_info.get("quote_asset")
                        or market_info.get("quoteSymbol")
                        or market_info.get("quoteToken")
                        or market_info.get("quoteAsset")
                        or symbol.split("/")[-1]
                    )

                    base_precision = market_info.get("base_precision")
                    quote_precision = market_info.get("quote_precision")

                    for candidate in (
                        "supported_size_decimals",
                        "size_decimals",
                        "base_precision_digits",
                        "basePrecision",
                    ):
                        if base_precision is None and market_info.get(candidate) is not None:
                            try:
                                base_precision = int(market_info[candidate])
                            except (TypeError, ValueError):
                                pass

                    for candidate in (
                        "supported_price_decimals",
                        "price_decimals",
                        "quote_precision_digits",
                        "pricePrecision",
                    ):
                        if quote_precision is None and market_info.get(candidate) is not None:
                            try:
                                quote_precision = int(market_info[candidate])
                            except (TypeError, ValueError):
                                pass

                    if base_precision is None:
                        base_precision = 3
                    if quote_precision is None:
                        quote_precision = 3

                    min_order_size = (
                        market_info.get("min_order_size")
                        or market_info.get("min_base_amount")
                        or market_info.get("minQuantity")
                        or market_info.get("minTradeSize")
                        or "0"
                    )
                    tick_size = (
                        market_info.get("tick_size")
                        or market_info.get("tickSize")
                        or market_info.get("priceIncrement")
                    )
                    if tick_size is None:
                        tick_size = self._infer_tick_size(int(quote_precision))

                    return {
                        "base_asset": base_asset,
                        "quote_asset": quote_asset,
                        "base_precision": int(base_precision),
                        "quote_precision": int(quote_precision),
                        "min_order_size": str(min_order_size),
                        "tick_size": str(tick_size),
                    }

            logger.error("Unable to find market metadata for %s", symbol)
            return None

        logger.error("Failed to retrieve market list when resolving %s: %s", symbol, markets_info)
        return None

    def get_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        market = self._lookup_market(symbol)
        if not market:
            return {"error": f"Unknown symbol {symbol}"}

        market_id = market.get("market_id")
        if market_id is None:
            return {"error": f"Market id missing for symbol {symbol}"}

        payload = self.make_request(
            "GET",
            "/api/v1/orderBookOrders",
            params={"market_id": market_id, "limit": max(1, min(limit, 100))},
        )
        if isinstance(payload, dict) and "error" in payload:
            return payload

        bids = self._convert_levels(payload.get("bids"))
        asks = self._convert_levels(payload.get("asks"))
        return {"symbol": symbol, "bids": bids, "asks": asks}

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        book = self.get_order_book(symbol, limit=50)
        if "error" in book:
            return book

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None

        market = self._lookup_market(symbol)
        last_price = market.get("last_price") if market else None
        if last_price is None:
            last_price = best_bid or best_ask

        return {
            "symbol": symbol,
            "bidPrice": f"{best_bid:.10f}" if best_bid is not None else None,
            "askPrice": f"{best_ask:.10f}" if best_ask is not None else None,
            "lastPrice": f"{last_price:.10f}" if last_price is not None else None,
        }

    def get_order_book_snapshot(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        return self.get_order_book(symbol, limit)

    def get_balance(self) -> Dict[str, Any]:
        account = self._fetch_account_details()
        if isinstance(account, dict) and "error" in account:
            return account

        available = self._safe_float(account.get("available_balance")) or 0.0
        collateral = self._safe_float(account.get("collateral")) or 0.0
        locked = max(collateral - available, 0.0)
        total = available + locked

        # Lighter使用USDC作為統一抵押品，同時提供USD/USDT別名以兼容不同策略
        balance_info = {
            "asset": "USDC",
            "available": available,
            "locked": locked,
            "total": total,
            "collateral": collateral,
        }

        return {
            "USDC": balance_info,
            "USD": balance_info,   # 別名，指向同一數據
            "USDT": balance_info,  # 別名，指向同一數據
        }

    def get_collateral(self, subaccount_id: Optional[str] = None) -> Dict[str, Any]:
        account = self._fetch_account_details()
        if isinstance(account, dict) and "error" in account:
            return account

        collateral = self._safe_float(account.get("collateral")) or 0.0
        available = self._safe_float(account.get("available_balance")) or 0.0
        total_asset_value = self._safe_float(account.get("total_asset_value"))
        cross_asset_value = self._safe_float(account.get("cross_asset_value"))

        response: Dict[str, Any] = {
            "totalCollateral": collateral,
            "availableCollateral": available,
            "assets": [
                {
                    "symbol": "USDC",
                    "totalQuantity": collateral,
                    "availableQuantity": available,
                }
            ],
        }
        if total_asset_value is not None:
            response["totalAssetValue"] = total_asset_value
        if cross_asset_value is not None:
            response["crossAssetValue"] = cross_asset_value
        return response

    def execute_order_batch(self, orders_details: List[Dict[str, Any]]) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """批量執行訂單

        Args:
            orders_details: List of order detail dictionaries

        Returns:
            List of order results or error dict
        """
        signer = self._ensure_signer_client()
        if not signer:
            return {"error": "Signer client is not configured"}

        if not orders_details:
            return {"error": "Empty order list"}

        # 預處理所有訂單
        processed_orders = []

        for order_details in orders_details:
            symbol = order_details.get("symbol")
            if not symbol:
                logger.warning("跳過無效訂單: 缺少 symbol")
                continue

            market = self._lookup_market(symbol)
            if not market:
                logger.warning("跳過無效訂單: 未知交易對 %s", symbol)
                continue

            market_id = market.get("market_id")
            if market_id is None:
                logger.warning("跳過無效訂單: %s 缺少 market_id", symbol)
                continue

            base_precision = int(market.get("base_precision", 3))
            quote_precision = int(market.get("quote_precision", 3))

            # 處理訂單類型
            order_type_raw = (order_details.get("orderType") or order_details.get("type") or "limit").upper()
            order_type_map = {
                "LIMIT": SimpleSignerClient.ORDER_TYPE_LIMIT,
                "MARKET": SimpleSignerClient.ORDER_TYPE_MARKET,
                "STOP": SimpleSignerClient.ORDER_TYPE_STOP_LOSS,
                "STOP_LIMIT": SimpleSignerClient.ORDER_TYPE_STOP_LOSS_LIMIT,
                "TAKE_PROFIT": SimpleSignerClient.ORDER_TYPE_TAKE_PROFIT,
                "TAKE_PROFIT_LIMIT": SimpleSignerClient.ORDER_TYPE_TAKE_PROFIT_LIMIT,
            }
            order_type = order_type_map.get(order_type_raw, SimpleSignerClient.ORDER_TYPE_LIMIT)

            # 處理買賣方向
            side_raw = str(order_details.get("side", "")).upper()
            is_ask = side_raw in ("ASK", "SELL", "SELL_SHORT")

            # 處理價格和數量
            price_value = order_details.get("price")
            quantity_value = order_details.get("quantity") or order_details.get("size")
            if quantity_value is None:
                logger.warning("跳過無效訂單: 缺少 quantity")
                continue

            scaled_price = self._scale_to_int(price_value, quote_precision)
            scaled_quantity = self._scale_to_int(quantity_value, base_precision)

            # 處理市價單
            if scaled_price is None:
                if order_type == SimpleSignerClient.ORDER_TYPE_MARKET:
                    book = self.get_order_book(symbol, limit=1)
                    reference_price: Optional[float] = None
                    if isinstance(book, dict) and "error" not in book:
                        if is_ask:
                            bids = book.get("bids") or []
                            if bids:
                                reference_price = float(bids[0][0]) * 0.999
                        else:
                            asks = book.get("asks") or []
                            if asks:
                                reference_price = float(asks[0][0]) * 1.001
                    if reference_price is None:
                        reference_price = market.get("last_price")
                    if reference_price is None:
                        logger.warning("跳過無效訂單: 無法獲取市價")
                        continue
                    price_value = reference_price
                    scaled_price = self._scale_to_int(price_value, quote_precision)
                else:
                    logger.warning("跳過無效訂單: 無效價格格式")
                    continue

            if scaled_price is None or scaled_quantity is None:
                logger.warning("跳過無效訂單: 無效價格或數量")
                continue

            # 處理 time_in_force
            time_in_force_raw = (order_details.get("timeInForce") or order_details.get("time_in_force") or "GTC").upper()
            post_only = bool(order_details.get("postOnly") or order_details.get("post_only"))
            if post_only:
                time_in_force = SimpleSignerClient.ORDER_TIME_IN_FORCE_POST_ONLY
            else:
                tif_map = {
                    "GTC": SimpleSignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    "IOC": SimpleSignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                    "FOK": SimpleSignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                    "PO": SimpleSignerClient.ORDER_TIME_IN_FORCE_POST_ONLY,
                    "POST_ONLY": SimpleSignerClient.ORDER_TIME_IN_FORCE_POST_ONLY,
                }
                time_in_force = tif_map.get(time_in_force_raw, SimpleSignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME)
            if order_type == SimpleSignerClient.ORDER_TYPE_MARKET:
                time_in_force = SimpleSignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL

            # 處理 reduce_only
            reduce_only = bool(order_details.get("reduceOnly") or order_details.get("reduce_only"))

            # 處理觸發價格
            trigger_price_raw = order_details.get("triggerPrice") or order_details.get("trigger_price")
            if trigger_price_raw is None:
                scaled_trigger_price = SimpleSignerClient.NIL_TRIGGER_PRICE
            else:
                scaled_trigger_price = self._scale_to_int(trigger_price_raw, quote_precision)
                if scaled_trigger_price is None:
                    logger.warning("跳過無效訂單: 無效觸發價格")
                    continue

            # 處理訂單過期時間
            expiry_raw = order_details.get("orderExpiry") or order_details.get("order_expiry")
            default_expiry = (
                SimpleSignerClient.DEFAULT_IOC_EXPIRY
                if order_type == SimpleSignerClient.ORDER_TYPE_MARKET
                else SimpleSignerClient.DEFAULT_28_DAY_ORDER_EXPIRY
            )
            order_expiry = self._as_int(expiry_raw, default=default_expiry)

            # 處理客戶端訂單 ID
            client_order_raw = (
                order_details.get("clientOrderIndex")
                or order_details.get("clientOrderId")
                or order_details.get("client_order_id")
            )
            client_order_index = self._as_int(client_order_raw)
            if client_order_index is None:
                client_order_index = self._next_client_order_index()

            # 添加到處理列表
            processed_orders.append({
                "market_index": int(market_id),
                "client_order_index": int(client_order_index),
                "base_amount": int(scaled_quantity),
                "price": int(scaled_price),
                "is_ask": is_ask,
                "order_type": order_type,
                "time_in_force": time_in_force,
                "reduce_only": reduce_only,
                "trigger_price": int(scaled_trigger_price),
                "order_expiry": int(order_expiry),
                # 保存原始信息用於返回
                "symbol": symbol,
                "original_price": self._safe_float(price_value),
                "original_quantity": self._safe_float(quantity_value),
            })

        if not processed_orders:
            return {"error": "No valid orders to submit"}

        # 批量簽名並發送訂單
        tx_payloads, tx_response, error = signer.create_order_batch(processed_orders)

        if error:
            return {"error": error}

        if not tx_response or tx_response.get("code") != 200:
            message = tx_response.get("message") if tx_response else "unknown error"
            return {"error": f"Batch orders rejected: {message}", "response": tx_response}

        # 構建返回結果
        results = []
        for i, order in enumerate(processed_orders):
            result = {
                "id": str(order["client_order_index"]),
                "clientOrderIndex": order["client_order_index"],
                "symbol": order["symbol"],
                "side": "Ask" if order["is_ask"] else "Bid",
                "price": order["original_price"],
                "quantity": order["original_quantity"],
                "status": "pending",
            }
            # 如果有 tx_hash，添加到結果中
            if tx_response:
                result["txHash"] = tx_response.get("tx_hash")
            results.append(result)

        logger.info("批量下單成功: %d 個訂單", len(results))
        return results

    def execute_order(self, order_details: Dict[str, Any]) -> Dict[str, Any]:
        signer = self._ensure_signer_client()
        if not signer:
            return {"error": "Signer client is not configured"}

        symbol = order_details.get("symbol")
        if not symbol:
            return {"error": "Order symbol is required"}

        market = self._lookup_market(symbol)
        if not market:
            return {"error": f"Unknown symbol {symbol}"}

        market_id = market.get("market_id")
        if market_id is None:
            return {"error": f"Market id missing for {symbol}"}

        base_precision = int(market.get("base_precision", 3))
        quote_precision = int(market.get("quote_precision", 3))

        order_type_raw = (order_details.get("orderType") or order_details.get("type") or "limit").upper()
        order_type_map = {
            "LIMIT": SimpleSignerClient.ORDER_TYPE_LIMIT,
            "MARKET": SimpleSignerClient.ORDER_TYPE_MARKET,
            "STOP": SimpleSignerClient.ORDER_TYPE_STOP_LOSS,
            "STOP_LIMIT": SimpleSignerClient.ORDER_TYPE_STOP_LOSS_LIMIT,
            "TAKE_PROFIT": SimpleSignerClient.ORDER_TYPE_TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": SimpleSignerClient.ORDER_TYPE_TAKE_PROFIT_LIMIT,
        }
        order_type = order_type_map.get(order_type_raw, SimpleSignerClient.ORDER_TYPE_LIMIT)

        side_raw = str(order_details.get("side", "")).upper()
        is_ask = side_raw in ("ASK", "SELL", "SELL_SHORT", "SELL_SHORT")

        price_value = order_details.get("price")
        quantity_value = order_details.get("quantity") or order_details.get("size")
        if quantity_value is None:
            return {"error": "Both price and quantity are required"}

        scaled_price = self._scale_to_int(price_value, quote_precision)
        scaled_quantity = self._scale_to_int(quantity_value, base_precision)

        if scaled_price is None:
            if order_type == SimpleSignerClient.ORDER_TYPE_MARKET:
                book = self.get_order_book(symbol, limit=1)
                reference_price: Optional[float] = None
                if isinstance(book, dict) and "error" not in book:
                    if is_ask:
                        bids = book.get("bids") or []
                        if bids:
                            reference_price = float(bids[0][0]) * 0.999
                    else:
                        asks = book.get("asks") or []
                        if asks:
                            reference_price = float(asks[0][0]) * 1.001
                if reference_price is None:
                    reference_price = market.get("last_price")
                if reference_price is None:
                    return {"error": "Market price unavailable for market order"}
                price_value = reference_price
                scaled_price = self._scale_to_int(price_value, quote_precision)
            else:
                return {"error": "Invalid price format"}

        if scaled_price is None or scaled_quantity is None:
            return {"error": "Invalid price or quantity format"}

        time_in_force_raw = (order_details.get("timeInForce") or order_details.get("time_in_force") or "GTC").upper()
        post_only = bool(order_details.get("postOnly") or order_details.get("post_only"))
        if post_only:
            time_in_force = SimpleSignerClient.ORDER_TIME_IN_FORCE_POST_ONLY
        else:
            tif_map = {
                "GTC": SimpleSignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                "IOC": SimpleSignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                "FOK": SimpleSignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                "PO": SimpleSignerClient.ORDER_TIME_IN_FORCE_POST_ONLY,
                "POST_ONLY": SimpleSignerClient.ORDER_TIME_IN_FORCE_POST_ONLY,
            }
            time_in_force = tif_map.get(time_in_force_raw, SimpleSignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME)
        if order_type == SimpleSignerClient.ORDER_TYPE_MARKET:
            time_in_force = SimpleSignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL

        reduce_only = bool(order_details.get("reduceOnly") or order_details.get("reduce_only"))

        trigger_price_raw = order_details.get("triggerPrice") or order_details.get("trigger_price")
        if trigger_price_raw is None:
            scaled_trigger_price = SimpleSignerClient.NIL_TRIGGER_PRICE
        else:
            scaled_trigger_price = self._scale_to_int(trigger_price_raw, quote_precision)
            if scaled_trigger_price is None:
                return {"error": "Invalid trigger price"}

        expiry_raw = order_details.get("orderExpiry") or order_details.get("order_expiry")
        default_expiry = (
            SimpleSignerClient.DEFAULT_IOC_EXPIRY
            if order_type == SimpleSignerClient.ORDER_TYPE_MARKET
            else SimpleSignerClient.DEFAULT_28_DAY_ORDER_EXPIRY
        )
        order_expiry = self._as_int(expiry_raw, default=default_expiry)

        client_order_raw = (
            order_details.get("clientOrderIndex")
            or order_details.get("clientOrderId")
            or order_details.get("client_order_id")
        )
        client_order_index = self._as_int(client_order_raw)
        if client_order_index is None:
            client_order_index = self._next_client_order_index()

        tx_payload, tx_response, error = signer.create_order(
            market_index=int(market_id),
            client_order_index=int(client_order_index),
            base_amount=int(scaled_quantity),
            price=int(scaled_price),
            is_ask=is_ask,
            order_type=order_type,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            trigger_price=int(scaled_trigger_price),
            order_expiry=int(order_expiry),
        )

        if error:
            return {"error": error}

        if not tx_response or tx_response.get("code") != 200:
            message = tx_response.get("message") if tx_response else "unknown error"
            return {"error": f"Order rejected: {message}", "response": tx_response}

        open_orders = self.get_open_orders(symbol)
        if isinstance(open_orders, list):
            for order in open_orders:
                candidate = (
                    order.get("clientOrderIndex")
                    or order.get("orderIndex")
                    or order.get("id")
                    or order.get("orderId")
                )
                try:
                    if candidate is not None and int(candidate) == int(client_order_index):
                        order["txHash"] = tx_response.get("tx_hash")
                        return order
                except (TypeError, ValueError):
                    continue

        return {
            "id": str(client_order_index),
            "clientOrderIndex": client_order_index,
            "symbol": symbol,
            "side": "Ask" if is_ask else "Bid",
            "price": self._safe_float(price_value),
            "quantity": self._safe_float(quantity_value),
            "status": "pending",
            "txHash": tx_response.get("tx_hash") if tx_response else None,
        }

    def get_open_orders(self, symbol: Optional[str] = None) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if not symbol:
            return {"error": "Symbol is required"}

        if self.account_index is None:
            return {"error": "Account index is not configured"}

        market = self._lookup_market(symbol)
        if not market:
            return {"error": f"Unknown symbol {symbol}"}

        market_id = market.get("market_id")
        if market_id is None:
            return {"error": f"Market id missing for {symbol}"}

        auth = self._get_auth_token()
        if auth is None:
            return {"error": "Unable to generate auth token"}

        payload = self.make_request(
            "GET",
            "/api/v1/accountActiveOrders",
            params={
                "account_index": int(self.account_index),
                "market_id": int(market_id),
                "auth": auth,
            },
            headers={"authorization": auth},
        )
        if isinstance(payload, dict) and "error" in payload:
            return payload

        source = payload.get("orders") if isinstance(payload, dict) else None
        orders: List[Dict[str, Any]] = []
        if isinstance(source, list):
            for entry in source:
                if isinstance(entry, dict):
                    orders.append(self._normalize_order_record(entry, symbol))
        return orders

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        open_orders = self.get_open_orders(symbol)
        if isinstance(open_orders, dict) and "error" in open_orders:
            return open_orders

        if not isinstance(open_orders, list):
            return {"error": "Unexpected open orders payload"}

        cancelled = 0
        errors: List[str] = []
        for order in open_orders:
            identifier = (
                order.get("clientOrderIndex")
                or order.get("orderIndex")
                or order.get("orderId")
                or order.get("id")
            )
            if identifier is None:
                continue
            result = self.cancel_order(str(identifier), symbol)
            if isinstance(result, dict) and "error" in result:
                errors.append(result["error"])
            else:
                cancelled += 1

        response: Dict[str, Any] = {"cancelled": cancelled}
        if errors:
            response["errors"] = errors
        return response

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        signer = self._ensure_signer_client()
        if not signer:
            return {"error": "Signer client is not configured"}

        market = self._lookup_market(symbol)
        if not market:
            return {"error": f"Unknown symbol {symbol}"}

        market_id = market.get("market_id")
        if market_id is None:
            return {"error": f"Market id missing for {symbol}"}

        order_index = self._as_int(order_id)
        if order_index is None:
            open_orders = self.get_open_orders(symbol)
            if isinstance(open_orders, list):
                for order in open_orders:
                    if str(order.get("id")) == str(order_id) or str(order.get("orderId")) == str(order_id):
                        candidate = order.get("clientOrderIndex") or order.get("orderIndex")
                        order_index = self._as_int(candidate)
                        if order_index is not None:
                            break
            elif isinstance(open_orders, dict) and "error" in open_orders:
                return open_orders

        if order_index is None:
            return {"error": f"Unable to resolve order index for {order_id}"}

        tx_payload, tx_response, error = signer.cancel_order(
            market_index=int(market_id),
            order_index=int(order_index),
        )

        if error:
            return {"error": error}

        if not tx_response or tx_response.get("code") != 200:
            message = tx_response.get("message") if tx_response else "unknown error"
            return {"error": f"Cancel order failed: {message}"}

        return {
            "id": str(order_index),
            "orderId": str(order_index),
            "symbol": symbol,
            "txHash": tx_response.get("tx_hash") if tx_response else None,
            "status": "cancelled",
            "payload": tx_payload,
        }

    def get_fill_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if self.account_index is None:
            return {"error": "Account index is not configured"}

        market_id: Optional[int] = None
        if symbol:
            market = self._lookup_market(symbol)
            if not market:
                return {"error": f"Unknown symbol {symbol}"}
            market_id = market.get("market_id")
            if market_id is None:
                return {"error": f"Market id missing for {symbol}"}

        auth = self._get_auth_token()
        if auth is None:
            return {"error": "Unable to generate auth token"}

        params: Dict[str, Any] = {
            "sort_by": "timestamp",
            "sort_dir": "desc",
            "limit": max(1, min(limit, 100)),
            "account_index": int(self.account_index),
            "auth": auth,
        }
        if market_id is not None:
            params["market_id"] = int(market_id)

        payload = self.make_request(
            "GET",
            "/api/v1/trades",
            params=params,
            headers={"authorization": auth},
        )
        if isinstance(payload, dict) and "error" in payload:
            return payload

        trades = payload.get("trades") if isinstance(payload, dict) else None
        records: List[Dict[str, Any]] = []
        if isinstance(trades, list):
            for entry in trades:
                if isinstance(entry, dict):
                    records.append(self._normalize_trade_record(entry))
        return records

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        market = self._lookup_market(symbol)
        if not market:
            return []

        market_id = market.get("market_id")
        if market_id is None:
            return []

        interval_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        duration = interval_seconds.get(interval, 3600)
        end_time = int(time.time())
        start_time = end_time - duration * max(limit, 1)

        payload = self.make_request(
            "GET",
            "/api/v1/candlesticks",
            params={
                "market_id": int(market_id),
                "resolution": interval,
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "count_back": max(limit, 1),
            },
        )
        if isinstance(payload, dict) and "error" in payload:
            return []

        items = payload.get("candlesticks") if isinstance(payload, dict) else None
        klines: List[Dict[str, Any]] = []
        if isinstance(items, list):
            for candle in items:
                if isinstance(candle, dict):
                    klines.append(
                        {
                            "open_time": candle.get("timestamp"),
                            "close_time": candle.get("timestamp"),
                            "open": self._safe_float(candle.get("open")),
                            "high": self._safe_float(candle.get("high")),
                            "low": self._safe_float(candle.get("low")),
                            "close": self._safe_float(candle.get("close")),
                            "volume": self._safe_float(candle.get("volume0")),
                            "quote_volume": self._safe_float(candle.get("volume1")),
                        }
                    )
        return klines

    def get_positions(self, symbol: Optional[str] = None) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        account = self._fetch_account_details()
        if isinstance(account, dict) and "error" in account:
            return account

        positions = account.get("positions", [])
        results: List[Dict[str, Any]] = []
        target_key = self._normalize_symbol_key(symbol) if symbol else None
        if isinstance(positions, list):
            for position in positions:
                if not isinstance(position, dict):
                    continue
                record = self._convert_position(position)
                if target_key:
                    pos_symbol = record.get("symbol")
                    if not pos_symbol or self._normalize_symbol_key(pos_symbol) != target_key:
                        continue
                results.append(record)
        return results

    # ---- Normalisation helpers -------------------------------------------------
    def _convert_levels(self, levels: Optional[Sequence[Dict[str, Any]]]) -> List[List[float]]:
        result: List[List[float]] = []
        if not levels:
            return result
        for level in levels:
            if not isinstance(level, dict):
                continue
            try:
                price = float(level.get("price"))
            except (TypeError, ValueError):
                continue
            quantity_value = (
                level.get("remaining_base_amount")
                or level.get("initial_base_amount")
                or level.get("size")
                or level.get("quantity")
                or 0
            )
            try:
                quantity = float(quantity_value)
            except (TypeError, ValueError):
                quantity = 0.0
            result.append([price, quantity])
        return result

    def _resolve_position_symbol(self, payload: Dict[str, Any]) -> Optional[str]:
        market_id = payload.get("market_id")
        symbol = payload.get("symbol")
        market_entry = self._lookup_market_by_id(market_id) if market_id is not None else None
        if market_entry:
            return market_entry.get("symbol") or symbol
        return symbol

    def _convert_position(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        symbol = self._resolve_position_symbol(payload)
        size_value = self._safe_float(payload.get("position")) or 0.0
        sign_flag = payload.get("sign")
        if sign_flag is not None:
            try:
                sign_int = int(float(sign_flag))
            except (TypeError, ValueError):
                sign_int = 0
            if sign_int > 0:
                size_value = abs(size_value)
            elif sign_int < 0:
                size_value = -abs(size_value)
            else:
                size_value = 0.0
        side = "LONG" if size_value > 0 else "SHORT" if size_value < 0 else "FLAT"

        return {
            "symbol": symbol,
            "side": side,
            "size": abs(size_value),
            "rawSize": size_value,
            "netQuantity": size_value,
            "avgEntryPrice": self._safe_float(payload.get("avg_entry_price")),
            "entryPrice": self._safe_float(payload.get("avg_entry_price")),
            "positionValue": self._safe_float(payload.get("position_value")),
            "unrealizedPnl": self._safe_float(payload.get("unrealized_pnl")),
            "pnlUnrealized": self._safe_float(payload.get("unrealized_pnl")),
            "realizedPnl": self._safe_float(payload.get("realized_pnl")),
            "liquidationPrice": self._safe_float(payload.get("liquidation_price")),
            "marginMode": payload.get("margin_mode"),
            "allocatedMargin": self._safe_float(payload.get("allocated_margin")),
            "market_id": payload.get("market_id"),
        }

    def _normalize_order_record(
        self,
        payload: Dict[str, Any],
        symbol: str,
    ) -> Dict[str, Any]:
        order_id = payload.get("order_id") or payload.get("order_index")
        client_order_index = payload.get("client_order_index") or payload.get("order_index")
        price = self._safe_float(payload.get("price"))
        remaining = self._safe_float(payload.get("remaining_base_amount"))
        initial = self._safe_float(payload.get("initial_base_amount"))

        if remaining is None and initial is not None:
            remaining = max(initial - (self._safe_float(payload.get("filled_base_amount")) or 0.0), 0.0)
        if initial is None:
            initial = remaining

        filled_amount = None
        if initial is not None and remaining is not None:
            filled_amount = max(initial - remaining, 0.0)

        trigger_price = self._safe_float(payload.get("trigger_price"))

        return {
            "id": str(order_id) if order_id is not None else None,
            "orderId": str(order_id) if order_id is not None else None,
            "orderIndex": payload.get("order_index"),
            "clientOrderId": str(client_order_index) if client_order_index is not None else None,
            "clientOrderIndex": client_order_index,
            "symbol": symbol,
            "price": price,
            "quantity": remaining,
            "remainingQuantity": remaining,
            "origQty": initial,
            "executedQty": filled_amount,
            "status": payload.get("status"),
            "side": "Ask" if payload.get("is_ask") else "Bid",
            "type": payload.get("type"),
            "timeInForce": payload.get("time_in_force"),
            "reduceOnly": payload.get("reduce_only"),
            "triggerPrice": trigger_price,
            "orderExpiry": payload.get("order_expiry"),
            "timestamp": payload.get("timestamp"),
            "txHash": payload.get("tx_hash"),
        }

    def _normalize_trade_record(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        trade_id = trade.get("trade_id")
        size = self._safe_float(trade.get("size"))
        price = self._safe_float(trade.get("price"))
        usd_amount = self._safe_float(trade.get("usd_amount"))

        maker_account = self._as_int(
            trade.get("maker_account_index") or trade.get("makerAccountIndex")
        )
        taker_account = self._as_int(
            trade.get("taker_account_index") or trade.get("takerAccountIndex")
        )
        ask_account = self._as_int(
            trade.get("ask_account_id") or trade.get("askAccountId")
        )
        bid_account = self._as_int(
            trade.get("bid_account_id") or trade.get("bidAccountId")
        )
        maker_is_ask = self._as_bool(
            trade.get("is_maker_ask")
            or trade.get("maker_is_ask")
            or trade.get("makerIsAsk")
            or trade.get("maker_side_is_ask")
        )
        taker_is_ask = self._as_bool(
            trade.get("is_taker_ask")
            or trade.get("taker_is_ask")
            or trade.get("takerIsAsk")
            or trade.get("taker_side_is_ask")
        )
        maker_is_buy = self._as_bool(
            trade.get("maker_is_buy") or trade.get("is_maker_buy") or trade.get("makerIsBuy")
        )
        taker_is_buy = self._as_bool(
            trade.get("taker_is_buy") or trade.get("is_taker_buy") or trade.get("takerIsBuy")
        )

        if maker_account is None or taker_account is None:
            if maker_is_ask is True:
                if maker_account is None:
                    maker_account = ask_account
                if taker_account is None:
                    taker_account = bid_account
            elif maker_is_ask is False:
                if maker_account is None:
                    maker_account = bid_account
                if taker_account is None:
                    taker_account = ask_account

        is_maker: Optional[bool] = None
        if self.account_index is not None:
            if maker_account is not None and maker_account == int(self.account_index):
                is_maker = True
            elif taker_account is not None and taker_account == int(self.account_index):
                is_maker = False

        side: Optional[str] = None
        if is_maker is True:
            if maker_is_ask is None and maker_is_buy is not None:
                maker_is_ask = not maker_is_buy
            if maker_is_ask is not None:
                side = "Ask" if maker_is_ask else "Bid"
        elif is_maker is False:
            if taker_is_ask is None and taker_is_buy is not None:
                taker_is_ask = not taker_is_buy
            if taker_is_ask is None and maker_is_ask is not None:
                taker_is_ask = not maker_is_ask
            if taker_is_ask is not None:
                side = "Ask" if taker_is_ask else "Bid"

        if side is None:
            generic_side = trade.get("side") or trade.get("taker_side") or trade.get("maker_side")
            if isinstance(generic_side, str):
                side_upper = generic_side.upper()
                if side_upper in ("BUY", "BID"):
                    side = "Bid"
                elif side_upper in ("SELL", "ASK"):
                    side = "Ask"

        if side is None and maker_is_ask is not None:
            side = "Ask" if maker_is_ask else "Bid"

        if side is None:
            side = "Bid"

        maker_fee_raw = self._safe_float(trade.get("maker_fee"))
        taker_fee_raw = self._safe_float(trade.get("taker_fee"))

        if is_maker is None:
            if maker_fee_raw is not None and (taker_fee_raw is None or abs(maker_fee_raw) >= abs(taker_fee_raw)):
                is_maker = True
            elif taker_fee_raw is not None:
                is_maker = False

        fee_amount: Optional[float] = None
        fee_amount_explicit = False
        for key in (
            "fee",
            "fee_amount",
            "fee_paid",
            "feeValue",
            "fee_value",
            "maker_fee_paid",
            "taker_fee_paid",
        ):
            value = self._safe_float(trade.get(key))
            if value is not None:
                fee_amount = value
                fee_amount_explicit = True
                break

        notional: Optional[float] = None
        if usd_amount is not None:
            notional = abs(usd_amount)
        elif size is not None and price is not None:
            notional = abs(size * price)

        rate_candidate: Optional[float] = None
        if is_maker is True and maker_fee_raw is not None:
            rate_candidate = maker_fee_raw
        elif is_maker is False and taker_fee_raw is not None:
            rate_candidate = taker_fee_raw
        elif maker_fee_raw is not None:
            rate_candidate = maker_fee_raw
        elif taker_fee_raw is not None:
            rate_candidate = taker_fee_raw

        if (
            self._allow_fee_rate_inference
            and rate_candidate is not None
            and notional is not None
            and not fee_amount_explicit
        ):
            rate = rate_candidate
            if abs(rate) > 1:
                rate = rate / 1_000_000.0
            computed_fee = notional * rate
            if (
                fee_amount is None
                or fee_amount == 0.0
                or (notional > 0 and abs(fee_amount) > notional * 0.1)
            ):
                fee_amount = computed_fee

        if fee_amount is None:
            fee_amount = 0.0

        fee_asset = (
            trade.get("fee_asset")
            or trade.get("feeAsset")
            or trade.get("maker_fee_asset")
            or trade.get("taker_fee_asset")
            or trade.get("fee_currency")
            or trade.get("feeCurrency")
        )

        order_identifier = (
            trade.get("order_id")
            or trade.get("orderId")
            or trade.get("order_index")
            or trade.get("orderIndex")
            or trade.get("ask_id")
            or trade.get("bid_id")
        )

        timestamp = trade.get("timestamp") or trade.get("time")

        return {
            "trade_id": str(trade_id) if trade_id is not None else None,
            "order_id": str(order_identifier) if order_identifier is not None else None,
            "market_id": trade.get("market_id"),
            "side": side,
            "size": size,
            "quantity": size,
            "price": price,
            "usd_amount": usd_amount,
            "fee": fee_amount,
            "fee_asset": fee_asset,
            "taker_fee": taker_fee_raw,
            "maker_fee": maker_fee_raw,
            # 如果沒有 fee 項，就用 maker_fee 或 taker_fee 推斷出來的，默認不推算
            "fee_rate_inferred": bool(
                self._allow_fee_rate_inference and not fee_amount_explicit and rate_candidate is not None
            ),
            "is_maker": is_maker,
            "maker_account_index": maker_account,
            "taker_account_index": taker_account,
            "timestamp": timestamp,
            "tx_hash": trade.get("tx_hash"),
        }
