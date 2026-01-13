"""APEX Omni exchange REST client implementation."""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
import sys
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from decimal import Decimal, InvalidOperation, ROUND_UP, ROUND_DOWN
from urllib.parse import urlencode

import requests

# Import zklink_sdk for order signing
try:
    from apexomni.pc import zklink_sdk as zk_sdk
    HAS_ZKLINK = True
except ImportError:
    HAS_ZKLINK = False
    zk_sdk = None

from .base_client import (
    BaseExchangeClient,
    ApiResponse,
    OrderResult,
    OrderInfo,
    BalanceInfo,
    CollateralInfo,
    PositionInfo,
    MarketInfo,
    TickerInfo,
    OrderBookInfo,
    OrderBookLevel,
    KlineInfo,
    TradeInfo,
    CancelResult,
    BatchOrderResult,
    safe_float,
    safe_decimal,
    safe_int,
)
from .proxy_utils import get_proxy_config
from logger import setup_logger

logger = setup_logger("api.apex_client")


class ApexClient(BaseExchangeClient):
    """REST client for the APEX Omni perpetual futures API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.passphrase = config.get("passphrase", "")
        self.zk_seeds = config.get("zk_seeds", "")  # zkKey seeds for order signing
        # APEX Omni base URL
        self.base_url = config.get("base_url", "https://omni.apex.exchange")
        self.timeout = float(config.get("timeout", 10))
        self.max_retries = int(config.get("max_retries", 3))
        self.session = requests.Session()

        # 從環境變量讀取代理配置
        proxies = get_proxy_config()
        if proxies:
            self.session.proxies.update(proxies)
            logger.info(f"APEX 客户端已配置代理: {proxies}")

        self._symbol_cache: Dict[str, str] = {}
        self._market_info_cache: Dict[str, Dict[str, Any]] = {}
        self._cross_symbol_cache: Dict[str, str] = {}  # symbol -> crossSymbolName
        self._account_data: Optional[Dict[str, Any]] = None  # Cached account data
        self._config_data: Optional[Dict[str, Any]] = None  # Cached config data

    def get_exchange_name(self) -> str:
        return "APEX"

    async def connect(self) -> None:
        logger.info("APEX 客户端已連接")

    async def disconnect(self) -> None:
        self.session.close()
        logger.info("APEX 客户端已斷開連接")

    def _current_timestamp(self) -> int:
        return int(time.time() * 1000)

    def _generate_timestamp(self) -> str:
        """Generate timestamp for APEX API.

        Returns milliseconds timestamp as string (e.g., '1763723904275')
        Official SDK uses: int(round(time.time() * 1000))
        """
        return str(int(round(time.time() * 1000)))

    def _normalize_order_fields(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize order fields to standard format."""
        if "id" in order and "order_id" not in order:
            order["order_id"] = order["id"]
        if "orderId" in order and "id" not in order:
            order["id"] = str(order["orderId"])

        side = order.get("side")
        if side:
            normalized = side.upper()
            if normalized == "BUY":
                order["side"] = "Bid"
            elif normalized == "SELL":
                order["side"] = "Ask"

        if "size" in order and "quantity" not in order:
            order["quantity"] = order["size"]

        return order

    def _lookup_key(self, symbol: str) -> str:
        """Generate a case-insensitive lookup key for exchange symbols."""
        return symbol.upper().replace("_", "-")

    def _ensure_symbol_cache(self) -> None:
        """Lazy-load the symbol cache from exchange info."""
        if self._symbol_cache and self._market_info_cache and self._cross_symbol_cache:
            return

        response = self.get_markets()
        if not response.success:
            logger.error("獲取交易對列表失敗: %s", response.error_message)
            self._symbol_cache = {}
            self._market_info_cache = {}
            self._cross_symbol_cache = {}
            return

        # APEX Omni v3 API returns contracts in contractConfig.perpetualContract
        # response.raw 是原始字典數據
        raw_data = response.raw or {}
        data = raw_data.get("data", {}) if isinstance(raw_data, dict) else {}
        contracts: List[Dict[str, Any]] = []

        # v3 structure: data.contractConfig.perpetualContract
        contract_config = data.get("contractConfig", {})
        if contract_config:
            perp_contracts = contract_config.get("perpetualContract", [])
            if perp_contracts:
                contracts.extend(perp_contracts)
                logger.debug(f"從 contractConfig 讀取 {len(perp_contracts)} 個合約")

        cache: Dict[str, str] = {}
        market_cache: Dict[str, Dict[str, Any]] = {}
        cross_cache: Dict[str, str] = {}

        for item in contracts:
            actual_symbol = item.get("symbol")
            if not actual_symbol:
                continue

            # Primary lookup by symbol (e.g., BTC-USDT)
            cache[self._lookup_key(actual_symbol)] = actual_symbol
            market_cache[actual_symbol] = item

            # Also add lookup by crossSymbolName (e.g., BTCUSDT)
            cross_name = item.get("crossSymbolName")
            if cross_name:
                cache[cross_name.upper()] = actual_symbol
                cross_cache[actual_symbol] = cross_name  # Reverse mapping

            # Also add lookup by symbolDisplayName (e.g., BTCUSDT)
            display_name = item.get("symbolDisplayName")
            if display_name and display_name.upper() != cross_name:
                cache[display_name.upper()] = actual_symbol

        self._symbol_cache = cache
        self._market_info_cache = market_cache
        self._cross_symbol_cache = cross_cache

    def _resolve_symbol(self, symbol: Optional[str]) -> Optional[str]:
        """Resolve user provided symbol aliases to APEX native symbols."""
        if not symbol:
            return None

        self._ensure_symbol_cache()
        if not self._symbol_cache:
            return None

        # Try direct lookup first
        sanitized = symbol.strip().upper().replace("_", "-")
        resolved = self._symbol_cache.get(self._lookup_key(sanitized))
        if resolved:
            return resolved

        # Try without separator
        no_sep = sanitized.replace("-", "")
        for key, value in self._symbol_cache.items():
            if key.replace("-", "") == no_sep:
                return value

        return None

    def _decimal_to_str(self, value: Decimal) -> str:
        """Format Decimal without scientific notation and trim trailing zeros."""
        normalized = value.normalize() if value != 0 else Decimal("0")
        text = format(normalized, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    def _find_symbol_suggestions(self, symbol: str, limit: int = 5) -> List[str]:
        """Suggest possible symbols when lookup fails."""
        self._ensure_symbol_cache()
        if not self._market_info_cache:
            return []

        sanitized = symbol.strip().upper().replace("_", "-")
        token = sanitized.replace("-", "")
        candidates: List[str] = []
        seen: Set[str] = set()

        # Fuzzy match by substring
        if token:
            for actual in self._market_info_cache.keys():
                if token in actual.upper().replace("-", "") and actual not in seen:
                    candidates.append(actual)
                    seen.add(actual)
                    if len(candidates) >= limit:
                        return candidates

        return candidates[:limit]

    def _unknown_symbol_error(self, symbol: str) -> Dict[str, Any]:
        suggestions = self._find_symbol_suggestions(symbol)
        message = f"無法解析交易對: {symbol}"
        if suggestions:
            message += f"。可能的交易對: {', '.join(suggestions)}"
            logger.error(message)
            return {"error": message, "status_code": 400, "details": {"candidates": suggestions}}
        logger.error(message)
        return {"error": message, "status_code": 400}

    def _ensure_account_data(self) -> None:
        """Lazy-load and cache account data."""
        if self._account_data:
            return

        result = self.make_request(
            "GET",
            "/api/v3/account",
            instruction=True,
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" not in result:
            self._account_data = result.get("data", {})
        else:
            logger.error("獲取帳戶資料失敗: %s", result.get("error", "unknown"))
            self._account_data = {}

    def _ensure_config_data(self) -> None:
        """Lazy-load and cache config data."""
        if self._config_data:
            return

        # Use /api/v3/symbols endpoint for config data (most reliable)
        response = self.get_markets()
        
        # get_markets 返回 ApiResponse，需要正確處理
        if not response.success:
            logger.error("獲取配置資料失敗: %s", response.error_message or "unknown")
            self._config_data = {}
            return
        
        # 從 ApiResponse.raw 獲取原始數據
        data = response.raw.get("data", {}) if isinstance(response.raw, dict) else {}

        # v3 structure: data.contractConfig.perpetualContract
        perpetual_contracts = []
        assets = []

        contract_config = data.get("contractConfig", {})
        if contract_config:
            perp_contracts = contract_config.get("perpetualContract", [])
            if perp_contracts:
                perpetual_contracts.extend(perp_contracts)
                logger.debug(f"從 contractConfig 讀取 {len(perp_contracts)} 個合約配置")
            config_assets = contract_config.get("assets", [])
            if config_assets:
                assets.extend(config_assets)

        self._config_data = {
            "contractConfig": {
                "perpetualContract": perpetual_contracts,
                "assets": assets,
            },
            "raw": data
        }
        logger.debug(f"配置數據已緩存: {len(perpetual_contracts)} 個合約, {len(assets)} 個資產")

    def _sign_order(self, symbol: str, side: str, size: str, price: str, client_id: str) -> Optional[str]:
        """Sign order using zkLink SDK.

        Returns the zkKey signature for the order.
        """
        if not HAS_ZKLINK or not zk_sdk:
            logger.error("zklink_sdk 未安裝，無法簽名訂單")
            return None

        if not self.zk_seeds:
            logger.error("缺少 zk_seeds，無法簽名訂單。請前往 https://omni.apex.exchange/keyManagement 點擊 'Omni Key' 獲取")
            return None

        self._ensure_account_data()
        self._ensure_config_data()

        if not self._account_data or not self._config_data:
            logger.error("缺少帳戶或配置資料")
            return None

        # Find symbol data
        symbol_data = None
        currency = {}

        contract_config = self._config_data.get("contractConfig", {})
        for contract in contract_config.get("perpetualContract", []):
            if contract.get("symbol") == symbol or contract.get("crossSymbolName") == symbol:
                symbol_data = contract
                break

        if not symbol_data:
            logger.error("找不到交易對配置: %s", symbol)
            return None

        # Find currency data (check both settleAssetId and settleCurrencyId)
        settle_asset = symbol_data.get("settleAssetId") or symbol_data.get("settleCurrencyId")
        for asset in contract_config.get("assets", []):
            if asset.get("token") == settle_asset or asset.get("id") == settle_asset:
                currency = asset
                break

        # Verify we found the currency
        if not currency:
            logger.error(f"找不到結算資產: {settle_asset}")
            return None

        # zkLink uses 18 decimals internally (like Ethereum Wei)
        # This is different from the settlement currency decimals (6 for USDT/USDC)
        decimals = 18

        logger.debug(f"簽名參數: symbol={symbol}, l2PairId={symbol_data.get('l2PairId')}, settle={settle_asset}, decimals={decimals}")

        # Get account info
        account_id = self._account_data.get("id")
        spot_account = self._account_data.get("spotAccount", {})
        contract_account = self._account_data.get("contractAccount", {})

        sub_account_id = spot_account.get("defaultSubAccountId", 1)
        taker_fee_rate = contract_account.get("takerFeeRate", "0.0005")
        maker_fee_rate = contract_account.get("makerFeeRate", "0.0002")

        # Calculate nonce and slot from client_id
        message_hash = hashlib.sha256(client_id.encode()).hexdigest()
        nonce_int = int(message_hash, 16)

        max_uint32 = np.iinfo(np.uint32).max
        max_uint64 = np.iinfo(np.uint64).max

        slot_id = int((nonce_int % max_uint64) / max_uint32)
        nonce = nonce_int % max_uint32
        account_id_int = int(account_id) % max_uint32

        # Convert price and size to contract format
        # Note: decimals was already calculated above
        price_dec = Decimal(price) * Decimal(10) ** decimals
        size_dec = Decimal(size) * Decimal(10) ** decimals

        price_str = str(int(price_dec.quantize(Decimal(1), rounding=ROUND_DOWN)))
        size_str = str(int(size_dec.quantize(Decimal(1), rounding=ROUND_DOWN)))

        # Convert fee rates
        taker_fee_int = int(Decimal(taker_fee_rate) * 10000)
        maker_fee_int = int(Decimal(maker_fee_rate) * 10000)

        # Build and sign the contract
        try:
            builder = zk_sdk.ContractBuilder(
                int(account_id_int),
                int(sub_account_id),
                int(slot_id),
                int(nonce),
                int(symbol_data.get("l2PairId", 0)),
                size_str,
                price_str,
                side.upper() == "BUY",
                taker_fee_int,
                maker_fee_int,
                False  # is_market
            )

            tx = zk_sdk.Contract(builder)
            seeds_bytes = bytes.fromhex(self.zk_seeds.removeprefix("0x"))
            signer = zk_sdk.ZkLinkSigner().new_from_seed(seeds_bytes)
            auth_data = signer.sign_musig(tx.get_bytes())

            return auth_data.signature
        except Exception as e:
            logger.error("zkKey 簽名失敗: %s", e)
            return None

    def _sign_request(self, request_path: str, method: str, timestamp: str, data: Dict[str, Any] = None, include_data: bool = True) -> str:
        """Generate APEX API signature.

        Args:
            request_path: API endpoint path (e.g., /v3/account)
            method: HTTP method (GET, POST, DELETE)
            timestamp: Timestamp string (milliseconds)
            data: Request parameters
            include_data: Whether to include data in signature message (False for GET requests)

        Returns:
            Base64 encoded HMAC-SHA256 signature
        """
        # For POST requests, sort parameters alphabetically and include in signature
        # For GET requests, do NOT include query parameters in signature
        if include_data and data:
            sorted_items = sorted(data.items(), key=lambda x: x[0])
            data_string = '&'.join(
                f'{key}={value}' for key, value in sorted_items if value is not None
            )
        else:
            data_string = ''

        # Build message: timestamp + method + path (+ dataString for POST only)
        message = timestamp + method.upper() + request_path + data_string

        # Create HMAC-SHA256 signature
        # Official SDK: base64.standard_b64encode(secret.encode('utf-8'))
        # The secret is encoded as UTF-8 string, then base64 encoded to get the HMAC key
        hashed = hmac.new(
            base64.standard_b64encode(self.secret_key.encode('utf-8')),
            msg=message.encode('utf-8'),
            digestmod=hashlib.sha256
        )

        signature = base64.standard_b64encode(hashed.digest()).decode()
        return signature

    def _generate_query_path(self, url: str, params: Dict[str, Any]) -> str:
        """Generate URL with query parameters appended.
        
        This mimics the official SDK's generate_query_path function.
        For GET requests, params are appended to the URL and the full path
        (including query string) is used for signing.
        """
        if not params:
            return url
        
        # Filter out None values and build query string
        entries = [(k, v) for k, v in params.items() if v is not None]
        if not entries:
            return url
        
        params_string = '&'.join(f'{k}={v}' for k, v in entries)
        return f'{url}?{params_string}'

    def make_request(
        self,
        method: str,
        endpoint: str,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        instruction: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 3,
    ) -> Dict[str, Any]:
        method_upper = method.upper()
        
        # For GET/DELETE: params go in URL query string
        # For POST: params go in request body as form data
        if params:
            clean_params = {k: v for k, v in params.items() if v is not None}
        else:
            clean_params = {}
            
        if data:
            clean_data = {k: v for k, v in data.items() if v is not None}
        else:
            clean_data = {}

        signed = bool(instruction)
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        if signed:
            if not self.api_key or not self.secret_key:
                return {"error": "缺少 API Key 或 Secret Key"}

            # 生成一次時間戳，用於簽名和 header（毫秒時間戳）
            timestamp = self._generate_timestamp()

            # Official SDK behavior:
            # - GET: params are appended to URL path, sign uses full path with query string, data={}
            # - POST: sign uses endpoint path + sorted data params
            if method_upper in {"GET", "DELETE"}:
                # Build full path with query params for signing
                sign_path = self._generate_query_path(endpoint, clean_params)
                # Sign with empty data (params are in the path)
                signature = self._sign_request(sign_path, method_upper, timestamp, {}, include_data=False)
            else:
                # POST: sign with endpoint and data
                signature = self._sign_request(endpoint, method_upper, timestamp, clean_data, include_data=True)

            headers.update({
                'APEX-SIGNATURE': signature,
                'APEX-TIMESTAMP': timestamp,
                'APEX-API-KEY': self.api_key,
                'APEX-PASSPHRASE': self.passphrase or ''
            })

        retry_total = retry_count or self.max_retries
        url = f"{self.base_url}{endpoint}"

        for attempt in range(retry_total):
            try:
                if method_upper in {"GET", "DELETE"}:
                    response = self.session.request(
                        method_upper,
                        url,
                        params=clean_params,
                        timeout=self.timeout,
                        headers=headers,
                    )
                else:
                    response = self.session.request(
                        method_upper,
                        url,
                        params=None,
                        data=clean_data,
                        timeout=self.timeout,
                        headers=headers,
                    )

                if 200 <= response.status_code < 300:
                    return response.json() if response.text else {}

                if response.status_code == 429:
                    wait_time = min(1 * (2 ** attempt), 8)
                    logger.warning("APEX API 達到速率限制，等待 %.1f 秒後重試", wait_time)
                    time.sleep(wait_time)
                    continue

                try:
                    error_body = response.json()
                    message = error_body.get("msg") or error_body.get("message") or str(error_body)
                except ValueError:
                    message = response.text or f"HTTP {response.status_code}"
                    error_body = {"msg": message}

                if attempt < retry_total - 1 and response.status_code >= 500:
                    time.sleep(1)
                    continue

                return {"error": message, "status_code": response.status_code, "details": error_body}
            except requests.RequestException as exc:
                if attempt < retry_total - 1:
                    logger.warning("APEX API 請求異常 (%s)，重試中...", exc)
                    time.sleep(1)
                    continue
                return {"error": f"請求失敗: {exc}"}

        return {"error": "達到最大重試次數"}

    def get_deposit_address(self, blockchain: str) -> ApiResponse:
        return ApiResponse.error("請使用 APEX 網頁界面獲取充值地址")

    def get_balance(self) -> ApiResponse:
        """獲取帳戶餘額。
        
        Returns:
            ApiResponse with List[BalanceInfo]
        """
        # 獲取 account-balance 的 totalEquityValue（可用保證金）
        balance_result = self.make_request(
            "GET",
            "/api/v3/account-balance",
            instruction=True,
            retry_count=self.max_retries,
        )

        total_equity = 0.0
        if isinstance(balance_result, dict) and "error" not in balance_result:
            data = balance_result.get("data", {})
            total_equity = float(data.get("totalEquityValue", 0))

        # 獲取 account 的 contractWallets（錢包餘額）
        account_result = self.make_request(
            "GET",
            "/api/v3/account",
            instruction=True,
            retry_count=self.max_retries,
        )

        if isinstance(account_result, dict) and "error" in account_result:
            return ApiResponse.error(
                account_result.get("error", "Unknown error"),
                raw=account_result
            )

        data = account_result.get("data", {})
        contract_wallets = data.get("contractWallets", [])
        
        balances: List[BalanceInfo] = []
        
        if contract_wallets:
            wallet = contract_wallets[0]
            wallet_balance = safe_float(wallet.get("balance", 0))
            token = wallet.get("token", "USDC")

            balances.append(BalanceInfo(
                asset=token,
                available=total_equity,
                locked=max(wallet_balance - total_equity, 0.0),
                total=wallet_balance,
                raw=data,
            ))
        else:
            balances.append(BalanceInfo(
                asset="USDC",
                available=total_equity,
                locked=0.0,
                total=total_equity,
                raw=data,
            ))

        return ApiResponse.ok(balances, raw=account_result)

    def get_collateral(self, subaccount_id: Optional[str] = None) -> ApiResponse:
        """獲取抵押品資訊。
        
        Returns:
            ApiResponse with CollateralInfo
        """
        result = self.make_request(
            "GET",
            "/api/v3/account",
            instruction=True,
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)

        data = result.get("data", {})

        # 從 contractWallets 獲取合約錢包餘額
        contract_wallets = data.get("contractWallets", [])
        total_balance = safe_decimal(0)
        token = "USDC"
        if contract_wallets:
            wallet = contract_wallets[0]
            total_balance = safe_decimal(wallet.get("balance", "0"))
            token = wallet.get("token", "USDC")

        # 從 contractAccount 獲取費率信息
        contract_account = data.get("contractAccount", {})

        collateral = CollateralInfo(
            asset=token,
            total_collateral=total_balance,
            free_collateral=total_balance,
            initial_margin=safe_decimal(0),
            maintenance_margin=safe_decimal(0),
            raw=data,
        )

        return ApiResponse.ok(collateral, raw=result)

    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        """執行訂單。
        
        Returns:
            ApiResponse with OrderResult
        """
        symbol = order_details.get("symbol")
        if not symbol:
            return ApiResponse.error("缺少交易對")

        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            message = f"無法解析交易對: {symbol}"
            if suggestions:
                message += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(message)

        side = order_details.get("side")
        if not side:
            return ApiResponse.error("缺少買賣方向")

        if side.lower() in {"bid", "buy"}:
            normalized_side = "BUY"
        elif side.lower() in {"ask", "sell"}:
            normalized_side = "SELL"
        else:
            return ApiResponse.error(f"不支持的方向: {side}")

        order_type = order_details.get("orderType") or order_details.get("type")
        if not order_type:
            return ApiResponse.error("缺少訂單類型")

        # Map order types to APEX format
        type_mapping = {
            "LIMIT": "LIMIT",
            "MARKET": "MARKET",
            "STOP_LIMIT": "STOP_LIMIT",
            "STOP_MARKET": "STOP_MARKET",
        }
        normalized_type = type_mapping.get(order_type.upper(), order_type.upper())

        # Order API expects the original symbol format (e.g., BTC-USDC)
        # NOT the crossSymbolName format (e.g., BTCUSDC)
        payload: Dict[str, Any] = {
            "symbol": resolved_symbol,
            "side": normalized_side,
            "type": normalized_type,
        }

        # Time in force
        post_only = order_details.get("postOnly", False)
        time_in_force = order_details.get("timeInForce")

        if post_only:
            payload["timeInForce"] = "POST_ONLY"
        elif time_in_force:
            tif_mapping = {
                "GTC": "GOOD_TIL_CANCEL",
                "FOK": "FILL_OR_KILL",
                "IOC": "IMMEDIATE_OR_CANCEL",
            }
            payload["timeInForce"] = tif_mapping.get(time_in_force.upper(), time_in_force.upper())
        else:
            payload["timeInForce"] = "GOOD_TIL_CANCEL"

        # Quantity
        quantity = order_details.get("quantity") or order_details.get("size")
        if quantity is not None:
            payload["size"] = str(quantity)

        # Price - 市價單需要特殊處理
        price = order_details.get("price")
        
        # APEX 的市價單實際上也需要價格參數（用於計算 limitFee 和簽名）
        # 如果是市價單且沒有提供價格，則獲取當前市場價格
        if normalized_type == "MARKET" and price is None:
            ticker_response = self.get_ticker(symbol)
            if ticker_response.success and ticker_response.data:
                market_price = ticker_response.data.last_price
                # 市價買單用較高價格，市價賣單用較低價格（確保能成交）
                slippage = 0.01  # 1% 滑點容忍
                if normalized_side == "BUY":
                    price = market_price * (1 + slippage)
                else:
                    price = market_price * (1 - slippage)
                logger.debug("市價單使用參考價格: %.4f (市場價: %.4f)", price, market_price)
            else:
                return ApiResponse.error("無法獲取市場價格，市價單需要價格參考")
        
        # 確保價格符合 tickSize 精度要求
        if price is not None:
            self._ensure_symbol_cache()
            symbol_info = self._market_info_cache.get(resolved_symbol, {})
            tick_size = Decimal(str(symbol_info.get("tickSize", "0.1")))
            price_dec = Decimal(str(price))
            # 根據買賣方向進行四捨五入：買單向上取整，賣單向下取整
            if normalized_side == "BUY":
                # 買單向上取整以確保能成交
                price_dec = (price_dec / tick_size).quantize(Decimal("1"), rounding=ROUND_UP) * tick_size
            else:
                # 賣單向下取整以確保能成交
                price_dec = (price_dec / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size
            price = float(price_dec)
            payload["price"] = str(price_dec)

        # Client order ID - generate if not provided
        client_id = order_details.get("clientId") or str(uuid.uuid4())
        payload["clientId"] = client_id

        # Reduce only
        if order_details.get("reduceOnly"):
            payload["reduceOnly"] = "true"

        # Calculate expiration (28 days from now) - must be in milliseconds
        expiration = int(time.time()) + 3600 * 24 * 28
        payload["expiration"] = expiration * 1000

        # Calculate limit fee
        size_dec = Decimal(str(quantity)) if quantity else Decimal("0")
        price_dec = Decimal(str(price)) if price else Decimal("0")

        self._ensure_account_data()
        taker_fee_rate = Decimal(self._account_data.get("contractAccount", {}).get("takerFeeRate", "0.0005"))

        # Calculate human cost (buy uses ROUND_UP, sell uses ROUND_DOWN)
        if normalized_side == "BUY":
            human_cost = (size_dec * price_dec).quantize(Decimal("0.000001"), rounding=ROUND_UP)
        else:
            human_cost = (size_dec * price_dec).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)

        # Calculate fee
        fee = human_cost * taker_fee_rate

        # IMPORTANT: limitFee must ALWAYS round UP (even for sell orders)
        limit_fee = fee.quantize(Decimal("0.000001"), rounding=ROUND_UP)

        payload["limitFee"] = str(limit_fee)

        # Generate zkKey signature
        zk_signature = self._sign_order(
            resolved_symbol,
            normalized_side,
            str(quantity),
            str(price),
            client_id
        )

        if zk_signature:
            payload["signature"] = zk_signature
        else:
            return ApiResponse.error("無法生成 zkKey 簽名。請前往 https://omni.apex.exchange/keyManagement 點擊 'Omni Key' 獲取 zk_seeds 並添加到配置中")

        result = self.make_request(
            "POST",
            "/api/v3/order",
            instruction=True,
            data=payload,
            retry_count=self.max_retries,
        )

        if isinstance(result, dict):
            # Check for error in response (can be 'error' or 'code' field)
            if "error" in result:
                return ApiResponse.error(result.get("error", "Unknown error"), raw=result)
            if result.get("code") and result.get("code") != 0:
                return ApiResponse.error(
                    result.get("msg", f"Order failed with code {result.get('code')}"),
                    raw=result
                )

        data = result.get("data", result)
        normalized = self._normalize_order_fields(data)
        
        order_result = OrderResult(
            success=True,
            order_id=str(normalized.get("id", normalized.get("orderId", ""))),
            client_order_id=normalized.get("clientId", client_id),
            symbol=resolved_symbol,
            side=normalized.get("side", normalized_side),
            order_type=normalized.get("type", normalized_type),
            price=safe_decimal(normalized.get("price", price)),
            size=safe_decimal(normalized.get("size", quantity)),
            status=normalized.get("status", "NEW"),
            raw=data,
        )
        
        return ApiResponse.ok(order_result, raw=result)

    def get_open_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        """Get all open orders.
        
        Note: APEX API v3/open-orders returns all open orders without symbol filter.
        Client-side filtering is applied if symbol is specified.
        
        Returns:
            ApiResponse with List[OrderInfo]
        """
        # Resolve symbol for client-side filtering if provided
        filter_symbol = None
        if symbol:
            filter_symbol = self._resolve_symbol(symbol)
            if not filter_symbol:
                suggestions = self._find_symbol_suggestions(symbol)
                message = f"無法解析交易對: {symbol}"
                if suggestions:
                    message += f"。可能的交易對: {', '.join(suggestions)}"
                return ApiResponse.error(message)

        result = self.make_request(
            "GET",
            "/api/v3/open-orders",
            instruction=True,
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)

        # Handle different response formats
        # API returns: {"data": [{order1}, {order2}, ...]}
        if isinstance(result, list):
            orders = result
        else:
            data = result.get("data", result)
            if isinstance(data, list):
                orders = data
            else:
                orders = data.get("orders", [])

        order_list: List[OrderInfo] = []
        for item in orders:
            # Client-side filtering by symbol if specified
            if filter_symbol:
                order_symbol = item.get("symbol", "")
                if order_symbol != filter_symbol:
                    continue
            
            normalized = self._normalize_order_fields(dict(item))
            size_val = safe_decimal(normalized.get("size", normalized.get("quantity")))
            filled_val = safe_decimal(normalized.get("filledQty", normalized.get("filledSize", 0)))
            order_info = OrderInfo(
                order_id=str(normalized.get("id", normalized.get("orderId", ""))),
                client_order_id=normalized.get("clientId"),
                symbol=normalized.get("symbol", ""),
                side=normalized.get("side", ""),
                order_type=normalized.get("type", ""),
                price=safe_decimal(normalized.get("price")),
                size=size_val,
                filled_size=filled_val,
                remaining_size=size_val - filled_val if size_val and filled_val else size_val,
                status=normalized.get("status", ""),
                created_at=normalized.get("createdAt"),
                raw=item,
            )
            order_list.append(order_info)
        
        return ApiResponse.ok(order_list, raw=result)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        """Cancel all open orders for a symbol or all symbols.
        
        Args:
            symbol: Optional symbol to cancel orders for (e.g., "BTC-USDC").
                   If None, cancels all open orders.
                   Can also be comma-separated for multiple symbols (e.g., "BTC-USDC,ETH-USDC").
        
        Returns:
            ApiResponse with CancelResult
        """
        data: Dict[str, Any] = {}
        
        if symbol:
            # Handle comma-separated symbols
            if "," in symbol:
                symbols = [s.strip() for s in symbol.split(",")]
                resolved_symbols = []
                for s in symbols:
                    resolved = self._resolve_symbol(s)
                    if not resolved:
                        suggestions = self._find_symbol_suggestions(s)
                        message = f"無法解析交易對: {s}"
                        if suggestions:
                            message += f"。可能的交易對: {', '.join(suggestions)}"
                        return ApiResponse.error(message)
                    resolved_symbols.append(resolved)
                # API expects format: "BTC-USDC,ETH-USDC"
                data["symbol"] = ",".join(resolved_symbols)
            else:
                resolved_symbol = self._resolve_symbol(symbol)
                if not resolved_symbol:
                    suggestions = self._find_symbol_suggestions(symbol)
                    message = f"無法解析交易對: {symbol}"
                    if suggestions:
                        message += f"。可能的交易對: {', '.join(suggestions)}"
                    return ApiResponse.error(message)
                # Use original symbol format (e.g., BTC-USDC), NOT crossSymbolName
                data["symbol"] = resolved_symbol

        result = self.make_request(
            "POST",
            "/api/v3/delete-open-orders",
            instruction=True,
            data=data if data else None,
            retry_count=self.max_retries,
        )
        
        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)
        
        cancel_result = CancelResult(
            success=True,
            order_id=None,
            error_message=None,
            raw=result,
        )
        return ApiResponse.ok(cancel_result, raw=result)

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> ApiResponse:
        """Cancel a single order by order ID.
        
        Args:
            order_id: The order ID to cancel.
            symbol: Optional symbol (not required by API, kept for compatibility).
        
        Returns:
            ApiResponse with CancelResult
        """
        result = self.make_request(
            "POST",
            "/api/v3/delete-order",
            instruction=True,
            data={"id": order_id},
            retry_count=self.max_retries,
        )

        if isinstance(result, dict):
            if "error" in result:
                return ApiResponse.error(result.get("error", "Unknown error"), raw=result)
            if result.get("code") and result.get("code") != 0:
                return ApiResponse.error(
                    result.get("msg", f"Cancel failed with code {result.get('code')}"),
                    raw=result
                )
            # API returns: {"data": "123456"} where data is the order ID
            data = result.get("data", result)
            cancelled_id = data if isinstance(data, str) else str(data.get("id", order_id)) if isinstance(data, dict) else order_id
            
            cancel_result = CancelResult(
                success=True,
                order_id=cancelled_id,
                cancelled_count=1,
                raw=result,
            )
            return ApiResponse.ok(cancel_result, raw=result)

        cancel_result = CancelResult(
            success=True,
            order_id=order_id,
            cancelled_count=1,
            raw=result,
        )
        return ApiResponse.ok(cancel_result, raw=result)

    def get_ticker(self, symbol: str) -> ApiResponse:
        """獲取行情報價。
        
        Returns:
            ApiResponse with TickerInfo
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            message = f"無法解析交易對: {symbol}"
            if suggestions:
                message += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(message)

        # Ticker API requires crossSymbolName format (e.g., BTCUSDT instead of BTC-USDT)
        self._ensure_symbol_cache()
        api_symbol = self._cross_symbol_cache.get(resolved_symbol, resolved_symbol)

        result = self.make_request(
            "GET",
            "/api/v3/ticker",
            params={"symbol": api_symbol},
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)

        data = result.get("data", result)

        # Handle empty data - try to get price from orderbook as fallback
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0]
            else:
                # Fallback: get price from orderbook
                orderbook_response = self.get_order_book(resolved_symbol, limit=1)
                if not orderbook_response.success:
                    return ApiResponse.error(f"無法獲取 {resolved_symbol} 的價格數據，APEX API 返回空數據")

                orderbook = orderbook_response.data
                if orderbook.bids and orderbook.asks:
                    best_bid = float(orderbook.bids[0].price)
                    best_ask = float(orderbook.asks[0].price)
                    mid_price = (best_bid + best_ask) / 2
                    
                    ticker = TickerInfo(
                        symbol=resolved_symbol,
                        last_price=mid_price,
                        raw={"lastPrice": str(mid_price), "symbol": resolved_symbol},
                    )
                    return ApiResponse.ok(ticker, raw=result)
                else:
                    return ApiResponse.error(f"無法獲取 {resolved_symbol} 的價格數據，訂單簿為空")

        # Normalize to standard format
        last_price = safe_float(data.get("lastPrice", data.get("price")))
        
        ticker = TickerInfo(
            symbol=resolved_symbol,
            last_price=last_price,
            bid_price=safe_float(data.get("bestBid")),
            ask_price=safe_float(data.get("bestAsk")),
            volume_24h=safe_float(data.get("volume24h", data.get("volume"))),
            high_24h=safe_float(data.get("high24h", data.get("highPrice"))),
            low_24h=safe_float(data.get("low24h", data.get("lowPrice"))),
            raw=data,
        )

        return ApiResponse.ok(ticker, raw=result)

    def get_markets(self) -> ApiResponse:
        """獲取市場資訊。
        
        Returns:
            ApiResponse with List[MarketInfo]
        """
        result = self.make_request(
            "GET",
            "/api/v3/symbols",
            retry_count=self.max_retries,
        )
        
        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)
        
        data = result.get("data", {})
        contract_config = data.get("contractConfig", {})
        contracts = contract_config.get("perpetualContract", [])
        
        markets: List[MarketInfo] = []
        for item in contracts:
            symbol = item.get("symbol", "")
            parts = symbol.split("-")
            base_asset = parts[0] if len(parts) > 0 else symbol
            quote_asset = parts[1] if len(parts) > 1 else "USDT"
            
            market = MarketInfo(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                market_type="PERP",
                status="TRADING",
                min_order_size=safe_decimal(item.get("minOrderSize", "0.001")),
                tick_size=safe_decimal(item.get("tickSize", "0.1")),
                raw=item,
            )
            markets.append(market)
        
        return ApiResponse.ok(markets, raw=result)

    def get_server_time(self) -> ApiResponse:
        """Get server time to check clock sync.
        
        Returns:
            ApiResponse with server time dict
        """
        result = self.make_request(
            "GET",
            "/api/v3/time",
            retry_count=self.max_retries,
        )
        
        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)
        
        return ApiResponse.ok(result, raw=result)

    def get_order_book(self, symbol: str, limit: int = 20) -> ApiResponse:
        """獲取訂單簿。
        
        Returns:
            ApiResponse with OrderBookInfo
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            message = f"無法解析交易對: {symbol}"
            if suggestions:
                message += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(message)

        # Depth API requires crossSymbolName format (e.g., BTCUSDT instead of BTC-USDT)
        self._ensure_symbol_cache()
        api_symbol = self._cross_symbol_cache.get(resolved_symbol, resolved_symbol)

        result = self.make_request(
            "GET",
            "/api/v3/depth",
            params={"symbol": api_symbol, "limit": limit},
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)

        data = result.get("data", result)
        bids_raw = data.get("b", data.get("bids", []))
        asks_raw = data.get("a", data.get("asks", []))

        # Handle null data from API
        if bids_raw is None:
            bids_raw = []
        if asks_raw is None:
            asks_raw = []

        # Check if orderbook is empty
        if not bids_raw and not asks_raw:
            return ApiResponse.error(
                f"APEX API 返回空訂單簿數據 (symbol: {resolved_symbol})，請確認交易對是否有流動性",
                raw=result
            )

        # Sort bids descending, asks ascending
        try:
            bids_raw = sorted(bids_raw, key=lambda level: float(level[0]), reverse=True)
            asks_raw = sorted(asks_raw, key=lambda level: float(level[0]))
        except (ValueError, TypeError, IndexError):
            pass

        # Convert to OrderBookLevel
        bids = [OrderBookLevel(price=safe_decimal(b[0]), quantity=safe_decimal(b[1])) for b in bids_raw]
        asks = [OrderBookLevel(price=safe_decimal(a[0]), quantity=safe_decimal(a[1])) for a in asks_raw]
        
        orderbook = OrderBookInfo(
            symbol=resolved_symbol,
            bids=bids,
            asks=asks,
            raw=data,
        )
        
        return ApiResponse.ok(orderbook, raw=result)

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100, page: int = 0) -> ApiResponse:
        """獲取成交歷史（已成交的訂單）

        Args:
            symbol: 交易對符號 (如 BTC-USDT)，使用原始格式
            limit: 返回數量限制，預設 100
            page: 頁碼，從 0 開始
        
        Returns:
            ApiResponse with List[TradeInfo]
        """
        params = {"limit": limit, "page": page}
        resolved_symbol = None
        if symbol:
            resolved_symbol = self._resolve_symbol(symbol)
            if not resolved_symbol:
                suggestions = self._find_symbol_suggestions(symbol)
                message = f"無法解析交易對: {symbol}"
                if suggestions:
                    message += f"。可能的交易對: {', '.join(suggestions)}"
                return ApiResponse.error(message)
            # fills API 需要原始格式 (BTC-USDT)，不是 crossSymbolName (BTCUSDT)
            params["symbol"] = resolved_symbol

        result = self.make_request(
            "GET",
            "/api/v3/fills",
            instruction=True,
            params=params,
            retry_count=self.max_retries,
        )
        
        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)
        
        data = result.get("data", result)
        orders = data.get("orders", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        
        trades: List[TradeInfo] = []
        for item in orders:
            trade = TradeInfo(
                trade_id=str(item.get("id", item.get("orderId", ""))),
                order_id=str(item.get("orderId", "")),
                symbol=item.get("symbol", resolved_symbol or ""),
                side=item.get("side", ""),
                price=safe_decimal(item.get("price")),
                size=safe_decimal(item.get("size", item.get("quantity"))),
                fee=safe_decimal(item.get("fee", 0)),
                timestamp=item.get("createdAt"),
                raw=item,
            )
            trades.append(trade)
        
        return ApiResponse.ok(trades, raw=result)

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> ApiResponse:
        """獲取 K 線數據。
        
        Returns:
            ApiResponse with List[KlineInfo]
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            message = f"無法解析交易對: {symbol}"
            if suggestions:
                message += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(message)

        # Klines API requires crossSymbolName format (e.g., BTCUSDT instead of BTC-USDT)
        self._ensure_symbol_cache()
        api_symbol = self._cross_symbol_cache.get(resolved_symbol, resolved_symbol)

        # Map interval to APEX format
        interval_mapping = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "4h": "240",
            "1d": "D",
        }
        apex_interval = interval_mapping.get(interval, interval)

        params = {
            "symbol": api_symbol,
            "interval": apex_interval,
            "limit": limit
        }

        result = self.make_request(
            "GET",
            "/api/v3/klines",
            params=params,
            retry_count=self.max_retries,
        )
        
        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)
        
        data = result.get("data", result)
        klines_raw = data if isinstance(data, list) else []
        
        klines: List[KlineInfo] = []
        for item in klines_raw:
            # APEX kline format: [open_time, open, high, low, close, volume, close_time]
            if isinstance(item, list) and len(item) >= 6:
                kline = KlineInfo(
                    open_time=safe_int(item[0]),
                    open=safe_decimal(item[1]),
                    high=safe_decimal(item[2]),
                    low=safe_decimal(item[3]),
                    close=safe_decimal(item[4]),
                    volume=safe_decimal(item[5]),
                    close_time=safe_int(item[6]) if len(item) > 6 else None,
                    raw=item,
                )
                klines.append(kline)
            elif isinstance(item, dict):
                kline = KlineInfo(
                    open_time=safe_int(item.get("openTime", item.get("t"))),
                    open=safe_decimal(item.get("open", item.get("o"))),
                    high=safe_decimal(item.get("high", item.get("h"))),
                    low=safe_decimal(item.get("low", item.get("l"))),
                    close=safe_decimal(item.get("close", item.get("c"))),
                    volume=safe_decimal(item.get("volume", item.get("v"))),
                    close_time=safe_int(item.get("closeTime")),
                    raw=item,
                )
                klines.append(kline)
        
        return ApiResponse.ok(klines, raw=result)

    def get_market_limits(self, symbol: str) -> ApiResponse:
        """獲取交易對限制資訊。
        
        Returns:
            ApiResponse with MarketInfo
        """
        resolved_symbol = self._resolve_symbol(symbol)
        if not resolved_symbol:
            suggestions = self._find_symbol_suggestions(symbol)
            message = f"無法解析交易對: {symbol}"
            if suggestions:
                message += f"。可能的交易對: {', '.join(suggestions)}"
            return ApiResponse.error(message)

        self._ensure_symbol_cache()
        symbol_info = self._market_info_cache.get(resolved_symbol)
        if not symbol_info:
            logger.error("交易所返回的資料中找不到交易對 %s", resolved_symbol)
            return ApiResponse.error(f"交易所返回的資料中找不到交易對 {resolved_symbol}")

        # Parse symbol (e.g., BTC-USDT -> base=BTC, quote=USDT)
        parts = resolved_symbol.split("-")
        base_asset = parts[0] if len(parts) > 0 else resolved_symbol
        quote_asset = parts[1] if len(parts) > 1 else "USDT"

        # Calculate precision from step size (e.g., "0.001" -> 3 decimal places)
        def get_decimal_places(value_str: str) -> int:
            if '.' in value_str:
                return len(value_str.split('.')[1].rstrip('0')) or 1
            return 0

        step_size = str(symbol_info.get("stepSize", "0.001"))
        tick_size = str(symbol_info.get("tickSize", "0.1"))

        market = MarketInfo(
            symbol=resolved_symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            market_type="PERP",
            status="TRADING",
            min_order_size=safe_decimal(symbol_info.get("minOrderSize", "0.001")),
            tick_size=safe_decimal(tick_size),
            base_precision=get_decimal_places(step_size),
            quote_precision=get_decimal_places(tick_size),
            raw=symbol_info,
        )
        
        return ApiResponse.ok(market)

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取持倉資訊。
        
        Returns:
            ApiResponse with List[PositionInfo]
        """
        result = self.make_request(
            "GET",
            "/api/v3/account",
            instruction=True,
            retry_count=self.max_retries,
        )

        if isinstance(result, dict) and "error" in result:
            return ApiResponse.error(result.get("error", "Unknown error"), raw=result)

        data = result.get("data", {})
        # APEX 使用 "positions" 而不是 "openPositions"
        positions_raw = data.get("positions", data.get("openPositions", []))

        filter_symbol = None
        if symbol:
            filter_symbol = self._resolve_symbol(symbol)

        positions: List[PositionInfo] = []
        for item in positions_raw:
            item_symbol = item.get("symbol", "")

            # Filter by symbol if specified
            if filter_symbol and item_symbol != filter_symbol:
                continue

            raw_size = item.get("size", "0") or "0"
            try:
                pos_dec = Decimal(str(raw_size))
            except (InvalidOperation, TypeError):
                pos_dec = Decimal("0")

            # 跳過 size 為 0 的無效倉位（已平倉的歷史記錄）
            if pos_dec == 0:
                continue

            # APEX 的 side 欄位明確表示方向，size 永遠是正數
            # 必須以 side 欄位為準來判斷多空方向
            side_str = item.get("side", "").upper()
            if side_str == "LONG":
                mapped_side = "LONG"
            elif side_str == "SHORT":
                mapped_side = "SHORT"
            else:
                # 如果 side 欄位為空，則根據 size 正負判斷（通常不會發生）
                mapped_side = "FLAT"

            long_dec = abs(pos_dec) if mapped_side == "LONG" else Decimal("0")
            short_dec = abs(pos_dec) if mapped_side == "SHORT" else Decimal("0")

            entry_price = item.get("entryPrice")
            unrealized = item.get("unrealizedPnl", item.get("unrealizedProfit"))

            # netQuantity: 多頭為正，空頭為負
            net_qty = abs(pos_dec) if mapped_side == "LONG" else -abs(pos_dec)

            position = PositionInfo(
                symbol=item_symbol,
                side=mapped_side,
                size=abs(pos_dec),
                entry_price=safe_decimal(entry_price),
                unrealized_pnl=safe_decimal(unrealized),
                leverage=safe_decimal(item.get("leverage")),
                raw=item,
            )
            positions.append(position)

        return ApiResponse.ok(positions, raw=result)
