"""Paradex exchange REST client implementation with JWT authentication."""
from __future__ import annotations

import time
import base64
import binascii
import json
from typing import Any, Dict, List, Optional
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta

import requests
from starknet_py.net.signer.stark_curve_signer import StarkCurveSigner, KeyPair
from starknet_py.utils.typed_data import TypedData

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
    TradeInfo,
    CancelResult,
    BatchOrderResult,
    KlineInfo,
)
from .proxy_utils import get_proxy_config
from logger import setup_logger

logger = setup_logger("api.paradex_client")


class ParadexClient(BaseExchangeClient):
    """REST client for Paradex perpetual futures exchange with JWT authentication."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Paradex 使用 StarkNet 認證，不需要傳統的 API Key
        raw_private_key = config.get("private_key") or config.get("secret_key")
        self.private_key = raw_private_key.strip() if isinstance(raw_private_key, str) else raw_private_key  # StarkNet 私鑰
        raw_account = config.get("account_address")
        self.account_address = raw_account.strip() if isinstance(raw_account, str) else raw_account  # StarkNet 賬户地址

        if isinstance(self.private_key, str) and self.private_key.startswith("0X"):
            self.private_key = "0x" + self.private_key[2:]
        if isinstance(self.private_key, str):
            self.private_key = self.private_key.lower()

        if isinstance(self.account_address, str) and self.account_address.startswith("0X"):
            self.account_address = "0x" + self.account_address[2:]
        if isinstance(self.account_address, str):
            self.account_address = self.account_address.lower()
        self.base_url = config.get("base_url", "https://api.prod.paradex.trade/v1")
        self.timeout = float(config.get("timeout", 30))
        self.max_retries = int(config.get("max_retries", 3))
        self.session = requests.Session()

        # 從環境變量讀取代理配置
        proxies = get_proxy_config()
        if proxies:
            self.session.proxies.update(proxies)
            logger.info(f"Paradex 客户端已配置代理: {proxies}")

        # JWT 相關
        self._jwt_token: Optional[str] = None
        self._jwt_expiry: Optional[datetime] = None
        self._jwt_refresh_buffer = int(config.get("jwt_refresh_buffer", 120))  # 默認提前120秒刷新

        # 訂單簽名相關 - 創建 KeyPair 用於訂單簽名
        private_key_int = int(self.private_key, 16) if isinstance(self.private_key, str) else self.private_key
        self._key_pair = KeyPair.from_private_key(private_key_int)
        self._account_address_int = int(self.account_address, 16) if isinstance(self.account_address, str) else self.account_address

        # 簽名與時間同步設置
        signature_ttl_seconds_cfg = config.get("signature_ttl_seconds")
        if signature_ttl_seconds_cfg is None and config.get("signature_ttl_ms") is not None:
            # 兼容舊配置（毫秒）
            signature_ttl_seconds_cfg = int(int(config["signature_ttl_ms"]) / 1000)
        self._signature_ttl_seconds = int(signature_ttl_seconds_cfg or 30 * 60)  # 默認30分鐘
        self._time_offset_sec: float = 0.0
        self._last_time_sync: Optional[float] = None
        self._time_sync_interval = int(config.get("time_sync_interval", 300))  # 默認每5分鐘同步一次

        # 緩存
        self._market_info_cache: Dict[str, Dict[str, Any]] = {}
        self._last_market_fetch: Optional[float] = None
        self._market_cache_ttl = 300  # 市場信息緩存5分鐘

        # 系統配置緩存
        self._system_config: Optional[Dict[str, Any]] = None
        self._chain_id: Optional[int] = None

    def get_exchange_name(self) -> str:
        return "Paradex"

    async def connect(self) -> None:
        """初始化連接，生成首次 JWT token"""
        try:
            self._refresh_jwt_token()
            logger.info("Paradex 客户端已連接，JWT token 已生成")
        except Exception as e:
            logger.error(f"Paradex 連接失敗: {e}")
            raise

    async def disconnect(self) -> None:
        self.session.close()
        self._jwt_token = None
        self._jwt_expiry = None
        logger.info("Paradex 客户端已斷開連接")

    def _sync_server_time(self, force: bool = False) -> None:
        """與 Paradex 服務器同步時間，避免本地時鐘漂移導致的認證錯誤"""
        now = time.time()

        if not force and self._last_time_sync and now - self._last_time_sync < self._time_sync_interval:
            return

        url = f"{self.base_url}/system/time"

        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"同步 Paradex 服務器時間失敗: HTTP {response.status_code} - {response.text}")
                return

            data = response.json()
            server_time_raw = data.get("server_time")
            if server_time_raw is None:
                logger.warning(f"同步 Paradex 服務器時間失敗，返回內容缺少 server_time: {data}")
                return

            server_time_value = float(server_time_raw)
            if server_time_value > 1e12:
                # 如果返回毫秒，轉換為秒
                server_time_value /= 1000.0
            local_time_value = time.time()
            self._time_offset_sec = server_time_value - local_time_value
            self._last_time_sync = now
            logger.debug(f"已同步 Paradex 服務器時間，當前偏移: {self._time_offset_sec:.3f} 秒")
        except (ValueError, requests.RequestException) as exc:
            logger.warning(f"同步 Paradex 服務器時間時出錯: {exc}")

    def _current_timestamp(self, force_sync: bool = False) -> int:
        """獲取與服務器同步後的當前時間戳（秒）"""
        self._sync_server_time(force=force_sync)
        return int(time.time() + self._time_offset_sec)

    def _load_system_config(self) -> Dict[str, Any]:
        if self._system_config is not None:
            return self._system_config

        url = f"{self.base_url}/system/config"
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                raise ValueError(
                    f"無法獲取 Paradex 系統配置: HTTP {response.status_code} - {response.text}"
                )
            self._system_config = response.json()
        except requests.RequestException as exc:
            raise ValueError(f"獲取 Paradex 系統配置失敗: {exc}")

        return self._system_config

    def _get_chain_id(self) -> int:
        if self._chain_id is not None:
            return self._chain_id

        config = self._load_system_config()
        chain_id_text = config.get("starknet_chain_id")
        if not chain_id_text:
            raise ValueError("系統配置未返回 starknet_chain_id")

        self._chain_id = int.from_bytes(chain_id_text.encode(), "big")
        return self._chain_id

    def _flatten_signature(self, sig: List[int]) -> str:
        """將簽名格式化為 Paradex 要求的格式 ["r","s"]"""
        return f'["{sig[0]}","{sig[1]}"]'

    def _encode_short_string(self, value: str) -> str:
        """將字符串編碼為 felt (short string)"""
        encoded = value.encode("ascii")
        if len(encoded) > 31:
            raise ValueError("短字符串超出 31 字節限制")
        return str(int.from_bytes(encoded, "big"))

    def _to_chain_amount(self, amount: float, decimals: int = 8) -> str:
        """將金額轉換為鏈上格式（帶固定小數位）
        
        使用 Decimal 避免浮點數精度問題
        """
        # Paradex 使用 8 位小數的整數表示
        try:
            amount_decimal = Decimal(str(amount))
            multiplier = Decimal(10 ** decimals)
            chain_amount = int(amount_decimal * multiplier)
            return str(chain_amount)
        except (InvalidOperation, ValueError) as e:
            logger.error(f"金額轉換失敗: {amount}, 錯誤: {e}")
            # 回退到舊方法
            multiplier = 10 ** decimals
            return str(int(float(amount) * multiplier))

    def _build_order_message(self, order_data: Dict[str, Any], signature_timestamp: int) -> Dict[str, Any]:
        """構建訂單簽名消息（TypedData 格式）

        參考 Paradex SDK 的實現：
        https://github.com/tradeparadex/paradex-py/blob/main/paradex_py/message/order.py
        """
        chain_id = self._get_chain_id()

        # 轉換訂單方向：BUY -> "1", SELL -> "2"
        side = order_data["side"].upper()
        chain_side = "1" if side == "BUY" else "2"

        # 編碼市場名稱為 short string
        market_encoded = self._encode_short_string(order_data["market"])

        # 編碼訂單類型為 short string
        order_type_encoded = self._encode_short_string(order_data["type"].upper())

        # 轉換金額為鏈上格式（8位小數）
        size_chain = self._to_chain_amount(order_data["size"], 8)

        # 價格：限價單使用實際價格，市價單使用 0
        if order_data["type"].upper() == "LIMIT" and "price" in order_data:
            price_chain = self._to_chain_amount(order_data["price"], 8)
        else:
            price_chain = "0"

        typed_data = {
            "domain": {
                "name": "Paradex",
                "chainId": hex(chain_id),
                "version": "1"
            },
            "primaryType": "Order",
            "types": {
                "StarkNetDomain": [
                    {"name": "name", "type": "felt"},
                    {"name": "chainId", "type": "felt"},
                    {"name": "version", "type": "felt"},
                ],
                "Order": [
                    {"name": "timestamp", "type": "felt"},
                    {"name": "market", "type": "felt"},
                    {"name": "side", "type": "felt"},
                    {"name": "orderType", "type": "felt"},
                    {"name": "size", "type": "felt"},
                    {"name": "price", "type": "felt"},
                ],
            },
            "message": {
                "timestamp": str(signature_timestamp),
                "market": market_encoded,
                "side": chain_side,
                "orderType": order_type_encoded,
                "size": size_chain,
                "price": price_chain,
            },
        }

        return typed_data

    def _sign_order(self, order_data: Dict[str, Any], signature_timestamp: int) -> str:
        """為訂單生成 StarkNet 簽名

        返回格式化的簽名字符串：["r","s"]
        """
        # 構建 TypedData 消息
        typed_data_dict = self._build_order_message(order_data, signature_timestamp)

        # 使用 starknet-py 進行簽名
        typed_data = TypedData.from_dict(typed_data_dict)

        # 創建簽名器
        signer = StarkCurveSigner(
            account_address=hex(self._account_address_int),
            key_pair=self._key_pair,
            chain_id=self._get_chain_id()
        )

        # 簽名消息
        signature_list = signer.sign_message(typed_data, self._account_address_int)

        # 格式化簽名為 Paradex 要求的格式
        return self._flatten_signature(signature_list)

    def _decode_jwt_expiry(self, token: str) -> Optional[datetime]:
        """從 JWT 中解析過期時間（UTC）"""
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None
            payload_part = parts[1]
            padding = "=" * (-len(payload_part) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_part + padding)
            payload = json.loads(payload_bytes.decode())
            exp = payload.get("exp")
            if isinstance(exp, (int, float)):
                return datetime.utcfromtimestamp(exp)
        except (ValueError, json.JSONDecodeError, binascii.Error) as exc:
            logger.warning(f"解析 JWT 到期時間失敗: {exc}")
        return None

    def _generate_jwt_token(self) -> str:
        """生成 JWT token 用於 Paradex API 認證

        Paradex 使用 StarkNet 簽名來生成 JWT token
        注意：這需要 StarkNet 賬户地址和私鑰
        """
        if not self.private_key:
            raise ValueError("需要提供 private_key（StarkNet 私鑰）以生成 JWT token")

        if not self.account_address:
            raise ValueError("需要提供 account_address（StarkNet 賬户地址）以生成 JWT token")

        # 構建認證消息
        now = self._current_timestamp(force_sync=True)
        signature_ttl_seconds = max(self._signature_ttl_seconds, 60)  # 最少保持 1 分鐘有效期
        expiry = now + signature_ttl_seconds

        # 獲取 chain_id
        chain_id_value = self._get_chain_id()

        # 構建 TypedData (遵循 Paradex 的格式)
        typed_data_dict = {
            "types": {
                "StarkNetDomain": [
                    {"name": "name", "type": "felt"},
                    {"name": "chainId", "type": "felt"},
                    {"name": "version", "type": "felt"},
                ],
                "Request": [
                    {"name": "method", "type": "felt"},
                    {"name": "path", "type": "felt"},
                    {"name": "body", "type": "felt"},
                    {"name": "timestamp", "type": "felt"},
                    {"name": "expiration", "type": "felt"},
                ],
            },
            "primaryType": "Request",
            "domain": {
                "name": "Paradex",
                "chainId": hex(chain_id_value),
                "version": "1"
            },
            "message": {
                "method": "POST",
                "path": "/v1/auth",
                "body": "",
                "timestamp": now,
                "expiration": expiry,
            },
        }

        try:
            # 使用 starknet-py 庫進行簽名
            typed_data = TypedData.from_dict(typed_data_dict)

            # 創建簽名器
            private_key_int = int(self.private_key, 16) if isinstance(self.private_key, str) else self.private_key
            account_address_int = int(self.account_address, 16) if isinstance(self.account_address, str) else self.account_address

            key_pair = KeyPair.from_private_key(private_key_int)
            signer = StarkCurveSigner(
                account_address=hex(account_address_int),
                key_pair=key_pair,
                chain_id=chain_id_value
            )

            # 使用 StarkCurveSigner 簽名 TypedData
            signature_list = signer.sign_message(typed_data, account_address_int)

            # 格式化簽名為 Paradex 要求的格式 [r, s]
            # signature_list 應該是 [r, s] 的列表
            signature_header = f'["{signature_list[0]}","{signature_list[1]}"]'

            logger.debug(f"簽名生成成功: r={signature_list[0]}, s={signature_list[1]}")

        except Exception as exc:
            import traceback
            logger.error(f"簽名生成異常詳情: {traceback.format_exc()}")
            raise ValueError(f"生成 StarkNet 簽名失敗: {exc}") from exc

        # 構建請求頭部（Paradex 需要特定的頭部）
        headers = {
            "PARADEX-STARKNET-ACCOUNT": self.account_address,
            "PARADEX-STARKNET-SIGNATURE": signature_header,
            "PARADEX-TIMESTAMP": str(now),
            "PARADEX-SIGNATURE-EXPIRATION": str(expiry),
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        # 發送請求獲取 JWT token
        try:
            url = f"{self.base_url.replace('/v1', '')}/v1/auth"
            response = self.session.post(
                url,
                headers=headers,
                timeout=self.timeout
            )

            logger.info(f"JWT 認證響應狀態: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                token = result.get("jwt_token") or result.get("token")
                if token:
                    self._jwt_token = token
                    jwt_expiry = self._decode_jwt_expiry(token)
                    if jwt_expiry:
                        self._jwt_expiry = jwt_expiry
                    else:
                        # 如果無法解析，保守地設置 5 分鐘有效期
                        self._jwt_expiry = datetime.utcnow() + timedelta(minutes=5)
                    logger.info(f"JWT token 生成成功，有效期至 (UTC): {self._jwt_expiry}")
                    return token
                else:
                    raise ValueError(f"JWT token 響應中未找到 token: {result}")
            else:
                error_msg = f"JWT 認證失敗: HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                    logger.info(f"result: {error_detail}")
                except:
                    error_msg += f" - {response.text}"
                raise ValueError(error_msg)

        except requests.RequestException as e:
            raise ValueError(f"JWT 認證請求失敗: {e}")

    def _refresh_jwt_token(self) -> None:
        """刷新 JWT token（如果需要）"""
        now = datetime.utcnow()

        # 檢查是否需要刷新
        if self._jwt_token and self._jwt_expiry:
            time_until_expiry = (self._jwt_expiry - now).total_seconds()
            if time_until_expiry > self._jwt_refresh_buffer:
                # token 仍然有效，無需刷新
                return

        # 生成新的 token
        logger.info("刷新 JWT token...")
        self._generate_jwt_token()

    def _ensure_jwt_valid(self) -> None:
        """確保 JWT token 有效，如需要則自動刷新"""
        self._refresh_jwt_token()

        if not self._jwt_token:
            raise ValueError("JWT token 未初始化")

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
        """執行 HTTP 請求，自動處理 JWT 認證"""

        # 如果需要認證，確保 JWT 有效
        requires_auth = instruction is not None
        if requires_auth:
            try:
                self._ensure_jwt_valid()
            except Exception as e:
                return {"error": f"JWT 認證失敗: {e}"}

        url = f"{self.base_url}{endpoint}"
        headers = {}

        # 添加 JWT token
        if requires_auth and self._jwt_token:
            headers["Authorization"] = f"Bearer {self._jwt_token}"

        headers["Content-Type"] = "application/json"

        method_upper = method.upper()
        retry_total = retry_count or self.max_retries

        for attempt in range(retry_total):
            try:
                if method_upper == "GET":
                    response = self.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                elif method_upper == "POST":
                    response = self.session.post(
                        url,
                        json=data,
                        headers=headers,
                        timeout=self.timeout
                    )
                elif method_upper == "DELETE":
                    response = self.session.delete(
                        url,
                        json=data,
                        headers=headers,
                        timeout=self.timeout
                    )
                else:
                    return {"error": f"不支持的 HTTP 方法: {method}"}

                # 檢查響應
                if 200 <= response.status_code < 300:
                    return response.json() if response.text else {}

                # 處理速率限制
                if response.status_code == 429:
                    wait_time = min(2 ** attempt, 8)
                    logger.warning(f"Paradex API 達到速率限制，等待 {wait_time} 秒後重試")
                    time.sleep(wait_time)
                    continue

                # 處理認證失敗（可能需要刷新 token）
                if response.status_code == 401 and attempt < retry_total - 1:
                    logger.warning("JWT token 可能已過期，嘗試刷新...")
                    self._jwt_token = None  # 強制刷新
                    self._ensure_jwt_valid()
                    headers["Authorization"] = f"Bearer {self._jwt_token}"
                    continue

                # 解析錯誤信息
                try:
                    error_body = response.json()
                    message = error_body.get("message") or str(error_body)
                except ValueError:
                    message = response.text or f"HTTP {response.status_code}"
                    error_body = {"message": message}

                # 服務器錯誤重試
                if attempt < retry_total - 1 and response.status_code >= 500:
                    time.sleep(1)
                    continue

                return {
                    "error": message,
                    "status_code": response.status_code,
                    "details": error_body
                }

            except requests.RequestException as exc:
                if attempt < retry_total - 1:
                    logger.warning(f"Paradex API 請求異常 ({exc})，重試中...")
                    time.sleep(1)
                    continue
                return {"error": f"請求失敗: {exc}"}

        return {"error": "達到最大重試次數"}

    def _fetch_markets_if_needed(self) -> None:
        """按需獲取市場信息並緩存"""
        now = time.time()
        if (self._market_info_cache and self._last_market_fetch and
            now - self._last_market_fetch < self._market_cache_ttl):
            return

        result = self.make_request("GET", "/markets")
        if isinstance(result, dict) and "error" not in result:
            markets = result.get("results", [])
            for market in markets:
                symbol = market.get("symbol")
                if symbol:
                    self._market_info_cache[symbol] = market
            self._last_market_fetch = now
            logger.info(f"已緩存 {len(self._market_info_cache)} 個市場信息")

    def get_balance(self) -> ApiResponse:
        """獲取賬户餘額
        
        Paradex API 返回格式:
        {
            "results": [
                {
                    "token": "USDC",
                    "size": "1234.56",
                    "last_updated_at": 1681462770114
                }
            ]
        }
        """
        result = self.make_request(
            "GET",
            "/balance",
            instruction=True,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        # 轉換為標準格式（與其他客户端一致）
        balances: List[BalanceInfo] = []
        balance_list = result.get("results", []) if isinstance(result, dict) else []

        for item in balance_list:
            if isinstance(item, dict):
                token = item.get("token", "UNKNOWN")
                size = self.safe_float(item.get("size", "0")) or 0.0
                
                # Paradex 的 balance 端點只返回總額，沒有區分 available 和 locked
                balances.append(BalanceInfo(
                    asset=token,
                    available=size,
                    locked=0.0,
                    total=size,
                    raw=item,
                ))

        return ApiResponse.ok(balances, raw=result)

    def get_collateral(self) -> ApiResponse:
        """獲取賬户抵押品信息（通過 account summary 端點）
        
        Paradex API 返回格式:
        {
            "account": "0x...",
            "account_value": "136285.06918911",
            "free_collateral": "73276.47229774",
            "initial_margin_requirement": "63008.59689218",
            "maintenance_margin_requirement": "31504.29844641",
            "total_collateral": "123003.62047353",
            "updated_at": 1681471234972
        }
        """
        result = self.make_request(
            "GET",
            "/account",
            instruction=True,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        collateral_info = CollateralInfo(
            asset="USDC",
            total_collateral=self.safe_float(result.get("total_collateral")),
            free_collateral=self.safe_float(result.get("free_collateral")),
            account_value=self.safe_float(result.get("account_value")),
            initial_margin=self.safe_float(result.get("initial_margin_requirement")),
            maintenance_margin=self.safe_float(result.get("maintenance_margin_requirement")),
            raw=result,
        )
        return ApiResponse.ok(collateral_info, raw=result)

    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        """執行訂單

        Paradex 需要對每個訂單進行 StarkNet 簽名
        """
        symbol = order_details.get("symbol")
        if not symbol:
            return ApiResponse.error("缺少交易對")

        side = order_details.get("side", "").upper()
        if side not in ["BUY", "SELL", "BID", "ASK"]:
            return ApiResponse.error("無效的買賣方向")

        # 標準化方向
        if side in ["BID"]:
            side = "BUY"
        elif side in ["ASK"]:
            side = "SELL"

        order_type = order_details.get("orderType") or order_details.get("type", "LIMIT")
        size = order_details.get("quantity") or order_details.get("size")

        # 生成簽名時間戳（毫秒）
        signature_timestamp = int(time.time() * 1000)

        # 標準化數值（確保簽名和發送使用相同的值）
        normalized_size = float(size)
        normalized_price = None

        # 限價單需要價格
        if order_type.upper() == "LIMIT":
            price = order_details.get("price")
            if price:
                # 使用 Decimal 確保精度一致，避免浮點數精度問題
                try:
                    price_decimal = Decimal(str(price))
                    # 標準化為最多 8 位小數（Paradex 使用 8 位精度）
                    normalized_price = float(price_decimal.quantize(Decimal('0.00000001')))
                except (InvalidOperation, ValueError) as e:
                    logger.error(f"無效的價格格式: {price}, 錯誤: {e}")
                    return ApiResponse.error(f"無效的價格: {price}")
            else:
                return ApiResponse.error("限價單缺少價格")

        # 構建訂單數據用於簽名
        order_data_for_signature = {
            "market": symbol,
            "side": side,
            "type": order_type.upper(),
            "size": normalized_size,
        }

        if normalized_price is not None:
            order_data_for_signature["price"] = normalized_price

        # 生成訂單簽名
        try:
            signature = self._sign_order(order_data_for_signature, signature_timestamp)
        except Exception as e:
            logger.error(f"訂單簽名失敗: {e}")
            import traceback
            logger.error(f"簽名異常詳情: {traceback.format_exc()}")
            return ApiResponse.error(f"訂單簽名失敗: {e}")

        # 構建 API 請求 payload（使用與簽名相同的標準化值）
        payload = {
            "market": symbol,
            "side": side,
            "type": order_type.upper(),
            "size": str(normalized_size),
            "signature": signature,
            "signature_timestamp": signature_timestamp,
        }

        # 限價單價格 - 格式化為 8 位小數字符串，保留尾隨零
        if normalized_price is not None:
            payload["price"] = f"{normalized_price:.8f}"

        # 處理 instruction（時間有效性）- 與批量下單邏輯一致
        time_in_force = order_details.get("timeInForce", "GTC").upper()
        post_only = order_details.get("postOnly", False)

        if post_only:
            payload["instruction"] = "POST_ONLY"
        elif time_in_force == "IOC":
            payload["instruction"] = "IOC"
        else:
            payload["instruction"] = "GTC"

        # 可選參數
        if "clientId" in order_details:
            payload["client_order_id"] = order_details["clientId"]

        # 處理 reduceOnly 標誌
        flags = []
        if order_details.get("reduceOnly"):
            flags.append("REDUCE_ONLY")
        if flags:
            payload["flags"] = flags

        result = self.make_request(
            "POST",
            "/orders",
            instruction=True,
            data=payload,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        # 轉換為標準格式
        order_result = OrderResult(
            success=True,
            order_id=result.get("id") or result.get("order_id"),
            client_order_id=result.get("client_order_id"),
            symbol=symbol,
            side=side,
            price=normalized_price,
            size=normalized_size,
            status=result.get("status"),
            raw=result,
        )
        return ApiResponse.ok(order_result, raw=result)

    def execute_order_batch(self, orders_details: List[Dict[str, Any]]) -> ApiResponse:
        """批量執行訂單

        Paradex 批量下單限制：每批最多 10 個訂單

        Args:
            orders_details: 訂單詳情列表

        Returns:
            ApiResponse with BatchOrderResult
        """
        if not orders_details:
            return ApiResponse.error("訂單列表為空")

        # Paradex 限制每批最多 10 個訂單
        if len(orders_details) > 10:
            logger.warning("Paradex 批量下單限制為 10 個訂單，當前 %d 個，將拆分為多批", len(orders_details))

        # 將訂單拆分為多批（每批最多 10 個）
        batch_size = 10
        all_results: List[OrderResult] = []
        all_errors: List[str] = []

        for batch_start in range(0, len(orders_details), batch_size):
            batch = orders_details[batch_start:batch_start + batch_size]

            # 生成簽名時間戳（毫秒）- 批量訂單中所有訂單使用相同的時間戳
            signature_timestamp = int(time.time() * 1000)

            # 構建批量訂單請求
            batch_orders = []

            for order_details in batch:
                symbol = order_details.get("symbol")
                if not symbol:
                    logger.warning("跳過無效訂單: 缺少交易對")
                    all_errors.append("缺少交易對")
                    continue

                side = order_details.get("side", "").upper()
                if side not in ["BUY", "SELL", "BID", "ASK"]:
                    logger.warning("跳過無效訂單: 無效的買賣方向 %s", side)
                    all_errors.append(f"無效的買賣方向: {side}")
                    continue

                # 標準化方向
                if side == "BID":
                    side = "BUY"
                elif side == "ASK":
                    side = "SELL"

                order_type = order_details.get("orderType") or order_details.get("type", "LIMIT")
                size = order_details.get("quantity") or order_details.get("size")

                if not size:
                    logger.warning("跳過無效訂單: 缺少數量")
                    all_errors.append("缺少數量")
                    continue

                # 標準化數值（確保簽名和發送使用相同的值）
                normalized_size = float(size)
                normalized_price = None

                # 限價單需要價格
                if order_type.upper() == "LIMIT":
                    price = order_details.get("price")
                    if price:
                        # 使用 Decimal 確保精度一致，避免浮點數精度問題
                        try:
                            price_decimal = Decimal(str(price))
                            # 標準化為最多 8 位小數（Paradex 使用 8 位精度）
                            normalized_price = float(price_decimal.quantize(Decimal('0.00000001')))
                        except (InvalidOperation, ValueError) as e:
                            logger.warning("跳過無效訂單: 無效的價格格式 %s, 錯誤: %s", price, e)
                            all_errors.append(f"無效的價格: {price}")
                            continue
                    else:
                        logger.warning("跳過無效訂單: 限價單缺少價格")
                        all_errors.append("限價單缺少價格")
                        continue

                # 構建訂單數據用於簽名
                order_data_for_signature = {
                    "market": symbol,
                    "side": side,
                    "type": order_type.upper(),
                    "size": normalized_size,
                }

                if normalized_price is not None:
                    order_data_for_signature["price"] = normalized_price

                # 生成訂單簽名
                try:
                    signature = self._sign_order(order_data_for_signature, signature_timestamp)
                except Exception as e:
                    logger.error("訂單簽名失敗: %s", e)
                    all_errors.append(f"訂單簽名失敗: {e}")
                    continue

                # 構建單個訂單 payload（使用與簽名相同的標準化值）
                order_payload = {
                    "market": symbol,
                    "side": side,
                    "type": order_type.upper(),
                    "size": str(normalized_size),
                    "signature": signature,
                    "signature_timestamp": signature_timestamp,
                }

                # 添加價格（限價單）- 格式化為 8 位小數字符串，保留尾隨零
                if normalized_price is not None:
                    order_payload["price"] = f"{normalized_price:.8f}"

                # 處理 instruction（時間有效性）
                time_in_force = order_details.get("timeInForce", "GTC").upper()
                post_only = order_details.get("postOnly", False)

                if post_only:
                    order_payload["instruction"] = "POST_ONLY"
                elif time_in_force == "IOC":
                    order_payload["instruction"] = "IOC"
                else:
                    order_payload["instruction"] = "GTC"

                # 可選參數
                if "clientId" in order_details:
                    order_payload["client_id"] = order_details["clientId"]

                # 處理 reduceOnly 標誌
                flags = []
                if order_details.get("reduceOnly"):
                    flags.append("REDUCE_ONLY")
                if flags:
                    order_payload["flags"] = flags

                batch_orders.append(order_payload)

            # 如果這批沒有有效訂單，跳過
            if not batch_orders:
                continue

            # 發送批量請求
            logger.info("發送批量訂單請求: %d 個訂單", len(batch_orders))

            # Paradex API 期望直接接收訂單數組，而不是包裝在對象中
            result = self.make_request(
                "POST",
                "/orders/batch",
                instruction=True,
                data=batch_orders,  # 直接發送數組
                retry_count=self.max_retries
            )

            if isinstance(result, dict):
                # 處理成功的訂單
                if "orders" in result:
                    for order in result["orders"]:
                        all_results.append(OrderResult(
                            success=True,
                            order_id=order.get("id") or order.get("order_id"),
                            client_order_id=order.get("client_order_id"),
                            symbol=order.get("market"),
                            side=order.get("side"),
                            price=self.safe_float(order.get("price")),
                            size=self.safe_float(order.get("size")),
                            status=order.get("status"),
                            raw=order,
                        ))
                    logger.info("批量下單成功: %d 個訂單", len(result["orders"]))

                # 處理失敗的訂單（過濾掉 None 值）
                if "errors" in result and result["errors"]:
                    # Paradex API 返回的 errors 數組中，成功的訂單對應 None，失敗的訂單才有錯誤信息
                    real_errors = [str(e) for e in result["errors"] if e is not None]
                    if real_errors:
                        all_errors.extend(real_errors)
                        logger.warning("批量下單部分失敗: %d 個錯誤", len(real_errors))

                # 如果整個批次失敗
                if "error" in result and "orders" not in result:
                    logger.error("批量下單失敗: %s", result["error"])
                    all_errors.append(result["error"])

        # 返回結果
        batch_result = BatchOrderResult(
            success=len(all_results) > 0 or len(all_errors) == 0,
            orders=all_results,
            failed_count=len(all_errors),
            errors=all_errors if all_errors else [],
            raw={"successful_count": len(all_results), "failed_count": len(all_errors)},
        )
        
        if all_results:
            return ApiResponse.ok(batch_result, raw=batch_result.raw)
        else:
            return ApiResponse.error("批量下單全部失敗", raw={"errors": all_errors})

    def get_open_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取開放訂單"""
        params = {}
        if symbol:
            params["market"] = symbol

        result = self.make_request(
            "GET",
            "/orders",
            instruction=True,
            params=params,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        # 返回訂單列表
        orders_raw = result.get("results", []) if isinstance(result, dict) else []
        orders: List[OrderInfo] = []
        for order in orders_raw:
            size_val = self.safe_float(order.get("size"))
            filled_val = self.safe_float(order.get("filled_size"))
            orders.append(OrderInfo(
                order_id=order.get("id") or order.get("order_id"),
                client_order_id=order.get("client_order_id"),
                symbol=order.get("market"),
                side=order.get("side"),
                price=self.safe_float(order.get("price")),
                size=size_val,
                filled_size=filled_val,
                remaining_size=size_val - filled_val if size_val and filled_val else size_val,
                status=order.get("status"),
                order_type=order.get("type"),
                time_in_force=order.get("instruction"),
                created_at=order.get("created_at"),
                raw=order,
            ))
        return ApiResponse.ok(orders, raw=result)

    def cancel_order(self, order_id: str, symbol: str) -> ApiResponse:
        """取消指定訂單"""
        result = self.make_request(
            "DELETE",
            f"/orders/{order_id}",
            instruction=True,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        cancel_result = CancelResult(
            success=True,
            order_id=order_id,
            cancelled_count=1,
            raw=result,
        )
        return ApiResponse.ok(cancel_result, raw=result)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        """批量取消訂單

        Args:
            symbol: 交易對符號（可選）。如果提供，只取消該市場的訂單；否則取消所有訂單

        Returns:
            ApiResponse with CancelResult
        """
        params = {}
        if symbol:
            params["market"] = symbol

        result = self.make_request(
            "DELETE",
            "/orders",
            instruction=True,
            params=params,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        # 計算取消的訂單數量
        cancelled_count = 0
        if isinstance(result, dict):
            if "results" in result:
                cancelled_count = len(result["results"])
            elif "deleted" in result:
                cancelled_count = result["deleted"]
            elif isinstance(result.get("data"), list):
                cancelled_count = len(result["data"])
        
        cancel_result = CancelResult(
            success=True,
            cancelled_count=cancelled_count,
            raw=result,
        )
        return ApiResponse.ok(cancel_result, raw=result)

    def get_ticker(self, symbol: str) -> ApiResponse:
        """獲取行情信息 - 使用 markets/summary 端點

        注意：Paradex 使用 /markets/summary 端點來獲取市場行情數據
        """
        # 獲取市場摘要（需要認證）
        summary_result = self.make_request(
            "GET",
            "/markets/summary",
            instruction=True,
            params={"market": symbol},
            retry_count=self.max_retries
        )

        error = self._check_raw_error(summary_result)
        if error:
            return error

        # Paradex summary 返回格式：{"results": [{"symbol": "...", "bid": "...", "ask": "...", "last_traded_price": "...", ...}]}
        results = summary_result.get("results", [])

        if not results:
            return ApiResponse.error(f"未找到交易對 {symbol} 的市場數據")

        market_data = results[0]

        # 轉換為標準格式
        ticker = TickerInfo(
            symbol=symbol,
            last_price=self.safe_float(market_data.get("last_traded_price")),
            bid_price=self.safe_float(market_data.get("bid")),
            ask_price=self.safe_float(market_data.get("ask")),
            volume_24h=self.safe_float(market_data.get("volume_24h")),
            mark_price=self.safe_float(market_data.get("mark_price")),
            raw=summary_result,
        )
        return ApiResponse.ok(ticker, raw=summary_result)

    def get_markets(self) -> ApiResponse:
        """獲取市場列表"""
        result = self.make_request(
            "GET",
            "/markets",
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        # 轉換為標準格式
        markets_raw = result.get("results", []) if isinstance(result, dict) else []
        markets: List[MarketInfo] = []
        for market in markets_raw:
            markets.append(MarketInfo(
                symbol=market.get("symbol"),
                base_asset=market.get("base_currency"),
                quote_asset=market.get("quote_currency"),
                market_type="PERP",
                status=market.get("status"),
                min_order_size=str(market.get("min_order_size", 0)),
                tick_size=str(market.get("price_tick_size") or market.get("tick_size", "0.1")),
                base_precision=market.get("base_precision"),
                quote_precision=market.get("quote_precision"),
                raw=market,
            ))
        return ApiResponse.ok(markets, raw=result)

    def get_order_book(self, symbol: str, limit: int = 20) -> ApiResponse:
        """獲取訂單簿

        注意：Paradex 的訂單簿端點是 /orderbook/{symbol}
        """
        result = self.make_request(
            "GET",
            f"/orderbook/{symbol}",
            instruction=True,  # Paradex 的 orderbook 端點需要認證
            params={"depth": limit} if limit else None,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        # 轉換為標準格式
        orderbook_data = result.get("orderbook", result.get("results", result))

        raw_bids = orderbook_data.get("bids", [])
        raw_asks = orderbook_data.get("asks", [])

        # 轉換為 OrderBookLevel 格式
        bids = [OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in raw_bids] if raw_bids else []
        asks = [OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in raw_asks] if raw_asks else []

        order_book = OrderBookInfo(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=orderbook_data.get("timestamp"),
            raw=result,
        )
        return ApiResponse.ok(order_book, raw=result)

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取持倉信息

        Args:
            symbol: 交易對符號（可選），用於過濾特定市場

        Returns:
            ApiResponse with List[PositionInfo]

        注意：Paradex API 返回所有持倉，然後在客户端進行過濾
        """
        result = self.make_request(
            "GET",
            "/positions",
            instruction=True,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        # 轉換為標準格式
        positions_data = result.get("results", [])

        # 如果指定了交易對，則過濾
        if symbol:
            positions_data = [p for p in positions_data if p.get("market") == symbol]

        positions: List[PositionInfo] = []

        for pos in positions_data:
            # 讀取實際的字段名
            size_str = pos.get("size", "0")
            size = float(size_str) if size_str else 0.0

            # 根據 size 判斷方向
            if size > 0:
                side = "LONG"
            elif size < 0:
                side = "SHORT"
            else:
                side = "FLAT"

            positions.append(PositionInfo(
                symbol=pos.get("market"),
                side=side,
                size=abs(size),
                entry_price=self.safe_float(pos.get("average_entry_price")),
                mark_price=self.safe_float(pos.get("mark_price")),
                liquidation_price=self.safe_float(pos.get("liquidation_price")),
                unrealized_pnl=self.safe_float(pos.get("unrealized_pnl")),
                leverage=self.safe_float(pos.get("leverage")),
                raw=pos,
            ))

        return ApiResponse.ok(positions, raw=result)

    def get_market_limits(self, symbol: str) -> ApiResponse:
        """獲取市場限制信息"""
        self._fetch_markets_if_needed()

        market_info = self._market_info_cache.get(symbol)
        if not market_info:
            logger.error(f"未找到交易對 {symbol} 的信息")
            return ApiResponse.error(f"未找到交易對 {symbol} 的信息")

        # 獲取 tick_size，Paradex 使用 price_tick_size 字段
        tick_size_raw = market_info.get("price_tick_size") or market_info.get("tick_size")
        if tick_size_raw:
            tick_size = str(tick_size_raw)
        else:
            # 默認值 0.1（適用於大多數 Paradex 合約）
            logger.warning(f"市場 {symbol} 未返回 tick_size，使用默認值 0.1")
            tick_size = "0.1"

        logger.debug(f"市場 {symbol} 的 tick_size: {tick_size}")

        # 返回標準化格式
        market_limits = MarketInfo(
            symbol=symbol,
            base_asset=market_info.get("base_currency"),
            quote_asset=market_info.get("quote_currency"),
            market_type="PERP",
            status=market_info.get("status", "ACTIVE"),
            min_order_size=str(market_info.get("min_order_size", 0)),
            tick_size=tick_size,
            base_precision=market_info.get("base_precision", 8),
            quote_precision=market_info.get("quote_precision", 2),
            raw=market_info,
        )
        return ApiResponse.ok(market_limits, raw=market_info)

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100, **kwargs) -> ApiResponse:
        """獲取成交歷史

        Args:
            symbol: 交易對符號（可選），用於過濾特定市場
            limit: 每頁返回的最大記錄數（默認100）
            **kwargs: 其他可選參數
                - cursor: 分頁遊標
                - start_at: 開始時間戳（毫秒）
                - end_at: 結束時間戳（毫秒）

        Returns:
            ApiResponse with List[TradeInfo]
        """
        # Paradex API 使用 page_size 而不是 limit
        params = {"page_size": limit}

        # 添加市場過濾
        if symbol:
            params["market"] = symbol

        # 添加其他可選參數
        if "cursor" in kwargs:
            params["cursor"] = kwargs["cursor"]
        if "start_at" in kwargs:
            params["start_at"] = kwargs["start_at"]
        if "end_at" in kwargs:
            params["end_at"] = kwargs["end_at"]

        result = self.make_request(
            "GET",
            "/fills",
            instruction=True,
            params=params,
            retry_count=self.max_retries
        )

        error = self._check_raw_error(result)
        if error:
            return error

        # 轉換為標準格式
        fills_raw = result.get("results", []) if isinstance(result, dict) else result
        trades: List[TradeInfo] = []
        for fill in fills_raw:
            trades.append(TradeInfo(
                trade_id=fill.get("id") or fill.get("trade_id"),
                order_id=fill.get("order_id"),
                symbol=fill.get("market"),
                side=fill.get("side"),
                price=self.safe_float(fill.get("price")),
                size=self.safe_float(fill.get("size")),
                fee=self.safe_float(fill.get("fee")),
                fee_asset=fill.get("fee_currency"),
                is_maker=fill.get("liquidity") == "MAKER",
                timestamp=fill.get("created_at"),
                raw=fill,
            ))
        return ApiResponse.ok(trades, raw=result)
