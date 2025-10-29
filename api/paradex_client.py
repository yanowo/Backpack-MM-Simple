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
from typing import List

from .base_client import (
    BaseExchangeClient,
    ApiResponse,
    OrderResult,
    OrderInfo,
    BalanceInfo,
    PositionInfo,
    MarketInfo,
    TickerInfo,
    OrderBookInfo,
    OrderBookLevel,
    TradeInfo
)
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
        """將金額轉換為鏈上格式（帶固定小數位）"""
        # Paradex 使用 8 位小數的整數表示
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

    def get_balance(self) -> Dict[str, Any]:
        """獲取賬户餘額"""
        result = self.make_request(
            "GET",
            "/account",
            instruction=True,
            retry_count=self.max_retries
        )

        if isinstance(result, dict) and "error" in result:
            return result

        # 轉換為標準格式（與其他客户端一致）
        balances = {}
        account_data = result.get("results", {}) if isinstance(result, dict) else {}

        for asset, data in account_data.items():
            if isinstance(data, dict):
                balances[asset] = {
                    "available": str(data.get("available", 0)),
                    "locked": str(data.get("locked", 0)),
                    "total": str(data.get("total", 0))
                }

        return balances

    def execute_order(self, order_details: Dict[str, Any]) -> Dict[str, Any]:
        """執行訂單

        Paradex 需要對每個訂單進行 StarkNet 簽名
        """
        symbol = order_details.get("symbol")
        if not symbol:
            return {"error": "缺少交易對"}

        side = order_details.get("side", "").upper()
        if side not in ["BUY", "SELL", "BID", "ASK"]:
            return {"error": "無效的買賣方向"}

        # 標準化方向
        if side in ["BID"]:
            side = "BUY"
        elif side in ["ASK"]:
            side = "SELL"

        order_type = order_details.get("orderType") or order_details.get("type", "LIMIT")
        size = order_details.get("quantity") or order_details.get("size")

        # 生成簽名時間戳（毫秒）
        signature_timestamp = int(time.time() * 1000)

        # 構建訂單數據用於簽名
        order_data_for_signature = {
            "market": symbol,
            "side": side,
            "type": order_type.upper(),
            "size": float(size),
        }

        # 限價單需要價格
        if order_type.upper() == "LIMIT":
            price = order_details.get("price")
            if price:
                order_data_for_signature["price"] = float(price)
            else:
                return {"error": "限價單缺少價格"}

        # 生成訂單簽名
        try:
            signature = self._sign_order(order_data_for_signature, signature_timestamp)
        except Exception as e:
            logger.error(f"訂單簽名失敗: {e}")
            import traceback
            logger.error(f"簽名異常詳情: {traceback.format_exc()}")
            return {"error": f"訂單簽名失敗: {e}"}

        # 構建 API 請求 payload
        payload = {
            "market": symbol,
            "side": side,
            "type": order_type.upper(),
            "size": str(size),
            "signature": signature,
            "signature_timestamp": signature_timestamp,
        }

        # 限價單價格
        if order_type.upper() == "LIMIT" and "price" in order_data_for_signature:
            payload["price"] = str(order_data_for_signature["price"])

        # 可選參數
        if "clientId" in order_details:
            payload["client_order_id"] = order_details["clientId"]

        if "timeInForce" in order_details:
            payload["time_in_force"] = order_details["timeInForce"]

        if "postOnly" in order_details:
            payload["post_only"] = order_details["postOnly"]

        if "reduceOnly" in order_details:
            payload["reduce_only"] = order_details["reduceOnly"]

        result = self.make_request(
            "POST",
            "/orders",
            instruction=True,
            data=payload,
            retry_count=self.max_retries
        )

        # 返回原始結果或錯誤
        return result

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
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

        if isinstance(result, dict) and "error" in result:
            return result

        # 返回訂單列表
        return result.get("results", []) if isinstance(result, dict) else []

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """取消指定訂單"""
        result = self.make_request(
            "DELETE",
            f"/orders/{order_id}",
            instruction=True,
            retry_count=self.max_retries
        )

        return result

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """取消所有訂單"""
        result = self.make_request(
            "DELETE",
            "/orders",
            instruction=True,
            data={"market": symbol},
            retry_count=self.max_retries
        )

        return result

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
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

        if isinstance(summary_result, dict) and "error" in summary_result:
            return summary_result

        # Paradex summary 返回格式：{"results": [{"symbol": "...", "bid": "...", "ask": "...", "last_traded_price": "...", ...}]}
        results = summary_result.get("results", [])

        if not results:
            return {"error": f"未找到交易對 {symbol} 的市場數據"}

        market_data = results[0]

        # 轉換為標準格式（與其他客户端一致）
        return {
            "symbol": symbol,
            "lastPrice": str(market_data.get("last_traded_price", 0)) if market_data.get("last_traded_price") else None,
            "bidPrice": str(market_data.get("bid", 0)) if market_data.get("bid") else None,
            "askPrice": str(market_data.get("ask", 0)) if market_data.get("ask") else None,
            "volume": str(market_data.get("volume_24h", 0)) if market_data.get("volume_24h") else None,
            "markPrice": str(market_data.get("mark_price", 0)) if market_data.get("mark_price") else None,
            "raw": summary_result
        }

    def get_markets(self) -> Dict[str, Any]:
        """獲取市場列表"""
        result = self.make_request(
            "GET",
            "/markets",
            retry_count=self.max_retries
        )

        if isinstance(result, dict) and "error" in result:
            return result

        # 返回原始結果，與其他客户端保持一致
        return result

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
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

        if isinstance(result, dict) and "error" in result:
            return result

        # 轉換為標準格式（與其他客户端一致）
        orderbook_data = result.get("orderbook", result.get("results", result))

        bids = orderbook_data.get("bids", [])
        asks = orderbook_data.get("asks", [])

        # 轉換為 [price, quantity] 格式
        formatted_bids = [[float(b[0]), float(b[1])] for b in bids] if bids else []
        formatted_asks = [[float(a[0]), float(a[1])] for a in asks] if asks else []

        return {
            "bids": formatted_bids,
            "asks": formatted_asks,
            "symbol": symbol,
            "timestamp": orderbook_data.get("timestamp"),
        }

    def get_positions(self, symbol: Optional[str] = None) -> Any:
        """獲取持倉信息

        Args:
            symbol: 交易對符號（可選），用於過濾特定市場

        Returns:
            標準化的持倉信息列表

        注意：Paradex API 返回所有持倉，然後在客户端進行過濾
        """
        result = self.make_request(
            "GET",
            "/positions",
            instruction=True,
            retry_count=self.max_retries
        )

        if isinstance(result, dict) and "error" in result:
            return result

        # 轉換為標準格式（與其他客户端一致）
        positions_data = result.get("results", [])

        # 如果指定了交易對，則過濾
        if symbol:
            positions_data = [p for p in positions_data if p.get("market") == symbol]

        normalized = []

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

            # 使用正確的字段名 average_entry_price
            entry_price = pos.get("average_entry_price")
            unrealized_pnl = pos.get("unrealized_pnl", "0")

            normalized.append({
                "symbol": pos.get("market"),
                "side": side,
                "size": str(abs(size)),
                "netQuantity": str(size),
                "entryPrice": entry_price,  # 使用 average_entry_price
                "markPrice": pos.get("mark_price"),  # 注意：API 可能不返回 mark_price
                "unrealizedPnl": unrealized_pnl,
                "pnlUnrealized": unrealized_pnl,
                "liquidationPrice": pos.get("liquidation_price"),
                "leverage": pos.get("leverage"),
                "status": pos.get("status"),
                "raw": pos,
            })

        return normalized

    def get_market_limits(self, symbol: str) -> Optional[Dict[str, Any]]:
        """獲取市場限制信息"""
        self._fetch_markets_if_needed()

        market_info = self._market_info_cache.get(symbol)
        if not market_info:
            logger.error(f"未找到交易對 {symbol} 的信息")
            return None

        # 獲取 tick_size，Paradex 使用 price_tick_size 字段
        tick_size_raw = market_info.get("price_tick_size") or market_info.get("tick_size")
        if tick_size_raw:
            tick_size = str(tick_size_raw)
        else:
            # 默認值 0.1（適用於大多數 Paradex 合約）
            logger.warning(f"市場 {symbol} 未返回 tick_size，使用默認值 0.1")
            tick_size = "0.1"

        logger.debug(f"市場 {symbol} 的 tick_size: {tick_size}")

        # 返回字典格式，與其他客户端一致
        return {
            "symbol": symbol,
            "base_asset": market_info.get("base_currency"),
            "quote_asset": market_info.get("quote_currency"),
            "market_type": "PERP",
            "status": market_info.get("status", "ACTIVE"),
            "min_order_size": str(market_info.get("min_order_size", 0)),
            "tick_size": tick_size,
            "base_precision": market_info.get("base_precision", 8),
            "quote_precision": market_info.get("quote_precision", 2)
        }

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100, **kwargs) -> Any:
        """獲取成交歷史

        Args:
            symbol: 交易對符號（可選），用於過濾特定市場
            limit: 每頁返回的最大記錄數（默認100）
            **kwargs: 其他可選參數
                - cursor: 分頁遊標
                - start_at: 開始時間戳（毫秒）
                - end_at: 結束時間戳（毫秒）

        Returns:
            成交記錄列表或錯誤信息
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

        if isinstance(result, dict) and "error" in result:
            return result

        # 返回成交列表
        return result.get("results", []) if isinstance(result, dict) else result
