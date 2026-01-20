"""
API請求客户端模塊
"""
import json
import time
import requests
from decimal import Decimal
from typing import Dict, Any, Iterable, List, Optional, Tuple
from .auth import create_signature
from config import API_URL, API_VERSION, DEFAULT_WINDOW
from logger import setup_logger
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
    DepositAddressInfo,
    CancelResult,
    BatchOrderResult,
)
from .proxy_utils import get_proxy_config

logger = setup_logger("api.client")


class BPClient(BaseExchangeClient):
    """Backpack exchange client (REST).

    統一封裝 API 請求、簽名與重試邏輯。
    與早期函數式實現對齊（/api vs /wapi 端點與 instruction 名稱），方便遷移。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")

        # 從環境變量讀取代理配置
        self.proxies = get_proxy_config()
        if self.proxies:
            logger.info(f"Backpack 客户端已配置代理: {self.proxies}")

    def get_exchange_name(self) -> str:
        return "Backpack"

    async def connect(self) -> None:
        logger.info("Backpack 客户端已連接")

    async def disconnect(self) -> None:
        logger.info("Backpack 客户端已斷開連接")

    def make_request(self, method: str, endpoint: str, api_key=None, secret_key=None, instruction=None, 
                    params=None, data=None, retry_count=3) -> Dict:
        """
        執行API請求，支持重試機制
        
        Args:
            method: HTTP方法 (GET, POST, DELETE)
            endpoint: API端點
            api_key: API密鑰
            secret_key: API密鑰
            instruction: API指令
            params: 查詢參數
            data: 請求體數據
            retry_count: 重試次數
            
        Returns:
            API響應數據
        """
        url = f"{API_URL}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'X-Broker-Id': '1500'
        }
        
        # 構建簽名信息（如需要）
        if api_key and secret_key and instruction:
            timestamp = str(int(time.time() * 1000))
            window = DEFAULT_WINDOW
            
            # 構建簽名消息
            query_string = ""
            if params:
                sorted_params = sorted(params.items())
                query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
            
            sign_message = f"instruction={instruction}"
            if query_string:
                sign_message += f"&{query_string}"
            sign_message += f"&timestamp={timestamp}&window={window}"
            
            signature = create_signature(secret_key, sign_message)
            if not signature:
                return {"error": "簽名創建失敗"}
            
            headers.update({
                'X-API-KEY': api_key,
                'X-SIGNATURE': signature,
                'X-TIMESTAMP': timestamp,
                'X-WINDOW': window
            })
        
        # 添加查詢參數到URL
        if params and method.upper() in ['GET', 'DELETE']:
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url += f"?{query_string}"
        
        # 實施重試機制
        for attempt in range(retry_count):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, headers=headers, proxies=self.proxies or None, timeout=10)
                elif method.upper() == 'POST':
                    response = requests.post(url, headers=headers, data=json.dumps(data) if data else None, proxies=self.proxies or None, timeout=10)
                elif method.upper() == 'DELETE':
                    response = requests.delete(url, headers=headers, data=json.dumps(data) if data else None, proxies=self.proxies or None, timeout=10)
                else:
                    return {"error": f"不支持的請求方法: {method}"}
                
                # 處理響應
                if response.status_code in [200, 201]:
                    return response.json() if response.text.strip() else {}
                elif response.status_code == 429:  # 速率限制
                    wait_time = 1 * (2 ** attempt)  # 指數退避
                    logger.warning(f"遇到速率限制，等待 {wait_time} 秒後重試")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"狀態碼: {response.status_code}, 消息: {response.text}"
                    if attempt < retry_count - 1:
                        logger.warning(f"請求失敗 ({attempt+1}/{retry_count}): {error_msg}")
                        time.sleep(1)  # 簡單重試延遲
                        continue
                    return {"error": error_msg}
            
            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    logger.warning(f"請求超時 ({attempt+1}/{retry_count})，重試中...")
                    continue
                return {"error": "請求超時"}
            except requests.exceptions.ConnectionError:
                if attempt < retry_count - 1:
                    logger.warning(f"連接錯誤 ({attempt+1}/{retry_count})，重試中...")
                    time.sleep(2)  # 連接錯誤通常需要更長等待
                    continue
                return {"error": "連接錯誤"}
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"請求異常 ({attempt+1}/{retry_count}): {str(e)}，重試中...")
                    continue
                return {"error": f"請求失敗: {str(e)}"}
        
        return {"error": "達到最大重試次數"}

    # 各API端點函數
    def get_deposit_address(self, blockchain: str) -> ApiResponse:
        """獲取存款地址
        
        Returns:
            ApiResponse with data: DepositAddressInfo
        """
        endpoint = f"/wapi/{API_VERSION}/capital/deposit/address"
        instruction = "depositAddressQuery"
        params = {"blockchain": blockchain}
        raw = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        address_info = DepositAddressInfo(
            address=raw.get("address", ""),
            blockchain=blockchain,
            tag=raw.get("tag") or raw.get("memo"),
            raw=raw
        )
        return ApiResponse.ok(address_info, raw=raw)

    def get_balance(self) -> ApiResponse:
        """獲取賬户餘額
        
        Returns:
            ApiResponse with data: List[BalanceInfo]
        """
        endpoint = f"/api/{API_VERSION}/capital"
        instruction = "balanceQuery"
        raw = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction)
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        balances = []
        # Backpack 返回格式: {"USDC": {"available": "100", "locked": "10", "staked": "0"}, ...}
        if isinstance(raw, dict):
            for asset, data in raw.items():
                if isinstance(data, dict):
                    available = self.safe_decimal(data.get("available"), Decimal("0"))
                    locked = self.safe_decimal(data.get("locked"), Decimal("0"))
                    balances.append(BalanceInfo(
                        asset=asset,
                        available=available,
                        locked=locked,
                        total=available + locked,
                        raw=data
                    ))
        
        return ApiResponse.ok(balances, raw=raw)

    def get_collateral(self, subaccount_id: Optional[str] = None) -> ApiResponse:
        """獲取抵押品資產
        
        Returns:
            ApiResponse with data: List[CollateralInfo]
        """
        endpoint = f"/api/{API_VERSION}/capital/collateral"
        params = {}
        if subaccount_id is not None:
            params["subaccountId"] = str(subaccount_id)
        instruction = "collateralQuery" if self.api_key and self.secret_key else None
        raw = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        collaterals = []
        if isinstance(raw, dict):
            for asset, data in raw.items():
                if isinstance(data, dict):
                    total = self.safe_decimal(data.get("total") or data.get("available"), Decimal("0"))
                    free = self.safe_decimal(data.get("available"), Decimal("0"))
                    collaterals.append(CollateralInfo(
                        asset=asset,
                        total_collateral=total,
                        free_collateral=free,
                        raw=data
                    ))
        
        return ApiResponse.ok(collaterals, raw=raw)

    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        """執行訂單
        
        Returns:
            ApiResponse with data: OrderResult
        """
        endpoint = f"/api/{API_VERSION}/order"
        instruction = "orderExecute"
      
        # 根據實際請求體產生簽名參數，確保完全一致
        # 同時預處理 order_details，確保數值格式統一
        params = {}
        processed_order = {}
        for key, value in order_details.items():
            if value is None:
                continue
            if isinstance(value, bool):
                params[key] = str(value).lower()
                processed_order[key] = value  # 布爾值保持原樣，API 需要真正的 boolean
            elif isinstance(value, (int, float)):
                # 數值轉為字符串用於簽名和請求體
                str_value = str(value)
                params[key] = str_value
                processed_order[key] = str_value  # API 期望價格和數量是字符串
            else:
                params[key] = str(value)
                processed_order[key] = value

        raw = self.make_request("POST", endpoint, self.api_key, self.secret_key, instruction, params, processed_order)
        
        error = self._check_raw_error(raw)
        if error:
            return ApiResponse(
                success=False,
                data=OrderResult(
                    success=False,
                    error_message=error.error_message,
                    raw=raw
                ),
                error_message=error.error_message,
                raw=raw
            )
        
        order_result = OrderResult(
            success=True,
            order_id=raw.get("id") or raw.get("orderId"),
            client_order_id=raw.get("clientId"),
            symbol=raw.get("symbol"),
            side=raw.get("side"),
            order_type=raw.get("orderType"),
            size=self.safe_decimal(raw.get("quantity")),
            price=self.safe_decimal(raw.get("price")),
            filled_size=self.safe_decimal(raw.get("executedQuantity")),
            status=raw.get("status"),
            created_at=self.safe_int(raw.get("createdAt")),
            raw=raw
        )
        
        return ApiResponse.ok(order_result, raw=raw)

    def execute_order_batch(self, orders_list: List[Dict[str, Any]], max_batch_size: int = 50) -> ApiResponse:
        """批量執行訂單

        Args:
            orders_list: 訂單列表，每個訂單是一個字典，包含訂單詳情
            max_batch_size: 單次批量下單的最大訂單數量，默認50個

        Returns:
            ApiResponse with data: BatchOrderResult
        """
        # 如果訂單數量超過限制，分批處理
        if len(orders_list) > max_batch_size:
            logger.info(f"訂單數量 {len(orders_list)} 超過單次限制 {max_batch_size}，將分批下單")
            all_results = []
            all_errors = []
            
            for i in range(0, len(orders_list), max_batch_size):
                batch = orders_list[i:i + max_batch_size]
                logger.info(f"處理第 {i//max_batch_size + 1} 批，共 {len(batch)} 個訂單")
                result = self._execute_order_batch_internal(batch)

                if isinstance(result, dict) and "error" in result:
                    all_errors.append(result["error"])
                    continue

                if isinstance(result, list):
                    logger.debug(f"第 {i//max_batch_size + 1} 批返回 {len(result)} 個訂單結果")
                    all_results.extend(result)
                elif isinstance(result, dict):
                    logger.debug(f"第 {i//max_batch_size + 1} 批返回單個訂單結果")
                    all_results.append(result)

                # 批次之間添加短暫延遲，避免速率限制
                if i + max_batch_size < len(orders_list):
                    time.sleep(0.5)

            logger.info(f"所有批次完成，共返回 {len(all_results)} 個訂單結果")
            
            # 轉換為 OrderResult 列表
            order_results = []
            for raw in all_results:
                order_results.append(OrderResult(
                    success=True,
                    order_id=raw.get("id") or raw.get("orderId"),
                    client_order_id=raw.get("clientId"),
                    symbol=raw.get("symbol"),
                    side=raw.get("side"),
                    order_type=raw.get("orderType"),
                    size=self.safe_decimal(raw.get("quantity")),
                    price=self.safe_decimal(raw.get("price")),
                    status=raw.get("status"),
                    raw=raw
                ))
            
            batch_result = BatchOrderResult(
                success=len(order_results) > 0,
                orders=order_results,
                failed_count=len(all_errors),
                errors=all_errors,
                raw=all_results
            )
            return ApiResponse.ok(batch_result, raw=all_results)
        else:
            raw = self._execute_order_batch_internal(orders_list)
            
            if isinstance(raw, dict) and "error" in raw:
                return ApiResponse.error(raw["error"], raw=raw)
            
            order_results = []
            raw_list = raw if isinstance(raw, list) else [raw]
            for item in raw_list:
                order_results.append(OrderResult(
                    success=True,
                    order_id=item.get("id") or item.get("orderId"),
                    client_order_id=item.get("clientId"),
                    symbol=item.get("symbol"),
                    side=item.get("side"),
                    order_type=item.get("orderType"),
                    size=self.safe_decimal(item.get("quantity")),
                    price=self.safe_decimal(item.get("price")),
                    status=item.get("status"),
                    raw=item
                ))
            
            batch_result = BatchOrderResult(
                success=True,
                orders=order_results,
                raw=raw
            )
            return ApiResponse.ok(batch_result, raw=raw)

    def _execute_order_batch_internal(self, orders_list):
        """內部批量下單實現

        Args:
            orders_list: 訂單列表

        Returns:
            批量訂單結果
        """
        # 根據 Backpack API 文檔，批量下單端點是 POST /api/v1/orders
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderExecute"  # 每個訂單使用 orderExecute 指令

        # 預處理訂單，確保數值格式統一（與 execute_order 保持一致）
        processed_orders = []
        for order in orders_list:
            processed_order = {}
            for key, value in order.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    processed_order[key] = value  # 布爾值保持原樣
                elif isinstance(value, (int, float)):
                    processed_order[key] = str(value)  # 數值轉為字符串
                else:
                    processed_order[key] = value
            processed_orders.append(processed_order)

        # 請求體使用預處理後的訂單
        data = processed_orders

        # 構建簽名參數字符串
        # 使用預處理後的訂單，確保簽名和請求體格式一致
        param_strings = []

        for order in processed_orders:
            # 按字母順序排序訂單參數
            sorted_params = sorted(order.items())

            # 構建單個訂單的參數字符串
            order_params = []
            order_params.append(f"instruction={instruction}")

            for key, value in sorted_params:
                if value is None:
                    continue
                # 布爾值轉換為小寫字符串（用於簽名）
                if isinstance(value, bool):
                    order_params.append(f"{key}={str(value).lower()}")
                else:
                    order_params.append(f"{key}={value}")

            param_strings.append("&".join(order_params))

        # 拼接所有訂單的參數字符串
        sign_message = "&".join(param_strings)

        # 添加時間戳和窗口
        timestamp = str(int(time.time() * 1000))
        window = DEFAULT_WINDOW
        sign_message += f"&timestamp={timestamp}&window={window}"

        # 創建簽名
        signature = create_signature(self.secret_key, sign_message)
        if not signature:
            return {"error": "簽名創建失敗"}

        # 構建請求頭
        url = f"{API_URL}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': self.api_key,
            'X-SIGNATURE': signature,
            'X-TIMESTAMP': timestamp,
            'X-WINDOW': window,
            'X-Broker-Id': '1500'
        }

        # 執行請求（使用自定義頭，不通過 make_request）
        import json
        import requests

        retry_count = 3
        for attempt in range(retry_count):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data), proxies=self.proxies or None, timeout=30)

                if response.status_code in [200, 201]:
                    return response.json() if response.text.strip() else {}
                elif response.status_code == 429:  # 速率限制
                    wait_time = 1 * (2 ** attempt)
                    logger.warning(f"遇到速率限制，等待 {wait_time} 秒後重試")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"狀態碼: {response.status_code}, 消息: {response.text}"
                    if attempt < retry_count - 1:
                        logger.warning(f"請求失敗 ({attempt+1}/{retry_count}): {error_msg}")
                        time.sleep(1)
                        continue
                    return {"error": error_msg}

            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"請求異常 ({attempt+1}/{retry_count}): {str(e)}，重試中...")
                    time.sleep(1)
                    continue
                return {"error": f"請求失敗: {str(e)}"}

        return {"error": "達到最大重試次數"}

    def get_open_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取未成交訂單
        
        Returns:
            ApiResponse with data: List[OrderInfo]
        """
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderQueryAll"
        params = {}
        if symbol:
            params["symbol"] = symbol
        raw = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        orders = []
        raw_list = raw if isinstance(raw, list) else []
        for item in raw_list:
            orders.append(OrderInfo(
                order_id=item.get("id") or item.get("orderId"),
                symbol=item.get("symbol"),
                side=item.get("side"),
                order_type=item.get("orderType"),
                size=self.safe_decimal(item.get("quantity"), Decimal("0")),
                price=self.safe_decimal(item.get("price")),
                status=item.get("status"),
                filled_size=self.safe_decimal(item.get("executedQuantity"), Decimal("0")),
                remaining_size=self.safe_decimal(item.get("quantity"), Decimal("0")) - self.safe_decimal(item.get("executedQuantity"), Decimal("0")),
                client_order_id=item.get("clientId"),
                created_at=self.safe_int(item.get("createdAt")),
                time_in_force=item.get("timeInForce"),
                post_only=item.get("postOnly", False),
                raw=item
            ))
        
        return ApiResponse.ok(orders, raw=raw)

    def cancel_all_orders(self, symbol: str) -> ApiResponse:
        """取消所有訂單
        
        Returns:
            ApiResponse with data: CancelResult
        """
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderCancelAll"
        params = {"symbol": symbol}
        data = {"symbol": symbol}
        raw = self.make_request("DELETE", endpoint, self.api_key, self.secret_key, instruction, params, data)
        
        error = self._check_raw_error(raw)
        if error:
            return ApiResponse(
                success=False,
                data=CancelResult(success=False, error_message=error.error_message, raw=raw),
                error_message=error.error_message,
                raw=raw
            )
        
        cancelled_count = len(raw) if isinstance(raw, list) else 1
        return ApiResponse.ok(CancelResult(success=True, cancelled_count=cancelled_count, raw=raw), raw=raw)

    def cancel_order(self, order_id: str, symbol: str) -> ApiResponse:
        """取消指定訂單
        
        Returns:
            ApiResponse with data: CancelResult
        """
        endpoint = f"/api/{API_VERSION}/order"
        instruction = "orderCancel"
        params = {"orderId": order_id, "symbol": symbol}
        data = {"orderId": order_id, "symbol": symbol}
        raw = self.make_request("DELETE", endpoint, self.api_key, self.secret_key, instruction, params, data)
        
        error = self._check_raw_error(raw)
        if error:
            return ApiResponse(
                success=False,
                data=CancelResult(success=False, order_id=order_id, error_message=error.error_message, raw=raw),
                error_message=error.error_message,
                raw=raw
            )
        
        return ApiResponse.ok(CancelResult(success=True, order_id=order_id, cancelled_count=1, raw=raw), raw=raw)

    def get_ticker(self, symbol: str) -> ApiResponse:
        """獲取市場價格
        
        Returns:
            ApiResponse with data: TickerInfo
        """
        endpoint = f"/api/{API_VERSION}/ticker"
        params = {"symbol": symbol}
        raw = self.make_request("GET", endpoint, params=params)

        error = self._check_raw_error(raw)
        if error:
            return error

        parsed = self._parse_ticker_snapshot(raw)
        if not parsed:
            return ApiResponse.error("無法解析ticker數據", raw=raw)

        ticker = TickerInfo(
            symbol=symbol,
            last_price=self.safe_decimal(parsed.get("lastPrice")),
            bid_price=self.safe_decimal(parsed.get("bidPrice")),
            ask_price=self.safe_decimal(parsed.get("askPrice")),
            volume_24h=self.safe_decimal(parsed.get("volume")),
            change_percent_24h=self.safe_decimal(parsed.get("change24h")),
            raw=raw
        )
        
        return ApiResponse.ok(ticker, raw=raw)

    def get_markets(self) -> ApiResponse:
        """獲取所有交易對信息
        
        Returns:
            ApiResponse with data: List[MarketInfo]
        """
        endpoint = f"/api/{API_VERSION}/markets"
        raw = self.make_request("GET", endpoint)
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        markets = []
        raw_list = raw if isinstance(raw, list) else []
        for item in raw_list:
            filters = item.get("filters", {})
            tick_size = "0.00000001"
            min_order_size = "0"
            
            if "price" in filters:
                tick_size = filters["price"].get("tickSize", tick_size)
            if "quantity" in filters:
                min_order_size = filters["quantity"].get("minQuantity", min_order_size)
            
            markets.append(MarketInfo(
                symbol=item.get("symbol"),
                base_asset=item.get("baseSymbol"),
                quote_asset=item.get("quoteSymbol"),
                market_type="PERP" if item.get("type") == "perpetual" else "SPOT",
                status=item.get("status", "ACTIVE"),
                min_order_size=self.safe_decimal(min_order_size, Decimal("0")),
                tick_size=self.safe_decimal(tick_size, Decimal("0.00000001")),
                base_precision=len(min_order_size.split(".")[-1]) if "." in min_order_size else 0,
                quote_precision=len(tick_size.split(".")[-1]) if "." in tick_size else 0,
                raw=item
            ))
        
        return ApiResponse.ok(markets, raw=raw)

    def get_order_book(self, symbol: str, limit: Optional[int] = None) -> ApiResponse:
        """獲取市場深度
        
        Returns:
            ApiResponse with data: OrderBookInfo
        """
        endpoint = f"/api/{API_VERSION}/depth"
        params = {"symbol": symbol}
        if limit is not None:
            params["limit"] = str(limit)
        raw = self.make_request("GET", endpoint, params=params)

        error = self._check_raw_error(raw)
        if error:
            return error

        bids_raw, asks_raw = self._parse_order_book_snapshot(raw)
        
        bids = [OrderBookLevel(price=Decimal(str(b[0])), quantity=Decimal(str(b[1]))) for b in bids_raw]
        asks = [OrderBookLevel(price=Decimal(str(a[0])), quantity=Decimal(str(a[1]))) for a in asks_raw]
        
        order_book = OrderBookInfo(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=self.safe_int(self._extract_from_payload(raw, ("ts", "timestamp", "time"))),
            sequence=self.safe_int(self._extract_from_payload(raw, ("sequence", "seq", "lastUpdateId"))),
            raw=raw
        )

        return ApiResponse.ok(order_book, raw=raw)

    # ------------------------------------------------------------------
    # Snapshot parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_from_payload(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
        """從payload中提取指定鍵的值，支持在data節點下查找"""
        data = payload.get("data") if isinstance(payload, dict) else None
        for key in keys:
            if isinstance(payload, dict) and key in payload and payload[key] not in (None, ""):
                return payload[key]
            if isinstance(data, dict) and key in data and data[key] not in (None, ""):
                return data[key]
        return None

    @classmethod
    def _parse_order_book_snapshot(cls, payload: Dict[str, Any]) -> Tuple[List[List[float]], List[List[float]]]:
        """根據官方 API 規範解析訂單簿資料結構"""
        if not isinstance(payload, dict):
            return [], []

        data = payload.get("data", payload)
        bids_raw = data.get("bids", []) or []
        asks_raw = data.get("asks", []) or []

        def _normalise_level(level: Any) -> Optional[List[float]]:
            if isinstance(level, dict):
                price = cls._extract_from_payload(level, ("price", "px", "p"))
                quantity = cls._extract_from_payload(level, ("size", "quantity", "q", "sz"))
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                price, quantity = level[0], level[1]
            else:
                return None

            try:
                return [float(price), float(quantity)]
            except (TypeError, ValueError):
                return None

        bids = [item for item in (_normalise_level(level) for level in bids_raw) if item]
        asks = [item for item in (_normalise_level(level) for level in asks_raw) if item]

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return bids, asks

    @classmethod
    def _parse_ticker_snapshot(cls, payload: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """根據官方 API 規範解析 ticker 響應"""
        if not isinstance(payload, dict):
            return {}

        data = payload.get("data", payload)

        def _safe_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        bid = _safe_float(cls._extract_from_payload(data, ("bidPrice", "bestBidPrice", "bid", "bestBid", "buy")))
        ask = _safe_float(cls._extract_from_payload(data, ("askPrice", "bestAskPrice", "ask", "bestAsk", "sell")))
        last = _safe_float(cls._extract_from_payload(data, ("lastPrice", "price", "last", "close", "markPrice")))

        result: Dict[str, Optional[str]] = {}
        if bid is not None:
            result["bidPrice"] = f"{bid}"
            result["bestBidPrice"] = result["bidPrice"]
        if ask is not None:
            result["askPrice"] = f"{ask}"
            result["bestAskPrice"] = result["askPrice"]
        if last is not None:
            result["lastPrice"] = f"{last}"
            result["price"] = result["lastPrice"]

        volume = cls._extract_from_payload(data, ("volume", "baseVolume", "quoteVolume"))
        if volume is not None:
            result["volume"] = str(volume)

        change = cls._extract_from_payload(data, ("change24h", "priceChangePercent", "change"))
        if change is not None:
            result["change24h"] = str(change)

        return result

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100) -> ApiResponse:
        """獲取歷史成交記錄
        
        Returns:
            ApiResponse with data: List[TradeInfo]
        """
        endpoint = f"/wapi/{API_VERSION}/history/fills"
        instruction = "fillHistoryQueryAll"
        params = {"limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        raw = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        trades = []
        raw_list = raw if isinstance(raw, list) else []
        for item in raw_list:
            trades.append(TradeInfo(
                trade_id=str(item.get("tradeId", "")),
                order_id=item.get("orderId"),
                symbol=item.get("symbol"),
                side=item.get("side"),
                size=self.safe_decimal(item.get("quantity"), Decimal("0")),
                price=self.safe_decimal(item.get("price"), Decimal("0")),
                fee=self.safe_decimal(item.get("fee")),
                fee_asset=item.get("feeSymbol"),
                timestamp=self.safe_int(item.get("createdAt")),
                is_maker=item.get("isMaker"),
                raw=item
            ))
        
        return ApiResponse.ok(trades, raw=raw)

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> ApiResponse:
        """獲取K線數據
        
        Returns:
            ApiResponse with data: List[KlineInfo]
        """
        endpoint = f"/api/{API_VERSION}/klines"
        
        # 計算起始時間 (秒)
        current_time = int(time.time())
        
        # 各間隔對應的秒數
        interval_seconds = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
            "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800, "1month": 2592000
        }
        
        # 計算合適的起始時間
        duration = interval_seconds.get(interval, 3600)
        start_time = current_time - (duration * limit)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": str(start_time)
        }

        raw = self.make_request("GET", endpoint, params=params)
        
        error = self._check_raw_error(raw)
        if error:
            return error
        
        klines = []
        raw_list = raw if isinstance(raw, list) else []
        for item in raw_list:
            if isinstance(item, dict):
                klines.append(KlineInfo(
                    open_time=self.safe_int(item.get("start"), 0),
                    close_time=self.safe_int(item.get("end"), 0),
                    open_price=self.safe_decimal(item.get("open"), Decimal("0")),
                    high_price=self.safe_decimal(item.get("high"), Decimal("0")),
                    low_price=self.safe_decimal(item.get("low"), Decimal("0")),
                    close_price=self.safe_decimal(item.get("close"), Decimal("0")),
                    volume=self.safe_decimal(item.get("volume"), Decimal("0")),
                    quote_volume=self.safe_decimal(item.get("quoteVolume")),
                    trades_count=self.safe_int(item.get("trades")),
                    raw=item
                ))
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                # 數組格式: [open_time, open, high, low, close, volume, ...]
                klines.append(KlineInfo(
                    open_time=self.safe_int(item[0], 0),
                    close_time=self.safe_int(item[0], 0) + duration,
                    open_price=self.safe_decimal(item[1], Decimal("0")),
                    high_price=self.safe_decimal(item[2], Decimal("0")),
                    low_price=self.safe_decimal(item[3], Decimal("0")),
                    close_price=self.safe_decimal(item[4], Decimal("0")),
                    volume=self.safe_decimal(item[5], Decimal("0")),
                    raw=item
                ))
        
        return ApiResponse.ok(klines, raw=raw)

    def get_market_limits(self, symbol: str) -> ApiResponse:
        """獲取交易對的最低訂單量和價格精度
        
        Returns:
            ApiResponse with data: MarketInfo
        """
        markets_response = self.get_markets()

        if not markets_response.success:
            return markets_response
        
        markets_list = markets_response.data
        for market_info in markets_list:
            if market_info.symbol == symbol:
                return ApiResponse.ok(market_info, raw=market_info.raw)
        
        return ApiResponse.error(f"找不到交易對 {symbol} 的信息")

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取永續合約倉位
        
        Returns:
            ApiResponse with data: List[PositionInfo]
        """
        endpoint = f"/api/{API_VERSION}/position"
        instruction = "positionQuery"
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        # 對於倉位查詢，404是正常情況（表示沒有倉位），所以只重試1次
        raw = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params, retry_count=1)

        # 特殊處理404錯誤 - 對於倉位查詢，404表示沒有倉位，返回空列表
        if isinstance(raw, dict) and "error" in raw:
            error_msg = raw["error"]
            if "404" in error_msg or "RESOURCE_NOT_FOUND" in error_msg:
                logger.debug("倉位查詢返回404，表示沒有活躍倉位")
                return ApiResponse.ok([], raw=raw)
            return ApiResponse.error(error_msg, raw=raw)
        
        positions = []
        raw_list = raw if isinstance(raw, list) else [raw] if raw else []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            
            size = self.safe_decimal(item.get("netQuantity") or item.get("size"), Decimal("0"))
            if size > 0:
                side = "LONG"
            elif size < 0:
                side = "SHORT"
            else:
                side = "FLAT"
            
            positions.append(PositionInfo(
                symbol=item.get("symbol"),
                side=side,
                size=abs(size),
                entry_price=self.safe_decimal(item.get("entryPrice")),
                mark_price=self.safe_decimal(item.get("markPrice")),
                liquidation_price=self.safe_decimal(item.get("liquidationPrice")),
                unrealized_pnl=self.safe_decimal(item.get("pnlUnrealized")),
                realized_pnl=self.safe_decimal(item.get("pnlRealized")),
                margin=self.safe_decimal(item.get("margin")),
                leverage=self.safe_decimal(item.get("leverage")),
                raw=item
            ))
        
        return ApiResponse.ok(positions, raw=raw)