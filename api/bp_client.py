"""
API請求客户端模塊
"""
import json
import time
import requests
from typing import Dict, Any, Iterable, List, Optional, Tuple
from .auth import create_signature
from config import API_URL, API_VERSION, DEFAULT_WINDOW
from logger import setup_logger
from .base_client import BaseExchangeClient

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

        # 代理配置
        http_proxy = config.get("http_proxy")
        https_proxy = config.get("https_proxy")
        self.proxies = {}
        if http_proxy:
            self.proxies['http'] = http_proxy
        if https_proxy:
            self.proxies['https'] = https_proxy

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
    def get_deposit_address(self, blockchain):
        """獲取存款地址"""
        endpoint = f"/wapi/{API_VERSION}/capital/deposit/address"
        instruction = "depositAddressQuery"
        params = {"blockchain": blockchain}
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)

    def get_balance(self):
        """獲取賬户餘額"""
        endpoint = f"/api/{API_VERSION}/capital"
        instruction = "balanceQuery"
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction)

    def get_collateral(self, subaccount_id=None):
        """獲取抵押品資產"""
        endpoint = f"/api/{API_VERSION}/capital/collateral"
        params = {}
        if subaccount_id is not None:
            params["subaccountId"] = str(subaccount_id)
        instruction = "collateralQuery" if self.api_key and self.secret_key else None
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)

    def execute_order(self, order_details):
        """執行訂單"""
        endpoint = f"/api/{API_VERSION}/order"
        instruction = "orderExecute"
      
        # 根據實際請求體產生簽名參數，確保完全一致
        params = {}
        for key, value in order_details.items():
            if value is None:
                continue
            if isinstance(value, bool):
                params[key] = str(value).lower()
            else:
                params[key] = str(value)

        return self.make_request("POST", endpoint, self.api_key, self.secret_key, instruction, params, order_details)

    def execute_order_batch(self, orders_list, max_batch_size=50):
        """批量執行訂單

        Args:
            orders_list: 訂單列表，每個訂單是一個字典，包含訂單詳情
            max_batch_size: 單次批量下單的最大訂單數量，默認50個

        Returns:
            批量訂單結果
        """
        # 如果訂單數量超過限制，分批處理
        if len(orders_list) > max_batch_size:
            logger.info(f"訂單數量 {len(orders_list)} 超過單次限制 {max_batch_size}，將分批下單")
            all_results = []
            for i in range(0, len(orders_list), max_batch_size):
                batch = orders_list[i:i + max_batch_size]
                logger.info(f"處理第 {i//max_batch_size + 1} 批，共 {len(batch)} 個訂單")
                result = self._execute_order_batch_internal(batch)

                if isinstance(result, dict) and "error" in result:
                    # 如果某批次失敗，返回錯誤
                    return result

                if isinstance(result, list):
                    logger.debug(f"第 {i//max_batch_size + 1} 批返回 {len(result)} 個訂單結果")
                    all_results.extend(result)
                elif isinstance(result, dict):
                    logger.debug(f"第 {i//max_batch_size + 1} 批返回單個訂單結果")
                    all_results.append(result)

                # 批次之間添加短暫延遲，避免速率限制
                if i + max_batch_size < len(orders_list):
                    import time
                    time.sleep(0.5)

            logger.info(f"所有批次完成，共返回 {len(all_results)} 個訂單結果")
            return all_results
        else:
            return self._execute_order_batch_internal(orders_list)

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

        # 請求體直接是訂單數組，不需要包裝在 {orders: ...} 中
        data = orders_list

        # 構建簽名參數字符串
        # 根據文檔：為每個訂單構建 instruction=orderExecute&param1=value1&param2=value2...
        # 然後將所有訂單的參數字符串拼接起來
        param_strings = []

        for order in orders_list:
            # 按字母順序排序訂單參數
            sorted_params = sorted(order.items())

            # 構建單個訂單的參數字符串
            order_params = []
            order_params.append(f"instruction={instruction}")

            for key, value in sorted_params:
                if value is None:
                    continue
                # 布爾值轉換為小寫字符串
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

    def get_open_orders(self, symbol=None):
        """獲取未成交訂單"""
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderQueryAll"
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)

    def cancel_all_orders(self, symbol):
        """取消所有訂單"""
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderCancelAll"
        params = {"symbol": symbol}
        data = {"symbol": symbol}
        return self.make_request("DELETE", endpoint, self.api_key, self.secret_key, instruction, params, data)

    def cancel_order(self, order_id, symbol):
        """取消指定訂單"""
        endpoint = f"/api/{API_VERSION}/order"
        instruction = "orderCancel"
        params = {"orderId": order_id, "symbol": symbol}
        data = {"orderId": order_id, "symbol": symbol}
        return self.make_request("DELETE", endpoint, self.api_key, self.secret_key, instruction, params, data)

    def get_ticker(self, symbol):
        """獲取市場價格"""
        endpoint = f"/api/{API_VERSION}/ticker"
        params = {"symbol": symbol}
        response = self.make_request("GET", endpoint, params=params)

        if not isinstance(response, dict) or "error" in response:
            return response

        parsed = self._parse_ticker_snapshot(response)
        if not parsed:
            return {"error": "無法解析ticker數據"}

        symbol_value = self._extract_from_payload(response, ("symbol", "s"))
        if symbol_value:
            parsed.setdefault("symbol", symbol_value)

        return parsed

    def get_markets(self):
        """獲取所有交易對信息"""
        endpoint = f"/api/{API_VERSION}/markets"
        return self.make_request("GET", endpoint)

    def get_order_book(self, symbol, limit=None):
        """獲取市場深度"""
        endpoint = f"/api/{API_VERSION}/depth"
        params = {"symbol": symbol}
        if limit is not None:
            params["limit"] = str(limit)
        response = self.make_request("GET", endpoint, params=params)

        if not isinstance(response, dict) or "error" in response:
            return response

        bids, asks = self._parse_order_book_snapshot(response)
        result = {
            "bids": bids,
            "asks": asks,
        }

        # 保留部分關鍵欄位，方便上層使用
        timestamp = self._extract_from_payload(response, ("ts", "timestamp", "time"))
        if timestamp is not None:
            result["timestamp"] = timestamp

        sequence = self._extract_from_payload(response, ("sequence", "seq", "lastUpdateId"))
        if sequence is not None:
            result["sequence"] = sequence

        symbol_value = self._extract_from_payload(response, ("symbol", "s"))
        if symbol_value:
            result["symbol"] = symbol_value

        return result

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

    def get_fill_history(self, symbol=None, limit=100):
        """獲取歷史成交記錄"""
        endpoint = f"/wapi/{API_VERSION}/history/fills"
        instruction = "fillHistoryQueryAll"
        params = {"limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)

    def get_klines(self, symbol, interval="1h", limit=100):
        """獲取K線數據"""
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

        return self.make_request("GET", endpoint, params=params)

    def get_market_limits(self, symbol):
        """獲取交易對的最低訂單量和價格精度"""
        markets_info = self.get_markets()

        if not isinstance(markets_info, dict) and isinstance(markets_info, list):
            for market_info in markets_info:
                if market_info.get('symbol') == symbol:
                    base_asset = market_info.get('baseSymbol')
                    quote_asset = market_info.get('quoteSymbol')
                    
                    # 從filters中獲取精度和最小訂單量信息
                    filters = market_info.get('filters', {})
                    base_precision = 8  # 默認值
                    quote_precision = 8  # 默認值
                    min_order_size = "0"  # 默認值
                    tick_size = "0.00000001"  # 默認值
                    
                    if 'price' in filters:
                        tick_size = filters['price'].get('tickSize', '0.00000001')
                        quote_precision = len(tick_size.split('.')[-1]) if '.' in tick_size else 0
                    
                    if 'quantity' in filters:
                        min_order_size = filters['quantity'].get('minQuantity', '0')
                        min_value = filters['quantity'].get('minQuantity', '0.00000001')
                        base_precision = len(min_value.split('.')[-1]) if '.' in min_value else 0
                    
                    return {
                        'base_asset': base_asset,
                        'quote_asset': quote_asset,
                        'base_precision': base_precision,
                        'quote_precision': quote_precision,
                        'min_order_size': min_order_size,
                        'tick_size': tick_size
                    }
            
            logger.error(f"找不到交易對 {symbol} 的信息")
            return None
        else:
            logger.error(f"無法獲取交易對信息: {markets_info}")
            return None

    def get_positions(self, symbol=None):
        """獲取永續合約倉位"""
        endpoint = f"/api/{API_VERSION}/position"
        instruction = "positionQuery"
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        # 對於倉位查詢，404是正常情況（表示沒有倉位），所以只重試1次
        result = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params, retry_count=1)

        # 特殊處理404錯誤 - 對於倉位查詢，404表示沒有倉位，返回空列表
        if isinstance(result, dict) and "error" in result:
            error_msg = result["error"]
            if "404" in error_msg or "RESOURCE_NOT_FOUND" in error_msg:
                logger.debug("倉位查詢返回404，表示沒有活躍倉位")
                return []  # 返回空列表而不是錯誤
        
        return result