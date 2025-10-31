"""
API請求客户端模塊
"""
import json
import time
import requests
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Iterable, List, Optional, Tuple
from .auth import create_signature
from config import API_URL, API_VERSION, DEFAULT_WINDOW
from logger import setup_logger
from .base_client import (
    BaseExchangeClient, ApiResponse, BalanceInfo, OrderResult, 
    OrderInfo, TickerInfo, MarketInfo, OrderBookInfo, OrderBookLevel,
    KlineInfo, TradeInfo, PositionInfo
)

logger = setup_logger("api.client")

def time_to_int(time_str: str, fmt="%Y-%m-%d %H:%M:%S") -> int:
    dt = datetime.strptime(time_str, fmt)
    return int(dt.timestamp())

class BackpackClient(BaseExchangeClient):
    """Backpack exchange client (REST).
    
    統一封裝 API 請求、簽名與重試邏輯。
    與早期函數式實現對齊（/api vs /wapi 端點與 instruction 名稱），方便遷移。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")

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
                    response = requests.get(url, headers=headers, timeout=10)
                elif method.upper() == 'POST':
                    response = requests.post(url, headers=headers, data=json.dumps(data) if data else None, timeout=10)
                elif method.upper() == 'DELETE':
                    response = requests.delete(url, headers=headers, data=json.dumps(data) if data else None, timeout=10)
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
        """獲取存款地址"""
        endpoint = f"/wapi/{API_VERSION}/capital/deposit/address"
        instruction = "depositAddressQuery"
        params = {"blockchain": blockchain}
        response = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)
        return ApiResponse(success=True, data=response)

    # -------------------------------------------------
    # Balance
    # -------------------------------------------------
    def get_balance(self) -> ApiResponse:
        """獲取賬户餘額"""
        endpoint = f"/api/{API_VERSION}/capital"
        instruction = "balanceQuery"
        response = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction)
        print(response)
        balances = self._parse_balance(response)
        return ApiResponse(success=True, data=balances)

    def _parse_balance(self, raw: Dict) -> List[BalanceInfo]:
        result = []
        for symbol, asset_data in raw.items():
            b = BalanceInfo(
                asset=symbol,
                available=Decimal(asset_data.get("available", "0")),
                locked=Decimal(asset_data.get("locked", "0")),
                total=Decimal(asset_data.get("available", "0")) + Decimal(asset_data.get("locked", "0"))
            )
            result.append(b)
        return result

    def get_collateral(self, subaccount_id=None) -> ApiResponse:
        """獲取抵押品資產"""
        endpoint = f"/api/{API_VERSION}/capital/collateral"
        params = {}
        if subaccount_id is not None:
            params["subaccountId"] = str(subaccount_id)
        instruction = "collateralQuery" if self.api_key and self.secret_key else None
        response = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)
        return ApiResponse(success=True, data=response)

    # -------------------------------------------------
    # Ticker
    # -------------------------------------------------
    def get_ticker(self, symbol) -> ApiResponse:
        """獲取市場價格"""
        endpoint = f"/api/{API_VERSION}/ticker"
        params = {"symbol": symbol}
        response = self.make_request("GET", endpoint, params=params)
        print(response)
        ticker = self._parse_ticker(response, symbol)
        return ApiResponse(success=True, data=ticker)

    def _parse_ticker(self, raw: Dict, symbol: str) -> Optional[TickerInfo]:
        data = raw
        return TickerInfo(
            symbol=symbol,
            last_price=Decimal(data.get("lastPrice", "0")),
            volume_24h=Decimal(data.get("volume", "0")),
            change_24h=Decimal(data.get("priceChangePercent", "0")),
            timestamp=int(time.time() * 1000)
        )

    # -------------------------------------------------
    # Order Book
    # -------------------------------------------------
    def get_order_book(self, symbol, limit=20) -> ApiResponse:
        """獲取市場深度"""
        endpoint = f"/api/{API_VERSION}/depth"
        params = {"symbol": symbol}
        response = self.make_request("GET", endpoint, params=params)

        if "error" in response:
            return ApiResponse(success=False, error_message=response["error"])

        orderbook = self._parse_order_book(response, symbol, limit)
        return ApiResponse(success=True, data=orderbook)
    
    def _parse_order_book(self, raw: Dict, symbol: str, limit: int) -> OrderBookInfo:
        sorted_bids = raw.get("bids", [])
        sorted_bids.sort(key=lambda x: x[0], reverse=True)
        bids = [OrderBookLevel(price=Decimal(str(p)), quantity=Decimal(str(q)))
                for p, q in sorted_bids[:limit]]
        asks = [OrderBookLevel(price=Decimal(str(p)), quantity=Decimal(str(q)))
                for p, q in raw.get("asks", [])[:limit]]
        timestamp = raw.get("timestamp")
        return OrderBookInfo(symbol=symbol, bids=bids, asks=asks, timestamp=timestamp)

    # -------------------------------------------------
    # Market info
    # -------------------------------------------------
    def get_markets(self) -> ApiResponse:
        """獲取所有交易對信息"""
        endpoint = f"/api/{API_VERSION}/markets"
        response = self.make_request("GET", endpoint)
        if "error" in response:
            return ApiResponse(success=False, error_message=response["error"])
        markets = self._parse_markets(response)
        return ApiResponse(success=True, data=markets)

    def get_market_limits(self, symbol) -> ApiResponse:
        """獲取交易對的最低訂單量和價格精度"""
        markets_info_list = self.get_markets().data

        for markets_info in markets_info_list:
            if markets_info.symbol == symbol:
                return ApiResponse(success=True, data=markets_info)
            
        return ApiResponse(success=False, error_message=f"找不到交易對 {symbol} 的信息")

    def _parse_markets(self, raw: Dict) -> List[MarketInfo]:
        result = []
        for item in raw:
            base_asset = item.get('baseSymbol')
            quote_asset = item.get('quoteSymbol')
            market_type = item.get("marketType", "SPOT")
            status = item.get("visible", "ACTIVE")
            min_order_size = "0"  # 默認值
            tick_size = "0.00000001"  # 默認值
            base_precision = 8  # 默認值
            quote_precision = 8  # 默認值

            filters = item.get('filters', {})
            if 'price' in filters:
                tick_size = filters['price'].get('tickSize', '0.00000001')
                quote_precision = len(tick_size.split('.')[-1]) if '.' in tick_size else 0
            if 'quantity' in filters:
                min_order_size = filters['quantity'].get('minQuantity', '0')
                min_value = filters['quantity'].get('minQuantity', '0.00000001')
                base_precision = len(min_value.split('.')[-1]) if '.' in min_value else 0
            
            m = MarketInfo(
                symbol=item["symbol"],
                base_asset=base_asset,
                quote_asset=quote_asset,
                market_type=market_type,
                status=status,
                min_order_size=min_order_size,
                tick_size=tick_size,
                base_precision=base_precision,
                quote_precision=quote_precision
            )
            result.append(m)
        return result

    # -------------------------------------------------
    # Klines / Candles
    # -------------------------------------------------
    def get_klines(self, symbol, interval="1h", limit=100) -> ApiResponse:
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

        response = self.make_request("GET", endpoint, params=params)
        print(response)
        if "error" in response:
            return ApiResponse(success=False, error_message=response["error"])
        klines = self._parse_klines(response)
        return ApiResponse(success=True, data=klines)

    def _parse_klines(self, raw: Dict) -> List[KlineInfo]:
        result = []
        for k in raw:
            ki = KlineInfo(
                open_time=time_to_int(k["start"]),
                close_time=time_to_int(k["end"]),
                open_price=k["open"],
                high_price=k["high"],
                low_price=k["low"],
                close_price=k["close"],
                volume=k["volume"],
                quote_volume=k["quoteVolume"]
            )
            result.append(ki)
        return result

    # -------------------------------------------------
    # Orders
    # -------------------------------------------------
    def execute_order(self, order_details) -> ApiResponse:
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

        response = self.make_request("POST", endpoint, self.api_key, self.secret_key, instruction, params, order_details)
        if "error" in response:
            return ApiResponse(success=False, error_message=response["error"])
        result = self._parse_order_result(response)
        return ApiResponse(success=True, data=result)

    def _parse_order_result(self, raw: Dict) -> OrderResult:
        data = raw
        return OrderResult(
            success=True,
            order_id=data.get("id"),
            side=data.get("side"),
            size=Decimal(str(data.get("quantity", "0"))),
            price=Decimal(str(data.get("price", "0"))),
            error_message=None
        )

    def get_open_orders(self, symbol=None) -> ApiResponse:
        """獲取未成交訂單"""
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderQueryAll"
        params = {}
        if symbol:
            params["symbol"] = symbol
        response = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)
        if "error" in response:
            return ApiResponse(success=False, error_message=response["error"])
        orders = self._parse_open_orders(response)
        return ApiResponse(success=True, data=orders)

    def _parse_open_orders(self, raw: Dict) -> List[OrderInfo]:
        result = []
        for item in raw:
            size = Decimal(item.get("quantity", "0"))
            filled_size = Decimal(item.get("executedQuantity", "0"))
            remaining_size = size - filled_size
            o = OrderInfo(
                order_id=item["id"],
                side=item["side"],
                size=size,
                price=Decimal(item.get("price", "0")),
                status=item.get("status", "OPEN"),
                filled_size=filled_size,
                remaining_size=remaining_size
            )
            result.append(o)
        return result

    def cancel_all_orders(self, symbol) -> ApiResponse:
        """取消所有訂單"""
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderCancelAll"
        params = {"symbol": symbol}
        data = {"symbol": symbol}
        response =  self.make_request("DELETE", endpoint, self.api_key, self.secret_key, instruction, params, data)
        if "error" in response:
            return ApiResponse(success=False, error_message=response["error"])
        return ApiResponse(success=True, data={"symbol": symbol, "status": "ALL_CANCELLED"})

    def cancel_order(self, order_id, symbol) -> ApiResponse:
        """取消指定訂單"""
        endpoint = f"/api/{API_VERSION}/order"
        instruction = "orderCancel"
        params = {"orderId": order_id, "symbol": symbol}
        data = {"orderId": order_id, "symbol": symbol}
        response = self.make_request("DELETE", endpoint, self.api_key, self.secret_key, instruction, params, data)
        if "error" in response:
            return ApiResponse(success=False, error_message=response["error"])
        return ApiResponse(success=True, data={"order_id": order_id, "status": "CANCELLED"})

    # -------------------------------------------------
    # Positions
    # -------------------------------------------------
    def get_positions(self, symbol=None) -> ApiResponse:
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
                return ApiResponse(success=True, data=[])
        
        positions = self._parse_positions(result)
        return ApiResponse(success=True, data=positions)

    def _parse_positions(self, raw: Dict) -> List[PositionInfo]:
        result = []
        for pos in raw:
            p = PositionInfo(
                symbol=pos["symbol"],
                side=pos.get("side", "UNKNOWN"),
                size=Decimal(pos.get("netQuantity", "0")),
                entry_price=Decimal(pos.get("entryPrice", "0")),
                mark_price=Decimal(pos.get("markPrice", "0")),
                unrealized_pnl=Decimal(pos.get("pnlUnrealized", "0")),
                margin=Decimal(pos.get("mmf", "0"))
            )
            result.append(p)
        return result

    # -------------------------------------------------
    # Trade history
    # -------------------------------------------------
    def get_fill_history(self, symbol=None, limit=100) -> ApiResponse:
        """獲取歷史成交記錄"""
        endpoint = f"/wapi/{API_VERSION}/history/fills"
        instruction = "fillHistoryQueryAll"
        params = {"limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        response = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)
        if "error" in response:
            return ApiResponse(success=False, error_message=response["error"])
        print(response)
        trades = self._parse_fills(response)
        return ApiResponse(success=True, data=trades)

    def _parse_fills(self, raw: Dict) -> List[TradeInfo]:
        result = []
        for t in raw:
            timestamp = time_to_int(t.get("timestamp"), fmt="%Y-%m-%dT%H:%M:%S.%f")
            tr = TradeInfo(
                trade_id=str(t["clientId"]),
                order_id=str(t["orderId"]),
                symbol=t["symbol"],
                side=t["side"],
                size=Decimal(t["quantity"]),
                price=Decimal(t["price"]),
                fee=Decimal(t.get("fee", "0")),
                fee_asset=t.get("feeSymbol", ""),
                timestamp=timestamp,
                is_maker=bool(t.get("isMaker", False))
            )
            result.append(tr)
        return result
