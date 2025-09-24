"""
API请求客户端模块
"""
import json
import time
import requests
from typing import Dict, Any
from .auth import create_signature
from config import API_URL, API_VERSION, DEFAULT_WINDOW
from logger import setup_logger
from .base_client import BaseExchangeClient

logger = setup_logger("api.client")


class BPClient(BaseExchangeClient):
    """Backpack exchange client (REST).
    
    统一封装 API 请求、签名与重试逻辑。
    与早期函数式实现对齐（/api vs /wapi 端点与 instruction 名称），方便迁移。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")

    def get_exchange_name(self) -> str:
        return "Backpack"

    async def connect(self) -> None:
        logger.info("Backpack 客户端已连接")

    async def disconnect(self) -> None:
        logger.info("Backpack 客户端已断开连接")

    def make_request(self, method: str, endpoint: str, api_key=None, secret_key=None, instruction=None, 
                    params=None, data=None, retry_count=3) -> Dict:
        """
        执行API请求，支持重试机制
        
        Args:
            method: HTTP方法 (GET, POST, DELETE)
            endpoint: API端点
            api_key: API密钥
            secret_key: API密钥
            instruction: API指令
            params: 查询参数
            data: 请求体数据
            retry_count: 重试次数
            
        Returns:
            API响应数据
        """
        url = f"{API_URL}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'X-Broker-Id': '1500'
        }
        
        # 构建签名信息（如需要）
        if api_key and secret_key and instruction:
            timestamp = str(int(time.time() * 1000))
            window = DEFAULT_WINDOW
            
            # 构建签名消息
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
                return {"error": "签名创建失败"}
            
            headers.update({
                'X-API-KEY': api_key,
                'X-SIGNATURE': signature,
                'X-TIMESTAMP': timestamp,
                'X-WINDOW': window
            })
        
        # 添加查询参数到URL
        if params and method.upper() in ['GET', 'DELETE']:
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url += f"?{query_string}"
        
        # 实施重试机制
        for attempt in range(retry_count):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, headers=headers, timeout=10)
                elif method.upper() == 'POST':
                    response = requests.post(url, headers=headers, data=json.dumps(data) if data else None, timeout=10)
                elif method.upper() == 'DELETE':
                    response = requests.delete(url, headers=headers, data=json.dumps(data) if data else None, timeout=10)
                else:
                    return {"error": f"不支持的请求方法: {method}"}
                
                # 处理响应
                if response.status_code in [200, 201]:
                    return response.json() if response.text.strip() else {}
                elif response.status_code == 429:  # 速率限制
                    wait_time = 1 * (2 ** attempt)  # 指数退避
                    logger.warning(f"遇到速率限制，等待 {wait_time} 秒后重试")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"状态码: {response.status_code}, 消息: {response.text}"
                    if attempt < retry_count - 1:
                        logger.warning(f"请求失败 ({attempt+1}/{retry_count}): {error_msg}")
                        time.sleep(1)  # 简单重试延迟
                        continue
                    return {"error": error_msg}
            
            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    logger.warning(f"请求超时 ({attempt+1}/{retry_count})，重试中...")
                    continue
                return {"error": "请求超时"}
            except requests.exceptions.ConnectionError:
                if attempt < retry_count - 1:
                    logger.warning(f"连接错误 ({attempt+1}/{retry_count})，重试中...")
                    time.sleep(2)  # 连接错误通常需要更长等待
                    continue
                return {"error": "连接错误"}
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"请求异常 ({attempt+1}/{retry_count}): {str(e)}，重试中...")
                    continue
                return {"error": f"请求失败: {str(e)}"}
        
        return {"error": "达到最大重试次数"}

    # 各API端点函数
    def get_deposit_address(self, blockchain):
        """获取存款地址"""
        endpoint = f"/wapi/{API_VERSION}/capital/deposit/address"
        instruction = "depositAddressQuery"
        params = {"blockchain": blockchain}
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)

    def get_balance(self):
        """获取账户余额"""
        endpoint = f"/api/{API_VERSION}/capital"
        instruction = "balanceQuery"
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction)

    def get_collateral(self, subaccount_id=None):
        """获取抵押品资产"""
        endpoint = f"/api/{API_VERSION}/capital/collateral"
        params = {}
        if subaccount_id is not None:
            params["subaccountId"] = str(subaccount_id)
        instruction = "collateralQuery" if self.api_key and self.secret_key else None
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)

    def execute_order(self, order_details):
        """执行订单"""
        endpoint = f"/api/{API_VERSION}/order"
        instruction = "orderExecute"
        
        # 提取所有参数用于签名
        params = {
            "orderType": order_details["orderType"],
            "quantity": order_details["quantity"],
            "side": order_details["side"],
            "symbol": order_details["symbol"],
            "timeInForce": order_details.get("timeInForce", "GTC")
        }
        
        # 只有当订单包含价格时才添加 price 参数
        if "price" in order_details:
            params["price"] = order_details["price"]
        
        # 添加可选参数
        for key in ["postOnly", "reduceOnly", "clientId", "quoteQuantity", 
                    "autoBorrow", "autoLendRedeem", "autoBorrowRepay", "autoLend"]:
            if key in order_details:
                params[key] = str(order_details[key]).lower() if isinstance(order_details[key], bool) else str(order_details[key])

        return self.make_request("POST", endpoint, self.api_key, self.secret_key, instruction, params, order_details)

    def get_open_orders(self, symbol=None):
        """获取未成交订单"""
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderQueryAll"
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)

    def cancel_all_orders(self, symbol):
        """取消所有订单"""
        endpoint = f"/api/{API_VERSION}/orders"
        instruction = "orderCancelAll"
        params = {"symbol": symbol}
        data = {"symbol": symbol}
        return self.make_request("DELETE", endpoint, self.api_key, self.secret_key, instruction, params, data)

    def cancel_order(self, order_id, symbol):
        """取消指定订单"""
        endpoint = f"/api/{API_VERSION}/order"
        instruction = "orderCancel"
        params = {"orderId": order_id, "symbol": symbol}
        data = {"orderId": order_id, "symbol": symbol}
        return self.make_request("DELETE", endpoint, self.api_key, self.secret_key, instruction, params, data)

    def get_ticker(self, symbol):
        """获取市场价格"""
        endpoint = f"/api/{API_VERSION}/ticker"
        params = {"symbol": symbol}
        return self.make_request("GET", endpoint, params=params)

    def get_markets(self):
        """获取所有交易对信息"""
        endpoint = f"/api/{API_VERSION}/markets"
        return self.make_request("GET", endpoint)

    def get_order_book(self, symbol, limit=20):
        """获取市场深度"""
        endpoint = f"/api/{API_VERSION}/depth"
        params = {"symbol": symbol, "limit": str(limit)}
        return self.make_request("GET", endpoint, params=params)

    def get_fill_history(self, symbol=None, limit=100):
        """获取历史成交记录"""
        endpoint = f"/wapi/{API_VERSION}/history/fills"
        instruction = "fillHistoryQueryAll"
        params = {"limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        return self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params)

    def get_klines(self, symbol, interval="1h", limit=100):
        """获取K线数据"""
        endpoint = f"/api/{API_VERSION}/klines"
        
        # 计算起始时间 (秒)
        current_time = int(time.time())
        
        # 各间隔对应的秒数
        interval_seconds = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
            "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800, "1month": 2592000
        }
        
        # 计算合适的起始时间
        duration = interval_seconds.get(interval, 3600)
        start_time = current_time - (duration * limit)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": str(start_time)
        }

        return self.make_request("GET", endpoint, params=params)

    def get_market_limits(self, symbol):
        """获取交易对的最低订单量和价格精度"""
        markets_info = self.get_markets()

        if not isinstance(markets_info, dict) and isinstance(markets_info, list):
            for market_info in markets_info:
                if market_info.get('symbol') == symbol:
                    base_asset = market_info.get('baseSymbol')
                    quote_asset = market_info.get('quoteSymbol')
                    
                    # 从filters中获取精度和最小订单量信息
                    filters = market_info.get('filters', {})
                    base_precision = 8  # 默认值
                    quote_precision = 8  # 默认值
                    min_order_size = "0"  # 默认值
                    tick_size = "0.00000001"  # 默认值
                    
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
            
            logger.error(f"找不到交易对 {symbol} 的信息")
            return None
        else:
            logger.error(f"无法获取交易对信息: {markets_info}")
            return None

    def get_positions(self, symbol=None):
        """获取永续合约仓位"""
        endpoint = f"/api/{API_VERSION}/position"
        instruction = "positionQuery"
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        # 对于仓位查询，404是正常情况（表示没有仓位），所以只重试1次
        result = self.make_request("GET", endpoint, self.api_key, self.secret_key, instruction, params, retry_count=1)

        # 特殊处理404错误 - 对于仓位查询，404表示没有仓位，返回空列表
        if isinstance(result, dict) and "error" in result:
            error_msg = result["error"]
            if "404" in error_msg or "RESOURCE_NOT_FOUND" in error_msg:
                logger.debug("仓位查询返回404，表示没有活跃仓位")
                return []  # 返回空列表而不是错误
        
        return result