"""
API請求客戶端模塊
"""
import json
import time
import requests
from typing import Dict, Any, Optional, List, Union
from .auth import create_signature
from config import API_URL, API_VERSION, DEFAULT_WINDOW
from logger import setup_logger

logger = setup_logger("api.client")

def make_request(method: str, endpoint: str, api_key=None, secret_key=None, instruction=None, 
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
    headers = {'Content-Type': 'application/json'}
    
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
def get_deposit_address(api_key, secret_key, blockchain):
    """獲取存款地址"""
    endpoint = f"/wapi/{API_VERSION}/capital/deposit/address"
    instruction = "depositAddressQuery"
    params = {"blockchain": blockchain}
    return make_request("GET", endpoint, api_key, secret_key, instruction, params)

def get_balance(api_key, secret_key):
    """獲取賬戶餘額"""
    endpoint = f"/api/{API_VERSION}/capital"
    instruction = "balanceQuery"
    return make_request("GET", endpoint, api_key, secret_key, instruction)

def execute_order(api_key, secret_key, order_details):
    """執行訂單"""
    endpoint = f"/api/{API_VERSION}/order"
    instruction = "orderExecute"
    
    # 提取所有參數用於簽名
    params = {
        "orderType": order_details["orderType"],
        "price": order_details.get("price", "0"),
        "quantity": order_details["quantity"],
        "side": order_details["side"],
        "symbol": order_details["symbol"],
        "timeInForce": order_details.get("timeInForce", "GTC")
    }
    
    # 添加可選參數
    for key in ["postOnly", "reduceOnly", "clientId", "quoteQuantity", 
                "autoBorrow", "autoLendRedeem", "autoBorrowRepay", "autoLend"]:
        if key in order_details:
            params[key] = str(order_details[key]).lower() if isinstance(order_details[key], bool) else str(order_details[key])
    
    return make_request("POST", endpoint, api_key, secret_key, instruction, params, order_details)

def get_open_orders(api_key, secret_key, symbol=None):
    """獲取未成交訂單"""
    endpoint = f"/api/{API_VERSION}/orders"
    instruction = "orderQueryAll"
    params = {}
    if symbol:
        params["symbol"] = symbol
    return make_request("GET", endpoint, api_key, secret_key, instruction, params)

def cancel_all_orders(api_key, secret_key, symbol):
    """取消所有訂單"""
    endpoint = f"/api/{API_VERSION}/orders"
    instruction = "orderCancelAll"
    params = {"symbol": symbol}
    data = {"symbol": symbol}
    return make_request("DELETE", endpoint, api_key, secret_key, instruction, params, data)

def cancel_order(api_key, secret_key, order_id, symbol):
    """取消指定訂單"""
    endpoint = f"/api/{API_VERSION}/order"
    instruction = "orderCancel"
    params = {"orderId": order_id, "symbol": symbol}
    data = {"orderId": order_id, "symbol": symbol}
    return make_request("DELETE", endpoint, api_key, secret_key, instruction, params, data)

def get_ticker(symbol):
    """獲取市場價格"""
    endpoint = f"/api/{API_VERSION}/ticker"
    params = {"symbol": symbol}
    return make_request("GET", endpoint, params=params)

def get_markets():
    """獲取所有交易對信息"""
    endpoint = f"/api/{API_VERSION}/markets"
    return make_request("GET", endpoint)

def get_order_book(symbol, limit=20):
    """獲取市場深度"""
    endpoint = f"/api/{API_VERSION}/depth"
    params = {"symbol": symbol, "limit": str(limit)}
    return make_request("GET", endpoint, params=params)

def get_fill_history(api_key, secret_key, symbol=None, limit=100):
    """獲取歷史成交記錄"""
    endpoint = f"/wapi/{API_VERSION}/history/fills"
    instruction = "fillHistoryQueryAll"
    params = {"limit": str(limit)}
    if symbol:
        params["symbol"] = symbol
    return make_request("GET", endpoint, api_key, secret_key, instruction, params)

def get_klines(symbol, interval="1h", limit=100):
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
    
    return make_request("GET", endpoint, params=params)

def get_market_limits(symbol):
    """獲取交易對的最低訂單量和價格精度"""
    markets_info = get_markets()
    
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