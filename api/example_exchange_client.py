"""
示例交易所客户端实现模板

新交易所开发者只需要：
1. 继承 BaseExchangeClient
2. 实现必要的抽象方法
3. 按照标准格式返回数据
4. 无需关心兼容性问题
"""
from typing import Dict, Any, Optional
from decimal import Decimal
from .base_client import (
    BaseExchangeClient, ApiResponse, BalanceInfo, OrderResult, 
    OrderInfo, TickerInfo, MarketInfo, OrderBookInfo, OrderBookLevel,
    KlineInfo, TradeInfo, PositionInfo
)
from logger import setup_logger

logger = setup_logger("example_exchange")


class ExampleExchangeClient(BaseExchangeClient):
    """示例交易所客户端 - 展示如何快速实现新交易所"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.base_url = config.get("base_url", "https://api.example-exchange.com")

    def get_exchange_name(self) -> str:
        return "ExampleExchange"

    async def connect(self) -> None:
        logger.info("示例交易所客戶端已連接")

    async def disconnect(self) -> None:
        logger.info("示例交易所客戶端已斷開連接")

    def make_request(self, method: str, endpoint: str, api_key=None, secret_key=None,
                     instruction=None, params=None, data=None, retry_count: int = 3) -> Dict:
        """实现HTTP请求逻辑"""
        # TODO: 实现具体的HTTP请求逻辑
        # 这里只是示例
        return {"error": "Not implemented yet"}

    # =====================================================================
    # 以下是新交易所开发者需要实现的标准化方法
    # 直接返回标准格式，不需要考虑兼容性问题
    # =====================================================================

    def get_balance(self) -> ApiResponse:
        """获取账户余额 - 新交易所开发者只需要按这个格式返回即可"""
        try:
            # TODO: 调用交易所API获取余额
            raw_response = self.make_request("GET", "/api/v1/account")
            
            if "error" in raw_response:
                return ApiResponse(
                    success=False,
                    error_message=raw_response["error"]
                )
            
            # 转换为标准格式
            balances = []
            for item in raw_response.get("balances", []):
                balance = BalanceInfo(
                    asset=item["asset"],
                    available=Decimal(item["free"]),
                    locked=Decimal(item["locked"]),
                    total=Decimal(item["free"]) + Decimal(item["locked"])
                )
                balances.append(balance)
            
            return ApiResponse(success=True, data=balances)
            
        except Exception as e:
            return ApiResponse(success=False, error_message=str(e))

    def get_ticker(self, symbol: str) -> ApiResponse:
        """获取行情信息"""
        try:
            # TODO: 调用交易所API
            raw_response = self.make_request("GET", f"/api/v1/ticker/{symbol}")
            
            if "error" in raw_response:
                return ApiResponse(success=False, error_message=raw_response["error"])
            
            # 转换为标准格式
            ticker = TickerInfo(
                symbol=symbol,
                last_price=Decimal(raw_response["price"]),
                bid_price=Decimal(raw_response.get("bidPrice", "0")),
                ask_price=Decimal(raw_response.get("askPrice", "0")),
                volume_24h=Decimal(raw_response.get("volume", "0")),
                change_24h=Decimal(raw_response.get("priceChangePercent", "0")),
                timestamp=raw_response.get("timestamp")
            )
            
            return ApiResponse(success=True, data=ticker)
            
        except Exception as e:
            return ApiResponse(success=False, error_message=str(e))

    def get_order_book(self, symbol: str, limit: int = 20) -> ApiResponse:
        """获取订单簿"""
        try:
            # TODO: 调用交易所API
            raw_response = self.make_request("GET", f"/api/v1/depth", params={
                "symbol": symbol,
                "limit": limit
            })
            
            if "error" in raw_response:
                return ApiResponse(success=False, error_message=raw_response["error"])
            
            # 转换为标准格式
            bids = [
                OrderBookLevel(price=Decimal(price), quantity=Decimal(qty))
                for price, qty in raw_response.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=Decimal(price), quantity=Decimal(qty))
                for price, qty in raw_response.get("asks", [])
            ]
            
            order_book = OrderBookInfo(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=raw_response.get("timestamp")
            )
            
            return ApiResponse(success=True, data=order_book)
            
        except Exception as e:
            return ApiResponse(success=False, error_message=str(e))

    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        """执行订单"""
        try:
            # TODO: 调用交易所API
            raw_response = self.make_request("POST", "/api/v1/order", data=order_details)
            
            if "error" in raw_response:
                return ApiResponse(success=False, error_message=raw_response["error"])
            
            # 转换为标准格式
            order_result = OrderResult(
                success=True,
                order_id=raw_response.get("orderId"),
                side=raw_response.get("side"),
                size=Decimal(raw_response.get("quantity", "0")),
                price=Decimal(raw_response.get("price", "0"))
            )
            
            return ApiResponse(success=True, data=order_result)
            
        except Exception as e:
            return ApiResponse(success=False, error_message=str(e))

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        """获取持仓信息"""
        try:
            # TODO: 调用交易所API
            params = {"symbol": symbol} if symbol else {}
            raw_response = self.make_request("GET", "/api/v1/positions", params=params)
            
            if "error" in raw_response:
                return ApiResponse(success=False, error_message=raw_response["error"])
            
            # 转换为标准格式
            positions = []
            for item in raw_response.get("positions", []):
                position = PositionInfo(
                    symbol=item["symbol"],
                    side=item["side"],
                    size=Decimal(item["size"]),
                    entry_price=Decimal(item.get("entryPrice", "0")),
                    mark_price=Decimal(item.get("markPrice", "0")),
                    unrealized_pnl=Decimal(item.get("unrealizedPnl", "0")),
                    margin=Decimal(item.get("margin", "0"))
                )
                positions.append(position)
            
            return ApiResponse(success=True, data=positions)
            
        except Exception as e:
            return ApiResponse(success=False, error_message=str(e))

    # 其他方法按照相同模式实现...
    # get_markets(), get_open_orders(), cancel_order(), 等等


# =====================================================================
# 使用示例 - 展示新交易所如何被集成到现有系统中
# =====================================================================

def create_exchange_client(exchange_name: str, config: Dict[str, Any]) -> BaseExchangeClient:
    """工厂函数 - 创建指定交易所的客户端"""
    if exchange_name == "backpack":
        from .bp_client import BPClient
        return BPClient(config)
    elif exchange_name == "example":
        return ExampleExchangeClient(config)
    elif exchange_name == "binance":
        # from .binance_client import BinanceClient
        # return BinanceClient(config)
        raise NotImplementedError("Binance client not implemented yet")
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")

# 使用示例：
# config = {"api_key": "xxx", "secret_key": "yyy"}
# client = create_exchange_client("example", config)
# balance_response = client.get_balance()
# if balance_response.success:
#     for balance in balance_response.data:
#         print(f"{balance.asset}: {balance.available}")
