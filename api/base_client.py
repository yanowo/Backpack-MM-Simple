"""Abstract base exchange definitions.

This module isolates shared abstractions so concrete exchange clients (Backpack, xx, etc.)
can implement a consistent interface.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional, List, Union
import asyncio
from abc import ABC, abstractmethod
import functools


@dataclass
class OrderResult:
    """标准化订单执行结果"""
    success: bool
    order_id: Optional[str] = None
    side: Optional[str] = None
    size: Optional[Decimal] = None
    price: Optional[Decimal] = None
    error_message: Optional[str] = None


@dataclass
class OrderInfo:
    """标准化订单信息"""
    order_id: str
    side: str
    size: Decimal
    price: Decimal
    status: str
    filled_size: Decimal
    remaining_size: Decimal


@dataclass
class TickerInfo:
    """标准化行情信息"""
    symbol: str
    last_price: Decimal
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None
    timestamp: Optional[int] = None


@dataclass
class BalanceInfo:
    """标准化余额信息"""
    asset: str
    available: Decimal
    locked: Decimal
    total: Decimal


@dataclass
class PositionInfo:
    """标准化持仓信息"""
    symbol: str
    side: str  # "LONG", "SHORT", "FLAT"
    size: Decimal
    entry_price: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    margin: Optional[Decimal] = None


@dataclass
class MarketInfo:
    """标准化市场信息"""
    symbol: str
    base_asset: str
    quote_asset: str
    market_type: str  # "SPOT", "PERP", "FUTURE"
    status: str
    min_order_size: Decimal
    tick_size: Decimal
    base_precision: int
    quote_precision: int


@dataclass
class OrderBookLevel:
    """标准化订单簿层级"""
    price: Decimal
    quantity: Decimal


@dataclass
class OrderBookInfo:
    """标准化订单簿信息"""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: Optional[int] = None


@dataclass
class KlineInfo:
    """标准化K线信息"""
    open_time: int
    close_time: int
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    quote_volume: Optional[Decimal] = None


@dataclass
class TradeInfo:
    """标准化交易信息"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    size: Decimal
    price: Decimal
    fee: Decimal
    fee_asset: str
    timestamp: int
    is_maker: bool


# 统一的响应格式
@dataclass
class ApiResponse:
    """标准化API响应格式"""
    success: bool
    data: Optional[Any] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


def query_retry(max_retries: int = 3, delay: float = 1.0, default_return=None):
    """Coroutine-friendly retry decorator for async query functions.

    If the wrapped coroutine raises an exception it will retry up to max_retries times
    with a fixed delay (seconds). After exhausting retries, returns default_return.
    """
    def wrapper(func):
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("@query_retry can only decorate async functions")

        @functools.wraps(func)
        async def inner(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:  # noqa
                    last_exc = e
                    if attempt == max_retries:
                        return default_return
                    await asyncio.sleep(delay)
            return default_return
        return inner
    return wrapper


class BaseExchangeClient(ABC):
    """Abstract base class for all exchange clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        # concrete clients may set up session/loggers

    # ---- lifecycle ----
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    def get_exchange_name(self) -> str: ...

    # ---- HTTP / request layer (sync for simplicity) ----
    @abstractmethod
    def make_request(self, method: str, endpoint: str, api_key=None, secret_key=None,
                     instruction=None, params=None, data=None, retry_count: int = 3) -> Dict:
        """Perform HTTP request and return parsed JSON dict or {'error': ...}."""
        ...

    # Optional: signature helper (default no-op)
    def _create_signature(self, secret_key: str, message: str) -> str:
        raise NotImplementedError("Signature method not implemented for this exchange")

    # ---- Generic high-level methods with standardized return types ----
    def get_deposit_address(self, blockchain: str) -> ApiResponse:
        """获取充值地址
        
        Returns:
            ApiResponse with data containing address info
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_balance(self) -> ApiResponse:
        """获取账户余额
        
        Returns:
            ApiResponse with data: List[BalanceInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_collateral(self, subaccount_id: Optional[str] = None) -> ApiResponse:
        """获取抵押品余额
        
        Returns:
            ApiResponse with data: List[BalanceInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        """执行订单
        
        Returns:
            ApiResponse with data: OrderResult
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_open_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        """获取开放订单
        
        Returns:
            ApiResponse with data: List[OrderInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def cancel_all_orders(self, symbol: str) -> ApiResponse:
        """取消所有订单
        
        Returns:
            ApiResponse with data containing cancellation info
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def cancel_order(self, order_id: str, symbol: str) -> ApiResponse:
        """取消指定订单
        
        Returns:
            ApiResponse with data containing cancellation info
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_ticker(self, symbol: str) -> ApiResponse:
        """获取行情信息
        
        Returns:
            ApiResponse with data: TickerInfo
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_markets(self) -> ApiResponse:
        """获取市场信息
        
        Returns:
            ApiResponse with data: List[MarketInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_order_book(self, symbol: str, limit: int = 20) -> ApiResponse:
        """获取订单簿
        
        Returns:
            ApiResponse with data: OrderBookInfo
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100) -> ApiResponse:
        """获取成交历史
        
        Returns:
            ApiResponse with data: List[TradeInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> ApiResponse:
        """获取K线数据
        
        Returns:
            ApiResponse with data: List[KlineInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_market_limits(self, symbol: str) -> ApiResponse:
        """获取市场限制信息
        
        Returns:
            ApiResponse with data: MarketInfo
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        """获取持仓信息
        
        Returns:
            ApiResponse with data: List[PositionInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    # ---- Utility methods for format conversion ----
    def _convert_to_standardized_response(self, raw_data: Any, success: bool = True, 
                                        error_message: Optional[str] = None) -> ApiResponse:
        """将原始响应转换为标准化格式"""
        return ApiResponse(
            success=success,
            data=raw_data if success else None,
            error_message=error_message
        )
