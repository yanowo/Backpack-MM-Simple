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
    """標準化訂單執行結果"""
    success: bool
    order_id: Optional[str] = None
    side: Optional[str] = None
    size: Optional[Decimal] = None
    price: Optional[Decimal] = None
    error_message: Optional[str] = None


@dataclass
class OrderInfo:
    """標準化訂單信息"""
    order_id: str
    side: str
    size: Decimal
    price: Decimal
    status: str
    filled_size: Decimal
    remaining_size: Decimal


@dataclass
class TickerInfo:
    """標準化行情信息"""
    symbol: str
    last_price: Decimal
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None
    timestamp: Optional[int] = None


@dataclass
class BalanceInfo:
    """標準化餘額信息"""
    asset: str
    available: Decimal
    locked: Decimal
    total: Decimal


@dataclass
class PositionInfo:
    """標準化持倉信息"""
    symbol: str
    side: str  # "LONG", "SHORT", "FLAT"
    size: Decimal
    entry_price: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    margin: Optional[Decimal] = None


@dataclass
class MarketInfo:
    """標準化市場信息"""
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
    """標準化訂單簿層級"""
    price: Decimal
    quantity: Decimal


@dataclass
class OrderBookInfo:
    """標準化訂單簿信息"""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: Optional[int] = None


@dataclass
class KlineInfo:
    """標準化K線信息"""
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
    """標準化交易信息"""
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


# 統一的響應格式
@dataclass
class ApiResponse:
    """標準化API響應格式"""
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
        self._order_update_callback = None
        # concrete clients may set up session/loggers

    # ---- lifecycle ----
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    def get_exchange_name(self) -> str: ...

    # ---- Order update callback management ----
    def set_order_update_callback(self, callback):
        """设置订单更新回调函数"""
        self._order_update_callback = callback

    def _handle_order_update(self, order_data):
        """处理订单更新的通用方法"""
        if self._order_update_callback:
            try:
                self._order_update_callback(order_data)
            except Exception as e:
                # 防止回调函数的错误影响主流程
                print(f"Error in order update callback: {e}")

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
        """獲取充值地址
        
        Returns:
            ApiResponse with data containing address info
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_balance(self) -> ApiResponse:
        """獲取賬户餘額
        
        Returns:
            ApiResponse with data: List[BalanceInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_collateral(self, subaccount_id: Optional[str] = None) -> ApiResponse:
        """獲取抵押品餘額
        
        Returns:
            ApiResponse with data: List[BalanceInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def execute_order(self, order_details: Dict[str, Any]) -> ApiResponse:
        """執行訂單
        
        Returns:
            ApiResponse with data: OrderResult
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_open_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取開放訂單
        
        Returns:
            ApiResponse with data: List[OrderInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def cancel_all_orders(self, symbol: str) -> ApiResponse:
        """取消所有訂單
        
        Returns:
            ApiResponse with data containing cancellation info
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def cancel_order(self, order_id: str, symbol: str) -> ApiResponse:
        """取消指定訂單
        
        Returns:
            ApiResponse with data containing cancellation info
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_ticker(self, symbol: str) -> ApiResponse:
        """獲取行情信息
        
        Returns:
            ApiResponse with data: TickerInfo
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_markets(self) -> ApiResponse:
        """獲取市場信息
        
        Returns:
            ApiResponse with data: List[MarketInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_order_book(self, symbol: str, limit: int = 20) -> ApiResponse:
        """獲取訂單簿
        
        Returns:
            ApiResponse with data: OrderBookInfo
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 100) -> ApiResponse:
        """獲取成交歷史
        
        Returns:
            ApiResponse with data: List[TradeInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> ApiResponse:
        """獲取K線數據
        
        Returns:
            ApiResponse with data: List[KlineInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_market_limits(self, symbol: str) -> ApiResponse:
        """獲取市場限制信息
        
        Returns:
            ApiResponse with data: MarketInfo
        """
        return ApiResponse(success=False, error_message="Not implemented")

    def get_positions(self, symbol: Optional[str] = None) -> ApiResponse:
        """獲取持倉信息
        
        Returns:
            ApiResponse with data: List[PositionInfo]
        """
        return ApiResponse(success=False, error_message="Not implemented")

    # ---- Utility methods for format conversion ----
    def _convert_to_standardized_response(self, raw_data: Any, success: bool = True, 
                                        error_message: Optional[str] = None) -> ApiResponse:
        """將原始響應轉換為標準化格式"""
        return ApiResponse(
            success=success,
            data=raw_data if success else None,
            error_message=error_message
        )
