"""Abstract base exchange definitions.

This module isolates shared abstractions so concrete exchange clients (Backpack, xx, etc.)
can implement a consistent interface.

標準化返回格式說明：
====================
所有交易所客戶端的公開方法統一返回 ApiResponse 對象，包含：
- success: bool - 請求是否成功
- data: Optional[Any] - 成功時的數據（使用標準化的 dataclass）
- error_code: Optional[str] - 錯誤碼
- error_message: Optional[str] - 錯誤信息

使用範例：
---------
response = client.get_balance()
if response.success:
    for balance in response.data:  # List[BalanceInfo]
        print(f"{balance.asset}: {balance.available}")
else:
    print(f"錯誤: {response.error_message}")
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, List, Union
import asyncio
from abc import ABC, abstractmethod
import functools


# ==================== 標準化數據類 ====================

@dataclass
class OrderResult:
    """標準化訂單執行結果"""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    order_type: Optional[str] = None
    size: Optional[Decimal] = None
    price: Optional[Decimal] = None
    filled_size: Optional[Decimal] = None
    status: Optional[str] = None
    created_at: Optional[int] = None
    error_message: Optional[str] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class OrderInfo:
    """標準化訂單信息"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    size: Decimal
    price: Optional[Decimal]
    status: str
    filled_size: Decimal
    remaining_size: Decimal
    client_order_id: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    time_in_force: Optional[str] = None
    post_only: bool = False
    reduce_only: bool = False
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class TickerInfo:
    """標準化行情信息"""
    symbol: str
    last_price: Optional[Decimal] = None
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None
    index_price: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    turnover_24h: Optional[Decimal] = None
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None
    change_percent_24h: Optional[Decimal] = None
    open_interest: Optional[Decimal] = None
    funding_rate: Optional[Decimal] = None
    next_funding_time: Optional[int] = None
    timestamp: Optional[int] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class BalanceInfo:
    """標準化餘額信息"""
    asset: str
    available: Decimal
    locked: Decimal
    total: Decimal
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class CollateralInfo:
    """標準化抵押品信息（永續合約）"""
    asset: str
    total_collateral: Decimal
    free_collateral: Decimal
    initial_margin: Optional[Decimal] = None
    maintenance_margin: Optional[Decimal] = None
    account_value: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class PositionInfo:
    """標準化持倉信息"""
    symbol: str
    side: str  # "LONG", "SHORT", "FLAT"
    size: Decimal
    entry_price: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None
    liquidation_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    margin: Optional[Decimal] = None
    leverage: Optional[Decimal] = None
    margin_mode: Optional[str] = None  # "CROSS", "ISOLATED"
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class MarketInfo:
    """標準化市場信息"""
    symbol: str
    base_asset: str
    quote_asset: str
    market_type: str  # "SPOT", "PERP", "FUTURE"
    status: str
    min_order_size: Decimal
    max_order_size: Optional[Decimal] = None
    tick_size: Decimal = Decimal("0.00000001")
    step_size: Optional[Decimal] = None
    base_precision: int = 8
    quote_precision: int = 8
    min_notional: Optional[Decimal] = None
    maker_fee: Optional[Decimal] = None
    taker_fee: Optional[Decimal] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


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
    sequence: Optional[int] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """獲取最佳買價"""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """獲取最佳賣價"""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """獲取中間價"""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """獲取價差"""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None


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
    trades_count: Optional[int] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class TradeInfo:
    """標準化成交信息"""
    trade_id: str
    order_id: Optional[str]
    symbol: str
    side: str
    size: Decimal
    price: Decimal
    fee: Optional[Decimal] = None
    fee_asset: Optional[str] = None
    timestamp: Optional[int] = None
    is_maker: Optional[bool] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class DepositAddressInfo:
    """標準化充值地址信息"""
    address: str
    blockchain: str
    tag: Optional[str] = None  # memo/tag for some chains
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class CancelResult:
    """標準化取消訂單結果"""
    success: bool
    order_id: Optional[str] = None
    cancelled_count: int = 0
    error_message: Optional[str] = None
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class BatchOrderResult:
    """標準化批量訂單結果"""
    success: bool
    orders: List[OrderResult] = field(default_factory=list)
    failed_count: int = 0
    errors: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)


# ==================== 統一響應格式 ====================

@dataclass
class ApiResponse:
    """標準化API響應格式
    
    所有交易所客戶端方法統一返回此格式。
    
    Attributes:
        success: 請求是否成功
        data: 成功時的數據（標準化 dataclass 或其列表）
        error_code: 錯誤碼（可選）
        error_message: 錯誤信息（可選）
        raw: 原始響應數據（用於調試）
    
    Example:
        # 檢查響應
        response = client.get_balance()
        if response.success:
            balances = response.data  # List[BalanceInfo]
        else:
            logger.error(f"Error: {response.error_message}")
    """
    success: bool
    data: Optional[Any] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw: Optional[Any] = field(default=None, repr=False)

    @classmethod
    def ok(cls, data: Any, raw: Any = None) -> "ApiResponse":
        """創建成功響應"""
        return cls(success=True, data=data, raw=raw)

    @classmethod
    def error(cls, message: str, code: Optional[str] = None, raw: Any = None) -> "ApiResponse":
        """創建錯誤響應"""
        return cls(success=False, error_message=message, error_code=code, raw=raw)


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

    @staticmethod
    def is_error_response(response: Any) -> bool:
        """檢查 API 響應是否為錯誤

        統一的錯誤檢測方法，支持所有交易所的響應格式。

        Args:
            response: API 響應數據

        Returns:
            bool: 如果是錯誤響應返回 True
        """
        if response is None:
            return True
        if isinstance(response, dict):
            return "error" in response
        return False

    @staticmethod
    def get_error_message(response: Any) -> Optional[str]:
        """從 API 響應中提取錯誤信息

        Args:
            response: API 響應數據

        Returns:
            錯誤信息字符串，如果沒有錯誤則返回 None
        """
        if response is None:
            return "Empty response"
        if isinstance(response, dict) and "error" in response:
            return str(response["error"])
        return None

    @staticmethod
    def extract_field(response: Any, *keys: str, default: Any = None) -> Any:
        """從 API 響應中安全提取字段

        支持多個可能的字段名稱，用於處理不同交易所的字段命名差異。

        Args:
            response: API 響應數據
            *keys: 可能的字段名稱列表
            default: 默認值

        Returns:
            提取的字段值，如果未找到則返回默認值

        Example:
            # 從 ticker 響應中提取最新價格
            price = client.extract_field(ticker, "lastPrice", "last_price", "price", default=0.0)
        """
        if not isinstance(response, dict):
            return default

        # 檢查頂層
        for key in keys:
            if key in response and response[key] not in (None, ""):
                return response[key]

        # 檢查 data 節點（部分交易所會包裝在 data 中）
        data = response.get("data")
        if isinstance(data, dict):
            for key in keys:
                if key in data and data[key] not in (None, ""):
                    return data[key]

        return default

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """安全地將值轉換為浮點數

        Args:
            value: 要轉換的值
            default: 轉換失敗時的默認值

        Returns:
            轉換後的浮點數
        """
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
        """安全地將值轉換為 Decimal

        Args:
            value: 要轉換的值
            default: 轉換失敗時的默認值

        Returns:
            轉換後的 Decimal 或默認值
        """
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except Exception:
            return default

    @staticmethod
    def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        """安全地將值轉換為整數

        Args:
            value: 要轉換的值
            default: 轉換失敗時的默認值

        Returns:
            轉換後的整數或默認值
        """
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _parse_raw_to_error(self, raw: Any) -> ApiResponse:
        """從原始響應解析錯誤"""
        if raw is None:
            return ApiResponse.error("Empty response", raw=raw)
        if isinstance(raw, dict) and "error" in raw:
            return ApiResponse.error(str(raw["error"]), raw=raw)
        return ApiResponse.error("Unknown error", raw=raw)

    def _check_raw_error(self, raw: Any) -> Optional[ApiResponse]:
        """檢查原始響應是否包含錯誤，如果有則返回錯誤響應，否則返回 None"""
        if self.is_error_response(raw):
            return self._parse_raw_to_error(raw)
        return None


# 模組級別的工具函數（從 BaseExchangeClient 導出）
safe_float = BaseExchangeClient.safe_float
safe_decimal = BaseExchangeClient.safe_decimal
safe_int = BaseExchangeClient.safe_int
