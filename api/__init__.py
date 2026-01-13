# api/__init__.py
"""
API 模塊：與各交易所 API 通訊

標準化 API 響應使用 ApiResponse dataclass，所有方法返回統一格式：
- response.success: bool - 是否成功
- response.data: Any - 成功時的數據 (各種 dataclass)
- response.error_message: Optional[str] - 失敗時的錯誤訊息
- response.raw: Optional[Dict] - 原始 API 響應

使用範例：
    response = client.get_balance()
    if response.success:
        for balance in response.data:
            print(f"{balance.asset}: {balance.available}")
    else:
        print(f"Error: {response.error_message}")
"""

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
    CancelResult,
    BatchOrderResult,
    safe_float,
    safe_decimal,
    safe_int,
)
from .bp_client import BPClient
from .aster_client import AsterClient
from .lighter_client import LighterClient  # 輕量依賴，可安全頂層導入
from .apex_client import ApexClient

__all__ = [
    # 基礎類別
    "BaseExchangeClient",
    # 交易所客戶端
    "BPClient",
    "AsterClient",
    "LighterClient",
    "ApexClient",
    # 標準化響應
    "ApiResponse",
    # 訂單相關
    "OrderResult",
    "OrderInfo",
    "CancelResult",
    "BatchOrderResult",
    # 帳戶相關
    "BalanceInfo",
    "CollateralInfo",
    "PositionInfo",
    # 市場數據
    "MarketInfo",
    "TickerInfo",
    "OrderBookInfo",
    "OrderBookLevel",
    "KlineInfo",
    "TradeInfo",
    # 工具函數
    "safe_float",
    "safe_decimal",
    "safe_int",
    # 工廠函數
    "get_client",
]

def get_client(name: str, *args, **kwargs):
    """按名稱返回對應交易所客户端（懶加載 Paradex）"""
    name = (name or "").lower()
    if name == "lighter":
        return LighterClient(*args, **kwargs)
    elif name == "paradex":
        # 只有真的需要時才導入，避免在 lighter 路徑拉起 starkware/cairo 舊依賴
        from .paradex_client import ParadexClient
        return ParadexClient(*args, **kwargs)
    elif name == "bp":
        return BPClient(*args, **kwargs)
    elif name == "aster":
        return AsterClient(*args, **kwargs)
    elif name == "apex":
        return ApexClient(*args, **kwargs)
    else:
        raise ValueError(f"未知交易所: {name}")

# 兼容：允許 from api import ParadexClient，但仍保持懶加載
def __getattr__(name: str):
    if name == "ParadexClient":
        from .paradex_client import ParadexClient
        return ParadexClient
    raise AttributeError(name)
