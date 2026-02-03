# ws_client/__init__.py
"""
WebSocket 模塊，負責實時數據處理

支持多交易所的 WebSocket 連接，提供統一的抽象接口。

使用方式：

1. 直接實例化特定交易所客戶端:
    >>> from ws_client import BackpackWebSocket
    >>> ws = BackpackWebSocket(symbol="BTC_USDC")
    >>> ws.connect()

2. 為新交易所實現客戶端:
    - 參考 example_ws_client.py 模板
    - 繼承 BaseWebSocketClient
    - 實現所有抽象方法

支持的交易所:
    - BackpackWebSocket (Backpack Exchange)
"""

# 基類和數據結構
from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSTickerData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSFillData,
)

# 具體實現
from .backpack_ws_client import BackpackWebSocket
from .aster_ws_client import AsterWebSocket
from .paradex_ws_client import ParadexWebSocket
from .lighter_ws_client import LighterWebSocket
from .apex_ws_client import ApexWebSocket

__all__ = [
    # 基類
    "BaseWebSocketClient",
    "WSConnectionConfig",
    # 數據結構
    "WSTickerData",
    "WSOrderBookData",
    "WSOrderUpdateData",
    "WSFillData",
    # 具體實現
    "BackpackWebSocket",
    "AsterWebSocket",
    "ParadexWebSocket",
    "LighterWebSocket",
    "ApexWebSocket",
]
