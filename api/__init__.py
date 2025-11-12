# api/__init__.py
"""
API 模塊：與各交易所 API 通訊
"""

from .base_client import BaseExchangeClient
from .bp_client import BPClient
from .aster_client import AsterClient
from .lighter_client import LighterClient  # 輕量依賴，可安全頂層導入

__all__ = [
    "BaseExchangeClient",
    "BPClient",
    "AsterClient",
    "LighterClient",
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
    else:
        raise ValueError(f"未知交易所: {name}")

# 兼容：允許 from api import ParadexClient，但仍保持懶加載
def __getattr__(name: str):
    if name == "ParadexClient":
        from .paradex_client import ParadexClient
        return ParadexClient
    raise AttributeError(name)
