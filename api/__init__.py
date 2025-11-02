# api/__init__.py
"""
API 模块：与各交易所 API 通讯
"""

from .base_client import BaseExchangeClient
from .bp_client import BPClient
from .aster_client import AsterClient
from .lighter_client import LighterClient  # 轻量依赖，可安全顶层导入

__all__ = [
    "BaseExchangeClient",
    "BPClient",
    "AsterClient",
    "LighterClient",
    "get_client",
]

def get_client(name: str, *args, **kwargs):
    """按名称返回对应交易所客户端（懒加载 Paradex）"""
    name = (name or "").lower()
    if name == "lighter":
        return LighterClient(*args, **kwargs)
    elif name == "paradex":
        # 只有真的需要时才导入，避免在 lighter 路径拉起 starkware/cairo 旧依赖
        from .paradex_client import ParadexClient
        return ParadexClient(*args, **kwargs)
    elif name == "bp":
        return BPClient(*args, **kwargs)
    elif name == "aster":
        return AsterClient(*args, **kwargs)
    else:
        raise ValueError(f"未知交易所: {name}")

# 兼容：允许 from api import ParadexClient，但仍保持懒加载
def __getattr__(name: str):
    if name == "ParadexClient":
        from .paradex_client import ParadexClient
        return ParadexClient
    raise AttributeError(name)
