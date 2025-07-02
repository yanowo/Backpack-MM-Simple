"""
輔助函數模塊
"""
import math
import numpy as np
from typing import List, Union, Optional

def round_to_precision(value: float, precision: int) -> float:
    """
    根據精度四捨五入數字
    
    Args:
        value: 要四捨五入的數值
        precision: 小數點精度
        
    Returns:
        四捨五入後的數值
    """
    factor = 10 ** precision
    return math.floor(value * factor) / factor

def round_to_tick_size(price: float, tick_size: float) -> float:
    """
    根據tick_size四捨五入價格
    
    Args:
        price: 原始價格
        tick_size: 價格步長
        
    Returns:
        調整後的價格
    """
    tick_size_float = float(tick_size)
    tick_size_str = format(tick_size_float, 'f').rstrip('0')
    precision = len(tick_size_str.split('.')[-1]) if '.' in tick_size_str else 0
    rounded_price = round(price / tick_size_float) * tick_size_float
    return round(rounded_price, precision)

def calculate_volatility(prices: List[float], window: int = 20) -> float:
    """
    計算波動率
    
    Args:
        prices: 價格列表
        window: 計算窗口大小
        
    Returns:
        波動率百分比
    """
    if len(prices) < window:
        return 0
    
    # 使用最近N個價格計算標準差
    recent_prices = prices[-window:]
    returns = np.diff(recent_prices) / recent_prices[:-1]
    return np.std(returns) * 100  # 轉換為百分比