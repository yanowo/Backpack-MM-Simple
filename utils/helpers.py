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
        value: 要向下取整的數值
        precision: 小數點精度
        
    Returns:
        向下取整的數值
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
    if tick_size <= 0:
        return price

    tick_size_float = float(tick_size)

    # 計算應該四捨五入到的倍數
    rounded_price = round(price / tick_size_float) * tick_size_float

    # 計算 tick_size 的小數位數（更精確的方法）
    # 使用 Decimal 的方式來避免浮點數精度問題
    tick_size_str = f"{tick_size_float:.10f}".rstrip('0').rstrip('.')

    if '.' in tick_size_str:
        precision = len(tick_size_str.split('.')[1])
    else:
        precision = 0

    # 使用計算出的精度進行最終四捨五入
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