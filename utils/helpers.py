"""
辅助函数模块
"""
import math
import numpy as np
from typing import List, Union, Optional

def round_to_precision(value: float, precision: int) -> float:
    """
    根据精度四舍五入数字
    
    Args:
        value: 要向下取整的数值
        precision: 小数点精度
        
    Returns:
        向下取整的数值
    """
    factor = 10 ** precision
    return math.floor(value * factor) / factor

def round_to_tick_size(price: float, tick_size: float) -> float:
    """
    根据tick_size四舍五入价格
    
    Args:
        price: 原始价格
        tick_size: 价格步长
        
    Returns:
        调整后的价格
    """
    tick_size_float = float(tick_size)
    tick_size_str = format(tick_size_float, 'f').rstrip('0')
    precision = len(tick_size_str.split('.')[-1]) if '.' in tick_size_str else 0
    rounded_price = round(price / tick_size_float) * tick_size_float
    return round(rounded_price, precision)

def calculate_volatility(prices: List[float], window: int = 20) -> float:
    """
    计算波动率
    
    Args:
        prices: 价格列表
        window: 计算窗口大小
        
    Returns:
        波动率百分比
    """
    if len(prices) < window:
        return 0
    
    # 使用最近N个价格计算标准差
    recent_prices = prices[-window:]
    returns = np.diff(recent_prices) / recent_prices[:-1]
    return np.std(returns) * 100  # 转换为百分比