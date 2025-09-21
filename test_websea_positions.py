#!/usr/bin/env python3
"""
测试 Websea 持仓获取功能
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.websea_client import WebseaClient
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_websea_positions():
    """测试 Websea 持仓获取"""
    try:
        # 构建 Websea 配置
        websea_config = {
            'ticker': 'SOL-USDT',  # 使用连字符格式
            'leverage': 10,
            'is_full': 2  # 全仓模式
        }
        
        # 创建 Websea 客户端
        client = WebseaClient(websea_config)
        
        # 获取配置的交易对
        symbol = websea_config.get('ticker', 'SOL-USDT')
        logger.info(f"测试获取 {symbol} 的持仓信息...")
        
        # 测试获取持仓
        positions = client.get_positions(symbol)
        logger.info(f"持仓信息: {positions}")
        
        if positions:
            for pos in positions:
                logger.info(f"交易对: {pos.get('symbol')}")
                logger.info(f"净持仓量: {pos.get('netQuantity')}")
                logger.info(f"方向: {pos.get('side')}")
                logger.info(f"仓位大小: {pos.get('size')}")
                logger.info(f"开仓价: {pos.get('entry_price')}")
                logger.info(f"标记价格: {pos.get('mark_price')}")
                logger.info(f"未实现盈亏: {pos.get('unrealized_pnl')}")
                logger.info(f"保证金: {pos.get('margin')}")
        else:
            logger.info("当前没有持仓")
            
        # 测试直接调用 API
        logger.info("\n直接测试 API 调用...")
        res = client._get("/openApi/contract/position", {"symbol": symbol})
        logger.info(f"API 返回原始数据: {res}")
        
        # 测试不带 symbol 参数
        logger.info("\n测试不带 symbol 参数...")
        res2 = client._get("/openApi/contract/position", {})
        logger.info(f"不带 symbol 的 API 返回: {res2}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_websea_positions())
