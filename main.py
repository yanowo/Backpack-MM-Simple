#!/usr/bin/env python
"""
Backpack Exchange 做市交易程序主執行文件
"""
import argparse
import sys
import os

from logger import setup_logger
from config import API_KEY, SECRET_KEY, WS_PROXY
from cli.commands import main_cli
from strategies.market_maker import MarketMaker
from strategies.perp_market_maker import PerpetualMarketMaker

logger = setup_logger("main")

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Backpack Exchange 做市交易程序')
    
    # 基本參數
    parser.add_argument('--api-key', type=str, help='API Key (可選，默認使用環境變數或配置文件)')
    parser.add_argument('--secret-key', type=str, help='Secret Key (可選，默認使用環境變數或配置文件)')
    parser.add_argument('--cli', action='store_true', help='啟動命令行界面')
    parser.add_argument('--ws-proxy', type=str, help='WebSocket Proxy (可選，默認使用環境變數或配置文件)')
    
    # 做市參數
    parser.add_argument('--symbol', type=str, help='交易對 (例如: SOL_USDC)')
    parser.add_argument('--spread', type=float, help='價差百分比 (例如: 0.5)')
    parser.add_argument('--quantity', type=float, help='訂單數量 (可選)')
    parser.add_argument('--max-orders', type=int, default=3, help='每側最大訂單數量 (默認: 3)')
    parser.add_argument('--duration', type=int, default=3600, help='運行時間（秒）(默認: 3600)')
    parser.add_argument('--interval', type=int, default=60, help='更新間隔（秒）(默認: 60)')
    parser.add_argument('--market-type', choices=['spot', 'perp'], default='spot', help='市場類型 (spot 或 perp)')
    parser.add_argument('--target-position', type=float, default=0.0, help='永續合約目標淨倉位')
    parser.add_argument('--max-position', type=float, default=1.0, help='永續合約最大允許倉位')
    parser.add_argument('--position-threshold', type=float, default=0.1, help='永續倉位調整觸發值')
    parser.add_argument('--inventory-skew', type=float, default=0.25, help='永續倉位偏移調整係數 (0-1)')

    return parser.parse_args()

def run_market_maker(args, api_key, secret_key, ws_proxy=None):
    """運行做市策略"""
    # 檢查必要參數
    if not args.symbol:
        logger.error("缺少交易對參數 (--symbol)")
        return
    
    if not args.spread and args.spread != 0:
        logger.error("缺少價差參數 (--spread)")
        return
    
    try:
        # 初始化做市商
        market_type = getattr(args, 'market_type', 'spot')

        if market_type == 'perp':
            market_maker = PerpetualMarketMaker(
                api_key=api_key,
                secret_key=secret_key,
                symbol=args.symbol,
                base_spread_percentage=args.spread,
                order_quantity=args.quantity,
                max_orders=args.max_orders,
                target_position=args.target_position,
                max_position=args.max_position,
                position_threshold=args.position_threshold,
                inventory_skew=args.inventory_skew,
                ws_proxy=ws_proxy
            )
        else:
            market_maker = MarketMaker(
                api_key=api_key,
                secret_key=secret_key,
                symbol=args.symbol,
                base_spread_percentage=args.spread,
                order_quantity=args.quantity,
                max_orders=args.max_orders,
                ws_proxy=ws_proxy
            )

        # 執行做市策略
        market_maker.run(duration_seconds=args.duration, interval_seconds=args.interval)

    except KeyboardInterrupt:
        logger.info("收到中斷信號，正在退出...")
    except Exception as e:
        logger.error(f"做市過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函數"""
    args = parse_arguments()
    
    # 優先使用命令行參數中的API密鑰
    api_key = args.api_key or API_KEY
    secret_key = args.secret_key or SECRET_KEY
    # 读取wss代理
    ws_proxy = args.ws_proxy or WS_PROXY
    
    # 檢查API密鑰
    if not api_key or not secret_key:
        logger.error("缺少API密鑰，請通過命令行參數或環境變量提供")
        sys.exit(1)
    
    # 決定執行模式
    if args.cli:
        # 啟動命令行界面
        main_cli(api_key, secret_key, ws_proxy=ws_proxy)
    elif args.symbol:
        # 如果指定了交易對，直接運行做市策略
        run_market_maker(args, api_key, secret_key, ws_proxy=ws_proxy)
    else:
        # 默認啟動命令行界面
        main_cli(api_key, secret_key, ws_proxy=ws_proxy)

if __name__ == "__main__":
    main()