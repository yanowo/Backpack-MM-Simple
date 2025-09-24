#!/usr/bin/env python
"""
Backpack Exchange 做市交易程序统一入口
支持命令行模式和面板模式
"""
import argparse
import sys
import os
from logger import setup_logger

# 创建记录器
logger = setup_logger("main")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Backpack Exchange 做市交易程序')
    
    # 模式选择
    parser.add_argument('--panel', action='store_true', help='启动图形界面面板')
    parser.add_argument('--cli', action='store_true', help='启动命令行界面')
    
    # 基本参数
    parser.add_argument('--exchange', type=str, choices=['backpack', 'websea', 'aster'], default='backpack', help='交易所選擇 (backpack, websea 或 aster)')
    parser.add_argument('--api-key', type=str, help='API Key (可选，默认使用环境变数或配置文件)')
    parser.add_argument('--secret-key', type=str, help='Secret Key (可选，默认使用环境变数或配置文件)')
    parser.add_argument('--ws-proxy', type=str, help='WebSocket Proxy (可选，默认使用环境变数或配置文件)')
    
    # 做市参数
    parser.add_argument('--symbol', type=str, help='交易对 (例如: SOL_USDC)')
    parser.add_argument('--spread', type=float, help='价差百分比 (例如: 0.5)')
    parser.add_argument('--quantity', type=float, help='订单数量 (可选)')
    parser.add_argument('--max-orders', type=int, default=3, help='每侧最大订单数量 (默认: 3)')
    parser.add_argument('--duration', type=int, default=3600, help='运行时间（秒）(默认: 3600)')
    parser.add_argument('--interval', type=int, default=60, help='更新间隔（秒）(默认: 60)')
    parser.add_argument('--market-type', choices=['spot', 'perp'], default='spot', help='市场类型 (spot 或 perp)')
    parser.add_argument('--target-position', type=float, default=1.0, help='永续合约目标持仓量 (绝对值, 例如: 1.0)')
    parser.add_argument('--max-position', type=float, default=1.0, help='永续合约最大允许仓位(绝对值)')
    parser.add_argument('--position-threshold', type=float, default=0.1, help='永续仓位调整触发值')
    parser.add_argument('--inventory-skew', type=float, default=0.0, help='永续仓位偏移调整系数 (0-1)')
    
    # 重平设置参数
    parser.add_argument('--enable-rebalance', action='store_true', help='开启重平功能')
    parser.add_argument('--disable-rebalance', action='store_true', help='关闭重平功能')
    parser.add_argument('--base-asset-target', type=float, help='基础资产目标比例 (0-100, 默认: 30)')
    parser.add_argument('--rebalance-threshold', type=float, help='重平触发阈值 (>0, 默认: 15)')

    return parser.parse_args()

def validate_rebalance_args(args):
    """验证重平设置参数"""
    if getattr(args, 'market_type', 'spot') == 'perp':
        return
    if args.enable_rebalance and args.disable_rebalance:
        logger.error("不能同时设置 --enable-rebalance 和 --disable-rebalance")
        sys.exit(1)
    
    if args.base_asset_target is not None:
        if not 0 <= args.base_asset_target <= 100:
            logger.error("基础资产目标比例必须在 0-100 之间")
            sys.exit(1)
    
    if args.rebalance_threshold is not None:
        if args.rebalance_threshold <= 0:
            logger.error("重平触发阈值必须大于 0")
            sys.exit(1)

def main():
    """主函数"""
    args = parse_arguments()
    
    # 验证重平参数
    validate_rebalance_args(args)
    
    exchange = args.exchange
    if exchange == 'backpack':
        api_key = os.getenv('BACKPACK_KEY')
        secret_key = os.getenv('BACKPACK_SECRET')
        ws_proxy = os.getenv('BACKPACK_PROXY_WEBSOCKET')
        base_url = os.getenv('BASE_URL', 'https://api.backpack.work')
        exchange_config = {
            'api_key': api_key,
            'secret_key': secret_key,
            'base_url': base_url,
            'api_version': 'v1',
            'default_window': '5000'
        }
    elif exchange == 'websea':
        api_key = os.getenv('WEBSEA_TOKEN')
        secret_key = os.getenv('WEBSEA_SECRET')
        ws_proxy = os.getenv('WEBSEA_PROXY_WEBSOCKET')  # 添加 ws_proxy 定义
        base_url = os.getenv('WEBSEA_BASE_URL', 'https://coapi.websea.com')
        exchange_config = {
            'ticker': args.symbol,
            'leverage': 10,
            'is_full': 2  # 全仓模式
        }
    elif exchange == 'aster':
        api_key = os.getenv('ASTER_API_KEY')
        secret_key = os.getenv('ASTER_SECRET_KEY')
        ws_proxy = os.getenv('ASTER_PROXY_WEBSOCKET')
        exchange_config = {
            'api_key': api_key,
            'secret_key': secret_key,
        }
    else:
        logger.error("不支持的交易所，請選擇 'backpack', 'websea' 或 'aster'")
        sys.exit(1)

    
    # 检查API密钥
    if not api_key or not secret_key:
        logger.error("缺少API密钥，请通过命令行参数或环境变量提供")
        sys.exit(1)
    
    # 决定执行模式
    if args.panel:
        # 启动图形界面面板
        try:
            from panel.panel_main import run_panel
            run_panel(api_key=api_key, secret_key=secret_key, default_symbol=args.symbol)
        except ImportError as e:
            logger.error(f"启动面板时出错，缺少必要的库: {str(e)}")
            logger.error("请执行 pip install rich 安装所需库")
            sys.exit(1)
    elif args.cli:
        # 启动命令行界面
        try:
            from cli.commands import main_cli
            main_cli(api_key, secret_key, ws_proxy=ws_proxy)
        except ImportError as e:
            logger.error(f"启动命令行界面时出错: {str(e)}")
            sys.exit(1)
    elif args.symbol and args.spread is not None:
        # 如果指定了交易对和价差，直接运行做市策略
        try:
            from strategies.market_maker import MarketMaker
            from strategies.perp_market_maker import PerpetualMarketMaker
            
            # 处理重平设置
            market_type = args.market_type

            if market_type == 'perp':
                logger.info("启动永续合约做市模式")
                logger.info(f"启动永续合约做市模式 (交易所: {exchange})")
                logger.info(f"  目标持仓量: {abs(args.target_position)}")
                logger.info(f"  最大持仓量: {args.max_position}")
                logger.info(f"  仓位触发值: {args.position_threshold}")
                logger.info(f"  报价偏移系数: {args.inventory_skew}")

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
                    ws_proxy=ws_proxy,
                    exchange=exchange,
                    exchange_config=exchange_config
                )
            else:
                logger.info("启动现货做市模式")
                enable_rebalance = True  # 默认开启
                base_asset_target_percentage = 30.0  # 默认30%
                rebalance_threshold = 15.0  # 默认15%

                if args.disable_rebalance:
                    enable_rebalance = False
                elif args.enable_rebalance:
                    enable_rebalance = True

                if args.base_asset_target is not None:
                    base_asset_target_percentage = args.base_asset_target

                if args.rebalance_threshold is not None:
                    rebalance_threshold = args.rebalance_threshold

                logger.info(f"重平设置:")
                logger.info(f"  重平功能: {'开启' if enable_rebalance else '关闭'}")
                if enable_rebalance:
                    quote_asset_target_percentage = 100.0 - base_asset_target_percentage
                    logger.info(f"  目标比例: {base_asset_target_percentage}% 基础资产 / {quote_asset_target_percentage}% 报价资产")
                    logger.info(f"  触发阈值: {rebalance_threshold}%")

                market_maker = MarketMaker(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=args.symbol,
                    base_spread_percentage=args.spread,
                    order_quantity=args.quantity,
                    max_orders=args.max_orders,
                    enable_rebalance=enable_rebalance,
                    base_asset_target_percentage=base_asset_target_percentage,
                    rebalance_threshold=rebalance_threshold,
                    ws_proxy=ws_proxy,
                    exchange=exchange,
                    exchange_config=exchange_config
                )
            
            # 执行做市策略
            market_maker.run(duration_seconds=args.duration, interval_seconds=args.interval)
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在退出...")
        except Exception as e:
            logger.error(f"做市过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 没有指定执行模式时显示帮助
        print("请指定执行模式：")
        print("  --panel   启动图形界面面板")
        print("  --cli     启动命令行界面")
        print("  直接指定  --symbol 和 --spread 参数运行做市策略")
        print("\n重平设置参数：")
        print("  --enable-rebalance        开启重平功能")
        print("  --disable-rebalance       关闭重平功能")
        print("  --base-asset-target 30    设置基础资产目标比例为30%")
        print("  --rebalance-threshold 15  设置重平触发阈值为15%")
        print("\n范例：")
        print("  python run.py --symbol SOL_USDC --spread 0.5 --enable-rebalance --base-asset-target 25 --rebalance-threshold 12")
        print("  python run.py --symbol SOL_USDC --spread 0.5 --market-type perp --target-position 1.0 --max-position 2")
        print("\n使用 --help 查看完整帮助")

if __name__ == "__main__":
    main()