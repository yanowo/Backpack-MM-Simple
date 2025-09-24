#!/usr/bin/env python
"""
Backpack Exchange 做市交易程序統一入口
支持命令行模式和麪板模式
"""
import argparse
import sys
import os
from logger import setup_logger

# 創建記錄器
logger = setup_logger("main")

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Backpack Exchange 做市交易程序')
    
    # 模式選擇
    parser.add_argument('--panel', action='store_true', help='啟動圖形界面面板')
    parser.add_argument('--cli', action='store_true', help='啟動命令行界面')
    
    # 基本參數
    parser.add_argument('--exchange', type=str, choices=['backpack', 'xx'], default='backpack', help='交易所選擇 (backpack 或 xx)')
    parser.add_argument('--api-key', type=str, help='API Key (可選，默認使用環境變數或配置文件)')
    parser.add_argument('--secret-key', type=str, help='Secret Key (可選，默認使用環境變數或配置文件)')
    parser.add_argument('--ws-proxy', type=str, help='WebSocket Proxy (可選，默認使用環境變數或配置文件)')
    
    # 做市參數
    parser.add_argument('--symbol', type=str, help='交易對 (例如: SOL_USDC)')
    parser.add_argument('--spread', type=float, help='價差百分比 (例如: 0.5)')
    parser.add_argument('--quantity', type=float, help='訂單數量 (可選)')
    parser.add_argument('--max-orders', type=int, default=3, help='每側最大訂單數量 (默認: 3)')
    parser.add_argument('--duration', type=int, default=3600, help='運行時間（秒）(默認: 3600)')
    parser.add_argument('--interval', type=int, default=60, help='更新間隔（秒）(默認: 60)')
    parser.add_argument('--market-type', choices=['spot', 'perp'], default='spot', help='市場類型 (spot 或 perp)')
    parser.add_argument('--target-position', type=float, default=1.0, help='永續合約目標持倉量 (絕對值, 例如: 1.0)')
    parser.add_argument('--max-position', type=float, default=1.0, help='永續合約最大允許倉位(絕對值)')
    parser.add_argument('--position-threshold', type=float, default=0.1, help='永續倉位調整觸發值')
    parser.add_argument('--inventory-skew', type=float, default=0.0, help='永續倉位偏移調整係數 (0-1)')
    
    # 重平設置參數
    parser.add_argument('--enable-rebalance', action='store_true', help='開啟重平功能')
    parser.add_argument('--disable-rebalance', action='store_true', help='關閉重平功能')
    parser.add_argument('--base-asset-target', type=float, help='基礎資產目標比例 (0-100, 默認: 30)')
    parser.add_argument('--rebalance-threshold', type=float, help='重平觸發閾值 (>0, 默認: 15)')

    return parser.parse_args()

def validate_rebalance_args(args):
    """驗證重平設置參數"""
    if getattr(args, 'market_type', 'spot') == 'perp':
        return
    if args.enable_rebalance and args.disable_rebalance:
        logger.error("不能同時設置 --enable-rebalance 和 --disable-rebalance")
        sys.exit(1)
    
    if args.base_asset_target is not None:
        if not 0 <= args.base_asset_target <= 100:
            logger.error("基礎資產目標比例必須在 0-100 之間")
            sys.exit(1)
    
    if args.rebalance_threshold is not None:
        if args.rebalance_threshold <= 0:
            logger.error("重平觸發閾值必須大於 0")
            sys.exit(1)

def main():
    """主函數"""
    args = parse_arguments()
    
    # 驗證重平參數
    validate_rebalance_args(args)
    
    exchange = args.exchange
    if exchange == 'backpack':
        api_key = os.getenv('API_KEY')
        secret_key = os.getenv('SECRET_KEY')
        ws_proxy = os.getenv('PROXY_WEBSOCKET')
        base_url = os.getenv('BASE_URL', 'https://api.backpack.work')
        exchange_config = {
            'api_key': api_key,
            'secret_key': secret_key,
            'base_url': base_url,
            'api_version': 'v1',
            'default_window': '5000'
        }
    elif exchange == 'xx':
        """
        這裡是 xx 交易所的配置
        """
    else:
        logger.error("不支持的交易所，請選擇 'backpack' 或 'xx'")
        sys.exit(1)

    
    # 檢查API密鑰
    if not api_key or not secret_key:
        logger.error("缺少API密鑰，請通過命令行參數或環境變量提供")
        sys.exit(1)
    
    # 決定執行模式
    if args.panel:
        # 啟動圖形界面面板
        try:
            from panel.panel_main import run_panel
            run_panel(api_key=api_key, secret_key=secret_key, default_symbol=args.symbol)
        except ImportError as e:
            logger.error(f"啟動面板時出錯，缺少必要的庫: {str(e)}")
            logger.error("請執行 pip install rich 安裝所需庫")
            sys.exit(1)
    elif args.cli:
        # 啟動命令行界面
        try:
            from cli.commands import main_cli
            main_cli(api_key, secret_key, ws_proxy=ws_proxy)
        except ImportError as e:
            logger.error(f"啟動命令行界面時出錯: {str(e)}")
            sys.exit(1)
    elif args.symbol and args.spread is not None:
        # 如果指定了交易對和價差，直接運行做市策略
        try:
            from strategies.market_maker import MarketMaker
            from strategies.perp_market_maker import PerpetualMarketMaker
            
            # 處理重平設置
            market_type = args.market_type

            if market_type == 'perp':
                logger.info("啟動永續合約做市模式")
                logger.info(f"啟動永續合約做市模式 (交易所: {exchange})")
                logger.info(f"  目標持倉量: {abs(args.target_position)}")
                logger.info(f"  最大持倉量: {args.max_position}")
                logger.info(f"  倉位觸發值: {args.position_threshold}")
                logger.info(f"  報價偏移係數: {args.inventory_skew}")

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
                logger.info("啟動現貨做市模式")
                enable_rebalance = True  # 默認開啟
                base_asset_target_percentage = 30.0  # 默認30%
                rebalance_threshold = 15.0  # 默認15%

                if args.disable_rebalance:
                    enable_rebalance = False
                elif args.enable_rebalance:
                    enable_rebalance = True

                if args.base_asset_target is not None:
                    base_asset_target_percentage = args.base_asset_target

                if args.rebalance_threshold is not None:
                    rebalance_threshold = args.rebalance_threshold

                logger.info(f"重平設置:")
                logger.info(f"  重平功能: {'開啟' if enable_rebalance else '關閉'}")
                if enable_rebalance:
                    quote_asset_target_percentage = 100.0 - base_asset_target_percentage
                    logger.info(f"  目標比例: {base_asset_target_percentage}% 基礎資產 / {quote_asset_target_percentage}% 報價資產")
                    logger.info(f"  觸發閾值: {rebalance_threshold}%")

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
            
            # 執行做市策略
            market_maker.run(duration_seconds=args.duration, interval_seconds=args.interval)
            
        except KeyboardInterrupt:
            logger.info("收到中斷信號，正在退出...")
        except Exception as e:
            logger.error(f"做市過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 沒有指定執行模式時顯示幫助
        print("請指定執行模式：")
        print("  --panel   啟動圖形界面面板")
        print("  --cli     啟動命令行界面")
        print("  直接指定  --symbol 和 --spread 參數運行做市策略")
        print("\n重平設置參數：")
        print("  --enable-rebalance        開啟重平功能")
        print("  --disable-rebalance       關閉重平功能")
        print("  --base-asset-target 30    設置基礎資產目標比例為30%")
        print("  --rebalance-threshold 15  設置重平觸發閾值為15%")
        print("\n範例：")
        print("  python run.py --symbol SOL_USDC --spread 0.5 --enable-rebalance --base-asset-target 25 --rebalance-threshold 12")
        print("  python run.py --symbol SOL_USDC --spread 0.5 --market-type perp --target-position 1.0 --max-position 2")
        print("\n使用 --help 查看完整幫助")

if __name__ == "__main__":
    main()