#!/usr/bin/env python
"""
Backpack Exchange 做市交易程序統一入口
支持命令行模式
"""
import argparse
import sys
import os
from typing import Optional
from config import ENABLE_DATABASE
from logger import setup_logger

# 創建記錄器
logger = setup_logger("main")

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Backpack Exchange 做市交易程序')
    
    # 模式選擇
    parser.add_argument('--cli', action='store_true', help='啟動命令行界面')
    parser.add_argument('--web', action='store_true', help='啟動Web界面')
    
    # 基本參數

    parser.add_argument('--exchange', type=str, choices=['backpack', 'aster', 'paradex', 'lighter'], default='backpack', help='交易所選擇 (backpack、aster、paradex 或 lighter)')
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
    parser.add_argument('--stop-loss', type=float, help='永續倉位止損觸發值 (以報價資產計價)')
    parser.add_argument('--take-profit', type=float, help='永續倉位止盈觸發值 (以報價資產計價)')
    parser.add_argument('--strategy', choices=['standard', 'maker_hedge'], default='standard', help='策略選擇 (standard 或 maker_hedge)')

    # 數據庫選項
    parser.add_argument('--enable-db', dest='enable_db', action='store_true', help='啟用資料庫寫入功能')
    parser.add_argument('--disable-db', dest='enable_db', action='store_false', help='停用資料庫寫入功能')
    parser.set_defaults(enable_db=ENABLE_DATABASE)
    
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
    api_key = ''
    secret_key = ''
    account_address: Optional[str] = None
    ws_proxy = None
    exchange_config = {}

    if exchange == 'backpack':
        api_key = os.getenv('BACKPACK_KEY', '')
        secret_key = os.getenv('BACKPACK_SECRET', '')
        ws_proxy = os.getenv('BACKPACK_PROXY_WEBSOCKET')
        base_url = os.getenv('BASE_URL', 'https://api.backpack.work')
        exchange_config = {
            'api_key': api_key,
            'secret_key': secret_key,
            'base_url': base_url,
            'api_version': 'v1',
            'default_window': '5000'
        }
    elif exchange == 'aster':
        api_key = os.getenv('ASTER_API_KEY', '')
        secret_key = os.getenv('ASTER_SECRET_KEY', '')
        ws_proxy = os.getenv('ASTER_PROXY_WEBSOCKET')
        exchange_config = {
            'api_key': api_key,
            'secret_key': secret_key,
        }
    elif exchange == 'lighter':
        api_key = os.getenv('LIGHTER_PRIVATE_KEY') or os.getenv('LIGHTER_API_KEY')
        secret_key = os.getenv('LIGHTER_SECRET_KEY') or api_key
        ws_proxy = os.getenv('LIGHTER_PROXY_WEBSOCKET') or os.getenv('LIGHTER_WS_PROXY')
        base_url = os.getenv('LIGHTER_BASE_URL')
        account_index = os.getenv('LIGHTER_ACCOUNT_INDEX')
        account_address = os.getenv('LIGHTER_ADDRESS')
        if not account_index:
            from api.lighter_client import _get_lihgter_account_index
            account_index = _get_lihgter_account_index(account_address)
        api_key_index = os.getenv('LIGHTER_API_KEY_INDEX', '0')
        chain_id = os.getenv('LIGHTER_CHAIN_ID')

        exchange_config = {
            'api_private_key': api_key,
            'account_index': account_index,
            'api_key_index': api_key_index,
            'base_url': base_url,
        }
        if chain_id is not None:
            exchange_config['chain_id'] = chain_id
        if not api_key:
            logger.error("缺少 Lighter 私鑰，請使用 --api-key 或環境變量 LIGHTER_PRIVATE_KEY 提供")
            sys.exit(1)
        if not exchange_config.get('account_index'):
            logger.error("缺少 Lighter Account Index，請透過環境變量 LIGHTER_ACCOUNT_INDEX 提供")
    elif exchange == 'paradex':
        private_key = os.getenv('PARADEX_PRIVATE_KEY', '')  # StarkNet 私鑰
        account_address = os.getenv('PARADEX_ACCOUNT_ADDRESS')  # StarkNet 帳戶地址
        ws_proxy = os.getenv('PARADEX_PROXY_WEBSOCKET')
        base_url = os.getenv('PARADEX_BASE_URL', 'https://api.prod.paradex.trade/v1')

        secret_key = private_key
        api_key = ''  # Paradex 不需要 API Key

        exchange_config = {
            'private_key': private_key,
            'account_address': account_address,
            'base_url': base_url,
        }
    else:
        logger.error("不支持的交易所，請選擇 'backpack', 'aster', 'lighter' 或 'paradex'")
        sys.exit(1)

    # 檢查API密鑰
    if exchange == 'paradex':
        if not secret_key or not account_address:
            logger.error("Paradex 需要提供 StarkNet 私鑰與帳戶地址，請確認環境變數已設定")
            sys.exit(1)
    else:
        if not api_key or not secret_key:
            logger.error("缺少API密鑰，請通過命令行參數或環境變量提供")
            sys.exit(1)

    # 決定執行模式
    if args.web:
        # 啟動Web界面
        try:
            logger.info("啟動Web界面...")
            from web.server import run_server
            run_server(host='0.0.0.0', port=5000, debug=False)
        except ImportError as e:
            logger.error(f"啟動Web界面時出錯: {str(e)}")
            logger.error("請確保已安裝Flask和flask-socketio: pip install flask flask-socketio")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Web服務器錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.cli:
        # 啟動命令行界面
        try:
            from cli.commands import main_cli
            main_cli(api_key, secret_key, ws_proxy=ws_proxy, enable_database=args.enable_db, exchange=exchange)
        except ImportError as e:
            logger.error(f"啟動命令行界面時出錯: {str(e)}")
            sys.exit(1)
    elif args.symbol and args.spread is not None:
        # 如果指定了交易對和價差，直接運行做市策略
        try:
            from strategies.market_maker import MarketMaker
            from strategies.maker_taker_hedge import MakerTakerHedgeStrategy
            from strategies.perp_market_maker import PerpetualMarketMaker
            
            # 處理重平設置
            market_type = args.market_type

            strategy_name = args.strategy
            if market_type == 'perp':
                logger.info(f"啟動永續合約做市模式 (策略: {strategy_name}, 交易所: {exchange})")
                logger.info(f"  目標持倉量: {abs(args.target_position)}")
                logger.info(f"  最大持倉量: {args.max_position}")
                logger.info(f"  倉位觸發值: {args.position_threshold}")
                logger.info(f"  報價偏移係數: {args.inventory_skew}")

                if strategy_name == 'maker_hedge':
                    market_maker = MakerTakerHedgeStrategy(
                        api_key=api_key,
                        secret_key=secret_key,
                        symbol=args.symbol,
                        base_spread_percentage=args.spread,
                        order_quantity=args.quantity,
                        target_position=args.target_position,
                        max_position=args.max_position,
                        position_threshold=args.position_threshold,
                        inventory_skew=args.inventory_skew,
                        stop_loss=args.stop_loss,
                        take_profit=args.take_profit,
                        ws_proxy=ws_proxy,
                        exchange=exchange,
                        exchange_config=exchange_config,
                        enable_database=args.enable_db,
                        market_type='perp'
                    )
                else:
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
                        stop_loss=args.stop_loss,
                        take_profit=args.take_profit,
                        ws_proxy=ws_proxy,
                        exchange=exchange,
                        exchange_config=exchange_config,
                        enable_database=args.enable_db
                    )

                if args.stop_loss is not None:
                    logger.info(f"  止損閾值: {args.stop_loss} {market_maker.quote_asset}")
                if args.take_profit is not None:
                    logger.info(f"  止盈閾值: {args.take_profit} {market_maker.quote_asset}")
            else:
                if strategy_name == 'maker_hedge':
                    logger.info("啟動 Maker-Taker 對沖現貨模式")
                    market_maker = MakerTakerHedgeStrategy(
                        api_key=api_key,
                        secret_key=secret_key,
                        symbol=args.symbol,
                        base_spread_percentage=args.spread,
                        order_quantity=args.quantity,
                        ws_proxy=ws_proxy,
                        exchange=exchange,
                        exchange_config=exchange_config,
                        enable_database=args.enable_db,
                        market_type='spot'
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
                        exchange_config=exchange_config,
                        enable_database=args.enable_db
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
        print("  --web     啟動Web界面")
        print("  --cli     啟動命令行界面")
        print("  直接指定  --symbol 和 --spread 參數運行做市策略")
        print("\n資料庫參數：")
        print("  --enable-db            啟用資料庫寫入")
        print("  --disable-db           停用資料庫寫入 (預設)")
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
