"""
CLI命令模块，提供命令行交互功能
"""
import time
from datetime import datetime

from api.bp_client import BPClient
from ws_client.client import BackpackWebSocket
from strategies.market_maker import MarketMaker
from strategies.perp_market_maker import PerpetualMarketMaker
from utils.helpers import calculate_volatility
from database.db import Database
from config import API_KEY, SECRET_KEY
from logger import setup_logger

logger = setup_logger("cli")

# 缓存客户端实例以提高性能
_client_cache = {}

def _get_client(api_key=None, secret_key=None):
    """获取缓存的客户端实例，避免重复创建"""
    # 为无认证的公开API调用创建一个通用客户端
    if api_key is None and secret_key is None:
        cache_key = "public"
        if cache_key not in _client_cache:
            _client_cache[cache_key] = BPClient({})
        return _client_cache[cache_key]
    
    # 为认证API调用创建特定的客户端
    cache_key = f"{api_key}_{secret_key}"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = BPClient({'api_key': api_key, 'secret_key': secret_key})
    return _client_cache[cache_key]


def get_address_command(api_key, secret_key):
    """获取存款地址命令"""
    blockchain = input("请输入区块链名称(Solana, Ethereum, Bitcoin等): ")
    result = _get_client(api_key, secret_key).get_deposit_address(blockchain)
    print(result)

def get_balance_command(api_key, secret_key):
    """获取余额命令"""
    c = _get_client(api_key, secret_key)
    balances = c.get_balance()
    collateral = c.get_collateral()
    if isinstance(balances, dict) and "error" in balances and balances["error"]:
        print(f"获取余额失败: {balances['error']}")
    else:
        print("\n当前余额:")
        if isinstance(balances, dict):
            for coin, details in balances.items():
                if float(details.get('available', 0)) > 0 or float(details.get('locked', 0)) > 0:
                    print(f"{coin}: 可用 {details.get('available', 0)}, 冻结 {details.get('locked', 0)}")
        else:
            print(f"获取余额失败: 无法识别返回格式 {type(balances)}")

    if isinstance(collateral, dict) and "error" in collateral:
        print(f"获取抵押品失败: {collateral['error']}")
    elif isinstance(collateral, dict):
        assets = collateral.get('assets') or collateral.get('collateral', [])
        if assets:
            print("\n抵押品资产:")
            for item in assets:
                symbol = item.get('symbol', '')
                total = item.get('totalQuantity', '')
                available = item.get('availableQuantity', '')
                lend = item.get('lendQuantity', '')
                collateral_value = item.get('collateralValue', '')
                print(f"{symbol}: 总量 {total}, 可用 {available}, 出借中 {lend}, 抵押价值 {collateral_value}")

def get_markets_command():
    """获取市场信息命令"""
    print("\n获取市场信息...")
    markets_info = _get_client().get_markets()
    
    if isinstance(markets_info, dict) and "error" in markets_info:
        print(f"获取市场信息失败: {markets_info['error']}")
        return
    
    spot_markets = [m for m in markets_info if m.get('marketType') == 'SPOT']
    print(f"\n找到 {len(spot_markets)} 个现货市场:")
    for i, market in enumerate(spot_markets):
        symbol = market.get('symbol')
        base = market.get('baseSymbol')
        quote = market.get('quoteSymbol')
        market_type = market.get('marketType')
        print(f"{i+1}. {symbol} ({base}/{quote}) - {market_type}")

def get_orderbook_command(api_key, secret_key, ws_proxy=None):
    """获取市场深度命令"""
    symbol = input("请输入交易对 (例如: SOL_USDC): ")
    try:
        print("连接WebSocket获取实时订单簿...")
        ws = BackpackWebSocket(api_key, secret_key, symbol, auto_reconnect=True, proxy=ws_proxy)
        ws.connect()
        
        # 等待连接建立
        wait_time = 0
        max_wait_time = 5
        while not ws.connected and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not ws.connected:
            print("WebSocket连接超时，使用REST API获取订单簿")
            depth = _get_client().get_order_book(symbol)
        else:
            # 初始化订单簿并订阅深度流
            ws.initialize_orderbook()
            ws.subscribe_depth()
            
            # 等待数据更新
            time.sleep(2)
            depth = ws.get_orderbook()
        
        print("\n订单簿:")
        print("\n卖单 (从低到高):")
        if 'asks' in depth and depth['asks']:
            asks = sorted(depth['asks'], key=lambda x: x[0])[:10]  # 多展示几个深度
            for i, (price, quantity) in enumerate(asks):
                print(f"{i+1}. 价格: {price}, 数量: {quantity}")
        else:
            print("无卖单数据")
        
        print("\n买单 (从高到低):")
        if 'bids' in depth and depth['bids']:
            bids = sorted(depth['bids'], key=lambda x: x[0], reverse=True)[:10]  # 多展示几个深度
            for i, (price, quantity) in enumerate(bids):
                print(f"{i+1}. 价格: {price}, 数量: {quantity}")
        else:
            print("无买单数据")
        
        # 分析市场情绪
        if ws.connected:
            liquidity_profile = ws.get_liquidity_profile()
            if liquidity_profile:
                buy_volume = liquidity_profile['bid_volume']
                sell_volume = liquidity_profile['ask_volume']
                imbalance = liquidity_profile['imbalance']
                
                print("\n市场流动性分析:")
                print(f"买单量: {buy_volume:.4f}")
                print(f"卖单量: {sell_volume:.4f}")
                print(f"买卖比例: {(buy_volume/sell_volume):.2f}") if sell_volume > 0 else print("买卖比例: 无限")
                
                # 判断市场情绪
                sentiment = "买方压力较大" if imbalance > 0.2 else "卖方压力较大" if imbalance < -0.2 else "买卖压力平衡"
                print(f"市场情绪: {sentiment} ({imbalance:.2f})")
        
        # 关闭WebSocket连接
        ws.close()
        
    except Exception as e:
        print(f"获取订单簿失败: {str(e)}")
        # 尝试使用REST API
        try:
            depth = _get_client().get_order_book(symbol)
            if isinstance(depth, dict) and "error" in depth:
                print(f"获取订单簿失败: {depth['error']}")
                return
            
            print("\n订单簿 (REST API):")
            print("\n卖单 (从低到高):")
            if 'asks' in depth and depth['asks']:
                asks = sorted([
                    [float(price), float(quantity)] for price, quantity in depth['asks']
                ], key=lambda x: x[0])[:10]
                for i, (price, quantity) in enumerate(asks):
                    print(f"{i+1}. 价格: {price}, 数量: {quantity}")
            else:
                print("无卖单数据")
            
            print("\n买单 (从高到低):")
            if 'bids' in depth and depth['bids']:
                bids = sorted([
                    [float(price), float(quantity)] for price, quantity in depth['bids']
                ], key=lambda x: x[0], reverse=True)[:10]
                for i, (price, quantity) in enumerate(bids):
                    print(f"{i+1}. 价格: {price}, 数量: {quantity}")
            else:
                print("无买单数据")
        except Exception as e:
            print(f"使用REST API获取订单簿也失败: {str(e)}")

def configure_rebalance_settings():
    """配置重平设置"""
    print("\n=== 重平设置配置 ===")
    
    # 是否开启重平功能
    while True:
        enable_input = input("是否开启重平功能? (y/n，默认: y): ").strip().lower()
        if enable_input in ['', 'y', 'yes']:
            enable_rebalance = True
            break
        elif enable_input in ['n', 'no']:
            enable_rebalance = False
            break
        else:
            print("请输入 y 或 n")
    
    base_asset_target_percentage = 30.0  # 默认值
    rebalance_threshold = 15.0  # 默认值
    
    if enable_rebalance:
        # 设置基础资产目标比例
        while True:
            try:
                percentage_input = input("请输入基础资产目标比例 (0-100，默认: 30): ").strip()
                if percentage_input == '':
                    base_asset_target_percentage = 30.0
                    break
                else:
                    percentage = float(percentage_input)
                    if 0 <= percentage <= 100:
                        base_asset_target_percentage = percentage
                        break
                    else:
                        print("比例必须在 0-100 之间")
            except ValueError:
                print("请输入有效的数字")
        
        # 设置重平触发阈值
        while True:
            try:
                threshold_input = input("请输入重平触发阈值 (>0，默认: 15): ").strip()
                if threshold_input == '':
                    rebalance_threshold = 15.0
                    break
                else:
                    threshold = float(threshold_input)
                    if threshold > 0:
                        rebalance_threshold = threshold
                        break
                    else:
                        print("阈值必须大于 0")
            except ValueError:
                print("请输入有效的数字")
        
        quote_asset_target_percentage = 100.0 - base_asset_target_percentage
        
        print(f"\n重平设置:")
        print(f"重平功能: 开启")
        print(f"目标比例: {base_asset_target_percentage}% 基础资产 / {quote_asset_target_percentage}% 报价资产")
        print(f"触发阈值: {rebalance_threshold}%")
    else:
        print(f"\n重平设置:")
        print(f"重平功能: 关闭")
    
    return enable_rebalance, base_asset_target_percentage, rebalance_threshold

def run_market_maker_command(api_key, secret_key, ws_proxy=None):
    """执行做市策略命令"""
    market_type_input = input("请选择市场类型 (spot/perp，默认 spot): ").strip().lower()
    market_type = market_type_input if market_type_input in ("spot", "perp") else "spot"

    symbol = input("请输入要做市的交易对 (例如: SOL_USDC): ")
    markets_info = _get_client().get_markets()
    selected_market = None
    if isinstance(markets_info, list):
        for market in markets_info:
            if market.get('symbol') == symbol:
                selected_market = market
                break

    if not selected_market:
        print(f"交易对 {symbol} 不存在或不可交易")
        return

    if market_type == "spot":
        print(f"已选择现货市场 {symbol}")
    else:
        print(f"已选择永续合约市场 {symbol}")

    spread_percentage = float(input("请输入价差百分比 (例如: 0.5 表示0.5%): "))
    quantity_input = input("请输入每个订单的数量 (留空则自动根据余额计算): ")
    quantity = float(quantity_input) if quantity_input.strip() else None
    max_orders = int(input("请输入每侧(买/卖)最大订单数 (例如: 3): "))

    if market_type == "perp":
        try:
            target_position_input = input("请输入目标持仓量 (绝对值, 例如 1.0, 默认 1): ").strip()
            target_position = float(target_position_input) if target_position_input else 1.0

            max_position_input = input("最大允许持仓量(绝对值) (默认 1.0): ").strip()
            max_position = float(max_position_input) if max_position_input else 1.0

            threshold_input = input("仓位调整触发值 (默认 0.1): ").strip()
            position_threshold = float(threshold_input) if threshold_input else 0.1

            skew_input = input("仓位偏移调整系数 (0-1，默认 0.0): ").strip()
            inventory_skew = float(skew_input) if skew_input else 0.25

            if max_position <= 0:
                raise ValueError("最大持仓量必须大于0")
            if position_threshold <= 0:
                raise ValueError("仓位调整触发值必须大于0")
            if not 0 <= inventory_skew <= 1:
                raise ValueError("仓位偏移调整系数需介于0-1之间")
        except ValueError:
            print("仓位参数输入错误，取消操作")
            return

        enable_rebalance = False
        base_asset_target_percentage = 0.0
        rebalance_threshold = 0.0
    else:
        enable_rebalance, base_asset_target_percentage, rebalance_threshold = configure_rebalance_settings()
        target_position = 0.0
        max_position = 0.0
        position_threshold = 0.0
        inventory_skew = 0.0

    duration = int(input("请输入运行时间(秒) (例如: 3600 表示1小时): "))
    interval = int(input("请输入更新间隔(秒) (例如: 60 表示1分钟): "))

    try:
        db = Database()
        exchange_config = {
            'api_key': api_key,
            'secret_key': secret_key
        }

        if market_type == "perp":
            market_maker = PerpetualMarketMaker(
                api_key=api_key,
                secret_key=secret_key,
                symbol=symbol,
                db_instance=db,
                base_spread_percentage=spread_percentage,
                order_quantity=quantity,
                max_orders=max_orders,
                target_position=target_position,
                max_position=max_position,
                position_threshold=position_threshold,
                inventory_skew=inventory_skew,
                ws_proxy=ws_proxy,
                exchange_config=exchange_config
            )
        else:
            market_maker = MarketMaker(
                api_key=api_key,
                secret_key=secret_key,
                symbol=symbol,
                db_instance=db,
                base_spread_percentage=spread_percentage,
                order_quantity=quantity,
                max_orders=max_orders,
                enable_rebalance=enable_rebalance,
                base_asset_target_percentage=base_asset_target_percentage,
                rebalance_threshold=rebalance_threshold,
                ws_proxy=ws_proxy,
                exchange_config=exchange_config
            )

        market_maker.run(duration_seconds=duration, interval_seconds=interval)

    except Exception as e:
        print(f"做市过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def rebalance_settings_command():
    """重平设置管理命令"""
    print("\n=== 重平设置管理 ===")
    print("1 - 查看重平设置说明")
    print("2 - 测试重平设置")
    print("3 - 返回主菜单")
    
    choice = input("请选择操作: ")
    
    if choice == '1':
        print("\n=== 重平设置说明 ===")
        print("重平功能用于保持资产配置的平衡，避免因市场波动导致的资产比例失衡。")
        print("\n主要参数:")
        print("1. 重平功能开关: 控制是否启用自动重平衡")
        print("2. 基础资产目标比例: 基础资产应占总资产的百分比 (0-100%)")
        print("3. 重平触发阈值: 当实际比例偏离目标比例超过此阈值时触发重平衡")
        print("\n范例:")
        print("- 目标比例 30%: 假设总资产价值 1000 USDC，则理想基础资产价值为 300 USDC")
        print("- 触发阈值 15%: 当偏差超过总资产的 15% 时触发重平衡")
        print("- 如果基础资产价值变为 450 USDC，偏差为 150 USDC (15%)，将触发重平衡")
        print("\n注意事项:")
        print("- 重平衡会产生交易手续费")
        print("- 过低的阈值可能导致频繁重平衡")
        print("- 过高的阈值可能无法及时控制风险")
        
    elif choice == '2':
        print("\n=== 测试重平设置 ===")
        enable_rebalance, base_asset_target_percentage, rebalance_threshold = configure_rebalance_settings()
        
        # 模拟计算示例
        if enable_rebalance:
            print(f"\n=== 模拟计算示例 ===")
            total_assets = 1000  # 假设总资产 1000 USDC
            ideal_base_value = total_assets * (base_asset_target_percentage / 100)
            quote_asset_target_percentage = 100 - base_asset_target_percentage
            
            print(f"假设总资产: {total_assets} USDC")
            print(f"理想基础资产价值: {ideal_base_value} USDC ({base_asset_target_percentage}%)")
            print(f"理想报价资产价值: {total_assets - ideal_base_value} USDC ({quote_asset_target_percentage}%)")
            print(f"重平触发阈值: {rebalance_threshold}% = {total_assets * (rebalance_threshold / 100)} USDC")
            
            # 示例偏差情况
            print(f"\n触发重平衡的情况示例:")
            trigger_amount = total_assets * (rebalance_threshold / 100)
            high_threshold = ideal_base_value + trigger_amount
            low_threshold = ideal_base_value - trigger_amount
            
            print(f"- 当基础资产价值 > {high_threshold:.2f} USDC 时，将卖出基础资产")
            print(f"- 当基础资产价值 < {low_threshold:.2f} USDC 时，将买入基础资产")
            print(f"- 在 {low_threshold:.2f} - {high_threshold:.2f} USDC 范围内不会触发重平衡")
        
    elif choice == '3':
        return
    else:
        print("无效选择")

def trading_stats_command(api_key, secret_key):
    """查看交易统计命令"""
    symbol = input("请输入要查看统计的交易对 (例如: SOL_USDC): ")
    
    try:
        # 初始化数据库
        db = Database()
        
        # 获取今日统计
        today = datetime.now().strftime('%Y-%m-%d')
        today_stats = db.get_trading_stats(symbol, today)
        
        print("\n=== 做市商交易统计 ===")
        print(f"交易对: {symbol}")
        
        if today_stats and len(today_stats) > 0:
            stat = today_stats[0]
            maker_buy = stat['maker_buy_volume']
            maker_sell = stat['maker_sell_volume']
            taker_buy = stat['taker_buy_volume']
            taker_sell = stat['taker_sell_volume']
            profit = stat['realized_profit']
            fees = stat['total_fees']
            net = stat['net_profit']
            avg_spread = stat.get('avg_spread', 0)
            volatility = stat.get('volatility', 0)
            
            total_volume = maker_buy + maker_sell + taker_buy + taker_sell
            maker_percentage = ((maker_buy + maker_sell) / total_volume * 100) if total_volume > 0 else 0
            
            print(f"\n今日统计 ({today}):")
            print(f"总成交量: {total_volume}")
            print(f"Maker买入量: {maker_buy}")
            print(f"Maker卖出量: {maker_sell}")
            print(f"Taker买入量: {taker_buy}")
            print(f"Taker卖出量: {taker_sell}")
            print(f"Maker占比: {maker_percentage:.2f}%")
            print(f"平均价差: {avg_spread:.4f}%")
            print(f"波动率: {volatility:.4f}%")
            print(f"毛利润: {profit:.8f}")
            print(f"总手续费: {fees:.8f}")
            print(f"净利润: {net:.8f}")
        else:
            print(f"今日没有 {symbol} 的交易记录")
        
        # 获取所有时间的统计
        all_time_stats = db.get_all_time_stats(symbol)
        
        if all_time_stats:
            maker_buy = all_time_stats['total_maker_buy']
            maker_sell = all_time_stats['total_maker_sell']
            taker_buy = all_time_stats['total_taker_buy']
            taker_sell = all_time_stats['total_taker_sell']
            profit = all_time_stats['total_profit']
            fees = all_time_stats['total_fees']
            net = all_time_stats['total_net_profit']
            avg_spread = all_time_stats.get('avg_spread_all_time', 0)
            
            total_volume = maker_buy + maker_sell + taker_buy + taker_sell
            maker_percentage = ((maker_buy + maker_sell) / total_volume * 100) if total_volume > 0 else 0
            
            print(f"\n累计统计:")
            print(f"总成交量: {total_volume}")
            print(f"Maker买入量: {maker_buy}")
            print(f"Maker卖出量: {maker_sell}")
            print(f"Taker买入量: {taker_buy}")
            print(f"Taker卖出量: {taker_sell}")
            print(f"Maker占比: {maker_percentage:.2f}%")
            print(f"平均价差: {avg_spread:.4f}%")
            print(f"毛利润: {profit:.8f}")
            print(f"总手续费: {fees:.8f}")
            print(f"净利润: {net:.8f}")
        else:
            print(f"没有 {symbol} 的历史交易记录")
        
        # 获取最近交易
        recent_trades = db.get_recent_trades(symbol, 10)
        
        if recent_trades and len(recent_trades) > 0:
            print("\n最近10笔成交:")
            for i, trade in enumerate(recent_trades):
                maker_str = "Maker" if trade['maker'] else "Taker"
                print(f"{i+1}. {trade['timestamp']} - {trade['side']} {trade['quantity']} @ {trade['price']} ({maker_str}) 手续费: {trade['fee']:.8f}")
        else:
            print(f"没有 {symbol} 的最近成交记录")
        
        # 关闭数据库连接
        db.close()
        
    except Exception as e:
        print(f"查看交易统计时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def market_analysis_command(api_key, secret_key, ws_proxy=None):
    """市场分析命令"""
    symbol = input("请输入要分析的交易对 (例如: SOL_USDC): ")
    try:
        print("\n执行市场分析...")
        
        # 创建临时WebSocket连接
        ws = BackpackWebSocket(api_key, secret_key, symbol, auto_reconnect=True, proxy=ws_proxy)
        ws.connect()
        
        # 等待连接建立
        wait_time = 0
        max_wait_time = 5
        while not ws.connected and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not ws.connected:
            print("WebSocket连接超时，无法进行完整分析")
        else:
            # 初始化订单簿
            ws.initialize_orderbook()
            
            # 订阅必要数据流
            ws.subscribe_depth()
            ws.subscribe_bookTicker()
            
            # 等待数据更新
            print("等待数据更新...")
            time.sleep(3)
            
            # 获取K线数据分析趋势
            print("获取历史数据分析趋势...")
            klines = _get_client().get_klines(symbol, "15m")

            # 添加调试信息查看数据结构
            print("K线数据结构: ")
            if isinstance(klines, dict) and "error" in klines:
                print(f"获取K线数据出错: {klines['error']}")
            else:
                print(f"收到 {len(klines) if isinstance(klines, list) else type(klines)} 条K线数据")
                
                # 检查第一条记录以确定结构
                if isinstance(klines, list) and len(klines) > 0:
                    print(f"第一条K线数据: {klines[0]}")
                    
                    # 根据实际结构提取收盘价
                    try:
                        if isinstance(klines[0], dict):
                            if 'close' in klines[0]:
                                # 如果是包含'close'字段的字典
                                prices = [float(kline['close']) for kline in klines]
                            elif 'c' in klines[0]:
                                # 另一种常见格式
                                prices = [float(kline['c']) for kline in klines]
                            else:
                                print(f"无法识别的字典K线格式，可用字段: {list(klines[0].keys())}")
                                raise ValueError("无法识别的K线数据格式")
                        elif isinstance(klines[0], list):
                            # 如果是列表格式，打印元素数量和数据样例
                            print(f"K线列表格式，每条记录有 {len(klines[0])} 个元素")
                            if len(klines[0]) >= 5:
                                # 通常第4或第5个元素是收盘价
                                try:
                                    # 尝试第4个元素 (索引3)
                                    prices = [float(kline[3]) for kline in klines]
                                    print("使用索引3作为收盘价")
                                except (ValueError, IndexError):
                                    # 如果失败，尝试第5个元素 (索引4)
                                    prices = [float(kline[4]) for kline in klines]
                                    print("使用索引4作为收盘价")
                            else:
                                print("K线记录元素数量不足")
                                raise ValueError("K线数据格式不兼容")
                        else:
                            print(f"未知的K线数据类型: {type(klines[0])}")
                            raise ValueError("未知的K线数据类型")
                        
                        # 计算移动平均
                        short_ma = sum(prices[-5:]) / 5 if len(prices) >= 5 else sum(prices) / len(prices)
                        medium_ma = sum(prices[-20:]) / 20 if len(prices) >= 20 else short_ma
                        long_ma = sum(prices[-50:]) / 50 if len(prices) >= 50 else medium_ma
                        
                        # 判断趋势
                        trend = "上涨" if short_ma > medium_ma > long_ma else "下跌" if short_ma < medium_ma < long_ma else "盘整"
                        
                        # 计算波动率
                        volatility = calculate_volatility(prices)
                        
                        print("\n市场趋势分析:")
                        print(f"短期均价 (5周期): {short_ma:.6f}")
                        print(f"中期均价 (20周期): {medium_ma:.6f}")
                        print(f"长期均价 (50周期): {long_ma:.6f}")
                        print(f"当前趋势: {trend}")
                        print(f"波动率: {volatility:.2f}%")
                        
                        # 获取最新价格和波动性指标
                        current_price = ws.get_current_price()
                        liquidity_profile = ws.get_liquidity_profile()
                        
                        if current_price and liquidity_profile:
                            print(f"\n当前价格: {current_price}")
                            print(f"相对长期均价: {(current_price / long_ma - 1) * 100:.2f}%")
                            
                            # 流动性分析
                            buy_volume = liquidity_profile['bid_volume']
                            sell_volume = liquidity_profile['ask_volume']
                            imbalance = liquidity_profile['imbalance']
                            
                            print("\n市场流动性分析:")
                            print(f"买单量: {buy_volume:.4f}")
                            print(f"卖单量: {sell_volume:.4f}")
                            print(f"买卖比例: {(buy_volume/sell_volume):.2f}" if sell_volume > 0 else "买卖比例: 无限")
                            
                            # 判断市场情绪
                            sentiment = "买方压力较大" if imbalance > 0.2 else "卖方压力较大" if imbalance < -0.2 else "买卖压力平衡"
                            print(f"市场情绪: {sentiment} ({imbalance:.2f})")
                            
                            # 给出建议的做市参数
                            print("\n建议做市参数:")
                            
                            # 根据波动率调整价差
                            suggested_spread = max(0.2, min(2.0, volatility * 0.2))
                            print(f"建议价差: {suggested_spread:.2f}%")
                            
                            # 根据流动性调整订单数量
                            liquidity_score = (buy_volume + sell_volume) / 2
                            orders_suggestion = 3
                            if liquidity_score > 10:
                                orders_suggestion = 5
                            elif liquidity_score < 1:
                                orders_suggestion = 2
                            print(f"建议订单数: {orders_suggestion}")
                            
                            # 根据趋势和情绪建议执行模式
                            if trend == "上涨" and imbalance > 0:
                                mode = "adaptive"
                                print("建议执行模式: 自适应模式 (跟随上涨趋势)")
                            elif trend == "下跌" and imbalance < 0:
                                mode = "passive"
                                print("建议执行模式: 被动模式 (降低下跌风险)")
                            else:
                                mode = "standard"
                                print("建议执行模式: 标准模式")
                            
                            # 建议重平设置
                            print("\n建议重平设置:")
                            if volatility > 5:
                                print("高波动率市场，建议:")
                                print("- 基础资产比例: 20-25% (降低风险暴露)")
                                print("- 重平阈值: 10-12% (更频繁重平衡)")
                            elif volatility > 2:
                                print("中等波动率市场，建议:")
                                print("- 基础资产比例: 25-35% (标准配置)")
                                print("- 重平阈值: 12-18% (适中频率)")
                            else:
                                print("低波动率市场，建议:")
                                print("- 基础资产比例: 30-40% (可承受更高暴露)")
                                print("- 重平阈值: 15-25% (较少重平衡)")
                    except Exception as e:
                        print(f"处理K线数据时出错: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("未收到有效的K线数据")
        
        # 关闭WebSocket连接
        if ws:
            ws.close()
            
    except Exception as e:
        print(f"市场分析时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def main_cli(api_key=API_KEY, secret_key=SECRET_KEY, ws_proxy=None):
    """主CLI函数"""
    while True:
        print("\n===== Backpack Exchange 交易程序 =====")
        print("1 - 查询存款地址")
        print("2 - 查询余额")
        print("3 - 获取市场信息")
        print("4 - 获取订单簿")
        print("5 - 执行现货/合约做市策略")
        print("6 - 交易统计报表")
        print("7 - 市场分析")
        print("8 - 重平设置管理")
        print("9 - 退出")
        
        operation = input("请输入操作类型: ")
        
        if operation == '1':
            get_address_command(api_key, secret_key)
        elif operation == '2':
            get_balance_command(api_key, secret_key)
        elif operation == '3':
            get_markets_command()
        elif operation == '4':
            get_orderbook_command(api_key, secret_key, ws_proxy=ws_proxy)
        elif operation == '5':
            run_market_maker_command(api_key, secret_key, ws_proxy=ws_proxy)
        elif operation == '6':
            trading_stats_command(api_key, secret_key)
        elif operation == '7':
            market_analysis_command(api_key, secret_key, ws_proxy=ws_proxy)
        elif operation == '8':
            rebalance_settings_command()
        elif operation == '9':
            print("退出程序。")
            break
        else:
            print("输入错误，请重新输入。")