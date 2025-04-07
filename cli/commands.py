"""
CLI命令模塊，提供命令行交互功能
"""
import time
from datetime import datetime

from api.client import (
    get_deposit_address, get_balance, get_markets, get_order_book, 
    get_ticker, get_fill_history, get_klines
)
from ws_client.client import BackpackWebSocket
from strategies.market_maker import MarketMaker
from utils.helpers import calculate_volatility
from database.db import Database
from config import API_KEY, SECRET_KEY
from logger import setup_logger

logger = setup_logger("cli")

def get_address_command(api_key, secret_key):
    """獲取存款地址命令"""
    blockchain = input("請輸入區塊鏈名稱(Solana, Ethereum, Bitcoin等): ")
    result = get_deposit_address(api_key, secret_key, blockchain)
    print(result)

def get_balance_command(api_key, secret_key):
    """獲取餘額命令"""
    balances = get_balance(api_key, secret_key)
    if isinstance(balances, dict) and "error" in balances and balances["error"]:
        print(f"獲取餘額失敗: {balances['error']}")
    else:
        print("\n當前餘額:")
        if isinstance(balances, dict):
            for coin, details in balances.items():
                if float(details.get('available', 0)) > 0 or float(details.get('locked', 0)) > 0:
                    print(f"{coin}: 可用 {details.get('available', 0)}, 凍結 {details.get('locked', 0)}")
        else:
            print(f"獲取餘額失敗: 無法識別返回格式 {type(balances)}")

def get_markets_command():
    """獲取市場信息命令"""
    print("\n獲取市場信息...")
    markets_info = get_markets()
    
    if isinstance(markets_info, dict) and "error" in markets_info:
        print(f"獲取市場信息失敗: {markets_info['error']}")
        return
    
    spot_markets = [m for m in markets_info if m.get('marketType') == 'SPOT']
    print(f"\n找到 {len(spot_markets)} 個現貨市場:")
    for i, market in enumerate(spot_markets):
        symbol = market.get('symbol')
        base = market.get('baseSymbol')
        quote = market.get('quoteSymbol')
        market_type = market.get('marketType')
        print(f"{i+1}. {symbol} ({base}/{quote}) - {market_type}")

def get_orderbook_command(api_key, secret_key):
    """獲取市場深度命令"""
    symbol = input("請輸入交易對 (例如: SOL_USDC): ")
    try:
        print("連接WebSocket獲取實時訂單簿...")
        ws = BackpackWebSocket(api_key, secret_key, symbol, auto_reconnect=True)
        ws.connect()
        
        # 等待連接建立
        wait_time = 0
        max_wait_time = 5
        while not ws.connected and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not ws.connected:
            print("WebSocket連接超時，使用REST API獲取訂單簿")
            depth = get_order_book(symbol)
        else:
            # 初始化訂單簿並訂閲深度流
            ws.initialize_orderbook()
            ws.subscribe_depth()
            
            # 等待數據更新
            time.sleep(2)
            depth = ws.get_orderbook()
        
        print("\n訂單簿:")
        print("\n賣單 (從低到高):")
        if 'asks' in depth and depth['asks']:
            asks = sorted(depth['asks'], key=lambda x: x[0])[:10]  # 多展示幾個深度
            for i, (price, quantity) in enumerate(asks):
                print(f"{i+1}. 價格: {price}, 數量: {quantity}")
        else:
            print("無賣單數據")
        
        print("\n買單 (從高到低):")
        if 'bids' in depth and depth['bids']:
            bids = sorted(depth['bids'], key=lambda x: x[0], reverse=True)[:10]  # 多展示幾個深度
            for i, (price, quantity) in enumerate(bids):
                print(f"{i+1}. 價格: {price}, 數量: {quantity}")
        else:
            print("無買單數據")
        
        # 分析市場情緒
        if ws.connected:
            liquidity_profile = ws.get_liquidity_profile()
            if liquidity_profile:
                buy_volume = liquidity_profile['bid_volume']
                sell_volume = liquidity_profile['ask_volume']
                imbalance = liquidity_profile['imbalance']
                
                print("\n市場流動性分析:")
                print(f"買單量: {buy_volume:.4f}")
                print(f"賣單量: {sell_volume:.4f}")
                print(f"買賣比例: {(buy_volume/sell_volume):.2f}") if sell_volume > 0 else print("買賣比例: 無限")
                
                # 判斷市場情緒
                sentiment = "買方壓力較大" if imbalance > 0.2 else "賣方壓力較大" if imbalance < -0.2 else "買賣壓力平衡"
                print(f"市場情緒: {sentiment} ({imbalance:.2f})")
        
        # 關閉WebSocket連接
        ws.close()
        
    except Exception as e:
        print(f"獲取訂單簿失敗: {str(e)}")
        # 嘗試使用REST API
        try:
            depth = get_order_book(symbol)
            if isinstance(depth, dict) and "error" in depth:
                print(f"獲取訂單簿失敗: {depth['error']}")
                return
            
            print("\n訂單簿 (REST API):")
            print("\n賣單 (從低到高):")
            if 'asks' in depth and depth['asks']:
                asks = sorted([
                    [float(price), float(quantity)] for price, quantity in depth['asks']
                ], key=lambda x: x[0])[:10]
                for i, (price, quantity) in enumerate(asks):
                    print(f"{i+1}. 價格: {price}, 數量: {quantity}")
            else:
                print("無賣單數據")
            
            print("\n買單 (從高到低):")
            if 'bids' in depth and depth['bids']:
                bids = sorted([
                    [float(price), float(quantity)] for price, quantity in depth['bids']
                ], key=lambda x: x[0], reverse=True)[:10]
                for i, (price, quantity) in enumerate(bids):
                    print(f"{i+1}. 價格: {price}, 數量: {quantity}")
            else:
                print("無買單數據")
        except Exception as e:
            print(f"使用REST API獲取訂單簿也失敗: {str(e)}")

def run_market_maker_command(api_key, secret_key):
    """執行做市策略命令"""
    symbol = input("請輸入要做市的交易對 (例如: SOL_USDC): ")
    markets_info = get_markets()
    valid_symbol = False
    if isinstance(markets_info, list):
        for market in markets_info:
            if market.get('symbol') == symbol:
                valid_symbol = True
                break
    
    if not valid_symbol:
        print(f"交易對 {symbol} 不存在或不可交易")
        return
    
    spread_percentage = float(input("請輸入價差百分比 (例如: 0.5 表示0.5%): "))
    quantity_input = input("請輸入每個訂單的數量 (留空則自動根據餘額計算): ")
    quantity = float(quantity_input) if quantity_input.strip() else None
    max_orders = int(input("請輸入每側(買/賣)最大訂單數 (例如: 3): "))
    
    duration = int(input("請輸入運行時間(秒) (例如: 3600 表示1小時): "))
    interval = int(input("請輸入更新間隔(秒) (例如: 60 表示1分鐘): "))
    
    try:
        # 初始化數據庫
        db = Database()
        
        # 初始化做市商
        market_maker = MarketMaker(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            db_instance=db,
            base_spread_percentage=spread_percentage,
            order_quantity=quantity,
            max_orders=max_orders
        )
        
        # 執行做市策略
        market_maker.run(duration_seconds=duration, interval_seconds=interval)
        
    except Exception as e:
        print(f"做市過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

def trading_stats_command(api_key, secret_key):
    """查看交易統計命令"""
    symbol = input("請輸入要查看統計的交易對 (例如: SOL_USDC): ")
    
    try:
        # 初始化數據庫
        db = Database()
        
        # 獲取今日統計
        today = datetime.now().strftime('%Y-%m-%d')
        today_stats = db.get_trading_stats(symbol, today)
        
        print("\n=== 做市商交易統計 ===")
        print(f"交易對: {symbol}")
        
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
            
            print(f"\n今日統計 ({today}):")
            print(f"總成交量: {total_volume}")
            print(f"Maker買入量: {maker_buy}")
            print(f"Maker賣出量: {maker_sell}")
            print(f"Taker買入量: {taker_buy}")
            print(f"Taker賣出量: {taker_sell}")
            print(f"Maker佔比: {maker_percentage:.2f}%")
            print(f"平均價差: {avg_spread:.4f}%")
            print(f"波動率: {volatility:.4f}%")
            print(f"毛利潤: {profit:.8f}")
            print(f"總手續費: {fees:.8f}")
            print(f"凈利潤: {net:.8f}")
        else:
            print(f"今日沒有 {symbol} 的交易記錄")
        
        # 獲取所有時間的統計
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
            
            print(f"\n累計統計:")
            print(f"總成交量: {total_volume}")
            print(f"Maker買入量: {maker_buy}")
            print(f"Maker賣出量: {maker_sell}")
            print(f"Taker買入量: {taker_buy}")
            print(f"Taker賣出量: {taker_sell}")
            print(f"Maker佔比: {maker_percentage:.2f}%")
            print(f"平均價差: {avg_spread:.4f}%")
            print(f"毛利潤: {profit:.8f}")
            print(f"總手續費: {fees:.8f}")
            print(f"凈利潤: {net:.8f}")
        else:
            print(f"沒有 {symbol} 的歷史交易記錄")
        
        # 獲取最近交易
        recent_trades = db.get_recent_trades(symbol, 10)
        
        if recent_trades and len(recent_trades) > 0:
            print("\n最近10筆成交:")
            for i, trade in enumerate(recent_trades):
                maker_str = "Maker" if trade['maker'] else "Taker"
                print(f"{i+1}. {trade['timestamp']} - {trade['side']} {trade['quantity']} @ {trade['price']} ({maker_str}) 手續費: {trade['fee']:.8f}")
        else:
            print(f"沒有 {symbol} 的最近成交記錄")
        
        # 關閉數據庫連接
        db.close()
        
    except Exception as e:
        print(f"查看交易統計時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

def market_analysis_command(api_key, secret_key):
    """市場分析命令"""
    symbol = input("請輸入要分析的交易對 (例如: SOL_USDC): ")
    try:
        print("\n執行市場分析...")
        
        # 創建臨時WebSocket連接
        ws = BackpackWebSocket(api_key, secret_key, symbol, auto_reconnect=True)
        ws.connect()
        
        # 等待連接建立
        wait_time = 0
        max_wait_time = 5
        while not ws.connected and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not ws.connected:
            print("WebSocket連接超時，無法進行完整分析")
        else:
            # 初始化訂單簿
            ws.initialize_orderbook()
            
            # 訂閲必要數據流
            ws.subscribe_depth()
            ws.subscribe_bookTicker()
            
            # 等待數據更新
            print("等待數據更新...")
            time.sleep(3)
            
            # 獲取K線數據分析趨勢
            print("獲取歷史數據分析趨勢...")
            klines = get_klines(symbol, "15m")
            
            # 添加調試信息查看數據結構
            print("K線數據結構: ")
            if isinstance(klines, dict) and "error" in klines:
                print(f"獲取K線數據出錯: {klines['error']}")
            else:
                print(f"收到 {len(klines) if isinstance(klines, list) else type(klines)} 條K線數據")
                
                # 檢查第一條記錄以確定結構
                if isinstance(klines, list) and len(klines) > 0:
                    print(f"第一條K線數據: {klines[0]}")
                    
                    # 根據實際結構提取收盤價
                    try:
                        if isinstance(klines[0], dict):
                            if 'close' in klines[0]:
                                # 如果是包含'close'字段的字典
                                prices = [float(kline['close']) for kline in klines]
                            elif 'c' in klines[0]:
                                # 另一種常見格式
                                prices = [float(kline['c']) for kline in klines]
                            else:
                                print(f"無法識別的字典K線格式，可用字段: {list(klines[0].keys())}")
                                raise ValueError("無法識別的K線數據格式")
                        elif isinstance(klines[0], list):
                            # 如果是列表格式，打印元素數量和數據樣例
                            print(f"K線列表格式，每條記錄有 {len(klines[0])} 個元素")
                            if len(klines[0]) >= 5:
                                # 通常第4或第5個元素是收盤價
                                try:
                                    # 嘗試第4個元素 (索引3)
                                    prices = [float(kline[3]) for kline in klines]
                                    print("使用索引3作為收盤價")
                                except (ValueError, IndexError):
                                    # 如果失敗，嘗試第5個元素 (索引4)
                                    prices = [float(kline[4]) for kline in klines]
                                    print("使用索引4作為收盤價")
                            else:
                                print("K線記錄元素數量不足")
                                raise ValueError("K線數據格式不兼容")
                        else:
                            print(f"未知的K線數據類型: {type(klines[0])}")
                            raise ValueError("未知的K線數據類型")
                        
                        # 計算移動平均
                        short_ma = sum(prices[-5:]) / 5 if len(prices) >= 5 else sum(prices) / len(prices)
                        medium_ma = sum(prices[-20:]) / 20 if len(prices) >= 20 else short_ma
                        long_ma = sum(prices[-50:]) / 50 if len(prices) >= 50 else medium_ma
                        
                        # 判斷趨勢
                        trend = "上漲" if short_ma > medium_ma > long_ma else "下跌" if short_ma < medium_ma < long_ma else "盤整"
                        
                        # 計算波動率
                        volatility = calculate_volatility(prices)
                        
                        print("\n市場趨勢分析:")
                        print(f"短期均價 (5週期): {short_ma:.6f}")
                        print(f"中期均價 (20週期): {medium_ma:.6f}")
                        print(f"長期均價 (50週期): {long_ma:.6f}")
                        print(f"當前趨勢: {trend}")
                        print(f"波動率: {volatility:.2f}%")
                        
                        # 獲取最新價格和波動性指標
                        current_price = ws.get_current_price()
                        liquidity_profile = ws.get_liquidity_profile()
                        
                        if current_price and liquidity_profile:
                            print(f"\n當前價格: {current_price}")
                            print(f"相對長期均價: {(current_price / long_ma - 1) * 100:.2f}%")
                            
                            # 流動性分析
                            buy_volume = liquidity_profile['bid_volume']
                            sell_volume = liquidity_profile['ask_volume']
                            imbalance = liquidity_profile['imbalance']
                            
                            print("\n市場流動性分析:")
                            print(f"買單量: {buy_volume:.4f}")
                            print(f"賣單量: {sell_volume:.4f}")
                            print(f"買賣比例: {(buy_volume/sell_volume):.2f}" if sell_volume > 0 else "買賣比例: 無限")
                            
                            # 判斷市場情緒
                            sentiment = "買方壓力較大" if imbalance > 0.2 else "賣方壓力較大" if imbalance < -0.2 else "買賣壓力平衡"
                            print(f"市場情緒: {sentiment} ({imbalance:.2f})")
                            
                            # 給出建議的做市參數
                            print("\n建議做市參數:")
                            
                            # 根據波動率調整價差
                            suggested_spread = max(0.2, min(2.0, volatility * 0.2))
                            print(f"建議價差: {suggested_spread:.2f}%")
                            
                            # 根據流動性調整訂單數量
                            liquidity_score = (buy_volume + sell_volume) / 2
                            orders_suggestion = 3
                            if liquidity_score > 10:
                                orders_suggestion = 5
                            elif liquidity_score < 1:
                                orders_suggestion = 2
                            print(f"建議訂單數: {orders_suggestion}")
                            
                            # 根據趨勢和情緒建議執行模式
                            if trend == "上漲" and imbalance > 0:
                                mode = "adaptive"
                                print("建議執行模式: 自適應模式 (跟隨上漲趨勢)")
                            elif trend == "下跌" and imbalance < 0:
                                mode = "passive"
                                print("建議執行模式: 被動模式 (降低下跌風險)")
                            else:
                                mode = "standard"
                                print("建議執行模式: 標準模式")
                    except Exception as e:
                        print(f"處理K線數據時出錯: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("未收到有效的K線數據")
        
        # 關閉WebSocket連接
        if ws:
            ws.close()
            
    except Exception as e:
        print(f"市場分析時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

def main_cli(api_key=API_KEY, secret_key=SECRET_KEY):
    """主CLI函數"""
    while True:
        print("\n===== Backpack Exchange 交易程序 =====")
        print("1 - 查詢存款地址")
        print("2 - 查詢餘額")
        print("3 - 獲取市場信息")
        print("4 - 獲取訂單簿")
        print("5 - 執行做市策略")
        print("6 - 交易統計報表")
        print("7 - 市場分析")
        print("8 - 退出")
        
        operation = input("請輸入操作類型: ")
        
        if operation == '1':
            get_address_command(api_key, secret_key)
        elif operation == '2':
            get_balance_command(api_key, secret_key)
        elif operation == '3':
            get_markets_command()
        elif operation == '4':
            get_orderbook_command(api_key, secret_key)
        elif operation == '5':
            run_market_maker_command(api_key, secret_key)
        elif operation == '6':
            trading_stats_command(api_key, secret_key)
        elif operation == '7':
            market_analysis_command(api_key, secret_key)
        elif operation == '8':
            print("退出程序。")
            break
        else:
            print("輸入錯誤，請重新輸入。")