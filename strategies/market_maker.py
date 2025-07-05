"""
做市策略模塊
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

from api.client import (
    get_balance, execute_order, get_open_orders, cancel_all_orders, 
    cancel_order, get_market_limits, get_klines, get_ticker, get_order_book,
    get_collateral
)
from ws_client.client import BackpackWebSocket
from database.db import Database
from utils.helpers import round_to_precision, round_to_tick_size, calculate_volatility
from logger import setup_logger

logger = setup_logger("market_maker")

def format_balance(value, decimals=8, threshold=1e-8):
    """
    格式化餘額顯示，避免科學記號
    
    Args:
        value: 數值
        decimals: 小數位數
        threshold: 閾值，小於此值顯示為0
    """
    if abs(value) < threshold:
        return "0.00000000"
    return f"{value:.{decimals}f}"

class MarketMaker:
    def __init__(
        self, 
        api_key, 
        secret_key, 
        symbol, 
        db_instance=None,
        base_spread_percentage=0.2, 
        order_quantity=None, 
        max_orders=3, 
        rebalance_threshold=15.0,
        enable_rebalance=True,
        base_asset_target_percentage=30.0,
        ws_proxy=None
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_spread_percentage = base_spread_percentage
        self.order_quantity = order_quantity
        self.max_orders = max_orders
        self.rebalance_threshold = rebalance_threshold
        
        # 新增重平設置參數
        self.enable_rebalance = enable_rebalance
        self.base_asset_target_percentage = base_asset_target_percentage
        self.quote_asset_target_percentage = 100.0 - base_asset_target_percentage

        # 初始化數據庫
        self.db = db_instance if db_instance else Database()
        
        # 統計屬性
        self.session_start_time = datetime.now()
        self.session_buy_trades = []
        self.session_sell_trades = []
        self.session_fees = 0.0
        self.session_maker_buy_volume = 0.0
        self.session_maker_sell_volume = 0.0
        self.session_taker_buy_volume = 0.0
        self.session_taker_sell_volume = 0.0
        
        # 初始化市場限制
        self.market_limits = get_market_limits(symbol)
        if not self.market_limits:
            raise ValueError(f"無法獲取 {symbol} 的市場限制")
        
        self.base_asset = self.market_limits['base_asset']
        self.quote_asset = self.market_limits['quote_asset']
        self.base_precision = self.market_limits['base_precision']
        self.quote_precision = self.market_limits['quote_precision']
        self.min_order_size = float(self.market_limits['min_order_size'])
        self.tick_size = float(self.market_limits['tick_size'])
        
        # 交易量統計
        self.maker_buy_volume = 0
        self.maker_sell_volume = 0
        self.taker_buy_volume = 0
        self.taker_sell_volume = 0
        self.total_fees = 0

        # 添加代理参数
        self.ws_proxy = ws_proxy
        # 建立WebSocket連接
        self.ws = BackpackWebSocket(api_key, secret_key, symbol, self.on_ws_message, auto_reconnect=True, proxy=self.ws_proxy)
        self.ws.connect()
        
        # 跟蹤活躍訂單
        self.active_buy_orders = []
        self.active_sell_orders = []
        
        # 記錄買賣數量以便重新平衡
        self.total_bought = 0
        self.total_sold = 0
        
        # 交易記錄 - 用於計算利潤
        self.buy_trades = []
        self.sell_trades = []
        
        # 利潤統計
        self.total_profit = 0
        self.trades_executed = 0
        self.orders_placed = 0
        self.orders_cancelled = 0
        
        # 執行緒池用於後台任務
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # 等待WebSocket連接建立並進行初始化訂閲
        self._initialize_websocket()
        
        # 載入交易統計和歷史交易
        self._load_trading_stats()
        self._load_recent_trades()
        
        logger.info(f"初始化做市商: {symbol}")
        logger.info(f"基礎資產: {self.base_asset}, 報價資產: {self.quote_asset}")
        logger.info(f"基礎精度: {self.base_precision}, 報價精度: {self.quote_precision}")
        logger.info(f"最小訂單大小: {self.min_order_size}, 價格步長: {self.tick_size}")
        logger.info(f"基礎價差百分比: {self.base_spread_percentage}%, 最大訂單數: {self.max_orders}")
        logger.info(f"重平功能: {'開啟' if self.enable_rebalance else '關閉'}")
        if self.enable_rebalance:
            logger.info(f"重平目標比例: {self.base_asset_target_percentage}% {self.base_asset} / {self.quote_asset_target_percentage}% {self.quote_asset}")
            logger.info(f"重平觸發閾值: {self.rebalance_threshold}%")
    
    def set_rebalance_settings(self, enable_rebalance=None, base_asset_target_percentage=None, rebalance_threshold=None):
        """
        設置重平參數
        
        Args:
            enable_rebalance: 是否開啟重平功能
            base_asset_target_percentage: 基礎資產目標比例 (0-100)
            rebalance_threshold: 重平觸發閾值
        """
        if enable_rebalance is not None:
            self.enable_rebalance = enable_rebalance
            logger.info(f"重平功能設置為: {'開啟' if enable_rebalance else '關閉'}")
        
        if base_asset_target_percentage is not None:
            if not 0 <= base_asset_target_percentage <= 100:
                raise ValueError("基礎資產目標比例必須在0-100之間")
            
            self.base_asset_target_percentage = base_asset_target_percentage
            self.quote_asset_target_percentage = 100.0 - base_asset_target_percentage
            logger.info(f"重平目標比例設置為: {self.base_asset_target_percentage}% {self.base_asset} / {self.quote_asset_target_percentage}% {self.quote_asset}")
        
        if rebalance_threshold is not None:
            if rebalance_threshold <= 0:
                raise ValueError("重平觸發閾值必須大於0")
            
            self.rebalance_threshold = rebalance_threshold
            logger.info(f"重平觸發閾值設置為: {self.rebalance_threshold}%")
    
    def get_rebalance_settings(self):
        """
        獲取當前重平設置
        
        Returns:
            dict: 重平設置信息
        """
        return {
            'enable_rebalance': self.enable_rebalance,
            'base_asset_target_percentage': self.base_asset_target_percentage,
            'quote_asset_target_percentage': self.quote_asset_target_percentage,
            'rebalance_threshold': self.rebalance_threshold
        }
    
    def get_total_balance(self):
        """獲取總餘額，包含普通餘額和抵押品餘額"""
        try:
            # 獲取普通餘額
            balances = get_balance(self.api_key, self.secret_key)
            if isinstance(balances, dict) and "error" in balances:
                logger.error(f"獲取普通餘額失敗: {balances['error']}")
                return None
            
            # 獲取抵押品餘額
            collateral = get_collateral(self.api_key, self.secret_key)
            if isinstance(collateral, dict) and "error" in collateral:
                logger.warning(f"獲取抵押品餘額失敗: {collateral['error']}")
                collateral_assets = []
            else:
                collateral_assets = collateral.get('assets') or collateral.get('collateral', [])
            
            # 初始化總餘額字典
            total_balances = {}
            
            # 處理普通餘額
            if isinstance(balances, dict):
                for asset, details in balances.items():
                    available = float(details.get('available', 0))
                    locked = float(details.get('locked', 0))
                    total_balances[asset] = {
                        'available': available,
                        'locked': locked,
                        'total': available + locked,
                        'collateral_available': 0,
                        'collateral_total': 0
                    }
            
            # 添加抵押品餘額
            for item in collateral_assets:
                symbol = item.get('symbol', '')
                if symbol:
                    total_quantity = float(item.get('totalQuantity', 0))
                    available_quantity = float(item.get('availableQuantity', 0))
                    
                    if symbol not in total_balances:
                        total_balances[symbol] = {
                            'available': 0,
                            'locked': 0,
                            'total': 0,
                            'collateral_available': available_quantity,
                            'collateral_total': total_quantity
                        }
                    else:
                        total_balances[symbol]['collateral_available'] = available_quantity
                        total_balances[symbol]['collateral_total'] = total_quantity
                    
                    # 更新總可用量和總量
                    total_balances[symbol]['total_available'] = (
                        total_balances[symbol]['available'] + 
                        total_balances[symbol]['collateral_available']
                    )
                    total_balances[symbol]['total_all'] = (
                        total_balances[symbol]['total'] + 
                        total_balances[symbol]['collateral_total']
                    )
            
            # 確保所有資產都有total_available和total_all字段
            for asset in total_balances:
                if 'total_available' not in total_balances[asset]:
                    total_balances[asset]['total_available'] = total_balances[asset]['available']
                if 'total_all' not in total_balances[asset]:
                    total_balances[asset]['total_all'] = total_balances[asset]['total']
            
            return total_balances
            
        except Exception as e:
            logger.error(f"獲取總餘額時出錯: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_asset_balance(self, asset):
        """獲取指定資產的總可用餘額"""
        total_balances = self.get_total_balance()
        if not total_balances or asset not in total_balances:
            return 0, 0  # 返回 (可用餘額, 總餘額)
        
        balance_info = total_balances[asset]
        available = balance_info.get('total_available', 0)
        total = balance_info.get('total_all', 0)
        
        # 格式化顯示餘額，避免科學記號
        normal_available = balance_info.get('available', 0)
        collateral_available = balance_info.get('collateral_available', 0)
        
        logger.debug(f"{asset} 餘額詳情: 普通可用={format_balance(normal_available)}, "
                    f"抵押品可用={format_balance(collateral_available)}, "
                    f"總可用={format_balance(available)}, 總量={format_balance(total)}")
        
        return available, total
    
    def _initialize_websocket(self):
        """等待WebSocket連接建立並進行初始化訂閲"""
        wait_time = 0
        max_wait_time = 10
        while not self.ws.connected and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_time += 0.5
        
        if self.ws.connected:
            logger.info("WebSocket連接已建立，初始化數據流...")
            
            # 初始化訂單簿
            orderbook_initialized = self.ws.initialize_orderbook()
            
            # 訂閲深度流和行情數據
            if orderbook_initialized:
                depth_subscribed = self.ws.subscribe_depth()
                ticker_subscribed = self.ws.subscribe_bookTicker()
                
                if depth_subscribed and ticker_subscribed:
                    logger.info("數據流訂閲成功!")
            
            # 訂閲私有訂單更新流
            self.subscribe_order_updates()
        else:
            logger.warning(f"WebSocket連接建立超時，將在運行過程中繼續嘗試連接")
    
    def _load_trading_stats(self):
        """從數據庫加載交易統計數據"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 查詢今天的統計數據
            stats = self.db.get_trading_stats(self.symbol, today)
            
            if stats and len(stats) > 0:
                stat = stats[0]
                self.maker_buy_volume = stat['maker_buy_volume']
                self.maker_sell_volume = stat['maker_sell_volume']
                self.taker_buy_volume = stat['taker_buy_volume']
                self.taker_sell_volume = stat['taker_sell_volume']
                self.total_profit = stat['realized_profit']
                self.total_fees = stat['total_fees']
                
                logger.info(f"已從數據庫加載今日交易統計")
                logger.info(f"Maker買入量: {self.maker_buy_volume}, Maker賣出量: {self.maker_sell_volume}")
                logger.info(f"Taker買入量: {self.taker_buy_volume}, Taker賣出量: {self.taker_sell_volume}")
                logger.info(f"已實現利潤: {self.total_profit}, 總手續費: {self.total_fees}")
            else:
                logger.info("今日無交易統計記錄，將創建新記錄")
        except Exception as e:
            logger.error(f"加載交易統計時出錯: {e}")
    
    def _load_recent_trades(self):
        """從數據庫加載歷史成交記錄"""
        try:
            # 獲取訂單歷史
            trades = self.db.get_order_history(self.symbol, 1000)
            trades_count = len(trades) if trades else 0
            
            if trades_count > 0:
                for side, quantity, price, maker, fee in trades:
                    quantity = float(quantity)
                    price = float(price)
                    fee = float(fee)
                    
                    if side == 'Bid':  # 買入
                        self.buy_trades.append((price, quantity))
                        self.total_bought += quantity
                        if maker:
                            self.maker_buy_volume += quantity
                        else:
                            self.taker_buy_volume += quantity
                    elif side == 'Ask':  # 賣出
                        self.sell_trades.append((price, quantity))
                        self.total_sold += quantity
                        if maker:
                            self.maker_sell_volume += quantity
                        else:
                            self.taker_sell_volume += quantity
                    
                    self.total_fees += fee
                
                logger.info(f"已從數據庫載入 {trades_count} 條歷史成交記錄")
                logger.info(f"總買入: {self.total_bought} {self.base_asset}, 總賣出: {self.total_sold} {self.base_asset}")
                logger.info(f"Maker買入: {self.maker_buy_volume} {self.base_asset}, Maker賣出: {self.maker_sell_volume} {self.base_asset}")
                logger.info(f"Taker買入: {self.taker_buy_volume} {self.base_asset}, Taker賣出: {self.taker_sell_volume} {self.base_asset}")
                
                # 計算精確利潤
                self.total_profit = self._calculate_db_profit()
                logger.info(f"計算得出已實現利潤: {self.total_profit:.8f} {self.quote_asset}")
                logger.info(f"總手續費: {self.total_fees:.8f} {self.quote_asset}")
            else:
                logger.info("數據庫中沒有歷史成交記錄，將開始記錄新的交易")
                
        except Exception as e:
            logger.error(f"載入歷史成交記錄時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    def check_ws_connection(self):
        """檢查並恢復WebSocket連接"""
        if not self.ws:
            logger.warning("WebSocket對象不存在，嘗試重新創建...")
            return self._recreate_websocket()
            
        ws_connected = self.ws.is_connected()
        
        if not ws_connected and not getattr(self.ws, 'reconnecting', False):
            logger.warning("WebSocket連接已斷開，觸發重連...")
            # 使用 WebSocket 自己的重連機制
            self.ws.check_and_reconnect_if_needed()
        
        return self.ws.is_connected() if self.ws else False
    
    def _recreate_websocket(self):
        """重新創建WebSocket連接"""
        try:
            logger.info("重新創建WebSocket連接...")
            
            # 安全關閉現有連接
            if self.ws:
                try:
                    self.ws.running = False
                    self.ws.close()
                    time.sleep(0.5)
                except Exception as e:
                    logger.debug(f"關閉現有WebSocket時的預期錯誤: {e}")
            
            # 創建新的連接
            self.ws = BackpackWebSocket(
                self.api_key, 
                self.secret_key, 
                self.symbol, 
                self.on_ws_message, 
                auto_reconnect=True,
                proxy=self.ws_proxy
            )
            self.ws.connect()
            
            # 等待連接建立，但不要等太久
            wait_time = 0
            max_wait_time = 3  # 減少等待時間
            while not self.ws.is_connected() and wait_time < max_wait_time:
                time.sleep(0.5)
                wait_time += 0.5
                
            if self.ws.is_connected():
                logger.info("WebSocket重新創建成功")
                
                # 重新初始化
                self.ws.initialize_orderbook()
                self.ws.subscribe_depth()
                self.ws.subscribe_bookTicker()
                self.subscribe_order_updates()
                return True
            else:
                logger.warning("WebSocket重新創建後仍未連接，但繼續運行")
                return False
                
        except Exception as e:
            logger.error(f"重新創建WebSocket連接時出錯: {e}")
            return False
    
    def on_ws_message(self, stream, data):
        """處理WebSocket消息回調"""
        if stream.startswith("account.orderUpdate."):
            event_type = data.get('e')
            
            # 「訂單成交」事件
            if event_type == 'orderFill':
                try:
                    side = data.get('S')
                    quantity = float(data.get('l', '0'))  # 此次成交數量
                    price = float(data.get('L', '0'))     # 此次成交價格
                    order_id = data.get('i')             # 訂單 ID
                    maker = data.get('m', False)         # 是否是 Maker
                    fee = float(data.get('n', '0'))      # 手續費
                    fee_asset = data.get('N', '')        # 手續費資產

                    logger.info(f"訂單成交: ID={order_id}, 方向={side}, 數量={quantity}, 價格={price}, Maker={maker}, 手續費={fee:.8f}")
                    
                    # 判斷交易類型
                    trade_type = 'market_making'  # 默認為做市行為
                    
                    # 安全地檢查訂單是否是重平衡訂單
                    try:
                        is_rebalance = self.db.is_rebalance_order(order_id, self.symbol)
                        if is_rebalance:
                            trade_type = 'rebalance'
                    except Exception as db_err:
                        logger.error(f"檢查重平衡訂單時出錯: {db_err}")
                    
                    # 準備訂單數據
                    order_data = {
                        'order_id': order_id,
                        'symbol': self.symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'maker': maker,
                        'fee': fee,
                        'fee_asset': fee_asset,
                        'trade_type': trade_type
                    }
                    
                    # 安全地插入數據庫
                    def safe_insert_order():
                        try:
                            self.db.insert_order(order_data)
                        except Exception as db_err:
                            logger.error(f"插入訂單數據時出錯: {db_err}")
                    
                    # 直接在當前線程中插入訂單數據，確保先寫入基本數據
                    safe_insert_order()
                    
                    # 更新買賣量和做市商成交量統計
                    if side == 'Bid':  # 買入
                        self.total_bought += quantity
                        self.buy_trades.append((price, quantity))
                        logger.info(f"買入成交: {quantity} {self.base_asset} @ {price} {self.quote_asset}")
                        
                        # 更新做市商成交量
                        if maker:
                            self.maker_buy_volume += quantity
                            self.session_maker_buy_volume += quantity
                        else:
                            self.taker_buy_volume += quantity
                            self.session_taker_buy_volume += quantity
                        
                        self.session_buy_trades.append((price, quantity))
                            
                    elif side == 'Ask':  # 賣出
                        self.total_sold += quantity
                        self.sell_trades.append((price, quantity))
                        logger.info(f"賣出成交: {quantity} {self.base_asset} @ {price} {self.quote_asset}")
                        
                        # 更新做市商成交量
                        if maker:
                            self.maker_sell_volume += quantity
                            self.session_maker_sell_volume += quantity
                        else:
                            self.taker_sell_volume += quantity
                            self.session_taker_sell_volume += quantity
                            
                        self.session_sell_trades.append((price, quantity))
                    
                    # 更新累計手續費
                    self.total_fees += fee
                    self.session_fees += fee
                        
                    # 在單獨的線程中更新統計數據，避免阻塞主回調
                    def safe_update_stats_wrapper():
                        try:
                            self._update_trading_stats()
                        except Exception as e:
                            logger.error(f"更新交易統計時出錯: {e}")
                    
                    self.executor.submit(safe_update_stats_wrapper)
                    
                    # 重新計算利潤（基於數據庫記錄）
                    # 也在單獨的線程中進行計算，避免阻塞
                    def update_profit():
                        try:
                            profit = self._calculate_db_profit()
                            self.total_profit = profit
                        except Exception as e:
                            logger.error(f"更新利潤計算時出錯: {e}")
                    
                    self.executor.submit(update_profit)
                    
                    # 計算本次執行的簡單利潤（不涉及數據庫查詢）
                    session_profit = self._calculate_session_profit()
                    
                    # 執行簡要統計
                    logger.info(f"累計利潤: {self.total_profit:.8f} {self.quote_asset}")
                    logger.info(f"本次執行利潤: {session_profit:.8f} {self.quote_asset}")
                    logger.info(f"本次執行手續費: {self.session_fees:.8f} {self.quote_asset}")
                    logger.info(f"本次執行淨利潤: {(session_profit - self.session_fees):.8f} {self.quote_asset}")
                    
                    self.trades_executed += 1
                    logger.info(f"總買入: {self.total_bought} {self.base_asset}, 總賣出: {self.total_sold} {self.base_asset}")
                    logger.info(f"Maker買入: {self.maker_buy_volume} {self.base_asset}, Maker賣出: {self.maker_sell_volume} {self.base_asset}")
                    logger.info(f"Taker買入: {self.taker_buy_volume} {self.base_asset}, Taker賣出: {self.taker_sell_volume} {self.base_asset}")
                    
                except Exception as e:
                    logger.error(f"處理訂單成交消息時出錯: {e}")
                    import traceback
                    traceback.print_exc()
    
    def _calculate_db_profit(self):
        """基於數據庫記錄計算已實現利潤（FIFO方法）"""
        try:
            # 獲取訂單歷史，注意這裡將返回一個列表
            order_history = self.db.get_order_history(self.symbol)
            if not order_history:
                return 0
            
            buy_trades = []
            sell_trades = []
            for side, quantity, price, maker, fee in order_history:
                if side == 'Bid':
                    buy_trades.append((float(price), float(quantity), float(fee)))
                elif side == 'Ask':
                    sell_trades.append((float(price), float(quantity), float(fee)))

            if not buy_trades or not sell_trades:
                return 0

            buy_queue = buy_trades.copy()
            total_profit = 0
            total_fees = 0

            for sell_price, sell_quantity, sell_fee in sell_trades:
                remaining_sell = sell_quantity
                total_fees += sell_fee

                while remaining_sell > 0 and buy_queue:
                    buy_price, buy_quantity, buy_fee = buy_queue[0]
                    matched_quantity = min(remaining_sell, buy_quantity)

                    trade_profit = (sell_price - buy_price) * matched_quantity
                    allocated_buy_fee = buy_fee * (matched_quantity / buy_quantity)
                    total_fees += allocated_buy_fee

                    net_trade_profit = trade_profit
                    total_profit += net_trade_profit

                    remaining_sell -= matched_quantity
                    if matched_quantity >= buy_quantity:
                        buy_queue.pop(0)
                    else:
                        remaining_fee = buy_fee * (1 - matched_quantity / buy_quantity)
                        buy_queue[0] = (buy_price, buy_quantity - matched_quantity, remaining_fee)

            self.total_fees = total_fees
            return total_profit

        except Exception as e:
            logger.error(f"計算數據庫利潤時出錯: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _update_trading_stats(self):
        """更新每日交易統計數據"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 計算額外指標
            volatility = 0
            if self.ws and hasattr(self.ws, 'historical_prices'):
                volatility = calculate_volatility(self.ws.historical_prices)
            
            # 計算平均價差
            avg_spread = 0
            if self.ws and self.ws.bid_price and self.ws.ask_price:
                avg_spread = (self.ws.ask_price - self.ws.bid_price) / ((self.ws.ask_price + self.ws.bid_price) / 2) * 100
            
            # 準備統計數據
            stats_data = {
                'date': today,
                'symbol': self.symbol,
                'maker_buy_volume': self.maker_buy_volume,
                'maker_sell_volume': self.maker_sell_volume,
                'taker_buy_volume': self.taker_buy_volume,
                'taker_sell_volume': self.taker_sell_volume,
                'realized_profit': self.total_profit,
                'total_fees': self.total_fees,
                'net_profit': self.total_profit - self.total_fees,
                'avg_spread': avg_spread,
                'trade_count': self.trades_executed,
                'volatility': volatility
            }
            
            # 使用專門的函數來處理數據庫操作
            def safe_update_stats():
                try:
                    success = self.db.update_trading_stats(stats_data)
                    if not success:
                        logger.warning("更新交易統計失敗，下次再試")
                except Exception as db_err:
                    logger.error(f"更新交易統計時出錯: {db_err}")
            
            # 直接在當前線程執行，避免過多的並發操作
            safe_update_stats()
                
        except Exception as e:
            logger.error(f"更新交易統計數據時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_average_buy_cost(self):
        """計算平均買入成本"""
        if not self.buy_trades:
            return 0
            
        total_buy_cost = sum(price * quantity for price, quantity in self.buy_trades)
        total_buy_quantity = sum(quantity for _, quantity in self.buy_trades)
        
        if not self.sell_trades or total_buy_quantity <= 0:
            return total_buy_cost / total_buy_quantity if total_buy_quantity > 0 else 0
        
        buy_queue = self.buy_trades.copy()
        consumed_cost = 0
        consumed_quantity = 0
        
        for _, sell_quantity in self.sell_trades:
            remaining_sell = sell_quantity
            
            while remaining_sell > 0 and buy_queue:
                buy_price, buy_quantity = buy_queue[0]
                matched_quantity = min(remaining_sell, buy_quantity)
                consumed_cost += buy_price * matched_quantity
                consumed_quantity += matched_quantity
                remaining_sell -= matched_quantity
                
                if matched_quantity >= buy_quantity:
                    buy_queue.pop(0)
                else:
                    buy_queue[0] = (buy_price, buy_quantity - matched_quantity)
        
        remaining_buy_quantity = total_buy_quantity - consumed_quantity
        remaining_buy_cost = total_buy_cost - consumed_cost
        
        if remaining_buy_quantity <= 0:
            if self.ws and self.ws.connected and self.ws.bid_price:
                return self.ws.bid_price
            return 0
        
        return remaining_buy_cost / remaining_buy_quantity
    
    def _calculate_session_profit(self):
        """計算本次執行的已實現利潤"""
        if not self.session_buy_trades or not self.session_sell_trades:
            return 0

        buy_queue = self.session_buy_trades.copy()
        total_profit = 0

        for sell_price, sell_quantity in self.session_sell_trades:
            remaining_sell = sell_quantity

            while remaining_sell > 0 and buy_queue:
                buy_price, buy_quantity = buy_queue[0]
                matched_quantity = min(remaining_sell, buy_quantity)

                # 計算這筆交易的利潤
                trade_profit = (sell_price - buy_price) * matched_quantity
                total_profit += trade_profit

                remaining_sell -= matched_quantity
                if matched_quantity >= buy_quantity:
                    buy_queue.pop(0)
                else:
                    buy_queue[0] = (buy_price, buy_quantity - matched_quantity)

        return total_profit

    def calculate_pnl(self):
        """計算已實現和未實現PnL"""
        # 總的已實現利潤
        realized_pnl = self._calculate_db_profit()
        
        # 本次執行的已實現利潤
        session_realized_pnl = self._calculate_session_profit()
        
        # 計算未實現利潤
        unrealized_pnl = 0
        net_position = self.total_bought - self.total_sold
        
        if net_position > 0:
            current_price = self.get_current_price()
            if current_price:
                avg_buy_cost = self._calculate_average_buy_cost()
                unrealized_pnl = (current_price - avg_buy_cost) * net_position
        
        # 返回總的PnL和本次執行的PnL
        return realized_pnl, unrealized_pnl, self.total_fees, realized_pnl - self.total_fees, session_realized_pnl, self.session_fees, session_realized_pnl - self.session_fees
    
    def get_current_price(self):
        """獲取當前價格（優先使用WebSocket數據）"""
        self.check_ws_connection()
        price = None
        if self.ws and self.ws.connected:
            price = self.ws.get_current_price()
        
        if price is None:
            ticker = get_ticker(self.symbol)
            if isinstance(ticker, dict) and "error" in ticker:
                logger.error(f"獲取價格失敗: {ticker['error']}")
                return None
            
            if "lastPrice" not in ticker:
                logger.error(f"獲取到的價格數據不完整: {ticker}")
                return None
            return float(ticker['lastPrice'])
        return price
    
    def get_market_depth(self):
        """獲取市場深度（優先使用WebSocket數據）"""
        self.check_ws_connection()
        bid_price, ask_price = None, None
        if self.ws and self.ws.connected:
            bid_price, ask_price = self.ws.get_bid_ask()
        
        if bid_price is None or ask_price is None:
            order_book = get_order_book(self.symbol)
            if isinstance(order_book, dict) and "error" in order_book:
                logger.error(f"獲取訂單簿失敗: {order_book['error']}")
                return None, None
            
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            if not bids or not asks:
                return None, None
            
            highest_bid = float(bids[-1][0]) if bids else None
            lowest_ask = float(asks[0][0]) if asks else None
            
            return highest_bid, lowest_ask
        
        return bid_price, ask_price
    
    def calculate_dynamic_spread(self):
        """計算動態價差基於市場情況"""
        base_spread = self.base_spread_percentage
        
        # 返回基礎價差，不再進行動態計算
        return base_spread
    
    def calculate_prices(self):
        """計算買賣訂單價格"""
        try:
            bid_price, ask_price = self.get_market_depth()
            if bid_price is None or ask_price is None:
                current_price = self.get_current_price()
                if current_price is None:
                    logger.error("無法獲取價格信息，無法設置訂單")
                    return None, None
                mid_price = current_price
            else:
                mid_price = (bid_price + ask_price) / 2
            
            logger.info(f"市場中間價: {mid_price}")
            
            # 使用基礎價差
            spread_percentage = self.base_spread_percentage
            exact_spread = mid_price * (spread_percentage / 100)
            
            base_buy_price = mid_price - (exact_spread / 2)
            base_sell_price = mid_price + (exact_spread / 2)
            
            base_buy_price = round_to_tick_size(base_buy_price, self.tick_size)
            base_sell_price = round_to_tick_size(base_sell_price, self.tick_size)
            
            actual_spread = base_sell_price - base_buy_price
            actual_spread_pct = (actual_spread / mid_price) * 100
            logger.info(f"使用的價差: {actual_spread_pct:.4f}% (目標: {spread_percentage}%), 絕對價差: {actual_spread}")
            
            # 計算梯度訂單價格
            buy_prices = []
            sell_prices = []
            
            # 優化梯度分佈：較小的梯度以提高成交率
            for i in range(self.max_orders):
                # 非線性遞增的梯度，靠近中間的訂單梯度小，越遠離中間梯度越大
                gradient_factor = (i ** 1.5) * 1.5
                
                buy_adjustment = gradient_factor * self.tick_size
                sell_adjustment = gradient_factor * self.tick_size
                
                buy_price = round_to_tick_size(base_buy_price - buy_adjustment, self.tick_size)
                sell_price = round_to_tick_size(base_sell_price + sell_adjustment, self.tick_size)
                
                buy_prices.append(buy_price)
                sell_prices.append(sell_price)
            
            final_spread = sell_prices[0] - buy_prices[0]
            final_spread_pct = (final_spread / mid_price) * 100
            logger.info(f"最終價差: {final_spread_pct:.4f}% (最低賣價 {sell_prices[0]} - 最高買價 {buy_prices[0]} = {final_spread})")
            
            return buy_prices, sell_prices
        
        except Exception as e:
            logger.error(f"計算價格時出錯: {str(e)}")
            return None, None
    
    def need_rebalance(self):
        """判斷是否需要重平衡倉位（基於總餘額包含抵押品）"""
        # 檢查重平功能是否開啟
        if not self.enable_rebalance:
            logger.debug("重平功能已關閉，跳過重平衡檢查")
            return False
            
        logger.info("檢查是否需要重平衡倉位...")
        
        # 獲取當前價格
        current_price = self.get_current_price()
        if not current_price:
            logger.warning("無法獲取當前價格，跳過重平衡檢查")
            return False
        
        # 獲取基礎資產和報價資產的總可用餘額（包含抵押品）
        base_available, base_total = self.get_asset_balance(self.base_asset)
        quote_available, quote_total = self.get_asset_balance(self.quote_asset)
        
        logger.info(f"當前基礎資產餘額: 可用 {format_balance(base_available)} {self.base_asset}, 總計 {format_balance(base_total)} {self.base_asset}")
        logger.info(f"當前報價資產餘額: 可用 {format_balance(quote_available)} {self.quote_asset}, 總計 {format_balance(quote_total)} {self.quote_asset}")
        
        # 計算總資產價值（以報價貨幣計算）
        total_assets = quote_total + (base_total * current_price)
        
        # 檢查是否有足夠資產進行重平衡
        min_asset_value = self.min_order_size * current_price * 10  # 最小資產要求
        if total_assets < min_asset_value:
            logger.info(f"總資產價值 {total_assets:.2f} {self.quote_asset} 過小，跳過重平衡檢查")
            return False
        
        # 使用用戶設定的目標比例
        ideal_base_value = total_assets * (self.base_asset_target_percentage / 100)
        actual_base_value = base_total * current_price
        
        # 計算偏差
        deviation_value = abs(actual_base_value - ideal_base_value)
        risk_exposure = (deviation_value / total_assets) * 100 if total_assets > 0 else 0
        
        logger.info(f"總資產價值: {total_assets:.2f} {self.quote_asset}")
        logger.info(f"目標配置比例: {self.base_asset_target_percentage}% {self.base_asset} / {self.quote_asset_target_percentage}% {self.quote_asset}")
        logger.info(f"理想基礎資產價值: {ideal_base_value:.2f} {self.quote_asset}")
        logger.info(f"實際基礎資產價值: {actual_base_value:.2f} {self.quote_asset}")
        logger.info(f"偏差: {deviation_value:.2f} {self.quote_asset}")
        logger.info(f"風險暴露比例: {risk_exposure:.2f}% (閾值: {self.rebalance_threshold}%)")
        
        need_rebalance = risk_exposure > self.rebalance_threshold
        logger.info(f"重平衡檢查結果: {'需要重平衡' if need_rebalance else '不需要重平衡'}")
        
        return need_rebalance
    
    def rebalance_position(self):
        """重平衡倉位（使用總餘額包含抵押品）"""
        # 檢查重平功能是否開啟
        if not self.enable_rebalance:
            logger.warning("重平功能已關閉，取消重平衡操作")
            return
            
        logger.info("開始重新平衡倉位...")
        self.check_ws_connection()
        
        # 獲取當前價格
        current_price = self.get_current_price()
        if not current_price:
            logger.error("無法獲取價格，無法重新平衡")
            return
        
        # 獲取市場深度
        bid_price, ask_price = self.get_market_depth()
        if bid_price is None or ask_price is None:
            bid_price = current_price * 0.998
            ask_price = current_price * 1.002
        
        # 獲取總可用餘額（包含抵押品）
        base_available, base_total = self.get_asset_balance(self.base_asset)
        quote_available, quote_total = self.get_asset_balance(self.quote_asset)
        
        logger.info(f"基礎資產: 可用 {format_balance(base_available)}, 總計 {format_balance(base_total)} {self.base_asset}")
        logger.info(f"報價資產: 可用 {format_balance(quote_available)}, 總計 {format_balance(quote_total)} {self.quote_asset}")
        
        # 計算總資產價值
        total_assets = quote_total + (base_total * current_price)
        ideal_base_value = total_assets * (self.base_asset_target_percentage / 100)
        actual_base_value = base_total * current_price
        
        logger.info(f"使用目標配置比例: {self.base_asset_target_percentage}% {self.base_asset} / {self.quote_asset_target_percentage}% {self.quote_asset}")
        
        # 判斷需要買入還是賣出
        if actual_base_value > ideal_base_value:
            # 基礎資產過多，需要賣出
            excess_value = actual_base_value - ideal_base_value
            quantity_to_sell = excess_value / current_price
            
            
            max_sellable = base_total * 0.95  # 保留5%作為緩衝，基於總餘額
            quantity_to_sell = min(quantity_to_sell, max_sellable)
            quantity_to_sell = round_to_precision(quantity_to_sell, self.base_precision)
            
            if quantity_to_sell < self.min_order_size:
                logger.info(f"需要賣出的數量 {format_balance(quantity_to_sell)} 低於最小訂單大小 {format_balance(self.min_order_size)}，不進行重新平衡")
                return
                
            
            if quantity_to_sell > base_total:
                logger.warning(f"需要賣出 {format_balance(quantity_to_sell)} 但總餘額只有 {format_balance(base_total)}，調整為總餘額的90%")
                quantity_to_sell = round_to_precision(base_total * 0.9, self.base_precision)
            
            # 檢查可用餘額，如果為0則依靠自動贖回
            if base_available < quantity_to_sell:
                logger.info(f"可用餘額 {format_balance(base_available)} 不足，需要賣出 {format_balance(quantity_to_sell)}，將依靠自動贖回功能")
            
            # 使用略低於當前買價的價格來快速成交
            sell_price = round_to_tick_size(bid_price * 0.999, self.tick_size)
            logger.info(f"執行重新平衡: 賣出 {format_balance(quantity_to_sell)} {self.base_asset} @ {format_balance(sell_price)}")
            
            # 構建訂單
            order_details = {
                "orderType": "Limit",
                "price": str(sell_price),
                "quantity": str(quantity_to_sell),
                "side": "Ask",
                "symbol": self.symbol,
                "timeInForce": "IOC",  # 立即成交或取消，避免掛單
                "autoLendRedeem": True,
                "autoLend": True
            }
            
        elif actual_base_value < ideal_base_value:
            # 基礎資產不足，需要買入
            deficit_value = ideal_base_value - actual_base_value
            quantity_to_buy = deficit_value / current_price
            
            # 計算需要的報價資產
            cost = quantity_to_buy * ask_price
            max_affordable_cost = quote_total * 0.95  # 基於總餘額的95%
            max_affordable = max_affordable_cost / ask_price
            quantity_to_buy = min(quantity_to_buy, max_affordable)
            quantity_to_buy = round_to_precision(quantity_to_buy, self.base_precision)
            
            if quantity_to_buy < self.min_order_size:
                logger.info(f"需要買入的數量 {format_balance(quantity_to_buy)} 低於最小訂單大小 {format_balance(self.min_order_size)}，不進行重新平衡")
                return
                
            cost = quantity_to_buy * ask_price
            if cost > quote_total:
                logger.warning(f"需要 {format_balance(cost)} {self.quote_asset} 但總餘額只有 {format_balance(quote_total)}，調整買入數量")
                quantity_to_buy = round_to_precision((quote_total * 0.9) / ask_price, self.base_precision)
                cost = quantity_to_buy * ask_price
            
            # 檢查可用餘額
            if quote_available < cost:
                logger.info(f"可用餘額 {format_balance(quote_available)} {self.quote_asset} 不足，需要 {format_balance(cost)} {self.quote_asset}，將依靠自動贖回功能")
            
            # 使用略高於當前賣價的價格來快速成交
            buy_price = round_to_tick_size(ask_price * 1.001, self.tick_size)
            logger.info(f"執行重新平衡: 買入 {format_balance(quantity_to_buy)} {self.base_asset} @ {format_balance(buy_price)}")
            
            # 構建訂單
            order_details = {
                "orderType": "Limit",
                "price": str(buy_price),
                "quantity": str(quantity_to_buy),
                "side": "Bid",
                "symbol": self.symbol,
                "timeInForce": "IOC",  # 立即成交或取消，避免掛單
                "autoLendRedeem": True,
                "autoLend": True
            }
        else:
            logger.info("倉位已經均衡，無需重新平衡")
            return
        
        # 執行訂單
        result = execute_order(self.api_key, self.secret_key, order_details)
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"重新平衡訂單執行失敗: {result['error']}")
        else:
            logger.info(f"重新平衡訂單執行成功")
            # 記錄這是一個重平衡訂單
            if 'id' in result:
                self.db.record_rebalance_order(result['id'], self.symbol)
        
        logger.info("倉位重新平衡完成")
    
    def subscribe_order_updates(self):
        """訂閲訂單更新流"""
        if not self.ws or not self.ws.is_connected():
            logger.warning("無法訂閲訂單更新：WebSocket連接不可用")
            return False
        
        # 嘗試訂閲訂單更新流
        stream = f"account.orderUpdate.{self.symbol}"
        if stream not in self.ws.subscriptions:
            retry_count = 0
            max_retries = 3
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    success = self.ws.private_subscribe(stream)
                    if success:
                        logger.info(f"成功訂閲訂單更新: {stream}")
                        return True
                    else:
                        logger.warning(f"訂閲訂單更新失敗，嘗試重試... ({retry_count+1}/{max_retries})")
                except Exception as e:
                    logger.error(f"訂閲訂單更新時發生異常: {e}")
                
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)  # 重試前等待
            
            if not success:
                logger.error(f"在 {max_retries} 次嘗試後仍無法訂閲訂單更新")
                return False
        else:
            logger.info(f"已經訂閲了訂單更新: {stream}")
            return True
    
    def place_limit_orders(self):
        """下限價單（使用總餘額包含抵押品）"""
        self.check_ws_connection()
        self.cancel_existing_orders()
        
        buy_prices, sell_prices = self.calculate_prices()
        if buy_prices is None or sell_prices is None:
            logger.error("無法計算訂單價格，跳過下單")
            return
        
        # 處理訂單數量
        if self.order_quantity is None:
            # 獲取總可用餘額（包含抵押品）
            base_available, base_total = self.get_asset_balance(self.base_asset)
            quote_available, quote_total = self.get_asset_balance(self.quote_asset)
            
            logger.info(f"當前總餘額: {format_balance(base_total)} {self.base_asset}, {format_balance(quote_total)} {self.quote_asset}")
            logger.info(f"當前可用餘額: {format_balance(base_available)} {self.base_asset}, {format_balance(quote_available)} {self.quote_asset}")
            
            # 如果可用餘額很少但總餘額充足，說明資金在抵押品中
            if base_available < base_total * 0.1:
                logger.info(f"基礎資產主要在抵押品中，將依靠自動贖回功能")
            if quote_available < quote_total * 0.1:
                logger.info(f"報價資產主要在抵押品中，將依靠自動贖回功能")
            
            # 計算每個訂單的數量
            avg_price = sum(buy_prices) / len(buy_prices)
            
            # 使用更保守的分配比例，避免資金用盡
            allocation_percent = min(0.05, 1.0 / (self.max_orders * 4))  # 最多使用總資金的25%
            
            # 基於總餘額計算，而不是可用餘額
            quote_amount_per_side = quote_total * allocation_percent
            base_amount_per_side = base_total * allocation_percent
            
            buy_quantity = max(self.min_order_size, round_to_precision(quote_amount_per_side / avg_price, self.base_precision))
            sell_quantity = max(self.min_order_size, round_to_precision(base_amount_per_side, self.base_precision))
            
            logger.info(f"計算訂單數量: 買單 {format_balance(buy_quantity)} {self.base_asset}, 賣單 {format_balance(sell_quantity)} {self.base_asset}")
        else:
            buy_quantity = max(self.min_order_size, round_to_precision(self.order_quantity, self.base_precision))
            sell_quantity = max(self.min_order_size, round_to_precision(self.order_quantity, self.base_precision))
        
        # 下買單 (併發處理)
        buy_futures = []

        def place_buy(price):
            qty = self._adjust_quantity_by_market(buy_quantity, 'buy')
            order = {
                "orderType": "Limit",
                "price": str(price),
                "quantity": str(qty),
                "side": "Bid",
                "symbol": self.symbol,
                "timeInForce": "GTC",
                "postOnly": True,
                "autoLendRedeem": True,
                "autoLend": True
            }
            res = execute_order(self.api_key, self.secret_key, order)
            if isinstance(res, dict) and "error" in res and "POST_ONLY_TAKER" in str(res["error"]):
                logger.info("調整買單價格並重試...")
                order["price"] = str(round_to_tick_size(float(order["price"]) - self.tick_size, self.tick_size))
                res = execute_order(self.api_key, self.secret_key, order)
            
            # 特殊處理資金不足錯誤
            if isinstance(res, dict) and "error" in res and "INSUFFICIENT_FUNDS" in str(res["error"]):
                logger.warning(f"買單資金不足，可能需要手動贖回抵押品或等待自動贖回生效")
            
            return qty, order["price"], res

        with ThreadPoolExecutor(max_workers=self.max_orders) as executor:
            for p in buy_prices:
                if len(buy_futures) >= self.max_orders:
                    break
                buy_futures.append(executor.submit(place_buy, p))

        buy_order_count = 0
        for future in buy_futures:
            qty, p_used, res = future.result()
            if isinstance(res, dict) and "error" in res:
                logger.error(f"買單失敗: {res['error']}")
            else:
                logger.info(f"買單成功: 價格 {p_used}, 數量 {qty}")
                self.active_buy_orders.append(res)
                self.orders_placed += 1
                buy_order_count += 1

        # 下賣單
        sell_futures = []

        def place_sell(price):
            qty = self._adjust_quantity_by_market(sell_quantity, 'sell')
            order = {
                "orderType": "Limit",
                "price": str(price),
                "quantity": str(qty),
                "side": "Ask",
                "symbol": self.symbol,
                "timeInForce": "GTC",
                "postOnly": True,
                "autoLendRedeem": True,
                "autoLend": True
            }
            res = execute_order(self.api_key, self.secret_key, order)
            if isinstance(res, dict) and "error" in res and "POST_ONLY_TAKER" in str(res["error"]):
                logger.info("調整賣單價格並重試...")
                order["price"] = str(round_to_tick_size(float(order["price"]) + self.tick_size, self.tick_size))
                res = execute_order(self.api_key, self.secret_key, order)
            
            # 特殊處理資金不足錯誤
            if isinstance(res, dict) and "error" in res and "INSUFFICIENT_FUNDS" in str(res["error"]):
                logger.warning(f"賣單資金不足，可能需要手動贖回抵押品或等待自動贖回生效")
            
            return qty, order["price"], res

        with ThreadPoolExecutor(max_workers=self.max_orders) as executor:
            for p in sell_prices:
                if len(sell_futures) >= self.max_orders:
                    break
                sell_futures.append(executor.submit(place_sell, p))

        sell_order_count = 0
        for future in sell_futures:
            qty, p_used, res = future.result()
            if isinstance(res, dict) and "error" in res:
                logger.error(f"賣單失敗: {res['error']}")
            else:
                logger.info(f"賣單成功: 價格 {p_used}, 數量 {qty}")
                self.active_sell_orders.append(res)
                self.orders_placed += 1
                sell_order_count += 1
            
        logger.info(f"共下單: {buy_order_count} 個買單, {sell_order_count} 個賣單")
    
    def _adjust_quantity_by_market(self, base_quantity, side):
        """根據市場情況動態調整訂單數量"""
        # 直接返回基本數量，不進行任何調整
        return max(self.min_order_size, round_to_precision(base_quantity, self.base_precision))
    
    def cancel_existing_orders(self):
        """取消所有現有訂單"""
        open_orders = get_open_orders(self.api_key, self.secret_key, self.symbol)
        
        if isinstance(open_orders, dict) and "error" in open_orders:
            logger.error(f"獲取訂單失敗: {open_orders['error']}")
            return
        
        if not open_orders:
            logger.info("沒有需要取消的現有訂單")
            self.active_buy_orders = []
            self.active_sell_orders = []
            return
        
        logger.info(f"正在取消 {len(open_orders)} 個現有訂單")
        
        try:
            # 嘗試批量取消
            result = cancel_all_orders(self.api_key, self.secret_key, self.symbol)
            
            if isinstance(result, dict) and "error" in result:
                logger.error(f"批量取消訂單失敗: {result['error']}")
                logger.info("嘗試逐個取消...")
                
                # 初始化線程池
                with ThreadPoolExecutor(max_workers=5) as executor:
                    cancel_futures = []
                    
                    # 提交取消訂單任務
                    for order in open_orders:
                        order_id = order.get('id')
                        if not order_id:
                            continue
                        
                        future = executor.submit(
                            cancel_order, 
                            self.api_key, 
                            self.secret_key, 
                            order_id, 
                            self.symbol
                        )
                        cancel_futures.append((order_id, future))
                    
                    # 處理結果
                    for order_id, future in cancel_futures:
                        try:
                            res = future.result()
                            if isinstance(res, dict) and "error" in res:
                                logger.error(f"取消訂單 {order_id} 失敗: {res['error']}")
                            else:
                                logger.info(f"取消訂單 {order_id} 成功")
                                self.orders_cancelled += 1
                        except Exception as e:
                            logger.error(f"取消訂單 {order_id} 時出錯: {e}")
            else:
                logger.info("批量取消訂單成功")
                self.orders_cancelled += len(open_orders)
        except Exception as e:
            logger.error(f"取消訂單過程中發生錯誤: {str(e)}")
        
        # 等待一下確保訂單已取消
        time.sleep(1)
        
        # 檢查是否還有未取消的訂單
        remaining_orders = get_open_orders(self.api_key, self.secret_key, self.symbol)
        if remaining_orders and len(remaining_orders) > 0:
            logger.warning(f"警告: 仍有 {len(remaining_orders)} 個未取消的訂單")
        else:
            logger.info("所有訂單已成功取消")
        
        # 重置活躍訂單列表
        self.active_buy_orders = []
        self.active_sell_orders = []
    
    def check_order_fills(self):
        """檢查訂單成交情況"""
        open_orders = get_open_orders(self.api_key, self.secret_key, self.symbol)
        
        if isinstance(open_orders, dict) and "error" in open_orders:
            logger.error(f"獲取訂單失敗: {open_orders['error']}")
            return
        
        # 獲取當前所有訂單ID
        current_order_ids = set()
        if open_orders:
            for order in open_orders:
                order_id = order.get('id')
                if order_id:
                    current_order_ids.add(order_id)
        
        # 記錄更新前的訂單數量
        prev_buy_orders = len(self.active_buy_orders)
        prev_sell_orders = len(self.active_sell_orders)
        
        # 更新活躍訂單列表
        active_buy_orders = []
        active_sell_orders = []
        
        if open_orders:
            for order in open_orders:
                if order.get('side') == 'Bid':
                    active_buy_orders.append(order)
                elif order.get('side') == 'Ask':
                    active_sell_orders.append(order)
        
        # 檢查買單成交
        filled_buy_orders = []
        for order in self.active_buy_orders:
            order_id = order.get('id')
            if order_id and order_id not in current_order_ids:
                price = float(order.get('price', 0))
                quantity = float(order.get('quantity', 0))
                logger.info(f"買單已成交: {price} x {quantity}")
                filled_buy_orders.append(order)
        
        # 檢查賣單成交
        filled_sell_orders = []
        for order in self.active_sell_orders:
            order_id = order.get('id')
            if order_id and order_id not in current_order_ids:
                price = float(order.get('price', 0))
                quantity = float(order.get('quantity', 0))
                logger.info(f"賣單已成交: {price} x {quantity}")
                filled_sell_orders.append(order)
        
        # 更新活躍訂單列表
        self.active_buy_orders = active_buy_orders
        self.active_sell_orders = active_sell_orders
        
        # 輸出訂單數量變化，方便追蹤
        if prev_buy_orders != len(active_buy_orders) or prev_sell_orders != len(active_sell_orders):
            logger.info(f"訂單數量變更: 買單 {prev_buy_orders} -> {len(active_buy_orders)}, 賣單 {prev_sell_orders} -> {len(active_sell_orders)}")
        
        logger.info(f"當前活躍訂單: 買單 {len(self.active_buy_orders)} 個, 賣單 {len(self.active_sell_orders)} 個")
    
    def estimate_profit(self):
        """估算潛在利潤"""
        # 計算活躍買賣單的平均價格
        avg_buy_price = 0
        total_buy_quantity = 0
        for order in self.active_buy_orders:
            price = float(order.get('price', 0))
            quantity = float(order.get('quantity', 0))
            avg_buy_price += price * quantity
            total_buy_quantity += quantity
        
        if total_buy_quantity > 0:
            avg_buy_price /= total_buy_quantity
        
        avg_sell_price = 0
        total_sell_quantity = 0
        for order in self.active_sell_orders:
            price = float(order.get('price', 0))
            quantity = float(order.get('quantity', 0))
            avg_sell_price += price * quantity
            total_sell_quantity += quantity
        
        if total_sell_quantity > 0:
            avg_sell_price /= total_sell_quantity
        
        # 計算總的PnL和本次執行的PnL
        realized_pnl, unrealized_pnl, total_fees, net_pnl, session_realized_pnl, session_fees, session_net_pnl = self.calculate_pnl()
        
        # 計算活躍訂單的潛在利潤
        if avg_buy_price > 0 and avg_sell_price > 0:
            spread = avg_sell_price - avg_buy_price
            spread_percentage = (spread / avg_buy_price) * 100
            min_quantity = min(total_buy_quantity, total_sell_quantity)
            potential_profit = spread * min_quantity
            
            logger.info(f"估算利潤: 買入均價 {avg_buy_price:.8f}, 賣出均價 {avg_sell_price:.8f}")
            logger.info(f"價差: {spread:.8f} ({spread_percentage:.2f}%), 潛在利潤: {potential_profit:.8f} {self.quote_asset}")
            logger.info(f"已實現利潤(總): {realized_pnl:.8f} {self.quote_asset}")
            logger.info(f"總手續費(總): {total_fees:.8f} {self.quote_asset}")
            logger.info(f"凈利潤(總): {net_pnl:.8f} {self.quote_asset}")
            logger.info(f"未實現利潤: {unrealized_pnl:.8f} {self.quote_asset}")
            
            # 打印本次執行的統計信息
            logger.info(f"\n---本次執行統計---")
            logger.info(f"本次執行已實現利潤: {session_realized_pnl:.8f} {self.quote_asset}")
            logger.info(f"本次執行手續費: {session_fees:.8f} {self.quote_asset}")
            logger.info(f"本次執行凈利潤: {session_net_pnl:.8f} {self.quote_asset}")
            
            session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
            session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
            
            logger.info(f"本次執行買入量: {session_buy_volume} {self.base_asset}, 賣出量: {session_sell_volume} {self.base_asset}")
            logger.info(f"本次執行Maker買入: {self.session_maker_buy_volume} {self.base_asset}, Maker賣出: {self.session_maker_sell_volume} {self.base_asset}")
            logger.info(f"本次執行Taker買入: {self.session_taker_buy_volume} {self.base_asset}, Taker賣出: {self.session_taker_sell_volume} {self.base_asset}")
            
        else:
            logger.info(f"無法估算潛在利潤: 缺少活躍的買/賣訂單")
            logger.info(f"已實現利潤(總): {realized_pnl:.8f} {self.quote_asset}")
            logger.info(f"總手續費(總): {total_fees:.8f} {self.quote_asset}")
            logger.info(f"凈利潤(總): {net_pnl:.8f} {self.quote_asset}")
            logger.info(f"未實現利潤: {unrealized_pnl:.8f} {self.quote_asset}")
            
            # 打印本次執行的統計信息
            logger.info(f"\n---本次執行統計---")
            logger.info(f"本次執行已實現利潤: {session_realized_pnl:.8f} {self.quote_asset}")
            logger.info(f"本次執行手續費: {session_fees:.8f} {self.quote_asset}")
            logger.info(f"本次執行凈利潤: {session_net_pnl:.8f} {self.quote_asset}")
            
            session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
            session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
            
            logger.info(f"本次執行買入量: {session_buy_volume} {self.base_asset}, 賣出量: {session_sell_volume} {self.base_asset}")
            logger.info(f"本次執行Maker買入: {self.session_maker_buy_volume} {self.base_asset}, Maker賣出: {self.session_maker_sell_volume} {self.base_asset}")
            logger.info(f"本次執行Taker買入: {self.session_taker_buy_volume} {self.base_asset}, Taker賣出: {self.session_taker_sell_volume} {self.base_asset}")
    
    def print_trading_stats(self):
        """打印交易統計報表"""
        try:
            logger.info("\n=== 做市商交易統計 ===")
            logger.info(f"交易對: {self.symbol}")
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 獲取今天的統計數據
            today_stats = self.db.get_trading_stats(self.symbol, today)
            
            if today_stats and len(today_stats) > 0:
                stat = today_stats[0]
                maker_buy = stat['maker_buy_volume']
                maker_sell = stat['maker_sell_volume']
                taker_buy = stat['taker_buy_volume']
                taker_sell = stat['taker_sell_volume']
                profit = stat['realized_profit']
                fees = stat['total_fees']
                net = stat['net_profit']
                avg_spread = stat['avg_spread']
                volatility = stat['volatility']
                
                total_volume = maker_buy + maker_sell + taker_buy + taker_sell
                maker_percentage = ((maker_buy + maker_sell) / total_volume * 100) if total_volume > 0 else 0
                
                logger.info(f"\n今日統計 ({today}):")
                logger.info(f"Maker買入量: {maker_buy} {self.base_asset}")
                logger.info(f"Maker賣出量: {maker_sell} {self.base_asset}")
                logger.info(f"Taker買入量: {taker_buy} {self.base_asset}")
                logger.info(f"Taker賣出量: {taker_sell} {self.base_asset}")
                logger.info(f"總成交量: {total_volume} {self.base_asset}")
                logger.info(f"Maker佔比: {maker_percentage:.2f}%")
                logger.info(f"平均價差: {avg_spread:.4f}%")
                logger.info(f"波動率: {volatility:.4f}%")
                logger.info(f"毛利潤: {profit:.8f} {self.quote_asset}")
                logger.info(f"總手續費: {fees:.8f} {self.quote_asset}")
                logger.info(f"凈利潤: {net:.8f} {self.quote_asset}")
            
            # 獲取所有時間的總計
            all_time_stats = self.db.get_all_time_stats(self.symbol)
            
            if all_time_stats:
                total_maker_buy = all_time_stats['total_maker_buy']
                total_maker_sell = all_time_stats['total_maker_sell']
                total_taker_buy = all_time_stats['total_taker_buy']
                total_taker_sell = all_time_stats['total_taker_sell']
                total_profit = all_time_stats['total_profit']
                total_fees = all_time_stats['total_fees']
                total_net = all_time_stats['total_net_profit']
                avg_spread = all_time_stats['avg_spread_all_time']
                
                total_volume = total_maker_buy + total_maker_sell + total_taker_buy + total_taker_sell
                maker_percentage = ((total_maker_buy + total_maker_sell) / total_volume * 100) if total_volume > 0 else 0
                
                logger.info(f"\n累計統計:")
                logger.info(f"Maker買入量: {total_maker_buy} {self.base_asset}")
                logger.info(f"Maker賣出量: {total_maker_sell} {self.base_asset}")
                logger.info(f"Taker買入量: {total_taker_buy} {self.base_asset}")
                logger.info(f"Taker賣出量: {total_taker_sell} {self.base_asset}")
                logger.info(f"總成交量: {total_volume} {self.base_asset}")
                logger.info(f"Maker佔比: {maker_percentage:.2f}%")
                logger.info(f"平均價差: {avg_spread:.4f}%")
                logger.info(f"毛利潤: {total_profit:.8f} {self.quote_asset}")
                logger.info(f"總手續費: {total_fees:.8f} {self.quote_asset}")
                logger.info(f"凈利潤: {total_net:.8f} {self.quote_asset}")
            
            # 添加本次執行的統計
            session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
            session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
            session_total_volume = session_buy_volume + session_sell_volume
            session_maker_volume = self.session_maker_buy_volume + self.session_maker_sell_volume
            session_maker_percentage = (session_maker_volume / session_total_volume * 100) if session_total_volume > 0 else 0
            session_profit = self._calculate_session_profit()
            
            logger.info(f"\n本次執行統計 (從 {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')} 開始):")
            logger.info(f"Maker買入量: {self.session_maker_buy_volume} {self.base_asset}")
            logger.info(f"Maker賣出量: {self.session_maker_sell_volume} {self.base_asset}")
            logger.info(f"Taker買入量: {self.session_taker_buy_volume} {self.base_asset}")
            logger.info(f"Taker賣出量: {self.session_taker_sell_volume} {self.base_asset}")
            logger.info(f"總成交量: {session_total_volume} {self.base_asset}")
            logger.info(f"Maker佔比: {session_maker_percentage:.2f}%")
            logger.info(f"毛利潤: {session_profit:.8f} {self.quote_asset}")
            logger.info(f"總手續費: {self.session_fees:.8f} {self.quote_asset}")
            logger.info(f"凈利潤: {(session_profit - self.session_fees):.8f} {self.quote_asset}")
            
            # 添加重平設置信息
            logger.info(f"\n重平設置:")
            logger.info(f"重平功能: {'開啟' if self.enable_rebalance else '關閉'}")
            if self.enable_rebalance:
                logger.info(f"目標比例: {self.base_asset_target_percentage}% {self.base_asset} / {self.quote_asset_target_percentage}% {self.quote_asset}")
                logger.info(f"觸發閾值: {self.rebalance_threshold}%")
                
            # 查詢前10筆最新成交
            recent_trades = self.db.get_recent_trades(self.symbol, 10)
            
            if recent_trades and len(recent_trades) > 0:
                logger.info("\n最近10筆成交:")
                for i, trade in enumerate(recent_trades):
                    maker_str = "Maker" if trade['maker'] else "Taker"
                    logger.info(f"{i+1}. {trade['timestamp']} - {trade['side']} {trade['quantity']} @ {trade['price']} ({maker_str}) 手續費: {trade['fee']:.8f}")
        
        except Exception as e:
            logger.error(f"打印交易統計時出錯: {e}")
    
    def _ensure_data_streams(self):
        """確保所有必要的數據流訂閲都是活躍的"""
        # 檢查深度流訂閲
        if "depth" not in self.ws.subscriptions:
            logger.info("重新訂閲深度數據流...")
            self.ws.initialize_orderbook()  # 重新初始化訂單簿
            self.ws.subscribe_depth()
        
        # 檢查行情數據訂閲
        if "bookTicker" not in self.ws.subscriptions:
            logger.info("重新訂閲行情數據...")
            self.ws.subscribe_bookTicker()
        
        # 檢查私有訂單更新流
        if f"account.orderUpdate.{self.symbol}" not in self.ws.subscriptions:
            logger.info("重新訂閲私有訂單更新流...")
            self.subscribe_order_updates()
    
    def run(self, duration_seconds=3600, interval_seconds=60):
        """執行做市策略"""
        logger.info(f"開始運行做市策略: {self.symbol}")
        logger.info(f"運行時間: {duration_seconds} 秒, 間隔: {interval_seconds} 秒")
        
        # 打印重平設置
        logger.info(f"重平功能: {'開啟' if self.enable_rebalance else '關閉'}")
        if self.enable_rebalance:
            logger.info(f"重平目標比例: {self.base_asset_target_percentage}% {self.base_asset} / {self.quote_asset_target_percentage}% {self.quote_asset}")
            logger.info(f"重平觸發閾值: {self.rebalance_threshold}%")
        
        # 重置本次執行的統計數據
        self.session_start_time = datetime.now()
        self.session_buy_trades = []
        self.session_sell_trades = []
        self.session_fees = 0.0
        self.session_maker_buy_volume = 0.0
        self.session_maker_sell_volume = 0.0
        self.session_taker_buy_volume = 0.0
        self.session_taker_sell_volume = 0.0
        
        start_time = time.time()
        iteration = 0
        last_report_time = start_time
        report_interval = 300  # 5分鐘打印一次報表
        
        try:
            # 先確保 WebSocket 連接可用
            connection_status = self.check_ws_connection()
            if connection_status:
                # 初始化訂單簿和數據流
                if not self.ws.orderbook["bids"] and not self.ws.orderbook["asks"]:
                    self.ws.initialize_orderbook()
                
                # 檢查並確保所有數據流訂閱
                if "depth" not in self.ws.subscriptions:
                    self.ws.subscribe_depth()
                if "bookTicker" not in self.ws.subscriptions:
                    self.ws.subscribe_bookTicker()
                if f"account.orderUpdate.{self.symbol}" not in self.ws.subscriptions:
                    self.subscribe_order_updates()
            
            while time.time() - start_time < duration_seconds:
                iteration += 1
                current_time = time.time()
                logger.info(f"\n=== 第 {iteration} 次迭代 ===")
                logger.info(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 檢查連接並在必要時重連
                connection_status = self.check_ws_connection()
                
                # 如果連接成功，檢查並確保所有流訂閱
                if connection_status:
                    # 重新訂閱必要的數據流
                    self._ensure_data_streams()
                
                # 檢查訂單成交情況
                self.check_order_fills()
                
                # 檢查是否需要重平衡倉位
                if self.need_rebalance():
                    self.rebalance_position()
                
                # 下限價單
                self.place_limit_orders()
                
                # 估算利潤
                self.estimate_profit()
                
                # 定期打印交易統計報表
                if current_time - last_report_time >= report_interval:
                    self.print_trading_stats()
                    last_report_time = current_time
                
                # 計算總的PnL和本次執行的PnL
                realized_pnl, unrealized_pnl, total_fees, net_pnl, session_realized_pnl, session_fees, session_net_pnl = self.calculate_pnl()
                
                logger.info(f"\n統計信息:")
                logger.info(f"總交易次數: {self.trades_executed}")
                logger.info(f"總下單次數: {self.orders_placed}")
                logger.info(f"總取消訂單次數: {self.orders_cancelled}")
                logger.info(f"買入總量: {self.total_bought} {self.base_asset}")
                logger.info(f"賣出總量: {self.total_sold} {self.base_asset}")
                logger.info(f"Maker買入: {self.maker_buy_volume} {self.base_asset}, Maker賣出: {self.maker_sell_volume} {self.base_asset}")
                logger.info(f"Taker買入: {self.taker_buy_volume} {self.base_asset}, Taker賣出: {self.taker_sell_volume} {self.base_asset}")
                logger.info(f"總手續費: {total_fees:.8f} {self.quote_asset}")
                logger.info(f"已實現利潤: {realized_pnl:.8f} {self.quote_asset}")
                logger.info(f"凈利潤: {net_pnl:.8f} {self.quote_asset}")
                logger.info(f"未實現利潤: {unrealized_pnl:.8f} {self.quote_asset}")
                logger.info(f"WebSocket連接狀態: {'已連接' if self.ws and self.ws.is_connected() else '未連接'}")
                
                # 打印本次執行的統計數據
                logger.info(f"\n---本次執行統計---")
                session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
                session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
                logger.info(f"買入量: {session_buy_volume} {self.base_asset}, 賣出量: {session_sell_volume} {self.base_asset}")
                logger.info(f"Maker買入: {self.session_maker_buy_volume} {self.base_asset}, Maker賣出: {self.session_maker_sell_volume} {self.base_asset}")
                logger.info(f"Taker買入: {self.session_taker_buy_volume} {self.base_asset}, Taker賣出: {self.session_taker_sell_volume} {self.base_asset}")
                logger.info(f"本次執行已實現利潤: {session_realized_pnl:.8f} {self.quote_asset}")
                logger.info(f"本次執行手續費: {session_fees:.8f} {self.quote_asset}")
                logger.info(f"本次執行凈利潤: {session_net_pnl:.8f} {self.quote_asset}")
                
                wait_time = interval_seconds
                logger.info(f"等待 {wait_time} 秒後進行下一次迭代...")
                time.sleep(wait_time)
                
            # 結束運行時打印最終報表
            logger.info("\n=== 做市策略運行結束 ===")
            self.print_trading_stats()
            
            # 打印本次執行的最終統計摘要
            logger.info("\n=== 本次執行統計摘要 ===")
            session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
            session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
            session_total_volume = session_buy_volume + session_sell_volume
            session_profit = self._calculate_session_profit()
            
            # 計算執行時間
            td = datetime.now() - self.session_start_time
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            run_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            logger.info(f"執行時間: {run_time}")
            
            logger.info(f"總成交量: {session_total_volume} {self.base_asset}")
            logger.info(f"買入量: {session_buy_volume} {self.base_asset}, 賣出量: {session_sell_volume} {self.base_asset}")
            logger.info(f"Maker買入: {self.session_maker_buy_volume} {self.base_asset}, Maker賣出: {self.session_maker_sell_volume} {self.base_asset}")
            logger.info(f"Taker買入: {self.session_taker_buy_volume} {self.base_asset}, Taker賣出: {self.session_taker_sell_volume} {self.base_asset}")
            logger.info(f"已實現利潤: {session_profit:.8f} {self.quote_asset}")
            logger.info(f"總手續費: {self.session_fees:.8f} {self.quote_asset}")
            logger.info(f"凈利潤: {(session_profit - self.session_fees):.8f} {self.quote_asset}")
            
            if session_total_volume > 0:
                logger.info(f"每單位成交量利潤: {((session_profit - self.session_fees) / session_total_volume):.8f} {self.quote_asset}/{self.base_asset}")
        
        except KeyboardInterrupt:
            logger.info("\n用户中斷，停止做市")
            
            # 中斷時也打印本次執行的統計數據
            logger.info("\n=== 本次執行統計摘要(中斷) ===")
            session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
            session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
            session_total_volume = session_buy_volume + session_sell_volume
            session_profit = self._calculate_session_profit()
            
            # 計算執行時間
            td = datetime.now() - self.session_start_time
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            run_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            logger.info(f"執行時間: {run_time}")
            
            logger.info(f"總成交量: {session_total_volume} {self.base_asset}")
            logger.info(f"買入量: {session_buy_volume} {self.base_asset}, 賣出量: {session_sell_volume} {self.base_asset}")
            logger.info(f"Maker買入: {self.session_maker_buy_volume} {self.base_asset}, Maker賣出: {self.session_maker_sell_volume} {self.base_asset}")
            logger.info(f"Taker買入: {self.session_taker_buy_volume} {self.base_asset}, Taker賣出: {self.session_taker_sell_volume} {self.base_asset}")
            logger.info(f"已實現利潤: {session_profit:.8f} {self.quote_asset}")
            logger.info(f"總手續費: {self.session_fees:.8f} {self.quote_asset}")
            logger.info(f"凈利潤: {(session_profit - self.session_fees):.8f} {self.quote_asset}")
            
            if session_total_volume > 0:
                logger.info(f"每單位成交量利潤: {((session_profit - self.session_fees) / session_total_volume):.8f} {self.quote_asset}/{self.base_asset}")
        
        finally:
            logger.info("取消所有未成交訂單...")
            self.cancel_existing_orders()
            
            # 關閉 WebSocket
            if self.ws:
                self.ws.close()
            
            # 關閉數據庫連接
            if self.db:
                self.db.close()
                logger.info("數據庫連接已關閉")