"""
做市策略模塊
"""
import time
import threading
import unicodedata
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Deque
from concurrent.futures import ThreadPoolExecutor

from api.bp_client import BPClient
from api.aster_client import AsterClient
from api.lighter_client import LighterClient
from ws_client import BackpackWebSocket
from database.db import Database
from utils.helpers import round_to_precision, round_to_tick_size, calculate_volatility, format_quantity
from logger import setup_logger
import traceback

logger = setup_logger("market_maker")

def format_balance(value, decimals=8, threshold=1e-8) -> str:
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
        exchange='backpack',
        exchange_config=None,
        enable_database=False
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_spread_percentage = base_spread_percentage
        self.order_quantity = order_quantity
        self.exchange = exchange
        self.exchange_config = exchange_config or {}
        
        # 初始化交易所客户端
        if exchange == 'backpack':
            self.client = BPClient(self.exchange_config)
        elif exchange == 'aster':
            self.client = AsterClient(self.exchange_config)
        elif exchange == 'paradex':
            from api.paradex_client import ParadexClient
            self.client = ParadexClient(self.exchange_config)
        elif exchange == 'lighter':
            self.client = LighterClient(self.exchange_config)
        elif exchange == 'apex':
            from api.apex_client import ApexClient
            self.client = ApexClient(self.exchange_config)
        else:
            raise ValueError(f"不支持的交易所: {exchange}")
            
        self.max_orders = max_orders
        self.rebalance_threshold = rebalance_threshold
        
        # 新增重平設置參數
        self.enable_rebalance = enable_rebalance
        self.base_asset_target_percentage = base_asset_target_percentage
        self.quote_asset_target_percentage = 100.0 - base_asset_target_percentage

        # 初始化數據庫
        self.db_enabled = bool(enable_database)
        self.db = None
        if self.db_enabled:
            self.db = db_instance if db_instance else Database()
        elif db_instance and hasattr(db_instance, 'close'):
            try:
                db_instance.close()
            except Exception:
                pass

        if not self.db:
            self.db_enabled = False

        if not self.db_enabled:
            logger.info("資料庫寫入功能已關閉，本次執行僅在記憶體中追蹤交易統計。")
        
        # 統計屬性
        self.session_start_time = datetime.now()
        self.session_buy_trades = []
        self.session_sell_trades = []

        # 停止標誌
        self._stop_flag = False
        self.session_fees = 0.0
        self.session_maker_buy_volume = 0.0
        self.session_maker_sell_volume = 0.0
        self.session_taker_buy_volume = 0.0
        self.session_taker_sell_volume = 0.0
        self.session_quote_volume = 0.0

        # 初始化市場限制
        market_limits_response = self.client.get_market_limits(symbol)
        if not market_limits_response.success:
            raise ValueError(f"無法獲取 {symbol} 的市場限制: {market_limits_response.error_message}")
        
        market_info = market_limits_response.data
        if not market_info:
            raise ValueError(f"無法獲取 {symbol} 的市場限制")
        
        # 保存原始數據供後續使用（支援 MarketInfo dataclass 或 dict）
        if hasattr(market_info, 'raw') and market_info.raw:
            self.market_limits = market_info.raw
        elif isinstance(market_info, dict):
            self.market_limits = market_info
        else:
            # MarketInfo dataclass，轉換為 dict 方便存取
            self.market_limits = {
                'base_asset': getattr(market_info, 'base_asset', symbol),
                'quote_asset': getattr(market_info, 'quote_asset', ''),
                'base_precision': getattr(market_info, 'base_precision', 8),
                'quote_precision': getattr(market_info, 'quote_precision', 8),
                'min_order_size': getattr(market_info, 'min_order_size', 0),
                'tick_size': getattr(market_info, 'tick_size', 0.00000001),
            }
        
        # 從 MarketInfo dataclass 或 dict 取值
        if hasattr(market_info, 'base_asset'):
            self.base_asset = market_info.base_asset
            self.quote_asset = market_info.quote_asset
            self.base_precision = market_info.base_precision
            self.quote_precision = market_info.quote_precision
            self.min_order_size = float(market_info.min_order_size)
            self.tick_size = float(market_info.tick_size)
        else:
            self.base_asset = market_info.get('base_asset', symbol)
            self.quote_asset = market_info.get('quote_asset', '')
            self.base_precision = market_info.get('base_precision', 8)
            self.quote_precision = market_info.get('quote_precision', 8)
            self.min_order_size = float(market_info.get('min_order_size', 0))
            self.tick_size = float(market_info.get('tick_size', 0.00000001))
        
        # 交易量統計
        self.maker_buy_volume = 0
        self.maker_sell_volume = 0
        self.taker_buy_volume = 0
        self.taker_sell_volume = 0
        self.total_quote_volume = 0.0
        self.total_fees = 0

        # 關鍵：在任何可能出錯的代碼之前初始化這些屬性
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

        # 風控狀態
        self._stop_trading = False
        self.stop_reason: Optional[str] = None

        # WebSocket 重連冷卻時間追蹤
        self._last_reconnect_attempt = 0

        # 添加代理參數
        # 建立WebSocket連接（僅對Backpack）
        if exchange == 'backpack':
            self.ws = BackpackWebSocket(api_key, secret_key, symbol, self.on_ws_message, auto_reconnect=True)
            self.ws.connect()
        elif exchange == 'xx':
            ...
            self.ws = None
        else:
            self.ws = None  # 不使用WebSocket
        # 執行緒池用於後台任務
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Aster REST 成交流處理狀態
        self._fill_history_bootstrapped = False
        self._processed_fill_ids: Set[str] = set()
        self._recent_fill_ids: Deque[str] = deque(maxlen=500)
        self._last_fill_timestamp: int = 0

        # 等待WebSocket連接建立並進行初始化訂閲
        self._initialize_websocket()

        # 載入交易統計和歷史交易
        self._load_trading_stats()
        self._load_recent_trades()

        # 針對無 WebSocket 的交易所使用 REST 成交同步
        if self.exchange in ('aster', 'lighter', 'apex'):
            self._bootstrap_fill_history()
        
        logger.info(f"初始化做市商: {symbol}")
        logger.info(f"基礎資產: {self.base_asset}, 報價資產: {self.quote_asset}")
        logger.info(f"基礎精度: {self.base_precision}, 報價精度: {self.quote_precision}")
        logger.info(f"最小訂單大小: {self.min_order_size}, 價格步長: {self.tick_size}")
        logger.info(f"基礎價差百分比: {self.base_spread_percentage}%, 最大訂單數: {self.max_orders}")
        logger.info(f"重平功能: {'開啟' if self.enable_rebalance else '關閉'}")
        if self.enable_rebalance:
            logger.info(f"重平目標比例: {self.base_asset_target_percentage}% {self.base_asset} / {self.quote_asset_target_percentage}% {self.quote_asset}")
            logger.info(f"重平觸發閾值: {self.rebalance_threshold}%")

    def _db_available(self) -> bool:
        """檢查資料庫功能是否啟用且可用。"""
        return self.db_enabled and self.db is not None

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
            balance_response = self.client.get_balance()
            logger.debug(f"普通餘額原始數據: {balance_response.raw}")
            if not balance_response.success:
                logger.error(f"獲取普通餘額失敗: {balance_response.error_message}")
                return None

            # 獲取抵押品餘額
            collateral_response = self.client.get_collateral()
            logger.debug(f"抵押品原始數據: {collateral_response.raw}")
            if not collateral_response.success:
                logger.warning(f"獲取抵押品餘額失敗: {collateral_response.error_message}")
                collateral_assets = []
            else:
                # 從 raw 數據中提取抵押品資產
                raw_data = collateral_response.raw or {}
                collateral_assets = raw_data.get('assets') or raw_data.get('collateral', [])

            logger.debug(f"抵押品資產列表: {collateral_assets}")

            # 初始化總餘額字典
            total_balances = {}

            # 處理普通餘額 (List[BalanceInfo])
            for balance_info in balance_response.data:
                asset = balance_info.asset
                available = float(balance_info.available or 0)
                locked = float(balance_info.locked or 0)
                total_balances[asset] = {
                    'available': available,
                    'locked': locked,
                    'total': available + locked,
                    'collateral_available': 0,
                    'collateral_total': 0
                }

            logger.debug(f"處理普通餘額後的 total_balances: {total_balances}")

            # 添加抵押品餘額
            for item in collateral_assets:
                symbol = item.get('symbol', '')
                if symbol:
                    total_quantity = float(item.get('totalQuantity', 0))
                    available_quantity = float(item.get('availableQuantity', 0))
                    lend_quantity = float(item.get('lendQuantity', 0))

                    logger.debug(f"處理抵押品: {symbol}, 總量={total_quantity}, 可用={available_quantity}, 借出={lend_quantity}")

                    # 對於 Backpack 抵押品，使用 totalQuantity 作為可用餘額
                    # 因為借貸中的資產（lendQuantity）也可以用於交易
                    effective_available = total_quantity

                    if symbol not in total_balances:
                        total_balances[symbol] = {
                            'available': 0,
                            'locked': 0,
                            'total': 0,
                            'collateral_available': effective_available,
                            'collateral_total': total_quantity
                        }
                    else:
                        total_balances[symbol]['collateral_available'] = effective_available
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

            logger.debug(f"最終 total_balances 資產列表: {list(total_balances.keys())}")

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
        if self.ws is None:
            logger.info("使用 REST API 模式（無 WebSocket）")
            return

        wait_time = 0
        max_wait_time = 2  # 減少等待時間從 10 秒到 2 秒
        check_interval = 0.2  # 減少檢查間隔從 0.5 秒到 0.2 秒

        while not self.ws.connected and wait_time < max_wait_time:
            time.sleep(check_interval)
            wait_time += check_interval

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
            logger.info("WebSocket 初始連接未建立，使用 REST API 模式（WebSocket 將在後台自動重連）")
    
    def _load_trading_stats(self):
        """從數據庫加載交易統計數據"""
        if not self._db_available():
            logger.debug("資料庫未啟用，跳過交易統計載入。")
            return
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
        if not self._db_available():
            logger.debug("資料庫未啟用，跳過歷史成交載入。")
            return
        try:
            # 獲取訂單歷史
            trades = self.db.get_order_history(self.symbol, 1000)
            trades_count = len(trades) if trades else 0
            
            if trades_count > 0:
                for side, quantity, price, maker, fee in trades:
                    quantity = float(quantity)
                    price = float(price)
                    fee = float(fee)
                    quote_volume = abs(quantity * price)

                    if side == 'Bid':  # 買入
                        self.buy_trades.append((price, quantity))
                        self.total_bought += quantity
                        self.total_quote_volume += quote_volume
                        if maker:
                            self.maker_buy_volume += quantity
                        else:
                            self.taker_buy_volume += quantity
                    elif side == 'Ask':  # 賣出
                        self.sell_trades.append((price, quantity))
                        self.total_sold += quantity
                        self.total_quote_volume += quote_volume
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

    # ------------------------------------------------------------------
    # Aster REST 成交同步相關方法
    # ------------------------------------------------------------------
    def _bootstrap_fill_history(self) -> None:
        """初始化 REST 成交歷史，避免重複計數"""
        exchange_label = self.exchange.capitalize()
        try:
            self._sync_fill_history(bootstrap=True)
            logger.info("%s 成交歷史初始化完成，開始追蹤新成交", exchange_label)
        except Exception as e:
            logger.error(f"初始化 {exchange_label} 成交歷史時出錯: {e}")

    def _sync_fill_history(self, bootstrap: bool = False) -> None:
        """透過 REST API 同步最新成交"""
        if self.exchange not in ('aster', 'lighter', 'apex'):
            return

        exchange_label = self.exchange.capitalize()

        try:
            # APEX API 限制 limit 最大為 100
            limit = 100 if self.exchange == 'apex' else 200
            response = self.client.get_fill_history(self.symbol, limit=limit)
        except Exception as e:
            logger.error(f"獲取 {exchange_label} 成交歷史時出錯: {e}")
            return

        fills = self._normalize_fill_history_response(response)
        if not fills:
            return

        fills.sort(key=lambda item: item.get('timestamp', 0))

        if bootstrap or not self._fill_history_bootstrapped:
            for fill in fills:
                self._register_processed_fill(fill.get('fill_id'), fill.get('timestamp', 0))
            self._fill_history_bootstrapped = True
            return

        for fill in fills:
            fill_id = fill.get('fill_id')
            timestamp = fill.get('timestamp', 0)

            if self._has_seen_fill(fill_id, timestamp):
                continue

            order_id = fill.get('order_id')
            side = fill.get('side')
            quantity = fill.get('quantity')
            price = fill.get('price')

            if not order_id or quantity is None or price is None:
                continue

            maker = fill.get('is_maker', True)
            # 確保 fee 是有效數字，因為 fill.get('fee', 0.0) 在 key 存在但值為 None 時返回 None
            fee_raw = fill.get('fee')
            fee = float(fee_raw) if fee_raw is not None else 0.0
            fee_asset = fill.get('fee_asset') or self.quote_asset

            normalized_side = None
            if isinstance(side, str):
                side_upper = side.upper()
                if side_upper in ('BUY', 'BID'):
                    normalized_side = 'Bid'
                elif side_upper in ('SELL', 'ASK'):
                    normalized_side = 'Ask'

            if normalized_side is None:
                continue

            self._register_processed_fill(fill_id, timestamp)
            self._process_order_fill_event(
                side=normalized_side,
                quantity=quantity,
                price=price,
                order_id=order_id,
                maker=maker,
                fee=fee,
                fee_asset=fee_asset,
                trade_id=fill_id,
                source='rest',
                timestamp=timestamp,
                register_processed=False,
                client_id=fill.get('client_id'),  # APEX 使用 clientId
            )

    def _normalize_fill_history_response(self, response) -> List[Dict[str, Any]]:
        """將 REST API 回傳的成交資料轉換為統一格式"""
        # 支援新的 ApiResponse 格式
        if hasattr(response, 'success'):
            if not response.success:
                logger.error(f"獲取成交歷史失敗: {response.error_message}")
                return []
            # 如果 data 是空列表，直接返回
            if response.data is not None and isinstance(response.data, list):
                if len(response.data) == 0:
                    return []
                # 如果是 List[TradeInfo]，直接轉換
                if hasattr(response.data[0], 'trade_id'):
                    result = []
                    for t in response.data:
                        # 從 raw 中提取 clientId（APEX 用 clientId 追蹤訂單）
                        raw = t.raw if hasattr(t, 'raw') and t.raw else {}
                        client_id = raw.get('clientId') or raw.get('client_id') or raw.get('clientOrderId')
                        result.append({
                            'fill_id': t.trade_id,
                            'order_id': t.order_id,
                            'client_id': str(client_id) if client_id else None,  # 從 raw 提取 clientId
                            'symbol': t.symbol,
                            'side': t.side,
                            'price': float(t.price) if t.price is not None else None,
                            'quantity': float(t.size) if t.size is not None else None,  # TradeInfo 使用 size
                            'fee': float(t.fee) if t.fee is not None else 0.0,  # fee 可以是 0，不能用 if t.fee 判斷
                            'fee_asset': t.fee_asset,
                            'is_maker': t.is_maker,
                            'timestamp': t.timestamp,
                        })
                    return result
            data = response.raw
        else:
            data = response

        if isinstance(data, dict):
            # APEX 格式: {"data": {"orders": [...]}}
            inner_data = data.get('data', data)
            if isinstance(inner_data, dict):
                # APEX 成交在 orders 字段中
                data = inner_data.get('orders', inner_data.get('fills', inner_data))
            else:
                data = inner_data

        if not isinstance(data, list):
            # 只在非空 dict（不是純狀態響應）時顯示警告
            if isinstance(data, dict) and data.get('orders') is None and data.get('fills') is None:
                # APEX 返回 {"code": 0, "msg": "", ...} 表示沒有成交，這是正常的
                return []
            logger.warning(f"成交歷史返回格式異常: {type(data)}")
            return []

        fills: List[Dict[str, Any]] = []

        def _extract(entry: Dict[str, Any], *keys: str) -> Any:
            for key in keys:
                if key in entry and entry[key] not in (None, ""):
                    return entry[key]
            return None

        def _to_float(value: Any) -> Optional[float]:
            if value in (None, "", "NaN"):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        for entry in data:
            if not isinstance(entry, dict):
                continue

            fill_id = _extract(
                entry,
                "id",
                "fillId",
                "fill_id",
                "tradeId",
                "trade_id",
                "executionId",
                "execution_id",
                "matchFillId",  # APEX
                "t",
            )
            order_id = _extract(
                entry,
                "orderId",
                "order_id",
                "orderIndex",
                "order_index",
                "ask_id",
                "bid_id",
                "i",
            )
            side = _extract(entry, "side", "S")
            price = _to_float(_extract(entry, "price", "p", "L"))
            quantity = _to_float(_extract(entry, "quantity", "qty", "q", "l", "size"))
            fee_asset = _extract(
                entry,
                "fee_asset",
                "feeAsset",
                "commissionAsset",
                "N",
                "fee_currency",
                "feeCurrency",
            )
            # APEX 使用 direction: MAKER/TAKER
            maker_flag = _extract(entry, "maker", "isMaker", "m", "is_maker", "direction")
            timestamp_raw = _extract(entry, "time", "timestamp", "T", "ts", "createdAt", "updatedTime")

            maker_fee = _to_float(_extract(entry, "maker_fee", "makerFee"))
            taker_fee = _to_float(_extract(entry, "taker_fee", "takerFee"))
            fee_primary = _extract(entry, "fee", "commission", "n", "fee_value")
            fee_value = _to_float(fee_primary)

            derived_maker_flag: Optional[bool] = None
            if maker_fee is not None and abs(maker_fee) > 0:
                derived_maker_flag = True
            elif taker_fee is not None and abs(taker_fee) > 0:
                derived_maker_flag = False

            is_maker = True
            if isinstance(maker_flag, bool):
                is_maker = maker_flag
            elif maker_flag is not None:
                maker_str = str(maker_flag).lower()
                # APEX uses direction: MAKER/TAKER
                is_maker = maker_str in ("true", "1", "yes", "maker")
            elif derived_maker_flag is not None:
                is_maker = derived_maker_flag

            if fee_value is None:
                if is_maker and maker_fee is not None:
                    fee_value = maker_fee
                elif not is_maker and taker_fee is not None:
                    fee_value = taker_fee
                elif maker_fee is not None:
                    fee_value = maker_fee
                elif taker_fee is not None:
                    fee_value = taker_fee

            if fee_value is None:
                fee_value = 0.0

            try:
                timestamp_value = int(float(timestamp_raw)) if timestamp_raw is not None else 0
            except (TypeError, ValueError):
                timestamp_value = 0

            # APEX: 提取 clientId/clientOrderId 作為備用 ID（下單時使用的是 clientId）
            client_id = _extract(
                entry,
                "clientId",
                "client_id",
                "clientOrderId",
                "client_order_id",
            )

            fills.append({
                'fill_id': str(fill_id) if fill_id is not None else None,
                'order_id': str(order_id) if order_id is not None else None,
                'client_id': str(client_id) if client_id is not None else None,  # APEX 用 clientId 追蹤訂單
                'side': side,
                'price': price,
                'quantity': quantity,
                'fee': fee_value,
                'fee_asset': fee_asset,
                'is_maker': is_maker,
                'timestamp': timestamp_value,
            })

        return fills

    def _has_seen_fill(self, fill_id: Optional[str], timestamp: int) -> bool:
        """判斷成交是否已處理"""
        if fill_id and fill_id in self._processed_fill_ids:
            return True
        if (not fill_id or fill_id is None) and timestamp and timestamp <= self._last_fill_timestamp:
            return True
        return False

    def _register_processed_fill(self, fill_id: Optional[str], timestamp: int) -> None:
        """將成交標記為已處理"""
        if fill_id:
            if len(self._recent_fill_ids) >= self._recent_fill_ids.maxlen:
                oldest = self._recent_fill_ids.popleft()
                if oldest in self._processed_fill_ids:
                    self._processed_fill_ids.remove(oldest)
            self._recent_fill_ids.append(fill_id)
            self._processed_fill_ids.add(fill_id)

        if timestamp:
            self._last_fill_timestamp = max(self._last_fill_timestamp, timestamp)

    def _process_order_fill_event(
        self,
        *,
        side: str,
        quantity: float,
        price: float,
        order_id: Optional[str],
        maker: bool,
        fee: float,
        fee_asset: Optional[str],
        trade_id: Optional[str] = None,
        source: str = 'ws',
        timestamp: Optional[int] = None,
        register_processed: bool = True,
        client_id: Optional[str] = None,  # APEX 使用 clientId 追蹤訂單
    ) -> None:
        """統一處理成交事件來源 (WebSocket/REST)"""

        if register_processed:
            self._register_processed_fill(trade_id, timestamp or 0)

        # 確保 fee 是有效數字，避免 NoneType 格式化錯誤
        if fee is None:
            fee = 0.0
        else:
            try:
                fee = float(fee)
            except (TypeError, ValueError):
                fee = 0.0

        fee_asset = fee_asset or self.quote_asset

        normalized_side = side
        if isinstance(side, str):
            side_upper = side.upper()
            if side_upper in ("BUY", "BID"):
                normalized_side = "Bid"
            elif side_upper in ("SELL", "ASK"):
                normalized_side = "Ask"

        logger.info(
            f"訂單成交[{source}]: ID={order_id}, 方向={normalized_side}, 數量={quantity}, 價格={price}, Maker={maker}, 手續費={fee:.8f}"
        )

        trade_type = 'market_making'
        if order_id and self._db_available():
            try:
                if self.db.is_rebalance_order(order_id, self.symbol):
                    trade_type = 'rebalance'
            except Exception as db_err:
                logger.error(f"檢查重平衡訂單時出錯: {db_err}")

        order_data = {
            'order_id': order_id,
            'symbol': self.symbol,
            'side': normalized_side,
            'quantity': quantity,
            'price': price,
            'maker': maker,
            'fee': fee,
            'fee_asset': fee_asset,
            'trade_type': trade_type,
        }

        if self._db_available():
            def safe_insert_order():
                try:
                    self.db.insert_order(order_data)
                except Exception as db_err:
                    logger.error(f"插入訂單數據時出錯: {db_err}")

            safe_insert_order()

        trade_quote_volume = abs(quantity * price)
        self.total_quote_volume += trade_quote_volume
        self.session_quote_volume += trade_quote_volume

        if normalized_side == 'Bid':
            self.total_bought += quantity
            self.buy_trades.append((price, quantity))

            if maker:
                self.maker_buy_volume += quantity
                self.session_maker_buy_volume += quantity
            else:
                self.taker_buy_volume += quantity
                self.session_taker_buy_volume += quantity

            self.session_buy_trades.append((price, quantity))

        elif normalized_side == 'Ask':
            self.total_sold += quantity
            self.sell_trades.append((price, quantity))

            if maker:
                self.maker_sell_volume += quantity
                self.session_maker_sell_volume += quantity
            else:
                self.taker_sell_volume += quantity
                self.session_taker_sell_volume += quantity

            self.session_sell_trades.append((price, quantity))

        self.total_fees += fee
        self.session_fees += fee

        if self._db_available():
            def safe_update_stats_wrapper():
                try:
                    self._update_trading_stats()
                except Exception as e:
                    logger.error(f"更新交易統計時出錯: {e}")

            self.executor.submit(safe_update_stats_wrapper)

        if self._db_available():
            def update_profit():
                try:
                    profit = self._calculate_db_profit()
                    self.total_profit = profit
                except Exception as e:
                    logger.error(f"更新利潤計算時出錯: {e}")

            self.executor.submit(update_profit)

        session_profit = self._calculate_session_profit()

        logger.info(f"累計利潤: {self.total_profit:.8f} {self.quote_asset}")
        logger.info(f"本次執行利潤: {session_profit:.8f} {self.quote_asset}")
        logger.info(f"本次執行手續費: {self.session_fees:.8f} {self.quote_asset}")
        logger.info(f"本次執行淨利潤: {(session_profit - self.session_fees):.8f} {self.quote_asset}")

        self.trades_executed += 1
        logger.info(f"總買入: {self.total_bought} {self.base_asset}, 總賣出: {self.total_sold} {self.base_asset}")
        logger.info(f"Maker買入: {self.maker_buy_volume} {self.base_asset}, Maker賣出: {self.maker_sell_volume} {self.base_asset}")
        logger.info(f"Taker買入: {self.taker_buy_volume} {self.base_asset}, Taker賣出: {self.taker_sell_volume} {self.base_asset}")

        fill_info = {
            'side': normalized_side,
            'quantity': quantity,
            'price': price,
            'order_id': order_id,
            'client_id': client_id,  # APEX 使用 clientId 追蹤訂單
            'maker': maker,
            'fee': fee,
            'fee_asset': fee_asset,
            'trade_id': trade_id,
            'source': source,
            'timestamp': timestamp,
        }

        try:
            self._after_fill_processed(fill_info)
        except Exception as hook_error:
            logger.error(f"成交後置處理時出錯: {hook_error}")

    def _after_fill_processed(self, fill_info: Dict[str, Any]) -> None:
        """留給子類覆蓋的成交後置處理鈎子"""
        return

    def check_ws_connection(self):
        """檢查並恢復WebSocket連接"""
        if not self.ws:
            # aster, paradex, lighter, apex 沒有 WebSocket，直接返回 True
            if self.exchange in ('aster', 'paradex', 'lighter', 'apex'):
                return True
            logger.warning("WebSocket對象不存在，嘗試重新創建...")
            return self._recreate_websocket()

        ws_connected = self.ws.is_connected()

        if not ws_connected and not getattr(self.ws, 'reconnecting', False):
            # 檢查上次重連嘗試的時間，避免頻繁重連
            current_time = time.time()
            last_reconnect_attempt = getattr(self, '_last_reconnect_attempt', 0)
            reconnect_cooldown = 30  # 30秒冷卻時間

            if current_time - last_reconnect_attempt >= reconnect_cooldown:
                logger.warning("WebSocket連接已斷開，觸發重連...")
                self._last_reconnect_attempt = current_time
                # 使用 WebSocket 自己的重連機制
                self.ws.check_and_reconnect_if_needed()
            else:
                remaining = int(reconnect_cooldown - (current_time - last_reconnect_attempt))
                logger.debug(f"WebSocket 重連冷卻中，剩餘 {remaining} 秒")

        return self.ws.is_connected() if self.ws else False
    
    def _recreate_websocket(self):
        """重新創建WebSocket連接"""
        try:
            if self.exchange == 'aster':
                logger.info(f"{self.exchange} 交易所不使用 WebSocket")
                return True
            
            # 安全關閉現有連接
            if self.ws:
                try:
                    self.ws.running = False
                    self.ws.close()
                    time.sleep(0.5)
                except Exception as e:
                    logger.debug(f"關閉現有WebSocket時的預期錯誤: {e}")
            if self.exchange == 'backpack':
                # 創建新的連接
                self.ws = BackpackWebSocket(
                    self.api_key,
                    self.secret_key,
                    self.symbol,
                    self.on_ws_message,
                    auto_reconnect=True
                )
            elif self.exchange == 'xx':
                ...
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
                    quantity = float(data.get('l', '0'))
                    price = float(data.get('L', '0'))
                    order_id = data.get('i')
                    maker = data.get('m', False)
                    
                    # 解析手續費信息（處理各種可能的字段名）
                    fee = 0.0
                    fee_asset = self.quote_asset
                    fee_fields = [
                        ('n', 'N'),  # n: fee amount, N: fee asset
                        ('fee', 'fee_currency'),
                        ('commission', 'commissionAsset')
                    ]
                    for amount_field, asset_field in fee_fields:
                        fee_amount = data.get(amount_field)
                        if fee_amount is not None:
                            try:
                                fee = float(fee_amount)
                                fee_asset = data.get(asset_field, self.quote_asset)
                                break
                            except (TypeError, ValueError):
                                continue
                    
                    trade_id = data.get('t')
                    timestamp = data.get('T') or data.get('E')

                    trade_id_str = str(trade_id) if trade_id is not None else None
                    timestamp_int = None
                    if timestamp is not None:
                        try:
                            timestamp_int = int(timestamp)
                        except (TypeError, ValueError):
                            timestamp_int = None

                    logger.info(
                        f"WebSocket 成交通知: {'買' if side == 'BUY' else '賣'}單成交 "
                        f"{quantity} @ {price}, "
                        f"{'Maker' if maker else 'Taker'}, "
                        f"手續費: {fee} {fee_asset}"
                    )

                    self._process_order_fill_event(
                        side=side,
                        quantity=quantity,
                        price=price,
                        order_id=order_id,
                        maker=bool(maker),
                        fee=fee,
                        fee_asset=fee_asset,
                        trade_id=trade_id_str,
                        source=data.get('source', 'ws'),
                        timestamp=timestamp_int,
                    )
                    
                except Exception as e:
                    logger.error(f"處理訂單成交消息時出錯: {e}")
                    traceback.print_exc()
    
    def on_order_update(self, order_data):
        """處理所有交易所的訂單更新消息 - 統一接口"""
        try:
            order_id = order_data.get('order_id')
            side = order_data.get('side', '').lower()
            status = order_data.get('status')
            filled_size = float(order_data.get('filled_size', '0'))
            price = float(order_data.get('price', '0'))
            
            # 簡化日誌輸出 - 只記錄重要的狀態變化
            if status in ('FILLED', 'PARTIALLY_FILLED', 'filled', 'partial_filled'):
                if filled_size > 0:
                    direction = "買入" if side == 'buy' else "賣出"
                    logger.info(f"*** 成交通知: {direction} {filled_size:.3f} SOL @ {price:.3f} USDT ({status}) ***")
                
            # 通用處理邏輯 - 處理成交的訂單
            if status in ('FILLED', 'PARTIALLY_FILLED', 'filled', 'partial_filled') and filled_size > 0:
                # 模擬訂單成交數據格式
                is_maker = True  # 限價單通常是 maker
                
                # 準備訂單數據用於數據庫記錄
                order_data_db = {
                    'order_id': order_id,
                    'symbol': self.symbol,
                    'side': 'Bid' if side == 'buy' else 'Ask',  # 轉換為數據庫格式
                    'quantity': filled_size,
                    'price': price,
                    'maker': is_maker,
                    'fee': 0.0,  # 手續費可能需要單獨查詢
                    'fee_asset': self.quote_asset,
                    'trade_type': 'market_making'
                }
                
                # 更新統計
                quote_volume = abs(filled_size * price)
                self.total_quote_volume += quote_volume
                self.session_quote_volume += quote_volume

                if side == 'buy':
                    self.total_bought += filled_size
                    if is_maker:
                        self.maker_buy_volume += filled_size
                        self.session_maker_buy_volume += filled_size
                    else:
                        self.taker_buy_volume += filled_size
                        self.session_taker_buy_volume += filled_size
                    self.buy_trades.append((price, filled_size))
                    self.session_buy_trades.append((price, filled_size))
                elif side == 'sell':
                    self.total_sold += filled_size
                    if is_maker:
                        self.maker_sell_volume += filled_size
                        self.session_maker_sell_volume += filled_size
                    else:
                        self.taker_sell_volume += filled_size
                        self.session_taker_sell_volume += filled_size
                    self.sell_trades.append((price, filled_size))
                    self.session_sell_trades.append((price, filled_size))
                
                # 異步插入數據庫
                if self._db_available():
                    def safe_insert_order():
                        try:
                            self.db.insert_order(order_data_db)
                        except Exception as db_err:
                            logger.error(f"插入訂單數據時出錯: {db_err}")

                    self.executor.submit(safe_insert_order)

                    # 更新利潤計算
                    def update_profit():
                        try:
                            profit = self._calculate_db_profit()
                            self.total_profit = profit
                        except Exception as e:
                            logger.error(f"更新利潤計算時出錯: {e}")

                    self.executor.submit(update_profit)
                
                # 執行統計報告
                session_profit = self._calculate_session_profit()
                
                logger.info(f"累計利潤: {self.total_profit:.8f} {self.quote_asset}")
                logger.info(f"本次執行利潤: {session_profit:.8f} {self.quote_asset}")
                logger.info(f"總買入: {self.total_bought} {self.base_asset}, 總賣出: {self.total_sold} {self.base_asset}")
                
                self.trades_executed += 1
                
        except Exception as e:
            logger.error(f"處理訂單更新時出錯: {e}")
            traceback.print_exc()
    
    def _calculate_memory_profit(self) -> float:
        """使用記憶體中的成交記錄計算已實現利潤（FIFO）。"""
        if not self.buy_trades or not self.sell_trades:
            return 0.0

        buy_queue: List[Tuple[float, float]] = [
            (float(price), float(quantity)) for price, quantity in self.buy_trades
        ]
        total_profit = 0.0

        for sell_price, sell_quantity in self.sell_trades:
            remaining_sell = float(sell_quantity)
            sell_price = float(sell_price)

            while remaining_sell > 0 and buy_queue:
                buy_price, buy_quantity = buy_queue[0]
                matched_quantity = min(remaining_sell, buy_quantity)

                total_profit += (sell_price - buy_price) * matched_quantity

                remaining_sell -= matched_quantity
                if matched_quantity >= buy_quantity:
                    buy_queue.pop(0)
                else:
                    buy_queue[0] = (buy_price, buy_quantity - matched_quantity)

        return total_profit

    def _calculate_db_profit(self):
        """基於數據庫記錄計算已實現利潤（FIFO方法）"""
        if not self._db_available():
            return self._calculate_memory_profit()
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
        if not self._db_available():
            return
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
        # 只檢查連接狀態，不觸發重連（避免頻繁重連嘗試）
        price = None
        if self.ws and self.ws.is_connected():
            price = self.ws.get_current_price()
        
        if price is None:
            ticker_response = self.client.get_ticker(self.symbol)
            if not ticker_response.success:
                logger.error(f"獲取價格失敗: {ticker_response.error_message}")
                return None
            
            ticker = ticker_response.data
            if ticker.last_price is None:
                logger.error(f"獲取到的價格數據不完整: {ticker_response.raw}")
                return None
            return float(ticker.last_price)
        return price
    
    def get_market_depth(self):
        """獲取市場深度（優先使用WebSocket數據）"""
        # 只檢查連接狀態，不觸發重連（避免頻繁重連嘗試）
        bid_price, ask_price = None, None
        if self.ws and self.ws.is_connected():
            bid_price, ask_price = self.ws.get_bid_ask()
        
        if bid_price is None or ask_price is None:
            orderbook_response = self.client.get_order_book(self.symbol)
            if not orderbook_response.success:
                logger.error(f"獲取訂單簿失敗: {orderbook_response.error_message}")
                return None, None
            
            orderbook = orderbook_response.data
            if not orderbook.bids or not orderbook.asks:
                return None, None
            
            highest_bid = float(orderbook.bids[0].price)
            lowest_ask = float(orderbook.asks[0].price)
            
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
        
        # 使用用户設定的目標比例
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
                "quantity": format_quantity(quantity_to_sell, self.base_precision),
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
                "quantity": format_quantity(quantity_to_buy, self.base_precision),
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
        order_response = self.client.execute_order(order_details)
        
        if not order_response.success:
            logger.error(f"重新平衡訂單執行失敗: {order_response.error_message}")
        else:
            logger.info(f"重新平衡訂單執行成功")
            # 記錄這是一個重平衡訂單
            order_result = order_response.data
            if order_result and order_result.order_id and self._db_available():
                self.db.record_rebalance_order(order_result.order_id, self.symbol)
        
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
            
            # 如果可用餘額很少但總餘額充足，説明資金在抵押品中
            if base_available < base_total * 0.1:
                logger.info(f"基礎資產主要在抵押品中，將依靠自動贖回功能")
            if quote_available < quote_total * 0.1:
                logger.info(f"報價資產主要在抵押品中，將依靠自動贖回功能")
            
            # 計算每個訂單的數量
            avg_price = sum(buy_prices) / len(buy_prices) if buy_prices else 0
            
            if avg_price <= 0 or (quote_total <= 0 and base_total <= 0):
                # 無法計算，使用最小訂單量
                logger.warning("無法根據餘額計算訂單數量，使用最小訂單量")
                buy_quantity = self.min_order_size
                sell_quantity = self.min_order_size
            else:
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

        def place_buy(price, qty):
            # 使用 format_quantity 避免科學計數法
            qty_str = format_quantity(qty, self.base_precision)
            order = {
                "orderType": "Limit",
                "price": str(price),
                "quantity": qty_str,
                "side": "Bid",
                "symbol": self.symbol,
                "timeInForce": "GTC",
                "postOnly": True,
                "autoLendRedeem": True,
                "autoLend": True
            }
            res = self.client.execute_order(order)
            if not res.success and "POST_ONLY_TAKER" in str(res.error_message or ""):
                logger.info("調整買單價格並重試...")
                order["price"] = str(round_to_tick_size(float(order["price"]) - self.tick_size, self.tick_size))
                res = self.client.execute_order(order)
            
            # 特殊處理資金不足錯誤
            if not res.success and "INSUFFICIENT_FUNDS" in str(res.error_message or ""):
                logger.warning(f"買單資金不足，可能需要手動贖回抵押品或等待自動贖回生效")
            
            return qty_str, order["price"], res

        with ThreadPoolExecutor(max_workers=self.max_orders) as executor:
            for p in buy_prices:
                if len(buy_futures) >= self.max_orders:
                    break
                buy_futures.append(executor.submit(place_buy, p, buy_quantity))

        buy_order_count = 0
        for future in buy_futures:
            qty, p_used, res = future.result()
            if not res.success:
                logger.error(f"買單失敗: {res.error_message}")
            else:
                logger.info(f"買單成功: 價格 {p_used}, 數量 {qty}")
                self.active_buy_orders.append(res.raw)
                self.orders_placed += 1
                buy_order_count += 1

        # 下賣單
        sell_futures = []

        def place_sell(price, qty):
            # 使用 format_quantity 避免科學計數法
            qty_str = format_quantity(qty, self.base_precision)
            order = {
                "orderType": "Limit",
                "price": str(price),
                "quantity": qty_str,
                "side": "Ask",
                "symbol": self.symbol,
                "timeInForce": "GTC",
                "postOnly": True,
                "autoLendRedeem": True,
                "autoLend": True
            }
            res = self.client.execute_order(order)
            if not res.success and "POST_ONLY_TAKER" in str(res.error_message or ""):
                logger.info("調整賣單價格並重試...")
                order["price"] = str(round_to_tick_size(float(order["price"]) + self.tick_size, self.tick_size))
                res = self.client.execute_order(order)
            
            # 特殊處理資金不足錯誤
            if not res.success and "INSUFFICIENT_FUNDS" in str(res.error_message or ""):
                logger.warning(f"賣單資金不足，可能需要手動贖回抵押品或等待自動贖回生效")
            
            return qty_str, order["price"], res

        with ThreadPoolExecutor(max_workers=self.max_orders) as executor:
            for p in sell_prices:
                if len(sell_futures) >= self.max_orders:
                    break
                sell_futures.append(executor.submit(place_sell, p, sell_quantity))

        sell_order_count = 0
        for future in sell_futures:
            qty, p_used, res = future.result()
            if not res.success:
                logger.error(f"賣單失敗: {res.error_message}")
            else:
                logger.info(f"賣單成功: 價格 {p_used}, 數量 {qty}")
                self.active_sell_orders.append(res.raw)
                self.orders_placed += 1
                sell_order_count += 1
            
        logger.info(f"共下單: {buy_order_count} 個買單, {sell_order_count} 個賣單")
    
    def cancel_existing_orders(self):
        """取消所有現有訂單"""
        orders_response = self.client.get_open_orders(self.symbol)
        
        if not orders_response.success:
            logger.error(f"獲取訂單失敗: {orders_response.error_message}")
            return
        
        open_orders = orders_response.data
        if not open_orders:
            logger.info("沒有需要取消的現有訂單")
            self.active_buy_orders = []
            self.active_sell_orders = []
            return
        
        logger.info(f"正在取消 {len(open_orders)} 個現有訂單")
        
        try:
            # 嘗試批量取消
            cancel_response = self.client.cancel_all_orders(self.symbol)
            
            if not cancel_response.success:
                logger.error(f"批量取消訂單失敗: {cancel_response.error_message}")
                logger.info("嘗試逐個取消...")
                
                # 初始化線程池
                with ThreadPoolExecutor(max_workers=5) as executor:
                    cancel_futures = []
                    
                    # 提交取消訂單任務
                    for order_info in open_orders:
                        order_id = order_info.order_id if hasattr(order_info, 'order_id') else order_info.get('id')
                        if not order_id:
                            continue
                        
                        # Use legacy wrapper to keep existing logic; could be refactored to self.client.cancel_order
                        # Directly use instance client method now
                        future = executor.submit(
                            self.client.cancel_order,
                            order_id,
                            self.symbol
                        )
                        cancel_futures.append((order_id, future))
                    
                    # 處理結果
                    for order_id, future in cancel_futures:
                        try:
                            res = future.result()
                            if not res.success:
                                logger.error(f"取消訂單 {order_id} 失敗: {res.error_message}")
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
        remaining_response = self.client.get_open_orders(self.symbol)
        remaining_orders = remaining_response.data if remaining_response.success else []
        if remaining_orders and len(remaining_orders) > 0:
            logger.warning(f"警告: 仍有 {len(remaining_orders)} 個未取消的訂單")
        else:
            logger.info("所有訂單已成功取消")
        
        # 重置活躍訂單列表
        self.active_buy_orders = []
        self.active_sell_orders = []
    
    def check_order_fills(self):
        orders_response = self.client.get_open_orders(self.symbol)
        if not orders_response.success:
            logger.error(f"獲取訂單失敗: {orders_response.error_message}")
            return []
        open_orders = orders_response.data
        current_order_ids = set()
        if open_orders:
            for order_info in open_orders:
                order_id = order_info.order_id if hasattr(order_info, 'order_id') else order_info.get('id')
                if order_id:
                    current_order_ids.add(order_id)
        prev_buy_orders = len(self.active_buy_orders)
        prev_sell_orders = len(self.active_sell_orders)
        filled_order_ids = []
        for order in self.active_buy_orders + self.active_sell_orders:
            order_id = order.get('id') if isinstance(order, dict) else getattr(order, 'order_id', None)
            if order_id and order_id not in current_order_ids:
                filled_order_ids.append(order_id)
        filled_trades = []
        if filled_order_ids:
            try:
                fills_response = self.client.get_fill_history(self.symbol, limit=50)
                if fills_response.success:
                    # 使用統一的格式處理方法
                    recent_fills = self._normalize_fill_history_response(fills_response)
                    if not hasattr(self, '_processed_fill_ids'):
                        self._processed_fill_ids = set()
                    for fill in recent_fills:
                        fill_id = fill.get('fill_id') or fill.get('id')
                        fill_order_id = fill.get('order_id')
                        if fill_id in self._processed_fill_ids:
                            continue
                        if fill_order_id in filled_order_ids:
                            filled_trades.append(fill)
                            self._processed_fill_ids.add(fill_id)
                            side = fill.get('side', '').upper()
                            price = float(fill.get('price', 0) or 0)
                            size = float(fill.get('quantity', 0) or fill.get('size', 0) or 0)
                            is_maker = fill.get('is_maker', True)
                            liquidity = 'MAKER' if is_maker else 'TAKER'
                            realized_pnl = fill.get('realized_pnl', 0)
                            
                            # 獲取手續費信息
                            fee = float(fill.get('fee', 0) or 0)
                            fee_currency = fill.get('fee_asset', self.quote_asset) or self.quote_asset
                            
                            # 構建完整的成交資訊
                            fill_info = {
                                'side': 'Bid' if side == 'BUY' else 'Ask',
                                'quantity': size,
                                'price': price,
                                'maker': is_maker,
                                'order_id': fill.get('order_id'),
                                'client_id': fill.get('client_id'),  # APEX 使用 clientId
                                'trade_id': fill.get('id'),
                                'realized_pnl': realized_pnl,
                                'fee': fee,
                                'fee_currency': fee_currency
                            }
                            
                            logger.info(
                                f"✓ {'買' if side == 'BUY' else '賣'}單成交 ({liquidity}): "
                                f"{size} @ {price}, 已實現盈虧: {realized_pnl}, 手續費: {fee} {fee_currency}"
                            )
                            
                            # 觸發成交後處理
                            self._process_order_fill_event(
                                side=fill_info['side'],
                                quantity=fill_info['quantity'],
                                price=fill_info['price'],
                                order_id=fill_info['order_id'],
                                maker=fill_info['maker'],
                                fee=fee,
                                fee_asset=fee_currency,
                                trade_id=fill_info['trade_id'],
                                source='rest',
                                timestamp=int(time.time() * 1000),
                                client_id=fill_info.get('client_id'),  # APEX 使用 clientId
                            )
            except Exception as e:
                logger.error(f"獲取成交記錄失敗: {e}")
        active_buy_orders = []
        active_sell_orders = []
        if open_orders:
            for order in open_orders:
                # 支援 OrderInfo dataclass 或 dict
                if hasattr(order, 'side'):
                    side = str(order.side).upper()
                else:
                    side = str(order.get('side', '')).upper()
                if side in ('BID', 'BUY'):
                    active_buy_orders.append(order)
                elif side in ('ASK', 'SELL'):
                    active_sell_orders.append(order)
        self.active_buy_orders = active_buy_orders
        self.active_sell_orders = active_sell_orders
        if prev_buy_orders != len(active_buy_orders) or prev_sell_orders != len(active_sell_orders):
            logger.info(f"訂單數量變更: 買單 {prev_buy_orders} -> {len(active_buy_orders)}, 賣單 {prev_sell_orders} -> {len(active_sell_orders)}")
        logger.info(f"當前活躍訂單: 買單 {len(self.active_buy_orders)} 個, 賣單 {len(self.active_sell_orders)} 個")
        return filled_trades
    def estimate_profit(self, pnl_data=None):
        """輸出本次迭代的關鍵統計資訊。"""
        if pnl_data is None:
            pnl_data = self.calculate_pnl()

        (
            realized_pnl,
            unrealized_pnl,
            total_fees,
            net_pnl,
            session_realized_pnl,
            session_fees,
            session_net_pnl,
        ) = pnl_data

        session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
        session_sell_volume = sum(qty for _, qty in self.session_sell_trades)

        sections: List[Tuple[str, List[Union[str, Tuple[str, str]]]]] = []

        if session_buy_volume > 0 or session_sell_volume > 0:
            session_rows: List[Union[str, Tuple[str, str]]] = [
                ("成交量", f"買入 {session_buy_volume:.3f} {self.base_asset} | 賣出 {session_sell_volume:.3f} {self.base_asset}"),
                (
                    "盈虧",
                    f"已實現 {session_realized_pnl:.4f} {self.quote_asset} | 凈利潤 {session_net_pnl:.4f} {self.quote_asset} (手續費 {session_fees:.4f} {self.quote_asset})",
                ),
                ("Maker成交量", f"買 {self.session_maker_buy_volume:.3f} {self.base_asset} | 賣 {self.session_maker_sell_volume:.3f} {self.base_asset}"),
                ("Taker成交量", f"買 {self.session_taker_buy_volume:.3f} {self.base_asset} | 賣 {self.session_taker_sell_volume:.3f} {self.base_asset}"),
            ]
            session_rows.insert(1, ("成交額", f"{self.session_quote_volume:.2f} {self.quote_asset}"))
        else:
            session_rows = [
                "本次迭代沒有成交記錄",
                (
                    "盈虧",
                    f"已實現 {session_realized_pnl:.4f} {self.quote_asset} | 凈利潤 {session_net_pnl:.4f} {self.quote_asset} (手續費 {session_fees:.4f} {self.quote_asset})",
                ),
                ("Maker成交量", f"買 {self.session_maker_buy_volume:.3f} {self.base_asset} | 賣 {self.session_maker_sell_volume:.3f} {self.base_asset}"),
                ("Taker成交量", f"買 {self.session_taker_buy_volume:.3f} {self.base_asset} | 賣 {self.session_taker_sell_volume:.3f} {self.base_asset}"),
            ]

        sections.append(("本次執行", session_rows))

        sections.append(
            (
                "累計表現",
                [
                    ("累計盈虧", f"{net_pnl:.4f} {self.quote_asset}"),
                    ("未實現盈虧", f"{unrealized_pnl:.4f} {self.quote_asset}"),
                    ("累計手續費", f"{total_fees:.4f} {self.quote_asset}"),
                ],
            )
        )

        sections.append(
            (
                "交易計數",
                [
                    ("成交次數", f"{self.trades_executed} 次"),
                    ("下單次數", f"{self.orders_placed} 次"),
                    ("取消次數", f"{self.orders_cancelled} 次"),
                ],
            )
        )

        if self.total_quote_volume > 0:
            loss = min(net_pnl, 0)  # 僅取虧損
            wear_rate_value = abs(loss) / self.total_quote_volume * 100
            wear_rate_display = f"{wear_rate_value:.4f}%"
        else:
            wear_rate_display = "N/A"


        trade_rows = [
            ("總成交量", f"買 {self.total_bought:.3f} {self.base_asset} | 賣 {self.total_sold:.3f} {self.base_asset}"),
            ("總成交額", f"{self.total_quote_volume:.2f} {self.quote_asset}"),
            ("Maker總量", f"買 {self.maker_buy_volume:.3f} {self.base_asset} | 賣 {self.maker_sell_volume:.3f} {self.base_asset}"),
            ("Taker總量", f"買 {self.taker_buy_volume:.3f} {self.base_asset} | 賣 {self.taker_sell_volume:.3f} {self.base_asset}"),
            ("磨損率", wear_rate_display),
        ]

        sections.append(
            (
                "成交概況",
                trade_rows,
            )
        )

        if self.active_buy_orders and self.active_sell_orders:
            # 支援 dict 或 OrderInfo/OrderResult dataclass
            last_buy = self.active_buy_orders[-1]
            first_sell = self.active_sell_orders[0]
            
            if hasattr(last_buy, 'price'):
                buy_price = float(last_buy.price or 0)
            elif isinstance(last_buy, dict):
                buy_price = float(last_buy.get('price', 0) or 0)
            else:
                buy_price = 0
            
            if hasattr(first_sell, 'price'):
                sell_price = float(first_sell.price or 0)
            elif isinstance(first_sell, dict):
                sell_price = float(first_sell.get('price', 0) or 0)
            else:
                sell_price = 0
            
            spread = sell_price - buy_price
            spread_pct = (spread / buy_price * 100) if buy_price > 0 else 0
            order_line = f"買 {buy_price:.3f} | 賣 {sell_price:.3f} | 價差 {spread:.3f} ({spread_pct:.3f}%)"
        else:
            active_buy_count = len(self.active_buy_orders)
            active_sell_count = len(self.active_sell_orders)
            order_line = f"買單 {active_buy_count} | 賣單 {active_sell_count}"

        sections.append(
            (
                "市場狀態",
                [
                    ("活躍訂單", order_line),
                    ("WebSocket狀態", "已連接" if self.ws and self.ws.is_connected() else "未連接"),
                ],
            )
        )

        extra_sections = self._get_extra_summary_sections()
        if extra_sections:
            sections.extend(extra_sections)

        self._log_boxed_summary("做市統計總結", sections)

    def _get_extra_summary_sections(self) -> List[Tuple[str, List[Union[str, Tuple[str, str]]]]]:
        """提供子類擴展的統計輸出。"""
        return []

    def _log_boxed_summary(self, title: str, sections: List[Tuple[str, List[Union[str, Tuple[str, str]]]]]):
        """以框線格式輸出統計資訊。"""
        inner_width = 74
        border_top = f"┌{'─' * inner_width}┐"
        border_section = f"├{'─' * inner_width}┤"
        border_bottom = f"└{'─' * inner_width}┘"

        logger.info(border_top)
        self._log_box_text(title, inner_width, align="center")

        for index, (section_title, rows) in enumerate(sections):
            logger.info(border_section)
            self._log_box_text(f"▸ {section_title}", inner_width)

            for row in rows:
                if isinstance(row, tuple):
                    label, value = row
                    self._log_box_key_value(label, value, inner_width)
                else:
                    self._log_box_text(str(row), inner_width)

        logger.info(border_bottom)

    def _log_box_text(self, text: str, inner_width: int, align: str = "left"):
        """在框線內輸出單行或多行文字。"""
        if align == "center":
            logger.info(f"│ {self._center_display(text, inner_width)} │")
            return

        for line in self._wrap_display_text(text, inner_width):
            logger.info(f"│ {self._pad_display(line, inner_width)} │")

    def _log_box_key_value(self, label: str, value: str, inner_width: int):
        """以鍵值形式輸出內容，並處理換行。"""
        label_display = f"{label}："
        label_width = 18
        label_field = self._pad_display(label_display, label_width)
        value_width = max(10, inner_width - label_width - 1)
        empty_label = self._pad_display("", label_width)

        wrapped_values = self._wrap_display_text(value, value_width)
        for index, chunk in enumerate(wrapped_values):
            chunk_field = self._pad_display(chunk, value_width)
            if index == 0:
                line = f"{label_field} {chunk_field}"
            else:
                line = f"{empty_label} {chunk_field}"
            logger.info(f"│ {line} │")

    def _display_width(self, text: str) -> int:
        """計算字串的可視寬度，處理全形與半形字符。"""
        width = 0
        for char in text:
            east_asian_width = unicodedata.east_asian_width(char)
            if east_asian_width in ("F", "W", "A"):
                width += 2
            else:
                width += 1
        return width

    def _pad_display(self, text: str, width: int) -> str:
        """將字串填充至指定的顯示寬度。"""
        padding = max(0, width - self._display_width(text))
        return f"{text}{' ' * padding}"

    def _center_display(self, text: str, width: int) -> str:
        """以顯示寬度為基準進行置中。"""
        text_width = self._display_width(text)
        if text_width >= width:
            return text
        total_padding = width - text_width
        left = total_padding // 2
        right = total_padding - left
        return f"{' ' * left}{text}{' ' * right}"

    def _wrap_display_text(self, text: str, width: int) -> List[str]:
        """根據顯示寬度換行。"""
        if not text:
            return [""]

        lines: List[str] = []
        current = ""
        current_width = 0

        for char in text:
            char_width = self._display_width(char)
            if current and current_width + char_width > width:
                lines.append(current)
                current = char
                current_width = char_width
            else:
                current += char
                current_width += char_width

        if current:
            lines.append(current)
        else:
            lines.append("")

        return lines
    
    def print_trading_stats(self):
        """打印交易統計報表"""
        try:
            logger.info("\n=== 做市商交易統計 ===")
            logger.info(f"交易對: {self.symbol}")

            today = datetime.now().strftime('%Y-%m-%d')
            if self._db_available():
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
            else:
                logger.info("資料庫功能未啟用，僅顯示本次執行的統計資訊。")
            
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
            if self._db_available():
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
        # 如果使用 Websea，不需要 WebSocket 數據流
        if self.ws is None:
            return
        
        # 構建完整的頻道名稱（包含 symbol）
        depth_channel = f"depth.{self.symbol}"
        ticker_channel = f"bookTicker.{self.symbol}"
        order_update_channel = f"account.orderUpdate.{self.symbol}"
            
        # 檢查深度流訂閲
        if depth_channel not in self.ws.subscriptions:
            logger.info("重新訂閲深度數據流...")
            self.ws.initialize_orderbook()  # 重新初始化訂單簿
            self.ws.subscribe_depth()
        
        # 檢查行情數據訂閲
        if ticker_channel not in self.ws.subscriptions:
            logger.info("重新訂閲行情數據...")
            self.ws.subscribe_bookTicker()
        
        # 檢查私有訂單更新流
        if order_update_channel not in self.ws.subscriptions:
            logger.info("重新訂閲私有訂單更新流...")
            self.subscribe_order_updates()

    def check_stop_conditions(self, realized_pnl, unrealized_pnl, session_realized_pnl) -> bool:
        """檢查是否觸發提前停止條件。

        基類默認不啟用任何風控條件，返回 ``False``。

        Args:
            realized_pnl (float): 累計已實現盈虧。
            unrealized_pnl (float): 未實現盈虧。
            session_realized_pnl (float): 本次執行的已實現盈虧。

        Returns:
            bool: 是否應該提前停止策略。
        """

        return False

    def stop(self):
        """停止做市策略"""
        logger.info("收到停止信號，正在停止做市策略...")
        self._stop_flag = True

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
            if connection_status and self.ws is not None:
                # 構建完整的頻道名稱
                depth_channel = f"depth.{self.symbol}"
                ticker_channel = f"bookTicker.{self.symbol}"
                order_update_channel = f"account.orderUpdate.{self.symbol}"
                
                # 初始化訂單簿和數據流
                if not self.ws.orderbook["bids"] and not self.ws.orderbook["asks"]:
                    self.ws.initialize_orderbook()
                
                # 檢查並確保所有數據流訂閲
                if depth_channel not in self.ws.subscriptions:
                    self.ws.subscribe_depth()
                if ticker_channel not in self.ws.subscriptions:
                    self.ws.subscribe_bookTicker()
                if order_update_channel not in self.ws.subscriptions:
                    self.subscribe_order_updates()
            
            while time.time() - start_time < duration_seconds and not self._stop_flag:
                iteration += 1
                current_time = time.time()
                logger.info(f"\n=== 第 {iteration} 次迭代 ===")
                logger.info(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 檢查連接並在必要時重連
                connection_status = self.check_ws_connection()
                
                # 如果連接成功，檢查並確保所有流訂閲
                if connection_status:
                    # 重新訂閲必要的數據流
                    self._ensure_data_streams()
                
                # 檢查訂單成交情況
                self.check_order_fills()

                # 透過 REST API 同步最新成交
                if self.exchange in ('aster', 'lighter', 'apex'):
                    self._sync_fill_history()

                # 檢查是否需要重平衡倉位
                if self.need_rebalance():
                    self.rebalance_position()
                
                # 下限價單
                self.place_limit_orders()

                # 計算PnL並輸出簡化統計
                pnl_data = self.calculate_pnl()
                self.estimate_profit(pnl_data)

                # 定期打印交易統計報表
                if current_time - last_report_time >= report_interval:
                    self.print_trading_stats()
                    last_report_time = current_time

                (
                    realized_pnl,
                    unrealized_pnl,
                    _total_fees,
                    _net_pnl,
                    session_realized_pnl,
                    _session_fees,
                    _session_net_pnl,
                ) = pnl_data

                if self.check_stop_conditions(realized_pnl, unrealized_pnl, session_realized_pnl):
                    self._stop_trading = True
                    logger.warning("觸發風控條件，提前結束策略迭代")
                    break

                wait_time = interval_seconds
                logger.info(f"等待 {wait_time} 秒後進行下一次迭代...")
                time.sleep(wait_time)

            # 結束運行時打印最終報表
            logger.info("\n=== 做市策略運行結束 ===")
            if self._stop_trading and self.stop_reason:
                logger.warning(f"提前停止原因: {self.stop_reason}")
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
