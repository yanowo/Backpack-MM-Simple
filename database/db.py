"""
數據庫操作模塊
"""
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from config import DB_PATH
from logger import setup_logger

logger = setup_logger("database")

class Database:
    def __init__(self, db_path=DB_PATH):
        """
        初始化數據庫連接
        
        Args:
            db_path: 數據庫文件路徑
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._init_tables()
    
    def _connect(self):
        """建立數據庫連接"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # 主游標只用於初始化
            self.cursor = self.conn.cursor()
            logger.info(f"數據庫連接成功: {self.db_path}")
        except Exception as e:
            logger.error(f"數據庫連接失敗: {e}")
            raise
    
    def _init_tables(self):
        """初始化資料庫表結構"""
        try:
            # 改進的交易記錄表
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS completed_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    maker BOOLEAN,
                    fee REAL,
                    fee_asset TEXT,
                    trade_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            # 創建索引提高查詢效率
            self.cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_completed_orders_symbol 
                ON completed_orders(symbol)
                """
            )
            
            # 統計表來跟蹤每日/每週成交量和利潤
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    symbol TEXT,
                    maker_buy_volume REAL DEFAULT 0,
                    maker_sell_volume REAL DEFAULT 0,
                    taker_buy_volume REAL DEFAULT 0,
                    taker_sell_volume REAL DEFAULT 0,
                    realized_profit REAL DEFAULT 0,
                    total_fees REAL DEFAULT 0,
                    net_profit REAL DEFAULT 0,
                    avg_spread REAL DEFAULT 0,
                    trade_count INTEGER DEFAULT 0,
                    volatility REAL DEFAULT 0
                )
                """
            )
            
            # 重平衡訂單記錄表
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rebalance_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    symbol TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            # 市場數據表
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    price REAL,
                    volume REAL,
                    bid_ask_spread REAL,
                    liquidity_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            try:
                self.conn.commit()
                logger.info("數據庫表初始化成功")
            except sqlite3.OperationalError as e:
                logger.debug(f"提交表初始化時出錯: {e}")
        except Exception as e:
            logger.error(f"初始化資料庫表時出錯: {e}")
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
            raise

    def execute(self, query, params=None):
        """
        執行SQL查詢 - 創建新游標避免遞歸問題
        
        Args:
            query: SQL查詢字符串
            params: 查詢參數
            
        Returns:
            游標對象
        """
        try:
            # 每次操作創建新游標
            cursor = self.conn.cursor()
            if params:
                return cursor.execute(query, params)
            else:
                return cursor.execute(query)
        except sqlite3.OperationalError as e:
            error_str = str(e)
            if "within a transaction" in error_str:
                # 如果是事務錯誤，嘗試提交或回滾後重試
                logger.warning(f"事務錯誤: {e}, 嘗試重設事務後重試")
                try:
                    self.conn.commit()
                except:
                    try:
                        self.conn.rollback()
                    except:
                        pass
                
                # 重新嘗試
                cursor = self.conn.cursor()
                if params:
                    return cursor.execute(query, params)
                else:
                    return cursor.execute(query)
            else:
                logger.error(f"SQL執行錯誤: {e}, 查詢: {query}")
                raise
        except Exception as e:
            logger.error(f"SQL執行錯誤: {e}, 查詢: {query}")
            raise

    def executemany(self, query, params_list):
        """
        批量執行SQL查詢 - 創建新游標避免遞歸問題
        
        Args:
            query: SQL查詢字符串
            params_list: 參數列表，每個元素對應一次執行
            
        Returns:
            游標對象
        """
        try:
            # 每次操作創建新游標
            cursor = self.conn.cursor()
            return cursor.executemany(query, params_list)
        except sqlite3.OperationalError as e:
            error_str = str(e)
            if "within a transaction" in error_str:
                # 如果是事務錯誤，嘗試提交或回滾後重試
                logger.warning(f"批量事務錯誤: {e}, 嘗試重設事務後重試")
                try:
                    self.conn.commit()
                except:
                    try:
                        self.conn.rollback()
                    except:
                        pass
                
                # 重新嘗試
                cursor = self.conn.cursor()
                return cursor.executemany(query, params_list)
            else:
                logger.error(f"批量SQL執行錯誤: {e}, 查詢: {query}")
                raise
        except Exception as e:
            logger.error(f"批量SQL執行錯誤: {e}, 查詢: {query}")
            raise

    def commit(self):
        """提交事務"""
        try:
            self.conn.commit()
        except sqlite3.OperationalError as e:
            # 忽略"no transaction is active"錯誤
            logger.debug(f"提交事務時發生操作錯誤: {e}")
            pass

    def rollback(self):
        """回滾事務"""
        try:
            self.conn.rollback()
        except sqlite3.OperationalError as e:
            # 忽略"no transaction is active"錯誤
            logger.debug(f"回滾事務時發生操作錯誤: {e}")
            pass
    
    def close(self):
        """關閉數據庫連接"""
        if self.conn:
            self.conn.close()
            logger.info("數據庫連接已關閉")
    
    def insert_order(self, order_data):
        """
        插入訂單記錄
        
        Args:
            order_data: 訂單數據字典
            
        Returns:
            插入的行ID
        """
        try:
            # 檢查是否有活動交易
            try:
                # 先嘗試提交任何可能存在的交易
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
                
            query = """
            INSERT INTO completed_orders 
            (order_id, symbol, side, quantity, price, maker, fee, fee_asset, trade_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                order_data['order_id'],
                order_data['symbol'],
                order_data['side'],
                order_data['quantity'],
                order_data['price'],
                1 if order_data['maker'] else 0,
                order_data['fee'],
                order_data['fee_asset'],
                order_data['trade_type']
            )
            
            cursor = self.execute(query, params)
            self.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"插入訂單記錄時出錯: {e}")
            # 嘗試回滾可能存在的事務
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
            return None
    
    def record_rebalance_order(self, order_id, symbol):
        """
        記錄重平衡訂單
        
        Args:
            order_id: 訂單ID
            symbol: 交易對符號
            
        Returns:
            插入的行ID
        """
        try:
            # 檢查是否有活動交易
            try:
                # 先嘗試提交任何可能存在的交易
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
                
            query = """
            INSERT INTO rebalance_orders (order_id, symbol)
            VALUES (?, ?)
            """
            cursor = self.execute(query, (order_id, symbol))
            self.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"記錄重平衡訂單時出錯: {e}")
            # 嘗試回滾可能存在的事務
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
            return None
    
    def is_rebalance_order(self, order_id, symbol):
        """
        檢查訂單是否為重平衡訂單
        
        Args:
            order_id: 訂單ID
            symbol: 交易對符號
            
        Returns:
            布爾值，表示是否為重平衡訂單
        """
        try:
            # 檢查是否有活動交易
            try:
                # 先嘗試提交任何可能存在的交易
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
                
            query = """
            SELECT id FROM rebalance_orders 
            WHERE order_id = ? AND symbol = ?
            """
            cursor = self.execute(query, (order_id, symbol))
            result = cursor.fetchone()
            cursor.close()  # 關閉游標
            return result is not None
        except Exception as e:
            logger.error(f"檢查重平衡訂單時出錯: {e}")
            return False
    
    def update_market_data(self, market_data):
        """
        更新市場數據
        
        Args:
            market_data: 市場數據字典
            
        Returns:
            插入的行ID
        """
        cursor = None
        try:
            # 檢查是否有活動交易並嘗試提交
            try:
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
                
            # 創建新的游標
            cursor = self.conn.cursor()
            
            query = """
            INSERT INTO market_data 
            (symbol, price, volume, bid_ask_spread, liquidity_score)
            VALUES (?, ?, ?, ?, ?)
            """
            params = (
                market_data['symbol'],
                market_data['price'],
                market_data['volume'],
                market_data['bid_ask_spread'],
                market_data['liquidity_score']
            )
            
            cursor.execute(query, params)
            lastrowid = cursor.lastrowid
            
            # 提交事務
            self.conn.commit()
            return lastrowid
        except Exception as e:
            logger.error(f"更新市場數據時出錯: {e}")
            # 嘗試回滾可能存在的事務
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
            return None
        finally:
            # 確保無論如何都關閉游標
            if cursor:
                cursor.close()
    
    def update_trading_stats(self, stats_data):
        """
        更新交易統計數據
        
        Args:
            stats_data: 統計數據字典
            
        Returns:
            布爾值，表示更新是否成功
        """
        cursor = None
        try:
            # 嘗試提交任何可能存在的交易
            try:
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
                
            # 創建獨立的游標進行全部操作
            cursor = self.conn.cursor()
            
            # 檢查今天的記錄是否存在
            check_query = """
            SELECT id FROM trading_stats
            WHERE date = ? AND symbol = ?
            """
            cursor.execute(check_query, (stats_data['date'], stats_data['symbol']))
            record = cursor.fetchone()
            
            if record:
                # 更新現有記錄
                update_query = """
                UPDATE trading_stats
                SET maker_buy_volume = ?,
                    maker_sell_volume = ?,
                    taker_buy_volume = ?,
                    taker_sell_volume = ?,
                    realized_profit = ?,
                    total_fees = ?,
                    net_profit = ?,
                    avg_spread = ?,
                    trade_count = ?,
                    volatility = ?
                WHERE date = ? AND symbol = ?
                """
                params = (
                    stats_data['maker_buy_volume'],
                    stats_data['maker_sell_volume'],
                    stats_data['taker_buy_volume'],
                    stats_data['taker_sell_volume'],
                    stats_data['realized_profit'],
                    stats_data['total_fees'],
                    stats_data['net_profit'],
                    stats_data['avg_spread'],
                    stats_data['trade_count'],
                    stats_data['volatility'],
                    stats_data['date'],
                    stats_data['symbol']
                )
                cursor.execute(update_query, params)
            else:
                # 創建新記錄
                insert_query = """
                INSERT INTO trading_stats
                (date, symbol, maker_buy_volume, maker_sell_volume, taker_buy_volume, taker_sell_volume, 
                realized_profit, total_fees, net_profit, avg_spread, trade_count, volatility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    stats_data['date'],
                    stats_data['symbol'],
                    stats_data['maker_buy_volume'],
                    stats_data['maker_sell_volume'],
                    stats_data['taker_buy_volume'],
                    stats_data['taker_sell_volume'],
                    stats_data['realized_profit'],
                    stats_data['total_fees'],
                    stats_data['net_profit'],
                    stats_data['avg_spread'],
                    stats_data['trade_count'],
                    stats_data['volatility']
                )
                cursor.execute(insert_query, params)
            
            # 提交事務
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"更新交易統計時出錯: {e}")
            # 嘗試回滾可能存在的事務
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
            return False
        finally:
            # 確保無論如何都關閉游標
            if cursor:
                cursor.close()
    
    def get_trading_stats(self, symbol, date=None):
        """
        獲取交易統計數據
        
        Args:
            symbol: 交易對符號
            date: 日期字符串，如果為None則獲取所有日期的統計
            
        Returns:
            統計數據列表
        """
        try:
            # 檢查是否有活動交易
            try:
                # 先嘗試提交任何可能存在的交易
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"錯誤
                pass
                
            cursor = self.conn.cursor()  # 創建新游標
            
            if date:
                query = """
                SELECT * FROM trading_stats
                WHERE symbol = ? AND date = ?
                """
                cursor.execute(query, (symbol, date))
            else:
                query = """
                SELECT * FROM trading_stats
                WHERE symbol = ?
                ORDER BY date DESC
                """
                cursor.execute(query, (symbol,))
            
            columns = [description[0] for description in cursor.description]
            result = []
            for row in cursor.fetchall():
                result.append(dict(zip(columns, row)))
            
            cursor.close()  # 關閉游標
            return result
        except Exception as e:
            logger.error(f"獲取交易統計時出錯: {e}")
            return []
    
    def get_all_time_stats(self, symbol):
        """
        獲取所有時間的總計統計數據
        
        Args:
            symbol: 交易對符號
            
        Returns:
            總計統計數據字典
        """
        cursor = self.conn.cursor()  # 創建新游標
        
        query = """
        SELECT 
            SUM(maker_buy_volume) as total_maker_buy,
            SUM(maker_sell_volume) as total_maker_sell,
            SUM(taker_buy_volume) as total_taker_buy,
            SUM(taker_sell_volume) as total_taker_sell,
            SUM(realized_profit) as total_profit,
            SUM(total_fees) as total_fees,
            SUM(net_profit) as total_net_profit,
            AVG(avg_spread) as avg_spread_all_time
        FROM trading_stats
        WHERE symbol = ?
        """
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        
        if result and result[0] is not None:
            columns = ['total_maker_buy', 'total_maker_sell', 'total_taker_buy', 
                      'total_taker_sell', 'total_profit', 'total_fees', 
                      'total_net_profit', 'avg_spread_all_time']
            stat_dict = dict(zip(columns, result))
            cursor.close()  # 關閉游標
            return stat_dict
            
        cursor.close()  # 關閉游標
        return None
    
    def get_recent_trades(self, symbol, limit=10):
        """
        獲取最近的成交記錄
        
        Args:
            symbol: 交易對符號
            limit: 返回記錄數量限制
            
        Returns:
            成交記錄列表
        """
        cursor = self.conn.cursor()  # 創建新游標
        
        query = """
        SELECT side, quantity, price, maker, fee, timestamp
        FROM completed_orders
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        cursor.execute(query, (symbol, limit))
        
        columns = ['side', 'quantity', 'price', 'maker', 'fee', 'timestamp']
        result = []
        for row in cursor.fetchall():
            result.append(dict(zip(columns, row)))
        
        cursor.close()  # 關閉游標
        return result
    
    def get_order_history(self, symbol, limit=1000):
        """
        獲取訂單歷史
        
        Args:
            symbol: 交易對符號
            limit: 返回記錄數量限制
            
        Returns:
            訂單記錄列表
        """
        cursor = self.conn.cursor()  # 創建新游標
        
        query = """
        SELECT side, quantity, price, maker, fee
        FROM completed_orders
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        cursor.execute(query, (symbol, limit))
        
        result = cursor.fetchall()
        cursor.close()  # 關閉游標
        return result