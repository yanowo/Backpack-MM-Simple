"""
数据库操作模块
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
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._init_tables()
    
    def _connect(self):
        """建立数据库连接"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # 主游标只用于初始化
            self.cursor = self.conn.cursor()
            logger.info(f"数据库连接成功: {self.db_path}")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def _init_tables(self):
        """初始化资料库表结构"""
        try:
            # 改进的交易记录表
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
            
            # 创建索引提高查询效率
            self.cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_completed_orders_symbol 
                ON completed_orders(symbol)
                """
            )
            
            # 统计表来跟踪每日/每周成交量和利润
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
            
            # 重平衡订单记录表
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
            
            # 市场数据表
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
                logger.info("数据库表初始化成功")
            except sqlite3.OperationalError as e:
                logger.debug(f"提交表初始化时出错: {e}")
        except Exception as e:
            logger.error(f"初始化资料库表时出错: {e}")
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
            raise

    def execute(self, query, params=None):
        """
        执行SQL查询 - 创建新游标避免递归问题
        
        Args:
            query: SQL查询字符串
            params: 查询参数
            
        Returns:
            游标对象
        """
        try:
            # 每次操作创建新游标
            cursor = self.conn.cursor()
            if params:
                return cursor.execute(query, params)
            else:
                return cursor.execute(query)
        except sqlite3.OperationalError as e:
            error_str = str(e)
            if "within a transaction" in error_str:
                # 如果是事务错误，尝试提交或回滚后重试
                logger.warning(f"事务错误: {e}, 尝试重设事务后重试")
                try:
                    self.conn.commit()
                except:
                    try:
                        self.conn.rollback()
                    except:
                        pass
                
                # 重新尝试
                cursor = self.conn.cursor()
                if params:
                    return cursor.execute(query, params)
                else:
                    return cursor.execute(query)
            else:
                logger.error(f"SQL执行错误: {e}, 查询: {query}")
                raise
        except Exception as e:
            logger.error(f"SQL执行错误: {e}, 查询: {query}")
            raise

    def executemany(self, query, params_list):
        """
        批量执行SQL查询 - 创建新游标避免递归问题
        
        Args:
            query: SQL查询字符串
            params_list: 参数列表，每个元素对应一次执行
            
        Returns:
            游标对象
        """
        try:
            # 每次操作创建新游标
            cursor = self.conn.cursor()
            return cursor.executemany(query, params_list)
        except sqlite3.OperationalError as e:
            error_str = str(e)
            if "within a transaction" in error_str:
                # 如果是事务错误，尝试提交或回滚后重试
                logger.warning(f"批量事务错误: {e}, 尝试重设事务后重试")
                try:
                    self.conn.commit()
                except:
                    try:
                        self.conn.rollback()
                    except:
                        pass
                
                # 重新尝试
                cursor = self.conn.cursor()
                return cursor.executemany(query, params_list)
            else:
                logger.error(f"批量SQL执行错误: {e}, 查询: {query}")
                raise
        except Exception as e:
            logger.error(f"批量SQL执行错误: {e}, 查询: {query}")
            raise

    def commit(self):
        """提交事务"""
        try:
            self.conn.commit()
        except sqlite3.OperationalError as e:
            # 忽略"no transaction is active"错误
            logger.debug(f"提交事务时发生操作错误: {e}")
            pass

    def rollback(self):
        """回滚事务"""
        try:
            self.conn.rollback()
        except sqlite3.OperationalError as e:
            # 忽略"no transaction is active"错误
            logger.debug(f"回滚事务时发生操作错误: {e}")
            pass
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
    
    def insert_order(self, order_data):
        """
        插入订单记录
        
        Args:
            order_data: 订单数据字典
            
        Returns:
            插入的行ID
        """
        try:
            # 检查是否有活动交易
            try:
                # 先尝试提交任何可能存在的交易
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
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
            logger.error(f"插入订单记录时出错: {e}")
            # 尝试回滚可能存在的事务
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
            return None
    
    def record_rebalance_order(self, order_id, symbol):
        """
        记录重平衡订单
        
        Args:
            order_id: 订单ID
            symbol: 交易对符号
            
        Returns:
            插入的行ID
        """
        try:
            # 检查是否有活动交易
            try:
                # 先尝试提交任何可能存在的交易
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
                
            query = """
            INSERT INTO rebalance_orders (order_id, symbol)
            VALUES (?, ?)
            """
            cursor = self.execute(query, (order_id, symbol))
            self.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"记录重平衡订单时出错: {e}")
            # 尝试回滚可能存在的事务
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
            return None
    
    def is_rebalance_order(self, order_id, symbol):
        """
        检查订单是否为重平衡订单
        
        Args:
            order_id: 订单ID
            symbol: 交易对符号
            
        Returns:
            布尔值，表示是否为重平衡订单
        """
        try:
            # 检查是否有活动交易
            try:
                # 先尝试提交任何可能存在的交易
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
                
            query = """
            SELECT id FROM rebalance_orders 
            WHERE order_id = ? AND symbol = ?
            """
            cursor = self.execute(query, (order_id, symbol))
            result = cursor.fetchone()
            cursor.close()  # 关闭游标
            return result is not None
        except Exception as e:
            logger.error(f"检查重平衡订单时出错: {e}")
            return False
    
    def update_market_data(self, market_data):
        """
        更新市场数据
        
        Args:
            market_data: 市场数据字典
            
        Returns:
            插入的行ID
        """
        cursor = None
        try:
            # 检查是否有活动交易并尝试提交
            try:
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
                
            # 创建新的游标
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
            
            # 提交事务
            self.conn.commit()
            return lastrowid
        except Exception as e:
            logger.error(f"更新市场数据时出错: {e}")
            # 尝试回滚可能存在的事务
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
            return None
        finally:
            # 确保无论如何都关闭游标
            if cursor:
                cursor.close()
    
    def update_trading_stats(self, stats_data):
        """
        更新交易统计数据
        
        Args:
            stats_data: 统计数据字典
            
        Returns:
            布尔值，表示更新是否成功
        """
        cursor = None
        try:
            # 尝试提交任何可能存在的交易
            try:
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
                
            # 创建独立的游标进行全部操作
            cursor = self.conn.cursor()
            
            # 检查今天的记录是否存在
            check_query = """
            SELECT id FROM trading_stats
            WHERE date = ? AND symbol = ?
            """
            cursor.execute(check_query, (stats_data['date'], stats_data['symbol']))
            record = cursor.fetchone()
            
            if record:
                # 更新现有记录
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
                # 创建新记录
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
            
            # 提交事务
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"更新交易统计时出错: {e}")
            # 尝试回滚可能存在的事务
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
            return False
        finally:
            # 确保无论如何都关闭游标
            if cursor:
                cursor.close()
    
    def get_trading_stats(self, symbol, date=None):
        """
        获取交易统计数据
        
        Args:
            symbol: 交易对符号
            date: 日期字符串，如果为None则获取所有日期的统计
            
        Returns:
            统计数据列表
        """
        try:
            # 检查是否有活动交易
            try:
                # 先尝试提交任何可能存在的交易
                self.conn.commit()
            except sqlite3.OperationalError:
                # 忽略"no transaction is active"错误
                pass
                
            cursor = self.conn.cursor()  # 创建新游标
            
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
            
            cursor.close()  # 关闭游标
            return result
        except Exception as e:
            logger.error(f"获取交易统计时出错: {e}")
            return []
    
    def get_all_time_stats(self, symbol):
        """
        获取所有时间的总计统计数据
        
        Args:
            symbol: 交易对符号
            
        Returns:
            总计统计数据字典
        """
        cursor = self.conn.cursor()  # 创建新游标
        
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
            cursor.close()  # 关闭游标
            return stat_dict
            
        cursor.close()  # 关闭游标
        return None
    
    def get_recent_trades(self, symbol, limit=10):
        """
        获取最近的成交记录
        
        Args:
            symbol: 交易对符号
            limit: 返回记录数量限制
            
        Returns:
            成交记录列表
        """
        cursor = self.conn.cursor()  # 创建新游标
        
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
        
        cursor.close()  # 关闭游标
        return result
    
    def get_order_history(self, symbol, limit=1000):
        """
        获取订单历史
        
        Args:
            symbol: 交易对符号
            limit: 返回记录数量限制
            
        Returns:
            订单记录列表
        """
        cursor = self.conn.cursor()  # 创建新游标
        
        query = """
        SELECT side, quantity, price, maker, fee
        FROM completed_orders
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        cursor.execute(query, (symbol, limit))
        
        result = cursor.fetchall()
        cursor.close()  # 关闭游标
        return result