"""
WebSocket客户端模块
"""
import json
import time
import threading
import websocket as ws
from typing import Dict, List, Tuple, Any, Optional, Callable
from config import WS_URL, DEFAULT_WINDOW
from api.auth import create_signature
from api.bp_client import BPClient
from utils.helpers import calculate_volatility
from logger import setup_logger

logger = setup_logger("backpack_ws")

class BackpackWebSocket:
    def __init__(self, api_key, secret_key, symbol, on_message_callback=None, auto_reconnect=True, proxy=None):
        """
        初始化WebSocket客户端
        
        Args:
            api_key: API密钥
            secret_key: API密钥
            symbol: 交易对符号
            on_message_callback: 消息回调函数
            auto_reconnect: 是否自动重连
            proxy:  wss代理 支持格式为 http://user:pass@host:port/ 或者 http://host:port

        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.ws = None
        self.on_message_callback = on_message_callback
        self.connected = False
        self.last_price = None
        self.bid_price = None
        self.ask_price = None
        self.orderbook = {"bids": [], "asks": []}
        self.order_updates = []
        self.historical_prices = []  # 储存历史价格用于计算波动率
        self.max_price_history = 100  # 最多储存的价格数量
        
        # 重连相关参数
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = 1
        self.max_reconnect_delay = 30
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.running = False
        self.ws_thread = None
        
        # 记录已订阅的频道
        self.subscriptions = []
        
        # 添加WebSocket执行绪锁
        self.ws_lock = threading.Lock()
        
        # 添加心跳检测
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30
        self.heartbeat_thread = None

        # 添加代理参数
        self.proxy = proxy
        
        # 客户端缓存，避免重复创建实例
        self._client_cache = {}
        
        # 添加重连中标志，避免多次重连
        self.reconnecting = False

    def _get_client(self):
        """获取缓存的客户端实例，避免重复创建"""
        cache_key = "public"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = BPClient({})
        return self._client_cache[cache_key]

    def initialize_orderbook(self):
        """通过REST API获取订单簿初始快照"""
        try:
            # 使用REST API获取完整订单簿
            order_book = self._get_client().get_order_book(self.symbol, 100)  # 增加深度
            if isinstance(order_book, dict) and "error" in order_book:
                logger.error(f"初始化订单簿失败: {order_book['error']}")
                return False
            
            # 重置并填充orderbook数据结构
            self.orderbook = {
                "bids": [[float(price), float(quantity)] for price, quantity in order_book.get('bids', [])],
                "asks": [[float(price), float(quantity)] for price, quantity in order_book.get('asks', [])]
            }
            
            # 按价格排序
            self.orderbook["bids"] = sorted(self.orderbook["bids"], key=lambda x: x[0], reverse=True)
            self.orderbook["asks"] = sorted(self.orderbook["asks"], key=lambda x: x[0])
            
            logger.info(f"订单簿初始化成功: {len(self.orderbook['bids'])} 个买单, {len(self.orderbook['asks'])} 个卖单")
            
            # 初始化最高买价和最低卖价
            if self.orderbook["bids"]:
                self.bid_price = self.orderbook["bids"][0][0]
            if self.orderbook["asks"]:
                self.ask_price = self.orderbook["asks"][0][0]
            if self.bid_price and self.ask_price:
                self.last_price = (self.bid_price + self.ask_price) / 2
                self.add_price_to_history(self.last_price)
            
            return True
        except Exception as e:
            logger.error(f"初始化订单簿时出错: {e}")
            return False
    
    def add_price_to_history(self, price):
        """添加价格到历史记录用于计算波动率"""
        if price:
            self.historical_prices.append(price)
            # 保持历史记录在设定长度内
            if len(self.historical_prices) > self.max_price_history:
                self.historical_prices = self.historical_prices[-self.max_price_history:]
    
    def get_volatility(self, window=20):
        """获取当前波动率"""
        return calculate_volatility(self.historical_prices, window)
    
    def start_heartbeat(self):
        """开始心跳检测线程"""
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_check, daemon=True)
            self.heartbeat_thread.start()
    
    def _heartbeat_check(self):
        """定期检查WebSocket连接状态并在需要时重连"""
        while self.running:
            current_time = time.time()
            time_since_last_heartbeat = current_time - self.last_heartbeat
            
            if time_since_last_heartbeat > self.heartbeat_interval * 2:
                logger.warning(f"心跳检测超时 ({time_since_last_heartbeat:.1f}秒)，尝试重新连接")
                # 使用非阻塞方式触发重连
                threading.Thread(target=self._trigger_reconnect, daemon=True).start()
                
            time.sleep(5)  # 每5秒检查一次
    
    def _trigger_reconnect(self):
        """非阻塞触发重连"""
        if not self.reconnecting:
            self.reconnect()
        
    def connect(self):
        """建立WebSocket连接"""
        try:
            self.running = True
            self.reconnect_attempts = 0
            self.reconnecting = False
            ws.enableTrace(False)
            self.ws = ws.WebSocketApp(
                WS_URL,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_ping=self.on_ping,
                on_pong=self.on_pong
            )
            
            self.ws_thread = threading.Thread(target=self.ws_run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # 启动心跳检测
            self.start_heartbeat()
        except Exception as e:
            logger.error(f"初始化WebSocket连接时出错: {e}")
    
    def ws_run_forever(self):
        """WebSocket运行循环 - 修复版本"""
        try:
            # 确保在运行前检查socket状态
            if hasattr(self.ws, 'sock') and self.ws.sock and self.ws.sock.connected:
                logger.debug("发现socket已经打开，跳过run_forever")
                return

            proxy_type = None
            http_proxy_auth = None
            http_proxy_host = None
            http_proxy_port = None
            
            if self.proxy and 3 <= len(self.proxy.split(":")) <= 4:
                arrs = self.proxy.split(":")
                proxy_type = arrs[0]
                arrs[1] = arrs[1][2:]  # 去掉 //
                if len(arrs) == 3:
                    http_proxy_host = arrs[1]
                else:
                    password, http_proxy_host = arrs[2].split("@")
                    http_proxy_auth = (arrs[1], password)
                http_proxy_port = arrs[-1]

            # 添加ping_interval和ping_timeout参数
            self.ws.run_forever(
                ping_interval=self.heartbeat_interval, 
                ping_timeout=10, 
                http_proxy_auth=http_proxy_auth, 
                http_proxy_host=http_proxy_host, 
                http_proxy_port=http_proxy_port, 
                proxy_type=proxy_type
            )

        except Exception as e:
            logger.error(f"WebSocket运行时出错: {e}")
        finally:
            # 移除可能导致递归调用的重连逻辑
            logger.debug("WebSocket run_forever 执行结束")
    
    def on_pong(self, ws, message):
        """处理pong响应"""
        self.last_heartbeat = time.time()
        
    def reconnect(self):
        """完全断开并重新建立WebSocket连接 - 修复版本"""
        # 防止多次重连
        if self.reconnecting:
            logger.debug("重连已在进行中，跳过此次重连请求")
            return False
            
        with self.ws_lock:
            if not self.running or self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.warning(f"重连次数超过上限 ({self.max_reconnect_attempts})，停止重连")
                return False

            self.reconnecting = True
            self.reconnect_attempts += 1
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), self.max_reconnect_delay)
            
            logger.info(f"尝试第 {self.reconnect_attempts} 次重连，等待 {delay} 秒...")
            time.sleep(delay)
            
            # 确保完全断开连接前先标记连接状态
            self.connected = False
            
            # 优雅关闭现有连接
            self._force_close_connection()
            
            # 重置所有相关状态
            self.ws_thread = None
            self.subscriptions = []  # 清空订阅列表，以便重新订閴
            
            try:
                # 创建全新的WebSocket连接
                ws.enableTrace(False)
                self.ws = ws.WebSocketApp(
                    WS_URL,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_ping=self.on_ping,
                    on_pong=self.on_pong
                )
                
                # 创建新线程
                self.ws_thread = threading.Thread(target=self.ws_run_forever)
                self.ws_thread.daemon = True
                self.ws_thread.start()
                
                # 更新最后心跳时间，避免重连后立即触发心跳检测
                self.last_heartbeat = time.time()
                
                logger.info(f"第 {self.reconnect_attempts} 次重连已启动")
                
                self.reconnecting = False
                return True
                
            except Exception as e:
                logger.error(f"重连过程中发生错误: {e}")
                self.reconnecting = False
                return False
    
    def _force_close_connection(self):
        """强制关闭现有连接"""
        try:
            # 完全断开并清理之前的WebSocket连接
            if self.ws:
                try:
                    # 显式设置内部标记表明这是用户主动关闭
                    if hasattr(self.ws, '_closed_by_me'):
                        self.ws._closed_by_me = True
                    
                    # 关闭WebSocket
                    self.ws.close()
                    if hasattr(self.ws, 'keep_running'):
                        self.ws.keep_running = False
                    
                    # 强制关闭socket
                    if hasattr(self.ws, 'sock') and self.ws.sock:
                        try:
                            self.ws.sock.close()
                            self.ws.sock = None
                        except:
                            pass
                except Exception as e:
                    logger.debug(f"关闭WebSocket时的预期错误: {e}")
                
                self.ws = None
                
            # 处理旧线程 - 使用较短的超时时间
            if self.ws_thread and self.ws_thread.is_alive():
                try:
                    # 不要无限等待线程结束
                    self.ws_thread.join(timeout=1.0)
                    if self.ws_thread.is_alive():
                        logger.warning("旧线程未能在超时时间内结束，但继续重连过程")
                except Exception as e:
                    logger.debug(f"等待旧线程终止时出错: {e}")
            
            # 给系统少量时间清理资源
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"强制关闭连接时出错: {e}")
        
    def on_ping(self, ws, message):
        """处理ping消息"""
        try:
            self.last_heartbeat = time.time()
            if ws and hasattr(ws, 'sock') and ws.sock:
                ws.sock.pong(message)
            else:
                logger.debug("无法回应ping：WebSocket或sock为None")
        except Exception as e:
            logger.debug(f"回应ping失败: {e}")
        
    def on_open(self, ws):
        """WebSocket打开时的处理"""
        logger.info("WebSocket连接已建立")
        self.connected = True
        self.reconnect_attempts = 0
        self.reconnecting = False
        self.last_heartbeat = time.time()
        
        # 添加短暂延迟确保连接稳定
        time.sleep(0.5)
        
        # 初始化订单簿
        orderbook_initialized = self.initialize_orderbook()
        
        # 如果初始化成功，订阅深度和行情数据
        if orderbook_initialized:
            if "bookTicker" in self.subscriptions or not self.subscriptions:
                self.subscribe_bookTicker()
            
            if "depth" in self.subscriptions or not self.subscriptions:
                self.subscribe_depth()
        
        # 重新订阅私有订单更新流
        for sub in self.subscriptions:
            if sub.startswith("account."):
                self.private_subscribe(sub)
    
    def subscribe_bookTicker(self):
        """订阅最优价格"""
        logger.info(f"订阅 {self.symbol} 的bookTicker...")
        if not self.connected or not self.ws:
            logger.warning("WebSocket未连接，无法订阅bookTicker")
            return False
            
        try:
            message = {
                "method": "SUBSCRIBE",
                "params": [f"bookTicker.{self.symbol}"]
            }
            self.ws.send(json.dumps(message))
            if "bookTicker" not in self.subscriptions:
                self.subscriptions.append("bookTicker")
            return True
        except Exception as e:
            logger.error(f"订閴bookTicker失败: {e}")
            return False
    
    def subscribe_depth(self):
        """订閴深度信息"""
        logger.info(f"订閴 {self.symbol} 的深度信息...")
        if not self.connected or not self.ws:
            logger.warning("WebSocket未连接，无法订閴深度信息")
            return False
            
        try:
            message = {
                "method": "SUBSCRIBE",
                "params": [f"depth.{self.symbol}"]
            }
            self.ws.send(json.dumps(message))
            if "depth" not in self.subscriptions:
                self.subscriptions.append("depth")
            return True
        except Exception as e:
            logger.error(f"订閴深度信息失败: {e}")
            return False
    
    def private_subscribe(self, stream):
        """订閴私有数据流"""
        if not self.connected or not self.ws:
            logger.warning("WebSocket未连接，无法订閴私有数据流")
            return False
            
        try:
            timestamp = str(int(time.time() * 1000))
            window = DEFAULT_WINDOW
            sign_message = f"instruction=subscribe&timestamp={timestamp}&window={window}"
            signature = create_signature(self.secret_key, sign_message)
            
            if not signature:
                logger.error("签名创建失败，无法订閴私有数据流")
                return False
            
            message = {
                "method": "SUBSCRIBE",
                "params": [stream],
                "signature": [self.api_key, signature, timestamp, window]
            }
            
            self.ws.send(json.dumps(message))
            logger.info(f"已订閴私有数据流: {stream}")
            if stream not in self.subscriptions:
                self.subscriptions.append(stream)
            return True
        except Exception as e:
            logger.error(f"订閴私有数据流失败: {e}")
            return False
    
    def on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            # 处理ping pong消息
            if isinstance(data, dict) and data.get("ping"):
                pong_message = {"pong": data.get("ping")}
                if self.ws and self.connected:
                    self.ws.send(json.dumps(pong_message))
                    self.last_heartbeat = time.time()
                return
            
            if "stream" in data and "data" in data:
                stream = data["stream"]
                event_data = data["data"]
                
                # 处理bookTicker
                if stream.startswith("bookTicker."):
                    if 'b' in event_data and 'a' in event_data:
                        self.bid_price = float(event_data['b'])
                        self.ask_price = float(event_data['a'])
                        self.last_price = (self.bid_price + self.ask_price) / 2
                        # 记录历史价格用于计算波动率
                        self.add_price_to_history(self.last_price)
                
                # 处理depth
                elif stream.startswith("depth."):
                    if 'b' in event_data and 'a' in event_data:
                        self._update_orderbook(event_data)
                
                # 订单更新数据流
                elif stream.startswith("account.orderUpdate."):
                    self.order_updates.append(event_data)
                    
                if self.on_message_callback:
                    self.on_message_callback(stream, event_data)
            
        except Exception as e:
            logger.error(f"处理WebSocket消息时出错: {e}")
    
    def _update_orderbook(self, data):
        """更新订单簿（优化处理速度）"""
        # 处理买单更新
        if 'b' in data:
            for bid in data['b']:
                price = float(bid[0])
                quantity = float(bid[1])
                
                # 使用二分查找来优化插入位置查找
                if quantity == 0:
                    # 移除价位
                    self.orderbook["bids"] = [b for b in self.orderbook["bids"] if b[0] != price]
                else:
                    # 先查找是否存在相同价位
                    found = False
                    for i, b in enumerate(self.orderbook["bids"]):
                        if b[0] == price:
                            self.orderbook["bids"][i] = [price, quantity]
                            found = True
                            break
                    
                    # 如果不存在，插入并保持排序
                    if not found:
                        self.orderbook["bids"].append([price, quantity])
                        # 按价格降序排序
                        self.orderbook["bids"] = sorted(self.orderbook["bids"], key=lambda x: x[0], reverse=True)
        
        # 处理卖单更新
        if 'a' in data:
            for ask in data['a']:
                price = float(ask[0])
                quantity = float(ask[1])
                
                if quantity == 0:
                    # 移除价位
                    self.orderbook["asks"] = [a for a in self.orderbook["asks"] if a[0] != price]
                else:
                    # 先查找是否存在相同价位
                    found = False
                    for i, a in enumerate(self.orderbook["asks"]):
                        if a[0] == price:
                            self.orderbook["asks"][i] = [price, quantity]
                            found = True
                            break
                    
                    # 如果不存在，插入并保持排序
                    if not found:
                        self.orderbook["asks"].append([price, quantity])
                        # 按价格升序排序
                        self.orderbook["asks"] = sorted(self.orderbook["asks"], key=lambda x: x[0])
    
    def on_error(self, ws, error):
        """处理WebSocket错误"""
        logger.error(f"WebSocket发生错误: {error}")
        self.last_heartbeat = 0  # 强制触发重连
    
    def on_close(self, ws, close_status_code, close_msg):
        """处理WebSocket关闭"""
        previous_connected = self.connected
        self.connected = False
        logger.info(f"WebSocket连接已关闭: {close_msg if close_msg else 'No message'} (状态码: {close_status_code if close_status_code else 'None'})")
        
        # 清理当前socket资源
        if hasattr(ws, 'sock') and ws.sock:
            try:
                ws.sock.close()
                ws.sock = None
            except Exception as e:
                logger.debug(f"关闭socket时出错: {e}")
        
        if close_status_code == 1000 or getattr(ws, '_closed_by_me', False):
            logger.info("WebSocket正常关闭，不进行重连")
        elif previous_connected and self.running and self.auto_reconnect and not self.reconnecting:
            logger.info("WebSocket非正常关闭，将自动重连")
            # 使用线程触发重连，避免在回调中直接重连
            threading.Thread(target=self._trigger_reconnect, daemon=True).start()
    
    def close(self):
        """完全关闭WebSocket连接"""
        logger.info("主动关闭WebSocket连接...")
        self.running = False
        self.connected = False
        self.reconnecting = False
        
        # 停止心跳检测线程
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            try:
                self.heartbeat_thread.join(timeout=1)
            except Exception:
                pass
        self.heartbeat_thread = None
        
        # 强制关闭连接
        self._force_close_connection()
        
        # 重置订閴状态
        self.subscriptions = []
        
        logger.info("WebSocket连接已完全关闭")
    
    def get_current_price(self):
        """获取当前价格"""
        return self.last_price
    
    def get_bid_ask(self):
        """获取买卖价"""
        return self.bid_price, self.ask_price
    
    def get_orderbook(self):
        """获取订单簿"""
        return self.orderbook

    def is_connected(self):
        """检查连接状态"""
        if not self.connected:
            return False
        if not self.ws:
            return False
        if not hasattr(self.ws, 'sock') or not self.ws.sock:
            return False
        
        # 检查socket是否连接
        try:
            return self.ws.sock.connected
        except:
            return False
    
    def get_liquidity_profile(self, depth_percentage=0.01):
        """分析市场流动性特征"""
        if not self.orderbook["bids"] or not self.orderbook["asks"]:
            return None
        
        mid_price = (self.bid_price + self.ask_price) / 2 if self.bid_price and self.ask_price else None
        if not mid_price:
            return None
        
        # 计算价格范围
        min_price = mid_price * (1 - depth_percentage)
        max_price = mid_price * (1 + depth_percentage)
        
        # 分析买卖单流动性
        bid_volume = sum(qty for price, qty in self.orderbook["bids"] if price >= min_price)
        ask_volume = sum(qty for price, qty in self.orderbook["asks"] if price <= max_price)
        
        # 计算买卖比例
        ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        
        # 买卖压力差异
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'volume_ratio': ratio,
            'imbalance': imbalance,
            'mid_price': mid_price
        }
    
    def check_and_reconnect_if_needed(self):
        """检查连接状态并在需要时重连 - 供外部调用"""
        if not self.is_connected() and not self.reconnecting:
            logger.info("外部检查发现连接断开，触发重连...")
            threading.Thread(target=self._trigger_reconnect, daemon=True).start()
        return self.is_connected()