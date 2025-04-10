"""
WebSocket客户端模塊
"""
import json
import time
import threading
import signal
import websocket as ws
from typing import Dict, List, Tuple, Any, Optional, Callable
from config import WS_URL, DEFAULT_WINDOW
from api.auth import create_signature
from api.client import get_order_book
from utils.helpers import calculate_volatility
from logger import setup_logger

logger = setup_logger("backpack_ws")

class BackpackWebSocket:
    def __init__(self, api_key, secret_key, symbol, on_message_callback=None, auto_reconnect=True, proxy=None):
        """
        初始化WebSocket客户端
        
        Args:
            api_key: API密鑰
            secret_key: API密鑰
            symbol: 交易對符號
            on_message_callback: 消息回調函數
            auto_reconnect: 是否自動重連
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
        self.historical_prices = []  # 儲存歷史價格用於計算波動率
        self.max_price_history = 100  # 最多儲存的價格數量
        
        # 重連相關參數
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = 1
        self.max_reconnect_delay = 30
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.running = False
        self.ws_thread = None
        
        # 記錄已訂閲的頻道
        self.subscriptions = []
        
        # 添加WebSocket執行緒鎖
        self.ws_lock = threading.Lock()
        
        # 添加心跳檢測
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30
        self.heartbeat_thread = None

        # 添加代理参数
        self.proxy = proxy
        
        # 添加信號處理，使能夠響應 CTRL+C
        self._setup_signal_handlers()
        
        # 添加連接超時
        self.connect_timeout = 10  # 連接超時，秒
    
    def _setup_signal_handlers(self):
        """設置信號處理器以便正確響應中斷"""
        try:
            # 僅在主線程中設置信號處理，且只設置一次
            if threading.current_thread() is threading.main_thread() and not hasattr(self, '_signal_handlers_set'):
                signal.signal(signal.SIGINT, self._handle_signal)
                signal.signal(signal.SIGTERM, self._handle_signal)
                self._signal_handlers_set = True
                logger.debug("信號處理器已設置")
        except Exception as e:
            logger.error(f"設置信號處理器失敗: {e}")
    
    def _handle_signal(self, sig, frame):
        """處理終止信號"""
        logger.info(f"收到信號 {sig}，正在關閉連接...")
        self.close(manual_close=True)  # 明確指定這是手動關閉

    def initialize_orderbook(self):
        """通過REST API獲取訂單簿初始快照"""
        try:
            # 增加超時控制
            init_start_time = time.time()
            init_timeout = 10  # 設置超時為10秒
            
            # 使用REST API獲取完整訂單簿
            order_book = get_order_book(self.symbol, 100)  # 增加深度
            
            # 檢查超時
            if time.time() - init_start_time > init_timeout:
                logger.error("初始化訂單簿超時")
                return False
                
            if isinstance(order_book, dict) and "error" in order_book:
                logger.error(f"初始化訂單簿失敗: {order_book['error']}")
                return False
            
            # 重置並填充orderbook數據結構
            self.orderbook = {
                "bids": [[float(price), float(quantity)] for price, quantity in order_book.get('bids', [])],
                "asks": [[float(price), float(quantity)] for price, quantity in order_book.get('asks', [])]
            }
            
            # 按價格排序
            self.orderbook["bids"] = sorted(self.orderbook["bids"], key=lambda x: x[0], reverse=True)
            self.orderbook["asks"] = sorted(self.orderbook["asks"], key=lambda x: x[0])
            
            logger.info(f"訂單簿初始化成功: {len(self.orderbook['bids'])} 個買單, {len(self.orderbook['asks'])} 個賣單")
            
            # 初始化最高買價和最低賣價
            if self.orderbook["bids"]:
                self.bid_price = self.orderbook["bids"][0][0]
            if self.orderbook["asks"]:
                self.ask_price = self.orderbook["asks"][0][0]
            if self.bid_price and self.ask_price:
                self.last_price = (self.bid_price + self.ask_price) / 2
                self.add_price_to_history(self.last_price)
            
            return True
        except Exception as e:
            logger.error(f"初始化訂單簿時出錯: {e}")
            return False
    
    def add_price_to_history(self, price):
        """添加價格到歷史記錄用於計算波動率"""
        if price:
            self.historical_prices.append(price)
            # 保持歷史記錄在設定長度內
            if len(self.historical_prices) > self.max_price_history:
                self.historical_prices = self.historical_prices[-self.max_price_history:]
    
    def get_volatility(self, window=20):
        """獲取當前波動率"""
        return calculate_volatility(self.historical_prices, window)
    
    def start_heartbeat(self):
        """開始心跳檢測線程"""
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_check, daemon=True)
            self.heartbeat_thread.start()
    
    def _heartbeat_check(self):
        """定期檢查WebSocket連接狀態並在需要時重連"""
        while self.running:
            try:
                current_time = time.time()
                time_since_last_heartbeat = current_time - self.last_heartbeat
                
                if time_since_last_heartbeat > self.heartbeat_interval * 2:
                    logger.warning(f"心跳檢測超時 ({time_since_last_heartbeat:.1f}秒)，嘗試重新連接")
                    if self.auto_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
                        # 使用線程進行重連，避免阻塞心跳線程
                        reconnect_thread = threading.Thread(target=self.reconnect, daemon=True)
                        reconnect_thread.start()
                        # 等待重連完成，最多等待5秒
                        reconnect_thread.join(timeout=5)
                    else:
                        logger.warning("已達到最大重連次數或自動重連已禁用，停止重連")
                        self.running = False
                        break
                        
                # 使用小的sleep間隔，以便能更快響應運行狀態的變化
                for _ in range(5):  # 5次1秒的sleep，而不是1次5秒的sleep
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"心跳檢測過程中發生錯誤: {e}")
                time.sleep(5)  # 錯誤發生時等待，避免CPU使用率過高
        
    def connect(self):
        """建立WebSocket連接，增加超時控制"""
        try:
            with self.ws_lock:
                if self.connected or self.ws is not None:
                    logger.info("已存在連接，先關閉舊連接")
                    self.close(manual_close=False)
                
                self.running = True
                self.reconnect_attempts = 0
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
                
                # 使用線程並設置超時
                self.ws_thread = threading.Thread(target=self.ws_run_forever)
                self.ws_thread.daemon = True
                self.ws_thread.start()
                
                # 等待連接建立，增加超時
                wait_start = time.time()
                while not self.connected and time.time() - wait_start < self.connect_timeout:
                    if not self.running:  # 如果運行狀態被取消，提前退出
                        return False
                    time.sleep(0.1)
                
                # 檢查是否連接成功
                if not self.connected:
                    logger.error(f"WebSocket連接超時 (超過 {self.connect_timeout} 秒)")
                    self.close(manual_close=False)  # 關閉未成功的連接
                    return False
                
                # 啟動心跳檢測
                self.start_heartbeat()
                return True
        except Exception as e:
            logger.error(f"建立WebSocket連接時出錯: {e}")
            self.close(manual_close=False)  # 確保資源被清理
            return False
    
    def ws_run_forever(self):
        """運行WebSocket連接，增加了錯誤處理和超時控制"""
        try:
            # 確保在運行前檢查socket狀態
            if hasattr(self.ws, 'sock') and self.ws.sock and self.ws.sock.connected:
                logger.debug("發現socket已經打開，跳過run_forever")
                return

            proxy_type=None
            http_proxy_auth=None
            http_proxy_host=None
            http_proxy_port=None
            if self.proxy and 3<=len(self.proxy.split(":"))<=4:
                arrs=self.proxy.split(":")
                proxy_type = arrs[0]
                arrs[1]=arrs[1][2:] #去掉 //
                if len(arrs)==3:
                    http_proxy_host = arrs[1]
                else:
                    password,http_proxy_host = arrs[2].split("@")
                    http_proxy_auth=(arrs[1],password)
                http_proxy_port = arrs[-1]

            # 添加ping_interval和ping_timeout參數，增加較短的ping_timeout
            self.ws.run_forever(
                ping_interval=self.heartbeat_interval, 
                ping_timeout=5,  # 更短的ping超時，使連接問題更快被檢測到
                http_proxy_auth=http_proxy_auth, 
                http_proxy_host=http_proxy_host, 
                http_proxy_port=http_proxy_port, 
                proxy_type=proxy_type
            )

        except Exception as e:
            logger.error(f"WebSocket運行時出錯: {e}")
        finally:
            with self.ws_lock:
                if self.running and self.auto_reconnect and not self.connected and self.reconnect_attempts < self.max_reconnect_attempts:
                    # 使用線程進行重連，避免在finally塊中阻塞
                    threading.Thread(target=self.reconnect, daemon=True).start()
                else:
                    # 如果不再嘗試重連，確保狀態正確
                    if self.reconnect_attempts >= self.max_reconnect_attempts:
                        logger.warning(f"已達到最大重連嘗試次數 ({self.max_reconnect_attempts})，不再重連")
                    self.connected = False
    
    def on_pong(self, ws, message):
        """處理pong響應"""
        self.last_heartbeat = time.time()
        
    def reconnect(self):
        """完全斷開並重新建立WebSocket連接"""
        # 檢查鎖是否可用，避免死鎖
        if not self.ws_lock.acquire(blocking=False):
            logger.warning("無法獲取鎖進行重連，可能另一個重連操作正在進行")
            return False
        
        try:
            # 檢查是否需要重連
            if not self.running or self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.warning(f"重連次數超過上限 ({self.max_reconnect_attempts})，或運行已停止，停止重連")
                return False

            # 增加重連嘗試次數
            self.reconnect_attempts += 1
            
            # 計算退避延遲，但設置更合理的上限
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), self.max_reconnect_delay)
            
            logger.info(f"嘗試第 {self.reconnect_attempts} 次重連，等待 {delay} 秒...")
            
            # 分段等待以便更快地響應中斷
            segments = 10
            for i in range(segments):
                if not self.running:
                    logger.info("運行已停止，取消重連")
                    return False
                time.sleep(delay / segments)
            
            # 確保完全斷開連接前先標記連接狀態
            self.connected = False
            
            # 完全斷開並清理之前的WebSocket連接
            if self.ws:
                try:
                    # 標記為自動關閉以避免觸發重連
                    if not hasattr(self.ws, '_closed_by_me'):
                        self.ws._closed_by_me = True
                    else:
                        self.ws._closed_by_me = True
                    
                    # 設置不再運行的標誌
                    self.ws.keep_running = False
                    
                    # 關閉WebSocket
                    self.ws.close()
                    
                    # 強制關閉socket
                    if hasattr(self.ws, 'sock') and self.ws.sock:
                        self.ws.sock.close()
                        self.ws.sock = None
                except Exception as e:
                    logger.error(f"關閉之前的WebSocket連接時出錯: {e}")
                finally:
                    # 無論失敗與否，都確保清理
                    self.ws = None
                
            # 確保舊的線程已終止
            if self.ws_thread and self.ws_thread.is_alive():
                try:
                    # 更短的超時時間
                    self.ws_thread.join(timeout=1)
                except Exception as e:
                    logger.error(f"等待舊線程終止時出錯: {e}")
            
            # 重置所有相關狀態
            self.ws_thread = None
            self.subscriptions = []  # 清空訂閲列表，以便重新訂閲
            
            # 檢查是否仍要繼續重連
            if not self.running or not self.auto_reconnect:
                logger.info("運行已停止或自動重連已禁用，放棄重連")
                return False
            
            # 創建全新的WebSocket連接
            logger.info("建立新的WebSocket連接...")
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
            
            # 創建新線程
            self.ws_thread = threading.Thread(target=self.ws_run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # 更新最後心跳時間，避免重連後立即觸發心跳檢測
            self.last_heartbeat = time.time()
            
            # 等待連接建立，但有超時
            wait_start = time.time()
            while not self.connected and time.time() - wait_start < self.connect_timeout:
                if not self.running:  # 如果運行狀態被取消，提前退出
                    return False
                time.sleep(0.1)
            
            # 檢查是否連接成功
            if not self.connected:
                logger.error(f"重連後等待WebSocket連接建立超時")
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    logger.info("將在下一個心跳檢測週期再次嘗試重連")
                return False
                
            logger.info("WebSocket重連成功")
            return True
            
        except Exception as e:
            logger.error(f"重連過程中發生錯誤: {e}")
            return False
        finally:
            # 確保釋放鎖
            self.ws_lock.release()
        
    def on_ping(self, ws, message):
        """處理ping消息"""
        try:
            self.last_heartbeat = time.time()
            if ws and hasattr(ws, 'sock') and ws.sock:
                ws.sock.pong(message)
            else:
                logger.debug("無法迴應ping：WebSocket或sock為None")
        except Exception as e:
            logger.debug(f"迴應ping失敗: {e}")
        
    def on_open(self, ws):
        """WebSocket打開時的處理"""
        logger.info("WebSocket連接已建立")
        self.connected = True
        self.reconnect_attempts = 0  # 重置重連計數
        self.last_heartbeat = time.time()
        
        try:
            # 初始化訂單簿
            orderbook_initialized = self.initialize_orderbook()
            
            # 如果初始化成功，訂閲深度和行情數據
            if orderbook_initialized:
                if "bookTicker" in self.subscriptions or not self.subscriptions:
                    self.subscribe_bookTicker()
                
                if "depth" in self.subscriptions or not self.subscriptions:
                    self.subscribe_depth()
            
            # 重新訂閲私有訂單更新流
            for sub in self.subscriptions:
                if sub.startswith("account."):
                    self.private_subscribe(sub)
        except Exception as e:
            logger.error(f"WebSocket打開後的初始化過程中出錯: {e}")
    
    def subscribe_bookTicker(self):
        """訂閲最優價格"""
        logger.info(f"訂閲 {self.symbol} 的bookTicker...")
        if not self.connected or not self.ws:
            logger.warning("WebSocket未連接，無法訂閲bookTicker")
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
            logger.error(f"訂閲bookTicker失敗: {e}")
            return False
    
    def subscribe_depth(self):
        """訂閲深度信息"""
        logger.info(f"訂閲 {self.symbol} 的深度信息...")
        if not self.connected or not self.ws:
            logger.warning("WebSocket未連接，無法訂閲深度信息")
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
            logger.error(f"訂閲深度信息失敗: {e}")
            return False
    
    def private_subscribe(self, stream):
        """訂閲私有數據流"""
        if not self.connected or not self.ws:
            logger.warning("WebSocket未連接，無法訂閲私有數據流")
            return False
            
        try:
            timestamp = str(int(time.time() * 1000))
            window = DEFAULT_WINDOW
            sign_message = f"instruction=subscribe&timestamp={timestamp}&window={window}"
            signature = create_signature(self.secret_key, sign_message)
            
            if not signature:
                logger.error("簽名創建失敗，無法訂閲私有數據流")
                return False
            
            message = {
                "method": "SUBSCRIBE",
                "params": [stream],
                "signature": [self.api_key, signature, timestamp, window]
            }
            
            self.ws.send(json.dumps(message))
            logger.info(f"已訂閲私有數據流: {stream}")
            if stream not in self.subscriptions:
                self.subscriptions.append(stream)
            return True
        except Exception as e:
            logger.error(f"訂閲私有數據流失敗: {e}")
            return False
    
    def on_message(self, ws, message):
        """處理WebSocket消息"""
        try:
            # 更新心跳時間
            self.last_heartbeat = time.time()
            
            data = json.loads(message)
            
            # 處理ping pong消息
            if isinstance(data, dict) and data.get("ping"):
                pong_message = {"pong": data.get("ping")}
                if self.ws and self.connected:
                    self.ws.send(json.dumps(pong_message))
                    self.last_heartbeat = time.time()
                return
            
            if "stream" in data and "data" in data:
                stream = data["stream"]
                event_data = data["data"]
                
                # 處理bookTicker
                if stream.startswith("bookTicker."):
                    if 'b' in event_data and 'a' in event_data:
                        self.bid_price = float(event_data['b'])
                        self.ask_price = float(event_data['a'])
                        self.last_price = (self.bid_price + self.ask_price) / 2
                        # 記錄歷史價格用於計算波動率
                        self.add_price_to_history(self.last_price)
                
                # 處理depth
                elif stream.startswith("depth."):
                    if 'b' in event_data and 'a' in event_data:
                        self._update_orderbook(event_data)
                
                # 訂單更新數據流
                elif stream.startswith("account.orderUpdate."):
                    self.order_updates.append(event_data)
                    
                if self.on_message_callback:
                    try:
                        self.on_message_callback(stream, event_data)
                    except Exception as callback_error:
                        logger.error(f"執行消息回調時出錯: {callback_error}")
            
        except Exception as e:
            logger.error(f"處理WebSocket消息時出錯: {e}")
    
    def _update_orderbook(self, data):
        """更新訂單簿（優化處理速度）"""
        try:
            # 處理買單更新
            if 'b' in data:
                for bid in data['b']:
                    price = float(bid[0])
                    quantity = float(bid[1])
                    
                    # 使用二分查找來優化插入位置查找
                    if quantity == 0:
                        # 移除價位
                        self.orderbook["bids"] = [b for b in self.orderbook["bids"] if b[0] != price]
                    else:
                        # 先查找是否存在相同價位
                        found = False
                        for i, b in enumerate(self.orderbook["bids"]):
                            if b[0] == price:
                                self.orderbook["bids"][i] = [price, quantity]
                                found = True
                                break
                        
                        # 如果不存在，插入並保持排序
                        if not found:
                            self.orderbook["bids"].append([price, quantity])
                            # 按價格降序排序
                            self.orderbook["bids"] = sorted(self.orderbook["bids"], key=lambda x: x[0], reverse=True)
            
            # 處理賣單更新
            if 'a' in data:
                for ask in data['a']:
                    price = float(ask[0])
                    quantity = float(ask[1])
                    
                    if quantity == 0:
                        # 移除價位
                        self.orderbook["asks"] = [a for a in self.orderbook["asks"] if a[0] != price]
                    else:
                        # 先查找是否存在相同價位
                        found = False
                        for i, a in enumerate(self.orderbook["asks"]):
                            if a[0] == price:
                                self.orderbook["asks"][i] = [price, quantity]
                                found = True
                                break
                        
                        # 如果不存在，插入並保持排序
                        if not found:
                            self.orderbook["asks"].append([price, quantity])
                            # 按價格升序排序
                            self.orderbook["asks"] = sorted(self.orderbook["asks"], key=lambda x: x[0])
        except Exception as e:
            logger.error(f"更新訂單簿時出錯: {e}")
    
    def on_error(self, ws, error):
        """處理WebSocket錯誤"""
        logger.error(f"WebSocket發生錯誤: {error}")
        # 檢查常見的連接錯誤
        if "getaddrinfo failed" in str(error) or "connection refused" in str(error).lower():
            logger.error("檢測到網絡連接問題")
        self.last_heartbeat = 0  # 強制觸發重連
    
    def on_close(self, ws, close_status_code, close_msg):
        """處理WebSocket關閉"""
        previous_connected = self.connected
        self.connected = False
        logger.info(f"WebSocket連接已關閉: {close_msg if close_msg else 'No message'} (狀態碼: {close_status_code if close_status_code else 'None'})")
        
        # 清理當前socket資源
        if hasattr(ws, 'sock') and ws.sock:
            try:
                ws.sock.close()
                ws.sock = None
            except Exception as e:
                logger.debug(f"關閉socket時出錯: {e}")
        
        # 判斷是否為手動關閉
        is_manual_close = close_status_code == 1000 or getattr(ws, '_closed_by_me', False)
        
        if is_manual_close:
            logger.info("WebSocket正常關閉，不進行重連")
        elif previous_connected and self.running and self.auto_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
            logger.info("WebSocket非正常關閉，將自動重連")
            # 使用線程觸發重連，避免在回調中直接重連
            reconnect_thread = threading.Thread(target=self.reconnect, daemon=True)
            reconnect_thread.start()
            # 不等待重連線程完成，避免阻塞on_close回調
        elif self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.warning(f"已達到最大重連嘗試次數 ({self.max_reconnect_attempts})，不再重連")
    
    def close(self, manual_close=True):
        """完全關閉WebSocket連接
        
        Args:
            manual_close: 是否為手動關閉，如果是則禁用自動重連，否則保持重連設置不變
        """
        # 首先設置標誌，防止重複調用
        if hasattr(self, '_closing') and self._closing:
            return
        self._closing = True
        
        try:
            logger.info("關閉WebSocket連接..." + ("(手動關閉)" if manual_close else "(意外斷線)"))
            
            # 只有在手動關閉時才完全停止運行並禁用自動重連
            if manual_close:
                self.running = False
                self.auto_reconnect = False
                
            self.connected = False
            
            # 停止心跳檢測線程
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                try:
                    self.heartbeat_thread.join(timeout=1)
                except Exception:
                    pass
            self.heartbeat_thread = None
            
            # 安全關閉WebSocket
            if self.ws:
                # 標記為主動關閉
                if not hasattr(self.ws, '_closed_by_me'):
                    self.ws._closed_by_me = True
                
                try:
                    # 關閉WebSocket
                    self.ws.keep_running = False  # 設置停止運行標記
                    self.ws.close()
                    
                    # 強制關閉socket
                    if hasattr(self.ws, 'sock') and self.ws.sock:
                        try:
                            self.ws.sock.close()
                        except Exception as sock_error:
                            logger.debug(f"關閉socket時出錯: {sock_error}")
                except Exception as e:
                    logger.error(f"關閉WebSocket時出錯: {e}")
                finally:
                    # 無論如何都確保清理
                    self.ws = None
            
            # 處理線程終止
            if self.ws_thread and self.ws_thread.is_alive():
                if manual_close:
                    # 手動關閉時不等待太久
                    try:
                        self.ws_thread.join(timeout=1)
                    except Exception:
                        pass
                    # 不管是否成功終止，直接設為None
                    self.ws_thread = None
                else:
                    # 非手動關閉時給更多時間終止
                    try:
                        self.ws_thread.join(timeout=2)
                    except Exception:
                        pass
            
            self.ws_thread = None
            
            # 重置訂閲狀態
            self.subscriptions = []
            
            logger.info("WebSocket連接已完全關閉" + (" - 已禁用自動重連" if manual_close else ""))
            logger.info(f"等待保存資料庫並關閉...")
        except Exception as e:
            logger.error(f"關閉WebSocket過程中發生錯誤: {e}")
        finally:
            self._closing = False
    
    def get_current_price(self):
        """獲取當前價格"""
        return self.last_price
    
    def get_bid_ask(self):
        """獲取買賣價"""
        return self.bid_price, self.ask_price
    
    def get_orderbook(self):
        """獲取訂單簿"""
        return self.orderbook

    def is_connected(self):
        """檢查連接狀態，增強健壯性"""
        try:
            if not self.connected or not self.running:
                return False
            if not self.ws:
                return False
            if not hasattr(self.ws, 'sock') or not self.ws.sock:
                return False
            
            # 檢查socket是否連接
            return self.ws.sock.connected
        except Exception as e:
            logger.debug(f"檢查連接狀態時出錯: {e}")
            return False
    
    def get_liquidity_profile(self, depth_percentage=0.01):
        """分析市場流動性特徵"""
        try:
            if not self.orderbook["bids"] or not self.orderbook["asks"]:
                return None
            
            mid_price = (self.bid_price + self.ask_price) / 2 if self.bid_price and self.ask_price else None
            if not mid_price:
                return None
            
            # 計算價格範圍
            min_price = mid_price * (1 - depth_percentage)
            max_price = mid_price * (1 + depth_percentage)
            
            # 分析買賣單流動性
            bid_volume = sum(qty for price, qty in self.orderbook["bids"] if price >= min_price)
            ask_volume = sum(qty for price, qty in self.orderbook["asks"] if price <= max_price)
            
            # 計算買賣比例
            ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
            
            # 買賣壓力差異
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'volume_ratio': ratio,
                'imbalance': imbalance,
                'mid_price': mid_price
            }
        except Exception as e:
            logger.error(f"計算流動性分析時出錯: {e}")
            return None
