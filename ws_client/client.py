"""
WebSocket客户端模塊
"""
import json
import time
import threading
from collections import deque
from typing import Dict, Any, Optional, Callable
import websocket as ws
from config import WS_URL, DEFAULT_WINDOW
from api.auth import create_signature
from api.bp_client import BPClient
from utils.helpers import calculate_volatility
from logger import setup_logger
from urllib.parse import urlparse

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
            proxy:  wss代理 支持格式為 http://user:pass@host:port/ 或者 http://host:port

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
        self.max_reconnect_delay = 1800
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 2
        self.reconnect_cooldown_until = 0.0
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

        # 添加代理參數
        self.proxy = proxy
        
        # 客户端緩存，避免重複創建實例
        self._client_cache = {}
        
        # 添加重連中標誌，避免多次重連
        self.reconnecting = False

        # API 備援方案相關屬性
        self.api_fallback_thread = None
        self.api_fallback_active = False
        self.api_poll_interval = 2  # 秒

        # REST 訂單更新追蹤
        self._fallback_bootstrapped = False
        self._seen_fill_ids = deque(maxlen=200)
        self._seen_fill_id_set = set()
        self._last_fill_timestamp = 0

    def _get_client(self):
        """獲取緩存的客户端實例，避免重複創建"""
        cache_key = "public"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = BPClient({
                "api_key": self.api_key,
                "secret_key": self.secret_key,
            })
        return self._client_cache[cache_key]
    
    def _start_api_fallback(self):
        """啟動使用 REST API 的備援模式"""
        if self.api_fallback_active:
            return

        logger.warning("WebSocket 異常，啟動 API 備援模式以持續獲取數據")
        self.api_fallback_active = True
        self._fallback_bootstrapped = False
        self.api_fallback_thread = threading.Thread(target=self._api_fallback_loop, daemon=True)
        self.api_fallback_thread.start()

    def _stop_api_fallback(self):
        """停止 REST API 備援模式"""
        if not self.api_fallback_active:
            return

        self.api_fallback_active = False

        if self.api_fallback_thread and self.api_fallback_thread.is_alive():
            try:
                self.api_fallback_thread.join(timeout=1)
            except Exception:
                pass

        self.api_fallback_thread = None

    def _api_fallback_loop(self):
        """循環透過 REST API 更新行情資訊"""
        client = self._get_client()

        while self.running and self.api_fallback_active:
            try:
                order_book = client.get_order_book(self.symbol, 50)
                ticker = client.get_ticker(self.symbol)
                fills = client.get_fill_history(self.symbol, limit=100)

                if isinstance(order_book, dict) and "error" not in order_book:
                    bids = order_book.get("bids", [])
                    asks = order_book.get("asks", [])

                    if bids or asks:
                        self.orderbook = {"bids": bids, "asks": asks}

                        if bids:
                            self.bid_price = bids[0][0]
                        if asks:
                            self.ask_price = asks[0][0]

                        if self.on_message_callback:
                            depth_event = {
                                "b": [[str(price), str(quantity)] for price, quantity in bids],
                                "a": [[str(price), str(quantity)] for price, quantity in asks],
                                "source": "api"
                            }
                            self.on_message_callback(f"depth.{self.symbol}", depth_event)

                if isinstance(ticker, dict) and "error" not in ticker:
                    bid_raw = ticker.get("bidPrice") or ticker.get("bestBidPrice")
                    ask_raw = ticker.get("askPrice") or ticker.get("bestAskPrice")
                    last_raw = ticker.get("lastPrice") or ticker.get("price")

                    def _safe_float(value):
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return None

                    bid = _safe_float(bid_raw)
                    ask = _safe_float(ask_raw)
                    last = _safe_float(last_raw)

                    if bid is not None:
                        self.bid_price = bid
                    if ask is not None:
                        self.ask_price = ask
                    if last is not None:
                        self.last_price = last
                        self.add_price_to_history(self.last_price)

                    if self.on_message_callback:
                        ticker_event = {
                            "b": str(self.bid_price) if self.bid_price is not None else None,
                            "a": str(self.ask_price) if self.ask_price is not None else None,
                            "p": str(self.last_price) if self.last_price is not None else None,
                            "source": "api"
                        }
                        self.on_message_callback(f"bookTicker.{self.symbol}", ticker_event)

                # 若仍未獲得價格資訊，嘗試以訂單簿估算
                if self.last_price is None and self.bid_price and self.ask_price:
                    self.last_price = (self.bid_price + self.ask_price) / 2
                    self.add_price_to_history(self.last_price)

                # 透過 REST 補充訂單成交通知
                normalised_fills = self._normalise_fill_history_response(fills)
                if normalised_fills:
                    self._process_rest_fill_updates(normalised_fills)

            except Exception as e:
                logger.error(f"API 備援獲取數據時出錯: {e}")

            # 控制輪詢頻率，避免觸發限速
            time.sleep(self.api_poll_interval)

    def _normalise_fill_history_response(self, response):
        """解析 REST 回傳的成交列表"""
        if isinstance(response, dict) and "error" in response:
            return []

        data = response
        if isinstance(response, dict):
            data = response.get("data", response)
            if isinstance(data, dict):
                for key in ("fills", "items", "rows", "records", "list"):
                    value = data.get(key)
                    if isinstance(value, list):
                        data = value
                        break
        if not isinstance(data, list):
            return []

        fills = []

        def _extract(entry: Dict[str, Any], *keys):
            for key in keys:
                if key in entry and entry[key] not in (None, ""):
                    return entry[key]
            return None

        for entry in data:
            if not isinstance(entry, dict):
                continue

            fill_id = _extract(entry, "id", "fillId", "fill_id", "tradeId", "trade_id", "executionId", "execution_id", "t")
            order_id = _extract(entry, "orderId", "order_id", "i")
            side = _extract(entry, "side", "S")
            price = _extract(entry, "price", "p", "L")
            quantity = _extract(entry, "size", "quantity", "qty", "q", "l")
            fee = _extract(entry, "fee", "commission", "n")
            fee_asset = _extract(entry, "feeAsset", "commissionAsset", "N")
            maker_flag = _extract(entry, "isMaker", "maker", "m")
            timestamp = _extract(entry, "timestamp", "time", "ts", "T")

            try:
                price = float(price) if price is not None else None
            except (TypeError, ValueError):
                price = None

            try:
                quantity = float(quantity) if quantity is not None else None
            except (TypeError, ValueError):
                quantity = None

            try:
                fee = float(fee) if fee is not None else 0.0
            except (TypeError, ValueError):
                fee = 0.0

            try:
                timestamp = int(timestamp)
            except (TypeError, ValueError):
                timestamp = 0

            if not fill_id and timestamp:
                fill_id = str(timestamp)

            fills.append({
                "fill_id": str(fill_id) if fill_id is not None else None,
                "order_id": str(order_id) if order_id is not None else None,
                "side": side,
                "price": price,
                "quantity": quantity,
                "fee": fee,
                "fee_asset": fee_asset,
                "is_maker": bool(maker_flag) if isinstance(maker_flag, bool) else str(maker_flag).lower() in ("true", "1", "yes"),
                "timestamp": timestamp,
            })

        return [fill for fill in fills if fill["order_id"] and fill["quantity"]]

    def _process_rest_fill_updates(self, fills):
        """處理 REST 備援獲取到的成交資訊"""
        fills = sorted(fills, key=lambda item: item.get("timestamp", 0))

        if not self._fallback_bootstrapped:
            for fill in fills:
                self._register_fill_seen(fill)
            self._fallback_bootstrapped = True
            return

        for fill in fills:
            if self._is_new_fill(fill):
                self._register_fill_seen(fill)
                self._emit_rest_order_fill(fill)

    def _is_new_fill(self, fill: Dict[str, Any]) -> bool:
        fill_id = fill.get("fill_id")
        timestamp = fill.get("timestamp", 0)

        if fill_id and fill_id in self._seen_fill_id_set:
            return False

        if timestamp and timestamp <= self._last_fill_timestamp and not fill_id:
            return False

        return True

    def _register_fill_seen(self, fill: Dict[str, Any]):
        fill_id = fill.get("fill_id")
        timestamp = fill.get("timestamp", 0)

        if fill_id:
            if len(self._seen_fill_ids) >= self._seen_fill_ids.maxlen:
                oldest = self._seen_fill_ids.popleft()
                if oldest in self._seen_fill_id_set:
                    self._seen_fill_id_set.remove(oldest)
            self._seen_fill_ids.append(fill_id)
            self._seen_fill_id_set.add(fill_id)

        if timestamp:
            self._last_fill_timestamp = max(self._last_fill_timestamp, timestamp)

    def _emit_rest_order_fill(self, fill: Dict[str, Any]):
        if not self.on_message_callback:
            return

        side = (fill.get("side") or "").lower()
        if side in ("buy", "bid"):
            ws_side = "Bid"
        elif side in ("sell", "ask"):
            ws_side = "Ask"
        else:
            ws_side = side.upper() if side else None

        if ws_side is None:
            return

        event = {
            "e": "orderFill",
            "S": ws_side,
            "l": str(fill.get("quantity", "0")),
            "L": str(fill.get("price", "0")),
            "i": fill.get("order_id"),
            "m": fill.get("is_maker", True),
            "n": str(fill.get("fee", 0.0)),
            "N": fill.get("fee_asset"),
            "t": fill.get("fill_id"),
            "source": "api-fallback",
        }

        logger.info(
            f"REST備援檢測到成交: 訂單 {event['i']} | 方向 {event['S']} | 數量 {event['l']} | 價格 {event['L']}"
        )

        self.on_message_callback(f"account.orderUpdate.{self.symbol}", event)

    def initialize_orderbook(self):
        """通過REST API獲取訂單簿初始快照"""
        try:
            # 使用REST API獲取完整訂單簿
            order_book = self._get_client().get_order_book(self.symbol, 100)  # 增加深度
            if isinstance(order_book, dict) and "error" in order_book:
                logger.error(f"初始化訂單簿失敗: {order_book['error']}")
                return False
            
            # 重置並填充orderbook數據結構
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            self.orderbook = {"bids": bids, "asks": asks}
            
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
            current_time = time.time()
            
            # 檢查是否在冷卻期，如果是則跳過心跳檢測
            if self.reconnect_cooldown_until and current_time < self.reconnect_cooldown_until:
                remaining_cooldown = int(self.reconnect_cooldown_until - current_time)
                logger.debug(f"WebSocket 處於冷卻期，剩餘 {remaining_cooldown} 秒，使用 API 備援模式")
                time.sleep(5)
                continue
            
            time_since_last_heartbeat = current_time - self.last_heartbeat
            
            if time_since_last_heartbeat > self.heartbeat_interval * 2:
                logger.warning(f"心跳檢測超時 ({time_since_last_heartbeat:.1f}秒)，嘗試重新連接")
                # 使用非阻塞方式觸發重連
                threading.Thread(target=self._trigger_reconnect, daemon=True).start()
                
            time.sleep(5)  # 每5秒檢查一次
    
    def _trigger_reconnect(self):
        """非阻塞觸發重連"""
        current_time = time.time()
        if self.reconnect_cooldown_until and current_time < self.reconnect_cooldown_until:
            logger.debug("重連尚在冷卻期，跳過此次請求")
            return

        if self.reconnect_cooldown_until and current_time >= self.reconnect_cooldown_until:
            self.reconnect_attempts = 0
            self.reconnect_cooldown_until = 0.0

        if not self.reconnecting:
            self.reconnect()
        
    def connect(self):
        """建立WebSocket連接"""
        try:
            self.running = True
            self.reconnect_attempts = 0
            self.reconnect_cooldown_until = 0.0
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
            
            # 啟動心跳檢測
            self.start_heartbeat()
        except Exception as e:
            logger.error(f"初始化WebSocket連接時出錯: {e}")
            self._start_api_fallback()
    
    def ws_run_forever(self):
        """WebSocket運行循環 - 修復版本"""
        try:
            if hasattr(self.ws, 'sock') and self.ws.sock and self.ws.sock.connected:
                logger.debug("發現socket已經打開，跳過run_forever")
                return

            http_proxy_host = None
            http_proxy_port = None
            http_proxy_auth = None
            proxy_type = None

            if self.proxy:
                # 使用標準庫 urlparse 進行可靠的解析
                parsed_proxy = urlparse(self.proxy)
                
                # 建立安全的日誌訊息，隱藏密碼
                safe_proxy_display = f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
                if parsed_proxy.username:
                    safe_proxy_display = f"{parsed_proxy.scheme}://{parsed_proxy.username}:***@{parsed_proxy.hostname}:{parsed_proxy.port}"

                logger.info(f"正在使用 WebSocket 代理: {safe_proxy_display}")

                http_proxy_host = parsed_proxy.hostname
                http_proxy_port = parsed_proxy.port
                if parsed_proxy.username and parsed_proxy.password:
                    http_proxy_auth = (parsed_proxy.username, parsed_proxy.password)
                # 支援 http, socks4, socks5 代理
                proxy_type = parsed_proxy.scheme if parsed_proxy.scheme in ['http', 'socks4', 'socks5'] else 'http'
                
            # 將解析後的參數傳遞給 run_forever
            self.ws.run_forever(
                ping_interval=self.heartbeat_interval,
                ping_timeout=10,
                http_proxy_host=http_proxy_host,
                http_proxy_port=http_proxy_port,
                http_proxy_auth=http_proxy_auth,
                proxy_type=proxy_type
            )

        except Exception as e:
            logger.error(f"WebSocket運行時出錯: {e}")
        finally:
            logger.debug("WebSocket run_forever 執行結束")
    
    def on_pong(self, ws, message):
        """處理pong響應"""
        self.last_heartbeat = time.time()
        
    def reconnect(self):
        """完全斷開並重新建立WebSocket連接 - 修復版本"""
        # 防止多次重連
        if self.reconnecting:
            logger.debug("重連已在進行中，跳過此次重連請求")
            return False

        current_time = time.time()
        if self.reconnect_cooldown_until and current_time < self.reconnect_cooldown_until:
            logger.debug("重連尚未解除冷卻，跳過此次重連請求")
            return False

        with self.ws_lock:
            if not self.running or self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.warning(f"重連次數超過上限 ({self.max_reconnect_attempts})，暫停自動重連")
                cooldown_seconds = max(self.max_reconnect_delay, 60)
                self.reconnect_cooldown_until = time.time() + cooldown_seconds
                self.last_heartbeat = time.time()
                logger.warning(f"已啟動 {cooldown_seconds} 秒冷卻，將繼續使用備援模式")
                self._start_api_fallback()
                return False

            self.reconnecting = True
            self.reconnect_attempts += 1
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), self.max_reconnect_delay)
            
            logger.info(f"嘗試第 {self.reconnect_attempts} 次重連，等待 {delay} 秒...")
            time.sleep(delay)
            
            # 確保完全斷開連接前先標記連接狀態
            self.connected = False
            
            # 優雅關閉現有連接
            self._force_close_connection()
            
            # 重置所有相關狀態
            self.ws_thread = None
            self.subscriptions = []  # 清空訂閲列表，以便重新訂閴
            
            try:
                # 創建全新的WebSocket連接
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
                self.reconnect_cooldown_until = 0.0

                logger.info(f"第 {self.reconnect_attempts} 次重連已啟動")
                
                self.reconnecting = False
                return True
                
            except Exception as e:
                logger.error(f"重連過程中發生錯誤: {e}")
                self.reconnecting = False
                self._start_api_fallback()
                return False
    
    def _force_close_connection(self):
        """強制關閉現有連接"""
        try:
            # 完全斷開並清理之前的WebSocket連接
            if self.ws:
                try:
                    # 顯式設置內部標記表明這是用户主動關閉
                    if hasattr(self.ws, '_closed_by_me'):
                        self.ws._closed_by_me = True
                    
                    # 關閉WebSocket
                    self.ws.close()
                    if hasattr(self.ws, 'keep_running'):
                        self.ws.keep_running = False
                    
                    # 強制關閉socket
                    if hasattr(self.ws, 'sock') and self.ws.sock:
                        try:
                            self.ws.sock.close()
                            self.ws.sock = None
                        except:
                            pass
                except Exception as e:
                    logger.debug(f"關閉WebSocket時的預期錯誤: {e}")
                
                self.ws = None
                
            # 處理舊線程 - 使用較短的超時時間
            if self.ws_thread and self.ws_thread.is_alive():
                try:
                    # 不要無限等待線程結束
                    self.ws_thread.join(timeout=1.0)
                    if self.ws_thread.is_alive():
                        logger.warning("舊線程未能在超時時間內結束，但繼續重連過程")
                except Exception as e:
                    logger.debug(f"等待舊線程終止時出錯: {e}")
            
            # 給系統少量時間清理資源
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"強制關閉連接時出錯: {e}")
        
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
        self.reconnect_attempts = 0
        self.reconnecting = False
        self.last_heartbeat = time.time()
        
        # 停止 API 備援模式
        self._stop_api_fallback()

        # 添加短暫延遲確保連接穩定
        time.sleep(0.5)
        
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
            logger.error(f"訂閴bookTicker失敗: {e}")
            return False
    
    def subscribe_depth(self):
        """訂閴深度信息"""
        logger.info(f"訂閴 {self.symbol} 的深度信息...")
        if not self.connected or not self.ws:
            logger.warning("WebSocket未連接，無法訂閴深度信息")
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
            logger.error(f"訂閴深度信息失敗: {e}")
            return False
    
    def private_subscribe(self, stream):
        """訂閴私有數據流"""
        if not self.connected or not self.ws:
            logger.warning("WebSocket未連接，無法訂閴私有數據流")
            return False
            
        try:
            timestamp = str(int(time.time() * 1000))
            window = DEFAULT_WINDOW
            sign_message = f"instruction=subscribe&timestamp={timestamp}&window={window}"
            signature = create_signature(self.secret_key, sign_message)
            
            if not signature:
                logger.error("簽名創建失敗，無法訂閴私有數據流")
                return False
            
            message = {
                "method": "SUBSCRIBE",
                "params": [stream],
                "signature": [self.api_key, signature, timestamp, window]
            }
            
            self.ws.send(json.dumps(message))
            logger.info(f"已訂閴私有數據流: {stream}")
            if stream not in self.subscriptions:
                self.subscriptions.append(stream)
            return True
        except Exception as e:
            logger.error(f"訂閴私有數據流失敗: {e}")
            return False
    
    def on_message(self, ws, message):
        """處理WebSocket消息"""
        try:
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
                    self.on_message_callback(stream, event_data)
            
        except Exception as e:
            logger.error(f"處理WebSocket消息時出錯: {e}")
    
    def _update_orderbook(self, data):
        """更新訂單簿（優化處理速度）"""
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
    
    def on_error(self, ws, error):
        """處理WebSocket錯誤"""
        logger.error(f"WebSocket發生錯誤: {error}")
        self.last_heartbeat = 0  # 強制觸發重連

        self._start_api_fallback()
    
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
        
        if close_status_code == 1000 or getattr(ws, '_closed_by_me', False):
            logger.info("WebSocket正常關閉，不進行重連")
            self._start_api_fallback()
        elif previous_connected and self.running and self.auto_reconnect and not self.reconnecting:
            logger.info("WebSocket非正常關閉，將自動重連")
            # 使用線程觸發重連，避免在回調中直接重連
            threading.Thread(target=self._trigger_reconnect, daemon=True).start()
            self._start_api_fallback()
    
    def close(self):
        """完全關閉WebSocket連接"""
        logger.info("主動關閉WebSocket連接...")
        self.running = False
        self.connected = False
        self.reconnecting = False
        self.reconnect_cooldown_until = 0.0
        
        # 停止心跳檢測線程
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            try:
                self.heartbeat_thread.join(timeout=1)
            except Exception:
                pass
        self.heartbeat_thread = None
        
        # 強制關閉連接
        self._force_close_connection()
        
        # 重置訂閴狀態
        self.subscriptions = []

        # 確保停止 API 備援
        self._stop_api_fallback()
        
        logger.info("WebSocket連接已完全關閉")
    
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
        """檢查連接狀態"""
        if not self.connected:
            return False
        if not self.ws:
            return False
        if not hasattr(self.ws, 'sock') or not self.ws.sock:
            return False
        
        # 檢查socket是否連接
        try:
            return self.ws.sock.connected
        except:
            return False
    
    def get_liquidity_profile(self, depth_percentage=0.01):
        """分析市場流動性特徵"""
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
    
    def check_and_reconnect_if_needed(self):
        """檢查連接狀態並在需要時重連 - 供外部調用"""
        if self.reconnect_cooldown_until and time.time() < self.reconnect_cooldown_until:
            return self.is_connected()

        if not self.is_connected() and not self.reconnecting:
            logger.info("外部檢查發現連接斷開，觸發重連...")
            threading.Thread(target=self._trigger_reconnect, daemon=True).start()
            self._start_api_fallback()

        return self.is_connected()

