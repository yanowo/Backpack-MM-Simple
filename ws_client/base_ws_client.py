"""
抽象 WebSocket 客戶端基類
支持多交易所的 WebSocket 連接、訂閱和消息處理
"""
from __future__ import annotations

import json
import time
import threading
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import websocket as ws

from logger import setup_logger
from utils.helpers import calculate_volatility


# ==================== 標準化數據結構 ====================

@dataclass
class WSTickerData:
    """標準化行情數據"""
    symbol: str
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    last_price: Optional[Decimal] = None
    timestamp: Optional[int] = None
    source: str = "ws"


@dataclass
class WSOrderBookData:
    """標準化訂單簿數據"""
    symbol: str
    bids: List[Tuple[Decimal, Decimal]] = field(default_factory=list)  # [(price, quantity), ...]
    asks: List[Tuple[Decimal, Decimal]] = field(default_factory=list)
    timestamp: Optional[int] = None
    source: str = "ws"


@dataclass
class WSOrderUpdateData:
    """標準化訂單更新數據"""
    symbol: str
    order_id: str
    side: str  # "BUY" / "SELL"
    order_type: str  # "LIMIT", "MARKET"
    status: str  # "NEW", "FILLED", "PARTIALLY_FILLED", "CANCELLED"
    price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    filled_quantity: Optional[Decimal] = None
    remaining_quantity: Optional[Decimal] = None
    timestamp: Optional[int] = None
    source: str = "ws"


@dataclass
class WSFillData:
    """標準化成交數據"""
    symbol: str
    fill_id: str
    order_id: str
    side: str  # "BUY" / "SELL"
    price: Decimal
    quantity: Decimal
    fee: Decimal = Decimal("0")
    fee_asset: Optional[str] = None
    is_maker: bool = True
    timestamp: Optional[int] = None
    source: str = "ws"


@dataclass
class WSConnectionConfig:
    """WebSocket 連接配置"""
    ws_url: str
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    proxy: Optional[str] = None
    auto_reconnect: bool = True
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 1800.0
    max_reconnect_attempts: int = 2
    heartbeat_interval: int = 30
    ping_interval: int = 30
    ping_timeout: int = 10


# ==================== 抽象 WebSocket 基類 ====================

class BaseWebSocketClient(ABC):
    """
    抽象 WebSocket 客戶端基類
    
    所有交易所的 WebSocket 客戶端都應繼承此類並實現抽象方法
    """
    
    def __init__(
        self,
        config: WSConnectionConfig,
        symbol: str,
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
    ):
        """
        初始化 WebSocket 客戶端
        
        Args:
            config: WebSocket 連接配置
            symbol: 交易對符號
            on_message_callback: 消息回調函數，接收 (stream_name, data)
        """
        self.config = config
        self.symbol = symbol
        self.on_message_callback = on_message_callback
        
        # WebSocket 連接相關
        self.ws: Optional[ws.WebSocketApp] = None
        self.connected = False
        self.running = False
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_lock = threading.Lock()
        
        # 市場數據
        self.last_price: Optional[float] = None
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.orderbook: Dict[str, List] = {"bids": [], "asks": []}
        self.order_updates: List[Dict] = []
        
        # 歷史價格（用於波動率計算）
        self.historical_prices: List[float] = []
        self.max_price_history: int = 100
        
        # 重連相關
        self.reconnect_delay = config.reconnect_delay
        self.max_reconnect_delay = config.max_reconnect_delay
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.max_reconnect_attempts
        self.reconnect_cooldown_until: float = 0.0
        self.reconnecting = False
        
        # 訂閱管理
        self.subscriptions: List[str] = []
        
        # 心跳檢測
        self.last_heartbeat = time.time()
        self.heartbeat_interval = config.heartbeat_interval
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # 代理設置
        self.proxy = config.proxy
        if self.proxy is None:
            self.proxy = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
        
        # API 備援模式
        self.api_fallback_thread: Optional[threading.Thread] = None
        self.api_fallback_active = False
        self.api_poll_interval = 1.0
        
        # 成交記錄追蹤（用於 REST 備援）
        self._fallback_bootstrapped = False
        self._seen_fill_ids: deque = deque(maxlen=200)
        self._seen_fill_id_set: Set[str] = set()
        self._last_fill_timestamp: int = 0
        
        # Logger
        self._logger = setup_logger(f"{self.get_exchange_name().lower()}_ws")
    
    # ==================== 抽象方法（子類必須實現）====================
    
    @abstractmethod
    def get_exchange_name(self) -> str:
        """
        返回交易所名稱
        
        Returns:
            交易所名稱字符串（如 "Backpack", "Binance"）
        """
        ...
    
    @abstractmethod
    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        """
        創建認證消息
        
        Returns:
            認證消息字典，如果不需要認證則返回 None
        """
        ...
    
    @abstractmethod
    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        """
        創建訂閱消息
        
        Args:
            channel: 頻道名稱
            is_private: 是否為私有頻道
            
        Returns:
            訂閱消息字典
        """
        ...
    
    @abstractmethod
    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        """
        創建取消訂閱消息
        
        Args:
            channel: 頻道名稱
            
        Returns:
            取消訂閱消息字典
        """
        ...
    
    @abstractmethod
    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        """
        解析原始 WebSocket 消息
        
        Args:
            raw_message: 原始消息字符串
            
        Returns:
            (stream_name, parsed_data) 元組，如果無法解析則返回 None
        """
        ...
    
    @abstractmethod
    def _get_ticker_channel(self) -> str:
        """
        獲取行情頻道名稱
        
        Returns:
            行情頻道名稱
        """
        ...
    
    @abstractmethod
    def _get_depth_channel(self) -> str:
        """
        獲取深度頻道名稱
        
        Returns:
            深度頻道名稱
        """
        ...
    
    @abstractmethod
    def _get_order_update_channel(self) -> str:
        """
        獲取訂單更新頻道名稱
        
        Returns:
            訂單更新頻道名稱
        """
        ...
    
    @abstractmethod
    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        """
        處理行情消息
        
        Args:
            data: 原始行情數據
            
        Returns:
            標準化的行情數據
        """
        ...
    
    @abstractmethod
    def _handle_depth_message(self, data: Any) -> Optional[WSOrderBookData]:
        """
        處理深度消息
        
        Args:
            data: 原始深度數據
            
        Returns:
            標準化的訂單簿數據
        """
        ...
    
    @abstractmethod
    def _handle_order_update_message(self, data: Any) -> Optional[WSOrderUpdateData]:
        """
        處理訂單更新消息
        
        Args:
            data: 原始訂單數據
            
        Returns:
            標準化的訂單更新數據
        """
        ...
    
    @abstractmethod
    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        """
        處理成交消息
        
        Args:
            data: 原始成交數據
            
        Returns:
            標準化的成交數據
        """
        ...
    
    @abstractmethod
    def _get_rest_client(self) -> Any:
        """
        獲取對應交易所的 REST API 客戶端
        
        Returns:
            REST API 客戶端實例
        """
        ...
    
    # ==================== 可選重寫方法 ====================
    
    def _handle_ping(self, message: Any) -> Optional[Dict[str, Any]]:
        """
        處理 ping 消息並返回 pong 響應
        
        Args:
            message: ping 消息
            
        Returns:
            pong 響應字典，如果不需要則返回 None
        """
        return None
    
    def _on_auth_success(self):
        """認證成功後的回調"""
        self._logger.info(f"{self.get_exchange_name()} WebSocket 認證成功")
    
    def _on_auth_failure(self, error: str):
        """認證失敗後的回調"""
        self._logger.error(f"{self.get_exchange_name()} WebSocket 認證失敗: {error}")
    
    # ==================== 連接管理 ====================
    
    def connect(self):
        """建立 WebSocket 連接"""
        try:
            self.running = True
            self.reconnect_attempts = 0
            self.reconnect_cooldown_until = 0.0
            self.reconnecting = False
            
            ws.enableTrace(False)
            self.ws = ws.WebSocketApp(
                self.config.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_ping=self._on_ping,
                on_pong=self._on_pong
            )
            
            self.ws_thread = threading.Thread(target=self._ws_run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # 啟動心跳檢測
            self._start_heartbeat()
            
        except Exception as e:
            self._logger.error(f"初始化 WebSocket 連接時出錯: {e}")
            self._start_api_fallback()
    
    def _ws_run_forever(self):
        """WebSocket 運行循環"""
        try:
            if hasattr(self.ws, 'sock') and self.ws.sock and self.ws.sock.connected:
                self._logger.debug("發現 socket 已經打開，跳過 run_forever")
                return
            
            # 解析代理設置
            http_proxy_host = None
            http_proxy_port = None
            http_proxy_auth = None
            proxy_type = None
            
            if self.proxy:
                parsed_proxy = urlparse(self.proxy)
                
                safe_proxy_display = f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
                if parsed_proxy.username:
                    safe_proxy_display = f"{parsed_proxy.scheme}://{parsed_proxy.username}:***@{parsed_proxy.hostname}:{parsed_proxy.port}"
                
                self._logger.info(f"正在使用 WebSocket 代理: {safe_proxy_display}")
                
                http_proxy_host = parsed_proxy.hostname
                http_proxy_port = parsed_proxy.port
                if parsed_proxy.username and parsed_proxy.password:
                    http_proxy_auth = (parsed_proxy.username, parsed_proxy.password)
                proxy_type = parsed_proxy.scheme if parsed_proxy.scheme in ['http', 'socks4', 'socks5'] else 'http'
            
            self.ws.run_forever(
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                http_proxy_host=http_proxy_host,
                http_proxy_port=http_proxy_port,
                http_proxy_auth=http_proxy_auth,
                proxy_type=proxy_type
            )
            
        except Exception as e:
            self._logger.error(f"WebSocket 運行時出錯: {e}")
        finally:
            self._logger.debug("WebSocket run_forever 執行結束")
    
    def reconnect(self) -> bool:
        """完全斷開並重新建立 WebSocket 連接"""
        if self.reconnecting:
            self._logger.debug("重連已在進行中，跳過此次重連請求")
            return False
        
        current_time = time.time()
        if self.reconnect_cooldown_until and current_time < self.reconnect_cooldown_until:
            self._logger.debug("重連尚未解除冷卻，跳過此次重連請求")
            return False
        
        with self.ws_lock:
            if not self.running or self.reconnect_attempts >= self.max_reconnect_attempts:
                self._logger.warning(f"重連次數超過上限 ({self.max_reconnect_attempts})，暫停自動重連")
                cooldown_seconds = max(self.max_reconnect_delay, 60)
                self.reconnect_cooldown_until = time.time() + cooldown_seconds
                self.last_heartbeat = time.time()
                self._logger.warning(f"已啟動 {cooldown_seconds} 秒冷卻，將繼續使用備援模式")
                self._start_api_fallback()
                return False
            
            self.reconnecting = True
            self.reconnect_attempts += 1
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), self.max_reconnect_delay)
            
            self._logger.info(f"嘗試第 {self.reconnect_attempts} 次重連，等待 {delay} 秒...")
            time.sleep(delay)
            
            self.connected = False
            self._force_close_connection()
            
            self.ws_thread = None
            self.subscriptions = []
            
            try:
                ws.enableTrace(False)
                self.ws = ws.WebSocketApp(
                    self.config.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_ping=self._on_ping,
                    on_pong=self._on_pong
                )
                
                self.ws_thread = threading.Thread(target=self._ws_run_forever)
                self.ws_thread.daemon = True
                self.ws_thread.start()
                
                self.last_heartbeat = time.time()
                self.reconnect_cooldown_until = 0.0
                
                self._logger.info(f"第 {self.reconnect_attempts} 次重連已啟動")
                
                self.reconnecting = False
                return True
                
            except Exception as e:
                self._logger.error(f"重連過程中發生錯誤: {e}")
                self.reconnecting = False
                self._start_api_fallback()
                return False
    
    def _force_close_connection(self):
        """強制關閉現有連接"""
        try:
            if self.ws:
                try:
                    if hasattr(self.ws, '_closed_by_me'):
                        self.ws._closed_by_me = True
                    
                    self.ws.close()
                    if hasattr(self.ws, 'keep_running'):
                        self.ws.keep_running = False
                    
                    if hasattr(self.ws, 'sock') and self.ws.sock:
                        try:
                            self.ws.sock.close()
                            self.ws.sock = None
                        except:
                            pass
                except Exception as e:
                    self._logger.debug(f"關閉 WebSocket 時的預期錯誤: {e}")
                
                self.ws = None
            
            if self.ws_thread and self.ws_thread.is_alive():
                try:
                    self.ws_thread.join(timeout=1.0)
                    if self.ws_thread.is_alive():
                        self._logger.warning("舊線程未能在超時時間內結束，但繼續重連過程")
                except Exception as e:
                    self._logger.debug(f"等待舊線程終止時出錯: {e}")
            
            time.sleep(0.5)
            
        except Exception as e:
            self._logger.error(f"強制關閉連接時出錯: {e}")
    
    def close(self):
        """完全關閉 WebSocket 連接"""
        self._logger.info("主動關閉 WebSocket 連接...")
        self.running = False
        self.connected = False
        self.reconnecting = False
        self.reconnect_cooldown_until = 0.0
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            try:
                self.heartbeat_thread.join(timeout=1)
            except Exception:
                pass
        self.heartbeat_thread = None
        
        self._force_close_connection()
        self.subscriptions = []
        self._stop_api_fallback()
        
        self._logger.info("WebSocket 連接已完全關閉")
    
    # ==================== 心跳檢測 ====================
    
    def _start_heartbeat(self):
        """開始心跳檢測線程"""
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_check, daemon=True)
            self.heartbeat_thread.start()
    
    def _heartbeat_check(self):
        """定期檢查 WebSocket 連接狀態"""
        while self.running:
            current_time = time.time()
            
            if self.reconnect_cooldown_until and current_time < self.reconnect_cooldown_until:
                remaining_cooldown = int(self.reconnect_cooldown_until - current_time)
                self._logger.debug(f"WebSocket 處於冷卻期，剩餘 {remaining_cooldown} 秒")
                time.sleep(5)
                continue
            
            time_since_last_heartbeat = current_time - self.last_heartbeat
            
            if time_since_last_heartbeat > self.heartbeat_interval * 2:
                self._logger.warning(f"心跳檢測超時 ({time_since_last_heartbeat:.1f}秒)，嘗試重新連接")
                threading.Thread(target=self._trigger_reconnect, daemon=True).start()
            
            time.sleep(5)
    
    def _trigger_reconnect(self):
        """非阻塞觸發重連"""
        current_time = time.time()
        if self.reconnect_cooldown_until and current_time < self.reconnect_cooldown_until:
            self._logger.debug("重連尚在冷卻期，跳過此次請求")
            return
        
        if self.reconnect_cooldown_until and current_time >= self.reconnect_cooldown_until:
            self.reconnect_attempts = 0
            self.reconnect_cooldown_until = 0.0
        
        if not self.reconnecting:
            self.reconnect()
    
    # ==================== WebSocket 回調 ====================
    
    def _on_open(self, ws_app):
        """WebSocket 打開時的處理"""
        self._logger.info(f"{self.get_exchange_name()} WebSocket 連接已建立")
        self.connected = True
        self.reconnect_attempts = 0
        self.reconnecting = False
        self.last_heartbeat = time.time()
        
        self._stop_api_fallback()
        
        time.sleep(0.5)
        
        # 初始化訂單簿
        self.initialize_orderbook()
        
        # 訂閱默認頻道
        self.subscribe_ticker()
        self.subscribe_depth()
    
    def _on_message(self, ws_app, message: str):
        """處理 WebSocket 消息"""
        try:
            # 檢查是否為 ping 消息
            try:
                data = json.loads(message)
                pong_response = self._handle_ping(data)
                if pong_response:
                    if self.ws and self.connected:
                        self.ws.send(json.dumps(pong_response))
                        self.last_heartbeat = time.time()
                    return
            except json.JSONDecodeError:
                pass
            
            # 解析消息
            parsed = self._parse_message(message)
            if not parsed:
                return
            
            stream, data = parsed
            
            # 根據頻道類型處理消息
            ticker_channel = self._get_ticker_channel()
            depth_channel = self._get_depth_channel()
            order_channel = self._get_order_update_channel()
            
            if ticker_channel and stream.startswith(ticker_channel.split('.')[0]):
                ticker_data = self._handle_ticker_message(data)
                if ticker_data:
                    if ticker_data.bid_price:
                        self.bid_price = float(ticker_data.bid_price)
                    if ticker_data.ask_price:
                        self.ask_price = float(ticker_data.ask_price)
                    if ticker_data.last_price:
                        self.last_price = float(ticker_data.last_price)
                        self.add_price_to_history(self.last_price)
                    elif self.bid_price and self.ask_price:
                        self.last_price = (self.bid_price + self.ask_price) / 2
                        self.add_price_to_history(self.last_price)
            
            elif depth_channel and stream.startswith(depth_channel.split('.')[0]):
                depth_data = self._handle_depth_message(data)
                if depth_data:
                    self._update_orderbook_from_data(depth_data)
            
            elif order_channel and stream.startswith(order_channel.split('.')[0]):
                # 處理訂單更新
                order_data = self._handle_order_update_message(data)
                if order_data:
                    self.order_updates.append(data)
                
                # 也嘗試處理成交
                fill_data = self._handle_fill_message(data)
                if fill_data:
                    pass  # 成交數據可以通過回調傳遞
            
            # 調用用戶回調
            if self.on_message_callback:
                self.on_message_callback(stream, data)
                
        except Exception as e:
            self._logger.error(f"處理 WebSocket 消息時出錯: {e}")
    
    def _on_error(self, ws_app, error):
        """處理 WebSocket 錯誤"""
        self._logger.error(f"WebSocket 發生錯誤: {error}")
        self.last_heartbeat = 0
        self._start_api_fallback()
    
    def _on_close(self, ws_app, close_status_code, close_msg):
        """處理 WebSocket 關閉"""
        previous_connected = self.connected
        self.connected = False
        self._logger.info(
            f"WebSocket 連接已關閉: {close_msg if close_msg else 'No message'} "
            f"(狀態碼: {close_status_code if close_status_code else 'None'})"
        )
        
        if hasattr(ws_app, 'sock') and ws_app.sock:
            try:
                ws_app.sock.close()
                ws_app.sock = None
            except Exception as e:
                self._logger.debug(f"關閉 socket 時出錯: {e}")
        
        if close_status_code == 1000 or getattr(ws_app, '_closed_by_me', False):
            self._logger.info("WebSocket 正常關閉，不進行重連")
            self._start_api_fallback()
        elif previous_connected and self.running and self.config.auto_reconnect and not self.reconnecting:
            self._logger.info("WebSocket 非正常關閉，將自動重連")
            threading.Thread(target=self._trigger_reconnect, daemon=True).start()
            self._start_api_fallback()
    
    def _on_ping(self, ws_app, message):
        """處理 ping 消息"""
        try:
            self.last_heartbeat = time.time()
            if ws_app and hasattr(ws_app, 'sock') and ws_app.sock:
                ws_app.sock.pong(message)
            else:
                self._logger.debug("無法回應 ping：WebSocket 或 sock 為 None")
        except Exception as e:
            self._logger.debug(f"回應 ping 失敗: {e}")
    
    def _on_pong(self, ws_app, message):
        """處理 pong 響應"""
        self.last_heartbeat = time.time()
    
    # ==================== 訂閱管理 ====================
    
    def subscribe_ticker(self) -> bool:
        """訂閱行情數據"""
        channel = self._get_ticker_channel()
        return self._subscribe(channel, is_private=False)
    
    def subscribe_depth(self) -> bool:
        """訂閱深度數據"""
        channel = self._get_depth_channel()
        return self._subscribe(channel, is_private=False)
    
    def subscribe_order_updates(self) -> bool:
        """訂閱訂單更新"""
        channel = self._get_order_update_channel()
        return self._subscribe(channel, is_private=True)
    
    def _subscribe(self, channel: str, is_private: bool = False) -> bool:
        """
        訂閱頻道
        
        Args:
            channel: 頻道名稱
            is_private: 是否為私有頻道
            
        Returns:
            是否成功發送訂閱請求
        """
        if not self.connected or not self.ws:
            self._logger.warning(f"WebSocket 未連接，無法訂閱 {channel}")
            return False
        
        try:
            message = self._create_subscribe_message(channel, is_private)
            self.ws.send(json.dumps(message))
            
            if channel not in self.subscriptions:
                self.subscriptions.append(channel)
            
            self._logger.info(f"已訂閱頻道: {channel}")
            return True
            
        except Exception as e:
            self._logger.error(f"訂閱 {channel} 失敗: {e}")
            return False
    
    def unsubscribe(self, channel: str) -> bool:
        """
        取消訂閱頻道
        
        Args:
            channel: 頻道名稱
            
        Returns:
            是否成功發送取消訂閱請求
        """
        if not self.connected or not self.ws:
            self._logger.warning(f"WebSocket 未連接，無法取消訂閱 {channel}")
            return False
        
        try:
            message = self._create_unsubscribe_message(channel)
            self.ws.send(json.dumps(message))
            
            if channel in self.subscriptions:
                self.subscriptions.remove(channel)
            
            self._logger.info(f"已取消訂閱頻道: {channel}")
            return True
            
        except Exception as e:
            self._logger.error(f"取消訂閱 {channel} 失敗: {e}")
            return False
    
    # ==================== 數據管理 ====================
    
    def initialize_orderbook(self) -> bool:
        """通過 REST API 獲取訂單簿初始快照"""
        try:
            client = self._get_rest_client()
            if not client:
                return False
            
            order_book_response = client.get_order_book(self.symbol, 100)
            if not order_book_response.success:
                self._logger.error(f"初始化訂單簿失敗: {order_book_response.error_message}")
                return False
            
            # 支援 OrderBookInfo dataclass 或 dict
            order_book = order_book_response.data
            if hasattr(order_book, 'bids'):
                # OrderBookInfo dataclass - 需要轉換 OrderBookLevel 為列表格式
                bids = [[float(level.price), float(level.quantity)] for level in order_book.bids]
                asks = [[float(level.price), float(level.quantity)] for level in order_book.asks]
            elif hasattr(order_book, 'raw') and order_book.raw:
                bids = order_book.raw.get("bids", [])
                asks = order_book.raw.get("asks", [])
            elif isinstance(order_book, dict):
                bids = order_book.get("bids", [])
                asks = order_book.get("asks", [])
            else:
                bids = []
                asks = []
            self.orderbook = {"bids": bids, "asks": asks}
            
            self._logger.info(
                f"訂單簿初始化成功: {len(self.orderbook['bids'])} 個買單, "
                f"{len(self.orderbook['asks'])} 個賣單"
            )
            
            if self.orderbook["bids"]:
                self.bid_price = self.orderbook["bids"][0][0]
            if self.orderbook["asks"]:
                self.ask_price = self.orderbook["asks"][0][0]
            if self.bid_price and self.ask_price:
                self.last_price = (self.bid_price + self.ask_price) / 2
                self.add_price_to_history(self.last_price)
            
            return True
            
        except Exception as e:
            self._logger.error(f"初始化訂單簿時出錯: {e}")
            return False
    
    def _update_orderbook_from_data(self, data: WSOrderBookData):
        """從標準化數據更新訂單簿"""
        for price, quantity in data.bids:
            price_f = float(price)
            quantity_f = float(quantity)
            
            if quantity_f == 0:
                self.orderbook["bids"] = [b for b in self.orderbook["bids"] if b[0] != price_f]
            else:
                found = False
                for i, b in enumerate(self.orderbook["bids"]):
                    if b[0] == price_f:
                        self.orderbook["bids"][i] = [price_f, quantity_f]
                        found = True
                        break
                if not found:
                    self.orderbook["bids"].append([price_f, quantity_f])
                    self.orderbook["bids"] = sorted(self.orderbook["bids"], key=lambda x: x[0], reverse=True)
        
        for price, quantity in data.asks:
            price_f = float(price)
            quantity_f = float(quantity)
            
            if quantity_f == 0:
                self.orderbook["asks"] = [a for a in self.orderbook["asks"] if a[0] != price_f]
            else:
                found = False
                for i, a in enumerate(self.orderbook["asks"]):
                    if a[0] == price_f:
                        self.orderbook["asks"][i] = [price_f, quantity_f]
                        found = True
                        break
                if not found:
                    self.orderbook["asks"].append([price_f, quantity_f])
                    self.orderbook["asks"] = sorted(self.orderbook["asks"], key=lambda x: x[0])
    
    def add_price_to_history(self, price: float):
        """添加價格到歷史記錄"""
        if price:
            self.historical_prices.append(price)
            if len(self.historical_prices) > self.max_price_history:
                self.historical_prices = self.historical_prices[-self.max_price_history:]
    
    # ==================== API 備援模式 ====================
    
    def _start_api_fallback(self):
        """啟動 REST API 備援模式"""
        if self.api_fallback_active:
            return
        
        self._logger.warning("WebSocket 異常，啟動 API 備援模式")
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
        """REST API 備援循環"""
        client = self._get_rest_client()
        if not client:
            self._logger.error("無法獲取 REST 客戶端，API 備援模式失敗")
            self.api_fallback_active = False
            return
        
        while self.running and self.api_fallback_active:
            try:
                # 獲取訂單簿
                order_book = client.get_order_book(self.symbol, 50)
                if isinstance(order_book, dict) and "error" not in order_book:
                    bids = order_book.get("bids", [])
                    asks = order_book.get("asks", [])
                    
                    if bids or asks:
                        self.orderbook = {"bids": bids, "asks": asks}
                        if bids:
                            self.bid_price = bids[0][0]
                        if asks:
                            self.ask_price = asks[0][0]
                
                # 獲取行情
                ticker = client.get_ticker(self.symbol)
                if isinstance(ticker, dict) and "error" not in ticker:
                    bid_raw = ticker.get("bidPrice") or ticker.get("bestBidPrice")
                    ask_raw = ticker.get("askPrice") or ticker.get("bestAskPrice")
                    last_raw = ticker.get("lastPrice") or ticker.get("price")
                    
                    if bid_raw:
                        try:
                            self.bid_price = float(bid_raw)
                        except (TypeError, ValueError):
                            pass
                    if ask_raw:
                        try:
                            self.ask_price = float(ask_raw)
                        except (TypeError, ValueError):
                            pass
                    if last_raw:
                        try:
                            self.last_price = float(last_raw)
                            self.add_price_to_history(self.last_price)
                        except (TypeError, ValueError):
                            pass
                
                if self.last_price is None and self.bid_price and self.ask_price:
                    self.last_price = (self.bid_price + self.ask_price) / 2
                    self.add_price_to_history(self.last_price)
                    
            except Exception as e:
                self._logger.error(f"API 備援獲取數據時出錯: {e}")
            
            time.sleep(self.api_poll_interval)
    
    # ==================== 公共 API ====================
    
    def get_current_price(self) -> Optional[float]:
        """獲取當前價格"""
        return self.last_price
    
    def get_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """獲取買賣價"""
        return self.bid_price, self.ask_price
    
    def get_orderbook(self) -> Dict[str, List]:
        """獲取訂單簿"""
        return self.orderbook
    
    def get_volatility(self, window: int = 20) -> float:
        """獲取當前波動率"""
        return calculate_volatility(self.historical_prices, window)
    
    def is_connected(self) -> bool:
        """檢查連接狀態"""
        if not self.connected:
            return False
        if not self.ws:
            return False
        if not hasattr(self.ws, 'sock') or not self.ws.sock:
            return False
        
        try:
            return self.ws.sock.connected
        except:
            return False
    
    def get_liquidity_profile(self, depth_percentage: float = 0.01) -> Optional[Dict[str, Any]]:
        """分析市場流動性特徵"""
        if not self.orderbook["bids"] or not self.orderbook["asks"]:
            return None
        
        mid_price = (self.bid_price + self.ask_price) / 2 if self.bid_price and self.ask_price else None
        if not mid_price:
            return None
        
        min_price = mid_price * (1 - depth_percentage)
        max_price = mid_price * (1 + depth_percentage)
        
        bid_volume = sum(qty for price, qty in self.orderbook["bids"] if price >= min_price)
        ask_volume = sum(qty for price, qty in self.orderbook["asks"] if price <= max_price)
        
        ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'volume_ratio': ratio,
            'imbalance': imbalance,
            'mid_price': mid_price
        }
    
    def check_and_reconnect_if_needed(self) -> bool:
        """檢查連接狀態並在需要時重連"""
        current_time = time.time()
        
        if self.reconnect_cooldown_until and current_time < self.reconnect_cooldown_until:
            if not self.is_connected() and not self.api_fallback_active:
                remaining_cooldown = int(self.reconnect_cooldown_until - current_time)
                self._logger.debug(f"冷卻期內檢查到連接斷開（剩餘 {remaining_cooldown} 秒），啟動 API 備援模式")
                self._start_api_fallback()
            return self.is_connected()
        
        if self.reconnect_cooldown_until and current_time >= self.reconnect_cooldown_until:
            self.reconnect_attempts = 0
            self.reconnect_cooldown_until = 0.0
            self._logger.info("冷卻期結束，重置重連計數器")
        
        if not self.is_connected() and not self.reconnecting:
            self._logger.info("外部檢查發現連接斷開，觸發重連...")
            threading.Thread(target=self._trigger_reconnect, daemon=True).start()
            self._start_api_fallback()
        
        return self.is_connected()
