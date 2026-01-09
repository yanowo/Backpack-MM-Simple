"""
Backpack 交易所 WebSocket 客戶端
繼承自 BaseWebSocketClient，實現 Backpack 特定的協議
"""
from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    BACKPACK_API_KEY,
    BACKPACK_SECRET_KEY,
    BACKPACK_WS_URL,
    BACKPACK_DEFAULT_WINDOW,
)
from api.auth import create_signature
from api.bp_client import BPClient
from logger import setup_logger

from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSTickerData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSFillData,
)


logger = setup_logger("backpack_ws")


class BackpackWebSocket(BaseWebSocketClient):
    """
    Backpack 交易所 WebSocket 客戶端
    
    實現 Backpack 特定的 WebSocket 協議：
    - 訂閱格式：{"method": "SUBSCRIBE", "params": ["channel.symbol"]}
    - 私有頻道需要簽名認證
    - 支持 bookTicker、depth、account.orderUpdate 等頻道
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        symbol: str = "BTC_USDC",
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ):
        """
        初始化 Backpack WebSocket 客戶端
        
        Args:
            api_key: API 密鑰（默認從配置讀取）
            secret_key: API 私鑰（默認從配置讀取）
            symbol: 交易對符號
            on_message_callback: 消息回調函數
            auto_reconnect: 是否自動重連
            proxy: 代理地址
            ws_url: WebSocket URL（默認從配置讀取）
        """
        self.api_key = api_key or BACKPACK_API_KEY
        self.secret_key = secret_key or BACKPACK_SECRET_KEY
        self.default_window = BACKPACK_DEFAULT_WINDOW
        
        config = WSConnectionConfig(
            ws_url=ws_url or BACKPACK_WS_URL,
            api_key=self.api_key,
            secret_key=self.secret_key,
            proxy=proxy,
            auto_reconnect=auto_reconnect,
            reconnect_delay=1.0,
            max_reconnect_delay=1800.0,
            max_reconnect_attempts=2,
            heartbeat_interval=30,
            ping_interval=30,
            ping_timeout=10,
        )
        
        super().__init__(
            config=config,
            symbol=symbol,
            on_message_callback=on_message_callback,
        )
        
        # Backpack 特有的客戶端緩存
        self._client_cache: Dict[str, BPClient] = {}
    
    # ==================== 實現抽象方法 ====================
    
    def get_exchange_name(self) -> str:
        """返回交易所名稱"""
        return "Backpack"
    
    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        """
        創建 Backpack 認證消息
        
        Backpack 的認證是在訂閱私有頻道時進行的，不需要單獨的認證消息
        """
        return None
    
    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        """
        創建 Backpack 訂閱消息
        
        Args:
            channel: 頻道名稱（已包含 symbol）
            is_private: 是否為私有頻道
            
        Returns:
            訂閱消息字典
        """
        if is_private:
            # 私有頻道需要簽名
            timestamp = str(int(time.time() * 1000))
            window = self.default_window
            sign_message = f"instruction=subscribe&timestamp={timestamp}&window={window}"
            signature = create_signature(self.secret_key, sign_message)
            
            if not signature:
                logger.error("簽名創建失敗，無法訂閱私有數據流")
                return {}
            
            return {
                "method": "SUBSCRIBE",
                "params": [channel],
                "signature": [self.api_key, signature, timestamp, window]
            }
        else:
            # 公共頻道
            return {
                "method": "SUBSCRIBE",
                "params": [channel]
            }
    
    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        """創建 Backpack 取消訂閱消息"""
        return {
            "method": "UNSUBSCRIBE",
            "params": [channel]
        }
    
    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        """
        解析 Backpack WebSocket 消息
        
        Backpack 消息格式：
        {
            "stream": "bookTicker.BTC_USDC",
            "data": {...}
        }
        """
        try:
            data = json.loads(raw_message)
            
            if "stream" in data and "data" in data:
                return data["stream"], data["data"]
            
            return None
            
        except json.JSONDecodeError:
            return None
    
    def _get_ticker_channel(self) -> str:
        """獲取 Backpack 行情頻道名稱"""
        return f"bookTicker.{self.symbol}"
    
    def _get_depth_channel(self) -> str:
        """獲取 Backpack 深度頻道名稱"""
        return f"depth.{self.symbol}"
    
    def _get_order_update_channel(self) -> str:
        """獲取 Backpack 訂單更新頻道名稱"""
        return f"account.orderUpdate.{self.symbol}"
    
    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        """
        處理 Backpack 行情消息
        
        Backpack bookTicker 格式：
        {
            "b": "42000.00",  # 最佳買價
            "a": "42001.00",  # 最佳賣價
            "B": "1.5",       # 最佳買量
            "A": "2.0"        # 最佳賣量
        }
        """
        if not isinstance(data, dict):
            return None
        
        bid_price = None
        ask_price = None
        
        if 'b' in data:
            try:
                bid_price = Decimal(str(data['b']))
            except:
                pass
        
        if 'a' in data:
            try:
                ask_price = Decimal(str(data['a']))
            except:
                pass
        
        # 計算中間價作為 last_price
        last_price = None
        if bid_price and ask_price:
            last_price = (bid_price + ask_price) / 2
        
        return WSTickerData(
            symbol=self.symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            last_price=last_price,
            source="ws"
        )
    
    def _handle_depth_message(self, data: Any) -> Optional[WSOrderBookData]:
        """
        處理 Backpack 深度消息
        
        Backpack depth 格式：
        {
            "b": [["42000.00", "1.5"], ...],  # 買單列表 [price, quantity]
            "a": [["42001.00", "2.0"], ...]   # 賣單列表
        }
        """
        if not isinstance(data, dict):
            return None
        
        bids: List[Tuple[Decimal, Decimal]] = []
        asks: List[Tuple[Decimal, Decimal]] = []
        
        if 'b' in data:
            for bid in data['b']:
                try:
                    price = Decimal(str(bid[0]))
                    quantity = Decimal(str(bid[1]))
                    bids.append((price, quantity))
                except:
                    continue
        
        if 'a' in data:
            for ask in data['a']:
                try:
                    price = Decimal(str(ask[0]))
                    quantity = Decimal(str(ask[1]))
                    asks.append((price, quantity))
                except:
                    continue
        
        return WSOrderBookData(
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            source="ws"
        )
    
    def _handle_order_update_message(self, data: Any) -> Optional[WSOrderUpdateData]:
        """
        處理 Backpack 訂單更新消息
        
        Backpack orderUpdate 格式：
        {
            "e": "orderUpdate",
            "E": 1234567890,
            "s": "BTC_USDC",
            "i": "order_id",
            "S": "Bid",
            "o": "Limit",
            "X": "New",
            "p": "42000.00",
            "q": "1.0",
            "z": "0.5",
            ...
        }
        """
        if not isinstance(data, dict):
            return None
        
        event_type = data.get('e')
        if event_type not in ('orderUpdate', 'orderNew', 'orderCancel', 'orderFill'):
            return None
        
        # 解析方向
        side_raw = data.get('S', '')
        side = "BUY" if side_raw.lower() in ('bid', 'buy') else "SELL"
        
        # 解析狀態
        status_map = {
            'New': 'NEW',
            'PartiallyFilled': 'PARTIALLY_FILLED',
            'Filled': 'FILLED',
            'Cancelled': 'CANCELLED',
            'Expired': 'CANCELLED',
        }
        status_raw = data.get('X', 'NEW')
        status = status_map.get(status_raw, status_raw.upper())
        
        # 解析數值
        price = None
        quantity = None
        filled_quantity = None
        
        if 'p' in data:
            try:
                price = Decimal(str(data['p']))
            except:
                pass
        
        if 'q' in data:
            try:
                quantity = Decimal(str(data['q']))
            except:
                pass
        
        if 'z' in data:
            try:
                filled_quantity = Decimal(str(data['z']))
            except:
                pass
        
        remaining_quantity = None
        if quantity and filled_quantity:
            remaining_quantity = quantity - filled_quantity
        
        return WSOrderUpdateData(
            symbol=self.symbol,
            order_id=str(data.get('i', '')),
            side=side,
            order_type=data.get('o', 'LIMIT').upper(),
            status=status,
            price=price,
            quantity=quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            timestamp=data.get('E'),
            source="ws"
        )
    
    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        """
        處理 Backpack 成交消息
        
        Backpack orderFill 格式：
        {
            "e": "orderFill",
            "S": "Bid",
            "l": "0.5",       # 成交數量
            "L": "42000.00",  # 成交價格
            "i": "order_id",
            "m": true,        # 是否 maker
            "n": "0.001",     # 手續費
            "N": "USDC",      # 手續費資產
            "t": "trade_id"
        }
        """
        if not isinstance(data, dict):
            return None
        
        event_type = data.get('e')
        if event_type != 'orderFill':
            return None
        
        # 解析方向
        side_raw = data.get('S', '')
        side = "BUY" if side_raw.lower() in ('bid', 'buy') else "SELL"
        
        # 解析數值
        price = Decimal("0")
        quantity = Decimal("0")
        fee = Decimal("0")
        
        if 'L' in data:
            try:
                price = Decimal(str(data['L']))
            except:
                pass
        
        if 'l' in data:
            try:
                quantity = Decimal(str(data['l']))
            except:
                pass
        
        if 'n' in data:
            try:
                fee = Decimal(str(data['n']))
            except:
                pass
        
        return WSFillData(
            symbol=self.symbol,
            fill_id=str(data.get('t', '')),
            order_id=str(data.get('i', '')),
            side=side,
            price=price,
            quantity=quantity,
            fee=fee,
            fee_asset=data.get('N'),
            is_maker=bool(data.get('m', True)),
            timestamp=data.get('E'),
            source="ws"
        )
    
    def _get_rest_client(self) -> BPClient:
        """獲取 Backpack REST 客戶端"""
        cache_key = "bp_client"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = BPClient({
                "api_key": self.api_key,
                "secret_key": self.secret_key,
            })
        return self._client_cache[cache_key]
    
    def _handle_ping(self, message: Any) -> Optional[Dict[str, Any]]:
        """
        處理 Backpack ping 消息
        
        Backpack 使用 {"ping": timestamp} 格式
        """
        if isinstance(message, dict) and "ping" in message:
            return {"pong": message.get("ping")}
        return None
    
    # ==================== Backpack 特有方法 ====================
    
    def subscribe_bookTicker(self) -> bool:
        """訂閱最優價格（兼容舊 API）"""
        return self.subscribe_ticker()
    
    def private_subscribe(self, stream: str) -> bool:
        """
        訂閱私有數據流（兼容舊 API）
        
        Args:
            stream: 數據流名稱
            
        Returns:
            是否成功
        """
        return self._subscribe(stream, is_private=True)
    
    # ==================== 重寫基類方法以保持向後兼容 ====================
    
    def _on_open(self, ws_app):
        """WebSocket 打開時的處理（保持向後兼容）"""
        logger.info("Backpack WebSocket 連接已建立")
        self.connected = True
        self.reconnect_attempts = 0
        self.reconnecting = False
        self.last_heartbeat = time.time()
        
        self._stop_api_fallback()
        
        time.sleep(0.5)
        
        # 保存當前訂閱列表用於判斷
        existing_subscriptions = self.subscriptions.copy()
        
        # 初始化訂單簿
        orderbook_initialized = self.initialize_orderbook()
        
        # 訂閱頻道
        if orderbook_initialized:
            # 如果沒有已存在的訂閱，則訂閱默認頻道
            should_subscribe_default = len(existing_subscriptions) == 0
            
            if should_subscribe_default or "bookTicker" in existing_subscriptions:
                self.subscribe_ticker()
            
            if should_subscribe_default or "depth" in existing_subscriptions:
                self.subscribe_depth()
        
        # 重新訂閱私有訂單更新流
        for sub in existing_subscriptions:
            if sub.startswith("account."):
                self.private_subscribe(sub)
    
    def _on_message(self, ws_app, message: str):
        """處理 WebSocket 消息（Backpack 特定實現）"""
        try:
            data = json.loads(message)
            
            # 處理 ping pong 消息
            if isinstance(data, dict) and data.get("ping"):
                pong_message = {"pong": data.get("ping")}
                if self.ws and self.connected:
                    self.ws.send(json.dumps(pong_message))
                    self.last_heartbeat = time.time()
                return
            
            if "stream" in data and "data" in data:
                stream = data["stream"]
                event_data = data["data"]
                
                # 處理 bookTicker
                if stream.startswith("bookTicker."):
                    if 'b' in event_data and 'a' in event_data:
                        self.bid_price = float(event_data['b'])
                        self.ask_price = float(event_data['a'])
                        self.last_price = (self.bid_price + self.ask_price) / 2
                        self.add_price_to_history(self.last_price)
                
                # 處理 depth
                elif stream.startswith("depth."):
                    if 'b' in event_data and 'a' in event_data:
                        self._update_orderbook(event_data)
                
                # 訂單更新數據流
                elif stream.startswith("account.orderUpdate."):
                    self.order_updates.append(event_data)
                
                if self.on_message_callback:
                    self.on_message_callback(stream, event_data)
            
        except Exception as e:
            logger.error(f"處理 WebSocket 消息時出錯: {e}")
    
    def _update_orderbook(self, data: Dict[str, Any]):
        """更新訂單簿（Backpack 特定實現）"""
        # 處理買單更新
        if 'b' in data:
            for bid in data['b']:
                price = float(bid[0])
                quantity = float(bid[1])
                
                if quantity == 0:
                    self.orderbook["bids"] = [b for b in self.orderbook["bids"] if b[0] != price]
                else:
                    found = False
                    for i, b in enumerate(self.orderbook["bids"]):
                        if b[0] == price:
                            self.orderbook["bids"][i] = [price, quantity]
                            found = True
                            break
                    
                    if not found:
                        self.orderbook["bids"].append([price, quantity])
                        self.orderbook["bids"] = sorted(self.orderbook["bids"], key=lambda x: x[0], reverse=True)
        
        # 處理賣單更新
        if 'a' in data:
            for ask in data['a']:
                price = float(ask[0])
                quantity = float(ask[1])
                
                if quantity == 0:
                    self.orderbook["asks"] = [a for a in self.orderbook["asks"] if a[0] != price]
                else:
                    found = False
                    for i, a in enumerate(self.orderbook["asks"]):
                        if a[0] == price:
                            self.orderbook["asks"][i] = [price, quantity]
                            found = True
                            break
                    
                    if not found:
                        self.orderbook["asks"].append([price, quantity])
                        self.orderbook["asks"] = sorted(self.orderbook["asks"], key=lambda x: x[0])
