"""
範例 WebSocket 客戶端
展示如何為新交易所實現 WebSocket 接口

這個文件提供了一個模板，你可以複製並修改它來支持新的交易所。
需要實現的抽象方法都有詳細的說明和示例代碼。
"""
from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from logger import setup_logger

from .base_ws_client import (
    BaseWebSocketClient,
    WSConnectionConfig,
    WSTickerData,
    WSOrderBookData,
    WSOrderUpdateData,
    WSFillData,
)


logger = setup_logger("example_ws")


class ExampleWebSocket(BaseWebSocketClient):
    """
    範例 WebSocket 客戶端
    
    這是一個模板類，展示如何為新交易所實現 WebSocket 接口。
    
    實現新交易所時，你需要：
    1. 研究該交易所的 WebSocket API 文檔
    2. 了解其消息格式、認證方式、訂閱格式
    3. 繼承 BaseWebSocketClient 並實現所有抽象方法
    4. 在 factory.py 中註冊新的客戶端類
    
    主要需要實現的方法：
    - get_exchange_name(): 返回交易所名稱
    - _create_auth_message(): 創建認證消息
    - _create_subscribe_message(): 創建訂閱消息
    - _create_unsubscribe_message(): 創建取消訂閱消息
    - _parse_message(): 解析原始消息
    - _get_ticker_channel(): 返回行情頻道名稱
    - _get_depth_channel(): 返回深度頻道名稱
    - _get_order_update_channel(): 返回訂單更新頻道名稱
    - _handle_ticker_message(): 處理行情消息
    - _handle_depth_message(): 處理深度消息
    - _handle_order_update_message(): 處理訂單更新消息
    - _handle_fill_message(): 處理成交消息
    - _get_rest_client(): 獲取 REST 客戶端
    """
    
    # 範例交易所的 WebSocket URL（需要替換為真實的 URL）
    DEFAULT_WS_URL = "wss://api.example.exchange/ws"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        symbol: str = "BTC/USDT",
        on_message_callback: Optional[Callable[[str, Any], None]] = None,
        auto_reconnect: bool = True,
        proxy: Optional[str] = None,
        ws_url: Optional[str] = None,
    ):
        """
        初始化範例 WebSocket 客戶端
        
        Args:
            api_key: API 密鑰
            secret_key: API 私鑰
            symbol: 交易對符號
            on_message_callback: 消息回調函數
            auto_reconnect: 是否自動重連
            proxy: 代理地址
            ws_url: WebSocket URL
        """
        self.api_key = api_key
        self.secret_key = secret_key
        
        config = WSConnectionConfig(
            ws_url=ws_url or self.DEFAULT_WS_URL,
            api_key=api_key,
            secret_key=secret_key,
            proxy=proxy,
            auto_reconnect=auto_reconnect,
            # 根據交易所要求調整這些參數
            reconnect_delay=1.0,
            max_reconnect_delay=300.0,
            max_reconnect_attempts=5,
            heartbeat_interval=30,
            ping_interval=30,
            ping_timeout=10,
        )
        
        super().__init__(
            config=config,
            symbol=symbol,
            on_message_callback=on_message_callback,
        )
        
        # 交易所特有的屬性
        self._rest_client = None
    
    # ==================== 必須實現的抽象方法 ====================
    
    def get_exchange_name(self) -> str:
        """
        返回交易所名稱
        
        這個名稱會用於日誌記錄和識別
        """
        return "ExampleExchange"
    
    def _create_auth_message(self) -> Optional[Dict[str, Any]]:
        """
        創建認證消息
        
        不同交易所有不同的認證方式：
        - 有些在連接時發送認證消息
        - 有些在訂閱私有頻道時附加簽名
        - 有些使用 HTTP 頭部認證
        
        Returns:
            認證消息字典，如果不需要獨立認證則返回 None
        
        Example (假設使用 API Key + 簽名認證):
            timestamp = str(int(time.time() * 1000))
            signature = self._create_signature(timestamp)
            return {
                "op": "auth",
                "args": {
                    "apiKey": self.api_key,
                    "timestamp": timestamp,
                    "sign": signature
                }
            }
        """
        # 範例：不需要獨立認證消息
        return None
    
    def _create_subscribe_message(self, channel: str, is_private: bool = False) -> Dict[str, Any]:
        """
        創建訂閱消息
        
        Args:
            channel: 頻道名稱
            is_private: 是否為私有頻道
        
        Returns:
            訂閱消息字典
        
        常見的訂閱格式：
        
        1. Binance 風格:
            {"method": "SUBSCRIBE", "params": ["btcusdt@ticker"]}
        
        2. OKX 風格:
            {"op": "subscribe", "args": [{"channel": "tickers", "instId": "BTC-USDT"}]}
        
        3. Bybit 風格:
            {"op": "subscribe", "args": ["orderbook.50.BTCUSDT"]}
        """
        # 範例訂閱格式（類似 Binance）
        message = {
            "op": "subscribe",
            "args": [channel]
        }
        
        # 如果是私有頻道，可能需要添加認證信息
        if is_private and self.api_key:
            timestamp = str(int(time.time() * 1000))
            # 這裡應該實現實際的簽名邏輯
            message["apiKey"] = self.api_key
            message["timestamp"] = timestamp
            message["sign"] = self._create_signature(timestamp)
        
        return message
    
    def _create_unsubscribe_message(self, channel: str) -> Dict[str, Any]:
        """
        創建取消訂閱消息
        
        Args:
            channel: 頻道名稱
        
        Returns:
            取消訂閱消息字典
        """
        return {
            "op": "unsubscribe",
            "args": [channel]
        }
    
    def _parse_message(self, raw_message: str) -> Optional[Tuple[str, Any]]:
        """
        解析原始 WebSocket 消息
        
        Args:
            raw_message: 原始消息字符串
        
        Returns:
            (stream_name, parsed_data) 元組，如果無法解析則返回 None
        
        常見的消息格式：
        
        1. Binance 風格:
            {"stream": "btcusdt@ticker", "data": {...}}
        
        2. OKX 風格:
            {"arg": {"channel": "tickers", "instId": "BTC-USDT"}, "data": [...]}
        
        3. 簡單格式:
            {"channel": "ticker", "symbol": "BTC/USDT", ...}
        """
        try:
            data = json.loads(raw_message)
            
            # 範例解析邏輯（根據實際交易所格式調整）
            
            # 檢查是否為訂閱確認消息
            if data.get("event") in ("subscribed", "unsubscribed"):
                return None
            
            # 格式 1: {"channel": "xxx", "data": {...}}
            if "channel" in data and "data" in data:
                return data["channel"], data["data"]
            
            # 格式 2: {"stream": "xxx", "data": {...}}
            if "stream" in data and "data" in data:
                return data["stream"], data["data"]
            
            # 格式 3: {"arg": {...}, "data": [...]}
            if "arg" in data and "data" in data:
                channel = data["arg"].get("channel", "unknown")
                symbol = data["arg"].get("instId", "")
                stream_name = f"{channel}.{symbol}" if symbol else channel
                return stream_name, data["data"]
            
            return None
            
        except json.JSONDecodeError:
            return None
    
    def _get_ticker_channel(self) -> str:
        """
        獲取行情頻道名稱
        
        不同交易所的命名方式：
        - Binance: "btcusdt@ticker"
        - OKX: "tickers"
        - Backpack: "bookTicker.BTC_USDC"
        """
        # 範例：ticker.{symbol}
        return f"ticker.{self.symbol}"
    
    def _get_depth_channel(self) -> str:
        """
        獲取深度頻道名稱
        
        不同交易所的命名方式：
        - Binance: "btcusdt@depth"
        - OKX: "books"
        - Backpack: "depth.BTC_USDC"
        """
        # 範例：orderbook.{symbol}
        return f"orderbook.{self.symbol}"
    
    def _get_order_update_channel(self) -> str:
        """
        獲取訂單更新頻道名稱
        
        這通常是私有頻道，需要認證
        """
        # 範例：orders.{symbol}
        return f"orders.{self.symbol}"
    
    def _handle_ticker_message(self, data: Any) -> Optional[WSTickerData]:
        """
        處理行情消息
        
        Args:
            data: 原始行情數據
        
        Returns:
            標準化的行情數據 WSTickerData
        
        需要從原始數據中提取：
        - bid_price: 最佳買價
        - ask_price: 最佳賣價
        - last_price: 最新成交價
        """
        if not isinstance(data, dict):
            return None
        
        # 範例解析邏輯（根據實際交易所格式調整）
        bid_price = None
        ask_price = None
        last_price = None
        
        # 嘗試不同的字段名稱
        for bid_key in ('bestBid', 'bidPrice', 'bid', 'b'):
            if bid_key in data:
                try:
                    bid_price = Decimal(str(data[bid_key]))
                    break
                except:
                    pass
        
        for ask_key in ('bestAsk', 'askPrice', 'ask', 'a'):
            if ask_key in data:
                try:
                    ask_price = Decimal(str(data[ask_key]))
                    break
                except:
                    pass
        
        for last_key in ('last', 'lastPrice', 'price', 'p'):
            if last_key in data:
                try:
                    last_price = Decimal(str(data[last_key]))
                    break
                except:
                    pass
        
        return WSTickerData(
            symbol=self.symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            last_price=last_price,
            timestamp=data.get('timestamp') or data.get('ts'),
            source="ws"
        )
    
    def _handle_depth_message(self, data: Any) -> Optional[WSOrderBookData]:
        """
        處理深度消息
        
        Args:
            data: 原始深度數據
        
        Returns:
            標準化的訂單簿數據 WSOrderBookData
        
        需要從原始數據中提取：
        - bids: 買單列表 [(price, quantity), ...]
        - asks: 賣單列表 [(price, quantity), ...]
        """
        if not isinstance(data, dict):
            return None
        
        bids: List[Tuple[Decimal, Decimal]] = []
        asks: List[Tuple[Decimal, Decimal]] = []
        
        # 嘗試不同的字段名稱
        bids_raw = data.get('bids') or data.get('b') or []
        asks_raw = data.get('asks') or data.get('a') or []
        
        for bid in bids_raw:
            try:
                # 常見格式: [price, quantity] 或 {"price": x, "qty": y}
                if isinstance(bid, (list, tuple)):
                    price = Decimal(str(bid[0]))
                    quantity = Decimal(str(bid[1]))
                elif isinstance(bid, dict):
                    price = Decimal(str(bid.get('price') or bid.get('p')))
                    quantity = Decimal(str(bid.get('quantity') or bid.get('qty') or bid.get('q')))
                else:
                    continue
                bids.append((price, quantity))
            except:
                continue
        
        for ask in asks_raw:
            try:
                if isinstance(ask, (list, tuple)):
                    price = Decimal(str(ask[0]))
                    quantity = Decimal(str(ask[1]))
                elif isinstance(ask, dict):
                    price = Decimal(str(ask.get('price') or ask.get('p')))
                    quantity = Decimal(str(ask.get('quantity') or ask.get('qty') or ask.get('q')))
                else:
                    continue
                asks.append((price, quantity))
            except:
                continue
        
        return WSOrderBookData(
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            timestamp=data.get('timestamp') or data.get('ts'),
            source="ws"
        )
    
    def _handle_order_update_message(self, data: Any) -> Optional[WSOrderUpdateData]:
        """
        處理訂單更新消息
        
        Args:
            data: 原始訂單數據
        
        Returns:
            標準化的訂單更新數據 WSOrderUpdateData
        """
        if not isinstance(data, dict):
            return None
        
        # 範例解析邏輯
        order_id = str(data.get('orderId') or data.get('id') or data.get('i', ''))
        if not order_id:
            return None
        
        # 解析方向
        side_raw = str(data.get('side') or data.get('S', '')).lower()
        side = "BUY" if side_raw in ('buy', 'bid', 'long') else "SELL"
        
        # 解析狀態
        status_raw = str(data.get('status') or data.get('state', '')).upper()
        status_map = {
            'NEW': 'NEW',
            'OPEN': 'NEW',
            'PARTIALLY_FILLED': 'PARTIALLY_FILLED',
            'PARTIAL': 'PARTIALLY_FILLED',
            'FILLED': 'FILLED',
            'CANCELLED': 'CANCELLED',
            'CANCELED': 'CANCELLED',
        }
        status = status_map.get(status_raw, status_raw)
        
        # 解析數值
        price = None
        quantity = None
        filled_quantity = None
        
        for price_key in ('price', 'p', 'px'):
            if price_key in data:
                try:
                    price = Decimal(str(data[price_key]))
                    break
                except:
                    pass
        
        for qty_key in ('quantity', 'qty', 'q', 'size', 'sz'):
            if qty_key in data:
                try:
                    quantity = Decimal(str(data[qty_key]))
                    break
                except:
                    pass
        
        for filled_key in ('filledQty', 'filled', 'executedQty', 'z'):
            if filled_key in data:
                try:
                    filled_quantity = Decimal(str(data[filled_key]))
                    break
                except:
                    pass
        
        remaining_quantity = None
        if quantity and filled_quantity:
            remaining_quantity = quantity - filled_quantity
        
        return WSOrderUpdateData(
            symbol=self.symbol,
            order_id=order_id,
            side=side,
            order_type=str(data.get('type') or data.get('orderType', 'LIMIT')).upper(),
            status=status,
            price=price,
            quantity=quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            timestamp=data.get('timestamp') or data.get('ts'),
            source="ws"
        )
    
    def _handle_fill_message(self, data: Any) -> Optional[WSFillData]:
        """
        處理成交消息
        
        Args:
            data: 原始成交數據
        
        Returns:
            標準化的成交數據 WSFillData
        """
        if not isinstance(data, dict):
            return None
        
        # 檢查是否為成交消息
        event_type = data.get('event') or data.get('e', '')
        if event_type not in ('fill', 'trade', 'execution', 'orderFill'):
            return None
        
        fill_id = str(data.get('tradeId') or data.get('fillId') or data.get('t', ''))
        order_id = str(data.get('orderId') or data.get('i', ''))
        
        if not order_id:
            return None
        
        # 解析方向
        side_raw = str(data.get('side') or data.get('S', '')).lower()
        side = "BUY" if side_raw in ('buy', 'bid', 'long') else "SELL"
        
        # 解析數值
        price = Decimal("0")
        quantity = Decimal("0")
        fee = Decimal("0")
        
        for price_key in ('price', 'p', 'fillPrice', 'L'):
            if price_key in data:
                try:
                    price = Decimal(str(data[price_key]))
                    break
                except:
                    pass
        
        for qty_key in ('qty', 'quantity', 'fillQty', 'l', 'size'):
            if qty_key in data:
                try:
                    quantity = Decimal(str(data[qty_key]))
                    break
                except:
                    pass
        
        for fee_key in ('fee', 'commission', 'n'):
            if fee_key in data:
                try:
                    fee = Decimal(str(data[fee_key]))
                    break
                except:
                    pass
        
        return WSFillData(
            symbol=self.symbol,
            fill_id=fill_id,
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            fee=fee,
            fee_asset=data.get('feeAsset') or data.get('feeCurrency') or data.get('N'),
            is_maker=bool(data.get('isMaker') or data.get('maker') or data.get('m', True)),
            timestamp=data.get('timestamp') or data.get('ts'),
            source="ws"
        )
    
    def _get_rest_client(self) -> Any:
        """
        獲取 REST 客戶端
        
        用於初始化訂單簿和 API 備援模式
        
        Returns:
            REST API 客戶端實例
        
        注意：
        - 需要返回一個具有 get_order_book(symbol, limit) 和 get_ticker(symbol) 方法的對象
        - 如果交易所沒有對應的 REST 客戶端，可以返回 None（但會失去備援功能）
        """
        # 範例：返回 None（表示沒有 REST 客戶端）
        # 實際實現時應該返回對應的 REST 客戶端
        return None
    
    # ==================== 可選：覆蓋基類方法 ====================
    
    def _handle_ping(self, message: Any) -> Optional[Dict[str, Any]]:
        """
        處理 ping 消息
        
        不同交易所有不同的心跳格式：
        - Backpack: {"ping": timestamp}
        - Binance: 使用 WebSocket 協議層的 ping/pong
        - OKX: "ping" 字符串
        """
        if isinstance(message, dict) and "ping" in message:
            return {"pong": message.get("ping")}
        if message == "ping":
            return {"pong": "pong"}
        return None
    
    # ==================== 輔助方法 ====================
    
    def _create_signature(self, timestamp: str) -> str:
        """
        創建簽名
        
        這是一個示例方法，實際實現需要根據交易所的簽名算法來實現
        
        常見的簽名方式：
        - HMAC-SHA256
        - HMAC-SHA512
        - Ed25519
        - ECDSA
        """
        import hmac
        import hashlib
        
        if not self.secret_key:
            return ""
        
        message = timestamp
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature


# ==================== 使用示例 ====================

"""
如何為新交易所創建 WebSocket 客戶端：

1. 複製這個文件並重命名（如 binance_ws_client.py）

2. 修改類名和相關常量：
   - 修改類名為 BinanceWebSocket
   - 修改 DEFAULT_WS_URL
   - 修改 get_exchange_name() 返回值

3. 實現所有抽象方法：
   - 研究交易所 API 文檔
   - 根據文檔實現消息解析邏輯
   - 測試各個方法

4. 在 __init__.py 中導出新客戶端：
   from .binance_ws_client import BinanceWebSocket
   # 添加到 __all__ 列表

5. 測試：
   from ws_client import BinanceWebSocket
   
   ws = BinanceWebSocket(
       symbol="BTCUSDT",
       api_key="xxx",
       secret_key="xxx",
   )
   ws.connect()

使用範例：

    # 直接實例化
    from ws_client.example_ws_client import ExampleWebSocket
    
    ws = ExampleWebSocket(
        symbol="BTC/USDT",
        api_key="your_api_key",
        secret_key="your_secret_key",
        on_message_callback=lambda stream, data: print(f"{stream}: {data}")
    )
    ws.connect()
"""
