"""
網格交易策略模塊 - 現貨市場
基於 MarketMaker 基類實現，完全兼容現有的做市套利框架
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from logger import setup_logger
from strategies.market_maker import MarketMaker, format_balance
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("grid_strategy")


class GridStrategy(MarketMaker):
    """現貨網格交易策略

    特點：
    - 在價格區間內設置多個網格價格點位
    - 在每個點位掛限價單
    - 買單成交後，在上一個網格點位掛賣單
    - 賣單成交後，在下一個網格點位掛買單
    - 通過價格波動賺取網格利潤
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        grid_upper_price: Optional[float] = None,  # 網格上限價格
        grid_lower_price: Optional[float] = None,  # 網格下限價格
        grid_num: int = 10,                        # 網格數量
        order_quantity: Optional[float] = None,    # 每格訂單數量
        auto_price_range: bool = False,            # 自動設置價格範圍
        price_range_percent: float = 5.0,          # 自動模式下的價格範圍百分比
        grid_mode: str = "arithmetic",             # 網格模式: arithmetic(等差) 或 geometric(等比)
        ws_proxy: Optional[str] = None,
        exchange: str = 'backpack',
        exchange_config: Optional[Dict[str, Any]] = None,
        enable_database: bool = True,
        **kwargs,
    ) -> None:
        """
        初始化網格交易策略

        Args:
            api_key: API密鑰
            secret_key: API私鑰
            symbol: 交易對
            grid_upper_price: 網格上限價格
            grid_lower_price: 網格下限價格
            grid_num: 網格數量
            order_quantity: 每格訂單數量
            auto_price_range: 是否自動設置價格範圍
            price_range_percent: 自動模式下的價格範圍百分比
            grid_mode: 網格模式（arithmetic或geometric）
            ws_proxy: WebSocket代理地址
            exchange: 交易所名稱
            exchange_config: 交易所配置
            enable_database: 是否啓用數據庫
        """
        # 禁用rebalance功能，網格策略有自己的資金管理邏輯
        kwargs.setdefault("enable_rebalance", False)

        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            base_spread_percentage=0.1,  # 網格策略不使用spread，設置一個默認值
            order_quantity=order_quantity,
            max_orders=1,  # 網格策略不使用max_orders
            ws_proxy=ws_proxy,
            exchange=exchange,
            exchange_config=exchange_config,
            enable_database=enable_database,
            **kwargs,
        )

        # 網格參數
        self.grid_upper_price = grid_upper_price
        self.grid_lower_price = grid_lower_price
        self.grid_num = max(2, grid_num)  # 至少2個網格
        self.order_quantity = order_quantity
        self.auto_price_range = auto_price_range
        self.price_range_percent = price_range_percent
        self.grid_mode = grid_mode

        # 網格狀態
        self.grid_initialized = False
        self.grid_levels: List[float] = []  # 所有網格價格點位

        # 網格訂單跟蹤
        self.grid_orders_by_price: Dict[float, List[Dict]] = {}  # {價格: [訂單信息列表]}
        self.grid_orders_by_id: Dict[str, Dict] = {}             # {訂單ID: 訂單信息}
        self.grid_buy_orders_by_price: Dict[float, List[Dict]] = {}  # 買單
        self.grid_sell_orders_by_price: Dict[float, List[Dict]] = {}  # 賣單

        # 網格點位鎖定狀態（防止重複補單）
        # 買單成交後鎖定該買單價格，賣單成交後才解鎖
        # key: 買單價格, value: 對應的賣單價格
        self.grid_level_locks: Dict[float, float] = {}  # {買單價格: 賣單價格}

        # 統計
        self.grid_buy_filled_count = 0
        self.grid_sell_filled_count = 0
        self.grid_profit = 0.0

        logger.info("初始化網格交易策略: %s", symbol)
        logger.info("網格數量: %d | 模式: %s", self.grid_num, self.grid_mode)

    def _reset_grid_state(self) -> None:
        """清理當前追蹤的網格訂單狀態。"""
        self.grid_orders_by_price.clear()
        self.grid_orders_by_id.clear()
        self.grid_buy_orders_by_price.clear()
        self.grid_sell_orders_by_price.clear()
        self.grid_level_locks.clear()

    def _initialize_grid_prices(self) -> bool:
        """初始化網格價格點位"""
        # 強制使用REST API獲取準確的初始價格
        ticker = self.client.get_ticker(self.symbol)
        if isinstance(ticker, dict) and "error" in ticker:
            logger.error("無法獲取當前價格: %s", ticker.get('error'))
            return False
        
        if "lastPrice" not in ticker:
            logger.error("Ticker數據不完整: %s", ticker)
            return False
        
        current_price = float(ticker['lastPrice'])
        logger.info("從REST API獲取當前價格: %.4f", current_price)
        
        if not current_price or current_price <= 0:
            logger.error("無法獲取有效的當前價格，無法初始化網格")
            return False

        # 自動計算價格範圍
        if self.auto_price_range or self.grid_upper_price is None or self.grid_lower_price is None:
            price_range = current_price * (self.price_range_percent / 100)
            self.grid_upper_price = current_price + price_range
            self.grid_lower_price = current_price - price_range
            logger.info("自動設置網格價格範圍: %.4f ~ %.4f (當前價格: %.4f)",
                       self.grid_lower_price, self.grid_upper_price, current_price)

        # 驗證價格範圍
        if self.grid_lower_price >= self.grid_upper_price:
            logger.error("網格下限價格必須小於上限價格")
            return False

        if current_price < self.grid_lower_price or current_price > self.grid_upper_price:
            logger.warning("當前價格 %.4f 在網格範圍外 [%.4f, %.4f]",
                          current_price, self.grid_lower_price, self.grid_upper_price)

        # 生成網格價格點位
        self.grid_levels = []

        if self.grid_mode == "geometric":
            # 等比網格
            ratio = (self.grid_upper_price / self.grid_lower_price) ** (1 / (self.grid_num - 1))
            for i in range(self.grid_num):
                price = self.grid_lower_price * (ratio ** i)
                price = round_to_tick_size(price, self.tick_size)
                self.grid_levels.append(price)
        else:
            # 等差網格（默認）
            step = (self.grid_upper_price - self.grid_lower_price) / (self.grid_num - 1)
            for i in range(self.grid_num):
                price = self.grid_lower_price + step * i
                price = round_to_tick_size(price, self.tick_size)
                self.grid_levels.append(price)

        # 去重並排序
        self.grid_levels = sorted(list(set(self.grid_levels)))

        logger.info("網格價格點位初始化完成，共 %d 個點位:", len(self.grid_levels))
        for i, price in enumerate(self.grid_levels):
            logger.info("  網格 %d: %.4f", i + 1, price)

        return True

    def initialize_grid(self) -> bool:
        """初始化網格訂單"""
        if self.grid_initialized:
            logger.info("網格已經初始化")
            return True

        # 初始化價格點位
        if not self._initialize_grid_prices():
            return False

        # 獲取當前價格
        current_price = self.get_current_price()
        if not current_price:
            logger.error("無法獲取當前價格")
            return False

        # 取消現有訂單
        logger.info("取消現有訂單...")
        self.cancel_existing_orders()
        self._reset_grid_state()

        # 獲取賬户餘額
        balances = self.get_balance()
        if not balances:
            logger.error("無法獲取賬户餘額")
            return False

        base_balance = balances.get('base_available', 0)
        quote_balance = balances.get('quote_available', 0)

        logger.info("賬户餘額: %.4f %s, %.4f %s",
                   base_balance, self.base_asset,
                   quote_balance, self.quote_asset)

        # 計算每格訂單數量
        if not self.order_quantity:
            # 根據餘額自動計算
            # 買單需要quote資產，賣單需要base資產
            buy_levels = sum(1 for p in self.grid_levels if p < current_price)
            sell_levels = sum(1 for p in self.grid_levels if p > current_price)

            if buy_levels > 0 and sell_levels > 0:
                # 計算可用於買單的資金
                quote_per_order = quote_balance / buy_levels * 0.95  # 預留5%
                buy_quantity = quote_per_order / current_price

                # 計算可用於賣單的數量
                sell_quantity = base_balance / sell_levels * 0.95

                # 取較小值
                self.order_quantity = min(buy_quantity, sell_quantity)
                self.order_quantity = round_to_precision(self.order_quantity, self.base_precision)

                # 確保不小於最小訂單量
                if self.order_quantity < self.min_order_size:
                    self.order_quantity = self.min_order_size

                logger.info("自動計算每格訂單數量: %.4f %s", self.order_quantity, self.base_asset)
            else:
                self.order_quantity = self.min_order_size
                logger.warning("無法計算訂單數量，使用最小值: %.4f", self.order_quantity)

        # 批量構建網格訂單
        orders_to_place = []

        for price in self.grid_levels:
            if price < current_price:
                # 在當前價格下方掛買單
                if quote_balance >= price * self.order_quantity:
                    orders_to_place.append({
                        "orderType": "Limit",
                        "price": str(price),
                        "quantity": str(self.order_quantity),
                        "side": "Bid",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    })
                    quote_balance -= price * self.order_quantity
                else:
                    logger.warning("報價資產餘額不足，無法在價格 %.4f 掛買單", price)

            elif price > current_price:
                # 在當前價格上方掛賣單
                if base_balance >= self.order_quantity:
                    orders_to_place.append({
                        "orderType": "Limit",
                        "price": str(price),
                        "quantity": str(self.order_quantity),
                        "side": "Ask",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    })
                    base_balance -= self.order_quantity
                else:
                    logger.warning("基礎資產餘額不足，無法在價格 %.4f 掛賣單", price)

        # 批量下單
        placed_orders = 0
        if orders_to_place:
            # 檢查客戶端是否支持批量下單
            has_batch_method = hasattr(self.client, 'execute_order_batch') and callable(getattr(self.client, 'execute_order_batch'))

            if has_batch_method:
                logger.info("準備批量下單: %d 個訂單", len(orders_to_place))
                result = self.client.execute_order_batch(orders_to_place)

                if isinstance(result, dict) and "error" in result:
                    logger.error("批量下單失敗: %s", result['error'])
                    # 如果批量下單失敗，回退到逐個下單
                    logger.info("回退到逐個下單模式...")
                    for i, order in enumerate(orders_to_place):
                        price = float(order['price'])
                        side = order['side']
                        result_single = self.client.execute_order(order)

                        if isinstance(result_single, dict) and "error" not in result_single:
                            order_id = result_single.get('id')
                            self._record_grid_order(order_id, price, side, self.order_quantity)
                            placed_orders += 1
                            logger.info("成功掛單 %d/%d: %s %.4f", i+1, len(orders_to_place), side, price)
                        else:
                            logger.error("掛單失敗: %s", result_single.get('error', 'unknown'))
                else:
                    # 批量下單成功，記錄所有訂單
                    if isinstance(result, list):
                        # 創建原始訂單的映射表 (price, side) -> order
                        order_map = {}
                        for order in orders_to_place:
                            key = (float(order['price']), order['side'])
                            order_map[key] = order

                        for order_result in result:
                            if 'id' in order_result:
                                order_id = order_result['id']
                                # 從返回結果中獲取價格和方向
                                result_price = float(order_result.get('price', 0))
                                result_side = order_result.get('side', '')

                                # 查找對應的原始訂單
                                key = (result_price, result_side)
                                if key in order_map:
                                    original_order = order_map[key]
                                    price = float(original_order['price'])
                                    side = original_order['side']
                                    quantity = float(original_order['quantity'])
                                    self._record_grid_order(order_id, price, side, quantity)
                                    placed_orders += 1
                                else:
                                    # 如果無法匹配，使用返回結果中的數據
                                    logger.warning("無法匹配訂單 %s，使用返回數據", order_id)
                                    price = result_price
                                    side = result_side
                                    quantity = float(order_result.get('quantity', self.order_quantity))
                                    self._record_grid_order(order_id, price, side, quantity)
                                    placed_orders += 1
                        logger.info("批量下單成功: %d 個訂單", placed_orders)
            else:
                # 客戶端不支持批量下單，直接逐個下單
                logger.info("該交易所不支持批量下單，使用逐個下單模式: %d 個訂單", len(orders_to_place))
                for i, order in enumerate(orders_to_place):
                    price = float(order['price'])
                    side = order['side']
                    result_single = self.client.execute_order(order)

                    if isinstance(result_single, dict) and "error" not in result_single:
                        order_id = result_single.get('id')
                        self._record_grid_order(order_id, price, side, self.order_quantity)
                        placed_orders += 1
                        logger.info("成功掛單 %d/%d: %s %.4f", i+1, len(orders_to_place), side, price)
                    else:
                        logger.error("掛單失敗: %s", result_single.get('error', 'unknown'))

        logger.info("網格初始化完成: 共放置 %d 個訂單", placed_orders)
        self.grid_initialized = True

        return True

    def _record_grid_order(self, order_id: str, price: float, side: str, quantity: float) -> None:
        """記錄網格訂單信息"""
        order_info = {
            'order_id': order_id,
            'side': side,
            'quantity': quantity,
            'price': price,
            'created_time': datetime.now(),
            'created_from': 'GRID_INIT'
        }

        self.grid_orders_by_id[order_id] = order_info

        if price not in self.grid_orders_by_price:
            self.grid_orders_by_price[price] = []
        self.grid_orders_by_price[price].append(order_info)

        if side == 'Bid':
            if price not in self.grid_buy_orders_by_price:
                self.grid_buy_orders_by_price[price] = []
            self.grid_buy_orders_by_price[price].append(order_info)
        else:
            if price not in self.grid_sell_orders_by_price:
                self.grid_sell_orders_by_price[price] = []
            self.grid_sell_orders_by_price[price].append(order_info)

        self.orders_placed += 1

    def _place_grid_buy_order(self, price: float, quantity: float) -> bool:
        """在指定價格掛買單"""
        order_details = {
            "orderType": "Limit",
            "price": str(price),
            "quantity": str(quantity),
            "side": "Bid",
            "symbol": self.symbol,
            "timeInForce": "GTC",
            "postOnly": True
        }

        result = self.client.execute_order(order_details)

        if isinstance(result, dict) and "error" in result:
            logger.error("掛買單失敗 (價格 %.4f): %s", price, result['error'])
            return False

        order_id = result.get('id')
        logger.info("成功掛買單: 價格=%.4f, 數量=%.4f, 訂單ID=%s", price, quantity, order_id)

        # 記錄訂單信息
        order_info = {
            'order_id': order_id,
            'side': 'Bid',
            'quantity': quantity,
            'price': price,
            'created_time': datetime.now(),
            'created_from': 'INIT'
        }

        self.grid_orders_by_id[order_id] = order_info

        if price not in self.grid_orders_by_price:
            self.grid_orders_by_price[price] = []
        self.grid_orders_by_price[price].append(order_info)

        if price not in self.grid_buy_orders_by_price:
            self.grid_buy_orders_by_price[price] = []
        self.grid_buy_orders_by_price[price].append(order_info)

        self.orders_placed += 1
        return True

    def _place_grid_sell_order(self, price: float, quantity: float) -> bool:
        """在指定價格掛賣單"""
        order_details = {
            "orderType": "Limit",
            "price": str(price),
            "quantity": str(quantity),
            "side": "Ask",
            "symbol": self.symbol,
            "timeInForce": "GTC",
            "postOnly": True
        }

        result = self.client.execute_order(order_details)

        if isinstance(result, dict) and "error" in result:
            logger.error("掛賣單失敗 (價格 %.4f): %s", price, result['error'])
            return False

        order_id = result.get('id')
        logger.info("成功掛賣單: 價格=%.4f, 數量=%.4f, 訂單ID=%s", price, quantity, order_id)

        # 記錄訂單信息
        order_info = {
            'order_id': order_id,
            'side': 'Ask',
            'quantity': quantity,
            'price': price,
            'created_time': datetime.now(),
            'created_from': 'INIT'
        }

        self.grid_orders_by_id[order_id] = order_info

        if price not in self.grid_orders_by_price:
            self.grid_orders_by_price[price] = []
        self.grid_orders_by_price[price].append(order_info)

        if price not in self.grid_sell_orders_by_price:
            self.grid_sell_orders_by_price[price] = []
        self.grid_sell_orders_by_price[price].append(order_info)

        self.orders_placed += 1
        return True

    def on_ws_message(self, stream, data):
        """處理WebSocket消息回調"""
        # 先調用父類處理
        super().on_ws_message(stream, data)

        if not stream.startswith("account.orderUpdate."):
            return

        event_type = data.get('e')
        if event_type in {"orderCancel", "orderCanceled", "orderExpired", "orderReject", "orderRejected"}:
            self._handle_order_cancel(data)

    def _after_fill_processed(self, fill_info: Dict[str, Any]) -> None:
        super()._after_fill_processed(fill_info)

        order_id = fill_info.get('order_id')
        side = fill_info.get('side')
        quantity_raw = fill_info.get('quantity')
        price_raw = fill_info.get('price')

        try:
            quantity = float(quantity_raw or 0)
            price = float(price_raw or 0)
        except (TypeError, ValueError):
            return

        self._handle_grid_fill(order_id, side, quantity, price)

    def _handle_grid_fill(self, order_id: Optional[str], side: Optional[str], quantity: float, price: float) -> None:
        """統一處理網格訂單成交事件（WS 或 REST）。"""
        if not order_id or order_id not in self.grid_orders_by_id:
            return

        normalized_side = side
        if isinstance(side, str):
            side_upper = side.upper()
            if side_upper in ('BUY', 'BID'):
                normalized_side = 'Bid'
            elif side_upper in ('SELL', 'ASK'):
                normalized_side = 'Ask'

        order_info = self.grid_orders_by_id[order_id]
        grid_price = order_info['price']

        logger.info(
            "網格訂單成交: ID=%s, 方向=%s, 價格=%.4f, 數量=%.4f",
            order_id,
            normalized_side,
            price,
            quantity,
        )

        self._remove_grid_order(order_id, grid_price, normalized_side)

        if normalized_side == 'Bid':
            self.grid_buy_filled_count += 1
            self._place_sell_after_buy(grid_price, quantity)
        elif normalized_side == 'Ask':
            self.grid_sell_filled_count += 1
            self._place_buy_after_sell(grid_price, quantity)

    def _remove_grid_order(self, order_id: str, grid_price: float, side: str) -> None:
        """從訂單跟蹤中移除訂單"""
        if order_id in self.grid_orders_by_id:
            del self.grid_orders_by_id[order_id]

        # 從價格字典中移除
        if grid_price in self.grid_orders_by_price:
            self.grid_orders_by_price[grid_price] = [
                o for o in self.grid_orders_by_price[grid_price]
                if o.get('order_id') != order_id
            ]
            if not self.grid_orders_by_price[grid_price]:
                del self.grid_orders_by_price[grid_price]

        # 從買賣單字典中移除
        if side == 'Bid' and grid_price in self.grid_buy_orders_by_price:
            self.grid_buy_orders_by_price[grid_price] = [
                o for o in self.grid_buy_orders_by_price[grid_price]
                if o.get('order_id') != order_id
            ]
            if not self.grid_buy_orders_by_price[grid_price]:
                del self.grid_buy_orders_by_price[grid_price]

        elif side == 'Ask' and grid_price in self.grid_sell_orders_by_price:
            self.grid_sell_orders_by_price[grid_price] = [
                o for o in self.grid_sell_orders_by_price[grid_price]
                if o.get('order_id') != order_id
            ]
            if not self.grid_sell_orders_by_price[grid_price]:
                del self.grid_sell_orders_by_price[grid_price]

    def _handle_order_cancel(self, data: Dict[str, Any]) -> None:
        """當交易所取消訂單時，移除記錄並嘗試重新掛單。"""
        order_id = data.get('i')
        if not order_id:
            return

        order_info = self.grid_orders_by_id.get(order_id)
        if not order_info:
            return

        side = order_info['side']
        price = order_info['price']
        quantity = order_info['quantity']

        logger.warning("網格訂單被取消: ID=%s, 方向=%s, 價格=%.4f", order_id, side, price)
        self._remove_grid_order(order_id, price, side)

        # 嘗試立即重新掛單，確保網格完整
        if side == 'Bid':
            self._place_grid_buy_order(price, quantity)
        else:
            self._place_grid_sell_order(price, quantity)

    def _place_sell_after_buy(self, buy_price: float, quantity: float) -> None:
        """買單成交後，在上一個網格點位掛賣單"""
        # 找到下一個更高的網格點位
        next_price = None
        for price in sorted(self.grid_levels):
            if price > buy_price:
                next_price = price
                break

        if not next_price:
            logger.warning("買入價格 %.4f 已經是最高網格，無法掛賣單", buy_price)
            return

        # 扣除手續費後的實際數量
        actual_quantity = round_to_precision(quantity * 0.999, self.base_precision)
        if actual_quantity < self.min_order_size:
            actual_quantity = self.min_order_size

        logger.info("買單成交後在價格 %.4f 掛賣單 (買入價格: %.4f)", next_price, buy_price)

        success = self._place_grid_sell_order(next_price, actual_quantity)
        if success:
            # 鎖定買單價格，記錄對應的賣單價格
            self.grid_level_locks[buy_price] = next_price
            logger.debug("鎖定網格點位 %.4f，等待賣單 %.4f 成交", buy_price, next_price)

            # 計算網格利潤
            grid_profit = (next_price - buy_price) * actual_quantity
            self.grid_profit += grid_profit
            logger.info("潛在網格利潤: %.4f %s (累計: %.4f)",
                       grid_profit, self.quote_asset, self.grid_profit)
        else:
            logger.warning("掛賣單失敗，不鎖定買單價格 %.4f", buy_price)

    def _place_buy_after_sell(self, sell_price: float, quantity: float) -> None:
        """賣單成交後，在下一個網格點位掛買單"""
        # 找到下一個更低的網格點位
        next_price = None
        for price in sorted(self.grid_levels, reverse=True):
            if price < sell_price:
                next_price = price
                break

        if not next_price:
            logger.warning("賣出價格 %.4f 已經是最低網格，無法掛買單", sell_price)
            # 解鎖對應的買單價格
            for buy_price, locked_sell_price in list(self.grid_level_locks.items()):
                if locked_sell_price == sell_price:
                    del self.grid_level_locks[buy_price]
                    logger.debug("解鎖網格點位 %.4f（賣單在最低點成交）", buy_price)
            return

        # 計算可買入的數量（扣除手續費）
        sell_value = sell_price * quantity * 0.999  # 扣除手續費
        buy_quantity = round_to_precision(sell_value / next_price, self.base_precision)
        if buy_quantity < self.min_order_size:
            buy_quantity = self.min_order_size

        logger.info("賣單成交後在價格 %.4f 掛買單 (賣出價格: %.4f)", next_price, sell_price)

        success = self._place_grid_buy_order(next_price, buy_quantity)
        if success:
            # 解鎖對應的買單價格，允許重新補單
            for buy_price, locked_sell_price in list(self.grid_level_locks.items()):
                if locked_sell_price == sell_price:
                    del self.grid_level_locks[buy_price]
                    logger.debug("解鎖網格點位 %.4f，完成買賣循環", buy_price)

    def place_limit_orders(self) -> None:
        """放置限價單 - 覆蓋父類方法"""
        if not self.grid_initialized:
            success = self.initialize_grid()
            if not success:
                logger.error("網格初始化失敗")
                return
        else:
            # 檢查並補充缺失的網格訂單
            self._refill_grid_orders()

    def _reconcile_grid_orders(self) -> Dict[str, Dict[float, int]]:
        """統計實際掛單數量（按價格/方向），用於檢查是否缺單。"""
        buy_counts: Dict[float, int] = defaultdict(int)
        sell_counts: Dict[float, int] = defaultdict(int)

        try:
            open_orders = self.client.get_open_orders(self.symbol) or []
        except Exception as exc:
            logger.warning("無法同步網格訂單狀態: %s", exc)
            return {'Bid': {}, 'Ask': {}}

        if isinstance(open_orders, dict) and open_orders.get('error'):
            logger.warning("同步網格訂單時收到錯誤: %s", open_orders['error'])
            return {'Bid': {}, 'Ask': {}}

        for order in open_orders:
            if not isinstance(order, dict):
                continue

            side_raw = str(order.get('side', '')).upper()
            price_raw = order.get('price')
            if price_raw is None:
                continue

            try:
                price = round_to_tick_size(float(price_raw), self.tick_size)
            except (TypeError, ValueError):
                continue

            if side_raw in ('BUY', 'BID'):
                buy_counts[price] += 1
            elif side_raw in ('SELL', 'ASK'):
                sell_counts[price] += 1

        return {
            'Bid': dict(buy_counts),
            'Ask': dict(sell_counts),
        }

    def _refill_grid_orders(self) -> None:
        """補充缺失的網格訂單"""

        current_price = self.get_current_price()
        if not current_price:
            return

        active_counts = self._reconcile_grid_orders()
        active_buy_counts = active_counts.get('Bid', {})
        active_sell_counts = active_counts.get('Ask', {})

        # 獲取餘額
        balances = self.get_balance()
        if not balances:
            return

        base_balance = balances.get('base_available', 0)
        quote_balance = balances.get('quote_available', 0)

        refilled = 0

        for price in self.grid_levels:
            if price < current_price:
                # 檢查是否有買單
                if active_buy_counts.get(price, 0) == 0:
                    # 檢查網格點位是否被鎖定（買單成交等待賣單成交）
                    if price in self.grid_level_locks:
                        logger.debug(
                            "網格點位 %.4f 已鎖定，等待賣單 %.4f 成交，暫不補充買單",
                            price,
                            self.grid_level_locks[price],
                        )
                        continue

                    if quote_balance >= price * self.order_quantity:
                        if self._place_grid_buy_order(price, self.order_quantity):
                            quote_balance -= price * self.order_quantity
                            refilled += 1

            elif price > current_price:
                # 檢查是否有賣單
                if active_sell_counts.get(price, 0) == 0:
                    if base_balance >= self.order_quantity:
                        if self._place_grid_sell_order(price, self.order_quantity):
                            base_balance -= self.order_quantity
                            refilled += 1

        if refilled > 0:
            logger.info("補充了 %d 個網格訂單 (總計: %d)",
                       refilled, len(self.grid_orders_by_id))

    def calculate_prices(self) -> Tuple[List[float], List[float]]:
        """計算價格 - 網格策略不需要這個方法，返回空列表"""
        return [], []

    def need_rebalance(self) -> bool:
        """網格策略不需要rebalance"""
        return False

    def rebalance_position(self) -> None:
        """網格策略不需要rebalance"""
        pass

    def _get_extra_summary_sections(self):
        """添加網格特有的統計信息"""
        sections = list(super()._get_extra_summary_sections())

        sections.append((
            "網格交易統計",
            [
                ("網格數量", f"{len(self.grid_levels)}"),
                ("價格範圍", f"{self.grid_lower_price:.4f} ~ {self.grid_upper_price:.4f}"),
                ("網格模式", self.grid_mode),
                ("買單成交次數", f"{self.grid_buy_filled_count}"),
                ("賣單成交次數", f"{self.grid_sell_filled_count}"),
                ("網格利潤", f"{self.grid_profit:.4f} {self.quote_asset}"),
                ("活躍買單數", f"{sum(len(orders) for orders in self.grid_buy_orders_by_price.values())}"),
                ("活躍賣單數", f"{sum(len(orders) for orders in self.grid_sell_orders_by_price.values())}"),
            ],
        ))

        return sections

    def run(self, duration_seconds=3600, interval_seconds=60):
        """運行網格交易策略"""
        logger.info("開始運行網格交易策略: %s", self.symbol)
        logger.info("網格參數: 上限=%.4f, 下限=%.4f, 數量=%d",
                   self.grid_upper_price or 0, self.grid_lower_price or 0, self.grid_num)

        # 調用父類的run方法
        super().run(duration_seconds, interval_seconds)
