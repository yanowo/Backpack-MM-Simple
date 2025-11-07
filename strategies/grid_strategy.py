"""
網格交易策略模塊 - 現貨市場
基於 MarketMaker 基類實現，完全兼容現有的做市套利框架
"""
from __future__ import annotations

import math
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

        # 網格依賴關係（買單成交後需要在上方掛賣單）
        self.grid_dependencies: Dict[float, float] = {}  # {買入價格: 賣出價格}

        # 統計
        self.grid_buy_filled_count = 0
        self.grid_sell_filled_count = 0
        self.grid_profit = 0.0

        logger.info("初始化網格交易策略: %s", symbol)
        logger.info("網格數量: %d | 模式: %s", self.grid_num, self.grid_mode)

    def _initialize_grid_prices(self) -> bool:
        """初始化網格價格點位"""
        current_price = self.get_current_price()
        if not current_price:
            logger.error("無法獲取當前價格，無法初始化網格")
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
            if abs(price - current_price) / current_price < 0.001:  # 跳過太接近當前價格的點位
                logger.debug("跳過太接近當前價格的網格點位: %.4f", price)
                continue

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
                    for order_result in result:
                        if 'id' in order_result:
                            order_id = order_result['id']
                            # 從原始訂單列表中找到對應的訂單信息
                            idx = result.index(order_result)
                            if idx < len(orders_to_place):
                                original_order = orders_to_place[idx]
                                price = float(original_order['price'])
                                side = original_order['side']
                                quantity = float(original_order['quantity'])
                                self._record_grid_order(order_id, price, side, quantity)
                                placed_orders += 1
                    logger.info("批量下單成功: %d 個訂單", placed_orders)

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

        # 處理訂單成交事件
        if stream.startswith("account.orderUpdate."):
            event_type = data.get('e')

            if event_type == 'orderFill':
                self._handle_order_fill(data)

    def _handle_order_fill(self, data: Dict[str, Any]) -> None:
        """處理訂單成交"""
        try:
            order_id = data.get('i')
            side = data.get('S')
            quantity = float(data.get('l', '0'))
            price = float(data.get('L', '0'))

            # 檢查是否是網格訂單
            if order_id not in self.grid_orders_by_id:
                return

            order_info = self.grid_orders_by_id[order_id]
            grid_price = order_info['price']

            logger.info("網格訂單成交: ID=%s, 方向=%s, 價格=%.4f, 數量=%.4f",
                       order_id, side, price, quantity)

            # 從訂單跟蹤中移除
            self._remove_grid_order(order_id, grid_price, side)

            # 根據成交方向，在對應網格點位掛反向單
            if side == 'Bid':  # 買單成交
                self.grid_buy_filled_count += 1
                self._place_sell_after_buy(grid_price, quantity)
            elif side == 'Ask':  # 賣單成交
                self.grid_sell_filled_count += 1
                self._place_buy_after_sell(grid_price, quantity)

        except Exception as e:
            logger.error("處理網格訂單成交時出錯: %s", e, exc_info=True)

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
            # 建立依賴關係
            self.grid_dependencies[buy_price] = next_price
            # 計算網格利潤
            grid_profit = (next_price - buy_price) * actual_quantity
            self.grid_profit += grid_profit
            logger.info("潛在網格利潤: %.4f %s (累計: %.4f)",
                       grid_profit, self.quote_asset, self.grid_profit)

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
            return

        # 計算可買入的數量（扣除手續費）
        sell_value = sell_price * quantity * 0.999  # 扣除手續費
        buy_quantity = round_to_precision(sell_value / next_price, self.base_precision)
        if buy_quantity < self.min_order_size:
            buy_quantity = self.min_order_size

        logger.info("賣單成交後在價格 %.4f 掛買單 (賣出價格: %.4f)", next_price, sell_price)

        success = self._place_grid_buy_order(next_price, buy_quantity)
        if success:
            # 解除依賴關係
            for buy_price, dependent_sell_price in list(self.grid_dependencies.items()):
                if dependent_sell_price == sell_price:
                    del self.grid_dependencies[buy_price]
                    logger.debug("解除依賴: 買入價格 %.4f -> 賣出價格 %.4f",
                               buy_price, sell_price)

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

    def _refill_grid_orders(self) -> None:
        """補充缺失的網格訂單"""
        current_price = self.get_current_price()
        if not current_price:
            return

        # 獲取餘額
        balances = self.get_balance()
        if not balances:
            return

        base_balance = balances.get('base_available', 0)
        quote_balance = balances.get('quote_available', 0)

        refilled = 0

        for price in self.grid_levels:
            if abs(price - current_price) / current_price < 0.001:
                continue

            if price < current_price:
                # 檢查是否有買單
                if price not in self.grid_buy_orders_by_price or not self.grid_buy_orders_by_price[price]:
                    # 檢查是否有依賴關係
                    if price in self.grid_dependencies:
                        logger.debug("價格 %.4f 的買單依賴未解除，暫不補充", price)
                        continue

                    if quote_balance >= price * self.order_quantity:
                        if self._place_grid_buy_order(price, self.order_quantity):
                            quote_balance -= price * self.order_quantity
                            refilled += 1

            elif price > current_price:
                # 檢查是否有賣單
                if price not in self.grid_sell_orders_by_price or not self.grid_sell_orders_by_price[price]:
                    if base_balance >= self.order_quantity:
                        if self._place_grid_sell_order(price, self.order_quantity):
                            base_balance -= self.order_quantity
                            refilled += 1

        if refilled > 0:
            logger.info("補充了 %d 個網格訂單", refilled)

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
