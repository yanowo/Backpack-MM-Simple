"""
永續合約網格交易策略模塊
基於 PerpetualMarketMaker 基類實現，支持多空雙向網格
"""
from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from logger import setup_logger
from strategies.perp_market_maker import PerpetualMarketMaker, format_balance
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("perp_grid_strategy")


class PerpGridStrategy(PerpetualMarketMaker):
    """永續合約網格交易策略

    特點：
    - 支持做多網格和做空網格
    - 在價格區間內設置多個網格價格點位
    - 買入開多後，在上一個網格點位賣出平多
    - 賣出開空後，在下一個網格點位買入平空
    - 支持中性網格（同時做多做空）
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
        grid_type: str = "neutral",                # 網格類型: neutral(中性), long(做多), short(做空)
        target_position: float = 0.0,              # 目標持倉（中性網格時使用）
        max_position: float = 1.0,                 # 最大持倉
        position_threshold: float = 0.1,           # 持倉調整閾值
        inventory_skew: float = 0.0,               # 庫存偏移係數
        stop_loss: Optional[float] = None,         # 止損
        take_profit: Optional[float] = None,       # 止盈
        ws_proxy: Optional[str] = None,
        exchange: str = 'backpack',
        exchange_config: Optional[Dict[str, Any]] = None,
        enable_database: bool = True,
        **kwargs,
    ) -> None:
        """
        初始化永續合約網格交易策略

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
            grid_type: 網格類型（neutral/long/short）
            target_position: 目標持倉
            max_position: 最大持倉
            position_threshold: 持倉調整閾值
            inventory_skew: 庫存偏移係數
            stop_loss: 止損
            take_profit: 止盈
            ws_proxy: WebSocket代理地址
            exchange: 交易所名稱
            exchange_config: 交易所配置
            enable_database: 是否啓用數據庫
        """
        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            base_spread_percentage=0.1,  # 網格策略不使用spread
            order_quantity=order_quantity,
            max_orders=1,  # 網格策略不使用max_orders
            target_position=target_position,
            max_position=max_position,
            position_threshold=position_threshold,
            inventory_skew=inventory_skew,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ws_proxy=ws_proxy,
            exchange=exchange,
            exchange_config=exchange_config,
            enable_database=enable_database,
            **kwargs,
        )

        # 網格參數
        self.grid_upper_price = grid_upper_price
        self.grid_lower_price = grid_lower_price
        self.grid_num = max(2, grid_num)
        self.order_quantity = order_quantity
        self.auto_price_range = auto_price_range
        self.price_range_percent = price_range_percent
        self.grid_mode = grid_mode
        self.grid_type = grid_type  # neutral, long, short

        # 網格狀態
        self.grid_initialized = False
        self.grid_levels: List[float] = []

        # 網格訂單跟蹤
        self.grid_orders_by_price: Dict[float, List[Dict]] = {}
        self.grid_orders_by_id: Dict[str, Dict] = {}
        self.grid_long_orders_by_price: Dict[float, List[Dict]] = {}  # 做多訂單
        self.grid_short_orders_by_price: Dict[float, List[Dict]] = {}  # 做空訂單

        # 網格持倉跟蹤（每個網格點位的持倉）
        self.grid_positions: Dict[float, float] = {}  # {開倉價格: 持倉量}

        # 統計
        self.grid_long_filled_count = 0
        self.grid_short_filled_count = 0
        self.grid_profit = 0.0

        logger.info("初始化永續合約網格交易策略: %s", symbol)
        logger.info("網格數量: %d | 模式: %s | 類型: %s", self.grid_num, self.grid_mode, self.grid_type)

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

        # 計算每格訂單數量
        if not self.order_quantity:
            # 使用最小訂單量
            self.order_quantity = self.min_order_size
            logger.info("使用最小訂單量: %.4f %s", self.order_quantity, self.base_asset)

        # 批量構建網格訂單
        orders_to_place = []

        for price in self.grid_levels:
            if abs(price - current_price) / current_price < 0.001:
                logger.debug("跳過太接近當前價格的網格點位: %.4f", price)
                continue

            if self.grid_type == "neutral":
                # 中性網格：在當前價格下方掛開多單，上方掛開空單
                if price < current_price:
                    # 開多單（買入）
                    orders_to_place.append({
                        "orderType": "Limit",
                        "price": str(price),
                        "quantity": str(self.order_quantity),
                        "side": "Bid",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    })
                elif price > current_price:
                    # 開空單（賣出）
                    orders_to_place.append({
                        "orderType": "Limit",
                        "price": str(price),
                        "quantity": str(self.order_quantity),
                        "side": "Ask",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    })

            elif self.grid_type == "long":
                # 做多網格：只在下方掛開多單
                if price <= current_price:
                    orders_to_place.append({
                        "orderType": "Limit",
                        "price": str(price),
                        "quantity": str(self.order_quantity),
                        "side": "Bid",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    })

            elif self.grid_type == "short":
                # 做空網格：只在上方掛開空單
                if price >= current_price:
                    orders_to_place.append({
                        "orderType": "Limit",
                        "price": str(price),
                        "quantity": str(self.order_quantity),
                        "side": "Ask",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    })

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

        logger.info("網格初始化完成: 共放置 %d 個訂單", placed_orders)
        self.grid_initialized = True

        return True

    def _record_grid_order(self, order_id: str, price: float, side: str, quantity: float) -> None:
        """記錄網格訂單信息"""
        grid_type = 'long' if side == 'Bid' else 'short'

        order_info = {
            'order_id': order_id,
            'side': side,
            'quantity': quantity,
            'price': price,
            'created_time': datetime.now(),
            'created_from': 'GRID_INIT',
            'grid_type': grid_type
        }

        self.grid_orders_by_id[order_id] = order_info

        if price not in self.grid_orders_by_price:
            self.grid_orders_by_price[price] = []
        self.grid_orders_by_price[price].append(order_info)

        if grid_type == 'long':
            if price not in self.grid_long_orders_by_price:
                self.grid_long_orders_by_price[price] = []
            self.grid_long_orders_by_price[price].append(order_info)
        else:
            if price not in self.grid_short_orders_by_price:
                self.grid_short_orders_by_price[price] = []
            self.grid_short_orders_by_price[price].append(order_info)

        self.orders_placed += 1

    def _place_grid_long_order(self, price: float, quantity: float) -> bool:
        """在指定價格掛開多單"""
        result = self.open_long(
            quantity=quantity,
            price=price,
            order_type="Limit",
            reduce_only=False
        )

        if isinstance(result, dict) and "error" in result:
            logger.error("掛開多單失敗 (價格 %.4f): %s", price, result.get('error'))
            return False

        order_id = result.get('id')
        logger.info("成功掛開多單: 價格=%.4f, 數量=%.4f, 訂單ID=%s", price, quantity, order_id)

        # 記錄訂單信息
        order_info = {
            'order_id': order_id,
            'side': 'Bid',
            'quantity': quantity,
            'price': price,
            'created_time': datetime.now(),
            'created_from': 'GRID_INIT',
            'grid_type': 'long'
        }

        self.grid_orders_by_id[order_id] = order_info

        if price not in self.grid_orders_by_price:
            self.grid_orders_by_price[price] = []
        self.grid_orders_by_price[price].append(order_info)

        if price not in self.grid_long_orders_by_price:
            self.grid_long_orders_by_price[price] = []
        self.grid_long_orders_by_price[price].append(order_info)

        self.orders_placed += 1
        return True

    def _place_grid_short_order(self, price: float, quantity: float) -> bool:
        """在指定價格掛開空單"""
        result = self.open_short(
            quantity=quantity,
            price=price,
            order_type="Limit",
            reduce_only=False
        )

        if isinstance(result, dict) and "error" in result:
            logger.error("掛開空單失敗 (價格 %.4f): %s", price, result.get('error'))
            return False

        order_id = result.get('id')
        logger.info("成功掛開空單: 價格=%.4f, 數量=%.4f, 訂單ID=%s", price, quantity, order_id)

        # 記錄訂單信息
        order_info = {
            'order_id': order_id,
            'side': 'Ask',
            'quantity': quantity,
            'price': price,
            'created_time': datetime.now(),
            'created_from': 'GRID_INIT',
            'grid_type': 'short'
        }

        self.grid_orders_by_id[order_id] = order_info

        if price not in self.grid_orders_by_price:
            self.grid_orders_by_price[price] = []
        self.grid_orders_by_price[price].append(order_info)

        if price not in self.grid_short_orders_by_price:
            self.grid_short_orders_by_price[price] = []
        self.grid_short_orders_by_price[price].append(order_info)

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
            grid_type = order_info.get('grid_type', 'unknown')

            logger.info("網格訂單成交: ID=%s, 類型=%s, 方向=%s, 價格=%.4f, 數量=%.4f",
                       order_id, grid_type, side, price, quantity)

            # 從訂單跟蹤中移除
            self._remove_grid_order(order_id, grid_price, grid_type)

            # 根據成交方向和網格類型，掛平倉單
            if side == 'Bid':  # 開多成交
                self.grid_long_filled_count += 1
                # 記錄持倉
                if grid_price not in self.grid_positions:
                    self.grid_positions[grid_price] = 0
                self.grid_positions[grid_price] += quantity
                # 在上一個網格點位掛平多單
                self._place_close_long_after_open(grid_price, quantity)

            elif side == 'Ask':  # 開空成交
                self.grid_short_filled_count += 1
                # 記錄持倉
                if grid_price not in self.grid_positions:
                    self.grid_positions[grid_price] = 0
                self.grid_positions[grid_price] -= quantity
                # 在下一個網格點位掛平空單
                self._place_close_short_after_open(grid_price, quantity)

        except Exception as e:
            logger.error("處理網格訂單成交時出錯: %s", e, exc_info=True)

    def _remove_grid_order(self, order_id: str, grid_price: float, grid_type: str) -> None:
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

        # 從多空單字典中移除
        if grid_type == 'long' and grid_price in self.grid_long_orders_by_price:
            self.grid_long_orders_by_price[grid_price] = [
                o for o in self.grid_long_orders_by_price[grid_price]
                if o.get('order_id') != order_id
            ]
            if not self.grid_long_orders_by_price[grid_price]:
                del self.grid_long_orders_by_price[grid_price]

        elif grid_type == 'short' and grid_price in self.grid_short_orders_by_price:
            self.grid_short_orders_by_price[grid_price] = [
                o for o in self.grid_short_orders_by_price[grid_price]
                if o.get('order_id') != order_id
            ]
            if not self.grid_short_orders_by_price[grid_price]:
                del self.grid_short_orders_by_price[grid_price]

    def _place_close_long_after_open(self, open_price: float, quantity: float) -> None:
        """開多成交後，在上一個網格點位掛平多單"""
        # 找到下一個更高的網格點位
        next_price = None
        for price in sorted(self.grid_levels):
            if price > open_price:
                next_price = price
                break

        if not next_price:
            logger.warning("開多價格 %.4f 已經是最高網格，無法掛平多單", open_price)
            return

        logger.info("開多成交後在價格 %.4f 掛平多單 (開倉價格: %.4f)", next_price, open_price)

        # 延遲一下，等待持倉更新（解決 reduce-only 時序問題）
        time.sleep(0.5)

        # 查詢持倉確認
        net_position = self.get_net_position()
        logger.debug("當前淨持倉: %.4f", net_position)

        # 檢查是否有足夠的多頭持倉可以平倉
        if net_position < quantity * 0.9:  # 允許 10% 的誤差
            logger.warning("多頭持倉不足 (當前: %.4f, 需要: %.4f)，改為不使用 reduce_only",
                          net_position, quantity)
            # 不使用 reduce_only，讓訂單可以開反向倉位
            reduce_only = False
        else:
            reduce_only = True

        # 掛平倉單
        result = self.open_short(
            quantity=quantity,
            price=next_price,
            order_type="Limit",
            reduce_only=reduce_only
        )

        if isinstance(result, dict) and "error" in result:
            error_msg = result.get('error', '')
            logger.error("掛平多單失敗: %s", error_msg)

            # 如果還是 reduce-only 錯誤，重試一次不使用 reduce_only
            if "Reduce only" in error_msg and reduce_only:
                logger.info("重試不使用 reduce_only...")
                time.sleep(0.5)
                result = self.open_short(
                    quantity=quantity,
                    price=next_price,
                    order_type="Limit",
                    reduce_only=False
                )
                if isinstance(result, dict) and "error" not in result:
                    logger.info("重試成功")
                else:
                    logger.error("重試仍失敗: %s", result.get('error', ''))
                    return
            else:
                return

        # 計算網格利潤
        grid_profit = (next_price - open_price) * quantity
        self.grid_profit += grid_profit
        logger.info("潛在網格利潤: %.4f %s (累計: %.4f)",
                   grid_profit, self.quote_asset, self.grid_profit)

    def _place_close_short_after_open(self, open_price: float, quantity: float) -> None:
        """開空成交後，在下一個網格點位掛平空單"""
        # 找到下一個更低的網格點位
        next_price = None
        for price in sorted(self.grid_levels, reverse=True):
            if price < open_price:
                next_price = price
                break

        if not next_price:
            logger.warning("開空價格 %.4f 已經是最低網格，無法掛平空單", open_price)
            return

        logger.info("開空成交後在價格 %.4f 掛平空單 (開倉價格: %.4f)", next_price, open_price)

        # 延遲一下，等待持倉更新（解決 reduce-only 時序問題）
        time.sleep(0.5)

        # 查詢持倉確認
        net_position = self.get_net_position()
        logger.debug("當前淨持倉: %.4f", net_position)

        # 檢查是否有足夠的空頭持倉可以平倉（空頭持倉為負數）
        if net_position > -quantity * 0.9:  # 允許 10% 的誤差
            logger.warning("空頭持倉不足 (當前: %.4f, 需要: %.4f)，改為不使用 reduce_only",
                          net_position, -quantity)
            # 不使用 reduce_only，讓訂單可以開反向倉位
            reduce_only = False
        else:
            reduce_only = True

        # 掛平倉單
        result = self.open_long(
            quantity=quantity,
            price=next_price,
            order_type="Limit",
            reduce_only=reduce_only
        )

        if isinstance(result, dict) and "error" in result:
            error_msg = result.get('error', '')
            logger.error("掛平空單失敗: %s", error_msg)

            # 如果還是 reduce-only 錯誤，重試一次不使用 reduce_only
            if "Reduce only" in error_msg and reduce_only:
                logger.info("重試不使用 reduce_only...")
                time.sleep(0.5)
                result = self.open_long(
                    quantity=quantity,
                    price=next_price,
                    order_type="Limit",
                    reduce_only=False
                )
                if isinstance(result, dict) and "error" not in result:
                    logger.info("重試成功")
                else:
                    logger.error("重試仍失敗: %s", result.get('error', ''))
                    return
            else:
                return

        # 計算網格利潤
        grid_profit = (open_price - next_price) * quantity
        self.grid_profit += grid_profit
        logger.info("潛在網格利潤: %.4f %s (累計: %.4f)",
                   grid_profit, self.quote_asset, self.grid_profit)

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

        refilled = 0

        for price in self.grid_levels:
            if abs(price - current_price) / current_price < 0.001:
                continue

            if self.grid_type == "neutral":
                if price < current_price:
                    if price not in self.grid_long_orders_by_price or not self.grid_long_orders_by_price[price]:
                        if self._place_grid_long_order(price, self.order_quantity):
                            refilled += 1
                elif price > current_price:
                    if price not in self.grid_short_orders_by_price or not self.grid_short_orders_by_price[price]:
                        if self._place_grid_short_order(price, self.order_quantity):
                            refilled += 1

            elif self.grid_type == "long":
                if price <= current_price:
                    if price not in self.grid_long_orders_by_price or not self.grid_long_orders_by_price[price]:
                        if self._place_grid_long_order(price, self.order_quantity):
                            refilled += 1

            elif self.grid_type == "short":
                if price >= current_price:
                    if price not in self.grid_short_orders_by_price or not self.grid_short_orders_by_price[price]:
                        if self._place_grid_short_order(price, self.order_quantity):
                            refilled += 1

        if refilled > 0:
            logger.info("補充了 %d 個網格訂單", refilled)

    def calculate_prices(self) -> Tuple[List[float], List[float]]:
        """計算價格 - 網格策略不需要這個方法"""
        return [], []

    def manage_positions(self) -> bool:
        """管理持倉 - 網格策略有自己的持倉管理邏輯"""
        # 檢查是否超過最大持倉
        net = self.get_net_position()
        current_size = abs(net)

        if current_size > self.max_position:
            excess = current_size - self.max_position
            logger.warning("持倉量 %s 超過最大允許 %s，執行緊急平倉 %s",
                          format_balance(current_size),
                          format_balance(self.max_position),
                          format_balance(excess))
            return self.close_position(quantity=excess, order_type="Market")

        return False

    def _get_extra_summary_sections(self):
        """添加網格特有的統計信息"""
        sections = list(super()._get_extra_summary_sections())

        # 計算網格持倉
        total_long_positions = sum(max(0, pos) for pos in self.grid_positions.values())
        total_short_positions = abs(sum(min(0, pos) for pos in self.grid_positions.values()))

        sections.append((
            "永續合約網格統計",
            [
                ("網格數量", f"{len(self.grid_levels)}"),
                ("價格範圍", f"{self.grid_lower_price:.4f} ~ {self.grid_upper_price:.4f}"),
                ("網格模式", self.grid_mode),
                ("網格類型", self.grid_type),
                ("開多次數", f"{self.grid_long_filled_count}"),
                ("開空次數", f"{self.grid_short_filled_count}"),
                ("網格利潤", f"{self.grid_profit:.4f} {self.quote_asset}"),
                ("多頭持倉", f"{total_long_positions:.4f} {self.base_asset}"),
                ("空頭持倉", f"{total_short_positions:.4f} {self.base_asset}"),
                ("活躍開多單數", f"{sum(len(orders) for orders in self.grid_long_orders_by_price.values())}"),
                ("活躍開空單數", f"{sum(len(orders) for orders in self.grid_short_orders_by_price.values())}"),
            ],
        ))

        return sections

    def run(self, duration_seconds=3600, interval_seconds=60):
        """運行永續合約網格交易策略"""
        logger.info("開始運行永續合約網格交易策略: %s", self.symbol)
        logger.info("網格參數: 上限=%.4f, 下限=%.4f, 數量=%d, 類型=%s",
                   self.grid_upper_price or 0, self.grid_lower_price or 0,
                   self.grid_num, self.grid_type)

        # 調用父類的run方法
        super().run(duration_seconds, interval_seconds)
