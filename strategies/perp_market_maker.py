"""
永續合約做市策略模塊。

此模塊在現貨做市策略的基礎上擴展，提供專為永續合約設計的
開倉、平倉與倉位風險管理功能。
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# 全局函数导入已移除，现在使用客户端方法
from logger import setup_logger
from strategies.market_maker import MarketMaker, format_balance
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("perp_market_maker")


class PerpetualMarketMaker(MarketMaker):
    """專為永續合約設計的做市策略。"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        target_position: float = 0.0,
        max_position: float = 1.0,
        position_threshold: float = 0.1,
        inventory_skew: float = 0.0,
        leverage: float = 1.0,
        ws_proxy: Optional[str] = None,
        exchange: str = 'backpack',
        exchange_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        初始化永續合約做市策略。

        Args:
            api_key (str): API金鑰。
            secret_key (str): API私鑰。
            symbol (str): 交易對。
            target_position (float): 目標持倉量 (絕對值)，策略會將倉位大小維持在此數值附近。
            max_position (float): 最大允許持倉量（絕對值）。
            position_threshold (float): 觸發倉位調整的閾值。
            inventory_skew (float): 庫存偏移，影響報價的不對稱性，以將淨倉位推向0。
            leverage (float): 槓桿倍數。
            ws_proxy (Optional[str]): WebSocket代理地址。
        """
        kwargs.setdefault("enable_rebalance", False)
        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            ws_proxy=ws_proxy,
            exchange=exchange,
            exchange_config=exchange_config,
            **kwargs,
        )

        # 核心修改：target_position 現在是絕對持倉量目標
        self.target_position = abs(target_position)
        self.max_position = max(abs(max_position), self.min_order_size)
        self.position_threshold = max(position_threshold, self.min_order_size)
        self.inventory_skew = max(0.0, min(1.0, inventory_skew))
        self.leverage = max(1.0, leverage)

        self.position_state: Dict[str, Any] = {
            "net": 0.0,
            "avg_entry": 0.0,
            "direction": "FLAT",
            "unrealized": 0.0,
        }

        # 添加總成交量統計（以報價資產計價）
        self.total_volume_quote = 0.0
        self.session_total_volume_quote = 0.0

        logger.info(
            "初始化永續合約做市: %s | 目標持倉量: %s | 最大持倉量: %s | 觸發閾值: %s",
            symbol,
            format_balance(self.target_position),
            format_balance(self.max_position),
            format_balance(self.position_threshold),
        )
        self._update_position_state()

    def on_ws_message(self, stream, data):
        """處理WebSocket消息回調 - 添加總成交量統計"""
        # 先調用父類的消息處理
        super().on_ws_message(stream, data)
        
        # 添加總成交量統計
        if stream.startswith("account.orderUpdate."):
            event_type = data.get('e')
            
            if event_type == 'orderFill':
                try:
                    quantity = float(data.get('l', '0'))  # 成交數量
                    price = float(data.get('L', '0'))     # 成交價格
                    
                    # 計算成交額（以報價資產計價）
                    trade_volume = quantity * price
                    
                    # 更新總成交量統計
                    self.total_volume_quote += trade_volume
                    self.session_total_volume_quote += trade_volume
                    
                    logger.debug(f"更新總成交量: +{trade_volume:.2f} {self.quote_asset}, 累計: {self.total_volume_quote:.2f}")
                    
                except Exception as e:
                    logger.error(f"更新總成交量統計時出錯: {e}")

    # ------------------------------------------------------------------
    # 基礎資訊與工具方法 (此處函數未變動)
    # ------------------------------------------------------------------
    def get_net_position(self) -> float:
        """取得目前的永續合約淨倉位。"""
        try:
            # 直接查詢特定交易對的倉位
            result = self.client.get_positions(self.symbol)

            if isinstance(result, dict) and "error" in result:
                error_msg = result["error"]
                # 404 錯誤表示沒有倉位，這是正常情況
                if "404" in error_msg or "RESOURCE_NOT_FOUND" in error_msg:
                    logger.debug("未找到 %s 的倉位記錄(404)，倉位為0", self.symbol)
                    return 0.0
                else:
                    logger.info(f"result: {result}")
                    logger.error("查詢倉位失敗: %s", error_msg)
                    return 0.0
                
            if not isinstance(result, list):
                logger.warning("倉位API返回格式異常: %s", type(result))
                return 0.0
                
            # 如果返回空列表，說明沒有該交易對的倉位
            if not result:
                logger.debug("未找到 %s 的倉位記錄，倉位為0", self.symbol)
                return 0.0
            
            # 取第一個倉位（因為已經按symbol過濾了）
            position = result[0]
            net_quantity = float(position.get("netQuantity", 0))
            
            logger.debug("從API獲取 %s 永續倉位: %s", self.symbol, net_quantity)
            return net_quantity
            
        except Exception as e:
            logger.error("查詢永續倉位時發生錯誤: %s", e)
            # 發生錯誤時，fallback到本地計算（雖然可能不準確）
            logger.warning("使用本地計算的倉位作為備用")
            return self.total_bought - self.total_sold

    def _get_actual_position_info(self) -> Dict[str, Any]:
        """獲取完整的倉位信息。"""
        try:
            result = self.client.get_positions(self.symbol)
            
            if isinstance(result, dict) and "error" in result:
                error_msg = result["error"]
                # 404 錯誤表示沒有倉位，返回空字典
                if "404" in error_msg or "RESOURCE_NOT_FOUND" in error_msg:
                    logger.debug("未找到 %s 的倉位信息(404)", self.symbol)
                    return {}
                else:
                    logger.error("查詢倉位信息失敗: %s", error_msg)
                    return {}
                
            if not isinstance(result, list) or not result:
                return {}
            
            return result[0]  # 返回第一個（也是唯一的）倉位信息
            
        except Exception as e:
            logger.error("獲取倉位信息時發生錯誤: %s", e)
            return {}

    def _calculate_average_short_entry(self) -> float:
        """計算目前空頭倉位的平均開倉價格。"""
        if not self.sell_trades:
            return 0.0

        sell_queue: List[Tuple[float, float]] = self.sell_trades.copy()
        for _, buy_quantity in self.buy_trades:
            remaining_buy = buy_quantity
            while remaining_buy > 0 and sell_queue:
                sell_price, sell_quantity = sell_queue[0]
                matched = min(remaining_buy, sell_quantity)
                remaining_buy -= matched
                if matched >= sell_quantity:
                    sell_queue.pop(0)
                else:
                    sell_queue[0] = (sell_price, sell_quantity - matched)

        unmatched_quantity = sum(quantity for _, quantity in sell_queue)
        if unmatched_quantity <= 0:
            return 0.0

        unmatched_notional = sum(price * quantity for price, quantity in sell_queue)
        return unmatched_notional / unmatched_quantity

    def _update_position_state(self) -> None:
        """更新倉位相關統計。"""
        net = self.get_net_position()  # 現在這會從API獲取實際倉位
        current_price = self.get_current_price()
        direction = "FLAT"
        avg_entry = 0.0
        unrealized = 0.0

        # 嘗試從API獲取更準確的信息
        position_info = self._get_actual_position_info()
        
        if position_info:
            # 使用API返回的精確信息
            avg_entry = float(position_info.get("entryPrice", 0))
            unrealized = float(position_info.get("pnlUnrealized", 0))
        else:
            # 使用本地計算作為備用
            if net > 0:
                avg_entry = self._calculate_average_buy_cost()
                if current_price:
                    unrealized = (current_price - avg_entry) * net
            elif net < 0:
                avg_entry = self._calculate_average_short_entry()
                if current_price:
                    unrealized = (avg_entry - current_price) * abs(net)

        if net > 0:
            direction = "LONG"
        elif net < 0:
            direction = "SHORT"

        self.position_state = {
            "net": net,
            "avg_entry": avg_entry,
            "direction": direction,
            "unrealized": unrealized,
            "target": self.target_position,
            "max_position": self.max_position,
            "threshold": self.position_threshold,
            "inventory_skew": self.inventory_skew,
            "leverage": self.leverage,
            "timestamp": datetime.utcnow().isoformat(),
            "current_price": current_price or 0.0,
        }

    def get_position_state(self) -> Dict[str, Any]:
        """取得倉位資訊快照。"""
        self._update_position_state()
        return self.position_state

    def estimate_profit(self):
        """覆蓋父類方法，添加總成交量統計顯示"""
        # 先調用父類的方法
        super().estimate_profit()
        
        # 然後添加總成交量統計顯示
        logger.info(f"\n---永續合約總成交量統計---")
        logger.info(f"累計總成交量: {self.total_volume_quote:.2f} {self.quote_asset}")
        logger.info(f"本次執行總成交量: {self.session_total_volume_quote:.2f} {self.quote_asset}")

    def run(self, duration_seconds=3600, interval_seconds=60):
        """執行永續合約做市策略"""
        logger.info(f"開始運行永續合約做市策略: {self.symbol}")
        
        # 重置本次執行的總成交量統計
        self.session_total_volume_quote = 0.0
        
        # 調用父類的 run 方法
        super().run(duration_seconds, interval_seconds)

    # ------------------------------------------------------------------
    # 下單相關 (此處函數未變動)
    # ------------------------------------------------------------------
    def open_position(
        self,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "Limit",
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        client_id: Optional[str] = None,
    ) -> Dict:
        """提交開倉或平倉訂單。"""

        normalized_order_type = order_type.capitalize()
        if normalized_order_type not in {"Limit", "Market"}:
            raise ValueError("order_type 僅支持 'Limit' 或 'Market'")

        qty = round_to_precision(abs(quantity), self.base_precision)
        if qty < self.min_order_size:
            logger.warning(
                "下單數量 %s 低於最小單位 %s，取消下單",
                format_balance(qty),
                format_balance(self.min_order_size),
            )
            return {"error": "quantity_too_small"}

        order_details: Dict[str, object] = {
            "orderType": normalized_order_type,
            "quantity": str(qty),
            "side": side,
            "symbol": self.symbol,
            "timeInForce": time_in_force if normalized_order_type == "Limit" else "IOC",
            "reduceOnly": reduce_only,
        }

        if normalized_order_type == "Limit":
            if price is None:
                raise ValueError("Limit 訂單需要提供價格")
            price_value = round_to_tick_size(price, self.tick_size)
            order_details["price"] = str(price_value)
        else:
            # 使用當前深度推估價格方便記錄
            bid_price, ask_price = self.get_market_depth()
            reference_price = ask_price if side == "Bid" else bid_price
            if reference_price:
                logger.debug(
                    "使用市場價格 %.8f 作為 %s 市價訂單參考",
                    reference_price,
                    side,
                )

        if client_id:
            order_details["clientId"] = str(client_id)

        logger.info(
            "提交永續合約訂單: %s %s %s | reduceOnly=%s | 類型=%s",
            side,
            format_balance(qty),
            self.symbol,
            reduce_only,
            normalized_order_type,
        )

        result = self.client.execute_order(order_details)
        if isinstance(result, dict) and "error" in result:
            logger.error(f"永續合約訂單失敗: {result['error']}, 订单信息：{order_details}")
        else:
            self.orders_placed += 1
            logger.info("永續合約訂單提交成功: %s", result.get("id", "unknown"))

        return result

    def open_long(
        self,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "Limit",
        reduce_only: bool = False,
        **kwargs,
    ) -> Dict:
        """開啟或增加多頭倉位。"""
        return self.open_position(
            side="Bid",
            quantity=quantity,
            price=price,
            order_type=order_type,
            reduce_only=reduce_only,
            **kwargs,
        )

    def open_short(
        self,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "Limit",
        reduce_only: bool = False,
        **kwargs,
    ) -> Dict:
        """開啟或增加空頭倉位。"""
        return self.open_position(
            side="Ask",
            quantity=quantity,
            price=price,
            order_type=order_type,
            reduce_only=reduce_only,
            **kwargs,
        )

    def close_position(
        self,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        order_type: str = "Market",
        side: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> bool:
        """平倉操作。"""
        net = self.get_net_position()  # 使用API獲取實際倉位
        if math.isclose(net, 0.0, abs_tol=self.min_order_size / 10):
            logger.info("實際倉位為零，無需平倉")
            return False

        # 根據實際倉位確定平倉方向
        if net > 0:  # 多頭倉位，需要賣出平倉
            direction = "long"
            order_side = "Ask"
        else:  # 空頭倉位，需要買入平倉
            direction = "short"
            order_side = "Bid"

        # 如果手動指定了side，則使用指定的
        if side is not None:
            direction = side
            order_side = "Ask" if direction == "long" else "Bid"

        qty_available = abs(net)
        qty = qty_available if quantity is None else min(abs(quantity), qty_available)
        qty = round_to_precision(qty, self.base_precision)
        if qty < self.min_order_size:
            logger.info("平倉數量 %s 低於最小單位，忽略", format_balance(qty))
            return False

        logger.info(
            "執行平倉: 實際倉位=%s, 平倉數量=%s, 方向=%s",
            format_balance(net),
            format_balance(qty),
            direction
        )

        result = self.open_position(
            side=order_side,
            quantity=qty,
            price=price,
            order_type=order_type,
            reduce_only=True,
            client_id=client_id,
        )

        if isinstance(result, dict) and "error" in result:
            logger.error(
                "平倉失敗: %s | side=%s qty=%s price=%s type=%s reduceOnly=%s clientId=%s result=%s",
                result.get("error"),
                order_side,
                format_balance(qty),
                price if price is not None else "Market",
                order_type,
                True,
                client_id or "",
                result,
            )
            return False

        logger.info("平倉完成，數量 %s", format_balance(qty))
        self._update_position_state()
        return True

    # ------------------------------------------------------------------
    # 倉位管理 (核心修改)
    # ------------------------------------------------------------------
    def need_rebalance(self) -> bool:
        """判斷是否需要倉位調整 (僅減倉)。"""
        net = self.get_net_position()
        current_size = abs(net)

        if current_size > self.max_position:
            logger.warning(
                "持倉量 %s 超過最大允許 %s，準備執行風控平倉",
                format_balance(current_size),
                format_balance(self.max_position),
            )
            return True

        # 檢查是否超過目標 + 閾值（而非僅僅偏離目標）
        threshold_line = self.target_position + self.position_threshold
        if current_size > threshold_line:
            logger.info(
                "持倉量 %s 超過閾值線 %s (目標 %s + 閾值 %s)，觸發減倉調整",
                format_balance(current_size),
                format_balance(threshold_line),
                format_balance(self.target_position),
                format_balance(self.position_threshold)
            )
            return True

        return False

    def manage_positions(self) -> bool:
        """根據持倉量狀態主動調整，此函數只負責減倉以控制風險。"""
        net = self.get_net_position()
        current_size = abs(net)
        desired_size = self.target_position

        logger.debug(
            "倉位管理: 實際持倉量=%s, 目標持倉量=%s",
            format_balance(current_size),
            format_balance(desired_size)
        )

        # 1. 風控：檢查是否超過最大持倉量
        if current_size > self.max_position:
            excess = current_size - self.max_position
            logger.warning(
                "持倉量 %s 超過最大允許 %s，執行緊急平倉 %s",
                format_balance(current_size),
                format_balance(self.max_position),
                format_balance(excess)
            )
            return self.close_position(quantity=excess, order_type="Market")

        # 2. 減倉：檢查持倉量是否超過目標 + 閾值
        # 只有當實際持倉超過 (目標持倉 + 閾值) 時才減倉
        threshold_line = desired_size + self.position_threshold
        if current_size > threshold_line:
            # 只平掉超出閾值線的部分，而不是全部
            qty_to_close = current_size - threshold_line
            logger.info(
                "持倉量 %s 超過閾值線 %s (目標 %s + 閾值 %s)，執行減倉 %s",
                format_balance(current_size),
                format_balance(threshold_line),
                format_balance(desired_size),
                format_balance(self.position_threshold),
                format_balance(qty_to_close)
            )
            # close_position 會根據淨倉位自動判斷平倉方向
            return self.close_position(quantity=qty_to_close, order_type="Market")

        logger.debug("持倉量在目標範圍內，無需主動管理。")
        return False

    def rebalance_position(self) -> None:
        """覆寫現貨邏輯，改為永續倉位管理。"""
        logger.info("執行永續倉位管理")
        acted = self.manage_positions()
        if not acted:
            logger.info("倉位已在安全範圍內，無需調整")

    # ------------------------------------------------------------------
    # 報價調整 (核心修改)
    # ------------------------------------------------------------------
    def calculate_prices(self):  # type: ignore[override]
        """計算買賣訂單價格，並根據淨倉位進行偏移以控制方向風險。"""
        buy_prices, sell_prices = super().calculate_prices()
        if not buy_prices or not sell_prices:
            return buy_prices, sell_prices

        net = self.get_net_position()
        current_price = self.get_current_price()
        
        # 获取盘口信息
        orderbook = self.client.get_order_book(self.symbol)
        best_bid = best_ask = None
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            if orderbook['bids']:
                best_bid = float(orderbook['bids'][0][0])
            if orderbook['asks']:
                best_ask = float(orderbook['asks'][0][0])
        
        # 输出盘口和持仓信息
        logger.info("=== 市场状态 ===")
        if best_bid and best_ask:
            spread = best_ask - best_bid
            spread_pct = (spread / current_price * 100) if current_price else 0
            logger.info(f"盘口: Bid {best_bid:.3f} | Ask {best_ask:.3f} | 价差 {spread:.3f} ({spread_pct:.3f}%)")
        if current_price:
            logger.info(f"中间价: {current_price:.3f}")
        
        # 输出持仓信息
        direction = "空头" if net < 0 else "多头" if net > 0 else "无仓位"
        logger.info(f"持仓: {direction} {abs(net):.3f} SOL | 目标: {self.target_position:.1f} | 上限: {self.max_position:.1f}")

        # 如果沒有庫存偏移係數或沒有倉位，則不進行調整
        if self.inventory_skew <= 0 or abs(net) < self.min_order_size:
            logger.info(f"原始挂单: 买 {buy_prices[0]:.3f} | 卖 {sell_prices[0]:.3f} (无偏移)")
            return buy_prices, sell_prices

        if self.max_position <= 0:
            return buy_prices, sell_prices

        # 核心偏移邏輯：根據淨倉位(net)調整報價，目標是將淨倉位推向0 (Delta中性)
        # 偏離量就是淨倉位本身
        deviation = net
        skew_ratio = max(-1.0, min(1.0, deviation / self.max_position))

        if not current_price:
            return buy_prices, sell_prices

        # 如果是多頭 (net > 0)，skew_offset為正；如果是空頭 (net < 0)，skew_offset為負
        skew_offset = current_price * self.inventory_skew * skew_ratio

        # 調整價格以鼓勵反向交易，使淨倉位回歸0
        # 如果是多頭 (net > 0)，降低買賣價以鼓勵市場吃掉我們的賣單，同時降低我們買入的意願
        # 如果是空頭 (net < 0)，提高買賣價以鼓勵市場吃掉我們的買單，同時降低我們賣出的意願
        adjusted_buys = [
            round_to_tick_size(price - skew_offset, self.tick_size)
            for price in buy_prices
        ]
        adjusted_sells = [
            round_to_tick_size(price - skew_offset, self.tick_size)
            for price in sell_prices
        ]

        # 输出价格调整详情
        logger.info("=== 价格计算 ===")
        logger.info(f"原始挂单: 买 {buy_prices[0]:.3f} | 卖 {sell_prices[0]:.3f}")
        logger.info(f"偏移计算: 净持仓 {net:.3f} | 偏移系数 {self.inventory_skew:.2f} | 偏移量 {skew_offset:.4f}")
        logger.info(f"调整后挂单: 买 {adjusted_buys[0]:.3f} | 卖 {adjusted_sells[0]:.3f}")

        # 風控：確保調整後買賣價沒有交叉
        if adjusted_buys[0] >= adjusted_sells[0]:
            logger.warning("報價調整後買賣價交叉或價差過小，恢復原始報價。買: %s, 賣: %s", adjusted_buys[0], adjusted_sells[0])
            return buy_prices, sell_prices

        return adjusted_buys, adjusted_sells

    # ------------------------------------------------------------------
    # 其他輔助方法 (核心修改)
    # ------------------------------------------------------------------
    def set_target_position(self, target: float, threshold: Optional[float] = None) -> None:
        """更新目標持倉量 (絕對值) 及觸發閾值。"""
        self.target_position = abs(target)
        if threshold is not None:
            self.position_threshold = max(threshold, self.min_order_size)
        logger.info(
            "更新目標持倉量: %s (閾值: %s)",
            format_balance(self.target_position),
            format_balance(self.position_threshold),
        )
        self._update_position_state()

    def set_max_position(self, max_position: float) -> None:
        """更新最大允許倉位。"""
        self.max_position = max(abs(max_position), self.min_order_size)
        logger.info(
            "更新最大倉位限制: %s",
            format_balance(self.max_position),
        )
        self._update_position_state()