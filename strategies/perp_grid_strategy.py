"""
永續合約網格交易策略模塊
基於 PerpetualMarketMaker 基類實現，支持多空雙向網格
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set, Iterable

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

        # === 新的網格訂單追蹤系統 ===
        # 開倉單追蹤：{price: {order_id: order_info}}
        self.open_long_orders: Dict[float, Dict[str, Dict]] = defaultdict(dict)  # 開多單
        self.open_short_orders: Dict[float, Dict[str, Dict]] = defaultdict(dict)  # 開空單
        
        # 平倉單追蹤：{order_id: close_order_info}
        self.close_orders: Dict[str, Dict[str, Any]] = {}
        
        # 網格點位狀態：{price: GridLevelState}
        # GridLevelState: {'locked': bool, 'open_position': float, 'close_order_ids': List[str]}
        self.grid_level_states: Dict[float, Dict[str, Any]] = defaultdict(lambda: {
            'locked': False,           # 是否已鎖定（有持倉待平倉）
            'open_position': 0.0,      # 在該價格點位開倉的淨持倉量
            'close_order_ids': []      # 對應的平倉單ID列表
        })
        
        # 持倉快照（用於檢測倉位變化）
        self.last_position_snapshot: float = 0.0
        self.position_change_threshold: float = self.order_quantity * 0.5  # 倉位變化閾值
        
        # 舊的數據結構（保留兼容性）
        self.grid_orders_by_price: Dict[float, List[Dict]] = {}
        self.grid_orders_by_id: Dict[str, Dict] = {}
        self.grid_long_orders_by_price: Dict[float, List[Dict]] = {}
        self.grid_short_orders_by_price: Dict[float, List[Dict]] = {}
        self.grid_level_locks: Dict[float, str] = {}
        self.close_order_mapping: Dict[str, Dict[str, Any]] = {}

        # 統計
        self.grid_long_filled_count = 0
        self.grid_short_filled_count = 0
        self.grid_profit = 0.0

        # 訂單ID別名：clientOrderIndex 與交易所 order_id 對應
        self.order_alias_map: Dict[str, str] = {}
        self.order_aliases_by_primary: Dict[str, Set[str]] = {}
        
        # === 失敗重試機制 ===
        # 待重試的平倉單隊列：[(open_price, quantity, position_type, retry_count)]
        self.pending_close_orders: List[Tuple[float, float, str, int]] = []
        self.max_close_order_retries = 3  # 平倉單最大重試次數
        
        # === 週期性倉位同步 ===
        self._last_position_sync_time: float = 0.0  # 上次同步時間戳
        self._position_sync_interval: int = 600  # 同步間隔（秒），預設 10 分鐘

        logger.info("初始化永續合約網格交易策略: %s", symbol)
        logger.info("網格數量: %d | 模式: %s | 類型: %s", self.grid_num, self.grid_mode, self.grid_type)

    def _reset_grid_state(self) -> None:
        """清理當前追蹤的網格訂單狀態。"""
        self.open_long_orders.clear()
        self.open_short_orders.clear()
        self.close_orders.clear()
        self.grid_level_states.clear()
        self.order_alias_map.clear()
        self.order_aliases_by_primary.clear()
        
        # 清理舊的數據結構
        self.grid_orders_by_price.clear()
        self.grid_orders_by_id.clear()
        self.grid_long_orders_by_price.clear()
        self.grid_short_orders_by_price.clear()
        self.grid_level_locks.clear()
        self.close_order_mapping.clear()

    # ==================== 新的核心方法：訂單狀態管理 ====================

    def _normalize_order_id(self, value: Any) -> Optional[str]:
        """統一處理訂單ID格式 (str/int/float)。"""
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        try:
            return str(int(value))
        except (TypeError, ValueError):
            try:
                return str(value).strip()
            except Exception:  # pragma: no cover - 保底邏輯
                return None

    def _register_order_aliases(self, primary_id: str, alias_ids: Iterable[str]) -> None:
        """記錄主ID及其所有別名，方便後續對應交換。"""
        normalized_primary = self._normalize_order_id(primary_id)
        if not normalized_primary:
            return

        alias_set = self.order_aliases_by_primary.setdefault(normalized_primary, set())
        alias_set.add(normalized_primary)
        self.order_alias_map[normalized_primary] = normalized_primary

        for alias in alias_ids:
            normalized_alias = self._normalize_order_id(alias)
            if not normalized_alias:
                continue
            alias_set.add(normalized_alias)
            self.order_alias_map[normalized_alias] = normalized_primary

        self._propagate_aliases_to_order_info(normalized_primary, alias_set)

    def _remove_order_aliases(self, primary_id: str) -> None:
        normalized_primary = self._normalize_order_id(primary_id)
        if not normalized_primary:
            return

        alias_set = self.order_aliases_by_primary.pop(normalized_primary, set())
        if not alias_set:
            alias_set = {normalized_primary}
        for alias in alias_set:
            normalized_alias = self._normalize_order_id(alias)
            if normalized_alias:
                self.order_alias_map.pop(normalized_alias, None)

    def _resolve_order_id(self, order_id: Any) -> Optional[str]:
        normalized = self._normalize_order_id(order_id)
        if not normalized:
            return None

        if normalized in self.order_alias_map:
            return self.order_alias_map[normalized]

        for primary, info in self.grid_orders_by_id.items():
            alias_ids = info.get('alias_ids') if isinstance(info, dict) else None
            if alias_ids and normalized in alias_ids:
                self._register_order_aliases(primary, alias_ids)
                return primary

        return normalized

    def _propagate_aliases_to_order_info(self, primary_id: str, alias_set: Set[str]) -> None:
        grid_entry = self.grid_orders_by_id.get(primary_id)
        if isinstance(grid_entry, dict):
            grid_aliases = grid_entry.setdefault('alias_ids', set())
            grid_aliases.update(alias_set)

        for orders in self.open_long_orders.values():
            if primary_id in orders:
                order_info = orders[primary_id]
                alias_info = order_info.setdefault('alias_ids', set())
                alias_info.update(alias_set)
                break

        for orders in self.open_short_orders.values():
            if primary_id in orders:
                order_info = orders[primary_id]
                alias_info = order_info.setdefault('alias_ids', set())
                alias_info.update(alias_set)
                break

    def _extract_order_identifiers(self, order_data: Any) -> Tuple[Optional[str], List[str]]:
        if isinstance(order_data, dict):
            alias_ids: List[str] = []
            candidate_keys = [
                'clientOrderIndex', 'client_order_index', 'orderIndex', 'order_index',
                'id', 'orderId', 'order_id', 'clientOrderId', 'client_order_id',
                'clientId', 'client_id'
            ]
            for key in candidate_keys:
                normalized = self._normalize_order_id(order_data.get(key))
                if normalized and normalized not in alias_ids:
                    alias_ids.append(normalized)

            primary = None
            priority_keys = [
                'clientOrderIndex', 'client_order_index', 'orderIndex', 'order_index',
                'clientId', 'client_id'
            ]
            for key in priority_keys:
                normalized = self._normalize_order_id(order_data.get(key))
                if normalized:
                    primary = normalized
                    break

            if primary is None and alias_ids:
                primary = alias_ids[0]
            if primary and primary not in alias_ids:
                alias_ids.insert(0, primary)
            return primary, alias_ids

        normalized = self._normalize_order_id(order_data)
        if not normalized:
            return None, []
        return normalized, [normalized]

    def _update_aliases_from_open_orders(self, open_orders: Optional[List[Dict[str, Any]]]) -> Set[str]:
        normalized_ids: Set[str] = set()
        if not isinstance(open_orders, list):
            return normalized_ids

        for order in open_orders:
            if not isinstance(order, dict):
                continue
            primary, aliases = self._extract_order_identifiers(order)
            if primary and aliases:
                self._register_order_aliases(primary, aliases)
            for alias in aliases or []:
                normalized = self._normalize_order_id(alias)
                if normalized:
                    normalized_ids.add(normalized)

        return normalized_ids
    
    def _record_open_order(
        self,
        order_id: Any,
        price: float,
        side: str,
        quantity: float,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """記錄開倉單
        
        Args:
            order_id: 訂單ID
            price: 網格價格
            side: 'Bid' 或 'Ask'
            quantity: 數量
        """
        normalized_id = self._normalize_order_id(order_id)
        if not normalized_id:
            logger.warning("無法記錄開倉單，缺少有效ID: %s", order_id)
            return

        alias_list = aliases or []
        self._register_order_aliases(normalized_id, alias_list)
        alias_ids = set(self.order_aliases_by_primary.get(normalized_id, {normalized_id}))

        order_info = {
            'order_id': normalized_id,
            'price': price,
            'side': side,
            'quantity': quantity,
            'filled_quantity': 0.0,  # 已成交數量（用於部分成交追蹤）
            'created_time': datetime.now(),
            'status': 'open',
            'alias_ids': set(alias_ids),
        }
        
        if side == 'Bid':
            self.open_long_orders[price][normalized_id] = order_info
            logger.info("記錄開多單: 價格=%.4f, ID=%s, 別名=%s", price, normalized_id, alias_list)
        else:
            self.open_short_orders[price][normalized_id] = order_info
            logger.info("記錄開空單: 價格=%.4f, ID=%s, 別名=%s", price, normalized_id, alias_list)
        
        # 同時維護舊的數據結構
        self.grid_orders_by_id[normalized_id] = {
            'order_id': normalized_id,
            'price': price,
            'side': side,
            'quantity': quantity,
            'grid_type': 'long' if side == 'Bid' else 'short',
            'alias_ids': set(alias_ids),
        }
    
    def _record_close_order(
        self,
        order_id: str,
        open_price: float,
        quantity: float,
        position_type: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """記錄平倉單"""
        normalized_id = self._normalize_order_id(order_id)
        if not normalized_id:
            logger.warning("無法記錄平倉單，缺少有效ID: %s", order_id)
            return

        alias_list = aliases or []
        self._register_order_aliases(normalized_id, alias_list)
        alias_ids = set(self.order_aliases_by_primary.get(normalized_id, {normalized_id}))

        self.close_orders[normalized_id] = {
            'open_price': open_price,
            'quantity': quantity,
            'position_type': position_type,
            'alias_ids': alias_ids,
            'created_time': datetime.now(),
            'status': 'open'
        }

        state = self.grid_level_states[open_price]
        state['close_order_ids'].append(normalized_id)
        state['locked'] = True

        self.close_order_mapping[normalized_id] = {
            'open_price': open_price,
            'quantity': quantity,
            'position': position_type
        }
        self.grid_level_locks[open_price] = normalized_id
        
        logger.debug("記錄平倉單: 開倉價格=%.4f, ID=%s, 類型=%s", open_price, normalized_id, position_type)
    
    def _remove_open_order(self, order_id: Any, price: float, side: str) -> None:
        """移除開倉單記錄
        
        Args:
            order_id: 訂單ID
            price: 網格價格
            side: 'Bid' 或 'Ask'
        """
        normalized_id = self._resolve_order_id(order_id)
        if not normalized_id:
            return

        if side == 'Bid':
            if price in self.open_long_orders and normalized_id in self.open_long_orders[price]:
                del self.open_long_orders[price][normalized_id]
                if not self.open_long_orders[price]:
                    del self.open_long_orders[price]
                logger.debug("移除開多單記錄: 價格=%.4f, ID=%s", price, normalized_id)
        else:
            if price in self.open_short_orders and normalized_id in self.open_short_orders[price]:
                del self.open_short_orders[price][normalized_id]
                if not self.open_short_orders[price]:
                    del self.open_short_orders[price]
                logger.debug("移除開空單記錄: 價格=%.4f, ID=%s", price, normalized_id)
        
        # 同時維護舊的數據結構
        if normalized_id in self.grid_orders_by_id:
            del self.grid_orders_by_id[normalized_id]

        self._remove_order_aliases(normalized_id)
    
    def _remove_close_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """移除平倉單記錄並返回相關信息
        
        Args:
            order_id: 平倉單ID
            
        Returns:
            平倉單信息，如果不存在則返回None
        """
        normalized_id = self._resolve_order_id(order_id) or self._normalize_order_id(order_id)
        if not normalized_id or normalized_id not in self.close_orders:
            return None
        
        close_info = self.close_orders.pop(normalized_id)
        open_price = close_info['open_price']
        
        # 更新網格點位狀態
        state = self.grid_level_states[open_price]
        if normalized_id in state['close_order_ids']:
            state['close_order_ids'].remove(normalized_id)
        
        # 如果沒有其他平倉單，解鎖該點位
        if not state['close_order_ids']:
            state['locked'] = False
            state['open_position'] = 0.0
            logger.debug("解鎖網格點位: %.4f", open_price)
        
        # 維護舊的數據結構
        self.close_order_mapping.pop(normalized_id, None)
        if open_price in self.grid_level_locks and self.grid_level_locks[open_price] == normalized_id:
            del self.grid_level_locks[open_price]

        self._remove_order_aliases(normalized_id)
        
        logger.debug("移除平倉單記錄: 開倉價格=%.4f, ID=%s", open_price, normalized_id)
        return close_info
    
    # ==================== 新的核心方法：基於訂單狀態的處理 ====================
    
    def _handle_open_order_filled(
        self,
        order_id: str,
        price: float,
        side: str,
        quantity: float,
        raw_order_id: Optional[str] = None,
    ) -> None:
        """處理開倉單成交（基於訂單狀態）
        
        Args:
            order_id: 訂單ID
            price: 成交價格（實際）
            side: 'Bid' 或 'Ask'
            quantity: 成交數量
        """
        normalized_input_id = self._normalize_order_id(raw_order_id or order_id)
        resolved_id = self._resolve_order_id(order_id)
        tracking_id = resolved_id or normalized_input_id

        if not tracking_id:
            logger.warning("無法解析開倉單ID: %s", order_id)
            return

        logger.debug("處理開倉單成交: tracking_id=%s, side=%s", tracking_id, side)
        
        # 從開倉單記錄中查找訂單信息
        order_info = None
        grid_price = None

        # 先在對應方向的列表中查找
        if side == 'Bid':
            for p, orders in self.open_long_orders.items():
                if tracking_id in orders:
                    order_info = orders[tracking_id]
                    grid_price = p
                    logger.debug("在開多單列表中找到: price=%s", p)
                    break
        else:
            for p, orders in self.open_short_orders.items():
                if tracking_id in orders:
                    order_info = orders[tracking_id]
                    grid_price = p
                    logger.debug("在開空單列表中找到: price=%s", p)
                    break
        
        # 如果在對應方向找不到，嘗試在另一個方向查找（可能是 side 標準化問題）
        if not order_info:
            logger.debug("在 %s 方向找不到訂單，嘗試另一方向", side)
            other_orders = self.open_short_orders if side == 'Bid' else self.open_long_orders
            for p, orders in other_orders.items():
                if tracking_id in orders:
                    order_info = orders[tracking_id]
                    grid_price = p
                    # 修正 side
                    side = 'Ask' if side == 'Bid' else 'Bid'
                    logger.warning("訂單 %s 在 %s 方向找到（side 可能不匹配）", tracking_id, side)
                    break
        
        if not order_info or grid_price is None:
            log_id = normalized_input_id or tracking_id
            if normalized_input_id and normalized_input_id != tracking_id:
                log_id = f"{normalized_input_id}->{tracking_id}"
            # 輸出當前追蹤的所有訂單 ID 以便調試
            all_long_ids = [oid for orders in self.open_long_orders.values() for oid in orders.keys()]
            all_short_ids = [oid for orders in self.open_short_orders.values() for oid in orders.keys()]
            logger.warning(
                "開倉單 %s 不在追蹤列表中，可能已處理過。當前追蹤: 開多=%s, 開空=%s",
                log_id, all_long_ids[:5], all_short_ids[:5]
            )
            return
        
        display_id = tracking_id
        if normalized_input_id and normalized_input_id != tracking_id:
            display_id = f"{normalized_input_id}->{tracking_id}"

        # 更新已成交數量（支持部分成交）
        order_info['filled_quantity'] = order_info.get('filled_quantity', 0.0) + quantity
        original_qty = order_info.get('quantity', quantity)
        filled_qty = order_info['filled_quantity']
        is_fully_filled = filled_qty >= original_qty - 0.0001  # 考慮浮點誤差
        
        logger.info(
            "開倉單成交[訂單狀態]: ID=%s, 方向=%s, 網格價格=%.4f, 成交價格=%.4f, "
            "本次成交=%.4f, 累計成交=%.4f/%.4f, 完全成交=%s",
            display_id, side, grid_price, price, quantity, filled_qty, original_qty, is_fully_filled
        )
        
        # 只有完全成交時才移除開倉單記錄
        if is_fully_filled:
            self._remove_open_order(tracking_id, grid_price, side)
        else:
            logger.info("部分成交，保留訂單記錄等待後續成交")
        
        # 更新網格點位狀態的持倉（每次部分成交都要更新）
        state = self.grid_level_states[grid_price]
        if side == 'Bid':
            state['open_position'] += quantity
            if is_fully_filled:
                self.grid_long_filled_count += 1
        else:
            state['open_position'] -= quantity
            if is_fully_filled:
                self.grid_short_filled_count += 1
        
        # 每次部分成交都掛對應數量的平倉單
        if side == 'Bid':
            self._place_close_long_order(grid_price, quantity)
        else:
            self._place_close_short_order(grid_price, quantity)
    
    def _handle_close_order_filled(self, order_id: str, price: float, side: str, quantity: float) -> None:
        """處理平倉單成交（基於訂單狀態）
        
        Args:
            order_id: 訂單ID
            price: 成交價格
            side: 'Bid' 或 'Ask'
            quantity: 成交數量
        """
        close_info = self._remove_close_order(order_id)
        
        if not close_info:
            logger.warning("平倉單 %s 不在追蹤列表中，可能已經處理過", order_id)
            return
        
        open_price = close_info['open_price']
        position_type = close_info['position_type']
        
        logger.info(
            "平倉單成交[訂單狀態]: ID=%s, 方向=%s, 成交價格=%.4f, 數量=%.4f, 開倉價格=%.4f, 類型=%s",
            order_id, side, price, quantity, open_price, position_type
        )
        
        # 計算並記錄網格利潤
        if position_type == 'long':
            grid_profit = (price - open_price) * quantity
        else:
            grid_profit = (open_price - price) * quantity
        
        self.grid_profit += grid_profit
        logger.info("網格利潤實現: %.4f %s (累計: %.4f)", grid_profit, self.quote_asset, self.grid_profit)
    
    # ==================== 新的核心方法：基於倉位變化的檢測 ====================
    
    def _check_position_changes(self) -> None:
        """檢查倉位變化，補強訂單狀態檢測的遺漏
        
        通過比對當前持倉與上次快照，檢測是否有未被訂單狀態捕獲的成交
        """
        current_position = self.get_net_position()
        
        if self.last_position_snapshot == 0.0:
            # 首次運行，只記錄快照
            self.last_position_snapshot = current_position
            logger.debug("初始化持倉快照: %.4f", current_position)
            return
        
        position_change = current_position - self.last_position_snapshot
        
        # 如果倉位變化超過閾值，說明可能有訂單成交
        if abs(position_change) >= self.position_change_threshold:
            logger.warning(
                "檢測到顯著倉位變化: %.4f -> %.4f (變化: %.4f)",
                self.last_position_snapshot, current_position, position_change
            )
            
            # 同步訂單狀態，確保沒有遺漏的成交
            self._sync_orders_with_exchange()
            
            # 更新快照
            self.last_position_snapshot = current_position
        
        # 即使沒有顯著變化，也定期更新快照
        elif abs(position_change) > 0.001:
            logger.debug("倉位小幅變化: %.4f -> %.4f", self.last_position_snapshot, current_position)
            self.last_position_snapshot = current_position
    
    def _sync_orders_with_exchange(self) -> None:
        """與交易所同步訂單狀態，找出已成交但未被檢測到的訂單"""
        try:
            open_orders = self.client.get_open_orders(self.symbol) or []
        except Exception as exc:
            logger.error("同步訂單狀態失敗: %s", exc)
            return
        
        if isinstance(open_orders, dict) and open_orders.get('error'):
            logger.error("同步訂單狀態返回錯誤: %s", open_orders['error'])
            return

        exchange_order_ids = self._update_aliases_from_open_orders(open_orders)
        
        # 檢查開倉單
        filled_open_orders = []
        for price, orders in list(self.open_long_orders.items()):
            for order_id, order_info in list(orders.items()):
                tracked_aliases = {self._normalize_order_id(order_id)}
                alias_set = order_info.get('alias_ids', set()) if isinstance(order_info, dict) else set()
                for alias in alias_set:
                    normalized_alias = self._normalize_order_id(alias)
                    if normalized_alias:
                        tracked_aliases.add(normalized_alias)

                if exchange_order_ids.isdisjoint(tracked_aliases):
                    filled_open_orders.append((order_id, price, 'Bid', order_info['quantity']))
        
        for price, orders in list(self.open_short_orders.items()):
            for order_id, order_info in list(orders.items()):
                tracked_aliases = {self._normalize_order_id(order_id)}
                alias_set = order_info.get('alias_ids', set()) if isinstance(order_info, dict) else set()
                for alias in alias_set:
                    normalized_alias = self._normalize_order_id(alias)
                    if normalized_alias:
                        tracked_aliases.add(normalized_alias)

                if exchange_order_ids.isdisjoint(tracked_aliases):
                    filled_open_orders.append((order_id, price, 'Ask', order_info['quantity']))
        
        # 檢查平倉單
        filled_close_orders = []
        for order_id, close_info in list(self.close_orders.items()):
            tracked_aliases = {self._normalize_order_id(order_id)}
            alias_set = close_info.get('alias_ids', set()) if isinstance(close_info, dict) else set()
            for alias in alias_set:
                normalized_alias = self._normalize_order_id(alias)
                if normalized_alias:
                    tracked_aliases.add(normalized_alias)

            if exchange_order_ids.isdisjoint(tracked_aliases):
                filled_close_orders.append((order_id, close_info))
        
        # 處理發現的已成交訂單
        if filled_open_orders:
            logger.warning("發現 %d 個未被檢測的開倉單成交", len(filled_open_orders))
            for order_id, price, side, quantity in filled_open_orders:
                logger.info("補充處理開倉單成交: ID=%s, 價格=%.4f, 方向=%s", order_id, price, side)
                self._handle_open_order_filled(order_id, price, side, quantity)
        
        if filled_close_orders:
            logger.warning("發現 %d 個未被檢測的平倉單成交", len(filled_close_orders))
            for order_id, close_info in filled_close_orders:
                open_price = close_info['open_price']
                quantity = close_info['quantity']
                position_type = close_info['position_type']
                # 平多是賣出(Ask)，平空是買入(Bid)
                side = 'Ask' if position_type == 'long' else 'Bid'
                logger.info("補充處理平倉單成交: ID=%s, 開倉價格=%.4f, 類型=%s", order_id, open_price, position_type)
                self._handle_close_order_filled(order_id, open_price, side, quantity)

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

            if self.grid_type == "short":
                # 做空網格：只在當前價格以上設置網格
                self.grid_lower_price = current_price
                self.grid_upper_price = current_price + price_range
                logger.info("自動設置做空網格價格範圍: %.4f ~ %.4f (當前價格: %.4f)",
                           self.grid_lower_price, self.grid_upper_price, current_price)
            elif self.grid_type == "long":
                # 做多網格：只在當前價格以下設置網格
                self.grid_lower_price = current_price - price_range
                self.grid_upper_price = current_price
                logger.info("自動設置做多網格價格範圍: %.4f ~ %.4f (當前價格: %.4f)",
                           self.grid_lower_price, self.grid_upper_price, current_price)
            else:
                # 中性網格：在當前價格上下設置網格
                self.grid_upper_price = current_price + price_range
                self.grid_lower_price = current_price - price_range
                logger.info("自動設置中性網格價格範圍: %.4f ~ %.4f (當前價格: %.4f)",
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

        # 一次性獲取當前價格（減少請求次數）
        current_price = self.get_current_price()
        if not current_price:
            logger.error("無法獲取當前價格")
            return False

        # 取消現有訂單
        logger.info("取消現有訂單...")
        self.cancel_existing_orders()
        self._reset_grid_state()

        # 計算每格訂單數量
        if not self.order_quantity:
            # 使用最小訂單量
            self.order_quantity = self.min_order_size
            logger.info("使用最小訂單量: %.4f %s", self.order_quantity, self.base_asset)

        # 批量構建網格訂單
        orders_to_place = []

        for price in self.grid_levels:
            if self.grid_type == "neutral":
                # 中性網格：在當前價格下方掛開多單，上方掛開空單
                if price < current_price:
                    # 開多單（買入）
                    orders_to_place.append({
                        "orderType": "Limit",
                        "price": price,  # 保持為 float，讓 client 處理格式化
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
                        "price": price,  # 保持為 float，讓 client 處理格式化
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
                        "price": price,  # 保持為 float，讓 client 處理格式化
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
                        "price": price,  # 保持為 float，讓 client 處理格式化
                        "quantity": str(self.order_quantity),
                        "side": "Ask",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    })

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
                            self._record_grid_order(result_single, price, side, self.order_quantity)
                            placed_orders += 1
                            logger.info("成功掛單 %d/%d: %s %.4f", i+1, len(orders_to_place), side, price)
                        else:
                            logger.error("掛單失敗: %s", result_single.get('error', 'unknown'))
                else:
                    # 批量下單成功，記錄所有訂單
                    # 處理不同交易所的響應格式
                    orders_list = []

                    if isinstance(result, list):
                        # 直接返回訂單數組（Backpack、Lighter、Paradex 成功時）
                        orders_list = result
                    elif isinstance(result, dict):
                        # 包含 orders 字段的響應（Paradex 部分成功時）
                        if "orders" in result:
                            orders_list = result["orders"]
                            # 記錄錯誤（過濾掉 None 值）
                            if "errors" in result and result.get("errors"):
                                # Paradex API 返回的 errors 中，成功的訂單對應 None
                                real_errors = [e for e in result["errors"] if e is not None]
                                if real_errors:
                                    logger.warning("批量下單部分失敗，錯誤數量: %d", len(real_errors))
                                    for error_info in real_errors[:3]:  # 只記錄前3個錯誤
                                        logger.warning("錯誤詳情: %s", error_info)

                    # 記錄所有成功的訂單
                    for order_result in orders_list:
                        if not isinstance(order_result, dict):
                            continue

                        price = float(order_result.get('price', 0))
                        side = order_result.get('side', '')
                        if isinstance(side, str):
                            if side.upper() in ['BUY', 'LONG']:
                                side = 'Bid'
                            elif side.upper() in ['SELL', 'SHORT', 'ASK']:
                                side = 'Ask'

                        quantity_raw = (
                            order_result.get('quantity') or
                            order_result.get('size') or
                            order_result.get('origQty') or
                            self.order_quantity
                        )
                        quantity = float(quantity_raw)

                        self._record_grid_order(order_result, price, side, quantity)
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
                        self._record_grid_order(result_single, price, side, self.order_quantity)
                        placed_orders += 1
                        logger.info("成功掛單 %d/%d: %s %.4f", i+1, len(orders_to_place), side, price)
                    else:
                        logger.error("掛單失敗: %s", result_single.get('error', 'unknown'))

        logger.info("網格初始化完成: 共放置 %d 個訂單", placed_orders)
        self.grid_initialized = True

        # 啟動時同步倉位，為多餘的倉位補掛平倉單
        self._sync_position_with_grid()

        return True

    def _sync_position_with_grid(self) -> None:
        """同步倉位與網格狀態
        
        檢查當前實際倉位和已掛平倉單，為多餘的倉位補掛平倉單
        """
        net_position = self.get_net_position()
        
        if abs(net_position) < self.min_order_size:
            logger.info("倉位同步檢查: 當前無持倉，無需補掛平倉單")
            return
        
        # 計算已掛平倉單的總數量
        total_close_order_qty = 0.0
        for close_info in self.close_orders.values():
            total_close_order_qty += close_info.get('quantity', 0)
        
        # 計算待重試平倉單的總數量
        pending_qty = sum(qty for _, qty, _, _ in self.pending_close_orders)
        total_close_order_qty += pending_qty
        
        if net_position < 0:
            # 空頭倉位
            uncovered_qty = abs(net_position) - total_close_order_qty
            if uncovered_qty >= self.min_order_size:
                logger.warning(
                    "倉位同步: 發現 %.4f 空頭倉位沒有對應的平倉單，正在補掛...",
                    uncovered_qty
                )
                self._place_sync_close_orders(uncovered_qty, 'short')
            else:
                logger.info(
                    "倉位同步檢查: 空頭倉位 %.4f, 已掛平倉單 %.4f, 無需補掛",
                    abs(net_position), total_close_order_qty
                )
        else:
            # 多頭倉位
            uncovered_qty = net_position - total_close_order_qty
            if uncovered_qty >= self.min_order_size:
                logger.warning(
                    "倉位同步: 發現 %.4f 多頭倉位沒有對應的平倉單，正在補掛...",
                    uncovered_qty
                )
                self._place_sync_close_orders(uncovered_qty, 'long')
            else:
                logger.info(
                    "倉位同步檢查: 多頭倉位 %.4f, 已掛平倉單 %.4f, 無需補掛",
                    net_position, total_close_order_qty
                )

    def _place_sync_close_orders(self, uncovered_qty: float, position_type: str) -> None:
        """為未覆蓋的倉位補掛平倉單
        
        Args:
            uncovered_qty: 未覆蓋的倉位數量
            position_type: 'long' 或 'short'
        """
        current_price = self.get_current_price()
        if not current_price:
            logger.error("無法獲取當前價格，無法補掛平倉單")
            return
        
        # 計算需要補掛的訂單數量
        order_qty = self.order_quantity if self.order_quantity else self.min_order_size
        num_orders = int(uncovered_qty / order_qty)
        remaining_qty = uncovered_qty - (num_orders * order_qty)
        
        placed_count = 0
        
        if position_type == 'short':
            # 空頭倉位：需要在當前價格下方掛買單（平空）
            # 找到當前價格下方的網格點位
            close_prices = sorted([p for p in self.grid_levels if p < current_price], reverse=True)
            
            if not close_prices:
                # 如果沒有合適的網格點位，使用當前價格減去網格間距
                if len(self.grid_levels) >= 2:
                    sorted_levels = sorted(self.grid_levels)
                    grid_step = (sorted_levels[-1] - sorted_levels[0]) / (len(sorted_levels) - 1)
                else:
                    grid_step = current_price * 0.001
                close_prices = [round_to_tick_size(current_price - grid_step * (i + 1), self.tick_size) 
                               for i in range(num_orders + 1)]
            
            price_idx = 0
            for i in range(num_orders):
                if price_idx >= len(close_prices):
                    price_idx = 0  # 循環使用價格
                
                close_price = close_prices[price_idx]
                price_idx += 1
                
                logger.info("補掛平空單 %d/%d: 價格=%.4f, 數量=%.4f", i+1, num_orders, close_price, order_qty)
                result = self.open_long(
                    quantity=order_qty,
                    price=close_price,
                    order_type="Limit",
                    reduce_only=True,
                    post_only=False  # 補掛時不使用 post_only，確保能掛出
                )
                
                if isinstance(result, dict) and "error" not in result:
                    primary_id, alias_ids = self._extract_order_identifiers(result)
                    if primary_id:
                        # 同步補掛的訂單使用平倉價格作為開倉價格，這樣不會產生虛假利潤
                        # 網格利潤只計算正常流程（開倉->平倉）的利潤
                        self._record_close_order(primary_id, close_price, order_qty, 'short', aliases=alias_ids)
                        placed_count += 1
                        logger.info("同步平空單已記錄 (無利潤計算): ID=%s, 價格=%.4f", primary_id, close_price)
                else:
                    logger.error("補掛平空單失敗: %s", result.get('error', 'unknown') if isinstance(result, dict) else result)
            
            # 處理剩餘數量
            if remaining_qty >= self.min_order_size and close_prices:
                close_price = close_prices[0]
                logger.info("補掛剩餘平空單: 價格=%.4f, 數量=%.4f", close_price, remaining_qty)
                result = self.open_long(
                    quantity=remaining_qty,
                    price=close_price,
                    order_type="Limit",
                    reduce_only=True,
                    post_only=False
                )
                if isinstance(result, dict) and "error" not in result:
                    primary_id, alias_ids = self._extract_order_identifiers(result)
                    if primary_id:
                        # 同步補掛的訂單使用平倉價格作為開倉價格
                        self._record_close_order(primary_id, close_price, remaining_qty, 'short', aliases=alias_ids)
                        placed_count += 1
                        logger.info("同步平空單已記錄 (無利潤計算): ID=%s, 價格=%.4f", primary_id, close_price)
        
        else:
            # 多頭倉位：需要在當前價格上方掛賣單（平多）
            close_prices = sorted([p for p in self.grid_levels if p > current_price])
            
            if not close_prices:
                if len(self.grid_levels) >= 2:
                    sorted_levels = sorted(self.grid_levels)
                    grid_step = (sorted_levels[-1] - sorted_levels[0]) / (len(sorted_levels) - 1)
                else:
                    grid_step = current_price * 0.001
                close_prices = [round_to_tick_size(current_price + grid_step * (i + 1), self.tick_size) 
                               for i in range(num_orders + 1)]
            
            price_idx = 0
            for i in range(num_orders):
                if price_idx >= len(close_prices):
                    price_idx = 0
                
                close_price = close_prices[price_idx]
                price_idx += 1
                
                logger.info("補掛平多單 %d/%d: 價格=%.4f, 數量=%.4f", i+1, num_orders, close_price, order_qty)
                result = self.open_short(
                    quantity=order_qty,
                    price=close_price,
                    order_type="Limit",
                    reduce_only=True,
                    post_only=False
                )
                
                if isinstance(result, dict) and "error" not in result:
                    primary_id, alias_ids = self._extract_order_identifiers(result)
                    if primary_id:
                        # 同步補掛的訂單使用平倉價格作為開倉價格，這樣不會產生虛假利潤
                        self._record_close_order(primary_id, close_price, order_qty, 'long', aliases=alias_ids)
                        placed_count += 1
                        logger.info("同步平多單已記錄 (無利潤計算): ID=%s, 價格=%.4f", primary_id, close_price)
                else:
                    logger.error("補掛平多單失敗: %s", result.get('error', 'unknown') if isinstance(result, dict) else result)
            
            if remaining_qty >= self.min_order_size and close_prices:
                close_price = close_prices[0]
                logger.info("補掛剩餘平多單: 價格=%.4f, 數量=%.4f", close_price, remaining_qty)
                result = self.open_short(
                    quantity=remaining_qty,
                    price=close_price,
                    order_type="Limit",
                    reduce_only=True,
                    post_only=False
                )
                if isinstance(result, dict) and "error" not in result:
                    primary_id, alias_ids = self._extract_order_identifiers(result)
                    if primary_id:
                        # 同步補掛的訂單使用平倉價格作為開倉價格
                        self._record_close_order(primary_id, close_price, remaining_qty, 'long', aliases=alias_ids)
                        placed_count += 1
                        logger.info("同步平多單已記錄 (無利潤計算): ID=%s, 價格=%.4f", primary_id, close_price)
        
        logger.warning("倉位同步完成: 成功補掛 %d 個平倉單", placed_count)

    def _record_grid_order(self, order_data: Any, price: float, side: str, quantity: float) -> None:
        """記錄網格訂單信息（使用新的記錄系統）"""
        primary_id, alias_ids = self._extract_order_identifiers(order_data)
        if not primary_id:
            logger.warning("無法記錄網格訂單，缺少訂單ID: %s", order_data)
            return

        self._record_open_order(primary_id, price, side, quantity, aliases=alias_ids)
        self.orders_placed += 1

    def _place_grid_long_order(self, price: float, quantity: float) -> bool:
        """在指定價格掛開多單（使用新的記錄系統）
        
        重要：使用 post_only=True 確保只能作為 Maker 成交，
        避免市價成交導致成交價格與網格價格不符
        """
        result = self.open_long(
            quantity=quantity,
            price=price,
            order_type="Limit",
            reduce_only=False,
            post_only=True  # 強制 Post-Only，避免市價成交
        )

        if isinstance(result, dict) and "error" in result:
            logger.error("掛開多單失敗 (價格 %.4f): %s", price, result.get('error'))
            return False

        primary_id, alias_ids = self._extract_order_identifiers(result)
        logger.info("成功掛開多單: 價格=%.4f, 數量=%.4f, 訂單ID=%s", price, quantity, primary_id)

        # 使用新的記錄系統
        self._record_open_order(primary_id, price, 'Bid', quantity, aliases=alias_ids)
        
        self.orders_placed += 1
        return True

    def _place_grid_short_order(self, price: float, quantity: float) -> bool:
        """在指定價格掛開空單（使用新的記錄系統）
        
        重要：使用 post_only=True 確保只能作為 Maker 成交，
        避免市價成交導致成交價格與網格價格不符
        """
        result = self.open_short(
            quantity=quantity,
            price=price,
            order_type="Limit",
            reduce_only=False,
            post_only=True  # 強制 Post-Only，避免市價成交
        )

        if isinstance(result, dict) and "error" in result:
            logger.error("掛開空單失敗 (價格 %.4f): %s", price, result.get('error'))
            return False

        primary_id, alias_ids = self._extract_order_identifiers(result)
        logger.info("成功掛開空單: 價格=%.4f, 數量=%.4f, 訂單ID=%s", price, quantity, primary_id)

        # 使用新的記錄系統
        self._record_open_order(primary_id, price, 'Ask', quantity, aliases=alias_ids)
        
        self.orders_placed += 1
        return True

    def on_ws_message(self, stream, data):
        """處理WebSocket消息回調"""
        # 先調用父類處理
        super().on_ws_message(stream, data)

        # 處理訂單成交事件
        if not stream.startswith("account.orderUpdate."):
            return

        event_type = data.get('e')
        if event_type in {"orderCancel", "orderCanceled", "orderExpired", "orderReject", "orderRejected"}:
            self._handle_order_cancel(data)

    def _after_fill_processed(self, fill_info: Dict[str, Any]) -> None:
        """訂單成交後的處理（新的雙重確認機制）"""
        super()._after_fill_processed(fill_info)

        primary_id, alias_ids = self._extract_order_identifiers(fill_info)
        if primary_id and alias_ids:
            self._register_order_aliases(primary_id, alias_ids)

        # APEX 成交歷史返回: orderId (交易所ID), clientId (下單時的UUID)
        # 我們下單時用 clientId 追蹤，所以優先用 client_id 匹配
        order_id = (
            fill_info.get('order_id')
            or fill_info.get('id')
            or fill_info.get('orderId')
            or primary_id
        )
        # APEX: client_id 是下單時生成的 UUID，用於追蹤
        client_id = fill_info.get('client_id') or fill_info.get('clientId') or fill_info.get('clientOrderId')
        
        side = fill_info.get('side')
        quantity_raw = fill_info.get('quantity')
        price_raw = fill_info.get('price')

        try:
            quantity = float(quantity_raw or 0)
            price = float(price_raw or 0)
        except (TypeError, ValueError):
            return
        
        if not order_id and not client_id:
            return
        
        # 標準化方向
        normalized_side = side
        if isinstance(side, str):
            side_upper = side.upper()
            if side_upper in ('BUY', 'BID', 'LONG'):
                normalized_side = 'Bid'
            elif side_upper in ('SELL', 'ASK', 'SHORT'):
                normalized_side = 'Ask'
        
        # 調試：輸出當前追蹤的訂單
        logger.debug("成交處理: order_id=%s, client_id=%s, side=%s", order_id, client_id, normalized_side)
        logger.debug("當前追蹤的開多單數量: %d", sum(len(orders) for orders in self.open_long_orders.values()))
        logger.debug("當前追蹤的開空單數量: %d", sum(len(orders) for orders in self.open_short_orders.values()))
        
        # 嘗試用 client_id 或 order_id 解析追蹤 ID
        # APEX 下單時用 clientId (UUID) 追蹤，成交歷史返回 clientId 和 orderId
        tracking_id = None
        
        # 優先用 client_id 查找（因為我們下單時用的是 clientId）
        if client_id:
            resolved = self._resolve_order_id(client_id)
            if resolved:
                tracking_id = resolved
                logger.debug("通過 client_id 解析到 tracking_id: %s", tracking_id)
            else:
                # 直接檢查是否在追蹤列表中
                normalized_client = self._normalize_order_id(client_id)
                if normalized_client:
                    # 檢查開倉單
                    for orders in self.open_long_orders.values():
                        if normalized_client in orders:
                            tracking_id = normalized_client
                            logger.debug("在開多單列表中找到 client_id: %s", normalized_client)
                            break
                    if not tracking_id:
                        for orders in self.open_short_orders.values():
                            if normalized_client in orders:
                                tracking_id = normalized_client
                                break
                    # 檢查平倉單
                    if not tracking_id and normalized_client in self.close_orders:
                        tracking_id = normalized_client
        
        # 如果 client_id 沒找到，嘗試用 order_id
        if not tracking_id and order_id:
            resolved = self._resolve_order_id(order_id)
            if resolved:
                tracking_id = resolved
            else:
                tracking_id = self._normalize_order_id(order_id)
        
        if not tracking_id:
            logger.debug("無法解析訂單 ID: order_id=%s, client_id=%s", order_id, client_id)
            return
        
        # 判斷是開倉單還是平倉單
        if tracking_id in self.close_orders:
            # 處理平倉單成交
            self._handle_close_order_filled(tracking_id, price, normalized_side, quantity)
        else:
            # 處理開倉單成交
            self._handle_open_order_filled(
                tracking_id,
                price,
                normalized_side,
                quantity,
                raw_order_id=self._normalize_order_id(order_id),
            )





    def _handle_close_order_cancel(self, order_id: str) -> None:
        """處理平倉單被取消"""
        close_info = self._remove_close_order(order_id)
        if not close_info:
            return

        open_price = close_info['open_price']
        quantity = close_info['quantity']
        position_type = close_info.get('position_type')

        logger.warning("平倉單被取消: ID=%s, 類型=%s, 重新掛單", order_id, position_type)

        if position_type == 'long':
            self._place_close_long_order(open_price, quantity)
        else:
            self._place_close_short_order(open_price, quantity)

    def _handle_order_cancel(self, data: Dict[str, Any]) -> None:
        order_id = data.get('i')
        if not order_id:
            return

        normalized_id = self._resolve_order_id(order_id) or self._normalize_order_id(order_id)
        if not normalized_id:
            return

        if normalized_id in self.close_order_mapping:
            self._handle_close_order_cancel(normalized_id)
            return

        order_info = self.grid_orders_by_id.get(normalized_id)
        if not order_info:
            return

        price = order_info['price']
        quantity = order_info['quantity']
        grid_type = order_info.get('grid_type', 'long')

        logger.warning("網格開倉單被取消: ID=%s, 類型=%s, 價格=%.4f", normalized_id, grid_type, price)
        self._remove_grid_order(normalized_id, price, grid_type)

        if grid_type == 'long':
            self._place_grid_long_order(price, quantity)
        else:
            self._place_grid_short_order(price, quantity)

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

        self._remove_order_aliases(order_id)

    def _place_close_long_order(self, open_price: float, quantity: float, retry_count: int = 0) -> bool:
        """掛平多單
        
        Args:
            open_price: 原始掛單價格（不是成交價格）
            quantity: 成交數量
            retry_count: 當前重試次數
            
        Returns:
            是否成功掛單
        """
        # 確保數量符合交易所精度要求
        quantity = round_to_precision(quantity, self.base_precision)
        if quantity < self.min_order_size:
            logger.warning(
                "平多單數量 %.8f 小於最小訂單量 %.8f，跳過",
                quantity, self.min_order_size
            )
            return True
        
        # 避免在倉位已清空的情況下繼續掛賣單導致開空
        if self.grid_type == "long":
            net_position = self.get_net_position()
            if net_position <= 0:
                logger.warning(
                    "做多網格：當前無多頭倉位 (淨倉位: %.4f)，跳過平多單 (開倉價格: %.4f, 數量: %.4f)",
                    net_position, open_price, quantity
                )
                # 從待重試隊列中移除此訂單（如果存在）
                return True  # 返回 True 表示不需要再重試
        
        # 找到下一個更高的網格點位
        next_price = None
        for price in sorted(self.grid_levels):
            if price > open_price:
                next_price = price
                break

        if not next_price:
            # 如果沒有更高的網格點位（最高網格成交），計算一個合理的平倉價格
            # 使用網格間距作為目標利潤
            if len(self.grid_levels) >= 2:
                # 計算平均網格間距
                sorted_levels = sorted(self.grid_levels)
                grid_step = (sorted_levels[-1] - sorted_levels[0]) / (len(sorted_levels) - 1)
            else:
                # 只有一個網格點位，使用價格的 0.1% 作為間距
                grid_step = open_price * 0.001
            
            next_price = round_to_tick_size(open_price + grid_step, self.tick_size)
            logger.info(
                "開多價格 %.4f 是最高網格，使用計算的平倉價格 %.4f (間距: %.4f)",
                open_price, next_price, grid_step
            )

        logger.info("開多成交後在價格 %.4f 掛平多單 (開倉價格: %.4f)", next_price, open_price)

        # APEX 是 zkLink L2 架構，持倉更新有延遲
        # 對於網格策略，我們知道剛才開多成交了，所以直接掛平倉單
        # 不需要查詢 API 持倉（因為可能還沒更新）
        
        # 對於 APEX，由於 L2 結算延遲，我們信任本地狀態而不是 API
        # 開多成交後立即掛平多單，使用 reduce_only=False 避免因 API 延遲導致失敗
        reduce_only = False
        
        # 對於其他交易所，可以嘗試使用 reduce_only
        if self.exchange not in ('apex',):
            # 延遲一下，等待持倉更新
            time.sleep(0.5)
            net_position = self.get_net_position()
            logger.debug("當前淨持倉: %.4f, 檢查是否可以掛平多單", net_position)
            
            if net_position >= quantity * 0.9:
                reduce_only = True
                logger.debug("持倉足夠，使用 reduce_only 掛平倉單")
            elif self.grid_type == "long" and net_position <= 0:
                # 做多網格且無多頭倉位，直接跳過
                logger.warning(
                    "做多網格：多頭持倉不足且已無倉位 (當前: %.4f)，跳過平多單",
                    net_position
                )
                return True
            else:
                # 對於中性網格或仍有部分倉位的情況，使用 reduce_only=True
                reduce_only = True
                logger.warning(
                    "多頭持倉不足 (當前: %.4f, 需要: %.4f)，仍使用 reduce_only=True 以避免開新倉",
                    net_position, quantity
                )

        # 掛平倉單（使用 post_only 確保只能作為 Maker 成交）
        result = self.open_short(
            quantity=quantity,
            price=next_price,
            order_type="Limit",
            reduce_only=reduce_only,
            post_only=True  # 強制 Post-Only
        )

        if isinstance(result, dict) and "error" in result:
            error_msg = result.get('error', '')
            logger.error("掛平多單失敗: %s", error_msg)

            # 檢查是否是 post_only 導致的錯誤（價格會立即成交）
            is_post_only_error = "immediately match" in error_msg.lower() or "would cross" in error_msg.lower()
            
            # 如果是 reduce-only 錯誤，加入待重試隊列，等待倉位更新後重試
            if "Reduce only" in error_msg and reduce_only:
                logger.warning("reduce_only 報錯，持倉可能還沒更新，加入待重試隊列...")
                self._add_pending_close_order(open_price, quantity, 'long', retry_count)
                return False
            # 如果是 post_only 錯誤，重試不使用 post_only（接受 taker 成交）
            elif is_post_only_error:
                logger.warning("平倉價格會立即成交，重試不使用 post_only（將作為 taker 成交）...")
                time.sleep(0.3)
                result = self.open_short(
                    quantity=quantity,
                    price=next_price,
                    order_type="Limit",
                    reduce_only=reduce_only,
                    post_only=False  # 允許 taker 成交
                )
                if isinstance(result, dict) and "error" in result:
                    logger.error("重試仍失敗: %s", result.get('error', ''))
                    self._add_pending_close_order(open_price, quantity, 'long', retry_count)
                    return False
            else:
                # 其他錯誤，加入待重試隊列
                self._add_pending_close_order(open_price, quantity, 'long', retry_count)
                return False

        # 使用新的記錄系統記錄平倉單
        primary_id, alias_ids = self._extract_order_identifiers(result)
        order_id = primary_id or result.get('id')
        if order_id:
            self._record_close_order(order_id, open_price, quantity, 'long', aliases=alias_ids)
            
            # 計算潛在網格利潤
            grid_profit = (next_price - open_price) * quantity
            logger.info("掛出平多單，潛在利潤: %.4f %s", grid_profit, self.quote_asset)
            return True
        
        return False

    def _place_close_short_order(self, open_price: float, quantity: float, retry_count: int = 0) -> bool:
        """掛平空單
        
        Args:
            open_price: 原始掛單價格（不是成交價格）
            quantity: 成交數量
            retry_count: 當前重試次數
            
        Returns:
            是否成功掛單
        """
        # 確保數量符合交易所精度要求
        quantity = round_to_precision(quantity, self.base_precision)
        if quantity < self.min_order_size:
            logger.warning(
                "平空單數量 %.8f 小於最小訂單量 %.8f，跳過",
                quantity, self.min_order_size
            )
            return True
        
        # 對於做空網格，如果當前沒有空頭倉位，則不應該掛平空單
        # 避免在倉位已清空的情況下繼續掛買單導致開多
        if self.grid_type == "short":
            net_position = self.get_net_position()
            if net_position >= 0:
                logger.warning(
                    "做空網格：當前無空頭倉位 (淨倉位: %.4f)，跳過平空單 (開倉價格: %.4f, 數量: %.4f)",
                    net_position, open_price, quantity
                )
                # 從待重試隊列中移除此訂單（如果存在）
                return True  # 返回 True 表示不需要再重試
        
        # 找到下一個更低的網格點位
        next_price = None
        for price in sorted(self.grid_levels, reverse=True):
            if price < open_price:
                next_price = price
                break

        if not next_price:
            # 如果沒有更低的網格點位（最低網格成交），計算一個合理的平倉價格
            # 使用網格間距作為目標利潤
            if len(self.grid_levels) >= 2:
                # 計算平均網格間距
                sorted_levels = sorted(self.grid_levels)
                grid_step = (sorted_levels[-1] - sorted_levels[0]) / (len(sorted_levels) - 1)
            else:
                # 只有一個網格點位，使用價格的 0.1% 作為間距
                grid_step = open_price * 0.001
            
            next_price = round_to_tick_size(open_price - grid_step, self.tick_size)
            logger.info(
                "開空價格 %.4f 是最低網格，使用計算的平倉價格 %.4f (間距: %.4f)",
                open_price, next_price, grid_step
            )

        logger.info("開空成交後在價格 %.4f 掛平空單 (開倉價格: %.4f)", next_price, open_price)

        # APEX 是 zkLink L2 架構，持倉更新有延遲
        # 對於網格策略，我們知道剛才開空成交了，所以直接掛平倉單
        # 不需要查詢 API 持倉（因為可能還沒更新）
        
        # 對於 APEX，由於 L2 結算延遲，我們信任本地狀態而不是 API
        reduce_only = False
        
        # 對於其他交易所，可以嘗試使用 reduce_only
        if self.exchange not in ('apex',):
            # 延遲一下，等待持倉更新
            time.sleep(0.5)
            net_position = self.get_net_position()
            logger.debug("當前淨持倉: %.4f, 檢查是否可以掛平空單", net_position)
            
            if net_position <= -quantity * 0.9:
                reduce_only = True
                logger.debug("持倉足夠，使用 reduce_only 掛平倉單")
            elif self.grid_type == "short" and net_position >= 0:
                # 做空網格且無空頭倉位，直接跳過
                logger.warning(
                    "做空網格：空頭持倉不足且已無倉位 (當前: %.4f)，跳過平空單",
                    net_position
                )
                return True
            else:
                # 對於中性網格或仍有部分倉位的情況，使用 reduce_only=True
                reduce_only = True
                logger.warning(
                    "空頭持倉不足 (當前: %.4f, 需要: %.4f)，仍使用 reduce_only=True",
                    net_position, -quantity
                )

        # 掛平倉單（使用 post_only 確保只能作為 Maker 成交）
        result = self.open_long(
            quantity=quantity,
            price=next_price,
            order_type="Limit",
            reduce_only=reduce_only,
            post_only=True  # 強制 Post-Only
        )

        if isinstance(result, dict) and "error" in result:
            error_msg = result.get('error', '')
            logger.error("掛平空單失敗: %s", error_msg)

            # 檢查是否是 post_only 導致的錯誤（價格會立即成交）
            is_post_only_error = "immediately match" in error_msg.lower() or "would cross" in error_msg.lower()
            
            # 如果是 reduce-only 錯誤，加入待重試隊列，等待倉位更新後重試
            if "Reduce only" in error_msg and reduce_only:
                logger.warning("reduce_only 報錯，持倉可能還沒更新，加入待重試隊列...")
                self._add_pending_close_order(open_price, quantity, 'short', retry_count)
                return False
            # 如果是 post_only 錯誤，重試不使用 post_only（接受 taker 成交）
            elif is_post_only_error:
                logger.warning("平倉價格會立即成交，重試不使用 post_only（將作為 taker 成交）...")
                time.sleep(0.3)
                result = self.open_long(
                    quantity=quantity,
                    price=next_price,
                    order_type="Limit",
                    reduce_only=reduce_only,
                    post_only=False  # 允許 taker 成交
                )
                if isinstance(result, dict) and "error" in result:
                    logger.error("重試仍失敗: %s", result.get('error', ''))
                    self._add_pending_close_order(open_price, quantity, 'short', retry_count)
                    return False
            else:
                # 其他錯誤，加入待重試隊列
                self._add_pending_close_order(open_price, quantity, 'short', retry_count)
                return False

        # 使用新的記錄系統記錄平倉單
        primary_id, alias_ids = self._extract_order_identifiers(result)
        order_id = primary_id or result.get('id')
        if order_id:
            self._record_close_order(order_id, open_price, quantity, 'short', aliases=alias_ids)
            
            # 計算潛在網格利潤
            grid_profit = (open_price - next_price) * quantity
            logger.info("掛出平空單，潛在利潤: %.4f %s", grid_profit, self.quote_asset)
            return True
        
        return False
    
    def _add_pending_close_order(self, open_price: float, quantity: float, position_type: str, current_retry: int) -> None:
        """將失敗的平倉單加入待重試隊列
        
        Args:
            open_price: 開倉價格
            quantity: 數量
            position_type: 'long' 或 'short'
            current_retry: 當前重試次數
        """
        # 確保數量符合交易所精度要求
        quantity = round_to_precision(quantity, self.base_precision)
        if quantity < self.min_order_size:
            logger.warning(
                "待重試平倉單數量 %.8f 小於最小訂單量 %.8f，跳過",
                quantity, self.min_order_size
            )
            return
        
        next_retry = current_retry + 1
        if next_retry > self.max_close_order_retries:
            logger.warning(
                "平倉單已達最大限價重試次數 (%d)，嘗試市價強制平倉: 開倉價格=%.4f, 數量=%.4f, 類型=%s",
                self.max_close_order_retries, open_price, quantity, position_type
            )
            
            # 嘗試市價強制平倉 - 使用繼承的 close_position 方法
            try:
                # position_type='long' 表示要平多倉（賣出），'short' 表示要平空倉（買入）
                close_success = self.close_position(
                    quantity=quantity,
                    order_type="Market",
                    side=position_type,  # 'long' 或 'short'
                )
                
                if close_success:
                    logger.warning(
                        "市價強制平倉成功: 開倉價格=%.4f, 數量=%.4f, 類型=%s",
                        open_price, quantity, position_type
                    )
                    # 更新網格狀態
                    state = self.grid_level_states[open_price]
                    if position_type == 'long':
                        state['open_position'] = max(0, state['open_position'] - quantity)
                    else:
                        state['open_position'] = min(0, state['open_position'] + quantity)
                    
                    if abs(state['open_position']) < self.min_order_size:
                        state['locked'] = False
                        state['open_position'] = 0.0
                        logger.info("網格點位 %.4f 已解除鎖定", open_price)
                    return
                else:
                    logger.error("市價強制平倉失敗")
            except Exception as e:
                logger.error("市價強制平倉出錯: %s", e)
            
            # 市價平倉也失敗，重新加入隊列，下一輪繼續嘗試市價平倉
            logger.error(
                "*** 警告 ***: 市價平倉失敗! 開倉價格=%.4f, 數量=%.4f, 類型=%s，下一輪將繼續嘗試",
                open_price, quantity, position_type
            )
            # 保持在最大重試次數，這樣下一輪會繼續嘗試市價平倉
            self.pending_close_orders.append((open_price, quantity, position_type, self.max_close_order_retries))
            return
            
            # 市價平倉也失敗，重新加入隊列，下一輪繼續嘗試市價平倉
            logger.error(
                "*** 警告 ***: 市價平倉失敗! 開倉價格=%.4f, 數量=%.4f, 類型=%s，下一輪將繼續嘗試",
                open_price, quantity, position_type
            )
            # 保持在最大重試次數，這樣下一輪會繼續嘗試市價平倉
            self.pending_close_orders.append((open_price, quantity, position_type, self.max_close_order_retries))
            return
        
        # 檢查是否已在隊列中（避免重複添加）
        for pending in self.pending_close_orders:
            if pending[0] == open_price and pending[2] == position_type:
                logger.debug("平倉單已在隊列中，更新重試次數: %.4f %s", open_price, position_type)
                return
        
        self.pending_close_orders.append((open_price, quantity, position_type, next_retry))
        logger.warning(
            "平倉單加入重試隊列: 開倉價格=%.4f, 數量=%.4f, 類型=%s, 重試次數=%d/%d",
            open_price, quantity, position_type, next_retry, self.max_close_order_retries
        )
    
    def _retry_pending_close_orders(self) -> None:
        """重試待處理的平倉單"""
        if not self.pending_close_orders:
            return
        
        logger.info("開始重試 %d 個待處理的平倉單...", len(self.pending_close_orders))
        
        # 複製列表，因為處理過程中會修改
        pending_orders = list(self.pending_close_orders)
        self.pending_close_orders.clear()
        
        for open_price, quantity, position_type, retry_count in pending_orders:
            logger.info(
                "重試平倉單: 開倉價格=%.4f, 數量=%.4f, 類型=%s, 重試次數=%d",
                open_price, quantity, position_type, retry_count
            )
            
            if position_type == 'long':
                self._place_close_long_order(open_price, quantity, retry_count)
            else:
                self._place_close_short_order(open_price, quantity, retry_count)

    def _periodic_position_sync(self) -> None:
        """週期性檢查並同步倉位與平倉單。
        
        每隔 _position_sync_interval 秒執行一次，
        檢查目前持倉是否與追蹤的平倉單數量一致，如不一致則補掛平倉單。
        """
        current_time = time.time()
        
        # 檢查是否達到同步間隔
        if current_time - self._last_position_sync_time < self._position_sync_interval:
            return
        
        self._last_position_sync_time = current_time
        
        logger.info("=== 開始週期性倉位同步檢查 ===")
        
        # 獲取當前實際持倉
        position_state = self.get_position_state()
        current_position = float(position_state.get('net', 0.0))
        
        # 計算追蹤的平倉單總量
        tracked_close_qty = sum(
            abs(info.get('quantity', 0.0))
            for info in self.close_orders.values()
        )
        
        logger.info(
            "倉位同步: 實際持倉=%.4f, 追蹤平倉單量=%.4f",
            current_position, tracked_close_qty
        )
        
        # 計算差異
        if current_position > 0:
            # 多頭持倉，應有等量的平多單
            expected_close_qty = current_position
            gap = abs(current_position) - tracked_close_qty
        elif current_position < 0:
            # 空頭持倉，應有等量的平空單
            expected_close_qty = abs(current_position)
            gap = abs(current_position) - tracked_close_qty
        else:
            # 無持倉
            expected_close_qty = 0.0
            gap = 0.0
        
        # 如果差異超過閾值（半個訂單量），進行同步
        threshold = self.order_quantity * 0.5 if self.order_quantity else 1.0
        if gap > threshold:
            logger.warning(
                "偵測到倉位不同步！持倉=%.4f, 平倉單覆蓋=%.4f, 缺口=%.4f, 觸發同步",
                current_position, tracked_close_qty, gap
            )
            # 調用完整的倉位同步
            self._sync_position_with_grid()
        else:
            logger.info("倉位同步檢查完成，無異常")

    def place_limit_orders(self) -> None:
        """放置限價單 - 覆蓋父類方法（增加倉位變化檢測）"""
        if not self.grid_initialized:
            success = self.initialize_grid()
            if not success:
                logger.error("網格初始化失敗")
                return
            # 初始化後記錄持倉快照
            self.last_position_snapshot = self.get_net_position()
            # 記錄初始同步時間
            self._last_position_sync_time = time.time()
        else:
            # 【新增】倉位變化檢測，補強訂單狀態的遺漏
            self._check_position_changes()
            
            # 【新增】重試失敗的平倉單
            self._retry_pending_close_orders()
            
            # 【新增】週期性倉位同步檢查
            self._periodic_position_sync()
            
            # 一次性獲取所有必要數據，然後檢查並補充缺失的網格訂單
            current_price = self.get_current_price()
            try:
                open_orders = self.client.get_open_orders(self.symbol) or []
            except Exception as exc:
                logger.warning("無法獲取現有訂單: %s", exc)
                open_orders = []
            
            # 傳遞數據給 _refill_grid_orders，避免重複請求
            self._refill_grid_orders(current_price=current_price, 
                                    open_orders=open_orders)

    def _reconcile_grid_orders_from_list(self, open_orders: List) -> Dict[str, Dict[float, int]]:
        """統計目前在交易所實際掛出的開倉單數量，從訂單列表中統計。
        
        使用新的追蹤系統，只統計真正的開倉單。
        
        Args:
            open_orders: 現有訂單列表
            
        Returns:
            {'Bid': {price: count}, 'Ask': {price: count}}
        """
        long_open_counts: Dict[float, int] = defaultdict(int)  # 開多單
        short_open_counts: Dict[float, int] = defaultdict(int)  # 開空單

        if isinstance(open_orders, dict) and open_orders.get('error'):
            logger.warning("訂單列表包含錯誤: %s", open_orders['error'])
            return {'Bid': {}, 'Ask': {}}

        for order in open_orders:
            if not isinstance(order, dict):
                continue

            order_id = order.get('id') or order.get('orderId') or order.get('order_id')
            if not order_id:
                continue
            
            # 使用新的追蹤系統判斷是否為平倉單
            if str(order_id) in self.close_orders:
                logger.debug("訂單 %s 是平倉單，不計入開倉單統計", order_id)
                continue
            
            # 檢查是否標記為 reduce_only（平倉單）
            reduce_only = order.get('reduceOnly', False) or order.get('reduce_only', False)
            if reduce_only:
                logger.debug("訂單 %s 標記為 reduce_only，不計入開倉單統計", order_id)
                continue

            side_raw = str(order.get('side', '')).upper()
            price_raw = order.get('price')
            if price_raw is None:
                continue

            try:
                price = round_to_tick_size(float(price_raw), self.tick_size)
            except (TypeError, ValueError):
                continue

            # 只統計開倉單
            if side_raw in ('BUY', 'BID', 'LONG'):
                long_open_counts[price] += 1
                logger.debug("統計到開多單: 價格=%.4f, 訂單ID=%s", price, order_id)
            elif side_raw in ('SELL', 'ASK', 'SHORT'):
                short_open_counts[price] += 1
                logger.debug("統計到開空單: 價格=%.4f, 訂單ID=%s", price, order_id)

        return {
            'Bid': dict(long_open_counts),
            'Ask': dict(short_open_counts),
        }

    def _refill_grid_orders(self, current_price: Optional[float] = None,
                            open_orders: Optional[List] = None) -> None:
        """補充缺失的網格訂單（使用新的狀態管理系統）
        
        確認邏輯：
        1. 只統計開倉單（非 reduce_only）
        2. 檢查網格點位狀態（是否已鎖定）
        3. 根據持倉情況決定是否需要補單
        
        Args:
            current_price: 當前價格（如果未提供則會請求一次）
            open_orders: 現有訂單列表（如果未提供則會請求一次）
        """
        # 只在未提供數據時才請求
        if current_price is None:
            current_price = self.get_current_price()
            if not current_price:
                return

        if open_orders is None:
            try:
                open_orders = self.client.get_open_orders(self.symbol) or []
            except Exception as exc:
                logger.warning("無法獲取現有訂單: %s", exc)
                return

        # 統計開倉單（不包括平倉單）
        active_counts = self._reconcile_grid_orders_from_list(open_orders)
        active_long_open_counts = active_counts.get('Bid', {})
        active_short_open_counts = active_counts.get('Ask', {})
        
        # 獲取當前持倉（用於驗證補單邏輯）
        net_position = self.get_net_position()
        logger.debug("當前淨持倉: %.4f, 開始檢查網格訂單...", net_position)

        refilled = 0
        skipped_locked = 0
        skipped_has_order = 0
        
        # 追蹤本次迭代已補單的價格，避免重複補單
        filled_prices_this_round: Set[float] = set()

        for price in self.grid_levels:
            # 獲取該網格點位的狀態
            level_state = self.grid_level_states[price]
            is_locked = level_state['locked']
            
            if self.grid_type == "neutral":
                if price < current_price:
                    # 檢查是否已有開多單（交易所 API + 本地追蹤 + 本輪已補）
                    has_long_order = (
                        active_long_open_counts.get(price, 0) > 0 or
                        len(self.open_long_orders.get(price, {})) > 0 or
                        price in filled_prices_this_round
                    )
                    
                    if has_long_order:
                        logger.debug("網格點位 %.4f 已有開多單，不補單", price)
                        skipped_has_order += 1
                        continue
                    
                    # 檢查是否已鎖定（有平倉單掛出）
                    if is_locked:
                        logger.debug(
                            "網格點位 %.4f 已鎖定（持倉=%.4f, 平倉單數=%d），不補開多單",
                            price, level_state['open_position'], len(level_state['close_order_ids'])
                        )
                        skipped_locked += 1
                        continue
                    
                    # 需要補開多單
                    logger.info("補充開多單: 價格=%.4f, 數量=%.4f", price, self.order_quantity)
                    if self._place_grid_long_order(price, self.order_quantity):
                        refilled += 1
                        filled_prices_this_round.add(price)
                        
                elif price > current_price:
                    # 檢查是否已有開空單（交易所 API + 本地追蹤 + 本輪已補）
                    has_short_order = (
                        active_short_open_counts.get(price, 0) > 0 or
                        len(self.open_short_orders.get(price, {})) > 0 or
                        price in filled_prices_this_round
                    )
                    
                    if has_short_order:
                        logger.debug("網格點位 %.4f 已有開空單，不補單", price)
                        skipped_has_order += 1
                        continue
                    
                    # 檢查是否已鎖定（有平倉單掛出）
                    if is_locked:
                        logger.debug(
                            "網格點位 %.4f 已鎖定（持倉=%.4f, 平倉單數=%d），不補開空單",
                            price, level_state['open_position'], len(level_state['close_order_ids'])
                        )
                        skipped_locked += 1
                        continue
                    
                    # 需要補開空單
                    logger.info("補充開空單: 價格=%.4f, 數量=%.4f", price, self.order_quantity)
                    if self._place_grid_short_order(price, self.order_quantity):
                        refilled += 1
                        filled_prices_this_round.add(price)

            elif self.grid_type == "long":
                if price <= current_price:
                    # 檢查是否已有開多單（交易所 API + 本地追蹤 + 本輪已補）
                    has_long_order = (
                        active_long_open_counts.get(price, 0) > 0 or
                        len(self.open_long_orders.get(price, {})) > 0 or
                        price in filled_prices_this_round
                    )
                    
                    if has_long_order:
                        logger.debug("網格點位 %.4f 已有開多單，不補單", price)
                        skipped_has_order += 1
                        continue
                    
                    # 檢查是否已鎖定（有平倉單掛出）
                    if is_locked:
                        logger.debug(
                            "網格點位 %.4f 已鎖定（持倉=%.4f, 平倉單數=%d），不補開多單",
                            price, level_state['open_position'], len(level_state['close_order_ids'])
                        )
                        skipped_locked += 1
                        continue
                    
                    # 需要補開多單
                    logger.info("補充開多單: 價格=%.4f, 數量=%.4f", price, self.order_quantity)
                    if self._place_grid_long_order(price, self.order_quantity):
                        refilled += 1
                        filled_prices_this_round.add(price)

            elif self.grid_type == "short":
                if price >= current_price:
                    # 檢查是否已有開空單（交易所 API + 本地追蹤 + 本輪已補）
                    has_short_order = (
                        active_short_open_counts.get(price, 0) > 0 or
                        len(self.open_short_orders.get(price, {})) > 0 or
                        price in filled_prices_this_round
                    )
                    
                    if has_short_order:
                        logger.debug("網格點位 %.4f 已有開空單，不補單", price)
                        skipped_has_order += 1
                        continue
                    
                    # 檢查是否已鎖定（有平倉單掛出）
                    if is_locked:
                        logger.debug(
                            "網格點位 %.4f 已鎖定（持倉=%.4f, 平倉單數=%d），不補開空單",
                            price, level_state['open_position'], len(level_state['close_order_ids'])
                        )
                        skipped_locked += 1
                        continue
                    
                    # 需要補開空單
                    logger.info("補充開空單: 價格=%.4f, 數量=%.4f", price, self.order_quantity)
                    if self._place_grid_short_order(price, self.order_quantity):
                        refilled += 1
                        filled_prices_this_round.add(price)

        # 補單摘要
        if refilled > 0 or skipped_locked > 0:
            # 統計當前開倉單總數
            total_open_orders = sum(len(orders) for orders in self.open_long_orders.values())
            total_open_orders += sum(len(orders) for orders in self.open_short_orders.values())
            
            logger.info(
                "補單檢查完成: 補充了 %d 個訂單, 跳過 %d 個已鎖定點位, 跳過 %d 個已有訂單點位 (總開倉單: %d, 總平倉單: %d)",
                refilled, skipped_locked, skipped_has_order, total_open_orders, len(self.close_orders)
            )

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

        buy_fill_count = len(self.session_buy_trades)
        sell_fill_count = len(self.session_sell_trades)

        # 統計所有掛單數量（開倉單 + 平倉單）
        # 開多單 + 平空單 = 買單總數
        # 開空單 + 平多單 = 賣單總數
        total_buy_orders = len(self.active_buy_orders) if hasattr(self, 'active_buy_orders') else 0
        total_sell_orders = len(self.active_sell_orders) if hasattr(self, 'active_sell_orders') else 0
        locked_levels = sum(1 for state in self.grid_level_states.values() if state.get('locked', False))

        sections.append((
            "永續合約網格統計",
            [
                ("網格數量", f"{len(self.grid_levels)}"),
                ("價格範圍", f"{self.grid_lower_price:.4f} ~ {self.grid_upper_price:.4f}"),
                ("網格模式", self.grid_mode),
                ("網格類型", self.grid_type),
                ("買入次數", f"{buy_fill_count}"),
                ("賣出次數", f"{sell_fill_count}"),
                ("網格利潤", f"{self.grid_profit:.4f} {self.quote_asset}"),
                ("多單數量", f"{total_buy_orders}"),
                ("空單數量", f"{total_sell_orders}"),
                ("鎖定網格數", f"{locked_levels}"),
                ("待重試平倉單", f"{len(self.pending_close_orders)}"),
            ],
        ))

        return sections

    def check_stop_conditions(self, realized_pnl, unrealized_pnl, session_realized_pnl) -> bool:
        """覆寫止損止盈檢查，在平倉成功後清理網格狀態      """
        # 調用父類的止損止盈檢查
        result = super().check_stop_conditions(realized_pnl, unrealized_pnl, session_realized_pnl)
        
        # 檢查是否有觸發止損/止盈（通過檢查 last_protective_action）
        if hasattr(self, 'last_protective_action') and self.last_protective_action:
            if "已執行緊急平倉" in self.last_protective_action:
                logger.warning("止損/止盈平倉完成，清理網格狀態...")
                
                # 清理所有網格追蹤狀態
                self._reset_grid_state()
                
                # 清理待重試的平倉單
                self.pending_close_orders.clear()
                
                # 重置網格初始化狀態，讓下次迭代重新初始化
                self.grid_initialized = False
                
                # 清空 last_protective_action 避免重複觸發
                self.last_protective_action = None
                
                logger.info("網格狀態已清理，下次迭代將重新初始化網格")
        
        return result

    def run(self, duration_seconds=3600, interval_seconds=60):
        """運行永續合約網格交易策略"""
        logger.info("開始運行永續合約網格交易策略: %s", self.symbol)
        logger.info("網格參數: 上限=%.4f, 下限=%.4f, 數量=%d, 類型=%s",
                   self.grid_upper_price or 0, self.grid_lower_price or 0,
                   self.grid_num, self.grid_type)

        # 調用父類的run方法
        super().run(duration_seconds, interval_seconds)
