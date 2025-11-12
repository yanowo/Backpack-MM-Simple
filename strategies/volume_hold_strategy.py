"""Multi-account volume & holding time strategy dedicated to Lighter."""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from api.lighter_client import LighterClient
from logger import setup_logger
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("volume_hold_strategy")


class StrategyConfigError(ValueError):
    """Raised when the strategy configuration is invalid."""


@dataclass
class AccountCredentials:
    """Holds a single account configuration."""

    label: str
    api_private_key: str
    account_index: int
    api_key_index: int = 0
    base_url: Optional[str] = None
    chain_id: Optional[int] = None
    signer_lib_dir: Optional[str] = None

    def as_client_config(self, defaults: "VolumeHoldStrategyConfig") -> Dict[str, Any]:
        base_url = self.base_url or defaults.base_url
        if not base_url:
            raise StrategyConfigError(f"Base URL missing for account {self.label}")
        config: Dict[str, Any] = {
            "base_url": base_url,
            "api_private_key": self.api_private_key,
            "account_index": self.account_index,
            "api_key_index": self.api_key_index,
        }
        if self.chain_id or defaults.chain_id:
            config["chain_id"] = self.chain_id or defaults.chain_id
        signer_dir = self.signer_lib_dir or defaults.signer_lib_dir
        if signer_dir:
            config["signer_lib_dir"] = signer_dir
        return config


@dataclass
class SymbolPlan:
    """Per-symbol configuration."""

    symbol: str
    target_notional: float
    slice_notional: float
    entry_offset_bps: Optional[float] = None
    exit_offset_bps: Optional[float] = None
    hold_minutes: Optional[float] = None


@dataclass
class MarketConstraints:
    """Normalized market metadata."""

    base_precision: int
    quote_precision: int
    min_order_size: float
    tick_size: float
    min_quote_value: float = 10.0


@dataclass
class VolumeHoldStrategyConfig:
    """Runtime configuration for the volume hold strategy."""

    accounts: List[AccountCredentials]
    symbols: List[SymbolPlan]
    base_url: Optional[str] = None
    chain_id: Optional[int] = None
    signer_lib_dir: Optional[str] = None
    hold_minutes: float = 17.0
    entry_price_offset_bps: float = 5.0
    exit_price_offset_bps: float = 5.0
    slice_delay_seconds: float = 6.0
    slice_delay_jitter_seconds: float = 3.0
    slice_fill_timeout: float = 120.0
    order_poll_interval: float = 2.0
    post_only: bool = True
    pause_between_symbols: float = 30.0
    random_split_range: Tuple[float, float] = (0.45, 0.55)
    run_once: bool = False
    enable_hedge: bool = True
    primary_time_in_force: str = "GTC"

    @classmethod
    def from_file(cls, path: str) -> "VolumeHoldStrategyConfig":
        if not os.path.isfile(path):
            raise StrategyConfigError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VolumeHoldStrategyConfig":
        base_url = payload.get("base_url") or os.getenv("LIGHTER_BASE_URL")
        chain_id = payload.get("chain_id") or os.getenv("LIGHTER_CHAIN_ID")
        signer_lib_dir = payload.get("signer_lib_dir")
        default_target_notional = float(payload.get("default_target_notional", 5000))
        default_slice_count = max(1, int(payload.get("default_slice_count", 50)))

        accounts_payload = payload.get("accounts") or []
        if len(accounts_payload) < 3:
            raise StrategyConfigError("At least three Lighter accounts are required.")
        accounts: List[AccountCredentials] = []
        for index, entry in enumerate(accounts_payload):
            label = entry.get("label") or f"Acct-{index + 1}"
            api_private_key = entry.get("api_private_key") or entry.get("private_key")
            if not api_private_key:
                raise StrategyConfigError(f"Missing api_private_key for account {label}")
            account_index = entry.get("account_index")
            if account_index is None:
                account_address = entry.get('account_address')
                from api.lighter_client import _get_lihgter_account_index
                account_index = _get_lihgter_account_index(account_address)
            cred = AccountCredentials(
                label=label,
                api_private_key=str(api_private_key),
                account_index=int(account_index),
                api_key_index=int(entry.get("api_key_index") or 0),
                base_url=entry.get("base_url") or base_url,
                chain_id=int(entry.get("chain_id") or chain_id) if entry.get("chain_id") or chain_id else None,
                signer_lib_dir=entry.get("signer_lib_dir") or signer_lib_dir,
            )
            accounts.append(cred)

        symbol_payload = payload.get("coinlist") or payload.get("symbols") or payload.get("coins")
        if not symbol_payload:
            raise StrategyConfigError("coinlist/symbols configuration is required.")

        symbols: List[SymbolPlan] = []
        for entry in symbol_payload:
            symbol = entry.get("symbol")
            if not symbol:
                raise StrategyConfigError("Each coin entry must include symbol.")
            target_notional = float(entry.get("target_notional") or entry.get("target") or default_target_notional)
            slice_notional = entry.get("slice_notional")
            if slice_notional is None:
                slice_count = int(entry.get("slice_count") or default_slice_count)
                slice_notional = target_notional / max(slice_count, 1)
            slice_notional = float(slice_notional)
            symbols.append(
                SymbolPlan(
                    symbol=str(symbol),
                    target_notional=target_notional,
                    slice_notional=slice_notional,
                    entry_offset_bps=entry.get("entry_offset_bps"),
                    exit_offset_bps=entry.get("exit_offset_bps"),
                    hold_minutes=entry.get("hold_minutes"),
                )
            )

        random_split_range = payload.get("random_split_range") or payload.get("hedge_random_split") or [0.45, 0.55]
        if not isinstance(random_split_range, (list, tuple)) or len(random_split_range) != 2:
            raise StrategyConfigError("random_split_range must be a two-element list.")
        low, high = float(random_split_range[0]), float(random_split_range[1])
        if not 0 < low < high < 1:
            raise StrategyConfigError("random_split_range values must be between 0 and 1.")

        config = cls(
            accounts=accounts,
            symbols=symbols,
            base_url=base_url,
            chain_id=int(chain_id) if chain_id else None,
            signer_lib_dir=signer_lib_dir,
            hold_minutes=float(payload.get("hold_minutes", 17)),
            entry_price_offset_bps=float(payload.get("entry_price_offset_bps", 5)),
            exit_price_offset_bps=float(payload.get("exit_price_offset_bps", 5)),
            slice_delay_seconds=float(payload.get("slice_delay_seconds", 6)),
            slice_delay_jitter_seconds=float(payload.get("slice_delay_jitter_seconds", 3)),
            slice_fill_timeout=float(payload.get("slice_fill_timeout", 120)),
            order_poll_interval=float(payload.get("order_poll_interval", 2)),
            post_only=bool(payload.get("post_only", True)),
            pause_between_symbols=float(payload.get("pause_between_symbols", 30)),
            random_split_range=(low, high),
            run_once=bool(payload.get("run_once", False)),
            enable_hedge=bool(payload.get("enable_hedge", True)),
            primary_time_in_force=str(payload.get("primary_time_in_force", "GTC")),
        )
        if not config.base_url and any(acc.base_url is None for acc in accounts):
            raise StrategyConfigError("base_url missing; configure global base_url or per-account base_url entries.")
        return config


class VolumeHoldStrategy:
    """Implements the multi-account volume/holding strategy."""

    def __init__(self, config: VolumeHoldStrategyConfig, *, random_seed: Optional[int] = None) -> None:
        self.config = config
        self._random = random.Random(random_seed)
        self._clients = [LighterClient(acc.as_client_config(config)) for acc in config.accounts]
        self._account_labels = [acc.label for acc in config.accounts]
        self._market_cache: Dict[str, MarketConstraints] = {}
        self._stop_event = threading.Event()
        self._current_symbol_index = 0
        self._current_primary_index = 0
        self._hedge_backlog: Dict[int, Dict[str, Dict[str, float]]] = {
            idx: {} for idx in range(len(self._clients))
        }
        self._hedge_cycle_positions: Dict[str, Dict[int, float]] = {}
        self._theoretical_positions: Dict[str, Dict[int, float]] = {}
        self._maker_price_stats: Dict[str, Dict[str, float]] = {}
        self._hedge_price_stats: Dict[str, Dict[int, Dict[str, float]]] = {}
        self._hedge_recorded_trades: List[Set[str]] = [set() for _ in self._clients]
        self._wear_cumulative = {"wear_notional": 0.0, "maker_qty": 0.0}
        self._hedge_enabled = bool(config.enable_hedge)
        self._primary_time_in_force = str(config.primary_time_in_force or "GTC").upper()

    # ------------------------------------------------------------------ lifecycle
    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("VolumeHold strategy booted with %d symbols", len(self.config.symbols))
        cycles_completed = 0
        try:
            while not self._stop_event.is_set():
                symbol_plan = self.config.symbols[self._current_symbol_index]
                primary_idx = self._current_primary_index
                hedger_indices = self._resolve_hedgers(primary_idx)
                logger.info(
                    "Starting cycle: symbol=%s primary=%s hedgers=%s target=%s slice=%s",
                    symbol_plan.symbol,
                    self._account_labels[primary_idx],
                    [self._account_labels[i] for i in hedger_indices],
                    symbol_plan.target_notional,
                    symbol_plan.slice_notional,
                )

                try:
                    entry_summary = self._accumulate_position(primary_idx, hedger_indices, symbol_plan)
                    if entry_summary <= 0:
                        logger.warning("No fills recorded for %s, skipping exit leg", symbol_plan.symbol)
                    else:
                        self._hold_position(symbol_plan)
                        self._flatten_position(primary_idx, hedger_indices, symbol_plan, entry_summary)
                        self._print_cycle_summary(symbol_plan.symbol)
                except Exception:
                    logger.exception("Cycle for %s failed", symbol_plan.symbol)

                cycles_completed += 1
                if self.config.run_once and cycles_completed >= len(self.config.symbols):
                    logger.info("run_once enabled, stopping after one pass.")
                    break

                self._advance_pointers()
                self._sleep_with_stop(self.config.pause_between_symbols)
        except KeyboardInterrupt:
            logger.info("Interrupted by user, shutting down VolumeHold strategy...")
        finally:
            self.stop()

    # ------------------------------------------------------------------ core flow
    def _accumulate_position(self, primary_idx: int, hedgers: Sequence[int], plan: SymbolPlan) -> float:
        """Execute entry leg for a symbol; returns total filled base quantity."""
        client = self._clients[primary_idx]
        limits = self._get_market_limits(client, plan.symbol)
        target_remaining = float(plan.target_notional)
        total_base_filled = 0.0

        while not self._stop_event.is_set() and target_remaining > 0:
            slice_notional = min(plan.slice_notional, target_remaining)
            reference_price = self._compute_reference_price(client, plan.symbol, "Bid", plan.entry_offset_bps)
            if reference_price is None or reference_price <= 0:
                logger.warning("Unable to compute entry price for %s, retrying...", plan.symbol)
                self._sleep_with_stop(3)
                continue

            slice_quantity = self._notional_to_quantity(slice_notional, reference_price, limits)
            if slice_quantity < limits.min_order_size:
                required_notional = limits.min_order_size * reference_price
                if required_notional <= target_remaining:
                    logger.info(
                        "Adjusting slice from %.8f to %.8f %s so it meets venue minimum.",
                        slice_notional,
                        required_notional,
                        plan.symbol,
                    )
                    slice_notional = required_notional
                    slice_quantity = limits.min_order_size
                else:
                    is_final_slice = slice_notional < plan.slice_notional
                    if is_final_slice:
                        logger.info(
                            "Final slice too small (qty %.8f < min %.8f), treat as done; skipping last order.",
                            slice_quantity, limits.min_order_size
                        )
                        target_remaining = 0.0
                        break
                    else:
                        logger.info(
                            "Slice qty %.8f below min %.8f for %s; skipping this slice and retry later.",
                            slice_quantity, limits.min_order_size, plan.symbol
                        )
                        self._sleep_with_stop(self._slice_delay())
                        continue
            min_quote = float(limits.min_order_size * reference_price)
            if slice_notional < min_quote:
                logger.debug(
                    "Slice notional %.8f below minimum quote %.8f, adjusting",
                    slice_notional,
                    min_quote,
                )
                slice_notional = min_quote
                slice_quantity = max(
                    limits.min_order_size,
                    round_to_precision(slice_notional / reference_price, limits.base_precision),
                )
                is_final_slice = slice_notional < plan.slice_notional  # 最后一刀 = 剩余不足
                if is_final_slice:
                    logger.info(
                        "Final slice too small (qty %.8f < min %.8f), treat as done; skipping last order.",
                        slice_quantity, limits.min_order_size
                    )
                    target_remaining = 0.0
                    break
                else:
                    logger.info(
                        "Slice qty %.8f below min %.8f for %s; skipping this slice and retry later.",
                        slice_quantity, limits.min_order_size, plan.symbol
                    )
                    self._sleep_with_stop(self._slice_delay())
                    continue

            fills = self._submit_limit_order(
                client=client,
                symbol=plan.symbol,
                side="Bid",
                quantity=slice_quantity,
                price=reference_price,
                post_only=self.config.post_only,
                reduce_only=False,
                limits=limits,
                account_label=self._account_labels[primary_idx],
            )
            filled_base = sum(fill["quantity"] for fill in fills)
            if filled_base <= 0:
                logger.info("Slice produced no fills, retrying another order...")
                self._sleep_with_stop(self._slice_delay())
                continue
            self._record_maker_fill(plan.symbol, fills)
            self._update_theoretical_position(plan.symbol, primary_idx, filled_base, limits)

            avg_price = self._weighted_average(fills)
            filled_notional = filled_base * avg_price
            total_base_filled += filled_base
            target_remaining = max(target_remaining - filled_notional, 0.0)
            logger.info(
                "Primary %s filled %.8f %s (%.2f / %.2f notional done)",
                self._account_labels[primary_idx],
                filled_base,
                plan.symbol,
                plan.target_notional - target_remaining,
                plan.target_notional,
            )

            self._dispatch_hedges(primary_idx, plan.symbol, "Ask", filled_base, hedgers, limits, reduce_only=False)
            self._sleep_with_stop(self._slice_delay())

        return total_base_filled

    def _flatten_position(
        self,
        primary_idx: int,
        hedgers: Sequence[int],
        plan: SymbolPlan,
        target_base_quantity: float,
    ) -> None:
        """Close the position opened during the entry leg."""
        client = self._clients[primary_idx]
        limits = self._get_market_limits(client, plan.symbol)
        remaining_base = float(target_base_quantity)
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        if remaining_base <= 0:
            logger.warning("No base quantity to unwind for %s", plan.symbol)
            return
        self._sync_hedge_positions(plan.symbol, hedgers, limits)

        while not self._stop_event.is_set() and remaining_base > limits.min_order_size / 2:
            reference_price = self._compute_reference_price(client, plan.symbol, "Ask", plan.exit_offset_bps)
            if reference_price is None or reference_price <= 0:
                logger.warning("Unable to compute exit price for %s, retrying...", plan.symbol)
                self._sleep_with_stop(3)
                continue
            slice_quantity = min(
                self._notional_to_quantity(plan.slice_notional, reference_price, limits),
                remaining_base,
            )
            min_exit_qty = self._effective_min_quantity(reference_price, limits)
            if slice_quantity + epsilon < min_exit_qty:
                logger.info(
                    "Exit remainder %.8f %s below venue minimum %.8f; stopping flatten loop.",
                    remaining_base,
                    plan.symbol,
                    min_exit_qty,
                )
                break

            fills = self._submit_limit_order(
                client=client,
                symbol=plan.symbol,
                side="Ask",
                quantity=slice_quantity,
                price=reference_price,
                post_only=self.config.post_only,
                reduce_only=True,
                limits=limits,
                account_label=self._account_labels[primary_idx],
            )
            filled_base = sum(fill["quantity"] for fill in fills)
            if filled_base <= 0:
                logger.info("Exit slice produced no fills, retrying...")
                self._sleep_with_stop(self._slice_delay())
                continue
            self._record_maker_fill(plan.symbol, fills)
            self._update_theoretical_position(plan.symbol, primary_idx, -filled_base, limits)

            remaining_base = max(remaining_base - filled_base, 0.0)
            logger.info(
                "Primary %s closed %.8f %s (%.8f remaining)",
                self._account_labels[primary_idx],
                filled_base,
                plan.symbol,
                remaining_base,
            )
            self._dispatch_hedges(primary_idx, plan.symbol, "Bid", filled_base, hedgers, limits, reduce_only=True)
            self._sleep_with_stop(self._slice_delay())

        if hedgers:
            self._flush_backlog(plan.symbol, primary_idx, hedgers, limits)
        logger.info("Exit leg for %s finished; remaining base %.8f", plan.symbol, remaining_base)

    def _hold_position(self, plan: SymbolPlan) -> None:
        hold_minutes = plan.hold_minutes or self.config.hold_minutes
        hold_seconds = max(hold_minutes * 60, 1)
        logger.info("Holding %s exposure for %.1f minutes", plan.symbol, hold_minutes)
        elapsed = 0.0
        while not self._stop_event.is_set() and elapsed < hold_seconds:
            self._sleep_with_stop(min(30, hold_seconds - elapsed))
            elapsed += min(30, hold_seconds - elapsed)

    # ------------------------------------------------------------------ helpers
    def _submit_limit_order(
        self,
        client: LighterClient,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool,
        reduce_only: bool,
        limits: MarketConstraints,
        account_label: str,
    ) -> List[Dict[str, float]]:
        rounded_qty = max(round_to_precision(quantity, limits.base_precision), limits.min_order_size)
        rounded_price = round_to_tick_size(price, limits.tick_size)
        submitted_at = time.time()
        logger.info(
            "Submitting %s order: account=%s side=%s qty=%.8f price=%.8f post_only=%s reduce_only=%s",
            symbol,
            account_label,
            side,
            rounded_qty,
            rounded_price,
            post_only,
            reduce_only,
        )
        time_in_force = getattr(self, "_primary_time_in_force", "GTC")
        tif_upper = time_in_force.upper()
        post_only_flag = post_only
        if tif_upper in ("FOK", "IOC") and post_only_flag:
            logger.warning("PostOnly disabled because timeInForce=%s is incompatible.", tif_upper)
            post_only_flag = False
        # Lighter signer requires NilOrderExpiry (0) for IOC/FOK limit orders, otherwise it rejects the request.
        order_expiry: Optional[int] = None
        if tif_upper in ("FOK", "IOC"):
            order_expiry = 0
        order = {
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "quantity": str(rounded_qty),
            "price": str(rounded_price),
            "postOnly": post_only_flag,
            "reduceOnly": reduce_only,
            "timeInForce": tif_upper,
        }
        if order_expiry is not None:
            order["orderExpiry"] = order_expiry
        logger.debug(
            "execute_order payload=%s thread=%s",
            order,
            threading.get_ident(),
        )
        response = self._execute_with_nonce_retry(client, order, account_label)
        if isinstance(response, dict) and response.get("error"):
            logger.error("Limit order rejected for %s: %s", symbol, response["error"])
            return []

        identifiers = self._extract_order_identifiers(response)
        if not identifiers:
            logger.warning("Order identifiers missing for %s, falling back to timestamp matching", symbol)

        return self._wait_for_fills(
            client=client,
            symbol=symbol,
            match_order_ids=identifiers,
            target_qty=rounded_qty,
            limits=limits,
            order_start_time=submitted_at,
        )

    def _wait_for_fills(
        self,
        client: LighterClient,
        symbol: str,
        match_order_ids: Sequence[str],
        target_qty: float,
        limits: MarketConstraints,
        order_start_time: float,
    ) -> List[Dict[str, float]]:
        deadline = time.time() + self.config.slice_fill_timeout
        seen_trades: Set[str] = set()
        fills: List[Dict[str, float]] = []
        filled_qty = 0.0
        tolerance = max(limits.min_order_size * 0.1, 1e-9)
        match_ids: Set[str] = {str(value) for value in match_order_ids if value is not None}
        fallback_threshold = max(order_start_time - 1.0, 0.0)

        def consume_trades(trades_payload: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
            nonlocal filled_qty
            if isinstance(trades_payload, dict):
                return
            for trade in trades_payload or []:
                order_id = trade.get("order_id")
                trade_ts = self._normalize_timestamp(trade.get("timestamp"))
                matches_identifier = order_id is not None and str(order_id) in match_ids
                is_recent = trade_ts is not None and trade_ts >= fallback_threshold
                if match_ids:
                    if not matches_identifier and not is_recent:
                        continue
                else:
                    if not is_recent:
                        continue
                trade_id = str(trade.get("trade_id"))
                if trade_id in seen_trades:
                    continue
                qty = float(trade.get("quantity") or trade.get("size") or 0)
                price = float(trade.get("price") or 0)
                if qty <= 0 or price <= 0:
                    continue
                seen_trades.add(trade_id)
                fills.append({"quantity": qty, "price": price})
                filled_qty += qty

        while not self._stop_event.is_set():
            trades = client.get_fill_history(symbol, limit=50)
            if isinstance(trades, dict) and trades.get("error"):
                logger.error("Failed to fetch fills for %s: %s", symbol, trades["error"])
            else:
                consume_trades(trades)

            if filled_qty + tolerance >= target_qty:
                break
            if time.time() >= deadline:
                logger.warning("Fill wait timeout for %s (filled %.8f / %.8f)", symbol, filled_qty, target_qty)
                break
            self._sleep_with_stop(self.config.order_poll_interval)

        if filled_qty + tolerance < target_qty:
            extra_trades = client.get_fill_history(symbol, limit=200)
            if isinstance(extra_trades, dict) and extra_trades.get("error"):
                logger.error("Failed to fetch final fills for %s: %s", symbol, extra_trades["error"])
            else:
                consume_trades(extra_trades)

        if filled_qty + tolerance < target_qty:
            cancel_result = client.cancel_all_orders(symbol)
            if isinstance(cancel_result, dict) and cancel_result.get("error"):
                logger.error("Cancel all orders for %s failed: %s", symbol, cancel_result["error"])
            else:
                logger.info("Cancel all orders for %s response: %s", symbol, cancel_result)
            post_cancel_trades = client.get_fill_history(symbol, limit=200)
            if isinstance(post_cancel_trades, dict) and post_cancel_trades.get("error"):
                logger.error("Failed to fetch fills after cancel for %s: %s", symbol, post_cancel_trades["error"])
            else:
                consume_trades(post_cancel_trades)

        return fills

    def _dispatch_hedges(
        self,
        primary_idx: Optional[int],
        symbol: str,
        side: str,
        qty: float,
        hedger_indices: Sequence[int],
        limits: MarketConstraints,
        *,
        reduce_only: bool,
    ) -> None:
        if qty <= 0:
            return
        if not hedger_indices:
            logger.warning("No hedger accounts configured for %s fill %.8f", symbol, qty)
            return
        allocations = self._split_base_quantity(qty, hedger_indices, limits)
        min_trade_qty = self._estimate_minimum_trade_qty(symbol, side, hedger_indices, limits)
        if not reduce_only and len(hedger_indices) > 1:
            threshold = max(min_trade_qty, limits.min_order_size)
            if qty < threshold * len(hedger_indices):
                target_pos = self._select_small_fill_target(hedger_indices)
                allocations = [0.0 for _ in hedger_indices]
                allocations[target_pos] = round_to_precision(qty, limits.base_precision)
        if reduce_only:
            allocations = self._apply_remaining_caps(symbol, hedger_indices, allocations, qty, limits)

        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        for list_index, hedger_idx in enumerate(hedger_indices):
            alloc = allocations[list_index]
            alloc += self._drain_backlog(hedger_idx, symbol, side)
            rounded = round_to_precision(alloc, limits.base_precision)
            if rounded <= 0:
                logger.debug(
                    "Rounded hedge size is zero after precision clamp (alloc=%.8f, precision=%s)",
                    alloc,
                    limits.base_precision,
                )
                continue
            price = self._compute_aggressive_price(self._clients[hedger_idx], symbol, side)
            if price is None:
                logger.error("Unable to compute hedge price for %s", symbol)
                self._enqueue_backlog(hedger_idx, symbol, side, alloc)
                continue
            price = round_to_tick_size(price, limits.tick_size)
            min_required = self._effective_min_quantity(price, limits)
            if rounded > 0 and rounded + epsilon < min_required:
                logger.warning(
                    "Hedge qty %.8f below venue minimum %.8f at price %.8f for %s, deferring remainder",
                    rounded,
                    min_required,
                    price,
                    self._account_labels[hedger_idx],
                )
                self._enqueue_backlog(hedger_idx, symbol, side, rounded)
                continue
            payload = {
                "symbol": symbol,
                "side": side,
                "orderType": "Limit",
                "quantity": str(rounded),
                "price": str(price),
                "reduceOnly": reduce_only,
                "timeInForce": "GTC",
                "price_protection": False,
            }
            logger.info(
                "Submitting hedge order: account=%s side=%s qty=%.8f reduce_only=%s",
                self._account_labels[hedger_idx],
                side,
                rounded,
                reduce_only,
            )
            logger.debug(
                "execute_order hedge payload=%s thread=%s",
                payload,
                threading.get_ident(),
            )
            response = self._execute_with_nonce_retry(
                self._clients[hedger_idx],
                payload,
                self._account_labels[hedger_idx],
            )
            if isinstance(response, dict) and response.get("error"):
                logger.error(
                    "Hedge order failed (%s -> %s): %s",
                    self._account_labels[hedger_idx],
                    side,
                    response["error"],
                )
                self._enqueue_backlog(hedger_idx, symbol, side, alloc)
                continue
            order_ids = self._extract_order_identifiers(response)
            logger.info(
                "Hedge submitted: account=%s side=%s qty=%.8f id=%s",
                self._account_labels[hedger_idx],
                side,
                rounded,
                order_ids[0] if order_ids else "N/A",
            )
            delta = rounded if side == "Bid" else -rounded
            self._update_theoretical_position(symbol, hedger_idx, delta, limits)
            self._record_hedge_fill(hedger_idx, symbol, order_ids, side, limits)
        self._log_position_snapshot(symbol, primary_idx, hedger_indices)

    def _split_base_quantity(
        self,
        total_quantity: float,
        hedger_indices: Sequence[int],
        limits: MarketConstraints,
    ) -> List[float]:
        """Split base quantity across hedgers while respecting precision."""
        bucket_count = len(hedger_indices)
        if bucket_count <= 0:
            return []
        total_quantity = round_to_precision(total_quantity, limits.base_precision)
        if bucket_count == 1:
            return [total_quantity]

        min_total_needed = limits.min_order_size * bucket_count
        if total_quantity < min_total_needed:
            allocations = [0.0 for _ in hedger_indices]
            target_pos = self._select_small_fill_target(hedger_indices)
            allocations[target_pos] = total_quantity
            return allocations

        low, high = self.config.random_split_range
        raw_allocations: List[float]
        if bucket_count == 2:
            ratio = self._random.uniform(low, high)
            raw_allocations = [total_quantity * ratio, total_quantity * (1 - ratio)]
        else:
            even_share = total_quantity / bucket_count
            raw_allocations = [even_share for _ in range(bucket_count - 1)]
            remainder = total_quantity - even_share * (bucket_count - 1)
            raw_allocations.append(remainder)

        rounded: List[float] = []
        consumed = 0.0
        for idx, amount in enumerate(raw_allocations):
            if idx == bucket_count - 1:
                value = max(total_quantity - consumed, 0.0)
                value = round_to_precision(value, limits.base_precision)
            else:
                value = round_to_precision(amount, limits.base_precision)
                consumed += value
            rounded.append(value)

        # Adjust final bucket to fix rounding drift
        total_rounded = sum(rounded)
        drift = round_to_precision(total_quantity - total_rounded, limits.base_precision)
        if abs(drift) >= max(1e-12, 10 ** (-limits.base_precision)):
            rounded[-1] = round_to_precision(rounded[-1] + drift, limits.base_precision)
        return rounded

    def _effective_min_quantity(self, price: float, limits: MarketConstraints) -> float:
        if price <= 0:
            return limits.min_order_size
        min_qty = limits.min_order_size
        min_quote = max(float(getattr(limits, "min_quote_value", 0.0) or 0.0), 0.0)
        if min_quote > 0:
            precision = max(limits.base_precision, 0)
            factor = 10 ** precision
            required = math.ceil((min_quote / price) * factor) / factor
            min_qty = max(min_qty, required)
        return round_to_precision(min_qty, limits.base_precision)

    def _estimate_minimum_trade_qty(
        self,
        symbol: str,
        side: str,
        hedger_indices: Sequence[int],
        limits: MarketConstraints,
    ) -> float:
        if not hedger_indices:
            return limits.min_order_size
        reference_idx = hedger_indices[0]
        price = self._compute_aggressive_price(self._clients[reference_idx], symbol, side)
        if price is None:
            return limits.min_order_size
        return self._effective_min_quantity(price, limits)

    def _apply_remaining_caps(
        self,
        symbol: str,
        hedger_indices: Sequence[int],
        allocations: List[float],
        total_quantity: float,
        limits: MarketConstraints,
    ) -> List[float]:
        """Clamp hedge allocations so we only unwind what was previously assigned."""
        book = self._hedge_cycle_positions.get(symbol)
        if not book:
            return allocations
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        remaining = [max(book.get(idx, 0.0), 0.0) for idx in hedger_indices]
        total_capacity = sum(remaining)
        if total_capacity <= epsilon:
            return allocations

        adjusted = [0.0 for _ in allocations]
        qty_left = total_quantity

        # First pass: clamp planned allocation by each account's remaining exposure.
        for idx, planned in enumerate(allocations):
            allowance = min(planned, remaining[idx], qty_left)
            allowance = round_to_precision(allowance, limits.base_precision)
            adjusted[idx] = allowance
            remaining[idx] = max(round_to_precision(remaining[idx] - allowance, limits.base_precision), 0.0)
            qty_left = max(round_to_precision(qty_left - allowance, limits.base_precision), 0.0)

        if qty_left <= epsilon:
            return adjusted

        # Second pass: distribute any leftover to whoever still has exposure.
        for idx in range(len(hedger_indices)):
            if qty_left <= epsilon:
                break
            cap = remaining[idx]
            if cap <= epsilon:
                continue
            add = min(cap, qty_left)
            add = round_to_precision(add, limits.base_precision)
            adjusted[idx] = round_to_precision(adjusted[idx] + add, limits.base_precision)
            remaining[idx] = max(round_to_precision(cap - add, limits.base_precision), 0.0)
            qty_left = max(round_to_precision(qty_left - add, limits.base_precision), 0.0)

        if qty_left > epsilon:
            logger.warning(
                "Symbol %s hedge unwind still missing %.8f after capping; assigning remainder to last hedger.",
                symbol,
                qty_left,
            )
            adjusted[-1] = round_to_precision(adjusted[-1] + qty_left, limits.base_precision)
        return adjusted

    def _drain_backlog(self, account_idx: int, symbol: str, side: str) -> float:
        store = self._hedge_backlog.setdefault(account_idx, {}).setdefault(symbol, {})
        qty = store.pop(side, 0.0)
        return qty

    def _enqueue_backlog(self, account_idx: int, symbol: str, side: str, qty: float) -> None:
        if qty <= 0:
            return
        store = self._hedge_backlog.setdefault(account_idx, {}).setdefault(symbol, {})
        store[side] = store.get(side, 0.0) + qty

    def _peek_backlog(self, account_idx: int) -> float:
        total = 0.0
        symbol_map = self._hedge_backlog.get(account_idx, {})
        for side_map in symbol_map.values():
            total += sum(abs(val) for val in side_map.values())
        return total

    def _select_small_fill_target(self, hedger_indices: Sequence[int]) -> int:
        if not hedger_indices:
            return 0
        backlog_stats = [
            (self._peek_backlog(idx), pos) for pos, idx in enumerate(hedger_indices)
        ]
        backlog_stats.sort(key=lambda item: item[0])
        return backlog_stats[0][1]

    def _update_cycle_position(self, symbol: str, hedger_idx: int, delta: float) -> None:
        if abs(delta) <= 1e-12:
            return
        store = self._hedge_cycle_positions.setdefault(symbol, {})
        next_value = store.get(hedger_idx, 0.0) + delta
        if next_value <= 1e-12:
            store.pop(hedger_idx, None)
        else:
            store[hedger_idx] = next_value
        if not store:
            self._hedge_cycle_positions.pop(symbol, None)

    def _sync_hedge_positions(
        self,
        symbol: str,
        hedger_indices: Sequence[int],
        limits: MarketConstraints,
    ) -> None:
        if not hedger_indices:
            return
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        store = self._hedge_cycle_positions.setdefault(symbol, {})
        for hedger_idx in hedger_indices:
            client = self._clients[hedger_idx]
            try:
                positions = client.get_positions(symbol)
            except Exception as exc:  # pragma: no cover - defensive against HTTP issues
                logger.warning(
                    "Unable to sync hedge position for %s (%s): %s",
                    symbol,
                    self._account_labels[hedger_idx],
                    exc,
                )
                continue
            if isinstance(positions, dict) and positions.get("error"):
                logger.warning(
                    "Unable to sync hedge position for %s (%s): %s",
                    symbol,
                    self._account_labels[hedger_idx],
                    positions["error"],
                )
                continue
            qty = 0.0
            if isinstance(positions, list):
                for entry in positions:
                    if not isinstance(entry, dict):
                        continue
                    raw = entry.get("netQuantity") or entry.get("rawSize") or entry.get("size")
                    try:
                        qty = abs(float(raw))
                    except (TypeError, ValueError):
                        qty = 0.0
                    break
            qty = round_to_precision(qty, limits.base_precision)
            if qty <= epsilon:
                store.pop(hedger_idx, None)
            else:
                store[hedger_idx] = qty
        if store:
            self._hedge_cycle_positions[symbol] = store
        else:
            self._hedge_cycle_positions.pop(symbol, None)

    def _flush_backlog(
        self,
        symbol: str,
        primary_idx: Optional[int],
        hedger_indices: Sequence[int],
        limits: MarketConstraints,
    ) -> None:
        for hedger_idx in hedger_indices:
            store = self._hedge_backlog.get(hedger_idx, {}).get(symbol)
            if not store:
                continue
            for side, qty in list(store.items()):
                if qty <= 0:
                    continue
                rounded = max(round_to_precision(qty, limits.base_precision), limits.min_order_size)
                price = self._compute_aggressive_price(self._clients[hedger_idx], symbol, side)
                if price is None:
                    logger.error("Unable to flush backlog for %s (%s) due to missing price", symbol, self._account_labels[hedger_idx])
                    continue
                payload = {
                    "symbol": symbol,
                    "side": side,
                    "orderType": "Limit",
                    "quantity": str(rounded),
                    "price": str(round_to_tick_size(price, limits.tick_size)),
                    "reduceOnly": True,
                    "timeInForce": "GTC",
                    "price_protection": False,
                }
                logger.info(
                    "Flushing backlog: account=%s side=%s qty=%.8f",
                    self._account_labels[hedger_idx],
                    side,
                    rounded,
                )
                response = self._execute_with_nonce_retry(
                    self._clients[hedger_idx],
                    payload,
                    self._account_labels[hedger_idx],
                )
                if isinstance(response, dict) and response.get("error"):
                    logger.error(
                        "Backlog flush failed (%s %s): %s",
                        self._account_labels[hedger_idx],
                        symbol,
                        response["error"],
                    )
                else:
                    store[side] = 0.0
                    order_ids = self._extract_order_identifiers(response)
                    delta = rounded if side == "Bid" else -rounded
                    self._update_theoretical_position(symbol, hedger_idx, delta, limits)
                    self._record_hedge_fill(hedger_idx, symbol, order_ids, side, limits)
        self._log_position_snapshot(symbol, primary_idx, hedger_indices)


    def _update_theoretical_position(
        self,
        symbol: str,
        account_idx: int,
        delta: float,
        limits: MarketConstraints,
    ) -> None:
        if abs(delta) <= 1e-12:
            return
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        book = self._theoretical_positions.setdefault(symbol, {})
        next_value = round_to_precision(book.get(account_idx, 0.0) + delta, limits.base_precision)
        if abs(next_value) <= epsilon:
            book.pop(account_idx, None)
        else:
            book[account_idx] = next_value
        if not book:
            self._theoretical_positions.pop(symbol, None)

    def _get_theoretical_position(self, symbol: str, account_idx: int) -> float:
        return self._theoretical_positions.get(symbol, {}).get(account_idx, 0.0)

    def _get_actual_position(self, account_idx: int, symbol: str) -> Optional[float]:
        client = self._clients[account_idx]
        time.sleep(0.35)
        positions = client.get_positions(symbol)
        if isinstance(positions, list):
            for entry in positions:
                if not isinstance(entry, dict):
                    continue
                raw = entry.get("netQuantity") or entry.get("rawSize") or entry.get("size")
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _log_position_snapshot(
        self,
        symbol: str,
        primary_idx: Optional[int],
        hedger_indices: Sequence[int],
    ) -> None:
        participants: List[int] = []
        if primary_idx is not None:
            participants.append(primary_idx)
        for hedger_idx in hedger_indices:
            if hedger_idx not in participants:
                participants.append(hedger_idx)
        if not participants:
            return
        entries: List[str] = []
        for account_idx in participants:
            theory = self._get_theoretical_position(symbol, account_idx)
            actual = self._get_actual_position(account_idx, symbol)
            entries.append(self._format_position_entry(symbol, account_idx, theory, actual))
        logger.info("[POSITION] %s %s", symbol, " | ".join(entries))

    def _format_position_entry(
        self,
        symbol: str,
        account_idx: int,
        theoretical: float,
        actual: Optional[float],
    ) -> str:
        label = self._account_labels[account_idx]
        theory_side = self._describe_side(theoretical)
        theory_qty = abs(theoretical)
        if actual is None:
            actual_desc = "实际 N/A"
        else:
            actual_side = self._describe_side(actual)
            actual_desc = f"实际 {abs(actual):.8f} {symbol} {actual_side}"
        return f"{label}: 理论 {theory_qty:.8f} {symbol} {theory_side} / {actual_desc}"

    @staticmethod
    def _describe_side(value: Optional[float]) -> str:
        if value is None:
            return "UNKNOWN"
        if value > 0:
            return "LONG"
        if value < 0:
            return "SHORT"
        return "FLAT"

    def _weighted_average(self, fills: Iterable[Dict[str, float]]) -> float:
        total_notional = 0.0
        total_qty = 0.0
        for fill in fills:
            qty = fill.get("quantity", 0.0)
            price = fill.get("price", 0.0)
            total_qty += qty
            total_notional += qty * price
        return total_notional / total_qty if total_qty else 0.0

    def _notional_to_quantity(self, notional: float, price: float, limits: MarketConstraints) -> float:
        if price <= 0:
            return limits.min_order_size
        qty = notional / price
        qty = round_to_precision(qty, limits.base_precision)
        return max(qty, limits.min_order_size)

    def _compute_reference_price(
        self,
        client: LighterClient,
        symbol: str,
        side: str,
        override_offset_bps: Optional[float],
    ) -> Optional[float]:
        book = client.get_order_book(symbol, limit=5)
        if isinstance(book, dict) and book.get("error"):
            logger.error("Failed to fetch order book for %s: %s", symbol, book["error"])
            return None
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        offset = override_offset_bps
        if offset is None:
            offset = self.config.entry_price_offset_bps if side == "Bid" else self.config.exit_price_offset_bps
        offset = float(offset or 0)
        if side == "Bid":
            base = best_bid or best_ask
            if base is None:
                return None
            price = base * (1 + offset / 10_000.0)
            if best_ask:
                price = min(price, best_ask - 1e-9)
            if best_bid:
                price = max(price, best_bid)
            return price
        base = best_ask or best_bid
        if base is None:
            return None
        price = base * (1 - offset / 10_000.0)
        if best_bid:
            price = max(price, best_bid + 1e-9)
        if best_ask:
            price = min(price, best_ask)
        return price

    def _compute_aggressive_price(
        self,
        client: LighterClient,
        symbol: str,
        side: str,
    ) -> Optional[float]:
        """Derive aggressive limit price to emulate market order behaviour."""
        book = client.get_order_book(symbol, limit=5)
        if isinstance(book, dict) and book.get("error"):
            logger.error("Failed to fetch order book for %s: %s", symbol, book["error"])
            return None
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        if side == "Bid":
            base = best_ask or best_bid
            if base is None:
                return None
            price = base * 1.002
            if best_ask:
                price = max(price, best_ask)
            return price
        base = best_bid or best_ask
        if base is None:
            return None
        price = base * 0.998
        if best_bid:
            price = min(price, best_bid)
        return price

    def _get_market_limits(self, client: LighterClient, symbol: str) -> MarketConstraints:
        cached = self._market_cache.get(symbol)
        if cached:
            return cached
        metadata = client.get_market_limits(symbol)
        if not metadata:
            raise StrategyConfigError(f"Unable to load market metadata for {symbol}")
        limits = MarketConstraints(
            base_precision=int(metadata["base_precision"]),
            quote_precision=int(metadata["quote_precision"]),
            min_order_size=float(metadata["min_order_size"]),
            tick_size=float(metadata["tick_size"]),
            min_quote_value=float(metadata.get("min_quote_value") or 10.0),
        )
        self._market_cache[symbol] = limits
        return limits

    def _sleep_with_stop(self, seconds: float) -> None:
        target_time = time.time() + max(seconds, 0)
        while not self._stop_event.is_set() and time.time() < target_time:
            time.sleep(min(1, target_time - time.time()))

    def _slice_delay(self) -> float:
        jitter = self._random.uniform(0, max(self.config.slice_delay_jitter_seconds, 0))
        return max(self.config.slice_delay_seconds + jitter, 0)

    def _resolve_hedgers(self, primary_idx: int) -> List[int]:
        if not self._hedge_enabled:
            return []
        indices = list(range(len(self._clients)))
        indices.remove(primary_idx)
        return indices[:2]

    def _advance_pointers(self) -> None:
        self._current_symbol_index = (self._current_symbol_index + 1) % len(self.config.symbols)
        self._current_primary_index = (self._current_primary_index + 1) % len(self._clients)

    @staticmethod
    def _refresh_client_nonce(client: LighterClient) -> Optional[int]:
        refresh = getattr(client, "refresh_nonce", None)
        if callable(refresh):
            try:
                return refresh()
            except Exception as exc:
                logger.debug("Failed to refresh nonce: %s", exc)
        return None

    def _execute_with_nonce_retry(
        self,
        client: LighterClient,
        order: Dict[str, Any],
        account_label: str,
        *,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute an order, refreshing nonce when the exchange rejects it."""
        last_response: Dict[str, Any] = {}
        retries = max_retries if isinstance(max_retries, int) and max_retries > 0 else 1
        for attempt in range(1, retries + 1):
            response = client.execute_order(order)
            if not (isinstance(response, dict) and response.get("error")):
                return response
            last_response = response
            error_text = str(response.get("error") or "")
            invalid_nonce = "invalid nonce" in error_text.lower()
            if invalid_nonce and attempt < retries:
                refreshed = self._refresh_client_nonce(client)
                refreshed_next = refreshed + 1 if refreshed is not None else None
                current_nonce = getattr(client, "debug_current_nonce", lambda: None)()
                logger.warning(
                    "Invalid nonce for %s (%s %s). Local=%s refreshed=%s (next=%s). Retrying %d/%d...",
                    account_label,
                    order.get("symbol"),
                    order.get("side"),
                    current_nonce,
                    refreshed,
                    refreshed_next,
                    attempt,
                    retries,
                )
                time.sleep(0.25)
                continue
            if invalid_nonce:
                refreshed = self._refresh_client_nonce(client)
                refreshed_next = refreshed + 1 if refreshed is not None else None
                current_nonce = getattr(client, "debug_current_nonce", lambda: None)()
                logger.error(
                    "Nonce still invalid for %s after %d attempts. Local=%s refreshed=%s (next=%s). Last error: %s",
                    account_label,
                    retries,
                    current_nonce,
                    refreshed,
                    refreshed_next,
                    response.get("error"),
                )
            break
        return last_response

    def _record_maker_fill(self, symbol: str, fills: Iterable[Dict[str, float]]) -> None:
        if not fills:
            return
        stats = self._maker_price_stats.setdefault(symbol, {"qty": 0.0, "notional": 0.0})
        for fill in fills:
            qty = float(fill.get("quantity") or 0)
            price = float(fill.get("price") or 0)
            if qty <= 0 or price <= 0:
                continue
            stats["qty"] += qty
            stats["notional"] += qty * price

    def _record_hedge_fill(
        self,
        hedger_idx: int,
        symbol: str,
        order_ids: List[str],
        side: Optional[str],
        limits: MarketConstraints,
    ) -> None:
        client = self._clients[hedger_idx]
        stats = self._hedge_price_stats.setdefault(symbol, {}).setdefault(
            hedger_idx,
            {"qty": 0.0, "notional": 0.0},
        )
        recorded = self._hedge_recorded_trades[hedger_idx]
        id_filter: Set[str] = {str(order_id) for order_id in order_ids if order_id is not None}
        matched = False
        max_attempts = 3
        delay = max(min(self.config.order_poll_interval, 1.0), 0.25)

        for attempt in range(max_attempts):
            trades = client.get_fill_history(symbol, limit=20)
            if isinstance(trades, dict) and trades.get("error"):
                logger.warning(
                    "Unable to fetch hedge fills for %s (%s): %s",
                    self._account_labels[hedger_idx],
                    symbol,
                    trades["error"],
                )
                break
            fallback_threshold = time.time() - 5.0
            for trade in trades or []:
                order_id = str(trade.get("order_id") or "")
                trade_ts = self._normalize_timestamp(trade.get("timestamp"))
                matches_identifier = bool(id_filter) and order_id in id_filter
                is_recent = trade_ts is not None and trade_ts >= fallback_threshold
                if id_filter:
                    if not matches_identifier and not is_recent:
                        continue
                else:
                    if not is_recent:
                        continue
                trade_id = str(trade.get("trade_id") or "")
                if trade_id in recorded:
                    continue
                qty = float(trade.get("quantity") or trade.get("size") or 0)
                price = float(trade.get("price") or 0)
                if qty <= 0 or price <= 0:
                    continue
                recorded.add(trade_id)
                stats["qty"] += qty
                stats["notional"] += qty * price
                if side == "Ask":
                    self._update_cycle_position(symbol, hedger_idx, qty)
                elif side == "Bid":
                    self._update_cycle_position(symbol, hedger_idx, -qty)
                matched = True
            if matched or self._stop_event.is_set():
                break
            if attempt < max_attempts - 1:
                self._sleep_with_stop(delay)

        if not matched and side in ("Ask", "Bid"):
            self._sync_hedge_positions(symbol, [hedger_idx], limits)
            logger.warning(
                "No hedge fills matched for %s (%s); refreshed position snapshot instead (order_ids=%s)",
                self._account_labels[hedger_idx],
                symbol,
                order_ids or ["N/A"],
            )

    def _print_cycle_summary(self, symbol: str) -> None:
        maker_stats = self._maker_price_stats.get(symbol)
        hedger_stats = self._hedge_price_stats.get(symbol, {})
        if not maker_stats or maker_stats.get("qty", 0) <= 0:
            return
        maker_avg = maker_stats["notional"] / maker_stats["qty"]
        logger.info(
            "[SUMMARY] %s Maker total %.8f @ %.5f",
            symbol,
            maker_stats["qty"],
            maker_avg,
        )
        combined_qty = 0.0
        combined_notional = 0.0
        for hedger_idx, stats in hedger_stats.items():
            qty = stats.get("qty", 0.0)
            if qty <= 0:
                continue
            avg = stats["notional"] / qty
            slippage = avg - maker_avg
            logger.info(
                "[SUMMARY] Hedge %s total %.8f @ %.5f (slippage %.8f)",
                self._account_labels[hedger_idx],
                qty,
                avg,
                slippage,
            )
            combined_qty += qty
            combined_notional += stats["notional"]

        if combined_qty > 0:
            hedge_avg = combined_notional / combined_qty
            wear_per_unit = hedge_avg - maker_avg
            # Use the smaller of maker vs hedge executed size to avoid over-counting.
            nominal_qty = min(maker_stats["qty"], combined_qty)
            wear_notional = wear_per_unit * nominal_qty
            wear_rate = wear_notional / max(2 * maker_stats["qty"], 1e-12)
            logger.info(
                "[SUMMARY] Cycle wear per-unit=%.8f notional=%.8f rate=%.6f%%",
                wear_per_unit,
                wear_notional,
                wear_rate * 100,
            )
            self._wear_cumulative["wear_notional"] += wear_notional
            self._wear_cumulative["maker_qty"] += maker_stats["qty"]
            global_wear_rate = (
                self._wear_cumulative["wear_notional"]
                / max(2 * self._wear_cumulative["maker_qty"], 1e-12)
            )
            logger.info(
                "[SUMMARY] Global wear notional=%.8f rate=%.6f%% over maker_qty %.8f",
                self._wear_cumulative["wear_notional"],
                global_wear_rate * 100,
                self._wear_cumulative["maker_qty"],
            )

        self._maker_price_stats.pop(symbol, None)
        if symbol in self._hedge_price_stats:
            self._hedge_price_stats.pop(symbol, None)
        self._hedge_cycle_positions.pop(symbol, None)
        self._theoretical_positions.pop(symbol, None)
        for record in self._hedge_recorded_trades:
            record.clear()

    @staticmethod
    def _normalize_timestamp(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            ts = float(value)
        except (TypeError, ValueError):
            return None
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        return ts

    @staticmethod
    def _extract_order_identifiers(payload: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(payload, dict):
            return []
        identifiers: List[str] = []
        for key in ("orderIndex", "orderId", "id", "clientOrderIndex", "clientOrderId"):
            value = payload.get(key)
            if value is not None:
                identifiers.append(str(value))
        return identifiers
