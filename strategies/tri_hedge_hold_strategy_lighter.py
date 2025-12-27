from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from api.lighter_client import LighterClient, SimpleSignerError
from logger import setup_logger
from utils.helpers import round_to_precision, round_to_tick_size
from utils.telegram_notify import TelegramNotifier

logger = setup_logger("tri_hedge_strategy")


MIN_QUOTE_THRESHOLD = 11.0


class StrategyConfigError(ValueError):
    """Raised when the strategy configuration is invalid."""


class MarginFailsafeTriggered(RuntimeError):
    """Raised when margin protection flattens positions and stops the strategy."""


class OrderSubmissionError(RuntimeError):
    """Raised when an order submission fails and we need to unwind."""

    def __init__(self, symbol: str, account_idx: int, account_label: str, message: str) -> None:
        super().__init__(message)
        self.symbol = symbol
        self.account_idx = account_idx
        self.account_label = account_label


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
    proxy: Optional[str] = None

    def as_client_config(self, defaults: "TriHedgeHoldStrategyConfig") -> Dict[str, Any]:
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
        if self.proxy:
            config["proxy"] = self.proxy
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
    min_quote_value: float = MIN_QUOTE_THRESHOLD


@dataclass
class TriHedgeHoldStrategyConfig:
    """Runtime configuration for the volume hold strategy."""

    accounts: List[AccountCredentials]
    symbols: List[SymbolPlan]
    base_url: Optional[str] = None
    chain_id: Optional[int] = None
    signer_lib_dir: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    hold_minutes: float = 1.0
    entry_price_offset_bps: float = 5.0
    exit_price_offset_bps: float = 5.0
    slice_delay_seconds: float = 6.0
    slice_delay_jitter_seconds: float = 3.0
    slice_fill_timeout: float = 8.0
    # waiting fill 的时候查询仓位的间隔
    order_poll_interval: float = 1.0
    post_only: bool = True
    pause_between_symbols: float = 30.0
    random_split_range: Tuple[float, float] = (0.45, 0.55)
    run_once: bool = False
    enable_hedge: bool = True
    primary_time_in_force: str = "GTC"
    market_fetch_retries: int = 3
    order_submit_retries: int = 3

    @classmethod
    def from_file(cls, path: str) -> "TriHedgeHoldStrategyConfig":
        if not os.path.isfile(path):
            raise StrategyConfigError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TriHedgeHoldStrategyConfig":
        base_url = payload.get("base_url") or os.getenv("LIGHTER_BASE_URL")
        chain_id = payload.get("chain_id") or os.getenv("LIGHTER_CHAIN_ID")
        signer_lib_dir = payload.get("signer_lib_dir")
        default_target_notional = float(payload.get("default_target_notional", 5000))
        default_slice_count = max(1, int(payload.get("default_slice_count", 50)))
        telegram_bot_token = (
            payload.get("telegram_bot_token")
            or payload.get("tg_bot_token")
            or os.getenv("TELEGRAM_BOT_TOKEN")
            or os.getenv("TG_BOT_TOKEN")
        )
        telegram_chat_id = (
            payload.get("telegram_chat_id")
            or payload.get("tg_chat_id")
            or os.getenv("TELEGRAM_CHAT_ID")
            or os.getenv("TG_BOT_CHAT_ID")
        )

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
                account_address = entry.get("account_address")
                if not account_address:
                    raise StrategyConfigError(f"account_index/account_address required for {label}")
                from api.lighter_client import _get_lihgter_account_index  # lazy import

                account_index = _get_lihgter_account_index(account_address)
            cred = AccountCredentials(
                label=label,
                api_private_key=str(api_private_key),
                account_index=int(account_index),
                api_key_index=int(entry.get("api_key_index") or 0),
                base_url=entry.get("base_url") or base_url,
                chain_id=int(entry.get("chain_id") or chain_id)
                if entry.get("chain_id") or chain_id
                else None,
                signer_lib_dir=entry.get("signer_lib_dir") or signer_lib_dir,
                proxy=entry.get("proxy"),
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
            symbols.append(
                SymbolPlan(
                    symbol=str(symbol),
                    target_notional=float(target_notional),
                    slice_notional=float(slice_notional),
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
            telegram_bot_token=str(telegram_bot_token) if telegram_bot_token else None,
            telegram_chat_id=str(telegram_chat_id) if telegram_chat_id else None,
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
            primary_time_in_force=str(payload.get("primary_time_in_force", "GTC")),
            market_fetch_retries=int(payload.get("market_fetch_retries", 3)),
        )
        if not config.base_url and any(acc.base_url is None for acc in accounts):
            raise StrategyConfigError("base_url missing; configure global base_url or per-account base_url entries.")
        return config


class TriHedgeHoldStrategy:
    """Implements the refactored multi-account volume/holding strategy."""

    def __init__(self, config: TriHedgeHoldStrategyConfig, *, random_seed: Optional[int] = None) -> None:
        self.config = config
        self._random = random.Random(random_seed)
        self._clients = [LighterClient(acc.as_client_config(config)) for acc in config.accounts]
        self._account_labels = [acc.label for acc in config.accounts]
        self._market_cache: Dict[str, MarketConstraints] = {}
        self._stop_event = threading.Event()
        self._current_symbol_index = 0
        self._current_primary_index = 0
        self._maker_price_stats: Dict[str, Dict[str, float]] = {}
        self._hedge_price_stats: Dict[str, Dict[int, Dict[str, float]]] = {}
        self._wear_cumulative = {"wear_notional": 0.0, "maker_qty": 0.0}
        self._primary_time_in_force = str(config.primary_time_in_force or "GTC").upper()
        self._position_cache: Dict[int, Dict[str, float]] = {idx: {} for idx in range(len(self._clients))}
        self._position_price_cache: Dict[int, Dict[str, float]] = {idx: {} for idx in range(len(self._clients))}
        self._small_fill_selector: Dict[Tuple[int, ...], int] = {}
        # 磨损统计
        self._round_stats: List[Dict[str, Any]] = []
        self._current_round: Optional[Dict[str, Any]] = None
        self._round_start_time: Optional[float] = None
        # 手续费费率
        self._maker_fee_rate = 0.00002  # 0.002%
        self._taker_fee_rate = 0.0002  # 0.02%
        self._market_fetch_retries = max(1, int(config.market_fetch_retries))
        self._order_submit_retries = max(1, int(config.order_submit_retries))
        self._active_cycle_context: Optional[Dict[str, Any]] = None
        self._margin_failsafe_engaged = False
        self._telegram_notifier: Optional[TelegramNotifier] = None
        self._stop_notified = False
        self._start_notified = False
        if config.telegram_bot_token and config.telegram_chat_id:
            self._telegram_notifier = TelegramNotifier(
                config.telegram_bot_token,
                config.telegram_chat_id,
            )

    # ------------------------------------------------------------------ lifecycle
    def stop(self) -> None:
        self._stop_event.set()

    def _notify_strategy_started(self) -> None:
        if self._start_notified or not self._telegram_notifier:
            return
        self._start_notified = True
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message_lines = ["<b>TriHedge strategy started</b>"]
        symbols = ", ".join(plan.symbol for plan in self.config.symbols)
        if symbols:
            message_lines.append(f"Symbols: {symbols}")
        message_lines.append(f"Accounts: {', '.join(self._account_labels)}")
        message_lines.append(f"Time: {timestamp}")
        message = "\n".join(message_lines)
        try:
            ok = self._telegram_notifier.send_message(message)
        except Exception as exc:  # noqa: BLE001 - never block startup on notification errors
            logger.warning("Failed to send Telegram start notification: %s", exc)
            return
        if not ok:
            error = getattr(self._telegram_notifier, "last_error", None)
            if error:
                logger.warning("Failed to send Telegram start notification: %s", error)
            else:
                logger.warning("Failed to send Telegram start notification.")

    def _notify_strategy_stopped(self, reason: str, details: str = "") -> None:
        if self._stop_notified or not self._telegram_notifier:
            return
        self._stop_notified = True
        symbol = None
        context = self._active_cycle_context or {}
        if isinstance(context, dict):
            plan = context.get("plan")
            if isinstance(plan, SymbolPlan):
                symbol = plan.symbol
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message_lines = ["<b>TriHedge strategy stopped</b>"]
        if symbol:
            message_lines.append(f"Symbol: {symbol}")
        message_lines.append(f"Reason: {reason}")
        if details:
            message_lines.append(f"Details: {details}")
        message_lines.append(f"Time: {timestamp}")
        message = "\n".join(message_lines)
        try:
            ok = self._telegram_notifier.send_message(message)
        except Exception as exc:  # noqa: BLE001 - never block shutdown on notification errors
            logger.warning("Failed to send Telegram stop notification: %s", exc)
            return
        if not ok:
            error = getattr(self._telegram_notifier, "last_error", None)
            if error:
                logger.warning("Failed to send Telegram stop notification: %s", error)
            else:
                logger.warning("Failed to send Telegram stop notification.")

    def run(self) -> None:
        logger.info("TriHedgeHold Strategy booted with %d symbols", len(self.config.symbols))
        self._notify_strategy_started()
        symbol_count = len(self.config.symbols)
        account_count = len(self._clients)
        if symbol_count == 0 or account_count == 0:
            logger.error("No symbols or accounts configured; stopping.")
            return
        symbol_offset = self._current_symbol_index
        rounds_completed = 0
        epsilon_global = 1e-9
        try:
            while not self._stop_event.is_set():
                sessions = []
                for primary_idx in range(account_count):
                    symbol_idx = (symbol_offset + primary_idx) % symbol_count
                    plan = self.config.symbols[symbol_idx]
                    hedger_indices = [i for i in range(account_count) if i != primary_idx][:2]
                    limits = self._get_market_limits_with_retry(self._clients[primary_idx], plan.symbol)
                    sessions.append(
                        {
                            "primary_idx": primary_idx,
                            "plan": plan,
                            "hedgers": tuple(hedger_indices),
                            "limits": limits,
                            "status": "accumulating",
                            "hold_end_ts": None,
                            "next_allowed_time": 0.0,
                            "enter_start_time": time.time(),
                        }
                    )
                round_num = rounds_completed + 1
                logger.info(
                    "Round %d assignments: %s",
                    round_num,
                    ", ".join(
                        f"{self._account_labels[s['primary_idx']]}->{s['plan'].symbol} hedgers={[self._account_labels[i] for i in s['hedgers']]}"
                        for s in sessions
                    ),
                )
                
                # 初始化当前 round 的统计
                self._current_round = {
                    "round": round_num,
                    "enter": {
                        "time": {},
                        "slippage": {},
                        "fee": {},
                        "wear": {},
                    },
                    "exit": {
                        "time": {},
                        "slippage": {},
                        "fee": {},
                        "wear": {},
                    },
                }
                self._round_start_time = time.time()

                while not self._stop_event.is_set():
                    all_done = True
                    now = time.time()
                    for session in sessions:
                        if session["status"] == "done":
                            continue
                        all_done = False
                        if now < session.get("next_allowed_time", 0.0):
                            continue

                        primary_idx = session["primary_idx"]
                        plan = session["plan"]
                        hedger_indices = session["hedgers"]
                        limits: MarketConstraints = session["limits"]
                        status = session["status"]

                        self._active_cycle_context = {
                            "plan": plan,
                            "primary_idx": primary_idx,
                            "hedgers": tuple(hedger_indices),
                        }
                        try:
                            if status == "accumulating":
                                reference_price = self._compute_reference_price(
                                    self._clients[primary_idx],
                                    plan.symbol,
                                    "Bid",
                                    plan.entry_offset_bps,
                                )
                                if reference_price is None or reference_price <= 0:
                                    logger.warning("Unable to compute entry price for %s, retrying later", plan.symbol)
                                    session["next_allowed_time"] = now + 1.0
                                    continue

                                # 用当前价格估算 base 目标和 slice 数量
                                target_base = (
                                    round_to_precision(plan.target_notional / reference_price, limits.base_precision)
                                    if plan.target_notional > 0
                                    else 0.0
                                )
                                current_position = max(self._refresh_position(primary_idx, plan.symbol), 0.0)
                                target_remaining = max(target_base - current_position, 0.0)
                                # 当目标已经满足时，进入持仓阶段
                                if target_remaining <= epsilon_global:
                                    hold_minutes = plan.hold_minutes or self.config.hold_minutes
                                    enter_end_time = time.time()
                                    session["enter_end_time"] = enter_end_time
                                    session["hold_end_ts"] = enter_end_time + max(hold_minutes * 60, 1)
                                    session["status"] = "holding"
                                    
                                    # 计算 Enter 阶段的磨损
                                    self._calculate_enter_wear(session, primary_idx, hedger_indices, plan.symbol)
                                    
                                    logger.info(
                                        "%s target met (%.8f/%.8f %s); starting hold for %.1f minutes",
                                        plan.symbol,
                                        current_position,
                                        target_base,
                                        plan.symbol,
                                        hold_minutes,
                                    )
                                    continue

                                self._rebalance_exposure(
                                    symbol=plan.symbol,
                                    primary_idx=primary_idx,
                                    hedger_indices=hedger_indices,
                                    limits=limits,
                                    reference_price=reference_price,
                                    allow_additional_shorts=True,
                                    allow_reduce_shorts=False,
                                )

                                if plan.slice_notional and plan.slice_notional > 0:
                                    desired_slice = plan.slice_notional / max(reference_price, 1e-12)
                                    desired_slice = round_to_precision(desired_slice, limits.base_precision)
                                else:
                                    desired_slice = target_remaining
                                slice_quantity = min(desired_slice, target_remaining)
                                adjusted_slice = self._ensure_slice_meets_minimums(
                                    symbol=plan.symbol,
                                    remaining_quantity=target_remaining,
                                    reference_price=reference_price,
                                    limits=limits,
                                    slice_quantity=slice_quantity,
                                )
                                if adjusted_slice is None:
                                    # 剩余量不足最小单，直接进入持仓
                                    hold_minutes = plan.hold_minutes or self.config.hold_minutes
                                    session["hold_end_ts"] = time.time() + max(hold_minutes * 60, 1)
                                    session["status"] = "holding"
                                    logger.info(
                                        "%s remaining below venue minimum; start hold for %.1f minutes",
                                        plan.symbol,
                                        hold_minutes,
                                    )
                                    continue
                                slice_quantity = adjusted_slice
                                baseline_position = self._get_cached_position(primary_idx, plan.symbol)
                                order_index, response, rounded_price, rounded_qty = self._submit_limit_order(
                                    client=self._clients[primary_idx],
                                    client_idx=primary_idx,
                                    symbol=plan.symbol,
                                    side="Bid",
                                    quantity=slice_quantity,
                                    price=reference_price,
                                    post_only=self.config.post_only,
                                    reduce_only=False,
                                    limits=limits,
                                    account_label=self._account_labels[primary_idx],
                                )
                                if order_index is None:
                                    delay = self.config.slice_delay_seconds + self._random.uniform(
                                        0, max(self.config.slice_delay_jitter_seconds, 0)
                                    )
                                    session["next_allowed_time"] = now + max(delay, 0.0)
                                    continue
                                fills = self._wait_for_position_fill(
                                    client=self._clients[primary_idx],
                                    client_idx=primary_idx,
                                    symbol=plan.symbol,
                                    side="Bid",
                                    order_index=order_index,
                                    expected_quantity=rounded_qty,
                                    limit_price=rounded_price,
                                    limits=limits,
                                    baseline_position=baseline_position,
                                )
                                filled_base = sum(fill["quantity"] for fill in fills)
                                if filled_base > 0:
                                    self._record_maker_fill(plan.symbol, fills)
                                    logger.info(
                                        "Primary %s filled %.8f %s (target remaining %.8f)",
                                        self._account_labels[primary_idx],
                                        filled_base,
                                        plan.symbol,
                                        max(target_remaining - filled_base, 0.0),
                                    )
                                    self._dispatch_hedges(
                                        primary_idx,
                                        plan.symbol,
                                        "Ask",
                                        filled_base,
                                        hedger_indices,
                                        limits,
                                        reduce_only=False,
                                    )
                                    self._rebalance_exposure(
                                        symbol=plan.symbol,
                                        primary_idx=primary_idx,
                                        hedger_indices=hedger_indices,
                                        limits=limits,
                                        reference_price=reference_price,
                                        allow_additional_shorts=True,
                                        allow_reduce_shorts=False,
                                    )

                                delay = self.config.slice_delay_seconds + self._random.uniform(
                                    0, max(self.config.slice_delay_jitter_seconds, 0)
                                )
                                session["next_allowed_time"] = now + max(delay, 0.0)

                            elif status == "holding":
                                hold_end = session.get("hold_end_ts")
                                if hold_end is None:
                                    hold_minutes = plan.hold_minutes or self.config.hold_minutes
                                    session["hold_end_ts"] = time.time() + max(hold_minutes * 60, 1)
                                    continue
                                if time.time() >= hold_end:
                                    session["exit_start_time"] = time.time()
                                    session["status"] = "exiting"
                                    logger.info("%s hold complete; starting exit", plan.symbol)
                                else:
                                    # Allow others to proceed; minimal sleep to avoid busy loop.
                                    session["next_allowed_time"] = time.time() + min(
                                        self.config.order_poll_interval, 5.0
                                    )

                            elif status == "exiting":
                                reference_price = self._compute_reference_price(
                                    self._clients[primary_idx],
                                    plan.symbol,
                                    "Ask",
                                    plan.exit_offset_bps,
                                )
                                if reference_price is None or reference_price <= 0:
                                    logger.warning("Unable to compute exit price for %s, retrying later", plan.symbol)
                                    session["next_allowed_time"] = now + 1.0
                                    continue
                                current_position = max(self._refresh_position(primary_idx, plan.symbol), 0.0)
                                min_exit_qty = self._effective_min_quantity(reference_price, limits)
                                if current_position <= epsilon_global or current_position < min_exit_qty / 2:
                                    exit_end_time = time.time()
                                    session["exit_end_time"] = exit_end_time
                                    session["status"] = "done"
                                    
                                    # 计算 Exit 阶段的磨损
                                    self._calculate_exit_wear(session, primary_idx, hedger_indices, plan.symbol)
                                    
                                    self._log_position_snapshot(plan.symbol, primary_idx, hedger_indices)
                                    self._print_cycle_summary(plan.symbol)
                                    continue

                                if plan.slice_notional and plan.slice_notional > 0:
                                    slice_base = plan.slice_notional / max(reference_price, 1e-12)
                                else:
                                    slice_base = current_position
                                configured_slice_qty = max(
                                    round_to_precision(slice_base, limits.base_precision),
                                    limits.min_order_size,
                                )
                                slice_quantity = min(configured_slice_qty, current_position)
                                adjusted_exit = self._ensure_exit_slice_meets_minimums(
                                    symbol=plan.symbol,
                                    remaining_base=current_position,
                                    reference_price=reference_price,
                                    limits=limits,
                                    slice_quantity=slice_quantity,
                                )
                                if adjusted_exit is None:
                                    session["status"] = "done"
                                    self._log_position_snapshot(plan.symbol, primary_idx, hedger_indices)
                                    self._print_cycle_summary(plan.symbol)
                                    continue
                                slice_quantity = adjusted_exit
                                effective_min = self._effective_min_quantity(reference_price, limits)
                                if slice_quantity + epsilon_global < effective_min:
                                    logger.info(
                                        "Exit remainder %.8f %s below venue minimum %.8f; marking done.",
                                        current_position,
                                        plan.symbol,
                                        effective_min,
                                    )
                                    session["status"] = "done"
                                    self._log_position_snapshot(plan.symbol, primary_idx, hedger_indices)
                                    self._print_cycle_summary(plan.symbol)
                                    continue

                                baseline_position = self._get_cached_position(primary_idx, plan.symbol)
                                order_index, response, rounded_price, rounded_qty = self._submit_limit_order(
                                    client=self._clients[primary_idx],
                                    client_idx=primary_idx,
                                    symbol=plan.symbol,
                                    side="Ask",
                                    quantity=slice_quantity,
                                    price=reference_price,
                                    post_only=self.config.post_only,
                                    reduce_only=True,
                                    limits=limits,
                                    account_label=self._account_labels[primary_idx],
                                )
                                if order_index is None:
                                    delay = self.config.slice_delay_seconds + self._random.uniform(
                                        0, max(self.config.slice_delay_jitter_seconds, 0)
                                    )
                                    session["next_allowed_time"] = now + max(delay, 0.0)
                                    continue
                                fills = self._wait_for_position_fill(
                                    client=self._clients[primary_idx],
                                    client_idx=primary_idx,
                                    symbol=plan.symbol,
                                    side="Ask",
                                    order_index=order_index,
                                    expected_quantity=rounded_qty,
                                    limit_price=rounded_price,
                                    limits=limits,
                                    baseline_position=baseline_position,
                                )
                                filled_base = sum(fill["quantity"] for fill in fills)
                                if filled_base > 0:
                                    self._record_maker_fill(plan.symbol, fills)
                                    logger.info(
                                        "Primary %s closed %.8f %s (%.8f remaining est.)",
                                        self._account_labels[primary_idx],
                                        filled_base,
                                        plan.symbol,
                                        max(current_position - filled_base, 0.0),
                                    )
                                    self._dispatch_hedges(
                                        primary_idx,
                                        plan.symbol,
                                        "Bid",
                                        filled_base,
                                        hedger_indices,
                                        limits,
                                        reduce_only=True,
                                    )
                                    self._rebalance_exposure(
                                        symbol=plan.symbol,
                                        primary_idx=primary_idx,
                                        hedger_indices=hedger_indices,
                                        limits=limits,
                                        reference_price=reference_price,
                                        allow_additional_shorts=False,
                                        allow_reduce_shorts=True,
                                    )
                                remaining_base = max(self._get_cached_position(primary_idx, plan.symbol), 0.0)
                                if remaining_base <= epsilon_global or remaining_base < effective_min / 2:
                                    session["status"] = "done"
                                    self._log_position_snapshot(plan.symbol, primary_idx, hedger_indices)
                                    self._print_cycle_summary(plan.symbol)
                                delay = self.config.slice_delay_seconds + self._random.uniform(
                                    0, max(self.config.slice_delay_jitter_seconds, 0)
                                )
                                session["next_allowed_time"] = now + max(delay, 0.0)

                        except OrderSubmissionError as exc:
                            self._handle_order_failure(exc)
                        finally:
                            self._active_cycle_context = None

                    if all_done:
                        # 所有 session 完成，计算本轮总结并保存
                        self._finalize_round_stats()
                        break
                    # 防止空转占满 CPU
                    self._stop_event.wait(min(self.config.order_poll_interval, 0.5))

                rounds_completed += 1
                if self.config.run_once:
                    logger.info("run_once enabled, stopping after one round.")
                    break

                symbol_offset = (symbol_offset + 1) % symbol_count
                self._current_symbol_index = symbol_offset
                self._current_primary_index = 0
        except MarginFailsafeTriggered as exc:
            logger.critical("Margin failsafe triggered; shutting down strategy loop.")
            self._notify_strategy_stopped("margin_failsafe", str(exc))
        except Exception as exc:
            logger.exception("Strategy loop aborted: %s", exc)
            self._notify_strategy_stopped("unhandled_exception", str(exc))
            raise
        finally:
            self._active_cycle_context = None
            if self._stop_event.is_set():
                self._notify_strategy_stopped("stop_event", "stop() requested")

    # ------------------------------------------------------------------ core flow
    def _accumulate_position(self, primary_idx: int, hedgers: Sequence[int], plan: SymbolPlan) -> float:
        client = self._clients[primary_idx]
        limits = self._get_market_limits_with_retry(client, plan.symbol)
        target_base_total, configured_slice_qty, conversion_price = self._resolve_base_targets(client, plan, limits)
        total_base_filled = 0.0
        self._prime_positions(plan.symbol, [primary_idx, *hedgers])
        # primary 当前仓位
        current_position = self._get_cached_position(primary_idx, plan.symbol)
        target_remaining = target_base_total - current_position
        if target_remaining <= 0:
            logger.info(
                "Current %s position %.8f already meets/exceeds target %.8f, skipping accumulation.",
                plan.symbol,
                current_position,
                target_base_total,
            )
            return 0.0
        logger.info(
            "Converted target %.2f quote to %.8f base using price %.8f (slice target %.8f)",
            plan.target_notional,
            target_base_total,
            conversion_price,
            configured_slice_qty,
        )
        logger.info(
            "Need to accumulate %.8f %s to reach target %.8f (current %.8f).",
            target_remaining,
            plan.symbol,
            target_base_total,
            current_position,
        )

        while not self._stop_event.is_set() and target_remaining > 0:
            reference_price = self._compute_reference_price(client, plan.symbol, "Bid", plan.entry_offset_bps)
            if reference_price is None or reference_price <= 0:
                logger.warning("Unable to compute entry price for %s, retrying...", plan.symbol)
                self._stop_event.wait(3)
                continue
            self._rebalance_exposure(
                symbol=plan.symbol,
                primary_idx=primary_idx,
                hedger_indices=hedgers,
                limits=limits,
                reference_price=reference_price,
                allow_additional_shorts=True,
                allow_reduce_shorts=False,
            )

            slice_quantity = min(configured_slice_qty, target_remaining)
            continue_entry = self._ensure_slice_meets_minimums(
                symbol=plan.symbol,
                remaining_quantity=target_remaining,
                reference_price=reference_price,
                limits=limits,
                slice_quantity=slice_quantity,
            )
            # 剩下 remaining 太小时，continue_entry 返回 None，target_remaining 置 0，结束 accumulate
            if continue_entry is None:
                target_remaining = 0.0
                break
            slice_quantity = continue_entry

            baseline_position = self._get_cached_position(primary_idx, plan.symbol)
            # 默认主账号下买单。
            order_index, response, rounded_price, rounded_qty = self._submit_limit_order(
                client=client,
                client_idx=primary_idx,
                symbol=plan.symbol,
                side="Bid",
                quantity=slice_quantity,
                price=reference_price,
                post_only=self.config.post_only,
                reduce_only=False,
                limits=limits,
                account_label=self._account_labels[primary_idx],
            )
            if order_index is None:
                delay = self.config.slice_delay_seconds + self._random.uniform(
                    0, max(self.config.slice_delay_jitter_seconds, 0)
                )
                logger.info("No valid order generated for this slice; waiting %.2f seconds before retry.", delay)
                self._stop_event.wait(max(delay, 0.0))
                continue
            fills = self._wait_for_position_fill(
                client=client,
                client_idx=primary_idx,
                symbol=plan.symbol,
                side="Bid",
                order_index=order_index,
                expected_quantity=rounded_qty,
                limit_price=rounded_price,
                limits=limits,
                baseline_position=baseline_position,
            )
            filled_base = sum(fill["quantity"] for fill in fills)
            if filled_base <= 0:
                logger.info("Slice produced no fills, retrying another order...")
                delay = self.config.slice_delay_seconds + self._random.uniform(
                    0, max(self.config.slice_delay_jitter_seconds, 0)
                )
                self._stop_event.wait(max(delay, 0.0))
                continue
            self._record_maker_fill(plan.symbol, fills)

            total_base_filled += filled_base
            target_remaining = max(target_remaining - filled_base, 0.0)
            logger.info(
                "Primary %s filled %.8f %s (%.8f / %.8f base done)",
                self._account_labels[primary_idx],
                filled_base,
                plan.symbol,
                target_base_total - target_remaining,
                target_base_total,
            )

            self._dispatch_hedges(primary_idx, plan.symbol, "Ask", filled_base, hedgers, limits, reduce_only=False)
            self._rebalance_exposure(
                symbol=plan.symbol,
                primary_idx=primary_idx,
                hedger_indices=hedgers,
                limits=limits,
                reference_price=reference_price,
                allow_additional_shorts=True,
                allow_reduce_shorts=False,
            )
            delay = self.config.slice_delay_seconds + self._random.uniform(
                0, max(self.config.slice_delay_jitter_seconds, 0)
            )
            self._stop_event.wait(max(delay, 0.0))  # jittered delay between slices so we don't spam the venue

        reference_price = self._compute_reference_price(client, plan.symbol, "Bid", plan.entry_offset_bps)
        self._rebalance_exposure(
            symbol=plan.symbol,
            primary_idx=primary_idx,
            hedger_indices=hedgers,
            limits=limits,
            reference_price=reference_price,
            allow_additional_shorts=True,
            allow_reduce_shorts=True,
        )
        return total_base_filled

    def _flatten_position(
        self,
        primary_idx: int,
        hedgers: Sequence[int],
        plan: SymbolPlan,
        target_base_quantity: float,
    ) -> None:
        client = self._clients[primary_idx]
        limits = self._get_market_limits_with_retry(client, plan.symbol)
        remaining_base = float(target_base_quantity)
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        if remaining_base <= 0:
            logger.warning("No base quantity to unwind for %s", plan.symbol)
            return
        self._prime_positions(plan.symbol, [primary_idx, *hedgers])
        reference_price = self._compute_reference_price(client, plan.symbol, "Ask", plan.exit_offset_bps)
        self._rebalance_exposure(
            symbol=plan.symbol,
            primary_idx=primary_idx,
            hedger_indices=hedgers,
            limits=limits,
            reference_price=reference_price,
            allow_additional_shorts=False,
            allow_reduce_shorts=True,
        )

        while not self._stop_event.is_set() and remaining_base > limits.min_order_size / 2:
            reference_price = self._compute_reference_price(client, plan.symbol, "Ask", plan.exit_offset_bps)
            if reference_price is None or reference_price <= 0:
                logger.warning("Unable to compute exit price for %s, retrying...", plan.symbol)
                self._stop_event.wait(3)
                continue
            if plan.slice_notional and plan.slice_notional > 0:
                slice_base = plan.slice_notional / max(reference_price, 1e-12)
            else:
                slice_base = remaining_base
            configured_slice_qty = max(
                round_to_precision(slice_base, limits.base_precision),
                limits.min_order_size,
            )
            slice_quantity = min(configured_slice_qty, remaining_base)
            adjusted_exit = self._ensure_exit_slice_meets_minimums(
                symbol=plan.symbol,
                remaining_base=remaining_base,
                reference_price=reference_price,
                limits=limits,
                slice_quantity=slice_quantity,
            )
            if adjusted_exit is None:
                break
            slice_quantity = adjusted_exit
            min_exit_qty = self._effective_min_quantity(reference_price, limits)
            if slice_quantity + epsilon < min_exit_qty:
                logger.info(
                    "Exit remainder %.8f %s below venue minimum %.8f; stopping flatten loop.",
                    remaining_base,
                    plan.symbol,
                    min_exit_qty,
                )
                break

            baseline_position = self._get_cached_position(primary_idx, plan.symbol)
            order_index, response, rounded_price, rounded_qty = self._submit_limit_order(
                client=client,
                client_idx=primary_idx,
                symbol=plan.symbol,
                side="Ask",
                quantity=slice_quantity,
                price=reference_price,
                post_only=self.config.post_only,
                reduce_only=True,
                limits=limits,
                account_label=self._account_labels[primary_idx],
            )
            if order_index is None:
                delay = self.config.slice_delay_seconds + self._random.uniform(
                    0, max(self.config.slice_delay_jitter_seconds, 0)
                )
                self._stop_event.wait(max(delay, 0.0))
                continue
            fills = self._wait_for_position_fill(
                client=client,
                client_idx=primary_idx,
                symbol=plan.symbol,
                side="Ask",
                order_index=order_index,
                expected_quantity=rounded_qty,
                limit_price=rounded_price,
                limits=limits,
                baseline_position=baseline_position,
            )
            filled_base = sum(fill["quantity"] for fill in fills)
            if filled_base <= 0:
                logger.info("Exit slice produced no fills, retrying...")
                delay = self.config.slice_delay_seconds + self._random.uniform(
                    0, max(self.config.slice_delay_jitter_seconds, 0)
                )
                self._stop_event.wait(max(delay, 0.0))
                continue
            self._record_maker_fill(plan.symbol, fills)

            remaining_base = max(remaining_base - filled_base, 0.0)
            logger.info(
                "Primary %s closed %.8f %s (%.8f remaining)",
                self._account_labels[primary_idx],
                filled_base,
                plan.symbol,
                remaining_base,
            )
            self._dispatch_hedges(primary_idx, plan.symbol, "Bid", filled_base, hedgers, limits, reduce_only=True)
            self._rebalance_exposure(
                symbol=plan.symbol,
                primary_idx=primary_idx,
                hedger_indices=hedgers,
                limits=limits,
                reference_price=reference_price,
                allow_additional_shorts=False,
                allow_reduce_shorts=True,
            )
            delay = self.config.slice_delay_seconds + self._random.uniform(
                0, max(self.config.slice_delay_jitter_seconds, 0)
            )
            self._stop_event.wait(max(delay, 0.0))  # allow partial book updates / hedges to catch up

        self._log_position_snapshot(plan.symbol, primary_idx, hedgers)
        logger.info("Exit leg for %s finished; remaining base %.8f", plan.symbol, remaining_base)

    def _hold_position(self, plan: SymbolPlan) -> None:
        hold_minutes = plan.hold_minutes or self.config.hold_minutes
        hold_seconds = max(hold_minutes * 60, 1)
        logger.info("Holding %s exposure for %.1f minutes", plan.symbol, hold_minutes)
        elapsed = 0.0
        while not self._stop_event.is_set() and elapsed < hold_seconds:
            sleep_window = min(30, hold_seconds - elapsed)
            self._stop_event.wait(sleep_window)
            elapsed += sleep_window

    # ------------------------------------------------------------------ helpers
    def _submit_limit_order(
        self,
        client: LighterClient,
        client_idx: int,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool,
        reduce_only: bool,
        limits: MarketConstraints,
        account_label: str,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], float, float]:
        rounded_qty = max(round_to_precision(quantity, limits.base_precision), limits.min_order_size)
        rounded_price = round_to_tick_size(price, limits.tick_size)
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
        tif_upper = self._primary_time_in_force
        post_only_flag = post_only
        if tif_upper in ("FOK", "IOC") and post_only_flag:
            logger.warning("PostOnly disabled because timeInForce=%s is incompatible.", tif_upper)
            post_only_flag = False
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
        try:
            response = self._execute_with_nonce_retry(
                client,
                order,
                account_label,
                max_retries=self._order_submit_retries,
            )
        except Exception as exc:
            logger.error(
                "Limit order submission exception for %s on %s: %s",
                symbol,
                account_label,
                exc,
            )
            raise OrderSubmissionError(symbol, client_idx, account_label, str(exc)) from exc
        if isinstance(response, dict) and response.get("error"):
            logger.error("Limit order rejected for %s on %s: %s", symbol, account_label, response["error"])
            raise OrderSubmissionError(symbol, client_idx, account_label, str(response["error"]))
        order_index = self._extract_order_index(response)
        return order_index, response, rounded_price, rounded_qty

    def _wait_for_position_fill(
        self,
        client: LighterClient,
        client_idx: int,
        symbol: str,
        side: str,
        order_index: Optional[str],
        expected_quantity: float,
        limit_price: float,
        limits: MarketConstraints,
        baseline_position: float,
    ) -> List[Dict[str, float]]:
        if not order_index:
            return []
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        start = time.time()
        deadline = start + self.config.slice_fill_timeout
        filled = 0.0
        last_position = baseline_position
        while not self._stop_event.is_set() and time.time() < deadline and filled + epsilon < expected_quantity:
            self._stop_event.wait(self.config.order_poll_interval)
            current_position = self._refresh_position(client_idx, symbol)
            delta = current_position - last_position
            last_position = current_position
            signed_delta = delta if side == "Bid" else -delta
            if signed_delta > epsilon:
                filled += signed_delta
        if order_index:
            cancel_result = self._cancel_order_with_retry(client, order_index, symbol)
            if isinstance(cancel_result, dict) and cancel_result.get("error"):
                logger.debug("Cancel %s for %s response: %s", order_index, symbol, cancel_result["error"])
        final_position = last_position
        quantity_delta = final_position - baseline_position if side == "Bid" else baseline_position - final_position
        quantity_delta = round_to_precision(max(quantity_delta, 0.0), limits.base_precision)
        if quantity_delta <= epsilon:
            return []
        
        # 这里返回 limit_price 实际上是有误差的，有可能会用 aggressive price
        return [{"quantity": quantity_delta, "price": limit_price}]

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
        exposure_caps: Dict[int, float] = {}
        if reduce_only:
            for idx in hedger_indices:
                exposure_caps[idx] = max(-self._get_cached_position(idx, symbol), 0.0)
            logger.debug(
                "Reduce-only hedge planning for %s: qty=%.8f exposures=%s",
                symbol,
                qty,
                {self._account_labels[idx]: cap for idx, cap in exposure_caps.items()},
            )

        # 随机分两份
        low, high = self.config.random_split_range
        first_split = self._random.uniform(low, high)
        allocations: List[float] = []
        first_qty = round_to_precision(qty * first_split, limits.base_precision)
        allocations.append(first_qty)
        allocations.append(qty - first_qty)
    
        aggressive_price, min_trade_qty = self._estimate_minimum_trade_qty(symbol, side, hedger_indices, limits)
        if not reduce_only and len(hedger_indices) > 1:
            # 如果数量太小，无法平均分给所有 hedger，那就只分给其中一个 hedger（轮流选）
            if qty < min_trade_qty * len(hedger_indices):
                target_pos = self._select_small_fill_target(hedger_indices)
                allocations = [0.0 for _ in hedger_indices]
                allocations[target_pos] = round_to_precision(qty, limits.base_precision)
        if reduce_only:
            allocations = self._apply_exposure_caps(
                allocations, hedger_indices, exposure_caps, min_trade_qty, limits
            )
            logger.debug(
                "Reduce-only hedge allocations after caps for %s: %s",
                symbol,
                allocations,
            )

        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        for list_index, hedger_idx in enumerate(hedger_indices):
            alloc = allocations[list_index]
            rounded = round_to_precision(alloc, limits.base_precision)
            if rounded <= 0:
                logger.debug(
                    "Skipping hedge %s for %s because rounded allocation %.8f <= 0",
                    self._account_labels[hedger_idx],
                    symbol,
                    rounded,
                )
                continue
            
            min_required = self._effective_min_quantity(aggressive_price, limits)
            # 这里的检查应该是多余的？
            if rounded + epsilon < min_required:
                logger.debug(
                    "Hedge qty %.8f below venue minimum %.8f at price %.8f for %s, left as net exposure",
                    rounded,
                    min_required,
                    aggressive_price,
                    self._account_labels[hedger_idx],
                )
                continue
            baseline_position = self._get_cached_position(hedger_idx, symbol)
            order_index, response, rounded_price, rounded_qty = self._submit_limit_order(
                client=self._clients[hedger_idx],
                client_idx=hedger_idx,
                symbol=symbol,
                side=side,
                quantity=rounded,
                price=aggressive_price,
                post_only=False,
                reduce_only=reduce_only,
                limits=limits,
                account_label=self._account_labels[hedger_idx],
            )
            if order_index is None:
                logger.info(
                    "Hedge order submmit failed"
                )
                continue
            fills = self._wait_for_position_fill(
                client=self._clients[hedger_idx],
                client_idx=hedger_idx,
                symbol=symbol,
                side=side,
                order_index=order_index,
                expected_quantity=rounded_qty,
                limit_price=rounded_price,
                limits=limits,
                baseline_position=baseline_position,
            )
            filled_base = sum(fill["quantity"] for fill in fills)
            if filled_base <= 0:
                continue
            self._record_hedge_fill(hedger_idx, symbol, fills)
        if hedger_indices:
            self._log_position_snapshot(symbol, primary_idx, hedger_indices)


    def _apply_exposure_caps(
        self,
        allocations: List[float],
        hedger_indices: Sequence[int],
        exposure_caps: Dict[int, float],
        min_trade_qty: float,
        limits: MarketConstraints,
    ) -> List[float]:
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        adjusted = [0.0 for _ in allocations]
        remaining_caps = [max(exposure_caps.get(idx, 0.0), 0.0) for idx in hedger_indices]
        qty_left = round_to_precision(sum(allocations), limits.base_precision)

        for idx, planned in enumerate(allocations):
            allowance = min(planned, remaining_caps[idx])
            allowance = round_to_precision(allowance, limits.base_precision)
            adjusted[idx] = allowance
            remaining_caps[idx] = max(round_to_precision(remaining_caps[idx] - allowance, limits.base_precision), 0.0)
            qty_left = max(round_to_precision(qty_left - allowance, limits.base_precision), 0.0)

        if qty_left > epsilon:
            for idx in range(len(adjusted)):
                if qty_left <= epsilon:
                    break
                allowance = min(remaining_caps[idx], qty_left)
                allowance = round_to_precision(allowance, limits.base_precision)
                adjusted[idx] += allowance
                remaining_caps[idx] = max(round_to_precision(remaining_caps[idx] - allowance, limits.base_precision), 0.0)
                qty_left = max(round_to_precision(qty_left - allowance, limits.base_precision), 0.0)

        total_alloc = round_to_precision(sum(adjusted), limits.base_precision)
        if total_alloc < min_trade_qty:
            # Not enough to trade; zero everything so caller can decide.
            return [0.0 for _ in adjusted]

        # For sub-min allocations, merge the smaller ones into the largest allocation.
        largest_idx = max(range(len(adjusted)), key=lambda i: adjusted[i])
        for idx in range(len(adjusted)):
            if idx == largest_idx:
                continue
            if 0 < adjusted[idx] < min_trade_qty:
                transferable = adjusted[idx]
                adjusted[idx] = 0.0
                adjusted[largest_idx] = round_to_precision(adjusted[largest_idx] + transferable, limits.base_precision)

        # If the largest allocation is still below the minimum, drop others and move everything there (within cap).
        if adjusted[largest_idx] < min_trade_qty:
            adjusted = [0.0 for _ in adjusted]
            cap = exposure_caps.get(hedger_indices[largest_idx], total_alloc)
            adjusted[largest_idx] = min(total_alloc, cap)

        return adjusted

    def _effective_min_quantity(self, price: float, limits: MarketConstraints) -> float:
        # 根据 limit 和 11u 返回最小下单数量，
        min_quote = max(limits.min_quote_value, MIN_QUOTE_THRESHOLD)
        quote_based = min_quote / max(price, 1e-12)
        return max(limits.min_order_size, quote_based)

    def _ensure_slice_meets_minimums(
        self,
        symbol: str,
        remaining_quantity: float,
        reference_price: float,
        limits: MarketConstraints,
        slice_quantity: float,
    ) -> Optional[float]:
        min_trade_qty = self._effective_min_quantity(reference_price, limits)
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        if slice_quantity + epsilon >= min_trade_qty:
            return slice_quantity
        if remaining_quantity <= min_trade_qty + epsilon:
            logger.info(
                "Remaining target %.8f %s below venue minimum %.8f, finishing accumulation.",
                remaining_quantity,
                symbol,
                min_trade_qty,
            )
            return None
        # 当本次 slice 小于交易所最小下单量，但剩余目标量还不止这一口时，强制把 slice 调整到 min_trade_qty。
        logger.info(
            "Adjusting slice qty from %.8f to venue minimum %.8f for %s.",
            slice_quantity,
            min_trade_qty,
            symbol,
        )
        return min_trade_qty

    def _ensure_exit_slice_meets_minimums(
        self,
        symbol: str,
        remaining_base: float,
        reference_price: float,
        limits: MarketConstraints,
        slice_quantity: float,
    ) -> Optional[float]:
        min_trade_qty = self._effective_min_quantity(reference_price, limits)
        epsilon = max(1e-12, 10 ** (-limits.base_precision))
        if slice_quantity + epsilon >= min_trade_qty:
            return slice_quantity
        if remaining_base <= min_trade_qty + epsilon:
            logger.info(
                "Remaining exit size %.8f %s below venue minimum %.8f, finishing exit leg.",
                remaining_base,
                symbol,
                min_trade_qty,
            )
            return None
        logger.info(
            "Adjusting exit slice qty from %.8f to venue minimum %.8f for %s.",
            slice_quantity,
            min_trade_qty,
            symbol,
        )
        return min(min_trade_qty, remaining_base)

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

    @staticmethod
    def _extract_order_index(payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        for key in ("orderIndex", "order_index", "clientOrderIndex", "id", "orderId"):
            value = payload.get(key)
            if value is not None:
                return str(value)
        return None

    def _estimate_minimum_trade_qty(
        self,
        symbol: str,
        side: str,
        hedger_indices: Sequence[int],
        limits: MarketConstraints,
    ) -> float:
        if not hedger_indices:
            return None, limits.min_order_size
        reference_idx = hedger_indices[0]
        price = self._compute_aggressive_price(self._clients[reference_idx], symbol, side, limits.tick_size)
        return price, self._effective_min_quantity(price, limits)

    def _select_small_fill_target(self, hedger_indices: Sequence[int]) -> int:
        if not hedger_indices:
            raise StrategyConfigError("hedger_indices cannot be empty")
        key = tuple(hedger_indices)
        pointer = self._small_fill_selector.get(key, -1)
        pointer = (pointer + 1) % len(hedger_indices)
        self._small_fill_selector[key] = pointer
        return pointer

    def _prime_positions(self, symbol: str, indices: Sequence[int]) -> None:
        # 同时更新三家的仓位
        for idx in indices:
            self._refresh_position(idx, symbol)

    def _refresh_position(self, account_idx: int, symbol: str) -> float:
        """刷新仓位，返回数量"""
        client = self._clients[account_idx]
        time.sleep(0.1)
        positions = client.get_positions(symbol)
        value = 0.0
        entry_price = 0.0
        if isinstance(positions, list):
            for entry in positions:
                if not isinstance(entry, dict):
                    continue
                raw = entry.get("netQuantity") or entry.get("rawSize") or entry.get("size")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                # 获取均价
                entry_price = float(entry.get("entryPrice") or entry.get("avgEntryPrice") or 0)
                break
        self._position_cache.setdefault(account_idx, {})[symbol] = value
        if entry_price > 0:
            self._position_price_cache.setdefault(account_idx, {})[symbol] = entry_price
        return value
    
    def _get_position_with_price(self, account_idx: int, symbol: str) -> Tuple[float, float]:
        """获取仓位数量和均价"""
        qty = self._get_cached_position(account_idx, symbol)
        price = self._position_price_cache.get(account_idx, {}).get(symbol, 0.0)
        if price <= 0:
            # 如果缓存中没有均价，刷新一次
            self._refresh_position(account_idx, symbol)
            price = self._position_price_cache.get(account_idx, {}).get(symbol, 0.0)
        return qty, price

    def _get_cached_position(self, account_idx: int, symbol: str) -> float:
        account_store = self._position_cache.setdefault(account_idx, {})
        if symbol not in account_store:
            return self._refresh_position(account_idx, symbol)
        return account_store[symbol]

    def _rebalance_exposure(
        self,
        symbol: str,
        primary_idx: int,
        hedger_indices: Sequence[int],
        limits: MarketConstraints,
        reference_price: Optional[float],
        *,
        allow_additional_shorts: bool,
        allow_reduce_shorts: bool,
    ) -> None:
        if not hedger_indices:
            return
        maker_qty = max(self._get_cached_position(primary_idx, symbol), 0.0)
        hedge_total = sum(max(-self._get_cached_position(idx, symbol), 0.0) for idx in hedger_indices)
        net_base = maker_qty - hedge_total
        epsilon = max(1e-12, 10 ** (-limits.base_precision))

        if net_base >= epsilon and allow_additional_shorts:
            price = self._compute_aggressive_price(self._clients[primary_idx], symbol, "Ask", limits.tick_size)
            min_gap = self._effective_min_quantity(price, limits)
            if net_base >= min_gap:
                logger.info(
                    "Entry hedge lag detected for %s (gap %.8f ≈ %.2f quote); sending catch-up short.",
                    symbol,
                    net_base,
                    net_base * price,
                )
                self._dispatch_hedges(primary_idx, symbol, "Ask", net_base, hedger_indices, limits, reduce_only=False)

        elif net_base <= -epsilon and allow_reduce_shorts:
            price = self._compute_aggressive_price(self._clients[primary_idx], symbol, "Bid", limits.tick_size)
            min_gap = self._effective_min_quantity(price, limits)
            if abs(net_base) >= min_gap:
                logger.info(
                    "Flatten hedge lag detected for %s (gap %.8f ≈ %.2f quote); reducing hedge shorts.",
                    symbol,
                    abs(net_base),
                    abs(net_base) * price,
                )
                self._dispatch_hedges(primary_idx, symbol, "Bid", abs(net_base), hedger_indices, limits, reduce_only=True)

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
        snapshot: List[str] = []
        for idx in participants:
            qty = self._get_cached_position(idx, symbol)
            snapshot.append(self._format_position_entry(symbol, idx, qty))
        if snapshot:
            logger.info("Positions[%s]: %s", symbol, "; ".join(snapshot))

    def _format_position_entry(self, symbol: str, account_idx: int, qty: float) -> str:
        label = self._account_labels[account_idx]
        descr = self._describe_side(qty)
        return f"{label}={qty:.6f} {symbol} ({descr})"

    @staticmethod
    def _describe_side(value: Optional[float]) -> str:
        if value is None or abs(value) <= 1e-12:
            return "flat"
        return "long" if value > 0 else "short"

    def _record_maker_fill(self, symbol: str, fills: Iterable[Dict[str, float]]) -> None:
        stats = self._maker_price_stats.setdefault(symbol, {"qty": 0.0, "notional": 0.0})
        for fill in fills:
            qty = float(fill["quantity"])
            price = float(fill["price"])
            stats["qty"] += qty
            stats["notional"] += qty * price

    def _record_hedge_fill(
        self,
        hedger_idx: int,
        symbol: str,
        fills: Iterable[Dict[str, float]],
    ) -> None:
        stats = self._hedge_price_stats.setdefault(symbol, {}).setdefault(
            hedger_idx,
            {"qty": 0.0, "notional": 0.0},
        )
        for fill in fills or []:
            qty = float(fill.get("quantity") or 0.0)
            price = float(fill.get("price") or 0.0)
            if qty <= 0 or price <= 0:
                continue
            stats["qty"] += qty
            stats["notional"] += qty * price

    def _calculate_enter_wear(
        self,
        session: Dict[str, Any],
        primary_idx: int,
        hedger_indices: Sequence[int],
        symbol: str,
    ) -> None:
        """计算 Enter 阶段的磨损（价差 + 手续费）"""
        try:
            # 获取主账户和对冲账户的仓位和均价
            primary_qty, primary_price = self._get_position_with_price(primary_idx, symbol)
            
            if primary_qty <= 0 or primary_price <= 0:
                logger.warning("Cannot calculate enter wear for %s: invalid position", symbol)
                return
            
            # 获取对冲账户的仓位和均价
            hedge_total_qty = 0.0
            hedge_weighted_price = 0.0
            for hedger_idx in hedger_indices:
                hedge_qty, hedge_price = self._get_position_with_price(hedger_idx, symbol)
                # 对冲账户应该是空头（负数）
                hedge_qty = abs(hedge_qty) if hedge_qty < 0 else 0.0
                if hedge_qty > 0 and hedge_price > 0:
                    hedge_total_qty += hedge_qty
                    hedge_weighted_price += hedge_qty * hedge_price
            
            if hedge_total_qty <= 0:
                logger.warning("Cannot calculate enter wear for %s: no hedge position", symbol)
                return
            
            hedge_avg_price = hedge_weighted_price / hedge_total_qty
            
            # 计算价差（Enter: 主账户买入 Bid，对冲账户卖出 Ask）
            # 价差 = (对冲Ask价格 - 主账户Bid价格) * 数量
            slippage = abs(hedge_avg_price - primary_price) * min(primary_qty, hedge_total_qty)
            
            # 计算手续费
            # 主账户是 maker (0.002%), 对冲账户是 taker (0.02%)
            primary_value = primary_qty * primary_price
            hedge_value = hedge_total_qty * hedge_avg_price
            primary_fee = primary_value * self._maker_fee_rate
            hedge_fee = hedge_value * self._taker_fee_rate
            total_fee = primary_fee + hedge_fee
            
            # 总磨损
            total_wear = slippage + total_fee
            
            # 计算时间
            enter_start = session.get("enter_start_time", 0)
            enter_end = session.get("enter_end_time", time.time())
            enter_duration = enter_end - enter_start
            
            # 记录到当前 round
            account_label = self._account_labels[primary_idx]
            key = f"{account_label}_{symbol}"
            
            if self._current_round:
                self._current_round["enter"]["time"][key] = enter_duration
                self._current_round["enter"]["slippage"][key] = slippage
                self._current_round["enter"]["fee"][key] = total_fee
                self._current_round["enter"]["wear"][key] = total_wear
            
            logger.info(
                "[ENTER WEAR] %s: slippage=%.6f, fee=%.6f, total=%.6f, duration=%.1fs",
                key,
                slippage,
                total_fee,
                total_wear,
                enter_duration,
            )
            
            # 发送 Enter 阶段的 Telegram 通知
            if self._telegram_notifier:
                self._send_enter_notification(key, slippage, total_fee, total_wear, enter_duration)
        except Exception as exc:
            logger.error("Error calculating enter wear: %s", exc, exc_info=True)
    
    def _calculate_exit_wear(
        self,
        session: Dict[str, Any],
        primary_idx: int,
        hedger_indices: Sequence[int],
        symbol: str,
    ) -> None:
        """计算 Exit 阶段的磨损（价差 + 手续费）"""
        try:
            # 获取主账户的 entry price（从 Enter 阶段保存的均价）
            # 如果主账户已经平仓，从缓存中获取之前的 entry price
            primary_entry_qty, primary_entry_price = self._get_position_with_price(primary_idx, symbol)
            
            # 如果主账户仓位为0，尝试从 maker_price_stats 获取均价
            if primary_entry_qty <= 0 or primary_entry_price <= 0:
                maker_stats = self._maker_price_stats.get(symbol)
                if maker_stats and maker_stats.get("qty", 0) > 0:
                    primary_entry_price = maker_stats["notional"] / maker_stats["qty"]
                    primary_entry_qty = maker_stats["qty"]
            
            if primary_entry_price <= 0:
                logger.warning("Cannot calculate exit wear for %s: no entry price", symbol)
                return
            
            # 获取当前市场价格（主账户卖出价格）
            client = self._clients[primary_idx]
            limits = session.get("limits")
            if not limits:
                limits = self._get_market_limits_with_retry(client, symbol)
            
            # 获取当前市场价格（用于计算价差）
            current_price = self._compute_reference_price(client, symbol, "Ask", 0) or primary_entry_price
            
            # 获取对冲账户的仓位和均价（Exit 阶段，对冲账户买入 Bid）
            hedge_total_qty = 0.0
            hedge_weighted_price = 0.0
            for hedger_idx in hedger_indices:
                hedge_qty, hedge_price = self._get_position_with_price(hedger_idx, symbol)
                # Exit 阶段，对冲账户应该是多头（正数，因为已经平仓对冲）
                hedge_qty = abs(hedge_qty) if hedge_qty > 0 else 0.0
                if hedge_qty > 0 and hedge_price > 0:
                    hedge_total_qty += hedge_qty
                    hedge_weighted_price += hedge_qty * hedge_price
            
            if hedge_total_qty <= 0:
                logger.warning("Cannot calculate exit wear for %s: no hedge position", symbol)
                return
            
            hedge_avg_price = hedge_weighted_price / hedge_total_qty
            
            # 计算价差（Exit: 主账户卖出 Ask，对冲账户买入 Bid）
            # 价差 = (主账户Ask价格 - 对冲Bid价格) * 数量
            exit_qty = min(primary_entry_qty, hedge_total_qty)
            slippage = abs(current_price - hedge_avg_price) * exit_qty
            
            # 计算手续费
            # 主账户是 maker (0.002%), 对冲账户是 taker (0.02%)
            # 使用实际成交数量计算
            primary_value = exit_qty * current_price
            hedge_value = exit_qty * hedge_avg_price  # 只计算匹配的数量
            primary_fee = primary_value * self._maker_fee_rate
            hedge_fee = hedge_value * self._taker_fee_rate
            total_fee = primary_fee + hedge_fee
            
            # 总磨损
            total_wear = slippage + total_fee
            
            # 计算时间
            exit_start = session.get("exit_start_time")
            if not exit_start:
                # 如果 exit_start_time 未设置，从 holding 结束时间开始
                exit_start = session.get("hold_end_ts", time.time())
                session["exit_start_time"] = exit_start
            exit_end = session.get("exit_end_time", time.time())
            exit_duration = exit_end - exit_start
            
            # 记录到当前 round
            account_label = self._account_labels[primary_idx]
            key = f"{account_label}_{symbol}"
            
            if self._current_round:
                self._current_round["exit"]["time"][key] = exit_duration
                self._current_round["exit"]["slippage"][key] = slippage
                self._current_round["exit"]["fee"][key] = total_fee
                self._current_round["exit"]["wear"][key] = total_wear
            
            logger.info(
                "[EXIT WEAR] %s: slippage=%.6f, fee=%.6f, total=%.6f, duration=%.1fs",
                key,
                slippage,
                total_fee,
                total_wear,
                exit_duration,
            )
            
            # 发送 Exit 阶段的 Telegram 通知
            if self._telegram_notifier:
                self._send_exit_notification(key, slippage, total_fee, total_wear, exit_duration)
        except Exception as exc:
            logger.error("Error calculating exit wear: %s", exc, exc_info=True)
    
    def _finalize_round_stats(self) -> None:
        """完成本轮统计，计算总结并保存"""
        if not self._current_round:
            return
        
        try:
            # 计算时间统计
            enter_times = self._current_round["enter"]["time"]
            exit_times = self._current_round["exit"]["time"]
            
            # 总时间 = max(所有 enter 时间) + max(所有 exit 时间)
            max_enter_time = max(enter_times.values()) if enter_times else 0.0
            max_exit_time = max(exit_times.values()) if exit_times else 0.0
            total_time = max_enter_time + max_exit_time
            
            # 计算总磨损
            enter_wears = self._current_round["enter"]["wear"]
            exit_wears = self._current_round["exit"]["wear"]
            total_enter_wear = sum(enter_wears.values())
            total_exit_wear = sum(exit_wears.values())
            total_wear = total_enter_wear + total_exit_wear
            
            # 计算总价差和手续费
            total_enter_slippage = sum(self._current_round["enter"]["slippage"].values())
            total_exit_slippage = sum(self._current_round["exit"]["slippage"].values())
            total_enter_fee = sum(self._current_round["enter"]["fee"].values())
            total_exit_fee = sum(self._current_round["exit"]["fee"].values())
            
            # 填充 summary
            self._current_round["summary"] = {
                "round_times": {
                    "enter": max_enter_time,
                    "exit": max_exit_time,
                    "total": total_time,
                },
                "wears": {
                    "enter": total_enter_wear,
                    "exit": total_exit_wear,
                    "total": total_wear,
                },
                "slippage": {
                    "enter": total_enter_slippage,
                    "exit": total_exit_slippage,
                    "total": total_enter_slippage + total_exit_slippage,
                },
                "fee": {
                    "enter": total_enter_fee,
                    "exit": total_exit_fee,
                    "total": total_enter_fee + total_exit_fee,
                },
                "totaltime": total_time,
            }
            
            # 保存到列表
            self._round_stats.append(self._current_round.copy())
            
            # 保存到 JSON 文件
            self._save_round_stats()
            
            # 发送 Telegram 通知
            self._send_round_notification()
            
            logger.info(
                "[ROUND SUMMARY] Round %d: total_wear=%.6f, total_time=%.1fs",
                self._current_round["round"],
                total_wear,
                total_time,
            )
            
            # 重置 _current_round 以便下一个 round 使用
            self._current_round = None
        except Exception as exc:
            logger.error("Error finalizing round stats: %s", exc, exc_info=True)
            # 即使出错也要重置，避免影响下一个 round
            self._current_round = None
    
    def _save_round_stats(self) -> None:
        """保存轮次统计到 JSON 文件"""
        try:
            stats_file = "tri_hedge_round_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self._round_stats, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Round stats saved to %s", stats_file)
        except Exception as exc:
            logger.error("Error saving round stats: %s", exc, exc_info=True)
    
    def _send_enter_notification(self, key: str, slippage: float, fee: float, wear: float, duration: float) -> None:
        """发送 Enter 阶段的 Telegram 通知"""
        if not self._telegram_notifier:
            return
        
        try:
            duration_minutes = duration / 60.0
            message = f"<b>Enter Phase Complete: {key}</b>\n"
            message += f"Slippage: {slippage:.6f}\n"
            message += f"Fee: {fee:.6f}\n"
            message += f"Total Wear: {wear:.6f}\n"
            message += f"Duration: {duration_minutes:.2f} min"
            self._telegram_notifier.send_message(message)
        except Exception as exc:
            logger.error("Error sending enter notification: %s", exc, exc_info=True)
    
    def _send_exit_notification(self, key: str, slippage: float, fee: float, wear: float, duration: float) -> None:
        """发送 Exit 阶段的 Telegram 通知"""
        if not self._telegram_notifier:
            return
        
        try:
            duration_minutes = duration / 60.0
            message = f"<b>Exit Phase Complete: {key}</b>\n"
            message += f"Slippage: {slippage:.6f}\n"
            message += f"Fee: {fee:.6f}\n"
            message += f"Total Wear: {wear:.6f}\n"
            message += f"Duration: {duration_minutes:.2f} min"
            self._telegram_notifier.send_message(message)
        except Exception as exc:
            logger.error("Error sending exit notification: %s", exc, exc_info=True)
    
    def _send_round_notification(self) -> None:
        """发送本轮统计的 Telegram 通知"""
        if not self._telegram_notifier or not self._current_round:
            return
        
        try:
            summary = self._current_round.get("summary", {})
            round_num = self._current_round.get("round", 0)
            
            # 构建消息
            lines = [f"<b>Round {round_num} Summary</b>"]
            lines.append("")
            
            # Enter 阶段
            lines.append("<b>Enter Phase:</b>")
            enter_times = self._current_round["enter"]["time"]
            enter_wears = self._current_round["enter"]["wear"]
            for key, duration in enter_times.items():
                wear = enter_wears.get(key, 0)
                duration_minutes = duration / 60.0
                lines.append(f"  {key}: {duration_minutes:.2f} min, wear: {wear:.6f}")
            max_enter_time = summary.get('round_times', {}).get('enter', 0) / 60.0
            lines.append(f"  Max time: {max_enter_time:.2f} min")
            lines.append(f"  Total wear: {summary.get('wears', {}).get('enter', 0):.6f}")
            lines.append("")
            
            # Exit 阶段
            lines.append("<b>Exit Phase:</b>")
            exit_times = self._current_round["exit"]["time"]
            exit_wears = self._current_round["exit"]["wear"]
            for key, duration in exit_times.items():
                wear = exit_wears.get(key, 0)
                duration_minutes = duration / 60.0
                lines.append(f"  {key}: {duration_minutes:.2f} min, wear: {wear:.6f}")
            max_exit_time = summary.get('round_times', {}).get('exit', 0) / 60.0
            lines.append(f"  Max time: {max_exit_time:.2f} min")
            lines.append(f"  Total wear: {summary.get('wears', {}).get('exit', 0):.6f}")
            lines.append("")
            
            # Summary
            lines.append("<b>Summary:</b>")
            total_time_minutes = summary.get('totaltime', 0) / 60.0
            lines.append(f"  Total time: {total_time_minutes:.2f} min")
            lines.append(f"  Total wear: {summary.get('wears', {}).get('total', 0):.6f}")
            lines.append(f"  Total slippage: {summary.get('slippage', {}).get('total', 0):.6f}")
            lines.append(f"  Total fee: {summary.get('fee', {}).get('total', 0):.6f}")
            
            message = "\n".join(lines)
            self._telegram_notifier.send_message(message)
        except Exception as exc:
            logger.error("Error sending round notification: %s", exc, exc_info=True)
    
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

    def _weighted_average(self, fills: Iterable[Dict[str, float]]) -> float:
        total_qty = 0.0
        total_notional = 0.0
        for fill in fills:
            qty = float(fill["quantity"])
            price = float(fill["price"])
            total_qty += qty
            total_notional += qty * price
        return total_notional / total_qty if total_qty > 0 else 0.0

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

    def _resolve_base_targets(
        self,
        client: LighterClient,
        plan: SymbolPlan,
        limits: MarketConstraints,
    ) -> Tuple[float, float, float]:
        price = self._compute_reference_price(
            client, plan.symbol, "Bid", plan.entry_offset_bps
        )
        # quote 就是金额
        raw_target_quote = float(plan.target_notional)
        raw_slice_quote = float(plan.slice_notional)
        # 币的数量
        configured_target_qty = (
            round_to_precision(raw_target_quote / price, limits.base_precision)
            if raw_target_quote > 0
            else 0.0
        )
        configured_slice_qty = round_to_precision(
            raw_slice_quote / price, limits.base_precision
        )
        return configured_target_qty, configured_slice_qty, price

    def _compute_aggressive_price(
        self,
        client: LighterClient,
        symbol: str,
        side: str,
        tick_size,
    ) -> Optional[float]:
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
            price = round_to_tick_size(base * 1.002, tick_size)
            if best_ask:
                price = max(price, best_ask)
            return price
        base = best_bid or best_ask
        price = round_to_tick_size(base * 0.998, tick_size)
        if best_bid:
            price = min(price, best_bid)
        return price

    def _get_market_limits_with_retry(self, client: LighterClient, symbol: str) -> MarketConstraints:
        return self._call_with_retry(
            lambda: self._get_market_limits(client, symbol),
            attempts=self._market_fetch_retries,
            description=f"fetch market metadata for {symbol}",
        )

    def _cancel_order_with_retry(self, client: LighterClient, order_index: str, symbol: str) -> Any:
        return self._call_with_retry(
            lambda: client.cancel_order(order_index, symbol),
            attempts=self._order_submit_retries,
            description=f"cancel order {order_index} on {symbol}",
        )

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
            min_quote_value=float(metadata.get("min_quote_value") or MIN_QUOTE_THRESHOLD),
        )
        self._market_cache[symbol] = limits
        return limits

    def _call_with_retry(self, func: Callable[[], Any], attempts: int, description: str) -> Any:
        max_attempts = max(1, attempts)
        last_error: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001 - bubble original error after logging
                last_error = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Failed to %s (attempt %s/%s): %s",
                        description,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    # Allow stop_event to interrupt long retry loops
                    self._stop_event.wait(min(1.0 * attempt, 5.0))
                else:
                    logger.error(
                        "Giving up trying to %s after %s attempts: %s",
                        description,
                        max_attempts,
                        exc,
                    )
        assert last_error is not None
        raise last_error

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
        last_response: Dict[str, Any] = {}
        last_error: Optional[Exception] = None
        retries = max_retries if isinstance(max_retries, int) and max_retries > 0 else 1
        for attempt in range(1, retries + 1):
            try:
                response = client.execute_order(order)
            except (SimpleSignerError, Exception) as exc:  # noqa: BLE001 - propagate after retries
                last_error = exc
                if attempt < retries:
                    logger.warning(
                        "Order submit attempt %s/%s for %s failed (%s); retrying...",
                        attempt,
                        retries,
                        account_label,
                        exc,
                    )
                    self._stop_event.wait(min(0.5 * attempt, 2.0))
                    continue
                raise exc
            if isinstance(response, dict) and response.get("error"):
                error_text = str(response["error"]).lower()
                if "nonce" in error_text and attempt < retries:
                    logger.warning("Nonce error for %s, refreshing and retrying (attempt %s/%s)", account_label, attempt, retries)
                    refreshed = self._refresh_client_nonce(client)
                    if refreshed is None:
                        time.sleep(0.2 * attempt)
                    continue
                last_response = response
                break
            last_response = response
            break
        if last_error:
            raise last_error
        return last_response

    def _handle_order_failure(self, exc: OrderSubmissionError) -> None:
        if self._margin_failsafe_engaged:
            logger.critical(
                "Additional order failure for %s on %s while failsafe already engaged: %s",
                exc.symbol,
                exc.account_label,
                exc,
            )
            raise MarginFailsafeTriggered(str(exc))
        self._margin_failsafe_engaged = True
        logger.critical(
            "Order submission failed for %s on %s: %s",
            exc.symbol,
            exc.account_label,
            exc,
        )
        self._trigger_margin_failsafe(exc.symbol, exc.account_idx, str(exc))

    def _trigger_margin_failsafe(self, symbol: str, account_idx: int, error_text: str) -> None:
        context = self._active_cycle_context or {}
        plan = context.get("plan") if isinstance(context, dict) else None
        primary_idx = context.get("primary_idx") if isinstance(context, dict) else None
        hedgers: Sequence[int] = tuple(context.get("hedgers") or ()) if isinstance(context, dict) else ()
        target_symbol = plan.symbol if isinstance(plan, SymbolPlan) else symbol

        if primary_idx is None or not isinstance(plan, SymbolPlan):
            logger.critical(
                "Failsafe context missing while handling margin error for %s; stopping immediately (account=%s, error=%s).",
                symbol,
                self._account_labels[account_idx],
                error_text,
            )
            self.stop()
            raise MarginFailsafeTriggered(error_text)

        try:
            fresh_position = max(self._refresh_position(primary_idx, target_symbol), 0.0)
            logger.critical(
                "Failsafe flatten starting for %s: primary=%s hedgers=%s existing_base=%.8f",
                target_symbol,
                self._account_labels[primary_idx],
                [self._account_labels[idx] for idx in hedgers],
                fresh_position,
            )
            self._flatten_position(primary_idx, hedgers, plan, fresh_position)
            self._print_cycle_summary(target_symbol)
        except Exception as exc:
            logger.exception("Failsafe flatten encountered an error: %s", exc)
        finally:
            self.stop()
        raise MarginFailsafeTriggered(error_text)
