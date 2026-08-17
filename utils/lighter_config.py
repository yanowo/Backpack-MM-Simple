"""Shared configuration helpers for Lighter deployments."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Mapping, Optional, Tuple

import requests
from dotenv import load_dotenv


load_dotenv()


LIGHTER_EXCHANGE = "lighter"
LIGHTER_ROBINHOOD_EXCHANGE = "lighter_robinhood"

LIGHTER_BASE_URL = "https://mainnet.zklighter.elliot.ai"
LIGHTER_WS_URL = "wss://mainnet.zklighter.elliot.ai/stream"
LIGHTER_CHAIN_ID = 304

LIGHTER_ROBINHOOD_BASE_URL = "https://api.rh.lighter.xyz"
LIGHTER_ROBINHOOD_WS_URL = "wss://api.rh.lighter.xyz/stream"
LIGHTER_ROBINHOOD_CHAIN_ID = 466324

_LIGHTER_ALIASES = {
    "lighter": LIGHTER_EXCHANGE,
    "lighter_mainnet": LIGHTER_EXCHANGE,
    "lighter-mainnet": LIGHTER_EXCHANGE,
    "lighter_robinhood": LIGHTER_ROBINHOOD_EXCHANGE,
    "lighter-robinhood": LIGHTER_ROBINHOOD_EXCHANGE,
    "lighter_rh": LIGHTER_ROBINHOOD_EXCHANGE,
    "lighter-rh": LIGHTER_ROBINHOOD_EXCHANGE,
    "robinhood_lighter": LIGHTER_ROBINHOOD_EXCHANGE,
}

_DEPLOYMENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    LIGHTER_EXCHANGE: {
        "deployment": LIGHTER_EXCHANGE,
        "base_url": LIGHTER_BASE_URL,
        "ws_url": LIGHTER_WS_URL,
        "chain_id": LIGHTER_CHAIN_ID,
    },
    LIGHTER_ROBINHOOD_EXCHANGE: {
        "deployment": LIGHTER_ROBINHOOD_EXCHANGE,
        "base_url": LIGHTER_ROBINHOOD_BASE_URL,
        "ws_url": LIGHTER_ROBINHOOD_WS_URL,
        "chain_id": LIGHTER_ROBINHOOD_CHAIN_ID,
    },
}

_ENV_PREFIXES = {
    LIGHTER_EXCHANGE: ("LIGHTER",),
    LIGHTER_ROBINHOOD_EXCHANGE: ("LIGHTER_ROBINHOOD", "LIGHTER_RH"),
}


def normalize_lighter_exchange(name: Optional[str]) -> Optional[str]:
    """Return the canonical Lighter deployment id, or ``None``."""
    return _LIGHTER_ALIASES.get((name or "").strip().lower())


def is_lighter_exchange(name: Optional[str]) -> bool:
    return normalize_lighter_exchange(name) is not None


def get_lighter_defaults(name: Optional[str]) -> Dict[str, Any]:
    deployment = normalize_lighter_exchange(name)
    if deployment is None:
        raise ValueError(f"Unsupported Lighter deployment: {name}")
    return dict(_DEPLOYMENT_DEFAULTS[deployment])


def apply_lighter_defaults(name: Optional[str], config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge a user configuration on top of safe deployment defaults."""
    deployment = normalize_lighter_exchange(name)
    if deployment is None and config:
        deployment = normalize_lighter_exchange(str(config.get("deployment") or ""))
    deployment = deployment or LIGHTER_EXCHANGE

    merged = get_lighter_defaults(deployment)
    for key, value in dict(config or {}).items():
        if value is not None and value != "":
            merged[key] = value
    merged["deployment"] = deployment
    return merged


def infer_lighter_chain_id(base_url: Optional[str], default: int = LIGHTER_CHAIN_ID) -> int:
    """Infer the signing chain id for official Lighter hosts."""
    normalized = (base_url or "").strip().lower().rstrip("/")
    if normalized == LIGHTER_ROBINHOOD_BASE_URL or "api.rh.lighter.xyz" in normalized:
        return LIGHTER_ROBINHOOD_CHAIN_ID
    if "testnet" in normalized:
        return 300
    if "mainnet" in normalized:
        return LIGHTER_CHAIN_ID
    return int(default)


def _first_env(prefixes: Tuple[str, ...], *suffixes: str) -> Optional[str]:
    for prefix in prefixes:
        for suffix in suffixes:
            value = os.getenv(f"{prefix}_{suffix}")
            if value is not None and value.strip():
                return value.strip()
    return None


def get_lighter_account_index(
    address: str,
    base_url: str = LIGHTER_BASE_URL,
    *,
    timeout: float = 10.0,
    session: Optional[requests.Session] = None,
) -> int:
    """Resolve the first (master) account index associated with an L1 address."""
    normalized_address = (address or "").strip()
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", normalized_address):
        raise ValueError("Lighter address must be a 20-byte 0x-prefixed EVM address")

    requester = session or requests
    url = f"{base_url.rstrip('/')}/api/v1/accountsByL1Address"
    try:
        response = requester.get(
            url,
            params={"l1_address": normalized_address},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ValueError(f"Unable to resolve Lighter account index: {exc}") from exc

    accounts = payload.get("sub_accounts") or payload.get("accounts") or []
    if not isinstance(accounts, list):
        accounts = []
    for account in accounts:
        if not isinstance(account, dict):
            continue
        account_index = account.get("index")
        if account_index is None:
            account_index = account.get("account_index")
        if account_index is not None:
            return int(account_index)

    message = payload.get("message") if isinstance(payload, dict) else None
    raise ValueError(message or f"No Lighter account found for address: {normalized_address}")


def build_lighter_config_from_env(
    exchange: str,
    *,
    resolve_account_index: bool = False,
) -> Dict[str, Any]:
    """Build a deployment-specific Lighter config from environment variables."""
    deployment = normalize_lighter_exchange(exchange)
    if deployment is None:
        raise ValueError(f"Unsupported Lighter deployment: {exchange}")

    prefixes = _ENV_PREFIXES[deployment]
    config = get_lighter_defaults(deployment)

    base_url = _first_env(prefixes, "BASE_URL")
    ws_url = _first_env(prefixes, "WS_URL")
    chain_id = _first_env(prefixes, "CHAIN_ID")
    private_key = _first_env(prefixes, "PRIVATE_KEY", "API_KEY")
    account_index = _first_env(prefixes, "ACCOUNT_INDEX")
    address = _first_env(prefixes, "ADDRESS")
    api_key_index = _first_env(prefixes, "API_KEY_INDEX")
    signer_lib_dir = _first_env(prefixes, "SIGNER_LIB_DIR")
    verify_ssl = _first_env(prefixes, "VERIFY_SSL")

    if base_url:
        config["base_url"] = base_url.rstrip("/")
    if ws_url:
        config["ws_url"] = ws_url
    if chain_id:
        config["chain_id"] = int(chain_id)
    else:
        config["chain_id"] = infer_lighter_chain_id(
            config["base_url"],
            default=int(config["chain_id"]),
        )
    if private_key:
        config["api_private_key"] = private_key
    if account_index:
        config["account_index"] = int(account_index)
    if address:
        config["account_address"] = address
    if api_key_index:
        config["api_key_index"] = int(api_key_index)
    if signer_lib_dir:
        config["signer_lib_dir"] = signer_lib_dir
    if verify_ssl is not None:
        config["verify_ssl"] = verify_ssl.lower() not in ("0", "false", "no", "off")

    if resolve_account_index and config.get("account_index") is None and address:
        config["account_index"] = get_lighter_account_index(address, config["base_url"])

    return config
