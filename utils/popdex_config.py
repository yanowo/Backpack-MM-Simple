"""POPDEX environment/configuration helpers."""
from __future__ import annotations

import os
from typing import Any, Dict


POPDEX_EXCHANGE = "popdex"


def build_popdex_config_from_env() -> Dict[str, Any]:
    """Build a POPDEX client config without exposing secrets in call sites."""
    network = (os.getenv("POPDEX_NETWORK") or "mainnet").strip().lower()
    testnet = network in {"test", "testnet"}
    default_base_url = (
        "https://testnet-api.popdex.xyz" if testnet else "https://api.popdex.xyz"
    )
    default_ws_url = (
        "wss://testnet-ws.popdex.xyz/v1/ws/public"
        if testnet
        else "wss://ws.popdex.xyz/v1/ws/public"
    )
    base_url = (os.getenv("POPDEX_BASE_URL") or default_base_url).rstrip("/")

    config: Dict[str, Any] = {
        "account_address": os.getenv("POPDEX_ACCOUNT_ADDRESS", ""),
        "agent_private_key": os.getenv("POPDEX_AGENT_PRIVATE_KEY", ""),
        "base_url": base_url,
        "rpc_url": os.getenv("POPDEX_RPC_URL") or f"{base_url}/api/v1/web3/rpc",
        "ws_url": os.getenv("POPDEX_WS_URL") or default_ws_url,
        "chain_id": int(os.getenv("POPDEX_CHAIN_ID") or (34952 if testnet else 2184)),
        "network": "testnet" if testnet else "mainnet",
        "market_slippage": os.getenv("POPDEX_MARKET_SLIPPAGE") or "0.01",
    }

    if os.getenv("POPDEX_RECEIPT_TIMEOUT"):
        config["receipt_timeout"] = float(os.environ["POPDEX_RECEIPT_TIMEOUT"])
    return config
