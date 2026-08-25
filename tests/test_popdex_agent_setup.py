"""Tests for local-only POPDEX Agent Web onboarding."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eth_abi import decode as abi_decode
from eth_account import Account
from eth_utils import keccak

from utils import popdex_agent_setup as setup


MAIN_ACCOUNT = "0x2222222222222222222222222222222222222222"


class PopdexAgentSetupTest(unittest.TestCase):
    def test_prepare_generates_local_key_without_returning_secret(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True), patch.object(
            setup, "ENV_PATH", Path(temp_dir) / ".env"
        ), patch.object(
            setup,
            "get_agent_account_info",
            return_value={"available": True, "exists": False},
        ):
            result = setup.prepare_agent_setup(
                MAIN_ACCOUNT,
                network="testnet",
                expires_in_days=90,
                global_scope=False,
            )

            self.assertTrue(result["success"])
            self.assertTrue(result["generated"])
            self.assertNotIn("private_key", result)
            self.assertEqual(result["chain_id"], 34952)
            self.assertEqual(result["chain"]["chainId"], hex(34952))

            env_text = setup.ENV_PATH.read_text(encoding="utf-8")
            self.assertIn("POPDEX_ACCOUNT_ADDRESS=0x2222", env_text)
            self.assertIn("POPDEX_AGENT_PRIVATE_KEY=0x", env_text)
            self.assertIn("POPDEX_NETWORK=testnet", env_text)
            self.assertEqual(
                Account.from_key(os.environ["POPDEX_AGENT_PRIVATE_KEY"]).address,
                result["agent_address"],
            )

    def test_approve_agent_calldata_matches_official_signature(self):
        agent_key = "0x" + "11" * 32
        agent_address = Account.from_key(agent_key).address
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"POPDEX_AGENT_PRIVATE_KEY": agent_key},
            clear=True,
        ), patch.object(
            setup, "ENV_PATH", Path(temp_dir) / ".env"
        ), patch.object(
            setup,
            "get_agent_account_info",
            return_value={"available": True, "exists": False},
        ):
            result = setup.prepare_agent_setup(MAIN_ACCOUNT, "mainnet", 30, True)

        raw = bytes.fromhex(result["transaction"]["data"].removeprefix("0x"))
        self.assertEqual(raw[:4], keccak(text=setup.APPROVE_AGENT_SIGNATURE)[:4])
        decoded = abi_decode(
            ["address", "address", "bytes32", "uint64", "uint64", "bool"],
            raw[4:],
        )
        self.assertEqual(decoded[0].lower(), agent_address.lower())
        self.assertEqual(decoded[1].lower(), MAIN_ACCOUNT.lower())
        self.assertTrue(bytes(decoded[2]).rstrip(b"\0").startswith(b"BackpackMM-"))
        self.assertGreater(decoded[3], decoded[4])
        self.assertTrue(decoded[5])
        self.assertEqual(result["transaction"]["gasPrice"], "0x0")

    def test_existing_authorized_agent_is_reused(self):
        agent_key = "0x" + "11" * 32
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"POPDEX_AGENT_PRIVATE_KEY": agent_key},
            clear=True,
        ), patch.object(
            setup, "ENV_PATH", Path(temp_dir) / ".env"
        ), patch.object(
            setup,
            "get_agent_account_info",
            return_value={
                "available": True,
                "exists": True,
                "account_type": 1,
                "delegator": MAIN_ACCOUNT,
                "is_expired": False,
            },
        ):
            result = setup.prepare_agent_setup(MAIN_ACCOUNT, "mainnet")
        self.assertTrue(result["already_authorized"])
        self.assertNotIn("transaction", result)
        self.assertFalse(result["generated"])

    def test_expired_agent_is_not_reapproved(self):
        agent_key = "0x" + "11" * 32
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"POPDEX_AGENT_PRIVATE_KEY": agent_key},
            clear=True,
        ), patch.object(
            setup, "ENV_PATH", Path(temp_dir) / ".env"
        ), patch.object(
            setup,
            "get_agent_account_info",
            return_value={
                "available": True,
                "exists": True,
                "account_type": 1,
                "delegator": MAIN_ACCOUNT,
                "is_expired": True,
            },
        ):
            with self.assertRaisesRegex(ValueError, "不應以 approveAgent 重複授權"):
                setup.prepare_agent_setup(MAIN_ACCOUNT, "mainnet")

    def test_rpc_failure_does_not_build_unsafe_approval(self):
        agent_key = "0x" + "11" * 32
        with patch.dict(
            os.environ,
            {"POPDEX_AGENT_PRIVATE_KEY": agent_key},
            clear=True,
        ), patch.object(
            setup,
            "get_agent_account_info",
            return_value={"available": False, "error": "offline"},
        ):
            with self.assertRaisesRegex(ValueError, "避免重複授權"):
                setup.prepare_agent_setup(MAIN_ACCOUNT, "mainnet")

    def test_remote_agent_setup_route_is_denied(self):
        from web.server import app

        client = app.test_client()
        response = client.get(
            "/api/popdex-agent/status",
            environ_base={"REMOTE_ADDR": "203.0.113.10"},
        )
        self.assertEqual(response.status_code, 403)

    def test_replace_agent_uses_official_atomic_rotation_and_backup(self):
        old_key = "0x" + "11" * 32
        old_agent = Account.from_key(old_key).address
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "POPDEX_ACCOUNT_ADDRESS": MAIN_ACCOUNT,
                "POPDEX_AGENT_PRIVATE_KEY": old_key,
                "POPDEX_NETWORK": "mainnet",
            },
            clear=True,
        ), patch.object(
            setup, "ENV_PATH", Path(temp_dir) / ".env"
        ), patch.object(
            setup, "BACKUP_DIR", Path(temp_dir) / ".popdex-agent-backups"
        ), patch.object(
            setup,
            "get_agent_account_info",
            return_value={
                "available": True,
                "exists": True,
                "account_type": 1,
                "delegator": MAIN_ACCOUNT,
                "is_expired": False,
            },
        ):
            prepared = setup.prepare_agent_management(
                "replace", MAIN_ACCOUNT, "mainnet", 30
            )

            raw = bytes.fromhex(prepared["transaction"]["data"].removeprefix("0x"))
            self.assertEqual(raw[:4], keccak(text=setup.REPLACE_AGENT_SIGNATURE)[:4])
            decoded = abi_decode(
                ["address", "address", "uint64", "uint64"], raw[4:]
            )
            self.assertEqual(decoded[0].lower(), old_agent.lower())
            self.assertEqual(
                decoded[1].lower(), prepared["new_agent_address"].lower()
            )
            self.assertNotEqual(decoded[1].lower(), old_agent.lower())
            self.assertGreater(decoded[2], decoded[3])
            self.assertNotIn("private_key", str(prepared))

            backup_path = Path(prepared["backup_file"])
            self.assertTrue(backup_path.is_file())
            self.assertIn(
                f"POPDEX_AGENT_PRIVATE_KEY={old_key}",
                backup_path.read_text(encoding="utf-8"),
            )

            finalized = setup.finalize_agent_management(prepared["operation_id"])
            self.assertTrue(finalized["authorized"])
            self.assertEqual(
                Account.from_key(os.environ["POPDEX_AGENT_PRIVATE_KEY"]).address.lower(),
                prepared["new_agent_address"].lower(),
            )
            self.assertFalse(
                (setup.BACKUP_DIR / f'.pending-{prepared["operation_id"]}.json').exists()
            )
            self.assertTrue(backup_path.is_file())

    def test_revoke_agent_is_backed_up_and_clears_local_key_after_confirmation(self):
        old_key = "0x" + "11" * 32
        old_agent = Account.from_key(old_key).address
        active_info = {
            "available": True,
            "exists": True,
            "account_type": 1,
            "delegator": MAIN_ACCOUNT,
            "is_expired": False,
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "POPDEX_ACCOUNT_ADDRESS": MAIN_ACCOUNT,
                "POPDEX_AGENT_PRIVATE_KEY": old_key,
                "POPDEX_NETWORK": "mainnet",
            },
            clear=True,
        ), patch.object(
            setup, "ENV_PATH", Path(temp_dir) / ".env"
        ), patch.object(
            setup, "BACKUP_DIR", Path(temp_dir) / ".popdex-agent-backups"
        ), patch.object(
            setup, "get_agent_account_info", return_value=active_info
        ) as get_info:
            prepared = setup.prepare_agent_management(
                "revoke", MAIN_ACCOUNT, "mainnet"
            )
            raw = bytes.fromhex(prepared["transaction"]["data"].removeprefix("0x"))
            self.assertEqual(raw[:4], keccak(text=setup.REVOKE_AGENT_SIGNATURE)[:4])
            decoded = abi_decode(["address"], raw[4:])
            self.assertEqual(decoded[0].lower(), old_agent.lower())
            backup_path = Path(prepared["backup_file"])
            self.assertTrue(backup_path.is_file())

            get_info.return_value = {"available": True, "exists": False}
            finalized = setup.finalize_agent_management(prepared["operation_id"])
            self.assertEqual(finalized["action"], "revoke")
            self.assertFalse(finalized["configured"])
            self.assertEqual(os.environ["POPDEX_AGENT_PRIVATE_KEY"], "")
            self.assertIn(
                "POPDEX_AGENT_PRIVATE_KEY=",
                setup.ENV_PATH.read_text(encoding="utf-8"),
            )
            self.assertTrue(backup_path.is_file())

    def test_prepare_route_never_returns_private_key(self):
        from web.server import app

        fake_result = {
            "success": True,
            "agent_address": "0x3333333333333333333333333333333333333333",
            "transaction": {"to": setup.ACCOUNT_CONTRACT, "data": "0x1234"},
        }
        client = app.test_client()
        with patch("web.server.prepare_agent_setup", return_value=fake_result):
            response = client.post(
                "/api/popdex-agent/prepare",
                json={"account_address": MAIN_ACCOUNT, "network": "mainnet"},
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("private_key", response.get_json())

    def test_manage_route_never_returns_private_key(self):
        from web.server import app

        fake_result = {
            "success": True,
            "action": "replace",
            "operation_id": "a" * 32,
            "old_agent_address": "0x3333333333333333333333333333333333333333",
            "new_agent_address": "0x4444444444444444444444444444444444444444",
            "backup_file": ".popdex-agent-backups/example.env",
            "transaction": {"to": setup.ACCOUNT_CONTRACT, "data": "0x1234"},
        }
        client = app.test_client()
        with patch("web.server.prepare_agent_management", return_value=fake_result):
            response = client.post(
                "/api/popdex-agent/manage",
                json={
                    "action": "replace",
                    "account_address": MAIN_ACCOUNT,
                    "network": "mainnet",
                },
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("private_key", str(response.get_json()))


if __name__ == "__main__":
    unittest.main()
