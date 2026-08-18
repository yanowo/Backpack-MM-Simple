import os
import json
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from api import ApiResponse, OrderInfo, RobinhoodLighterClient, get_client
from api.lighter_client import LighterClient
from utils.lighter_config import (
    LIGHTER_ROBINHOOD_BASE_URL,
    LIGHTER_ROBINHOOD_CHAIN_ID,
    LIGHTER_ROBINHOOD_EXCHANGE,
    LIGHTER_ROBINHOOD_WS_URL,
    build_lighter_config_from_env,
    get_lighter_account_index,
)
from ws_client import RobinhoodLighterWebSocket


class LighterRobinhoodConfigTests(unittest.TestCase):
    def test_robinhood_defaults_are_isolated_from_mainnet(self):
        env = {
            "LIGHTER_PRIVATE_KEY": "mainnet-key",
            "LIGHTER_ACCOUNT_INDEX": "10",
            "LIGHTER_ROBINHOOD_PRIVATE_KEY": "robinhood-key",
            "LIGHTER_ROBINHOOD_ACCOUNT_INDEX": "20",
            "LIGHTER_ROBINHOOD_API_KEY_INDEX": "4",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_lighter_config_from_env(LIGHTER_ROBINHOOD_EXCHANGE)

        self.assertEqual(config["api_private_key"], "robinhood-key")
        self.assertEqual(config["account_index"], 20)
        self.assertEqual(config["api_key_index"], 4)
        self.assertEqual(config["base_url"], LIGHTER_ROBINHOOD_BASE_URL)
        self.assertEqual(config["ws_url"], LIGHTER_ROBINHOOD_WS_URL)
        self.assertEqual(config["chain_id"], LIGHTER_ROBINHOOD_CHAIN_ID)

    def test_factory_returns_robinhood_client(self):
        client = get_client(LIGHTER_ROBINHOOD_EXCHANGE, {})

        self.assertIsInstance(client, RobinhoodLighterClient)
        self.assertEqual(client.get_exchange_name(), "Lighter Robinhood")
        self.assertEqual(client.base_url, LIGHTER_ROBINHOOD_BASE_URL)
        self.assertEqual(client.chain_id, LIGHTER_ROBINHOOD_CHAIN_ID)

    def test_direct_client_infers_deployment_from_official_host(self):
        client = LighterClient({"base_url": LIGHTER_ROBINHOOD_BASE_URL})

        self.assertEqual(client.deployment, LIGHTER_ROBINHOOD_EXCHANGE)
        self.assertEqual(client.chain_id, LIGHTER_ROBINHOOD_CHAIN_ID)

    def test_websocket_uses_robinhood_rest_and_stream_endpoints(self):
        ws = RobinhoodLighterWebSocket(symbol="LIT")

        self.assertEqual(ws.config.ws_url, LIGHTER_ROBINHOOD_WS_URL)
        self.assertEqual(ws._rest_config["base_url"], LIGHTER_ROBINHOOD_BASE_URL)
        self.assertEqual(ws._rest_config["chain_id"], LIGHTER_ROBINHOOD_CHAIN_ID)

    def test_websocket_parses_robinhood_order_book_shape(self):
        ws = RobinhoodLighterWebSocket(symbol="LIT")
        payload = {
            "type": "subscribed/order_book",
            "channel": "order_book:5",
            "order_book": {
                "bids": [{"price": "2.3000", "size": "5.00"}],
                "asks": [{"price": "2.3001", "size": "6.00"}],
            },
        }

        stream, parsed_payload = ws._parse_message(json.dumps(payload))
        depth = ws._handle_depth_message(parsed_payload)

        self.assertEqual(stream, "order_book")
        self.assertEqual(str(depth.bids[0][0]), "2.3000")
        self.assertEqual(str(depth.asks[0][1]), "6.00")

    def test_account_lookup_supports_current_sub_accounts_schema(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"code": 200, "sub_accounts": [{"index": 12345}]}
        session = Mock()
        session.get.return_value = response

        account_index = get_lighter_account_index(
            "0x0000000000000000000000000000000000000001",
            LIGHTER_ROBINHOOD_BASE_URL,
            session=session,
        )

        self.assertEqual(account_index, 12345)
        session.get.assert_called_once_with(
            f"{LIGHTER_ROBINHOOD_BASE_URL}/api/v1/accountsByL1Address",
            params={"l1_address": "0x0000000000000000000000000000000000000001"},
            timeout=10.0,
        )

    def test_execute_order_uses_order_info_size_when_new_order_is_immediately_visible(self):
        client = RobinhoodLighterClient({})
        signer = Mock()
        signer.create_order.return_value = (
            {"client_order_index": 123},
            {"code": 200, "tx_hash": "0xtest"},
            None,
        )
        open_order = OrderInfo(
            order_id="456",
            client_order_id="123",
            symbol="LIT",
            side="Bid",
            order_type="LIMIT",
            size=Decimal("0.15"),
            price=Decimal("75.5"),
            status="open",
            filled_size=Decimal("0"),
            remaining_size=Decimal("0.15"),
            raw={"clientOrderIndex": 123},
        )

        with (
            patch.object(client, "_ensure_signer_client", return_value=signer),
            patch.object(
                client,
                "_lookup_market",
                return_value={
                    "market_id": 5,
                    "base_precision": 3,
                    "quote_precision": 3,
                    "min_order_size": 0.001,
                },
            ),
            patch.object(client, "get_open_orders", return_value=ApiResponse.ok([open_order])),
        ):
            result = client.execute_order(
                {
                    "symbol": "LIT",
                    "side": "Bid",
                    "type": "LIMIT",
                    "price": "75.5",
                    "quantity": "0.15",
                    "clientOrderIndex": 123,
                }
            )

        self.assertTrue(result.success)
        self.assertEqual(result.data.order_id, "456")
        self.assertEqual(result.data.size, Decimal("0.15"))


if __name__ == "__main__":
    unittest.main()
