"""Unit tests for the POPDEX WebSocket protocol adapter."""
from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from utils.popdex_config import build_popdex_config_from_env
from ws_client.popdex_ws_client import PopdexWebSocket


ACCOUNT = "0x2222222222222222222222222222222222222222"


class PopdexWebSocketTest(unittest.TestCase):
    def make_ws(self, callback=None, **overrides):
        return PopdexWebSocket(
            account_address=ACCOUNT,
            symbol="BTC_USDT_PERP",
            on_message_callback=callback,
            rest_config={"base_url": "https://api.popdex.xyz"},
            **overrides,
        )

    def test_network_defaults_and_symbol_normalization(self):
        main = self.make_ws()
        test = PopdexWebSocket(
            account_address=ACCOUNT,
            symbol="BTC-USDT-PERP",
            rest_config={"network": "testnet"},
        )
        self.assertEqual(main.symbol, "BTCUSDT")
        self.assertEqual(main.config.ws_url, PopdexWebSocket.MAINNET_WS_URL)
        self.assertEqual(test.config.ws_url, PopdexWebSocket.TESTNET_WS_URL)
        self.assertEqual(main.config.ping_interval, 0)

    def test_public_subscribe_and_unsubscribe_schema(self):
        client = self.make_ws()
        self.assertEqual(
            client._create_subscribe_message("books5.BTCUSDT"),
            {
                "op": "subscribe",
                "args": [
                    {
                        "category": "Futures",
                        "topic": "books5",
                        "symbol": "BTCUSDT",
                    }
                ],
            },
        )
        self.assertEqual(
            client._create_unsubscribe_message("ticker.BTCUSDT")["op"],
            "unsubscribe",
        )

    def test_wallet_topics_are_batched_without_auth(self):
        client = self.make_ws()
        client.connected = True
        client.ws = Mock()
        self.assertIsNone(client._create_auth_message())
        self.assertTrue(client.subscribe_order_updates())
        message = json.loads(client.ws.send.call_args.args[0])
        self.assertEqual(message["op"], "subscribe")
        self.assertEqual(
            message["args"],
            [
                {"walletId": ACCOUNT, "topic": "order"},
                {"walletId": ACCOUNT, "topic": "fill"},
                {"walletId": ACCOUNT, "topic": "position"},
                {"walletId": ACCOUNT, "topic": "account"},
            ],
        )
        self.assertEqual(client.subscriptions, list(PopdexWebSocket.PRIVATE_TOPICS))
        self.assertTrue(client.subscribe_order_updates())
        self.assertEqual(client.ws.send.call_count, 1)

    def test_ticker_and_books5_mapping(self):
        client = self.make_ws()
        ticker = client._handle_ticker_message(
            {
                "symbol": "BTCUSDT",
                "bid1Price": "77100",
                "ask1Price": "77104",
                "lastPrice": "77086",
                "_wsTs": "1787542445846",
            }
        )
        self.assertEqual(str(ticker.bid_price), "77100")
        self.assertEqual(str(ticker.ask_price), "77104")
        self.assertEqual(ticker.timestamp, 1787542445846)

        depth = client._handle_depth_message(
            {
                "a": [["77105", "0.1"], ["77104", "0.2"]],
                "b": [["77099", "0.3"], ["77100", "0.4"]],
                "ts": "1787542445808",
            }
        )
        self.assertEqual(str(depth.bids[0][0]), "77100")
        self.assertEqual(str(depth.asks[0][0]), "77104")
        self.assertEqual(client.orderbook, {"bids": [], "asks": []})

    def test_order_and_fill_mapping(self):
        client = self.make_ws()
        order = client._handle_order_update_message(
            {
                "orderId": "order-1",
                "clientOid": "client-1",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "type": "Limit",
                "status": "PartiallyFilled",
                "price": "77000",
                "qty": "0.01",
                "filledQty": "0.004",
                "remainingQty": "0.006",
                "updatedAt": "1787542445000",
            }
        )
        self.assertEqual(order.status, "PARTIALLY_FILLED")
        self.assertEqual(order.side, "BUY")
        self.assertEqual(str(order.filled_quantity), "0.004")

        fill = client._handle_fill_message(
            {
                "execId": "fill-1",
                "orderId": "order-1",
                "symbol": "BTCUSDT",
                "side": "Sell",
                "execPrice": "77100",
                "execQty": "0.004",
                "tradeScope": "Taker",
                "feeDetail": [{"feeToken": "USDT", "fee": "0.02"}],
                "updatedAt": "1787542445100",
            }
        )
        self.assertEqual(fill.side, "SELL")
        self.assertFalse(fill.is_maker)
        self.assertEqual(str(fill.fee), "0.02")
        self.assertEqual(fill.fee_asset, "USDT")

    def test_batched_wallet_data_fans_out_every_record(self):
        received = []
        client = self.make_ws(callback=lambda stream, data: received.append((stream, data)))
        client._on_message(
            None,
            json.dumps(
                {
                    "action": "snapshot",
                    "arg": {"walletId": ACCOUNT, "topic": "fill"},
                    "data": [
                        {"execId": "fill-1", "execPrice": "1", "execQty": "1"},
                        {"execId": "fill-2", "execPrice": "2", "execQty": "2"},
                    ],
                    "ts": "1787542445000",
                }
            ),
        )
        self.assertEqual([item[1]["execId"] for item in received], ["fill-1", "fill-2"])

    def test_application_text_ping_and_pong(self):
        client = self.make_ws()
        client.ws = Mock()
        client.connected = True
        client.running = True
        client._last_application_ping = 0

        def stop_after_first_sleep(_seconds):
            client.running = False

        with patch("ws_client.popdex_ws_client.time.sleep", side_effect=stop_after_first_sleep):
            client._heartbeat_check()
        client.ws.send.assert_called_once_with("ping")
        self.assertTrue(client._awaiting_application_pong)

        client._on_message(None, "pong")
        self.assertFalse(client._awaiting_application_pong)
        self.assertGreater(client.last_heartbeat, 0)

    def test_environment_config_includes_ws_url(self):
        with patch.dict("os.environ", {"POPDEX_NETWORK": "testnet"}, clear=True):
            config = build_popdex_config_from_env()
        self.assertEqual(config["ws_url"], PopdexWebSocket.TESTNET_WS_URL)


if __name__ == "__main__":
    unittest.main()
