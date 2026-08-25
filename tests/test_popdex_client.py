"""Unit tests for POPDEX mapping and transaction construction."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode

from api.popdex_client import PopdexClient


ACCOUNT = "0x2222222222222222222222222222222222222222"
AGENT_KEY = "0x" + "11" * 32


class PopdexClientTest(unittest.TestCase):
    def make_client(self, **overrides):
        config = {
            "account_address": ACCOUNT,
            "agent_private_key": AGENT_KEY,
            "wait_for_receipt": False,
            "position_mode": "Oneway",
        }
        config.update(overrides)
        return PopdexClient(config)

    def test_pack_post_only_open_long(self):
        packed = self.make_client()._pack_order_params(
            {
                "orderType": "Limit",
                "side": "Bid",
                "postOnly": True,
            }
        )
        self.assertEqual(packed[:10], bytes([2, 0, 0, 4, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(len(packed), 32)

    def test_pack_reduce_only_buy_closes_short(self):
        packed = self.make_client()._pack_order_params(
            {
                "orderType": "Market",
                "side": "Buy",
                "timeInForce": "IOC",
                "reduceOnly": True,
            }
        )
        self.assertEqual(packed[1], 1)
        self.assertEqual(packed[2], 0)
        self.assertEqual(packed[3], 2)
        self.assertEqual(packed[6], 1)
        self.assertEqual(packed[7], 0)

    def test_account_position_mode_sets_valid_position_side(self):
        oneway = self.make_client(position_mode="Oneway")
        hedge = self.make_client(position_mode="Hedge")

        oneway_buy = oneway._apply_account_position_mode({"side": "Bid"})
        hedge_buy = hedge._apply_account_position_mode({"side": "Bid"})
        hedge_reduce_buy = hedge._apply_account_position_mode(
            {"side": "Bid", "reduceOnly": True}
        )

        self.assertEqual(oneway_buy["positionSide"], "None")
        self.assertEqual(hedge_buy["positionSide"], "Long")
        self.assertEqual(hedge_reduce_buy["positionSide"], "Short")

    def test_position_mode_is_loaded_from_account_settings(self):
        client = self.make_client(position_mode="")
        client.make_request = Mock(
            return_value={
                "code": "200",
                "data": {"positionMode": "Oneway"},
            }
        )
        prepared = client._apply_account_position_mode({"side": "Ask"})
        self.assertEqual(prepared["positionSide"], "None")
        self.assertEqual(client._position_mode, "ONEWAY")
        client.make_request.assert_called_once_with(
            "GET", f"/api/v1/account/{ACCOUNT}/settings"
        )

    def test_market_and_ticker_mapping(self):
        client = self.make_client()
        client.make_request = Mock(
            side_effect=[
                {
                    "code": "200",
                    "data": [
                        {
                            "symbolId": "20000",
                            "symbol": "BTCUSDT",
                            "baseToken": "BTC",
                            "quoteToken": "USDT",
                            "status": "Trading",
                            "tickSize": "1",
                            "lotSize": "0.0001",
                            "minQty": "0.0001",
                            "maxQty": "100",
                            "pricePrecision": "0",
                            "baseTokenPrecision": "4",
                        }
                    ],
                },
                {
                    "code": "200",
                    "data": [
                        {
                            "symbol": "BTCUSDT",
                            "lastPrice": "77241",
                            "bid1Price": "77240",
                            "ask1Price": "77242",
                            "updatedTime": "1787540948532",
                        }
                    ],
                },
            ]
        )
        market = client.get_market_limits("BTC_USDT_PERP")
        ticker = client.get_ticker("BTC_USDT_PERP")
        self.assertTrue(market.success)
        self.assertEqual(str(market.data.min_order_size), "0.0001")
        self.assertTrue(ticker.success)
        self.assertEqual(str(ticker.data.bid_price), "77240")
        self.assertEqual(ticker.data.symbol, "BTCUSDT")

    def test_execute_order_signs_legacy_agent_transaction(self):
        client = self.make_client()
        client._market_cache["BTCUSDT"] = {"symbol": "BTCUSDT", "symbolId": "20000"}
        sent = {}

        def fake_rpc(method, params):
            sent["method"] = method
            sent["raw_tx"] = params[0]
            return {"result": "0x" + "ab" * 32}

        client._rpc = fake_rpc
        response = client.execute_order(
            {
                "symbol": "BTCUSDT",
                "orderType": "Limit",
                "side": "Bid",
                "price": "77000",
                "quantity": "0.001",
                "postOnly": True,
                "clientId": "strategy-order-1",
            }
        )
        self.assertTrue(response.success, response.error_message)
        self.assertEqual(sent["method"], "eth_sendRawTransaction")
        self.assertTrue(sent["raw_tx"].startswith("0x"))
        self.assertEqual(response.data.order_id, "strategy-order-1")
        self.assertEqual(response.data.client_order_id, "strategy-order-1")

    def test_market_order_uses_configured_default_slippage(self):
        client = self.make_client(market_slippage="0.01")
        client._market_cache["BTCUSDT"] = {"symbol": "BTCUSDT", "symbolId": "20000"}
        captured = {}

        def fake_send(call_data):
            captured["call_data"] = call_data
            return "0x" + "ab" * 32, None, {}

        client._send_contract_call = fake_send
        response = client.execute_order(
            {
                "symbol": "BTCUSDT",
                "orderType": "Market",
                "side": "Ask",
                "quantity": "0.001",
            }
        )
        self.assertTrue(response.success, response.error_message)
        decoded = abi_decode(
            ["address", "bytes32", "uint16", "bytes32", "uint256", "uint256", "uint256", "address", "uint256"],
            captured["call_data"][4:],
        )
        self.assertEqual(decoded[4], 0)
        self.assertEqual(decoded[6], 10**16)
        self.assertEqual(bytes(decoded[3])[3], 2)

    def test_cancel_by_client_id_uses_zero_order_id(self):
        client = self.make_client()
        captured = {}

        def fake_send(call_data):
            captured["call_data"] = call_data
            return "0x" + "ab" * 32, None, {}

        client._send_contract_call = fake_send
        response = client.cancel_order("strategy-order-1", "BTCUSDT")
        self.assertTrue(response.success, response.error_message)
        decoded = abi_decode(
            ["address", "uint128", "bytes32"], captured["call_data"][4:]
        )
        self.assertEqual(decoded[1], 0)
        self.assertEqual(bytes(decoded[2]).rstrip(b"\x00"), b"strategy-order-1")

    def test_cancel_event_business_error_is_reported(self):
        client = self.make_client()
        receipt = {
            "logs": [
                {
                    "topics": ["0x" + client.ORDER_CANCEL_EVENT],
                    "data": "0x"
                    + abi_encode(
                        ["uint128", "bytes32", "bool", "uint32"],
                        [123, bytes(32), False, 7001],
                    ).hex(),
                }
            ]
        }
        error = client._event_business_error(
            receipt,
            client.ORDER_CANCEL_EVENT,
            ["uint128", "bytes32", "bool", "uint32"],
            2,
            3,
            "OrderCancel",
        )
        self.assertEqual(error, "POPDEX OrderCancel rejected with code 7001")

    def test_order_business_error_is_reported_before_reverted_status(self):
        client = self.make_client(wait_for_receipt=True)
        client._market_cache["BTCUSDT"] = {"symbol": "BTCUSDT", "symbolId": "20000"}
        client._send_contract_call = Mock(return_value=("0xdeadbeef", None, {}))
        client._wait_receipt = Mock(
            return_value={
                "status": "0x0",
                "logs": [
                    {
                        "topics": ["0x" + client.ORDER_CREATE_EVENT],
                        "data": "0x"
                        + abi_encode(
                            ["uint128", "bytes32", "int256", "int256", "int256", "uint8", "bool", "uint32"],
                            [0, bytes(32), 0, 0, 0, 0, False, 15003],
                        ).hex(),
                    }
                ],
            }
        )
        response = client.execute_order(
            {
                "symbol": "BTCUSDT",
                "orderType": "Limit",
                "side": "Bid",
                "price": "80000",
                "quantity": "0.0002",
                "postOnly": True,
            }
        )
        self.assertFalse(response.success)
        self.assertIn("code 15003", response.error_message)
        self.assertIn("one-way position mode", response.error_message)

    def test_fill_mapping_uses_popdex_exec_fields(self):
        client = self.make_client()
        client.make_request = Mock(
            return_value={
                "code": "200",
                "data": [
                    {
                        "execId": "fill-1",
                        "orderId": "order-1",
                        "clientOid": "client-1",
                        "symbol": "BTCUSDT",
                        "side": "Sell",
                        "execPrice": "106950.1",
                        "execQty": "0.01",
                        "tradeScope": "Taker",
                        "feeDetail": [{"feeToken": "USDT", "fee": "0.6417006"}],
                        "updatedAt": "1750141421728",
                    }
                ],
            }
        )
        response = client.get_fill_history("BTCUSDT")
        self.assertTrue(response.success)
        fill = response.data[0]
        self.assertEqual(fill.trade_id, "fill-1")
        self.assertEqual(str(fill.price), "106950.1")
        self.assertEqual(str(fill.size), "0.01")
        self.assertEqual(fill.fee_asset, "USDT")
        self.assertFalse(fill.is_maker)


if __name__ == "__main__":
    unittest.main()
