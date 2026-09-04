import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from api.base_client import ApiResponse, MarketInfo
from api.bp_client import BPClient
from strategies.market_maker import MarketMaker
from ws_client.backpack_ws_client import BackpackWebSocket


SNDK_MARKET = {
    "symbol": "SNDK.US_USDC_PERP",
    "baseSymbol": "SNDK.US",
    "quoteSymbol": "USDC",
    "marketType": "PERP",
    "rwaMarketType": "STOCK",
    "orderBookState": "Open",
    "visible": True,
    "filters": {
        "price": {
            "minPrice": "0.01",
            "maxPrice": "5000",
            "tickSize": "0.01",
        },
        "quantity": {
            "minQuantity": "0.001",
            "maxQuantity": None,
            "stepSize": "0.001",
        },
    },
}


class BackpackTradFiPerpTests(unittest.TestCase):
    def test_market_metadata_uses_official_market_type_and_filters(self):
        client = BPClient({})
        with patch.object(client, "make_request", return_value=[SNDK_MARKET]):
            response = client.get_markets()

        self.assertTrue(response.success)
        market = response.data[0]
        self.assertEqual(market.symbol, "SNDK.US_USDC_PERP")
        self.assertEqual(market.market_type, "PERP")
        self.assertEqual(market.status, "OPEN")
        self.assertEqual(market.min_order_size, Decimal("0.001"))
        self.assertEqual(market.step_size, Decimal("0.001"))
        self.assertEqual(market.tick_size, Decimal("0.01"))
        self.assertEqual(market.base_precision, 3)
        self.assertEqual(market.quote_precision, 2)
        self.assertEqual(market.raw["rwaMarketType"], "STOCK")

    def test_tradfi_perp_aliases_resolve_to_official_symbol(self):
        client = BPClient({})
        with patch.object(client, "make_request", return_value=[SNDK_MARKET]):
            self.assertEqual(client.resolve_market_symbol("SNDK-PERP"), "SNDK.US_USDC_PERP")

        self.assertEqual(client.resolve_market_symbol("sndk_perp"), "SNDK.US_USDC_PERP")
        self.assertEqual(client.resolve_market_symbol("SNDK_USDC_PERP"), "SNDK.US_USDC_PERP")
        self.assertEqual(client.resolve_market_symbol("SNDK.US-PERP"), "SNDK.US_USDC_PERP")
        self.assertEqual(client.resolve_market_symbol("SNDK.US_USDC_PERP"), "SNDK.US_USDC_PERP")

    def test_market_limits_accept_tradfi_perp_alias(self):
        client = BPClient({})
        with patch.object(client, "make_request", return_value=[SNDK_MARKET]):
            response = client.get_market_limits("SNDK-PERP")

        self.assertTrue(response.success)
        self.assertEqual(response.data.symbol, "SNDK.US_USDC_PERP")
        self.assertEqual(response.data.base_asset, "SNDK.US")

    def test_order_payload_uses_official_tradfi_symbol(self):
        client = BPClient({"api_key": "test-key", "secret_key": "test-secret"})
        client._market_aliases = {"SNDKPERP": "SNDK.US_USDC_PERP"}
        order_response = {
            "id": "123",
            "symbol": "SNDK.US_USDC_PERP",
            "side": "Bid",
            "orderType": "Limit",
            "quantity": "0.001",
            "price": "1500.00",
            "status": "New",
        }

        with patch.object(client, "make_request", return_value=order_response) as request:
            response = client.execute_order({
                "symbol": "SNDK-PERP",
                "side": "Bid",
                "orderType": "Limit",
                "quantity": "0.001",
                "price": "1500.00",
            })

        self.assertTrue(response.success)
        params = request.call_args.args[5]
        data = request.call_args.args[6]
        self.assertEqual(params["symbol"], "SNDK.US_USDC_PERP")
        self.assertEqual(data["symbol"], "SNDK.US_USDC_PERP")

    def test_websocket_streams_use_official_tradfi_symbol(self):
        with patch.object(
            BPClient,
            "resolve_market_symbol",
            return_value="SNDK.US_USDC_PERP",
        ):
            ws = BackpackWebSocket(symbol="SNDK-PERP")

        self.assertEqual(ws.symbol, "SNDK.US_USDC_PERP")
        self.assertEqual(ws._get_ticker_channel(), "bookTicker.SNDK.US_USDC_PERP")
        self.assertEqual(ws._get_depth_channel(), "depth.SNDK.US_USDC_PERP")
        self.assertEqual(
            ws._get_order_update_channel(),
            "account.orderUpdate.SNDK.US_USDC_PERP",
        )

    def test_strategy_uses_resolved_symbol_for_rest_and_websocket(self):
        market = MarketInfo(
            symbol="SNDK.US_USDC_PERP",
            base_asset="SNDK.US",
            quote_asset="USDC",
            market_type="PERP",
            status="OPEN",
            min_order_size=Decimal("0.001"),
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            base_precision=3,
            quote_precision=2,
            raw=SNDK_MARKET,
        )
        rest_client = Mock()
        rest_client.get_market_limits.return_value = ApiResponse.ok(market)
        ws = Mock()

        with (
            patch("strategies.market_maker.get_client", return_value=rest_client),
            patch("strategies.market_maker.BackpackWebSocket", return_value=ws) as ws_class,
            patch.object(MarketMaker, "_initialize_websocket"),
            patch.object(MarketMaker, "_load_trading_stats"),
            patch.object(MarketMaker, "_load_recent_trades"),
        ):
            strategy = MarketMaker(
                api_key="test-key",
                secret_key="test-secret",
                symbol="SNDK-PERP",
                exchange="backpack",
                enable_database=False,
            )

        try:
            self.assertEqual(strategy.symbol, "SNDK.US_USDC_PERP")
            rest_client.get_market_limits.assert_called_once_with("SNDK-PERP")
            self.assertEqual(ws_class.call_args.args[2], "SNDK.US_USDC_PERP")
        finally:
            strategy.executor.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
