"""Regression tests for the POPDEX Agent Web interface."""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "web" / "static" / "js" / "app.js"
WALLET_ASSETS = ROOT / "web" / "static" / "images" / "wallets"


class PopdexWebUiTest(unittest.TestCase):
    def test_agent_wizard_uses_conditional_launcher_and_modal(self):
        from web.server import app

        response = app.test_client().get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="popdexAgentLauncher"', html)
        self.assertIn('aria-controls="popdexAgentModal"', html)
        self.assertIn('id="popdexAgentModal" class="agent-modal" hidden role="dialog"', html)
        self.assertIn('id="closePopdexAgentModal"', html)
        self.assertIn('id="replacePopdexAgentBtn"', html)
        self.assertIn('id="revokePopdexAgentBtn"', html)
        self.assertIn('id="agentManagementActions"', html)
        self.assertLess(html.index("基本設置"), html.index('id="popdexAgentLauncher"'))
        self.assertLess(html.index('id="popdexAgentLauncher"'), html.index('id="exchange"'))

    def test_agent_status_is_preloaded_without_selecting_popdex(self):
        app_js = APP_JS.read_text(encoding="utf-8")
        init_start = app_js.index("function initPopdexAgentSetup()")
        toggle_start = app_js.index("function togglePopdexAgentPanel()")
        init_body = app_js[init_start:toggle_start]

        self.assertIn("togglePopdexAgentPanel();", init_body)
        self.assertIn("loadPopdexAgentStatus();", init_body)
        self.assertNotIn(
            "exchangeSelect.value !== 'popdex'",
            app_js[app_js.index("async function loadPopdexAgentStatus()"):app_js.index("function applyPopdexAgentStatus")],
        )

    def test_author_contact_links_are_present_in_header_and_footer(self):
        from web.server import app

        html = app.test_client().get("/").get_data(as_text=True)
        self.assertGreaterEqual(html.count("https://x.com/practice_y11"), 2)
        self.assertIn("https://t.me/crypto_practice", html)
        self.assertIn("https://github.com/yanowo/Backpack-MM-Simple", html)
        self.assertIn("YanOwO", html)

    def test_quantitative_trading_brand_copy_is_concise(self):
        from web.server import app

        html = app.test_client().get("/").get_data(as_text=True)
        self.assertIn("量化交易控制台", html)
        self.assertNotIn("做市交易控制枱", html)
        self.assertNotIn("專業交易儀錶板", html)
        self.assertNotIn("Backpack 做市交易儀錶板", html)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the JavaScript regression test")
    def test_wallet_discovery_merges_eip6963_and_legacy_duplicates(self):
        script = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const sandbox = {
    console,
    Event: function Event(type) { this.type = type; },
    window: {
        addEventListener() {},
        dispatchEvent() {},
        requestAnimationFrame(callback) { callback(); },
        setTimeout(callback) { callback(); },
    },
    document: {
        addEventListener() {},
        getElementById() { return null; },
    },
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox, {filename: process.argv[1]});
vm.runInContext(`
    registerEvmWallet({request() {}, isOkxWallet: true});
    registerEvmWallet(
        {request() {}},
        {uuid: 'okx-eip-a', name: 'OKX Wallet', rdns: 'com.okx.wallet'}
    );
    registerEvmWallet(
        {request() {}},
        {uuid: 'okx-eip-b', name: 'OKX Wallet', rdns: 'com.okex.wallet'}
    );

    registerEvmWallet({request() {}, isBackpack: true});
    registerEvmWallet(
        {request() {}},
        {uuid: 'backpack-eip-a', name: 'Backpack', rdns: 'app.backpack'}
    );
    registerEvmWallet(
        {request() {}},
        {uuid: 'backpack-eip-b', name: 'Backpack Wallet', rdns: 'app.backpack'}
    );

    registerEvmWallet({request() {}, isMetaMask: true});
    if (discoveredEvmWallets.size !== 3) {
        throw new Error('expected 3 unique wallets, received ' + discoveredEvmWallets.size);
    }
    if (!getWalletFallbackLogo('wallet:phantom').endsWith('/phantom.svg')) {
        throw new Error('Phantom fallback logo is missing');
    }
    if (!getWalletFallbackLogo('wallet:metamask').endsWith('/metamask.svg')) {
        throw new Error('MetaMask fallback logo is missing');
    }
`, sandbox);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script, str(APP_JS)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_phantom_and_metamask_have_local_fallback_logos(self):
        app_js = APP_JS.read_text(encoding="utf-8")

        for wallet in ("phantom", "metamask"):
            asset = WALLET_ASSETS / f"{wallet}.svg"
            self.assertTrue(asset.is_file())
            self.assertIn(f"/static/images/wallets/{wallet}.svg", app_js)
            self.assertIn("<svg", asset.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
