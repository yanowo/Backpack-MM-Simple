"""Regression tests for the POPDEX Agent Web interface."""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "web" / "static" / "js" / "app.js"


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
        self.assertLess(html.index("基本設置"), html.index('id="popdexAgentLauncher"'))
        self.assertLess(html.index('id="popdexAgentLauncher"'), html.index('id="exchange"'))

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


if __name__ == "__main__":
    unittest.main()
