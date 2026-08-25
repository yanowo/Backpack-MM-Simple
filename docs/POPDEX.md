# POPDEX 接入說明

POPDEX 在本專案中支援永續做市、Maker-Taker 對沖與永續網格。市場與帳戶快照使用 REST；即時 ticker、五檔訂單簿、訂單、成交、倉位與帳戶事件使用官方 WebSocket；下單、撤單與全撤使用 POPDEX 固定 Order 合約，在本地以 Agent Key 簽署後透過 JSON-RPC 廣播。WS 中斷時會以 REST 暫時備援並自動重連。

## 安全設定

1. 在 POPDEX 建立並授權 secp256k1 Agent Key，只開啟策略所需的交易權限。
2. `POPDEX_ACCOUNT_ADDRESS` 填寫資金帳戶的 EVM 錢包地址。
3. `POPDEX_AGENT_PRIVATE_KEY` 只填已授權的 Agent Key 私鑰。不要填入存放資金的主錢包私鑰。
4. `.env` 已被 Git 忽略；不要把 Agent Key 寫進程式碼、日誌或 commit。

### Web 控制台快速設定

啟動 `python run.py --web` 後，控制台會立即在背景檢查本機金鑰與鏈上 Agent 狀態，不需要先選擇交易所。在「基本設置」選擇 `POPDEX` 時，右側會顯示已預先更新狀態的「設定／管理 POPDEX Agent」按鈕。點擊按鈕會以彈出視窗開啟授權精靈，完成或關閉後不會持續佔用策略設定頁面：

1. 連接任一支援 EIP-1193 / EIP-6963 的 EVM 瀏覽器錢包，例如 OKX Wallet、MetaMask、Rabby 或 Coinbase Wallet。
2. 按「建立本機 Agent」；程式會在本機產生獨立 Agent Key，寫入 `.env`，Web API 不會回傳或顯示私鑰。
3. 以剛連接的主錢包確認 `approveAgent` 鏈上交易；確認後頁面會查詢 Account 合約並顯示授權完成。

授權完成後，精靈會額外提供兩個管理動作：

- 「備份並更換授權」會先備份目前設定，再產生全新的 Agent Key，並透過官方 `replaceAgent(oldAgent, newAgent, expiresAt, initialNonce)` 在單一鏈上交易中原子輪換；交易確認後才會把 `.env` 切換至新金鑰。
- 「備份並取消授權」會先備份目前設定，再由主錢包送出官方 `revokeAgent(agent)`；鏈上確認後會清空 `.env` 的舊 Agent 私鑰，保留帳戶與網路設定，方便日後使用全新地址重新授權。

備份只包含 `POPDEX_*` Agent 相關設定，保存在 `.popdex-agent-backups/` 並由 Git 忽略；不會複製 Backpack 或其他交易所憑證。備份仍含舊 Agent 私鑰，請視同敏感檔案妥善保管。已撤銷的地址依官方安全建議不得再次授權，備份用途是稽核與緊急查閱，不應用來恢復已撤銷金鑰。

這個設定 API 僅接受 loopback（`127.0.0.1` / `::1`）請求。OKX 等錢包只要在目前瀏覽器注入 EVM provider，或在其行動版 DApp 瀏覽器開啟控制台，即可被偵測；同一擴充錢包即使同時透過 EIP-6963 與舊式 injected provider 公告，也只會顯示一次。純行動版的 WalletConnect QR 流程需要另外配置 WalletConnect Project ID，目前沒有內建。

主錢包只負責簽署 Agent 授權與權限管理，策略執行時只使用 Agent Key。已撤銷、過期或綁定其他帳戶的 Agent 不會被 `approveAgent` 自動覆寫或重複授權；有效 Agent 的更換會使用官方 `replaceAgent` 並產生全新地址。

```dotenv
POPDEX_ACCOUNT_ADDRESS=0xyour_popdex_wallet_address
POPDEX_AGENT_PRIVATE_KEY=0xyour_authorized_agent_private_key
POPDEX_NETWORK=mainnet
POPDEX_BASE_URL=https://api.popdex.xyz
POPDEX_RPC_URL=https://api.popdex.xyz/api/v1/web3/rpc
POPDEX_WS_URL=wss://ws.popdex.xyz/v1/ws/public
POPDEX_CHAIN_ID=2184
POPDEX_MARKET_SLIPPAGE=0.01
POPDEX_RECEIPT_TIMEOUT=8
```

`POPDEX_NETWORK=testnet` 時的官方預設值為：

```dotenv
POPDEX_NETWORK=testnet
POPDEX_BASE_URL=https://testnet-api.popdex.xyz
POPDEX_RPC_URL=https://testnet-api.popdex.xyz/api/v1/web3/rpc
POPDEX_WS_URL=wss://testnet-ws.popdex.xyz/v1/ws/public
POPDEX_CHAIN_ID=34952
```

`POPDEX_BASE_URL` / `POPDEX_RPC_URL` / `POPDEX_WS_URL` / `POPDEX_CHAIN_ID` 未設定時，程式會根據 `POPDEX_NETWORK` 自動套用上述值。

## WebSocket

POPDEX 的市場與錢包頻道共用同一個 public endpoint，不需要額外登入握手。程式會訂閱 `ticker`、`books5`、`order`、`fill`、`position`、`account`；錢包頻道以 `POPDEX_ACCOUNT_ADDRESS` 作為 `walletId`。依官方規格每 30 秒送出純文字 `ping` 並等待純文字 `pong`，未收到回覆時觸發重連。`books5` 每次推送皆為完整快照，因此重連後不需要補增量序號。

## 策略所需介面覆蓋

| 用途 | POPDEX 介面 | 本專案用途 |
|---|---|---|
| 市場規格 | `GET /api/v1/config/symbols` | tick size、lot size、最小委託量 |
| 行情快照 | `GET /api/v1/public/market/tickers` | WS 尚未就緒或中斷時備援 |
| 訂單簿快照 | `GET /api/v1/public/market/orderbook` | 啟動與 WS 中斷時備援 |
| 帳戶概況 | `GET /api/v1/account/{walletId}/overview` | 餘額與保證金資訊 |
| 委託 | `GET /api/v1/account/{walletId}/orders` 與 client OID 查詢 | 掛單同步與撤單定位 |
| 倉位 | `GET /api/v1/account/{walletId}/positions` | 永續風控與對沖 |
| 成交 | `GET /api/v1/account/{walletId}/trade/fills` | 啟動去重與 WS 中斷備援 |
| 交易寫入 | `/api/v1/web3/rpc` + Order precompile | 下單、單撤、全撤與 receipt |
| 即時行情 | WS `ticker`、`books5` | 做市報價 |
| 即時帳戶 | WS `order`、`fill`、`position`、`account` | 成交統計、訂單與風控事件 |

K 線、public trades 等 POPDEX 亦提供的頻道目前沒有訂閱，因為現有做市、對沖與網格策略不依賴它們；這不是策略執行所需介面的缺口。

## 使用

POPDEX 僅開啟永續市場，symbol 可使用 `BTCUSDT`，程式也會將 `BTC_USDT_PERP` 正規化為 `BTCUSDT`。

```bash
python run.py --exchange popdex --market-type perp --symbol BTCUSDT --spread 0.01 --quantity 0.001 --max-orders 2 --target-position 0 --max-position 1 --position-threshold 0.1 --duration 3600 --interval 10
```

交易寫入使用 POPDEX 官方規格：Order 合約 `0x0000000000000000000000000000000000001000`、legacy transaction、`gasPrice=0`、`gas=1,000,000`，Agent Key nonce 使用單調遞增的毫秒時間戳。價格、數量、滑點與 builder fee rate 在合約參數中均使用 18 位定點數。策略未單獨提供滑點時，市價單預設使用 `POPDEX_MARKET_SLIPPAGE=0.01`（1%）；設為 `0` 代表官方語意的「不限滑點」，請謹慎使用。

程式會在首次下單前讀取 `/api/v1/account/{walletId}/settings`，依帳戶的 `Oneway`／`Hedge` 模式設定 Order 合約的 `positionSide`。One-way 模式固定使用 `None`；Hedge 模式才會使用 `Long`／`Short`，避免 POPDEX 錯誤碼 `15002`／`15003`。

## 驗證

先用小額與測試網驗證 Agent Key 權限、交易對最小數量、PostOnly 行為與 reduce-only 平倉。程式會等待上鏈 receipt；超過 `POPDEX_RECEIPT_TIMEOUT` 時會回傳 client order ID，之後可以該 ID 撤單，不會重送可能已上鏈的訂單。

官方參考：[POPDEX Product Docs](https://popdex.xyz/docs/product-docs/intro)、[Quick Start](https://popdex.xyz/docs/common/quick-start)、[Agent Accounts](https://popdex.xyz/docs/product-docs/chain/agent-key)、[Account Contract](https://popdex.xyz/docs/smart-contract/Account)、[WebSocket Introduction](https://popdex.xyz/docs/websocket/intro)。
