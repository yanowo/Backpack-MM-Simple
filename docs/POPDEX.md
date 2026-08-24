# POPDEX 接入說明

POPDEX 在本專案中支援永續做市、Maker-Taker 對沖與永續網格。市場與帳戶快照使用 REST；即時 ticker、五檔訂單簿、訂單、成交、倉位與帳戶事件使用官方 WebSocket；下單、撤單與全撤使用 POPDEX 固定 Order 合約，在本地以 Agent Key 簽署後透過 JSON-RPC 廣播。WS 中斷時會以 REST 暫時備援並自動重連。

## 安全設定

1. 在 POPDEX 建立並授權 secp256k1 Agent Key，只開啟策略所需的交易權限。
2. `POPDEX_ACCOUNT_ADDRESS` 填寫資金帳戶的 EVM 錢包地址。
3. `POPDEX_AGENT_PRIVATE_KEY` 只填已授權的 Agent Key 私鑰。不要填入存放資金的主錢包私鑰。
4. `.env` 已被 Git 忽略；不要把 Agent Key 寫進程式碼、日誌或 commit。

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

## 驗證

先用小額與測試網驗證 Agent Key 權限、交易對最小數量、PostOnly 行為與 reduce-only 平倉。程式會等待上鏈 receipt；超過 `POPDEX_RECEIPT_TIMEOUT` 時會回傳 client order ID，之後可以該 ID 撤單，不會重送可能已上鏈的訂單。

官方參考：[POPDEX Product Docs](https://popdex.xyz/docs/product-docs/intro)、[Quick Start](https://popdex.xyz/docs/common/quick-start)、[WebSocket Introduction](https://popdex.xyz/docs/websocket/intro)。
