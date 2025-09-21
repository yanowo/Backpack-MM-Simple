# websea_client.py
"""
Websea exchange client implementation.
Validated against https://webseaex.github.io/en/ (Futures Trading & Futures Market docs).
"""

import os
import time
import math
import random
import string
from typing import Dict, Any, Optional, List
from decimal import Decimal
import hashlib
import hmac
import json
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import asyncio
import json
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, List, Optional, Tuple, Callable

import requests
import websockets

from .base_client import BaseExchangeClient, OrderResult, OrderInfo, query_retry
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger


# -------------------------- WebSocket (public market) --------------------------

class WebseaWSManager:
    """
    Public market WebSocket manager.
    Docs (Futures WS):
      - kline:   ws wss://coapi.websea.com/future-ws/v1/market channel=kline{period}  e.g. kline1min
      - tickers: ws wss://coapi.websea.com/future-ws/v1/market channel=tickers
      - trade:   ws wss://coapi.websea.com/future-ws/v1/market channel=trade
    """
    def __init__(self, url: str = "wss://coapi.websea.com/future-ws/v1/market"):
        self.url = url
        self._handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self._conn: Optional[websockets.WebSocketClientProtocol] = None
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self):
        self._conn = await websockets.connect(self.url)
        self._listen_task = asyncio.create_task(self._listen())

    async def disconnect(self):
        if self._listen_task:
            self._listen_task.cancel()
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _listen(self):
        try:
            async for raw in self._conn:
                data = json.loads(raw)
                ch = data.get("channel")
                if ch and ch in self._handlers:
                    try:
                        self._handlers[ch](data)
                    except Exception:
                        # swallow handler exceptions so listener keeps running
                        pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[WebseaWS] listener error: {e}")

    async def sub(self, channel: str, symbol: str):
        if not self._conn:
            raise RuntimeError("WebSocket not connected")
        await self._conn.send(json.dumps({"op": "sub", "channel": channel, "symbol": symbol}))

    async def unsub(self, channel: str, symbol: str):
        if not self._conn:
            raise RuntimeError("WebSocket not connected")
        await self._conn.send(json.dumps({"op": "unsub", "channel": channel, "symbol": symbol}))

    def on_message(self, channel: str, handler: Callable[[Dict[str, Any]], None]):
        self._handlers[channel] = handler


# ------------------------------- REST Client ----------------------------------

class WebseaClient(BaseExchangeClient):
    """
    Websea exchange client.
    Auth: headers Token / Nonce / Signature(sha1 over sorted [TOKEN, SECRET, Nonce] + all k=v)
    Endpoints (Futures):
      - Place Order:           POST /openApi/contract/add           (open/close, buy/sell-{limit|market})
      - Cancel Order:          POST /openApi/contract/cancel        (order_id)
      - Current Orders:        GET  /openApi/contract/currentList   (status int 1..6)
      - Order Detail:          GET  /openApi/contract/getOrderDetail(order_id)
      - Positions:             GET  /openApi/contract/position
      - Wallet (full):         GET  /openApi/contract/walletList/full?trade_area=USDT
      - Wallet (isolated):     GET  /openApi/contract/walletList
      - Precision:             GET  /openApi/contract/precision     (amount, price, minQuantity, minPrice ...)
      - Orderbook (depth):     GET  /openApi/contract/depth?symbol=...&limit=...
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.token = os.getenv("WEBSEA_TOKEN")
        self.secret = os.getenv("WEBSEA_SECRET")
        self.base_url = os.getenv("WEBSEA_BASE_URL", "https://coapi.websea.com")

        if not self.token or not self.secret:
            raise ValueError("WEBSEA_TOKEN and WEBSEA_SECRET must be set in environment variables")

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "websea-client/1.0"
        })

        # 检查config是否有ticker属性
        ticker = getattr(self.config, 'ticker', None) if hasattr(self.config, 'ticker') else self.config.get('ticker', 'DEFAULT')
        self.logger = setup_logger(f"websea_{ticker}")

        # Order update via REST polling (since no private WS)
        self._order_update_handler: Optional[Callable[[Dict[str, Any]], None]] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._known_status: Dict[str, int] = {}   # order_id -> last_status_code

        # Precision cache: {symbol: {"price": int_decimals, "amount": int_decimals, "minQuantity": str, ...}}
        self._precision: Dict[str, Dict[str, Any]] = {}

    # ---------- lifecycle ----------

    def _validate_config(self) -> None:
        required_env_vars = ['WEBSEA_TOKEN', 'WEBSEA_SECRET']
        missing = [v for v in required_env_vars if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

    async def connect(self) -> None:
        # No private WS to connect; just warm precision cache
        try:
            ticker = getattr(self.config, 'ticker', None) if hasattr(self.config, 'ticker') else self.config.get('ticker')
            if ticker:
                await self._warm_precision(ticker)
        except Exception as e:
            self.logger.warning(f"[Websea] precision warmup failed: {e}")

    async def disconnect(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
        self.session.close()

    def get_exchange_name(self) -> str:
        return "websea"

    # ---------- 实现BaseExchangeClient的抽象方法 ----------

    def make_request(self, method: str, endpoint: str, api_key=None, secret_key=None,
                     instruction=None, params=None, data=None, retry_count: int = 3) -> Dict:
        """Perform HTTP request and return parsed JSON dict or {'error': ...}."""
        try:
            url = self.base_url.rstrip("/") + endpoint
            
            # 使用websea的认证方法
            if method.upper() == "GET":
                headers = self._auth_headers(params or {})
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == "POST":
                headers = self._auth_headers(data or {})
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                response = self.session.post(url, data=data, headers=headers, timeout=10)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    # ---------- signing ----------

    def _make_nonce(self) -> str:
        ts = int(time.time())
        rand5 = ''.join(random.sample(string.ascii_letters + string.digits, 5))
        return f"{ts}_{rand5}"

    def _sign(self, nonce: str, params: Dict[str, Any]) -> str:
        arr = [self.token, self.secret, nonce]
        for k, v in (params or {}).items():
            arr.append(f"{str(k)}={str(v)}")
        arr.sort()
        return hashlib.sha1(''.join(arr).encode("utf-8")).hexdigest()

    def _auth_headers(self, params: Dict[str, Any]) -> Dict[str, str]:
        nonce = self._make_nonce()
        return {
            "Token": self.token,
            "Nonce": nonce,
            "Signature": self._sign(nonce, params or {}),
        }

    # ---------- HTTP helpers ----------

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        headers = self._auth_headers(params)
        r = self.session.get(url, params=params, headers=headers, timeout=10)
        return r.json()

    def _post(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        headers = self._auth_headers(params)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        r = self.session.post(url, data=params, headers=headers, timeout=10)
        return r.json()

    # ---------- precision / rounding ----------

    async def _warm_precision(self, symbol: str):
        # GET /openApi/contract/precision  (optional symbol param)
        res = self._get("/openApi/contract/precision", {"symbol": symbol})
        if res.get("errno") == 0 and "result" in res:
            data = res["result"]
            # API returns dict keyed by symbol OR a flat dict when only one symbol?
            if isinstance(data, dict) and symbol in data:
                self._precision[symbol] = data[symbol]
            else:
                # if passing no symbol, it returns { "ETH-USDT": {...}, "BTC-USDT": {...} }
                for k, v in data.items():
                    self._precision[k] = v
        else:
            raise RuntimeError(f"precision failed: {res}")

    def _price_tick(self, symbol: str) -> Decimal:
        # Tick = 10^(-price_decimals)
        decimals = int(self._precision.get(symbol, {}).get("price", "2"))
        return Decimal(10) ** Decimal(-decimals)

    def _amount_step(self, symbol: str) -> Decimal:
        decimals = int(self._precision.get(symbol, {}).get("amount", "0"))
        return Decimal(10) ** Decimal(-decimals)

    def _round_price(self, symbol: str, px: Decimal) -> Decimal:
        step = self._price_tick(symbol)
        # floor to step to be safe
        quant = step
        return (px // quant) * quant

    def _calculate_contract(self, symbol: str, qty: Decimal) -> int:
        """
        把实际标的数量转换为“合约张数”（最小下单单位的倍数，向下取整）。
        逻辑说明：
        - minQuantity 为纯数字字符串，表示数量精度位数 (如 "1" -> 最小步进 0.1, "3" -> 最小步进 0.001)
        - 若没有 minQuantity，则回退到 amount 字段（同样表示小数位数）
        示例：
        qty = 0.1,   minQuantity = "1" -> step=0.1  => 返回 1
        qty = -0.37, minQuantity = "2" -> step=0.01 => 返回 -37
        注意：Websea 文档中 minQuantity 不会返回诸如 "0.05" 这样的字符串，因此不再支持该分支。
        返回：张数（int，保留符号）
        """
        if qty is None or qty == 0:
            return 0

        sign = 1 if qty > 0 else -1
        q = abs(Decimal(qty))

        prec = self._precision.get(symbol, {}) or {}

        # 优先使用 minQuantity
        raw = prec.get("minQuantity")
        step = None
        if raw not in (None, "", "0", 0):
            try:
                digits = int(str(raw).strip())
                step = Decimal(10) ** (-digits)
            except Exception:
                pass

        # 回退到 amount
        if step is None:
            try:
                digits = int(prec.get("amount", 0))
                if digits < 0:
                    digits = 0
                step = Decimal(10) ** (-digits)
            except Exception:
                step = Decimal(1)

        # 张数 = 数量 / step，向下取整
        units = (q / step).to_integral_value(rounding=ROUND_DOWN)

        return int(sign * units)

    
    def _calculate_qty(self, symbol: str, contract_units: Decimal) -> Decimal:
        """
        把“合约张数”转换为实际标的数量。
        逻辑说明：
          - minQuantity 为纯数字字符串，表示数量精度位数 (如 "1" -> 最小步进 0.1, "3" -> 最小步进 0.001)
          - 若没有 minQuantity，则回退到 amount 字段（同样表示小数位数）
        示例：
            contract_units = 1,  minQuantity = "1" -> 最小步进 0.1  => 返回 0.1
            contract_units = 37, minQuantity = "2" -> 最小步进 0.01 => 返回 0.37
        """
        if contract_units <= 0:
            return Decimal(0)

        prec = self._precision.get(symbol, {}) or {}
        raw = prec.get("minQuantity")
        if float(raw)%1 != 0:
            return Decimal(0)
        
        digits = int(str(raw).strip())
        step = Decimal(10) ** Decimal(-digits)

        qty = contract_units * step
        return qty if qty > 0 else Decimal(0)

    # ---------- 添加round_to_tick方法以确保向后兼容 ----------
    def round_to_tick(self, price: Decimal, symbol: str = None) -> Decimal:
        """Round price to tick size"""
        if symbol is None:
            # 尝试从config获取ticker
            symbol = getattr(self.config, 'ticker', None) if hasattr(self.config, 'ticker') else self.config.get('ticker')
        
        if symbol and symbol in self._precision:
            return self._round_price(symbol, price)
        else:
            # 如果没有精度信息，使用默认的两位小数
            return price.quantize(Decimal('0.01'), rounding=ROUND_DOWN)

    # ---------- order placement (EdgeX-like API) ----------

    async def _best_bid_ask(self, symbol: str) -> Tuple[Decimal, Decimal]:
        # GET /openApi/contract/depth
        res = self._get("/openApi/contract/depth", {"symbol": symbol, "limit": 15})
        if res.get("errno") != 0:
            raise RuntimeError(f"depth error: {res.get('errmsg')}")
        data = res.get("result", {})
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        best_bid = Decimal(str(bids[0][0])) if bids else Decimal("0")
        best_ask = Decimal(str(asks[0][0])) if asks else Decimal("0")
        return best_bid, best_ask
    
    async def _quick_check_and_emit(self, order_id: str, symbol: str, side: str, order_type: str = "OPEN"):
        # 在 2.5 秒内，每 250ms 查一次订单详情；若成交/部分成交，立刻发回调
        for _ in range(10):
            info = await self.get_order_info(order_id)
            if info and info.status in ("FILLED", "PARTIALLY_FILLED"):
                if self._order_update_handler:
                    self._order_update_handler({
                        "order_id": order_id,
                        "side": side,
                        "order_type": order_type,
                        "status": info.status,
                        "size": str(info.size),
                        "price": str(info.price),
                        "contract_id": symbol,
                        "filled_size": str(info.filled_size),
                    })
                break
            await asyncio.sleep(0.25)

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """
        EdgeX 风格签名：传 contract_id(=symbol)、数量、方向('buy'/'sell')，自动择价做 maker 限价单。
        实际下单接口：POST /openApi/contract/add  (docs: Futures opening/closing)
        
        注意：Websea的amount字段支持小数，精度由API返回的amount字段决定
        例如SOL-USDT支持1位小数(0.1)，BTC-USDT支持3位小数(0.001)
        """
        symbol = contract_id
        try:
            # ensure precision cache
            if symbol not in self._precision:
                await self._warm_precision(symbol)

            best_bid, best_ask = await self._best_bid_ask(symbol)
            if best_bid <= 0 or best_ask <= 0:
                return OrderResult(success=False, error_message="empty orderbook")

            tick = self._price_tick(symbol)
            # maker price: buy -> just below best ask; sell -> just above best bid
            if direction.lower() == "buy":
                price = self._round_price(symbol, best_ask - tick)
                order_type = "buy-limit"
            else:
                price = self._round_price(symbol, best_bid + tick)
                order_type = "sell-limit"
            contract_quantity = self._calculate_contract(symbol, Decimal(quantity))
            amount_str = str(contract_quantity)  # 使用精度对齐后的小数字符串

            # 获取杠杆和is_full配置
            leverage = getattr(self.config, 'leverage', None) if hasattr(self.config, 'leverage') else self.config.get('leverage', 10)
            is_full = getattr(self.config, 'is_full', None) if hasattr(self.config, 'is_full') else self.config.get('is_full', 2)

            params = {
                "contract_type": "open",
                "type": order_type,          # "buy-limit" / "sell-limit"
                "symbol": symbol,
                "amount": amount_str,        # 使用精度对齐的小数字符串
                "price": str(price),
                "lever_rate": str(int(leverage)),
                "is_full": str(int(is_full)),
            }

            res = self._post("/openApi/contract/add", params)

            if res.get("errno") == 0:
                order_id = res["result"]["order_id"]
                # ✅ 立刻做一次快速轮询，减少超时-撤单竞态
                asyncio.create_task(self._quick_check_and_emit(order_id, symbol, direction))
                return OrderResult(success=True, order_id=order_id, side=direction, size=contract_quantity, price=price)
            else:
                return OrderResult(success=False, error_message=res.get("errmsg", "place order failed"))
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """
        Websea平仓：单函数内完成"可平数量等待 + 提交下单 + Insufficient position 微重试"。
        - 维持 maker 价：按 tick_size 把价格对齐（不强制贴近盘口，交给你传入的 close_price）。
        - 数量规则：根据symbol的amount精度对齐，支持小数（非整数）
        """
        try:
            symbol = contract_id
            # 修正：使用精度对齐而不是强制整数
            contract_quantity = self._calculate_contract(symbol, quantity)
            if contract_quantity <= 0:
                return OrderResult(success=False, error_message="Quantity too small for symbol precision")

            px = self.round_to_tick(price, symbol)

            max_retries = 15           # 最多尝试 15 次
            backoff_s   = 0.25         # 每次失败后等 250ms
            is_full     = "2"          # 逐仓=1 / 全仓=2 —— 与开仓保持一致，如你需要可改成 self.is_full

            for attempt in range(1, max_retries + 1):
                # 1) 轻量查询"可平数量"是否到位（Websea 成交到账有 200~500ms 延迟）
                try:
                    pos = self._get("/openApi/contract/position", {"symbol": symbol, "is_full": is_full})
                    if pos.get("errno") == 0:
                        avail_amt = Decimal("0")
                        for p in pos.get("result", []):
                            if p.get("symbol") == symbol:
                                # Websea 字段命名可能是 avail_amount / avail，二者择其一
                                avail_amt = Decimal(str(p.get("avail_amount") or p.get("avail") or "0"))
                                break
                        if avail_amt < contract_quantity:
                            await asyncio.sleep(backoff_s)
                            continue
                except Exception:
                    # 读仓位失败不致命，继续尝试
                    pass

                # 2) 提交平仓限价单
                params = {
                    "contract_type": "close",
                    "type": f"{side.lower()}-limit",   # buy-limit / sell-limit
                    "symbol": symbol,
                    "amount": str(contract_quantity),  # 使用精度对齐的小数字符串
                    "price": str(px),
                    "is_full": is_full,
                }
                res = self._post("/openApi/contract/add", params)

                # 3) 成功直接返回
                if res and res.get("errno") == 0:
                    oid = res["result"]["order_id"]
                    return OrderResult(
                        success=True,
                        order_id=oid,
                        side=side.lower(),
                        size=contract_quantity,
                        price=px,
                        status="OPEN"  # 下出去的委托状态
                    )

                # 4) 失败分类处理：常见的是"Insufficient position"，做微重试
                err = (res or {}).get("errmsg", "Unknown error")
                low = err.lower()
                if "insufficient position" in low or "not enough" in low:
                    await asyncio.sleep(backoff_s)
                    continue

                # 其它错误（价格精度、参数问题等）直接返回
                return OrderResult(success=False, error_message=err)

            return OrderResult(success=False, error_message="Insufficient position (max retries exceeded)")
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    async def cancel_order(self, order_id: str) -> OrderResult:
        """撤单"""
        try:
            # Websea 要求使用 order_ids（可逗号分隔多个）
            params = {"order_ids": str(order_id)}
            result = self._post("/openApi/contract/cancel", params)
            if result.get("errno") == 0:
                return OrderResult(success=True)
            else:
                return OrderResult(success=False, error_message=result.get("errmsg"))
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    # ---------- order queries ----------

    @staticmethod
    def _map_status_int_to_text(code: int) -> str:
        # 1=Pending, 2=Partially Filled, 3=Filled, 4=Cancelling, 5=Partially Cancelled, 6=Cancelled
        return {
            1: "OPEN",
            2: "PARTIALLY_FILLED",
            3: "FILLED",
            4: "CANCELLING",
            5: "PARTIALLY_CANCELLED",
            6: "CANCELED",
        }.get(code, str(code))
    
    @query_retry()
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """查询订单详情：用于取消超时 fallback 场景"""
        params = {"order_id": order_id}
        result = self._get("/openApi/contract/getOrderDetail", params)
        if result and result.get("errno") == 0 and "result" in result:
            o = result["result"] or {}
            raw_type = o.get("type", "")
            side = raw_type.lower().split("-")[0] if raw_type else ""
            try:
                status_code = int(o.get("status", 1))
            except Exception:
                status_code = 1
            status_map = {
                1: "OPEN",
                2: "PARTIALLY_FILLED",
                3: "FILLED",
                4: "CANCELING",
                5: "PARTIALLY_CANCELED",
                6: "CANCELED",
            }
            status = status_map.get(status_code, "OPEN")
            amount = Decimal(str(o.get("amount", "0")))
            filled = Decimal(str(o.get("deal_amount", "0")))
            price = Decimal(str(o.get("price", "0")))
            return OrderInfo(
                order_id=str(o.get("order_id") or order_id),
                side=side,
                size=amount,
                price=price,
                status=status,
                filled_size=filled,
                remaining_size=amount - filled
            )
        return None

    @query_retry(default_return=[])
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """GET /openApi/contract/currentList?symbol=...  (active/open orders)"""
        symbol = contract_id
        res = self._get("/openApi/contract/currentList", {"symbol": symbol})
        if res.get("errno") != 0:
            return []
        out: List[OrderInfo] = []
        for o in res.get("result", []):
            status_txt = self._map_status_int_to_text(int(o.get("status", 0)))
            out.append(OrderInfo(
                order_id=o.get("order_id", ""),
                side=str(o.get("type", "")).split("-")[0],
                size=Decimal(str(o.get("amount", "0"))),
                price=Decimal(str(o.get("price", "0"))),
                status=status_txt,
                filled_size=Decimal(str(o.get("deal_amount", "0"))),
                remaining_size=Decimal(str(Decimal(str(o.get("amount", "0"))) - Decimal(str(o.get("deal_amount", "0")))))
            ))
        return out

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """
        返回 (symbol, tick_size)。tick_size 由 /precision 的 price 位数换算：10^(-price).
        """
        symbol = getattr(self.config, 'ticker', None) if hasattr(self.config, 'ticker') else self.config.get('ticker')
        if not symbol:
            raise ValueError("ticker empty")
        await self._warm_precision(symbol)
        tick = self._price_tick(symbol)
        # 在外部保存 contract_id=symbol
        if hasattr(self.config, 'contract_id'):
            self.config.contract_id = symbol
        elif isinstance(self.config, dict):
            self.config['contract_id'] = symbol
        
        if hasattr(self.config, 'tick_size'):
            self.config.tick_size = tick
        elif isinstance(self.config, dict):
            self.config['tick_size'] = tick
            
        return symbol, tick

    # ---------- order update (polling) ----------
    def setup_order_update_handler(self, handler, poll_interval: float = 2.0):
        """
        轮询 /getOrderDetail 模拟订单更新：
        - 归一 side: buy-limit/sell-limit/... → buy/sell
        - 用 contract_type 判定 OPEN/CLOSE
        - 映射状态码到 OPEN/PARTIALLY_FILLED/FILLED/...
        - 回调里带上 contract_id 让 bot 不会丢
        """
        # 使用基类的回调机制
        self.set_order_update_callback(handler)
        self._poll_interval = poll_interval
        self._known_status: Dict[str, str] = {}

        STATUS_MAP = {
            1: "OPEN",
            2: "PARTIALLY_FILLED",
            3: "FILLED",
            4: "CANCELING",
            5: "PARTIALLY_CANCELED",
            6: "CANCELED",
        }

        def _normalize_side(order_type: str) -> str:
            if not order_type:
                return ""
            ot = str(order_type).lower()
            if ot.startswith("buy"):
                return "buy"
            if ot.startswith("sell"):
                return "sell"
            return ot

        # 存储已知订单，用于检测成交
        self._tracked_orders: Dict[str, Dict] = {}  # order_id -> order_info

        async def poll_orders():
            while True:
                try:
                    ticker = getattr(self.config, 'ticker', None) if hasattr(self.config, 'ticker') else self.config.get('ticker')
                    if not ticker:
                        await asyncio.sleep(poll_interval)
                        continue
                        
                    current_active_orders = await self.get_active_orders(ticker)
                    current_active_ids = {order.order_id for order in current_active_orders}
                    
                    # 检查之前追踪的订单是否消失（可能已成交）
                    for order_id in list(self._tracked_orders.keys()):
                        if order_id not in current_active_ids:
                            # 订单消失了，检查其最终状态
                            detail = self._get("/openApi/contract/getOrderDetail", {"order_id": order_id})
                            if detail and detail.get("errno") == 0:
                                d = detail.get("result", {}) or {}
                                status_code = d.get("status", 3)  # 默认假设已成交
                                try:
                                    status_code = int(status_code)
                                except Exception:
                                    status_code = 3
                                
                                # 如果状态是已成交或已取消，发送最终状态更新
                                if status_code in [3, 5, 6]:  # FILLED, PARTIALLY_CANCELED, CANCELED
                                    prev_order = self._tracked_orders[order_id]
                                    side = _normalize_side(d.get("type", prev_order.get('side', '')))
                                    ct = str(d.get("contract_type") or d.get("contractType") or "").lower()
                                    order_type = "CLOSE" if ct in ("close", "2") else "OPEN"
                                    status = STATUS_MAP.get(status_code, "FILLED")
                                    amount = Decimal(str(d.get("amount", prev_order.get('amount', 0))))
                                    price = Decimal(str(d.get("price", prev_order.get('price', 0))))
                                    filled = Decimal(str(d.get("deal_amount", 0)))
                                    remaining = amount - filled
                                    
                                    contract_id = getattr(self.config, 'contract_id', None) if hasattr(self.config, 'contract_id') else self.config.get('contract_id', ticker)
                                    self._handle_order_update({
                                        "order_id": order_id,
                                        "side": side,
                                        "order_type": order_type,
                                        "status": status,
                                        "size": str(amount),
                                        "price": str(price),
                                        "filled_size": str(filled),
                                        "remaining_size": str(remaining),
                                        "contract_id": contract_id,
                                    })
                                    self.logger.info(f"[Websea] 訂單 {order_id} 最終状态: {status}, 成交: {filled}")
                            
                            # 从追踪列表中移除
                            del self._tracked_orders[order_id]
                    
                    # 处理当前活跃订单
                    for order in current_active_orders:
                        # 添加到追踪列表
                        if order.order_id not in self._tracked_orders:
                            self._tracked_orders[order.order_id] = {
                                'side': order.side,
                                'amount': str(order.size),
                                'price': str(order.price)
                            }
                        
                        detail = self._get("/openApi/contract/getOrderDetail", {"order_id": order.order_id})
                        if detail and detail.get("errno") == 0:
                            d = detail.get("result", {}) or {}
                            side = _normalize_side(d.get("type", order.side))
                            ct = str(d.get("contract_type") or d.get("contractType") or "").lower()
                            order_type = "CLOSE" if ct in ("close", "2") else "OPEN"
                            status_code = d.get("status", 1)
                            try:
                                status_code = int(status_code)
                            except Exception:
                                status_code = 1
                            status = STATUS_MAP.get(status_code, "OPEN")
                            amount = Decimal(str(d.get("amount", order.size or 0)))
                            price = Decimal(str(d.get("price", order.price or 0)))
                            filled = Decimal(str(d.get("deal_amount", order.filled_size or 0)))
                            remaining = amount - filled

                            prev_status = self._known_status.get(order.order_id)
                            if prev_status != status:
                                self._known_status[order.order_id] = status
                                # 使用基类的回调机制
                                contract_id = getattr(self.config, 'contract_id', None) if hasattr(self.config, 'contract_id') else self.config.get('contract_id', ticker)
                                self._handle_order_update({
                                    "order_id": order.order_id,
                                    "side": side,
                                    "order_type": order_type,
                                    "status": status,
                                    "size": str(amount),
                                    "price": str(price),
                                    "filled_size": str(filled),
                                    "remaining_size": str(remaining),
                                    "contract_id": contract_id,
                                })
                except Exception as e:
                    self.logger.error(f"[Websea] poll_orders error: {e}")

                await asyncio.sleep(poll_interval)

        # 存储轮询函数，稍后在异步上下文中启动
        self._poll_orders_func = poll_orders

    def start_order_polling(self):
        """在异步上下文中启动订单轮询"""
        if hasattr(self, '_poll_orders_func') and not self._poll_task:
            try:
                self._poll_task = asyncio.create_task(self._poll_orders_func())
            except RuntimeError:
                # 如果没有运行的事件循环，记录但不抛出异常
                self.logger.warning("No running event loop, order polling will start when strategy runs")
                pass

    # ---------- 添加与策略兼容的方法 ----------
    
    def get_market_limits(self, symbol: str) -> Optional[Dict]:
        """Get market limits for Websea exchange (simplified)"""
        try:
            if symbol not in self._precision:
                import asyncio
                try:
                    asyncio.run(self._warm_precision(symbol))
                except Exception as e:
                    self.logger.error(f"[Websea] Failed to warm precision for {symbol}: {e}")
                    return None

            p = self._precision.get(symbol, {})
            if not p:
                return None

            # 精度
            base_precision = int(p.get("amount", "0"))
            quote_precision = int(p.get("price", "2"))

            # 步进：由精度推导
            tick_size = (
                str(Decimal(1) / (Decimal(10) ** quote_precision))
                if quote_precision > 0 else "1"
            )
            min_order_size = (
                str(Decimal(1) / (Decimal(10) ** base_precision))
                if base_precision > 0 else "1"
            )

            # 解析交易对
            if "-" in symbol:
                base_asset, quote_asset = symbol.split("-", 1)
            else:
                base_asset, quote_asset = symbol, "USDT"

            return {
                "base_asset": base_asset,
                "quote_asset": quote_asset,
                "base_precision": base_precision,   # 数量小数位
                "quote_precision": quote_precision, # 价格小数位
                "min_order_size": min_order_size,   # 用 10^-amount，更符合实盘
                "tick_size": tick_size,             # 10^-price
            }

        except Exception as e:
            self.logger.error(f"[Websea] get_market_limits error for {symbol}: {e}")
            return None

    def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """获取持仓信息，与永续合约策略兼容 - 返回列表格式"""
        try:
            if symbol is None:
                symbol = getattr(self.config, 'ticker', None) if hasattr(self.config, 'ticker') else self.config.get('ticker')
            
            res = self._get("/openApi/contract/position", {"symbol": symbol})
            if res.get("errno") != 0:
                # 返回错误时也要符合列表格式
                return []
            
            rows = res.get("result", [])
            if not isinstance(rows, list):
                rows = rows.get("positionList", []) if isinstance(rows, dict) else []
            
            # 处理多个持仓记录，计算净持仓
            positions_for_symbol = [p for p in rows if p.get("symbol") == symbol]
            
            if not positions_for_symbol:
                # 没有持仓，返回空列表（表示0持仓）
                return []
            
            # 计算净持仓量
            net_contracts = 0
            long_contracts = 0
            short_contracts = 0

            total_margin = 0
            weighted_entry_price = 0
            total_size = 0
            total_unrealized_pnl = 0
            
            for pos in positions_for_symbol:
                amount = float(pos.get("amount", "0"))
                pos_type = pos.get("type", 1)  # 1=多头, 2=空头
                
                # 根据 type 计算方向和数量
                if pos_type == 1:  # 多头
                    net_contracts += amount
                    long_contracts += amount
                elif pos_type == 2:  # 空头
                    net_contracts -= amount
                    short_contracts += amount
                
                total_size += abs(amount)
                total_margin += float(pos.get("bood", "0"))
                total_unrealized_pnl += float(pos.get("un_profit", "0"))
                
                # 计算加权平均开仓价
                entry_price = float(pos.get("open_price_avg", "0"))
                if entry_price > 0:
                    weighted_entry_price += entry_price * amount
            
            # 计算最终的加权平均价格
            if total_size > 0:
                weighted_entry_price = weighted_entry_price / total_size
            
            # 确定方向
            if net_contracts > 0:
                side = "LONG"
            elif net_contracts < 0:
                side = "SHORT"
            else:
                side = "FLAT"
            
            # 获取当前标记价格（使用第一个持仓的equity作为参考）
            mark_price = positions_for_symbol[0].get("equity", "0")
            
            net_qty = self._calculate_qty(symbol, Decimal(abs(net_contracts)))
            long_qty = self._calculate_qty(symbol, Decimal(long_contracts))
            short_qty = self._calculate_qty(symbol, Decimal(short_contracts))
            
            # 转换为 perp_market_maker.py 期望的格式
            return [{
                "symbol": symbol,
                "netQuantity": str(net_qty),
                "longQuantity": str(long_qty),
                "shortQuantity": str(short_qty),
                "side": side,
                "size": str(abs(short_qty)),
                "entry_price": str(weighted_entry_price),
                "mark_price": str(mark_price),
                "unrealized_pnl": str(total_unrealized_pnl),
                "margin": str(total_margin)
            }]
            
        except Exception as e:
            self.logger.error(f"Failed to get positions for {symbol}: {e}")
            return []

    # ---------- 实现BaseExchangeClient的标准API方法 ----------
    
    def execute_order(self, order_details: Dict[str, Any]):
        """执行订单 - 与策略兼容的接口"""
        try:
            side = order_details.get('side', '').lower()
            symbol = order_details.get('symbol')
            quantity = float(order_details.get('quantity', 0))
            order_type = order_details.get('orderType', 'limit').lower()
            time_in_force = order_details.get('timeInForce', 'GTC')
            
            if order_type == 'market':
                # 市价单 - 使用当前最佳价格
                if side == 'bid' or side == 'buy':
                    # 买单使用ask价格
                    best_bid, best_ask = asyncio.run(self._best_bid_ask(symbol))
                    price = best_ask
                    result = asyncio.run(self.place_open_order_explicit(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        side='buy'
                    ))
                else:
                    # 卖单使用bid价格
                    best_bid, best_ask = asyncio.run(self._best_bid_ask(symbol))
                    price = best_bid
                    result = asyncio.run(self.place_open_order_explicit(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        side='sell'
                    ))
            else:
                # 限价单
                price = float(order_details.get('price', 0))
                if price <= 0:
                    return {"success": False, "error": "Invalid price for limit order"}
                
                order_side = 'buy' if side in ('bid', 'buy') else 'sell'
                result = asyncio.run(self.place_open_order_explicit(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    side=order_side
                ))
            
            if result.success:
                return {
                    "success": True,
                    "data": {
                        "orderId": result.order_id,
                        "status": "NEW",
                        "symbol": symbol,
                        "side": result.side,
                        "quantity": str(result.size),
                        "price": str(result.price)
                    }
                }
            else:
                return {"success": False, "error": result.error_message}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_open_orders(self, symbol: Optional[str] = None):
        """获取开放订单 - 与策略兼容的格式"""
        try:
            if symbol is None:
                symbol = getattr(self.config, 'ticker', None) if hasattr(self.config, 'ticker') else self.config.get('ticker')
            
            orders = asyncio.run(self.get_active_orders(symbol))
            
            # 转换为策略期望的格式 - 直接返回订单列表
            result_orders = []
            for order in orders:
                result_orders.append({
                    "id": order.order_id,  # 注意使用 'id' 而不是 'orderId'
                    "orderId": order.order_id,
                    "symbol": symbol,
                    "side": order.side.upper(),
                    "quantity": str(order.size),
                    "price": str(order.price),
                    "status": order.status,
                    "type": "LIMIT",
                    "timeInForce": "GTC"
                })
            
            return result_orders  # 直接返回列表，不包装在字典中
            
        except Exception as e:
            return {"error": str(e)}  # 出错时返回错误字典

    def cancel_all_orders(self, symbol: str):
        """取消所有订单"""
        try:
            # 先获取活跃订单
            orders = asyncio.run(self.get_active_orders(symbol))
            
            cancelled_count = 0
            for order in orders:
                result = asyncio.run(self.cancel_order(order.order_id))
                if result.success:
                    cancelled_count += 1
            
            return {
                "success": True,
                "data": {
                    "cancelled_count": cancelled_count,
                    "message": f"Cancelled {cancelled_count} orders"
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_ticker(self, symbol: str):
        """获取行情信息 - 与策略兼容的格式"""
        try:
            # 通过深度数据获取最佳买卖价
            best_bid, best_ask = asyncio.run(self._best_bid_ask(symbol))
            last_price = (best_bid + best_ask) / 2  # 使用中间价作为最新价
            
            # 返回策略期望的格式 - 直接返回数据字典
            return {
                "symbol": symbol,
                "lastPrice": str(last_price),  # 使用 lastPrice 而不是 price
                "bidPrice": str(best_bid),
                "askPrice": str(best_ask),
                "volume": "0",  # Websea可能需要额外API获取
                "changePercent": "0"
            }
            
        except Exception as e:
            return {"error": str(e)}

    def get_order_book(self, symbol: str, limit: int = 20):
        """获取订单簿 - 与策略兼容的格式"""
        try:
            res = self._get("/openApi/contract/depth", {"symbol": symbol, "limit": limit})
            if res.get("errno") != 0:
                return {"error": res.get("errmsg", "Failed to get order book")}
            
            data = res.get("result", {})
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            # 直接返回策略期望的格式
            return {
                "symbol": symbol,
                "bids": [[str(bid[0]), str(bid[1])] for bid in bids],
                "asks": [[str(ask[0]), str(ask[1])] for ask in asks]
            }
            
        except Exception as e:
            return {"error": str(e)}

    def get_balance(self):
        """获取账户余额 - 简化实现"""
        try:
            # Websea 可能需要特定的余额查询API
            # 这里返回一个基本的空余额结构
            return {
                "success": True,
                "data": {
                    "USDT": {"available": "0", "locked": "0"},
                    "SOL": {"available": "0", "locked": "0"}
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_collateral(self, subaccount_id: Optional[str] = None):
        """获取抵押品余额 - 简化实现"""
        return {"success": True, "data": {"assets": []}}
    
    # ---------- convenience (optional explicit price API) ----------

    async def place_open_order_explicit(self, symbol: str, quantity: float, price: Decimal, side: str,
                                        lever_rate: int = 20, is_full: int = 2) -> OrderResult:
        """
        你也可以像早前脚本那样传入明确价格与方向。
        """
        try:
            if symbol not in self._precision:
                await self._warm_precision(symbol)
            # quantity 为 SOL 个数（标的个数）
            contract_quantity = self._calculate_contract(symbol, Decimal(quantity))
            px = self._round_price(symbol, Decimal(price))
            if contract_quantity <= 0:
                return OrderResult(success=False, error_message=f"{quantity} 向下取整后小于最小下单数量，({contract_quantity} <= 0)")
            params = {
                "contract_type": "open",
                "type": f"{side.lower()}-limit",
                "symbol": symbol,
                "amount": str(contract_quantity),
                "price": str(px),
                "lever_rate": str(lever_rate),
                "is_full": str(is_full),
            }
            res = self._post("/openApi/contract/add", params)
            if res.get("errno") == 0:
                return OrderResult(success=True, order_id=res["result"]["order_id"], side=side, size=contract_quantity, price=px)
            else:
                return OrderResult(success=False, error_message=res.get("errmsg", "place order failed"))
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))
