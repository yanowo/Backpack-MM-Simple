"""
交互式做市策略命令面板
"""
import time
import threading
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.prompt import Prompt, Confirm
from rich import box
from rich.console import Group

# 導入配置模塊
try:
    from config import API_KEY, SECRET_KEY
except ImportError:
    API_KEY = os.getenv('API_KEY')
    SECRET_KEY = os.getenv('SECRET_KEY')

# 導入設定模塊
try:
    from panel.settings import get_setting, set_setting, update_settings, load_settings
except ImportError:
    # 在直接運行面板文件時，可能會遇到導入問題，嘗試直接導入
    try:
        from settings import get_setting, set_setting, update_settings, load_settings
    except ImportError:
        # 如果無法導入，創建空的設定函數
        def get_setting(key, default=None): return default
        def set_setting(key, value): pass
        def update_settings(settings_dict): pass
        def load_settings(): return {}

class InteractivePanel:
    def __init__(self):
        """初始化交互面板"""
        # 初始化默認設定
        self.settings = load_settings()
        
        # 策略參數
        self.strategy_params = {
            'base_spread_percentage': self.settings.get('base_spread_percentage', 0.1),
            'order_quantity': self.settings.get('order_quantity', None),
            'max_orders': self.settings.get('max_orders', 3),
            'duration': self.settings.get('duration', 24*3600),
            'interval': self.settings.get('interval', 60),
        }
        
        # 策略狀態
        self.strategy_running = False
        self._initializing_strategy = False
        self.current_symbol = None
        self.market_maker = None
        self.strategy_thread = None
        self.last_market_update = datetime.now()
        
        # 市場數據
        self.market_data = {
            'bp_prices': {},    # 基準價格
            'bid_prices': {},   # 買價
            'ask_prices': {},   # 賣價
            'spread_pct': {},   # 價差百分比
            'buy_orders': {},   # 買單數量
            'sell_orders': {},  # 賣單數量
            'positions': {},    # 持倉狀態 (多/空/平)
        }
        
        # 策略數據
        self.strategy_data = {
            'base_spread': 0.0,      # 基礎價差
            'total_bought': 0.0,     # 總購買量
            'total_sold': 0.0,       # 總賣出量
            'maker_buy_volume': 0.0, # Maker買入量
            'maker_sell_volume': 0.0,# Maker賣出量
            'taker_buy_volume': 0.0, # Taker買入量
            'taker_sell_volume': 0.0,# Taker賣出量
            'session_profit': 0.0,   # 本次利潤
            'total_profit': 0.0,     # 總利潤
            'orders_placed': 0,      # 訂單數量
            'trades_executed': 0,    # 成交數量
        }
        
        # API 密鑰 (從環境變數或設定讀取)
        self.api_key = os.environ.get('API_KEY', '')
        self.secret_key = os.environ.get('SECRET_KEY', '')
        
        self.console = Console()
        self.layout = self.create_layout()
        self.live = None
        self.running = False
        self.update_thread = None
        
        # 命令和狀態
        self.command_handlers = {}
        self.command_mode = False  # 切換命令模式
        self.current_command = ""
        self.command_history = []
        self.max_command_history = 20
        
        # 系統日誌
        self.logs = []
        self.max_logs = 15  # 最多顯示日誌條數
        
        # 註冊命令處理函數
        self.register_commands()
    
    def add_log(self, message, level="INFO"):
        """添加日誌"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append((timestamp, level, message))
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
    
    def register_commands(self):
        """註冊所有命令和處理函數"""
        self.command_handlers = {
            'help': self.cmd_help,
            'clear': self.cmd_clear,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'symbols': self.cmd_list_symbols,
            'start': self.cmd_start_strategy,
            'stop': self.cmd_stop_strategy,
            'params': self.cmd_show_params,
            'set': self.cmd_set_param,
            'status': self.cmd_show_status,
            'balance': self.cmd_show_balance,
            'orders': self.cmd_show_orders,
            'cancel': self.cmd_cancel_orders,
            'diagnose': self.cmd_diagnose,
        }
    
    def create_layout(self):
        """創建UI布局"""
        layout = Layout()
        
        # 分成上中下三部分
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="command", size=3)
        )
        
        # 主區域分成左右兩部分
        layout["main"].split_row(
            Layout(name="market_data", ratio=3),
            Layout(name="logs", ratio=2)
        )
        
        return layout
    
    def generate_header(self):
        """生成頭部面板"""
        status = "閒置中"
        if self.strategy_running:
            status = f"運行中 - {self.current_symbol}" if self.current_symbol else "運行中"
        
        title = f"做市交易機器人 - [{status}]"
        return Panel(
            Align.center(title, vertical="middle"),
            style="bold white on blue"
        )
    
    def generate_market_table(self):
        """生成市場數據表格"""
        # 創建表格
        last_update_str = self.last_market_update.strftime("%H:%M:%S")
        table = Table(title=f"市場數據 (更新: {last_update_str})", show_header=True, header_style="bold white on dark_blue", box=box.SIMPLE)
        
        
        # 添加列
        table.add_column("幣種", style="cyan")
        table.add_column("BP價格", justify="right", style="green")
        table.add_column("買價", justify="right", style="green")
        table.add_column("賣價", justify="right", style="green")
        table.add_column("價差%", justify="right", style="magenta")
        table.add_column("買單數", justify="right", style="blue")
        table.add_column("賣單數", justify="right", style="red")
        table.add_column("持倉", justify="center", style="yellow")
        
        # 添加活躍交易對的數據
        if self.current_symbol:
            symbol = self.current_symbol
            bp_price = self.market_data['bp_prices'].get(symbol, "-")
            bid_price = self.market_data['bid_prices'].get(symbol, "-")
            ask_price = self.market_data['ask_prices'].get(symbol, "-")
            spread_pct = self.market_data['spread_pct'].get(symbol, "-")
            buy_orders = self.market_data['buy_orders'].get(symbol, 0)
            sell_orders = self.market_data['sell_orders'].get(symbol, 0)
            position = self.market_data['positions'].get(symbol, "-")
            
            table.add_row(
                symbol,
                f"{bp_price}" if bp_price != "-" else "-",
                f"{bid_price}" if bid_price != "-" else "-",
                f"{ask_price}" if ask_price != "-" else "-",
                f"{spread_pct}" if spread_pct != "-" else "-",
                str(buy_orders),
                str(sell_orders),
                position
            )
        
        # 添加策略數據
        strategy_table = Table(title="策略數據", show_header=True, header_style="bold white on dark_blue", box=box.SIMPLE)
        strategy_table.add_column("參數", style="yellow")
        strategy_table.add_column("數值", style="cyan", justify="right")
        
        # 添加重要的策略參數
        strategy_table.add_row("基礎價差", f"{self.strategy_data['base_spread']:.4f}%")
        
        # 顯示訂單數量
        order_quantity = self.strategy_params.get('order_quantity')
        if order_quantity is not None:
            strategy_table.add_row("訂單數量", f"{order_quantity}")
        else:
            strategy_table.add_row("訂單數量", "自動")
        
        # 顯示利潤表
        profit_table = Table(title="利潤統計", show_header=True, header_style="bold white on dark_blue", box=box.SIMPLE)
        profit_table.add_column("指標", style="yellow")
        profit_table.add_column("數值", style="cyan", justify="right")
        
        total_profit = self.strategy_data['total_profit']
        session_profit = self.strategy_data['session_profit']
        
        profit_style = "green" if total_profit >= 0 else "red"
        session_style = "green" if session_profit >= 0 else "red"
        
        profit_table.add_row("總利潤", f"{total_profit:.6f}")
        profit_table.add_row("本次利潤", Text(f"{session_profit:.6f}", style=session_style))
        
        # 添加倉位表
        position_table = Table(title="倉位統計", show_header=True, header_style="bold white on dark_blue", box=box.SIMPLE)
        position_table.add_column("指標", style="yellow")
        position_table.add_column("數值", style="cyan", justify="right")
        
        total_bought = self.strategy_data['total_bought']
        total_sold = self.strategy_data['total_sold']
        imbalance = total_bought - total_sold
        imbalance_pct = abs(imbalance) / max(total_bought, total_sold) * 100 if max(total_bought, total_sold) > 0 else 0
        
        position_table.add_row("買入總量", f"{total_bought:.6f}")
        position_table.add_row("賣出總量", f"{total_sold:.6f}")
        position_table.add_row("淨倉位", f"{imbalance:.6f}")
        position_table.add_row("不平衡%", f"{imbalance_pct:.2f}%")
        position_table.add_row("Maker買入", f"{self.strategy_data['maker_buy_volume']:.6f}")
        position_table.add_row("Maker賣出", f"{self.strategy_data['maker_sell_volume']:.6f}")
        
        return Group(table, strategy_table, profit_table, position_table)
    
    def generate_log_panel(self):
        """生成日誌面板"""
        log_text = ""
        
        for timestamp, level, message in self.logs:
            if level == "ERROR":
                log_text += f"[bold red][{timestamp}] {message}[/bold red]\n"
            elif level == "WARNING":
                log_text += f"[yellow][{timestamp}] {message}[/yellow]\n"
            elif level == "COMMAND":
                log_text += f"[bold cyan][{timestamp}] {message}[/bold cyan]\n"
            elif level == "SYSTEM":
                log_text += f"[bold magenta][{timestamp}] {message}[/bold magenta]\n"
            else:
                log_text += f"[{timestamp}] {message}\n"
        
        return Panel(
            log_text,
            title="系統日誌",
            border_style="bright_blue"
        )
    
    def generate_command_panel(self):
        """生成命令面板"""
        if self.command_mode:
            command_text = f"> {self.current_command}"
        else:
            command_text = "按 : 或 / 進入命令模式  |  幫助命令: help"
            
        return Panel(
            Text(command_text, style="bold cyan"),
            title="命令",
            border_style="green"
        )
    
    def update_display(self):
        """更新顯示內容"""
        self.layout["header"].update(self.generate_header())
        self.layout["market_data"].update(self.generate_market_table())
        self.layout["logs"].update(self.generate_log_panel())
        self.layout["command"].update(self.generate_command_panel())
    
    def update_thread_function(self):
        """更新線程函數"""
        while self.running:
            self.update_display()
            time.sleep(0.5)  # 每0.5秒更新一次，避免過高的CPU使用率
    
    def handle_input(self, key):
        """處理鍵盤輸入"""
        if self.command_mode:
            # 命令模式下的按鍵處理
            if key == "enter":
                self.execute_command(self.current_command)
                self.command_mode = False
                self.current_command = ""
            elif key == "escape":
                self.command_mode = False
                self.current_command = ""
            elif key == "backspace":
                self.current_command = self.current_command[:-1]
            else:
                # 添加普通字符到命令
                self.current_command += key
        else:
            # 非命令模式下的按鍵處理
            if key == ":" or key == "/":
                self.command_mode = True
                self.current_command = ""
            elif key == "q":
                self.running = False
    
    def execute_command(self, command):
        """執行命令"""
        # 添加到命令歷史
        if command.strip():
            self.command_history.append(command)
            if len(self.command_history) > self.max_command_history:
                self.command_history = self.command_history[-self.max_command_history:]
        
        # 解析命令
        parts = command.strip().split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # 執行對應的命令處理函數
        if cmd in self.command_handlers:
            self.add_log(f"執行命令: {command}", "COMMAND")
            try:
                self.command_handlers[cmd](args)
            except Exception as e:
                self.add_log(f"執行命令出錯: {str(e)}", "ERROR")
        else:
            self.add_log(f"未知命令: {cmd}", "ERROR")
    
    def cmd_help(self, args):
        """顯示幫助信息"""
        self.add_log("可用命令:", "SYSTEM")
        self.add_log("help - 顯示幫助", "SYSTEM")
        self.add_log("symbols - 列出可用交易對", "SYSTEM")
        self.add_log("start <symbol> - 啟動指定交易對的做市策略", "SYSTEM")
        self.add_log("stop - 停止當前做市策略", "SYSTEM")
        self.add_log("params - 顯示當前策略參數", "SYSTEM")
        self.add_log("set <參數> <值> - 設置策略參數", "SYSTEM")
        self.add_log("status - 顯示當前狀態", "SYSTEM")
        self.add_log("balance - 查詢餘額", "SYSTEM")
        self.add_log("orders - 顯示活躍訂單", "SYSTEM")
        self.add_log("cancel - 取消所有訂單", "SYSTEM")
        self.add_log("clear - 清除日誌", "SYSTEM")
        self.add_log("diagnose - 執行系統診斷檢查", "SYSTEM")
        self.add_log("exit/quit - 退出程序", "SYSTEM")
    
    def cmd_clear(self, args):
        """清除日誌"""
        self.logs = []
        self.add_log("日誌已清除", "SYSTEM")
    
    def cmd_exit(self, args):
        """退出程序"""
        self.running = False
    
    def cmd_list_symbols(self, args):
        """列出可用交易對"""
        self.add_log("正在獲取可用交易對...", "SYSTEM")
        
        try:
            # 導入需要的模塊
            from api.client import get_markets
            
            markets_info = get_markets()
            if isinstance(markets_info, dict) and "error" in markets_info:
                self.add_log(f"獲取市場信息失敗: {markets_info['error']}", "ERROR")
                return
            
            spot_markets = [m for m in markets_info if m.get('marketType') == 'SPOT']
            self.market_data['symbols'] = [m.get('symbol') for m in spot_markets]
            
            self.add_log(f"找到 {len(spot_markets)} 個現貨市場:", "SYSTEM")
            
            # 分組顯示，每行最多5個
            symbols_per_line = 5
            for i in range(0, len(spot_markets), symbols_per_line):
                group = spot_markets[i:i+symbols_per_line]
                symbols_line = ", ".join([m.get('symbol') for m in group])
                self.add_log(symbols_line, "SYSTEM")
                
        except Exception as e:
            self.add_log(f"獲取交易對時出錯: {str(e)}", "ERROR")
    
    def cmd_start_strategy(self, args):
        """啟動做市策略"""
        if not args:
            self.add_log("請指定交易對，例如: start SOL_USDC", "ERROR")
            return
        
        symbol = args[0]
        
        if self.strategy_running:
            self.add_log("已有策略運行中，請先停止當前策略", "ERROR")
            return
        
        self.add_log(f"正在啟動 {symbol} 的做市策略...", "SYSTEM")
        
        try:
            # 導入必要的類
            from database.db import Database
            from strategies.market_maker import MarketMaker
            
            # 導入或獲取API密鑰
            try:
                from config import API_KEY as CONFIG_API_KEY, SECRET_KEY as CONFIG_SECRET_KEY
            except ImportError:
                CONFIG_API_KEY = os.getenv('API_KEY')
                CONFIG_SECRET_KEY = os.getenv('SECRET_KEY')
            
            api_key = CONFIG_API_KEY
            secret_key = CONFIG_SECRET_KEY
            
            if not api_key or not secret_key:
                self.add_log("缺少API密鑰，請檢查config.py或環境變量", "ERROR")
                return
                
            # 默認策略參數
            params = {
                'base_spread_percentage': 0.1,  # 默認價差0.1%
                'order_quantity': None,  # 添加訂單數量
                'max_orders': 3,               # 每側3個訂單
                'execution_mode': 'standard',   # 標準執行模式
                'risk_factor': 0.5,            # 默認風險因子
                'duration': 24*3600,           # 運行24小時
                'interval': 60                 # 每分鐘更新一次
            }
            
            # 合併用戶設置的參數
            for key, value in self.strategy_params.items():
                if key in params:
                    params[key] = value
            
            # 初始化數據庫
            db = Database()
            
            # 設置當前交易對和標記策略為運行狀態
            self.current_symbol = symbol
            
            # 記錄當前正在初始化
            self._initializing_strategy = True
            
            # 更新策略數據
            self.strategy_data['base_spread'] = params['base_spread_percentage']
            
            # 初始化做市商
            self.market_maker = MarketMaker(
                api_key=api_key,
                secret_key=secret_key,
                symbol=symbol,
                db_instance=db,
                base_spread_percentage=params['base_spread_percentage'],
                order_quantity=params['order_quantity'],  # 使用設定的訂單數量
                max_orders=params['max_orders']
            )
            
            # 標記策略為運行狀態
            self.strategy_running = True
            self._initializing_strategy = False
            
            # 啟動策略在單獨的線程中
            self.strategy_thread = threading.Thread(
                target=self._run_strategy_thread,
                args=(params['duration'], params['interval']),
                daemon=True
            )
            self.strategy_thread.start()
            
            self.add_log(f"{symbol} 做市策略已啟動", "SYSTEM")
            
        except Exception as e:
            self._initializing_strategy = False
            self.strategy_running = False  # 確保在出錯時重置狀態
            self.current_symbol = None
            self.add_log(f"啟動策略時出錯: {str(e)}", "ERROR")
            import traceback
            self.add_log(f"詳細錯誤: {traceback.format_exc()}", "ERROR")
    
    def _run_strategy_thread(self, duration_seconds, interval_seconds):
        """在單獨線程中運行策略"""
        if not self.market_maker:
            self.add_log("做市商未初始化", "ERROR")
            self.strategy_running = False  # 確保重置狀態
            return
        
        try:
            start_time = time.time()
            iteration = 0
            
            # 記錄開始信息
            self.add_log(f"開始運行做市策略: {self.market_maker.symbol}")
            
            # 確保WebSocket連接
            try:
                # 首先檢查WebSocket連接
                self.add_log("檢查WebSocket連接...")
                connection_status = self.market_maker.check_ws_connection()
                if not connection_status:
                    self.add_log("WebSocket未連接，嘗試建立連接...")
                    # 在MarketMaker中應該有初始化WebSocket的方法
                    if hasattr(self.market_maker, 'initialize_websocket'):
                        self.market_maker.initialize_websocket()
                    elif hasattr(self.market_maker, 'reconnect_websocket'):
                        self.market_maker.reconnect_websocket()
                
                # 再次檢查連接狀態
                connection_status = self.market_maker.check_ws_connection()
                if connection_status:
                    self.add_log("WebSocket連接成功")
                    
                    # 等待WebSocket就緒
                    self.add_log("等待WebSocket就緒...")
                    time.sleep(2)
                    
                    # 初始化訂單簿
                    if hasattr(self.market_maker, 'ws') and self.market_maker.ws:
                        if not self.market_maker.ws.orderbook.get("bids") and not self.market_maker.ws.orderbook.get("asks"):
                            self.add_log("初始化訂單簿...")
                            if hasattr(self.market_maker.ws, 'initialize_orderbook'):
                                self.market_maker.ws.initialize_orderbook()
                                # 等待訂單簿填充
                                time.sleep(1)
                    
                    # 確保所有數據流訂閱
                    self.add_log("確保數據流訂閱...")
                    if hasattr(self.market_maker, '_ensure_data_streams'):
                        self.market_maker._ensure_data_streams()
                    
                    # 增加小延遲確保訂閱成功
                    time.sleep(2)
                    self.add_log("數據流訂閱完成，進入主循環...")
                else:
                    self.add_log("WebSocket連接失敗，請檢查網絡或API配置", "ERROR")
                    self.strategy_running = False
                    return
            except Exception as ws_error:
                self.add_log(f"WebSocket設置出錯: {str(ws_error)}", "ERROR")
                import traceback
                self.add_log(f"WebSocket錯誤詳情: {traceback.format_exc()}", "ERROR")
                self.strategy_running = False
                return
                
            # 主循環前檢查策略運行狀態
            if not self.strategy_running:
                self.add_log("策略在初始化後被停止", "WARNING")
                return
            
            # 檢查訂單簿是否已填充
            if hasattr(self.market_maker, 'ws') and self.market_maker.ws and hasattr(self.market_maker.ws, 'orderbook'):
                if not self.market_maker.ws.orderbook.get("bids") or not self.market_maker.ws.orderbook.get("asks"):
                    self.add_log("警告: 訂單簿可能未完全初始化", "WARNING")
                
            # 主循環
            self.add_log("開始執行策略主循環...")
            while time.time() - start_time < duration_seconds and self.strategy_running:
                iteration += 1
                
                self.add_log(f"第 {iteration} 次迭代")
                
                try:
                    # 檢查連接
                    connection_status = self.market_maker.check_ws_connection()
                    if not connection_status:
                        self.add_log("WebSocket連接已斷開，嘗試重新連接...", "WARNING")
                        reconnected = self.market_maker.reconnect_websocket()
                        if not reconnected:
                            self.add_log("重新連接失敗，停止策略", "ERROR")
                            break
                        # 給連接一些時間重新建立
                        time.sleep(2)
                        continue  # 跳過這次迭代
                    
                    # 更新面板數據
                    self._update_strategy_data()
                    
                    # 檢查訂單成交情況
                    self.add_log("檢查訂單成交情況...")
                    self.market_maker.check_order_fills()
                    
                    # 檢查是否需要重平衡倉位
                    needs_rebalance = self.market_maker.need_rebalance()
                    if needs_rebalance:
                        self.add_log("執行倉位重平衡")
                        self.market_maker.rebalance_position()
                    
                    # 下限價單
                    self.add_log("下限價單...")
                    self.market_maker.place_limit_orders()
                    
                    # 估算利潤
                    self.market_maker.estimate_profit()
                    
                except Exception as loop_error:
                    self.add_log(f"策略迭代中出錯: {str(loop_error)}", "ERROR")
                    import traceback
                    self.add_log(f"迭代錯誤詳情: {traceback.format_exc()}", "ERROR")
                    # 不因為單次循環錯誤停止整個策略，繼續下一次循環
                    time.sleep(5)  # 出錯時等待更長時間
                    continue
                
                # 等待下一次迭代
                time.sleep(interval_seconds)
            
            # 結束時記錄信息
            if not self.strategy_running:
                self.add_log("策略已手動停止")
            else:
                self.add_log("策略運行完成")
            
            # 清理資源
            self._cleanup_strategy()
            
        except Exception as e:
            self.add_log(f"策略運行出錯: {str(e)}", "ERROR")
            import traceback
            self.add_log(f"錯誤詳情: {traceback.format_exc()}", "ERROR")
            
            # 確保清理資源
            self._cleanup_strategy()
    
    def _update_strategy_data(self):
        """從市場做市商更新數據到面板"""
        if not self.market_maker:
            return
            
        try:
            # 更新最後一次市場數據更新時間
            self.last_market_update = datetime.now()
            
            # 檢查 WebSocket 連接
            if not hasattr(self.market_maker, 'ws') or not self.market_maker.ws:
                self.add_log("WebSocket 連接不可用", "WARNING")
                return
            
            # 更新市場數據
            symbol = self.current_symbol
            if symbol:
                try:
                    # 更新價格數據
                    bid_price = getattr(self.market_maker.ws, 'bid_price', None)
                    ask_price = getattr(self.market_maker.ws, 'ask_price', None)
                    
                    if bid_price and ask_price:
                        bp_price = (bid_price + ask_price) / 2
                        self.market_data['bp_prices'][symbol] = bp_price
                        self.market_data['bid_prices'][symbol] = bid_price
                        self.market_data['ask_prices'][symbol] = ask_price
                        
                        # 計算價差
                        spread_pct = (ask_price - bid_price) / bp_price * 100
                        self.market_data['spread_pct'][symbol] = f"{spread_pct:.6f}%"
                except Exception as price_err:
                    self.add_log(f"更新價格數據時出錯: {str(price_err)}", "WARNING")
                
                try:
                    # 更新訂單數量
                    buy_orders = getattr(self.market_maker, 'active_buy_orders', [])
                    sell_orders = getattr(self.market_maker, 'active_sell_orders', [])
                    self.market_data['buy_orders'][symbol] = len(buy_orders)
                    self.market_data['sell_orders'][symbol] = len(sell_orders)
                except Exception as order_err:
                    self.add_log(f"更新訂單數量時出錯: {str(order_err)}", "WARNING")
                
                try:
                    # 更新持倉狀態
                    total_bought = getattr(self.market_maker, 'total_bought', 0)
                    total_sold = getattr(self.market_maker, 'total_sold', 0)
                    
                    if total_bought > total_sold:
                        self.market_data['positions'][symbol] = "多"
                    elif total_bought < total_sold:
                        self.market_data['positions'][symbol] = "空"
                    else:
                        self.market_data['positions'][symbol] = "平"
                except Exception as pos_err:
                    self.add_log(f"更新持倉狀態時出錯: {str(pos_err)}", "WARNING")
            
            # 更新交易量數據
            try:
                self.strategy_data['total_bought'] = getattr(self.market_maker, 'total_bought', 0)
                self.strategy_data['total_sold'] = getattr(self.market_maker, 'total_sold', 0)
                self.strategy_data['maker_buy_volume'] = getattr(self.market_maker, 'maker_buy_volume', 0)
                self.strategy_data['maker_sell_volume'] = getattr(self.market_maker, 'maker_sell_volume', 0)
                self.strategy_data['taker_buy_volume'] = getattr(self.market_maker, 'taker_buy_volume', 0)
                self.strategy_data['taker_sell_volume'] = getattr(self.market_maker, 'taker_sell_volume', 0)
                self.strategy_data['orders_placed'] = getattr(self.market_maker, 'orders_placed', 0)
                self.strategy_data['trades_executed'] = getattr(self.market_maker, 'trades_executed', 0)
                
                # 利潤統計
                self.strategy_data['session_profit'] = getattr(self.market_maker, 'session_profit', 0.0)
                self.strategy_data['total_profit'] = getattr(self.market_maker, 'total_profit', 0.0)
            except Exception as vol_err:
                self.add_log(f"更新交易量數據時出錯: {str(vol_err)}", "WARNING")
                
        except Exception as e:
            self.add_log(f"更新面板數據時出錯: {str(e)}", "ERROR")
    
    def _cleanup_strategy(self):
        """清理策略資源"""
        if not self.market_maker:
            return
            
        # 標記清理開始
        was_running = self.strategy_running
        # 標記策略為停止狀態，以防止任何進一步的操作
        self.strategy_running = False
        
        try:
            # 記錄清理消息
            if was_running:
                self.add_log("正在清理策略資源...", "SYSTEM")
                
            # 取消所有活躍訂單
            self.add_log("取消所有未成交訂單...")
            try:
                if hasattr(self.market_maker, 'cancel_existing_orders'):
                    self.market_maker.cancel_existing_orders()
                    self.add_log("所有訂單已取消")
                else:
                    self.add_log("無法取消訂單: 方法不可用", "WARNING")
            except Exception as cancel_err:
                self.add_log(f"取消訂單時出錯: {str(cancel_err)}", "ERROR")
            
            # 關閉WebSocket連接
            try:
                if hasattr(self.market_maker, 'ws') and self.market_maker.ws:
                    self.add_log("關閉WebSocket連接...")
                    self.market_maker.ws.close()
                    self.add_log("WebSocket連接已關閉")
            except Exception as ws_err:
                self.add_log(f"關閉WebSocket時出錯: {str(ws_err)}", "ERROR")
            
            # 關閉數據庫連接
            try:
                if hasattr(self.market_maker, 'db') and self.market_maker.db:
                    self.add_log("關閉數據庫連接...")
                    self.market_maker.db.close()
                    self.add_log("數據庫連接已關閉")
            except Exception as db_err:
                self.add_log(f"關閉數據庫時出錯: {str(db_err)}", "ERROR")
            
            # 確認清理完成
            if was_running:
                self.add_log("策略資源清理完成", "SYSTEM")
                
        except Exception as e:
            self.add_log(f"清理資源時遇到未知錯誤: {str(e)}", "ERROR")
            import traceback
            self.add_log(f"錯誤詳情: {traceback.format_exc()}", "ERROR")
        finally:
            # 清空策略實例
            self.current_symbol = None
            self.market_maker = None
    
    def cmd_stop_strategy(self, args):
        """停止當前運行的策略"""
        if not self.strategy_running and not self._initializing_strategy:
            self.add_log("沒有正在運行的策略", "ERROR")
            return
            
        if self._initializing_strategy:
            self.add_log("策略正在初始化中，請稍後再試", "WARNING")
            return
        
        self.add_log("正在停止策略...")
        self.strategy_running = False
        
        # 等待策略線程結束
        if hasattr(self, 'strategy_thread') and self.strategy_thread and self.strategy_thread.is_alive():
            try:
                self.strategy_thread.join(timeout=3)
                if self.strategy_thread.is_alive():
                    self.add_log("策略線程未能在3秒內結束，可能需要手動重啟程序", "WARNING")
            except Exception as join_err:
                self.add_log(f"等待策略線程時出錯: {str(join_err)}", "ERROR")
            
        self.add_log("策略已停止")
    
    def cmd_show_params(self, args):
        """顯示當前策略參數"""
        self.add_log("當前策略參數:", "SYSTEM")
        
        if not self.strategy_params:
            self.add_log("尚未設置任何參數，使用默認值", "SYSTEM")
            self.add_log("可用參數:", "SYSTEM")
            self.add_log("base_spread - 基礎價差百分比", "SYSTEM")
            self.add_log("order_quantity - 訂單數量 (例如: 0.5 SOL)", "SYSTEM")
            self.add_log("max_orders - 每側最大訂單數", "SYSTEM")
            self.add_log("duration - 運行時間(秒)", "SYSTEM")
            self.add_log("interval - 更新間隔(秒)", "SYSTEM")
            return
        
        for param, value in self.strategy_params.items():
            # 訂單數量可能為空，特殊處理
            if param == 'order_quantity' and value is None:
                self.add_log(f"{param} = 自動 (根據餘額決定)", "SYSTEM")
            else:
                self.add_log(f"{param} = {value}", "SYSTEM")
                
        # 添加使用說明
        self.add_log("\n設置參數示例:", "SYSTEM")
        self.add_log("set base_spread 0.2    - 設置價差為0.2%", "SYSTEM")
        self.add_log("set order_quantity 0.5 - 設置訂單數量為0.5", "SYSTEM")
        self.add_log("set max_orders 5       - 設置每側最大訂單數為5", "SYSTEM")
    
    def cmd_set_param(self, args):
        """設置策略參數"""
        if len(args) < 2:
            self.add_log("用法: set <參數名> <參數值>", "ERROR")
            return
        
        param = args[0]
        value = args[1]
        
        valid_params = {
            'base_spread_percentage': float,
            'order_quantity': float,
            'max_orders': int,
            'duration': int,
            'interval': int
        }
        
        if param not in valid_params:
            self.add_log(f"無效的參數名: {param}", "ERROR")
            self.add_log("有效參數: " + ", ".join(valid_params.keys()), "SYSTEM")
            return
        
        # 轉換參數值
        try:
            # 處理特殊值：auto, none, null
            if value.lower() in ('auto', 'none', 'null', 'auto'):
                typed_value = None
                self.add_log(f"訂單數量將設為自動 (由程序根據餘額決定)", "SYSTEM")
            else:
                # 數值處理
                typed_value = float(value)
                if typed_value <= 0:
                    raise ValueError("訂單數量必須大於0")
            
            # 存儲參數
            self.strategy_params[param] = typed_value
            
            # 保存到設定文件
            try:
                set_setting(param, typed_value)
                self.add_log(f"參數已設置並保存: {param} = {typed_value}", "SYSTEM")
            except Exception as e:
                self.add_log(f"參數已設置但保存失敗: {str(e)}", "WARNING")
            
        except ValueError as e:
            self.add_log(f"參數值轉換錯誤: {str(e)}", "ERROR")
            self.add_log(f"參數 {param} 需要 {valid_params[param].__name__} 類型", "SYSTEM")
    
    def cmd_show_status(self, args):
        """顯示當前狀態"""
        if not self.strategy_running:
            self.add_log("沒有正在運行的策略", "SYSTEM")
            return
        
        self.add_log(f"正在運行 {self.current_symbol} 的做市策略", "SYSTEM")
        
        # 顯示策略參數
        self.add_log("策略參數:", "SYSTEM")
        self.add_log(f"基礎價差: {self.strategy_data['base_spread']:.4f}%", "SYSTEM")
        
        # 顯示訂單數量
        order_quantity = self.strategy_params.get('order_quantity')
        if order_quantity is not None:
            self.add_log(f"訂單數量: {order_quantity}", "SYSTEM")
        else:
            self.add_log("訂單數量: 自動", "SYSTEM")
            
        self.add_log(f"最大訂單數: {self.strategy_params.get('max_orders', 3)}", "SYSTEM")
        
        # 顯示重要狀態指標
        self.add_log("\n倉位統計:", "SYSTEM")
        total_bought = self.strategy_data['total_bought']
        total_sold = self.strategy_data['total_sold']
        imbalance = total_bought - total_sold
        imbalance_pct = abs(imbalance) / max(total_bought, total_sold) * 100 if max(total_bought, total_sold) > 0 else 0
        
        self.add_log(f"總買入: {total_bought} - 總賣出: {total_sold}", "SYSTEM")
        self.add_log(f"倉位不平衡度: {imbalance_pct:.2f}%", "SYSTEM")
        
        # 顯示利潤信息
        self.add_log("\n利潤統計:", "SYSTEM")
        total_profit = self.strategy_data['total_profit']
        
        self.add_log(f"總利潤: {total_profit:.6f}", "SYSTEM")
    
    def cmd_show_balance(self, args):
        """顯示當前餘額"""
        self.add_log("正在查詢餘額...", "SYSTEM")
        
        try:
            # 導入API客戶端
            from api.client import get_balance
            
            balances = get_balance(API_KEY, SECRET_KEY)
            if isinstance(balances, dict) and "error" in balances and balances["error"]:
                self.add_log(f"獲取餘額失敗: {balances['error']}", "ERROR")
                return
            
            self.add_log("當前餘額:", "SYSTEM")
            if isinstance(balances, dict):
                for coin, details in balances.items():
                    available = float(details.get('available', 0))
                    locked = float(details.get('locked', 0))
                    if available > 0 or locked > 0:
                        self.add_log(f"{coin}: 可用 {available}, 凍結 {locked}", "SYSTEM")
            else:
                self.add_log(f"獲取餘額失敗: 無法識別返回格式", "ERROR")
                
        except Exception as e:
            self.add_log(f"查詢餘額時出錯: {str(e)}", "ERROR")
    
    def cmd_show_orders(self, args):
        """顯示活躍訂單"""
        if not self.strategy_running:
            self.add_log("沒有正在運行的策略", "ERROR")
            return
        
        # 顯示活躍買單
        self.add_log(f"活躍買單 ({len(self.market_maker.active_buy_orders)}):", "SYSTEM")
        for i, order in enumerate(self.market_maker.active_buy_orders[:5]):  # 只顯示前5個
            price = float(order.get('price', 0))
            quantity = float(order.get('quantity', 0))
            self.add_log(f"{i+1}. 買入 {quantity} @ {price}", "SYSTEM")
        
        if len(self.market_maker.active_buy_orders) > 5:
            self.add_log(f"... 還有 {len(self.market_maker.active_buy_orders) - 5} 個買單", "SYSTEM")
        
        # 顯示活躍賣單
        self.add_log(f"活躍賣單 ({len(self.market_maker.active_sell_orders)}):", "SYSTEM")
        for i, order in enumerate(self.market_maker.active_sell_orders[:5]):  # 只顯示前5個
            price = float(order.get('price', 0))
            quantity = float(order.get('quantity', 0))
            self.add_log(f"{i+1}. 賣出 {quantity} @ {price}", "SYSTEM")
        
        if len(self.market_maker.active_sell_orders) > 5:
            self.add_log(f"... 還有 {len(self.market_maker.active_sell_orders) - 5} 個賣單", "SYSTEM")
    
    def cmd_cancel_orders(self, args):
        """取消所有訂單"""
        if not self.strategy_running:
            self.add_log("沒有正在運行的策略", "ERROR")
            return
        
        self.add_log("正在取消所有訂單...", "SYSTEM")
        
        try:
            self.market_maker.cancel_existing_orders()
            self.add_log("所有訂單已取消", "SYSTEM")
        except Exception as e:
            self.add_log(f"取消訂單時出錯: {str(e)}", "ERROR")
    
    def cmd_diagnose(self, args):
        """執行系統診斷以檢查問題"""
        self.add_log("開始系統診斷...", "SYSTEM")
        
        # 檢查API密鑰
        try:
            from config import API_KEY, SECRET_KEY
            if not API_KEY or not SECRET_KEY:
                self.add_log("診斷問題: API密鑰未設置在config.py中", "ERROR")
            else:
                self.add_log("API密鑰已在config.py中設置", "SYSTEM")
        except ImportError:
            api_key = os.getenv('API_KEY')
            secret_key = os.getenv('SECRET_KEY')
            if not api_key or not secret_key:
                self.add_log("診斷問題: API密鑰未在環境變量中設置", "ERROR")
            else:
                self.add_log("API密鑰已在環境變量中設置", "SYSTEM")
        
        # 檢查網絡連接
        self.add_log("檢查網絡連接...", "SYSTEM")
        try:
            import socket
            try:
                # 嘗試連接到Backpack Exchange API域名
                socket.create_connection(("api.backpack.exchange", 443), timeout=10)
                self.add_log("網絡連接正常，可訪問Backpack Exchange API", "SYSTEM")
            except (socket.timeout, socket.error):
                self.add_log("診斷問題: 無法連接到Backpack Exchange API，請檢查網絡連接", "ERROR")
        except ImportError:
            self.add_log("無法檢查網絡連接：缺少socket模塊", "WARNING")
        
        # 檢查必要模塊
        self.add_log("檢查必要模塊...", "SYSTEM")
        modules_to_check = [
            ('WebSocket庫', 'websocket'),
            ('API客戶端', 'api.client'),
            ('數據庫模塊', 'database.db'),
            ('策略模塊', 'strategies.market_maker')
        ]
        
        for module_name, module_path in modules_to_check:
            try:
                __import__(module_path)
                self.add_log(f"{module_name}可用", "SYSTEM")
            except ImportError as e:
                self.add_log(f"診斷問題: {module_name}導入失敗: {str(e)}", "ERROR")
        
        # 檢查設定目錄
        settings_dir = 'settings'
        if not os.path.exists(settings_dir):
            self.add_log(f"診斷問題: 設定目錄不存在: {settings_dir}", "ERROR")
            try:
                os.makedirs(settings_dir, exist_ok=True)
                self.add_log("已創建設定目錄", "SYSTEM")
            except Exception as e:
                self.add_log(f"無法創建設定目錄: {str(e)}", "ERROR")
        else:
            self.add_log("設定目錄已存在", "SYSTEM")
        
        # 如果當前正在運行策略，檢查策略狀態
        if self.strategy_running and self.market_maker:
            self.add_log("檢查策略狀態...", "SYSTEM")
            
            # 檢查WebSocket連接
            if not hasattr(self.market_maker, 'ws') or not self.market_maker.ws:
                self.add_log("診斷問題: WebSocket連接不可用", "ERROR")
            elif not getattr(self.market_maker.ws, '_thread', None) or not self.market_maker.ws._thread.is_alive():
                self.add_log("診斷問題: WebSocket線程未運行", "ERROR")
            else:
                self.add_log("WebSocket連接正常", "SYSTEM")
            
            # 檢查訂單簿數據
            if hasattr(self.market_maker, 'ws') and self.market_maker.ws:
                if not self.market_maker.ws.orderbook or (not self.market_maker.ws.orderbook.get('bids') and not self.market_maker.ws.orderbook.get('asks')):
                    self.add_log("診斷問題: 訂單簿數據為空", "ERROR")
                else:
                    self.add_log("訂單簿數據正常", "SYSTEM")
        
        self.add_log("診斷完成", "SYSTEM")
        self.add_log("如遇問題，請檢查API密鑰是否正確，網絡連接是否正常", "SYSTEM")
        self.add_log("或者嘗試重新啟動程序", "SYSTEM")
    
    def start(self):
        """啟動交互式面板"""
        # 設置初始日誌
        self.add_log("做市交易面板已啟動", "SYSTEM")
        self.add_log("按 : 或 / 進入命令模式", "SYSTEM")
        self.add_log("輸入 help 查看可用命令", "SYSTEM")
        
        self.running = True
        
        # 創建並啟動更新線程
        self.update_thread = threading.Thread(target=self.update_thread_function, daemon=True)
        self.update_thread.start()
        
        try:
            # 創建實時顯示並處理按鍵
            with Live(self.layout, refresh_per_second=2, screen=True) as live:
                while self.running:
                    # 更新顯示
                    self.update_display()
                    
                    # 等待按鍵輸入
                    # 注意：這裡需要在真實環境中實現鍵盤輸入捕獲
                    # 但在這個示例中，我們只是簡單地休眠
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理資源"""
        # 停止策略（如果正在運行）
        if self.strategy_running and hasattr(self, '_cleanup_strategy'):
            self._cleanup_strategy()

# 主函數
def main():
    panel = InteractivePanel()
    panel.start()

if __name__ == "__main__":
    main()