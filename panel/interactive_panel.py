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

# 导入配置模块
try:
    from config import API_KEY, SECRET_KEY
except ImportError:
    API_KEY = os.getenv('API_KEY')
    SECRET_KEY = os.getenv('SECRET_KEY')

# 导入设定模块
try:
    from panel.settings import get_setting, set_setting, update_settings, load_settings
except ImportError:
    # 在直接运行面板文件时，可能会遇到导入问题，尝试直接导入
    try:
        from settings import get_setting, set_setting, update_settings, load_settings
    except ImportError:
        # 如果无法导入，创建空的设定函数
        def get_setting(key, default=None): return default
        def set_setting(key, value): pass
        def update_settings(settings_dict): pass
        def load_settings(): return {}

class InteractivePanel:
    def __init__(self):
        """初始化交互面板"""
        # 初始化默认设定
        self.settings = load_settings()
        
        # 客户端缓存，避免重复创建实例拖慢速度
        self._client_cache = {}
        
        # 策略参数
        self.strategy_params = {
            'base_spread_percentage': self.settings.get('base_spread_percentage', 0.1),
            'order_quantity': self.settings.get('order_quantity', None),
            'max_orders': self.settings.get('max_orders', 3),
            'duration': self.settings.get('duration', 24*3600),
            'interval': self.settings.get('interval', 60),
            'market_type': self.settings.get('market_type', 'spot'),
            'target_position': self.settings.get('target_position', 0.0),
            'max_position': self.settings.get('max_position', 1.0),
            'position_threshold': self.settings.get('position_threshold', 0.1),
            'inventory_skew': self.settings.get('inventory_skew', 0.25),
        }
        
        # 策略状态
        self.strategy_running = False
        self._initializing_strategy = False
        self.current_symbol = None
        self.market_maker = None
        self.strategy_thread = None
        self.last_market_update = datetime.now()
        
        # 市场数据
        self.market_data = {
            'bp_prices': {},    # 基准价格
            'bid_prices': {},   # 买价
            'ask_prices': {},   # 卖价
            'spread_pct': {},   # 价差百分比
            'buy_orders': {},   # 买单数量
            'sell_orders': {},  # 卖单数量
            'positions': {},    # 持仓状态 (多/空/平)
        }
        
        # 策略数据
        self.strategy_data = {
            'base_spread': 0.0,      # 基础价差
            'total_bought': 0.0,     # 总购买量
            'total_sold': 0.0,       # 总卖出量
            'maker_buy_volume': 0.0, # Maker买入量
            'maker_sell_volume': 0.0,# Maker卖出量
            'taker_buy_volume': 0.0, # Taker买入量
            'taker_sell_volume': 0.0,# Taker卖出量
            'session_profit': 0.0,   # 本次利润
            'total_profit': 0.0,     # 总利润
            'orders_placed': 0,      # 订单数量
            'trades_executed': 0,    # 成交数量
            'market_type': self.settings.get('market_type', 'spot'),
            'target_position': self.settings.get('target_position', 0.0),
            'max_position': self.settings.get('max_position', 1.0),
            'position_threshold': self.settings.get('position_threshold', 0.1),
            'inventory_skew': self.settings.get('inventory_skew', 0.25),
            'position_state': None,
        }
        
        # API 密钥 (从环境变数或设定读取)
        self.api_key = os.environ.get('API_KEY', '')
        self.secret_key = os.environ.get('SECRET_KEY', '')
        
        self.console = Console()
        self.layout = self.create_layout()
        self.live = None
        self.running = False
        self.update_thread = None
        
        # 命令和状态
        self.command_handlers = {}
        self.command_mode = False  # 切换命令模式
        self.current_command = ""
        self.command_history = []
        self.max_command_history = 20
        
        # 系统日志
        self.logs = []
        self.max_logs = 15  # 最多显示日志条数
        
        # 注册命令处理函数
        self.register_commands()
        
    def _get_client(self, api_key=None, secret_key=None):
        """获取缓存的客户端实例，避免重复创建"""
        from api.bp_client import BPClient
        
        # 为无认证的公开API调用创建一个通用客户端
        if api_key is None and secret_key is None:
            cache_key = "public"
            if cache_key not in self._client_cache:
                self._client_cache[cache_key] = BPClient({})
            return self._client_cache[cache_key]
        
        # 为认证API调用创建特定的客户端
        cache_key = f"{api_key}_{secret_key}"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = BPClient({'api_key': api_key, 'secret_key': secret_key})
        return self._client_cache[cache_key]
    
    def add_log(self, message, level="INFO"):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append((timestamp, level, message))
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
    
    def register_commands(self):
        """注册所有命令和处理函数"""
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
        """创建UI布局"""
        layout = Layout()
        
        # 分成上中下三部分
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="command", size=3)
        )
        
        # 主区域分成左右两部分
        layout["main"].split_row(
            Layout(name="market_data", ratio=3),
            Layout(name="logs", ratio=2)
        )
        
        return layout
    
    def generate_header(self):
        """生成头部面板"""
        status = "闲置中"
        if self.strategy_running:
            status = f"运行中 - {self.current_symbol}" if self.current_symbol else "运行中"
        
        title = f"做市交易机器人 - [{status}]"
        return Panel(
            Align.center(title, vertical="middle"),
            style="bold white on blue"
        )
    
    def generate_market_table(self):
        """生成市场数据表格"""
        # 创建表格
        last_update_str = self.last_market_update.strftime("%H:%M:%S")
        table = Table(title=f"市场数据 (更新: {last_update_str})", show_header=True, header_style="bold white on dark_blue", box=box.SIMPLE)
        
        
        # 添加列
        table.add_column("币种", style="cyan")
        table.add_column("BP价格", justify="right", style="green")
        table.add_column("买价", justify="right", style="green")
        table.add_column("卖价", justify="right", style="green")
        table.add_column("价差%", justify="right", style="magenta")
        table.add_column("买单数", justify="right", style="blue")
        table.add_column("卖单数", justify="right", style="red")
        table.add_column("持仓", justify="center", style="yellow")
        
        # 添加活跃交易对的数据
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
        
        # 添加策略数据
        strategy_table = Table(title="策略数据", show_header=True, header_style="bold white on dark_blue", box=box.SIMPLE)
        strategy_table.add_column("参数", style="yellow")
        strategy_table.add_column("数值", style="cyan", justify="right")
        
        # 添加重要的策略参数
        strategy_table.add_row("基础价差", f"{self.strategy_data['base_spread']:.4f}%")
        strategy_table.add_row("市场类型", self.strategy_data.get('market_type', 'spot'))

        if self.strategy_data.get('market_type') == 'perp':
            strategy_table.add_row("目标净仓", f"{self.strategy_data.get('target_position', 0.0):.4f}")
            strategy_table.add_row("最大仓位", f"{self.strategy_data.get('max_position', 0.0):.4f}")
            strategy_table.add_row("调整阈值", f"{self.strategy_data.get('position_threshold', 0.0):.4f}")
            strategy_table.add_row("报价偏移", f"{self.strategy_data.get('inventory_skew', 0.0):.2f}")

        # 显示订单数量
        order_quantity = self.strategy_params.get('order_quantity')
        if order_quantity is not None:
            strategy_table.add_row("订单数量", f"{order_quantity}")
        else:
            strategy_table.add_row("订单数量", "自动")
        
        # 显示利润表
        profit_table = Table(title="利润统计", show_header=True, header_style="bold white on dark_blue", box=box.SIMPLE)
        profit_table.add_column("指标", style="yellow")
        profit_table.add_column("数值", style="cyan", justify="right")
        
        total_profit = self.strategy_data['total_profit']
        session_profit = self.strategy_data['session_profit']
        
        profit_style = "green" if total_profit >= 0 else "red"
        session_style = "green" if session_profit >= 0 else "red"
        
        profit_table.add_row("总利润", f"{total_profit:.6f}")
        profit_table.add_row("本次利润", Text(f"{session_profit:.6f}", style=session_style))
        
        # 添加仓位表
        position_table = Table(title="仓位统计", show_header=True, header_style="bold white on dark_blue", box=box.SIMPLE)
        position_table.add_column("指标", style="yellow")
        position_table.add_column("数值", style="cyan", justify="right")
        
        total_bought = self.strategy_data['total_bought']
        total_sold = self.strategy_data['total_sold']
        imbalance = total_bought - total_sold
        imbalance_pct = abs(imbalance) / max(total_bought, total_sold) * 100 if max(total_bought, total_sold) > 0 else 0
        
        position_table.add_row("买入总量", f"{total_bought:.6f}")
        position_table.add_row("卖出总量", f"{total_sold:.6f}")
        position_table.add_row("净仓位", f"{imbalance:.6f}")
        position_table.add_row("不平衡%", f"{imbalance_pct:.2f}%")
        position_table.add_row("Maker买入", f"{self.strategy_data['maker_buy_volume']:.6f}")
        position_table.add_row("Maker卖出", f"{self.strategy_data['maker_sell_volume']:.6f}")

        position_state = self.strategy_data.get('position_state')
        if position_state:
            position_table.add_row("当前仓位", f"{position_state.get('net', 0.0):.6f}")
            position_table.add_row("仓位方向", position_state.get('direction', '-'))
            position_table.add_row("平均开仓价", f"{position_state.get('avg_entry', 0.0):.6f}")
            position_table.add_row("未实现PnL", f"{position_state.get('unrealized', 0.0):.6f}")

        return Group(table, strategy_table, profit_table, position_table)
    
    def generate_log_panel(self):
        """生成日志面板"""
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
            title="系统日志",
            border_style="bright_blue"
        )
    
    def generate_command_panel(self):
        """生成命令面板"""
        if self.command_mode:
            command_text = f"> {self.current_command}"
        else:
            command_text = "按 : 或 / 进入命令模式  |  帮助命令: help"
            
        return Panel(
            Text(command_text, style="bold cyan"),
            title="命令",
            border_style="green"
        )
    
    def update_display(self):
        """更新显示内容"""
        self.layout["header"].update(self.generate_header())
        self.layout["market_data"].update(self.generate_market_table())
        self.layout["logs"].update(self.generate_log_panel())
        self.layout["command"].update(self.generate_command_panel())
    
    def update_thread_function(self):
        """更新线程函数"""
        while self.running:
            self.update_display()
            time.sleep(0.5)  # 每0.5秒更新一次，避免过高的CPU使用率
    
    def handle_input(self, key):
        """处理键盘输入"""
        if self.command_mode:
            # 命令模式下的按键处理
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
            # 非命令模式下的按键处理
            if key == ":" or key == "/":
                self.command_mode = True
                self.current_command = ""
            elif key == "q":
                self.running = False
    
    def execute_command(self, command):
        """执行命令"""
        # 添加到命令历史
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
        
        # 执行对应的命令处理函数
        if cmd in self.command_handlers:
            self.add_log(f"执行命令: {command}", "COMMAND")
            try:
                self.command_handlers[cmd](args)
            except Exception as e:
                self.add_log(f"执行命令出错: {str(e)}", "ERROR")
        else:
            self.add_log(f"未知命令: {cmd}", "ERROR")
    
    def cmd_help(self, args):
        """显示帮助信息"""
        self.add_log("可用命令:", "SYSTEM")
        self.add_log("help - 显示帮助", "SYSTEM")
        self.add_log("symbols - 列出可用交易对", "SYSTEM")
        self.add_log("start <symbol> - 启动指定交易对的做市策略", "SYSTEM")
        self.add_log("  现货示例: start SOL_USDC", "SYSTEM")
        self.add_log("  永续示例: start SOL_USDC_PERP (需先设置 market_type perp)", "SYSTEM")
        self.add_log("stop - 停止当前做市策略", "SYSTEM")
        self.add_log("params - 显示当前策略参数", "SYSTEM")
        self.add_log("set <参数> <值> - 设置策略参数", "SYSTEM")
        self.add_log("  基本参数: set base_spread 0.1", "SYSTEM")
        self.add_log("  市场类型: set market_type perp", "SYSTEM")
        self.add_log("  永续参数: set target_position 0.5", "SYSTEM")
        self.add_log("status - 显示当前状态", "SYSTEM")
        self.add_log("balance - 查询余额", "SYSTEM")
        self.add_log("orders - 显示活跃订单", "SYSTEM")
        self.add_log("cancel - 取消所有订单", "SYSTEM")
        self.add_log("clear - 清除日志", "SYSTEM")
        self.add_log("diagnose - 执行系统诊断检查", "SYSTEM")
        self.add_log("exit/quit - 退出程序", "SYSTEM")
    
    def cmd_clear(self, args):
        """清除日志"""
        self.logs = []
        self.add_log("日志已清除", "SYSTEM")
    
    def cmd_exit(self, args):
        """退出程序"""
        self.running = False
    
    def cmd_list_symbols(self, args):
        """列出可用交易对"""
        self.add_log("正在获取可用交易对...", "SYSTEM")
        
        try:
            markets_info = self._get_client().get_markets()
            if isinstance(markets_info, dict) and "error" in markets_info:
                self.add_log(f"获取市场信息失败: {markets_info['error']}", "ERROR")
                return
            
            spot_markets = [m for m in markets_info if m.get('marketType') == 'SPOT']
            self.market_data['symbols'] = [m.get('symbol') for m in spot_markets]
            
            self.add_log(f"找到 {len(spot_markets)} 个现货市场:", "SYSTEM")
            
            # 分组显示，每行最多5个
            symbols_per_line = 5
            for i in range(0, len(spot_markets), symbols_per_line):
                group = spot_markets[i:i+symbols_per_line]
                symbols_line = ", ".join([m.get('symbol') for m in group])
                self.add_log(symbols_line, "SYSTEM")
                
        except Exception as e:
            self.add_log(f"获取交易对时出错: {str(e)}", "ERROR")
    
    def cmd_start_strategy(self, args):
        """启动做市策略"""
        if not args:
            current_market_type = self.strategy_params.get('market_type', 'spot')
            if current_market_type == 'perp':
                self.add_log("请指定交易对，例如: start SOL_USDC_PERP", "ERROR")
                self.add_log("永续合约交易对示例: SOL_USDC_PERP, BTC_USDC_PERP, ETH_USDC_PERP", "SYSTEM")
            else:
                self.add_log("请指定交易对，例如: start SOL_USDC", "ERROR")
                self.add_log("现货交易对示例: SOL_USDC, BTC_USDC, ETH_USDC", "SYSTEM")
            return
        
        symbol = args[0]
        
        if self.strategy_running:
            self.add_log("已有策略运行中，请先停止当前策略", "ERROR")
            return
        
        self.add_log(f"正在启动 {symbol} 的做市策略...", "SYSTEM")
        
        try:
            # 导入必要的类
            from database.db import Database
            from strategies.market_maker import MarketMaker
            from strategies.perp_market_maker import PerpetualMarketMaker
            
            # 导入或获取API密钥
            try:
                from config import API_KEY as CONFIG_API_KEY, SECRET_KEY as CONFIG_SECRET_KEY, WS_PROXY as CONFIG_WS_PROXY
            except ImportError:
                CONFIG_API_KEY = os.getenv('API_KEY')
                CONFIG_SECRET_KEY = os.getenv('SECRET_KEY')
                CONFIG_WS_PROXY = os.getenv('PROXY_WEBSOCKET')
                
            api_key = CONFIG_API_KEY
            secret_key = CONFIG_SECRET_KEY
            ws_proxy = CONFIG_WS_PROXY
            
            if not api_key or not secret_key:
                self.add_log("缺少API密钥，请检查config.py或环境变量", "ERROR")
                return
                
            # 默认策略参数
            params = {
                'base_spread_percentage': 0.1,  # 默认价差0.1%
                'order_quantity': None,  # 添加订单数量
                'max_orders': 3,               # 每侧3个订单
                'execution_mode': 'standard',   # 标准执行模式
                'risk_factor': 0.5,            # 默认风险因子
                'duration': 24*3600,           # 运行24小时
                'interval': 60,                # 每分钟更新一次
                'market_type': 'spot',
                'target_position': 0.0,
                'max_position': 1.0,
                'position_threshold': 0.1,
                'inventory_skew': 0.25,
            }
            
            # 合并用户设置的参数
            for key, value in self.strategy_params.items():
                if key in params:
                    params[key] = value
            
            # 初始化数据库
            db = Database()
            
            # 设置当前交易对和标记策略为运行状态
            self.current_symbol = symbol
            
            # 记录当前正在初始化
            self._initializing_strategy = True
            
            # 更新策略数据
            self.strategy_data['base_spread'] = params['base_spread_percentage']
            self.strategy_data['market_type'] = params['market_type']
            self.strategy_data['target_position'] = params['target_position']
            self.strategy_data['max_position'] = params['max_position']
            self.strategy_data['position_threshold'] = params['position_threshold']
            self.strategy_data['inventory_skew'] = params['inventory_skew']
            
            # 初始化做市商
            if params['market_type'] == 'perp':
                self.add_log("使用永续合约策略参数", "SYSTEM")
                self.market_maker = PerpetualMarketMaker(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=symbol,
                    db_instance=db,
                    base_spread_percentage=params['base_spread_percentage'],
                    order_quantity=params['order_quantity'],
                    max_orders=params['max_orders'],
                    target_position=params['target_position'],
                    max_position=params['max_position'],
                    position_threshold=params['position_threshold'],
                    inventory_skew=params['inventory_skew'],
                    ws_proxy=ws_proxy
                )
            else:
                self.market_maker = MarketMaker(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=symbol,
                    db_instance=db,
                    base_spread_percentage=params['base_spread_percentage'],
                    order_quantity=params['order_quantity'],  # 使用设定的订单数量
                    max_orders=params['max_orders'],
                    ws_proxy=ws_proxy
                )

            if hasattr(self.market_maker, 'get_position_state'):
                try:
                    self.strategy_data['position_state'] = self.market_maker.get_position_state()
                except Exception:
                    self.strategy_data['position_state'] = None
            
            # 标记策略为运行状态
            self.strategy_running = True
            self._initializing_strategy = False
            
            # 启动策略在单独的线程中
            self.strategy_thread = threading.Thread(
                target=self._run_strategy_thread,
                args=(params['duration'], params['interval']),
                daemon=True
            )
            self.strategy_thread.start()
            
            self.add_log(f"{symbol} 做市策略已启动", "SYSTEM")
            
        except Exception as e:
            self._initializing_strategy = False
            self.strategy_running = False  # 确保在出错时重置状态
            self.current_symbol = None
            self.add_log(f"启动策略时出错: {str(e)}", "ERROR")
            import traceback
            self.add_log(f"详细错误: {traceback.format_exc()}", "ERROR")
    
    def _run_strategy_thread(self, duration_seconds, interval_seconds):
        """在单独线程中运行策略"""
        if not self.market_maker:
            self.add_log("做市商未初始化", "ERROR")
            self.strategy_running = False  # 确保重置状态
            return
        
        try:
            start_time = time.time()
            iteration = 0
            
            # 记录开始信息
            self.add_log(f"开始运行做市策略: {self.market_maker.symbol}")
            
            # 确保WebSocket连接
            try:
                # 首先检查WebSocket连接
                self.add_log("检查WebSocket连接...")
                connection_status = self.market_maker.check_ws_connection()
                if not connection_status:
                    self.add_log("WebSocket未连接，尝试建立连接...")
                    # 在MarketMaker中应该有初始化WebSocket的方法
                    if hasattr(self.market_maker, 'initialize_websocket'):
                        self.market_maker.initialize_websocket()
                    elif hasattr(self.market_maker, 'reconnect_websocket'):
                        self.market_maker.reconnect_websocket()
                
                # 再次检查连接状态
                connection_status = self.market_maker.check_ws_connection()
                if connection_status:
                    self.add_log("WebSocket连接成功")
                    
                    # 等待WebSocket就绪
                    self.add_log("等待WebSocket就绪...")
                    time.sleep(2)
                    
                    # 初始化订单簿
                    if hasattr(self.market_maker, 'ws') and self.market_maker.ws:
                        if not self.market_maker.ws.orderbook.get("bids") and not self.market_maker.ws.orderbook.get("asks"):
                            self.add_log("初始化订单簿...")
                            if hasattr(self.market_maker.ws, 'initialize_orderbook'):
                                self.market_maker.ws.initialize_orderbook()
                                # 等待订单簿填充
                                time.sleep(1)
                    
                    # 确保所有数据流订阅
                    self.add_log("确保数据流订阅...")
                    if hasattr(self.market_maker, '_ensure_data_streams'):
                        self.market_maker._ensure_data_streams()
                    
                    # 增加小延迟确保订阅成功
                    time.sleep(2)
                    self.add_log("数据流订阅完成，进入主循环...")
                else:
                    self.add_log("WebSocket连接失败，请检查网络或API配置", "ERROR")
                    self.strategy_running = False
                    return
            except Exception as ws_error:
                self.add_log(f"WebSocket设置出错: {str(ws_error)}", "ERROR")
                import traceback
                self.add_log(f"WebSocket错误详情: {traceback.format_exc()}", "ERROR")
                self.strategy_running = False
                return
                
            # 主循环前检查策略运行状态
            if not self.strategy_running:
                self.add_log("策略在初始化后被停止", "WARNING")
                return
            
            # 检查订单簿是否已填充
            if hasattr(self.market_maker, 'ws') and self.market_maker.ws and hasattr(self.market_maker.ws, 'orderbook'):
                if not self.market_maker.ws.orderbook.get("bids") or not self.market_maker.ws.orderbook.get("asks"):
                    self.add_log("警告: 订单簿可能未完全初始化", "WARNING")
                
            # 主循环
            self.add_log("开始执行策略主循环...")
            while time.time() - start_time < duration_seconds and self.strategy_running:
                iteration += 1
                
                self.add_log(f"第 {iteration} 次迭代")
                
                try:
                    # 检查连接
                    connection_status = self.market_maker.check_ws_connection()
                    if not connection_status:
                        self.add_log("WebSocket连接已断开，尝试重新连接...", "WARNING")
                        reconnected = self.market_maker.reconnect_websocket()
                        if not reconnected:
                            self.add_log("重新连接失败，停止策略", "ERROR")
                            break
                        # 给连接一些时间重新建立
                        time.sleep(2)
                        continue  # 跳过这次迭代
                    
                    # 更新面板数据
                    self._update_strategy_data()
                    
                    # 检查订单成交情况
                    self.add_log("检查订单成交情况...")
                    self.market_maker.check_order_fills()
                    
                    # 检查是否需要重平衡仓位
                    needs_rebalance = self.market_maker.need_rebalance()
                    if needs_rebalance:
                        self.add_log("执行仓位重平衡")
                        self.market_maker.rebalance_position()
                    
                    # 下限价单
                    self.add_log("下限价单...")
                    self.market_maker.place_limit_orders()
                    
                    # 估算利润
                    self.market_maker.estimate_profit()
                    
                except Exception as loop_error:
                    self.add_log(f"策略迭代中出错: {str(loop_error)}", "ERROR")
                    import traceback
                    self.add_log(f"迭代错误详情: {traceback.format_exc()}", "ERROR")
                    # 不因为单次循环错误停止整个策略，继续下一次循环
                    time.sleep(5)  # 出错时等待更长时间
                    continue
                
                # 等待下一次迭代
                time.sleep(interval_seconds)
            
            # 结束时记录信息
            if not self.strategy_running:
                self.add_log("策略已手动停止")
            else:
                self.add_log("策略运行完成")
            
            # 清理资源
            self._cleanup_strategy()
            
        except Exception as e:
            self.add_log(f"策略运行出错: {str(e)}", "ERROR")
            import traceback
            self.add_log(f"错误详情: {traceback.format_exc()}", "ERROR")
            
            # 确保清理资源
            self._cleanup_strategy()
    
    def _update_strategy_data(self):
        """从市场做市商更新数据到面板"""
        if not self.market_maker:
            return
            
        try:
            # 更新最后一次市场数据更新时间
            self.last_market_update = datetime.now()
            
            # 检查 WebSocket 连接
            if not hasattr(self.market_maker, 'ws') or not self.market_maker.ws:
                self.add_log("WebSocket 连接不可用", "WARNING")
                return
            
            # 更新市场数据
            symbol = self.current_symbol
            if symbol:
                try:
                    # 更新价格数据
                    bid_price = getattr(self.market_maker.ws, 'bid_price', None)
                    ask_price = getattr(self.market_maker.ws, 'ask_price', None)
                    
                    if bid_price and ask_price:
                        bp_price = (bid_price + ask_price) / 2
                        self.market_data['bp_prices'][symbol] = bp_price
                        self.market_data['bid_prices'][symbol] = bid_price
                        self.market_data['ask_prices'][symbol] = ask_price
                        
                        # 计算价差
                        spread_pct = (ask_price - bid_price) / bp_price * 100
                        self.market_data['spread_pct'][symbol] = f"{spread_pct:.6f}%"
                except Exception as price_err:
                    self.add_log(f"更新价格数据时出错: {str(price_err)}", "WARNING")
                
                try:
                    # 更新订单数量
                    buy_orders = getattr(self.market_maker, 'active_buy_orders', [])
                    sell_orders = getattr(self.market_maker, 'active_sell_orders', [])
                    self.market_data['buy_orders'][symbol] = len(buy_orders)
                    self.market_data['sell_orders'][symbol] = len(sell_orders)
                except Exception as order_err:
                    self.add_log(f"更新订单数量时出错: {str(order_err)}", "WARNING")
                
                try:
                    # 更新持仓状态
                    total_bought = getattr(self.market_maker, 'total_bought', 0)
                    total_sold = getattr(self.market_maker, 'total_sold', 0)
                    
                    if total_bought > total_sold:
                        self.market_data['positions'][symbol] = "多"
                    elif total_bought < total_sold:
                        self.market_data['positions'][symbol] = "空"
                    else:
                        self.market_data['positions'][symbol] = "平"
                except Exception as pos_err:
                    self.add_log(f"更新持仓状态时出错: {str(pos_err)}", "WARNING")
            
            # 更新交易量数据
            try:
                self.strategy_data['total_bought'] = getattr(self.market_maker, 'total_bought', 0)
                self.strategy_data['total_sold'] = getattr(self.market_maker, 'total_sold', 0)
                self.strategy_data['maker_buy_volume'] = getattr(self.market_maker, 'maker_buy_volume', 0)
                self.strategy_data['maker_sell_volume'] = getattr(self.market_maker, 'maker_sell_volume', 0)
                self.strategy_data['taker_buy_volume'] = getattr(self.market_maker, 'taker_buy_volume', 0)
                self.strategy_data['taker_sell_volume'] = getattr(self.market_maker, 'taker_sell_volume', 0)
                self.strategy_data['orders_placed'] = getattr(self.market_maker, 'orders_placed', 0)
                self.strategy_data['trades_executed'] = getattr(self.market_maker, 'trades_executed', 0)

                # 利润统计
                self.strategy_data['session_profit'] = getattr(self.market_maker, 'session_profit', 0.0)
                self.strategy_data['total_profit'] = getattr(self.market_maker, 'total_profit', 0.0)

                self.strategy_data['market_type'] = self.strategy_params.get('market_type', 'spot')
                self.strategy_data['target_position'] = getattr(self.market_maker, 'target_position', self.strategy_params.get('target_position', 0.0))
                self.strategy_data['max_position'] = getattr(self.market_maker, 'max_position', self.strategy_params.get('max_position', 1.0))
                self.strategy_data['position_threshold'] = getattr(self.market_maker, 'position_threshold', self.strategy_params.get('position_threshold', 0.1))
                self.strategy_data['inventory_skew'] = getattr(self.market_maker, 'inventory_skew', self.strategy_params.get('inventory_skew', 0.25))

                position_state = None
                if hasattr(self.market_maker, 'get_position_state'):
                    try:
                        position_state = self.market_maker.get_position_state()
                    except Exception as pos_err:
                        self.add_log(f"获取仓位资讯时出错: {str(pos_err)}", "WARNING")
                self.strategy_data['position_state'] = position_state
            except Exception as vol_err:
                self.add_log(f"更新交易量数据时出错: {str(vol_err)}", "WARNING")
                
        except Exception as e:
            self.add_log(f"更新面板数据时出错: {str(e)}", "ERROR")
    
    def _cleanup_strategy(self):
        """清理策略资源"""
        if not self.market_maker:
            return
            
        # 标记清理开始
        was_running = self.strategy_running
        # 标记策略为停止状态，以防止任何进一步的操作
        self.strategy_running = False
        
        try:
            # 记录清理消息
            if was_running:
                self.add_log("正在清理策略资源...", "SYSTEM")
                
            # 取消所有活跃订单
            self.add_log("取消所有未成交订单...")
            try:
                if hasattr(self.market_maker, 'cancel_existing_orders'):
                    self.market_maker.cancel_existing_orders()
                    self.add_log("所有订单已取消")
                else:
                    self.add_log("无法取消订单: 方法不可用", "WARNING")
            except Exception as cancel_err:
                self.add_log(f"取消订单时出错: {str(cancel_err)}", "ERROR")
            
            # 关闭WebSocket连接
            try:
                if hasattr(self.market_maker, 'ws') and self.market_maker.ws:
                    self.add_log("关闭WebSocket连接...")
                    self.market_maker.ws.close()
                    self.add_log("WebSocket连接已关闭")
            except Exception as ws_err:
                self.add_log(f"关闭WebSocket时出错: {str(ws_err)}", "ERROR")
            
            # 关闭数据库连接
            try:
                if hasattr(self.market_maker, 'db') and self.market_maker.db:
                    self.add_log("关闭数据库连接...")
                    self.market_maker.db.close()
                    self.add_log("数据库连接已关闭")
            except Exception as db_err:
                self.add_log(f"关闭数据库时出错: {str(db_err)}", "ERROR")
            
            # 确认清理完成
            if was_running:
                self.add_log("策略资源清理完成", "SYSTEM")
                
        except Exception as e:
            self.add_log(f"清理资源时遇到未知错误: {str(e)}", "ERROR")
            import traceback
            self.add_log(f"错误详情: {traceback.format_exc()}", "ERROR")
        finally:
            # 清空策略实例
            self.current_symbol = None
            self.market_maker = None
    
    def cmd_stop_strategy(self, args):
        """停止当前运行的策略"""
        if not self.strategy_running and not self._initializing_strategy:
            self.add_log("没有正在运行的策略", "ERROR")
            return
            
        if self._initializing_strategy:
            self.add_log("策略正在初始化中，请稍后再试", "WARNING")
            return
        
        self.add_log("正在停止策略...")
        self.strategy_running = False
        
        # 等待策略线程结束
        if hasattr(self, 'strategy_thread') and self.strategy_thread and self.strategy_thread.is_alive():
            try:
                self.strategy_thread.join(timeout=3)
                if self.strategy_thread.is_alive():
                    self.add_log("策略线程未能在3秒内结束，可能需要手动重启程序", "WARNING")
            except Exception as join_err:
                self.add_log(f"等待策略线程时出错: {str(join_err)}", "ERROR")
            
        self.add_log("策略已停止")
    
    def cmd_show_params(self, args):
        """显示当前策略参数"""
        self.add_log("当前策略参数:", "SYSTEM")
        
        if not self.strategy_params:
            self.add_log("尚未设置任何参数，使用默认值", "SYSTEM")
            self.add_log("可用参数:", "SYSTEM")
            self.add_log("base_spread - 基础价差百分比 (例: 0.1 = 0.1%)", "SYSTEM")
            self.add_log("order_quantity - 订单数量 (例: 0.5 SOL，auto为自动)", "SYSTEM")
            self.add_log("max_orders - 每侧最大订单数 (例: 3)", "SYSTEM")
            self.add_log("duration - 运行时间(秒) (例: 3600 = 1小时)", "SYSTEM")
            self.add_log("interval - 更新间隔(秒) (例: 60 = 1分钟)", "SYSTEM")
            self.add_log("market_type - 市场类型: spot(现货) 或 perp(永续)", "SYSTEM")
            self.add_log("", "SYSTEM")
            self.add_log("永续合约专用参数:", "SYSTEM")
            self.add_log("target_position - 目标净仓位 (例: 0.0为中性)", "SYSTEM")
            self.add_log("max_position - 最大仓位限制 (例: 1.0)", "SYSTEM")
            self.add_log("position_threshold - 仓位调整触发值 (例: 0.1)", "SYSTEM")
            self.add_log("inventory_skew - 报价偏移系数 (例: 0.25)", "SYSTEM")
            return
        
        for param, value in self.strategy_params.items():
            # 订单数量可能为空，特殊处理
            if param == 'order_quantity' and value is None:
                self.add_log(f"{param} = 自动 (根据余额决定)", "SYSTEM")
            else:
                self.add_log(f"{param} = {value}", "SYSTEM")
                
        # 添加使用说明
        current_market_type = self.strategy_params.get('market_type', 'spot')
        self.add_log("\n设置参数示例:", "SYSTEM")
        self.add_log("set base_spread 0.2         - 设置价差为0.2%", "SYSTEM")
        self.add_log("set order_quantity 0.5      - 设置订单数量为0.5", "SYSTEM")
        self.add_log("set order_quantity auto     - 设为自动订单数量", "SYSTEM")
        self.add_log("set max_orders 5            - 设置每侧最大订单数为5", "SYSTEM")
        self.add_log("set market_type spot        - 设为现货交易模式", "SYSTEM")
        self.add_log("set market_type perp        - 设为永续合约模式", "SYSTEM")
        
        if current_market_type == 'perp':
            self.add_log("\n永续合约模式专用参数:", "SYSTEM")
            self.add_log("set target_position 0.0     - 目标中性仓位", "SYSTEM")
            self.add_log("set target_position 0.5     - 目标做多0.5", "SYSTEM")
            self.add_log("set max_position 1.0        - 最大仓位限制", "SYSTEM")
            self.add_log("set position_threshold 0.1  - 仓位调整触发", "SYSTEM")
            self.add_log("set inventory_skew 0.25     - 报价偏移系数", "SYSTEM")
            self.add_log("\n永续合约交易对示例: SOL_USDC_PERP", "SYSTEM")
        else:
            self.add_log("\n现货交易对示例: SOL_USDC, BTC_USDC, ETH_USDC", "SYSTEM")
            self.add_log("切换到永续模式: set market_type perp", "SYSTEM")
    
    def cmd_set_param(self, args):
        """设置策略参数"""
        if len(args) < 2:
            self.add_log("用法: set <参数名> <参数值>", "ERROR")
            current_market_type = self.strategy_params.get('market_type', 'spot')
            self.add_log("\n常用参数示例:", "SYSTEM")
            self.add_log("set base_spread 0.2         - 基础价差0.2%", "SYSTEM")
            self.add_log("set order_quantity 0.5      - 订单数量0.5", "SYSTEM")
            self.add_log("set order_quantity auto     - 自动订单数量", "SYSTEM")
            self.add_log("set max_orders 5            - 每侧最大5个订单", "SYSTEM")
            self.add_log("set market_type spot        - 现货模式", "SYSTEM")
            self.add_log("set market_type perp        - 永续合约模式", "SYSTEM")
            if current_market_type == 'perp':
                self.add_log("\n永续合约参数:", "SYSTEM")
                self.add_log("set target_position 0.0     - 中性仓位", "SYSTEM")
                self.add_log("set max_position 1.0        - 最大仓位", "SYSTEM")
                self.add_log("set position_threshold 0.1  - 调整触发", "SYSTEM")
                self.add_log("set inventory_skew 0.25     - 报价偏移", "SYSTEM")
            return
        
        param = args[0]
        value = args[1]
        
        float_params = {'base_spread_percentage', 'target_position', 'max_position', 'position_threshold', 'inventory_skew'}
        int_params = {'max_orders', 'duration', 'interval'}
        str_params = {'market_type'}

        all_params = float_params | int_params | str_params | {'order_quantity'}

        if param not in all_params:
            self.add_log(f"无效的参数名: {param}", "ERROR")
            self.add_log("有效参数: " + ", ".join(sorted(all_params)), "SYSTEM")
            return

        try:
            if param == 'order_quantity':
                if value.lower() in ('auto', 'none', 'null'):
                    typed_value = None
                    self.add_log("订单数量设为自动 (由程式决定)", "SYSTEM")
                else:
                    typed_value = float(value)
                    if typed_value <= 0:
                        raise ValueError("订单数量必须大于0")
            elif param in int_params:
                typed_value = int(value)
                if typed_value <= 0:
                    raise ValueError("整数参数必须大于0")
            elif param in float_params:
                typed_value = float(value)
                if param in {'base_spread_percentage', 'position_threshold'} and typed_value <= 0:
                    raise ValueError("参数必须大于0")
                if param == 'max_position' and typed_value <= 0:
                    raise ValueError("最大仓位必须大于0")
                if param == 'inventory_skew' and not 0 <= typed_value <= 1:
                    raise ValueError("inventory_skew 必须介于 0-1 之间")
            elif param in str_params:
                typed_value = value.lower()
                if typed_value not in ('spot', 'perp'):
                    raise ValueError("market_type 仅支援 spot 或 perp")
            else:
                typed_value = value

            self.strategy_params[param] = typed_value

            try:
                set_setting(param, typed_value)
                self.add_log(f"参数已设置并保存: {param} = {typed_value}", "SYSTEM")
                
                # 根据设置的参数给出相应提示
                if param == 'market_type':
                    if typed_value == 'perp':
                        self.add_log("已切换到永续合约模式！", "SYSTEM")
                        self.add_log("永续合约交易对示例: SOL_USDC_PERP, BTC_USDC_PERP", "SYSTEM")
                        self.add_log("建议设置永续参数: target_position, max_position", "SYSTEM")
                    else:
                        self.add_log("已切换到现货交易模式！", "SYSTEM")
                        self.add_log("现货交易对示例: SOL_USDC, BTC_USDC, ETH_USDC", "SYSTEM")
                elif param == 'base_spread_percentage':
                    self.add_log(f"价差已设为 {typed_value}%，较小的价差可能提高成交率但减少利润", "SYSTEM")
                elif param == 'order_quantity':
                    if typed_value is None:
                        self.add_log("订单数量设为自动，将根据余额动态计算", "SYSTEM")
                    else:
                        self.add_log(f"订单数量已固定为 {typed_value}，确保有足够余额", "SYSTEM")
                elif param in ['target_position', 'max_position', 'position_threshold', 'inventory_skew']:
                    if self.strategy_params.get('market_type') != 'perp':
                        self.add_log("注意：此参数仅在永续合约模式下生效", "WARNING")
                        self.add_log("切换到永续模式: set market_type perp", "SYSTEM")
                    
            except Exception as e:
                self.add_log(f"参数已设置但保存失败: {str(e)}", "WARNING")

        except ValueError as e:
            self.add_log(f"参数值转换错误: {str(e)}", "ERROR")
    
    def cmd_show_status(self, args):
        """显示当前状态"""
        if not self.strategy_running:
            self.add_log("没有正在运行的策略", "SYSTEM")
            return
        
        self.add_log(f"正在运行 {self.current_symbol} 的做市策略", "SYSTEM")
        
        # 显示策略参数
        self.add_log("策略参数:", "SYSTEM")
        self.add_log(f"基础价差: {self.strategy_data['base_spread']:.4f}%", "SYSTEM")
        
        # 显示订单数量
        order_quantity = self.strategy_params.get('order_quantity')
        if order_quantity is not None:
            self.add_log(f"订单数量: {order_quantity}", "SYSTEM")
        else:
            self.add_log("订单数量: 自动", "SYSTEM")
            
        self.add_log(f"最大订单数: {self.strategy_params.get('max_orders', 3)}", "SYSTEM")

        market_type = self.strategy_data.get('market_type', 'spot')
        self.add_log(f"市场类型: {market_type}", "SYSTEM")
        if market_type == 'perp':
            self.add_log(f"目标净仓: {self.strategy_data.get('target_position', 0.0)}", "SYSTEM")
            self.add_log(f"最大仓位: {self.strategy_data.get('max_position', 1.0)}", "SYSTEM")
            self.add_log(f"调整触发: {self.strategy_data.get('position_threshold', 0.1)}", "SYSTEM")
            self.add_log(f"报价偏移: {self.strategy_data.get('inventory_skew', 0.25)}", "SYSTEM")

        # 显示重要状态指标
        self.add_log("\n仓位统计:", "SYSTEM")
        total_bought = self.strategy_data['total_bought']
        total_sold = self.strategy_data['total_sold']
        imbalance = total_bought - total_sold
        imbalance_pct = abs(imbalance) / max(total_bought, total_sold) * 100 if max(total_bought, total_sold) > 0 else 0

        self.add_log(f"总买入: {total_bought} - 总卖出: {total_sold}", "SYSTEM")
        self.add_log(f"仓位不平衡度: {imbalance_pct:.2f}%", "SYSTEM")

        position_state = self.strategy_data.get('position_state')
        if position_state:
            self.add_log(f"当前净仓: {position_state.get('net', 0.0):.6f} ({position_state.get('direction', '-')})", "SYSTEM")
            self.add_log(f"平均开仓价: {position_state.get('avg_entry', 0.0):.6f}", "SYSTEM")
            self.add_log(f"未实现PnL: {position_state.get('unrealized', 0.0):.6f}", "SYSTEM")

        # 显示利润信息
        self.add_log("\n利润统计:", "SYSTEM")
        total_profit = self.strategy_data['total_profit']
        
        self.add_log(f"总利润: {total_profit:.6f}", "SYSTEM")
    
    def cmd_show_balance(self, args):
        """显示当前余额"""
        self.add_log("正在查询余额...", "SYSTEM")
        
        try:
            balances = self._get_client(API_KEY, SECRET_KEY).get_balance()
            if isinstance(balances, dict) and "error" in balances and balances["error"]:
                self.add_log(f"获取余额失败: {balances['error']}", "ERROR")
                return
            
            self.add_log("当前余额:", "SYSTEM")
            if isinstance(balances, dict):
                for coin, details in balances.items():
                    available = float(details.get('available', 0))
                    locked = float(details.get('locked', 0))
                    if available > 0 or locked > 0:
                        self.add_log(f"{coin}: 可用 {available}, 冻结 {locked}", "SYSTEM")
            else:
                self.add_log(f"获取余额失败: 无法识别返回格式", "ERROR")
                
        except Exception as e:
            self.add_log(f"查询余额时出错: {str(e)}", "ERROR")
    
    def cmd_show_orders(self, args):
        """显示活跃订单"""
        if not self.strategy_running:
            self.add_log("没有正在运行的策略", "ERROR")
            return
        
        # 显示活跃买单
        self.add_log(f"活跃买单 ({len(self.market_maker.active_buy_orders)}):", "SYSTEM")
        for i, order in enumerate(self.market_maker.active_buy_orders[:5]):  # 只显示前5个
            price = float(order.get('price', 0))
            quantity = float(order.get('quantity', 0))
            self.add_log(f"{i+1}. 买入 {quantity} @ {price}", "SYSTEM")
        
        if len(self.market_maker.active_buy_orders) > 5:
            self.add_log(f"... 还有 {len(self.market_maker.active_buy_orders) - 5} 个买单", "SYSTEM")
        
        # 显示活跃卖单
        self.add_log(f"活跃卖单 ({len(self.market_maker.active_sell_orders)}):", "SYSTEM")
        for i, order in enumerate(self.market_maker.active_sell_orders[:5]):  # 只显示前5个
            price = float(order.get('price', 0))
            quantity = float(order.get('quantity', 0))
            self.add_log(f"{i+1}. 卖出 {quantity} @ {price}", "SYSTEM")
        
        if len(self.market_maker.active_sell_orders) > 5:
            self.add_log(f"... 还有 {len(self.market_maker.active_sell_orders) - 5} 个卖单", "SYSTEM")
    
    def cmd_cancel_orders(self, args):
        """取消所有订单"""
        if not self.strategy_running:
            self.add_log("没有正在运行的策略", "ERROR")
            return
        
        self.add_log("正在取消所有订单...", "SYSTEM")
        
        try:
            self.market_maker.cancel_existing_orders()
            self.add_log("所有订单已取消", "SYSTEM")
        except Exception as e:
            self.add_log(f"取消订单时出错: {str(e)}", "ERROR")
    
    def cmd_diagnose(self, args):
        """执行系统诊断以检查问题"""
        self.add_log("开始系统诊断...", "SYSTEM")
        
        # 检查API密钥
        try:
            from config import API_KEY, SECRET_KEY
            if not API_KEY or not SECRET_KEY:
                self.add_log("诊断问题: API密钥未设置在config.py中", "ERROR")
            else:
                self.add_log("API密钥已在config.py中设置", "SYSTEM")
        except ImportError:
            api_key = os.getenv('API_KEY')
            secret_key = os.getenv('SECRET_KEY')
            if not api_key or not secret_key:
                self.add_log("诊断问题: API密钥未在环境变量中设置", "ERROR")
            else:
                self.add_log("API密钥已在环境变量中设置", "SYSTEM")
        
        # 检查网络连接
        self.add_log("检查网络连接...", "SYSTEM")
        try:
            import socket
            try:
                # 尝试连接到Backpack Exchange API域名
                socket.create_connection(("api.backpack.exchange", 443), timeout=10)
                self.add_log("网络连接正常，可访问Backpack Exchange API", "SYSTEM")
            except (socket.timeout, socket.error):
                self.add_log("诊断问题: 无法连接到Backpack Exchange API，请检查网络连接", "ERROR")
        except ImportError:
            self.add_log("无法检查网络连接：缺少socket模块", "WARNING")
        
        # 检查必要模块
        self.add_log("检查必要模块...", "SYSTEM")
        modules_to_check = [
            ('WebSocket库', 'websocket'),
            ('API客户端', 'api.bp_client'),
            ('数据库模块', 'database.db'),
            ('策略模块', 'strategies.market_maker')
        ]
        
        for module_name, module_path in modules_to_check:
            try:
                __import__(module_path)
                self.add_log(f"{module_name}可用", "SYSTEM")
            except ImportError as e:
                self.add_log(f"诊断问题: {module_name}导入失败: {str(e)}", "ERROR")
        
        # 检查设定目录
        settings_dir = 'settings'
        if not os.path.exists(settings_dir):
            self.add_log(f"诊断问题: 设定目录不存在: {settings_dir}", "ERROR")
            try:
                os.makedirs(settings_dir, exist_ok=True)
                self.add_log("已创建设定目录", "SYSTEM")
            except Exception as e:
                self.add_log(f"无法创建设定目录: {str(e)}", "ERROR")
        else:
            self.add_log("设定目录已存在", "SYSTEM")
        
        # 如果当前正在运行策略，检查策略状态
        if self.strategy_running and self.market_maker:
            self.add_log("检查策略状态...", "SYSTEM")
            
            # 检查WebSocket连接
            if not hasattr(self.market_maker, 'ws') or not self.market_maker.ws:
                self.add_log("诊断问题: WebSocket连接不可用", "ERROR")
            elif not getattr(self.market_maker.ws, '_thread', None) or not self.market_maker.ws._thread.is_alive():
                self.add_log("诊断问题: WebSocket线程未运行", "ERROR")
            else:
                self.add_log("WebSocket连接正常", "SYSTEM")
            
            # 检查订单簿数据
            if hasattr(self.market_maker, 'ws') and self.market_maker.ws:
                if not self.market_maker.ws.orderbook or (not self.market_maker.ws.orderbook.get('bids') and not self.market_maker.ws.orderbook.get('asks')):
                    self.add_log("诊断问题: 订单簿数据为空", "ERROR")
                else:
                    self.add_log("订单簿数据正常", "SYSTEM")
        
        self.add_log("诊断完成", "SYSTEM")
        self.add_log("如遇问题，请检查API密钥是否正确，网络连接是否正常", "SYSTEM")
        self.add_log("或者尝试重新启动程序", "SYSTEM")
    
    def start(self):
        """启动交互式面板"""
        # 设置初始日志
        self.add_log("做市交易面板已启动", "SYSTEM")
        self.add_log("按 : 或 / 进入命令模式", "SYSTEM")
        self.add_log("输入 help 查看可用命令", "SYSTEM")
        self.add_log("", "SYSTEM")
        
        # 显示当前市场类型和相关提示
        current_market_type = self.strategy_params.get('market_type', 'spot')
        self.add_log(f"当前市场类型: {current_market_type}", "SYSTEM")
        if current_market_type == 'spot':
            self.add_log("现货交易示例: start SOL_USDC", "SYSTEM")
            self.add_log("切换永续模式: set market_type perp", "SYSTEM")
        else:
            self.add_log("永续交易示例: start SOL_USDC_PERP", "SYSTEM")
            self.add_log("切换现货模式: set market_type spot", "SYSTEM")
        
        self.running = True
        
        # 创建并启动更新线程
        self.update_thread = threading.Thread(target=self.update_thread_function, daemon=True)
        self.update_thread.start()
        
        try:
            # 创建实时显示并处理按键
            with Live(self.layout, refresh_per_second=2, screen=True) as live:
                while self.running:
                    # 更新显示
                    self.update_display()
                    
                    # 等待按键输入
                    # 注意：这里需要在真实环境中实现键盘输入捕获
                    # 但在这个示例中，我们只是简单地休眠
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        # 停止策略（如果正在运行）
        if self.strategy_running and hasattr(self, '_cleanup_strategy'):
            self._cleanup_strategy()

# 主函数
def main():
    panel = InteractivePanel()
    panel.start()

if __name__ == "__main__":
    main()