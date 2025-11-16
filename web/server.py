"""
Flask Web服務器
提供簡單的Web界面來控制做市交易機器人
"""
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import os
import sys
import traceback
import time
import socket
from typing import Optional, Dict, Any
from datetime import datetime

# 添加父目錄到路徑以導入項目模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import setup_logger

logger = setup_logger("web_server")

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = os.urandom(24)

# 配置SocketIO以提高連接穩定性
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading',
    logger=False,
    engineio_logger=False
)

# 全局狀態
bot_status = {
    'running': False,
    'strategy': None,
    'start_time': None,
    'last_update': None,
    'stats': {}
}

# 存儲當前運行的策略實例
current_strategy: Optional[Any] = None
strategy_thread: Optional[threading.Thread] = None

# 統計數據更新線程
stats_update_thread: Optional[threading.Thread] = None
stats_update_running = False

# 保留的最後統計數據（停止時不清除，啟動時才清除）
last_stats: Dict[str, Any] = {}

# 餘額緩存（避免頻繁查詢）
balance_cache: Dict[str, Any] = {
    'data': None,
    'timestamp': 0,
    'update_interval': 60  # 默認60秒更新一次餘額
}


@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """獲取機器人狀態"""
    return jsonify(bot_status)


@app.route('/api/start', methods=['POST'])
def start_bot():
    """啟動做市機器人"""
    global current_strategy, strategy_thread, bot_status, last_stats, balance_cache

    if bot_status['running']:
        return jsonify({'success': False, 'message': '機器人已在運行中'}), 400

    # 啟動新機器人時清除舊的統計數據和餘額緩存
    last_stats = {}
    bot_status['stats'] = {}
    balance_cache['data'] = None
    balance_cache['timestamp'] = 0

    try:
        data = request.json
        logger.info(f"收到啟動請求: {data}")

        # 驗證必要參數
        required_fields = ['exchange', 'symbol', 'spread']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'success': False, 'message': f'缺少必要參數: {field}'}), 400

        # 從環境變量獲取API密鑰
        exchange = data['exchange']
        api_key = ''
        secret_key = ''
        account_address: Optional[str] = None
        ws_proxy = None
        exchange_config = {}

        if exchange == 'backpack':
            api_key = os.getenv('BACKPACK_KEY', '')
            secret_key = os.getenv('BACKPACK_SECRET', '')
            ws_proxy = os.getenv('BACKPACK_PROXY_WEBSOCKET')
            base_url = os.getenv('BASE_URL', 'https://api.backpack.work')
            exchange_config = {
                'api_key': api_key,
                'secret_key': secret_key,
                'base_url': base_url,
                'api_version': 'v1',
                'default_window': '5000'
            }
        elif exchange == 'aster':
            api_key = os.getenv('ASTER_API_KEY', '')
            secret_key = os.getenv('ASTER_SECRET_KEY', '')
            ws_proxy = os.getenv('ASTER_PROXY_WEBSOCKET')
            exchange_config = {
                'api_key': api_key,
                'secret_key': secret_key,
            }
        elif exchange == 'lighter':
            api_key = os.getenv('LIGHTER_PRIVATE_KEY') or os.getenv('LIGHTER_API_KEY')
            secret_key = os.getenv('LIGHTER_SECRET_KEY') or api_key
            ws_proxy = os.getenv('LIGHTER_PROXY_WEBSOCKET') or os.getenv('LIGHTER_WS_PROXY')
            base_url = os.getenv('LIGHTER_BASE_URL')
            account_index = os.getenv('LIGHTER_ACCOUNT_INDEX')
            account_address = os.getenv('LIGHTER_ADDRESS')
            if not account_index:
                from api.lighter_client import _get_lihgter_account_index
                account_index = _get_lihgter_account_index(account_address)
            api_key_index = os.getenv('LIGHTER_API_KEY_INDEX')
            chain_id = os.getenv('LIGHTER_CHAIN_ID')

            exchange_config = {
                'api_private_key': api_key,
                'account_index': account_index,
                'api_key_index': api_key_index,
                'base_url': base_url,
            }
            if chain_id is not None:
                exchange_config['chain_id'] = chain_id
            if not api_key:
                logger.error("缺少 Lighter 私鑰，請使用 --api-key 或環境變量 LIGHTER_PRIVATE_KEY 提供")
                sys.exit(1)
            if not exchange_config.get('account_index'):
                logger.error("缺少 Lighter Account Index，請透過環境變量 LIGHTER_ACCOUNT_INDEX 提供")
                
        elif exchange == 'paradex':
            private_key = os.getenv('PARADEX_PRIVATE_KEY', '')
            account_address = os.getenv('PARADEX_ACCOUNT_ADDRESS')
            ws_proxy = os.getenv('PARADEX_PROXY_WEBSOCKET')
            base_url = os.getenv('PARADEX_BASE_URL', 'https://api.prod.paradex.trade/v1')

            secret_key = private_key
            api_key = ''

            exchange_config = {
                'private_key': private_key,
                'account_address': account_address,
                'base_url': base_url,
            }
        else:
            return jsonify({'success': False, 'message': f'不支持的交易所: {exchange}'}), 400

        # 檢查API密鑰
        if exchange == 'paradex':
            if not secret_key or not account_address:
                return jsonify({'success': False, 'message': 'Paradex需要提供StarkNet私鑰與帳户地址'}), 400
        else:
            if not api_key or not secret_key:
                return jsonify({'success': False, 'message': 'API密鑰未配置，請檢查環境變量'}), 400

        # 獲取策略參數
        symbol = data['symbol']
        spread = float(data['spread'])
        quantity = float(data.get('quantity', 0)) if data.get('quantity') else None
        max_orders = int(data.get('max_orders', 3))
        duration = int(data.get('duration', 3600))
        interval = int(data.get('interval', 60))
        market_type = data.get('market_type', 'spot')
        strategy_name = data.get('strategy', 'standard')
        enable_db = data.get('enable_db', False)

        # 永續合約參數
        target_position = float(data.get('target_position', 1.0))
        max_position = float(data.get('max_position', 1.0))
        position_threshold = float(data.get('position_threshold', 0.1))
        inventory_skew = float(data.get('inventory_skew', 0.0))
        stop_loss = float(data['stop_loss']) if data.get('stop_loss') else None
        take_profit = float(data['take_profit']) if data.get('take_profit') else None

        # 現貨重平參數
        enable_rebalance = data.get('enable_rebalance', True)
        base_asset_target = float(data.get('base_asset_target', 30.0))
        rebalance_threshold = float(data.get('rebalance_threshold', 15.0))

        # 網格策略參數
        grid_upper_price = float(data['grid_upper_price']) if data.get('grid_upper_price') else None
        grid_lower_price = float(data['grid_lower_price']) if data.get('grid_lower_price') else None
        grid_num = int(data.get('grid_num', 10))
        grid_mode = data.get('grid_mode', 'arithmetic')
        auto_price_range = data.get('auto_price_range', False)
        price_range_percent = float(data.get('price_range_percent', 5.0))
        grid_type = data.get('grid_type', 'neutral')  # 永續合約網格類型

        # 創建策略實例
        if market_type == 'perp':
            from strategies.perp_market_maker import PerpetualMarketMaker
            from strategies.maker_taker_hedge import MakerTakerHedgeStrategy
            from strategies.perp_grid_strategy import PerpGridStrategy

            if strategy_name == 'grid':
                # 永續合約網格策略
                current_strategy = PerpGridStrategy(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=symbol,
                    grid_upper_price=grid_upper_price,
                    grid_lower_price=grid_lower_price,
                    grid_num=grid_num,
                    order_quantity=quantity,
                    auto_price_range=auto_price_range,
                    price_range_percent=price_range_percent,
                    grid_mode=grid_mode,
                    grid_type=grid_type,
                    target_position=target_position,
                    max_position=max_position,
                    position_threshold=position_threshold,
                    inventory_skew=inventory_skew,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    ws_proxy=ws_proxy,
                    exchange=exchange,
                    exchange_config=exchange_config,
                    enable_database=enable_db
                )
            elif strategy_name == 'maker_hedge':
                # 永續合約對沖策略
                current_strategy = MakerTakerHedgeStrategy(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=symbol,
                    base_spread_percentage=spread,
                    order_quantity=quantity,
                    target_position=target_position,
                    max_position=max_position,
                    position_threshold=position_threshold,
                    inventory_skew=inventory_skew,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    ws_proxy=ws_proxy,
                    exchange=exchange,
                    exchange_config=exchange_config,
                    enable_database=enable_db,
                    market_type='perp'
                )
            else:
                # 永續合約標準策略
                current_strategy = PerpetualMarketMaker(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=symbol,
                    base_spread_percentage=spread,
                    order_quantity=quantity,
                    max_orders=max_orders,
                    target_position=target_position,
                    max_position=max_position,
                    position_threshold=position_threshold,
                    inventory_skew=inventory_skew,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    ws_proxy=ws_proxy,
                    exchange=exchange,
                    exchange_config=exchange_config,
                    enable_database=enable_db
                )
        else:
            from strategies.market_maker import MarketMaker
            from strategies.maker_taker_hedge import MakerTakerHedgeStrategy
            from strategies.grid_strategy import GridStrategy

            if strategy_name == 'grid':
                # 現貨網格策略
                current_strategy = GridStrategy(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=symbol,
                    grid_upper_price=grid_upper_price,
                    grid_lower_price=grid_lower_price,
                    grid_num=grid_num,
                    order_quantity=quantity,
                    auto_price_range=auto_price_range,
                    price_range_percent=price_range_percent,
                    grid_mode=grid_mode,
                    ws_proxy=ws_proxy,
                    exchange=exchange,
                    exchange_config=exchange_config,
                    enable_database=enable_db
                )
            elif strategy_name == 'maker_hedge':
                # 現貨對沖策略
                current_strategy = MakerTakerHedgeStrategy(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=symbol,
                    base_spread_percentage=spread,
                    order_quantity=quantity,
                    ws_proxy=ws_proxy,
                    exchange=exchange,
                    exchange_config=exchange_config,
                    enable_database=enable_db,
                    market_type='spot'
                )
            else:
                # 現貨標準策略
                current_strategy = MarketMaker(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=symbol,
                    base_spread_percentage=spread,
                    order_quantity=quantity,
                    max_orders=max_orders,
                    enable_rebalance=enable_rebalance,
                    base_asset_target_percentage=base_asset_target,
                    rebalance_threshold=rebalance_threshold,
                    ws_proxy=ws_proxy,
                    exchange=exchange,
                    exchange_config=exchange_config,
                    enable_database=enable_db
                )

        # 在新線程中運行策略
        def run_strategy():
            global bot_status, current_strategy, strategy_thread
            try:
                bot_status['running'] = True
                bot_status['start_time'] = datetime.now().isoformat()
                socketio.emit('status_update', {'status': 'running', 'message': '機器人已啟動'})

                # 啟動統計數據更新線程
                start_stats_update()

                current_strategy.run(duration_seconds=duration, interval_seconds=interval)

            except Exception as e:
                logger.error(f"策略運行錯誤: {e}")
                socketio.emit('error', {'message': f'策略運行錯誤: {str(e)}'})
            finally:
                # 停止統計數據更新線程
                stop_stats_update()

                # 清空策略實例和線程引用
                current_strategy = None
                strategy_thread = None

                bot_status['running'] = False
                # 不清除統計數據，保留最後的狀態
                # bot_status['stats'] = {}  # 註釋掉這行
                socketio.emit('status_update', {'status': 'stopped', 'message': '機器人已停止'})
                logger.info("策略線程已完全結束，資源已清理")

        strategy_thread = threading.Thread(target=run_strategy)
        strategy_thread.daemon = False  # 改為非守護線程，確保能正常停止
        strategy_thread.start()

        return jsonify({
            'success': True,
            'message': '機器人啟動成功',
            'config': {
                'exchange': exchange,
                'symbol': symbol,
                'spread': spread,
                'market_type': market_type,
                'strategy': strategy_name
            }
        })

    except Exception as e:
        logger.error(f"啟動機器人失敗: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'啟動失敗: {str(e)}'}), 500


@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """停止做市機器人"""
    global current_strategy, bot_status, strategy_thread

    if not bot_status['running']:
        return jsonify({'success': False, 'message': '機器人未在運行'}), 400

    try:
        logger.info("收到停止請求，正在停止機器人...")

        # 保存線程引用，防止在等待期間被清空
        thread_to_wait = strategy_thread

        if current_strategy:
            # 設置停止標誌
            current_strategy.stop()
            logger.info("已設置停止標誌")

            # 等待策略線程結束（最多等待60秒）
            if thread_to_wait and thread_to_wait.is_alive():
                logger.info("等待策略線程結束...")
                thread_to_wait.join(timeout=60)

                if thread_to_wait.is_alive():
                    logger.warning("策略線程未在60秒內結束，強制標記為停止")
                else:
                    logger.info("策略線程已正常結束")

        # 停止統計數據更新
        stop_stats_update()

        # 清空策略實例和線程引用，防止繼續運行
        current_strategy = None
        strategy_thread = None

        bot_status['running'] = False
        # 不清除統計數據，保留最後的狀態
        # bot_status['stats'] = {}  # 註釋掉這行
        socketio.emit('status_update', {'status': 'stopped', 'message': '機器人已停止'})

        logger.info("機器人已成功停止，策略實例已清空")
        return jsonify({'success': True, 'message': '機器人停止成功'})

    except Exception as e:
        logger.error(f"停止機器人失敗: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'停止失敗: {str(e)}'}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """獲取配置信息"""
    return jsonify({
        'exchanges': ['backpack', 'aster', 'paradex'],
        'market_types': ['spot', 'perp'],
        'strategies': ['standard', 'maker_hedge', 'grid'],
        'env_configured': {
            'backpack': bool(os.getenv('BACKPACK_KEY') and os.getenv('BACKPACK_SECRET')),
            'aster': bool(os.getenv('ASTER_API_KEY') and os.getenv('ASTER_SECRET_KEY')),
            'paradex': bool(os.getenv('PARADEX_PRIVATE_KEY') and os.getenv('PARADEX_ACCOUNT_ADDRESS')),
            'lighter': bool(os.getenv('LIGHTER_PRIVATE_KEY') and os.getenv('LIGHTER_PUBLIC_KEY'))
        }
    })


@socketio.on('connect')
def handle_connect():
    """WebSocket連接建立"""
    logger.info("客户端已連接")
    emit('connected', {'message': '已連接到服務器'})
    # 如果機器人正在運行，發送當前狀態
    if bot_status['running']:
        emit('status_update', {'status': 'running', 'message': '機器人運行中'})


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket連接斷開"""
    logger.info("客户端已斷開連接")


@socketio.on('ping')
def handle_ping():
    """處理心跳ping"""
    emit('pong')


def collect_strategy_stats():
    """收集策略統計數據"""
    global current_strategy, last_stats, balance_cache

    if not current_strategy:
        # 返回保留的最後統計數據
        return last_stats.copy() if last_stats else {}

    try:
        stats = {
            'symbol': current_strategy.symbol,
            'exchange': current_strategy.exchange,
            'base_asset': current_strategy.base_asset,
            'quote_asset': current_strategy.quote_asset,
        }

        # 獲取餘額 - 使用緩存機制，根據策略的更新週期來控制
        current_time = time.time()
        should_update_balance = False
        
        # 如果有緩存且還在有效期內，使用緩存
        if balance_cache['data'] is not None:
            elapsed = current_time - balance_cache['timestamp']
            
            # 獲取策略的更新間隔（如果有的話）
            strategy_interval = getattr(current_strategy, 'interval_seconds', 60)
            balance_cache['update_interval'] = max(strategy_interval, 30)  # 最少30秒
            
            if elapsed < balance_cache['update_interval']:
                # 使用緩存的餘額數據
                stats.update(balance_cache['data'])
                logger.debug(f"使用緩存的餘額數據（{elapsed:.1f}s 前更新，間隔{balance_cache['update_interval']}s）")
            else:
                should_update_balance = True
        else:
            # 首次查詢
            should_update_balance = True
        
        # 需要更新餘額
        if should_update_balance:
            try:
                # 獲取客户端實例
                client = current_strategy.client if hasattr(current_strategy, 'client') else None

                balance_data = {}

                if client:
                    # 獲取普通餘額
                    balances = client.get_balance()

                    # 初始化報價資產餘額變量
                    quote_balance = 0.0
                    quote_collateral_balance = 0.0

                    # 檢查是否有錯誤
                    has_error = isinstance(balances, dict) and "error" in balances and balances.get("error")

                    if not has_error and isinstance(balances, dict):
                        # 確定報價資產的鍵名（處理不同交易所的命名差異）
                        quote_asset_key = current_strategy.quote_asset

                        # Paradex 特殊處理：將 USD 映射到 USDC
                        if current_strategy.exchange.lower() == 'paradex':
                            if quote_asset_key == 'USD' and 'USDC' in balances:
                                quote_asset_key = 'USDC'

                        # Lighter 特殊處理：統一使用 USDC（因為返回了別名）
                        if current_strategy.exchange.lower() == 'lighter':
                            if 'USDC' in balances:
                                quote_asset_key = 'USDC'

                        # 嘗試多個可能的報價資產名稱（USDT, USDC, USD）
                        possible_quote_keys = [quote_asset_key]
                        if quote_asset_key not in ['USDT', 'USDC', 'USD']:
                            possible_quote_keys.extend(['USDC', 'USDT', 'USD'])

                        # 獲取普通餘額中的報價資產
                        for key in possible_quote_keys:
                            if key in balances:
                                quote_info = balances[key]
                                try:
                                    if isinstance(quote_info, dict):
                                        # 字典格式：{available: "xxx", locked: "xxx", total: "xxx"}
                                        # 優先使用 available，其次 total
                                        available_value = quote_info.get('available', quote_info.get('total', 0))
                                        if available_value not in (None, ''):
                                            quote_balance = float(available_value)
                                        else:
                                            quote_balance = 0.0
                                    else:
                                        # 直接是數值
                                        if quote_info not in (None, ''):
                                            quote_balance = float(quote_info)
                                        else:
                                            quote_balance = 0.0

                                    break  # 找到就退出循環
                                except (ValueError, TypeError) as e:
                                    logger.error(f"轉換報價資產餘額失敗: {e}")
                                    continue
                        else:
                            quote_balance = 0.0

                        # 對於 Backpack，還需要獲取抵押品餘額
                        if current_strategy.exchange.lower() == 'backpack':
                            try:
                                collateral = client.get_collateral()
                                if isinstance(collateral, dict) and "error" in collateral:
                                    logger.warning(f"獲取 Backpack 抵押品餘額失敗: {collateral.get('error')}")
                                elif isinstance(collateral, dict):
                                    collateral_assets = collateral.get('assets') or collateral.get('collateral', [])

                                    if collateral_assets:
                                        # 遍歷抵押品資產，查找報價資產
                                        for item in collateral_assets:
                                            symbol = item.get('symbol', '')
                                            if symbol in possible_quote_keys:
                                                try:
                                                    # 使用 totalQuantity（包含借貸中的資產）
                                                    total_quantity = float(item.get('totalQuantity', 0))
                                                    if total_quantity > 0:
                                                        quote_collateral_balance += total_quantity
                                                except (ValueError, TypeError) as e:
                                                    logger.error(f"轉換抵押品餘額失敗: {e}")
                            except Exception as e:
                                logger.warning(f"獲取 Backpack 抵押品餘額時出錯: {e}")

                    else:
                        if has_error:
                            logger.error(f"[{current_strategy.exchange}] 獲取餘額返回錯誤: {balances.get('error')}")
                        else:
                            logger.error(f"[{current_strategy.exchange}] 獲取餘額返回格式不正確: type={type(balances)}")

                    # 計算總餘額（普通餘額 + 抵押品餘額）
                    total_quote_balance = quote_balance + quote_collateral_balance

                    logger.info(f"[{current_strategy.exchange}] 報價資產總餘額: 普通={quote_balance:.2f}, 抵押品={quote_collateral_balance:.2f}, 總計={total_quote_balance:.2f}")

                    # 設置統計數據
                    balance_data['base_balance'] = 0.0  # 不再顯示基礎資產
                    balance_data['quote_balance'] = round(total_quote_balance, 2)
                    balance_data['total_balance_usd'] = round(total_quote_balance, 2)  # 總餘額包含普通和抵押品
                else:
                    # 如果沒有客户端，使用原有方法獲取報價資產
                    quote_balance_result = current_strategy.get_asset_balance(current_strategy.quote_asset)

                    # 處理可能返回元組的情況
                    quote_balance = quote_balance_result[0] if isinstance(quote_balance_result, tuple) else quote_balance_result

                    balance_data['base_balance'] = 0.0  # 不再顯示基礎資產
                    balance_data['quote_balance'] = round(float(quote_balance), 2)
                    balance_data['total_balance_usd'] = round(float(quote_balance), 2)

                # 更新緩存
                balance_cache['data'] = balance_data
                balance_cache['timestamp'] = current_time
                
                # 應用到統計數據
                stats.update(balance_data)
                
            except Exception as e:
                logger.error(f"獲取餘額時出錯: {e}")
                traceback.print_exc()
                balance_data = {
                    'base_balance': 0.0,
                    'quote_balance': 0.0,
                    'total_balance_usd': 0.0
                }
                balance_cache['data'] = balance_data
                balance_cache['timestamp'] = current_time
                stats.update(balance_data)

        # 獲取當前價格
        try:
            stats['current_price'] = round(current_strategy.get_current_price(), 8)
        except Exception:
            stats['current_price'] = 0

        # 獲取交易統計
        stats['total_trades'] = len(current_strategy.session_buy_trades) + len(current_strategy.session_sell_trades)
        stats['buy_trades'] = len(current_strategy.session_buy_trades)
        stats['sell_trades'] = len(current_strategy.session_sell_trades)

        # 計算成交量
        stats['maker_buy_volume'] = round(current_strategy.session_maker_buy_volume, 4)
        stats['maker_sell_volume'] = round(current_strategy.session_maker_sell_volume, 4)
        stats['taker_buy_volume'] = round(current_strategy.session_taker_buy_volume, 4)
        stats['taker_sell_volume'] = round(current_strategy.session_taker_sell_volume, 4)

        # 計算手續費
        stats['total_fees'] = round(current_strategy.session_fees, 4)

        # 計算成交額 (USDC)
        try:
            # 從交易記錄計算總成交額
            total_volume_usdc = 0.0
            current_price = 0.0

            try:
                current_price = float(current_strategy.get_current_price())
            except:
                pass

            buy_count = len(current_strategy.session_buy_trades)
            sell_count = len(current_strategy.session_sell_trades)

            # 計算買單成交額
            for trade in current_strategy.session_buy_trades:
                try:
                    if isinstance(trade, dict):
                        qty = float(trade.get('quantity', trade.get('qty', 0)))
                        price = float(trade.get('price', current_price))
                        volume = qty * price
                        total_volume_usdc += volume
                    elif isinstance(trade, (list, tuple)) and len(trade) >= 2:
                        # 處理列表/元組格式 (price, qty)
                        price = float(trade[0])
                        qty = float(trade[1])
                        total_volume_usdc += qty * price
                except Exception:
                    continue

            # 計算賣單成交額
            for trade in current_strategy.session_sell_trades:
                try:
                    if isinstance(trade, dict):
                        qty = float(trade.get('quantity', trade.get('qty', 0)))
                        price = float(trade.get('price', current_price))
                        volume = qty * price
                        total_volume_usdc += volume
                    elif isinstance(trade, (list, tuple)) and len(trade) >= 2:
                        # 處理列表/元組格式
                        price = float(trade[0])
                        qty = float(trade[1])
                        total_volume_usdc += qty * price
                except Exception:
                    continue

            stats['total_volume_usdc'] = round(total_volume_usdc, 2)

        except Exception as e:
            logger.error(f"計算成交額失敗: {e}")
            traceback.print_exc()
            stats['total_volume_usdc'] = 0.0
            stats['slippage_rate'] = 0.0

        # 計算盈虧
        try:
            pnl_result = current_strategy.calculate_pnl()

            # 處理不同的返回格式
            if isinstance(pnl_result, tuple):
                if len(pnl_result) == 3:
                    realized_pnl, unrealized_pnl, session_realized_pnl = pnl_result
                elif len(pnl_result) == 2:
                    realized_pnl, unrealized_pnl = pnl_result
                    session_realized_pnl = 0
                elif len(pnl_result) >= 7:
                    # 7個返回值的情況，根據常見格式解析
                    # 通常格式: (realized, unrealized, session, ...)
                    realized_pnl = pnl_result[0]
                    unrealized_pnl = pnl_result[1]
                    session_realized_pnl = pnl_result[2]
                    logger.debug(f"PnL返回7個值，使用前3個: {pnl_result[:3]}")
                else:
                    # 其他情況，嘗試使用前面的值
                    logger.error(f"PnL返回值數量異常: {len(pnl_result)} 個值")
                    realized_pnl = pnl_result[0] if len(pnl_result) > 0 else 0
                    unrealized_pnl = pnl_result[1] if len(pnl_result) > 1 else 0
                    session_realized_pnl = pnl_result[2] if len(pnl_result) > 2 else 0
            elif isinstance(pnl_result, dict):
                # 處理dict返回
                realized_pnl = pnl_result.get('realized_pnl', 0)
                unrealized_pnl = pnl_result.get('unrealized_pnl', 0)
                session_realized_pnl = pnl_result.get('session_realized_pnl', 0)
            else:
                # 單一返回值
                realized_pnl = float(pnl_result)
                unrealized_pnl = 0
                session_realized_pnl = 0

            stats['realized_pnl'] = round(float(realized_pnl), 4)
            stats['unrealized_pnl'] = round(float(unrealized_pnl), 4)
            stats['session_realized_pnl'] = round(float(session_realized_pnl), 4)
            stats['total_pnl'] = round(float(realized_pnl) + float(unrealized_pnl), 4)

            total_fees = stats.get('total_fees', 0)
            stats['cumulative_pnl'] = round(float(realized_pnl) + (-abs(float(total_fees))), 4)

        except Exception as e:
            logger.error(f"計算盈虧失敗: {e}")
            traceback.print_exc()
            stats['realized_pnl'] = 0
            stats['unrealized_pnl'] = 0
            stats['session_realized_pnl'] = 0
            stats['total_pnl'] = 0
            stats['cumulative_pnl'] = 0

        # 計算磨損率 (累計盈虧 / 成交額 * 100%)
        try:
            if stats.get('total_volume_usdc', 0) > 0 and stats.get('cumulative_pnl') is not None:
                # 磨損率 = 累計盈虧 / 成交額 * 100%
                slippage_rate = (stats['cumulative_pnl'] / stats['total_volume_usdc']) * 100
                stats['slippage_rate'] = round(slippage_rate, 4)
            else:
                stats['slippage_rate'] = 0.0
        except Exception as e:
            logger.error(f"計算磨損率失敗: {e}")
            stats['slippage_rate'] = 0.0

        # 計算運行時間
        if current_strategy.session_start_time:
            elapsed = datetime.now() - current_strategy.session_start_time
            stats['runtime_seconds'] = int(elapsed.total_seconds())
            stats['runtime_formatted'] = str(elapsed).split('.')[0]
        else:
            stats['runtime_seconds'] = 0
            stats['runtime_formatted'] = '00:00:00'

        # 網格策略特有的統計數據
        if hasattr(current_strategy, 'grid_levels'):
            stats['grid_profit'] = stats.get('realized_pnl', 0)
            stats['grid_buy_filled'] = len(current_strategy.session_buy_trades)
            stats['grid_sell_filled'] = len(current_strategy.session_sell_trades)
            stats['grid_count'] = len(current_strategy.grid_levels)

            # 添加網格價格區間
            if hasattr(current_strategy, 'grid_upper_price') and hasattr(current_strategy, 'grid_lower_price'):
                stats['grid_upper_price'] = current_strategy.grid_upper_price
                stats['grid_lower_price'] = current_strategy.grid_lower_price

            if hasattr(current_strategy, 'grid_orders_by_id'):
                stats['active_grid_orders'] = len(current_strategy.grid_orders_by_id)

        return stats

    except Exception as e:
        logger.error(f"收集統計數據失敗: {e}")
        traceback.print_exc()
        return {}


def stats_update_worker():
    """統計數據更新工作線程"""
    global stats_update_running, current_strategy, last_stats

    while stats_update_running:
        try:
            stats = collect_strategy_stats()
            if stats:
                # 保存最後的統計數據
                last_stats = stats.copy()
                socketio.emit('stats_update', stats)
                bot_status['stats'] = stats
                bot_status['last_update'] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"更新統計數據失敗: {e}")

        # 每5秒更新一次
        time.sleep(5)


def start_stats_update():
    """啟動統計數據更新線程"""
    global stats_update_thread, stats_update_running

    if stats_update_thread and stats_update_thread.is_alive():
        return

    stats_update_running = True
    stats_update_thread = threading.Thread(target=stats_update_worker, daemon=True)
    stats_update_thread.start()
    logger.info("統計數據更新線程已啟動")


def stop_stats_update():
    """停止統計數據更新線程"""
    global stats_update_running, last_stats

    stats_update_running = False

    # 發送最後一次統計數據
    if last_stats:
        socketio.emit('stats_update', last_stats)

    logger.info("統計數據更新線程已停止，最後數據已發送")


def broadcast_status_update(data: Dict[str, Any]):
    """廣播狀態更新到所有連接的客户端"""
    global bot_status
    bot_status['last_update'] = datetime.now().isoformat()
    bot_status['stats'] = data
    socketio.emit('status_update', data)


def is_port_available(host: str, port: int) -> bool:
    """檢查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except (OSError, socket.error) as e:
        logger.debug(f"端口 {port} 不可用: {e}")
        return False


def find_available_port(host: str, start_port: int = 5001, end_port: int = 6000) -> Optional[int]:
    """在指定範圍內查找可用端口"""
    for port in range(start_port, end_port + 1):
        if is_port_available(host, port):
            return port
    return None


def run_server(host='0.0.0.0', port=5000, debug=False):
    """運行Web服務器"""
    # 從 .env 讀取配置
    web_host = os.getenv('WEB_HOST', '127.0.0.1')
    web_port = int(os.getenv('WEB_PORT', '5000'))
    web_debug = os.getenv('WEB_DEBUG', 'false').lower() in ('true', '1', 'yes')

    # 使用環境變量的配置（如果有的話）
    host = web_host if web_host else host
    port = web_port if web_port else port
    debug = web_debug

    # 檢查端口是否可用
    if not is_port_available(host, port):
        logger.warning(f"端口 {port} 已被占用，正在尋找可用端口...")
        new_port = find_available_port(host, 5001, 6000)

        if new_port:
            logger.info(f"找到可用端口: {new_port}")
            port = new_port
        else:
            logger.error("無法在 5001-6000 範圍內找到可用端口，服務器啟動失敗")
            return

    logger.info(f"啟動Web服務器於 http://{host}:{port}")
    logger.info(f"調試模式: {'開啟' if debug else '關閉'}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server()
