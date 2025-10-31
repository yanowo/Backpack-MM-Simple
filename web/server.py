"""
Flask Web服務器
提供簡單的Web界面來控制做市交易機器人
"""
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import os
import sys
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

# 配置SocketIO以提高连接稳定性
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
    global current_strategy, strategy_thread, bot_status, last_stats

    if bot_status['running']:
        return jsonify({'success': False, 'message': '機器人已在運行中'}), 400

    # 啟動新機器人時清除舊的統計數據
    last_stats = {}
    bot_status['stats'] = {}

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
                return jsonify({'success': False, 'message': 'Paradex需要提供StarkNet私鑰與帳戶地址'}), 400
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

        # 創建策略實例
        if market_type == 'perp':
            from strategies.perp_market_maker import PerpetualMarketMaker
            from strategies.maker_taker_hedge import MakerTakerHedgeStrategy

            if strategy_name == 'maker_hedge':
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

            if strategy_name == 'maker_hedge':
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
        import traceback
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
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'停止失敗: {str(e)}'}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """獲取配置信息"""
    return jsonify({
        'exchanges': ['backpack', 'aster', 'paradex'],
        'market_types': ['spot', 'perp'],
        'strategies': ['standard', 'maker_hedge'],
        'env_configured': {
            'backpack': bool(os.getenv('BACKPACK_KEY') and os.getenv('BACKPACK_SECRET')),
            'aster': bool(os.getenv('ASTER_API_KEY') and os.getenv('ASTER_SECRET_KEY')),
            'paradex': bool(os.getenv('PARADEX_PRIVATE_KEY') and os.getenv('PARADEX_ACCOUNT_ADDRESS'))
        }
    })


@socketio.on('connect')
def handle_connect():
    """WebSocket連接建立"""
    logger.info("客戶端已連接")
    emit('connected', {'message': '已連接到服務器'})
    # 如果機器人正在運行，發送當前狀態
    if bot_status['running']:
        emit('status_update', {'status': 'running', 'message': '機器人運行中'})


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket連接斷開"""
    logger.info("客戶端已斷開連接")


@socketio.on('ping')
def handle_ping():
    """處理心跳ping"""
    emit('pong')


def collect_strategy_stats():
    """收集策略統計數據"""
    global current_strategy, last_stats

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

        # 獲取餘額
        try:
            base_balance_result = current_strategy.get_asset_balance(current_strategy.base_asset)
            quote_balance_result = current_strategy.get_asset_balance(current_strategy.quote_asset)

            # 處理可能返回元組的情況
            base_balance = base_balance_result[0] if isinstance(base_balance_result, tuple) else base_balance_result
            quote_balance = quote_balance_result[0] if isinstance(quote_balance_result, tuple) else quote_balance_result

            stats['base_balance'] = round(float(base_balance), 8)
            stats['quote_balance'] = round(float(quote_balance), 8)

            total_balance_result = current_strategy.get_total_balance()

            # 處理多種返回類型：dict, tuple, float, None
            if total_balance_result is None:
                total_balance = 0
            elif isinstance(total_balance_result, dict):
                # 如果返回dict，嘗試從中提取數值
                total_balance = total_balance_result.get('total_balance_usd', 0)
                if total_balance == 0:
                    # 嘗試其他可能的鍵名
                    total_balance = total_balance_result.get('total', 0)
                    if total_balance == 0:
                        total_balance = total_balance_result.get('balance', 0)
            elif isinstance(total_balance_result, tuple):
                total_balance = total_balance_result[0] if total_balance_result[0] is not None else 0
            else:
                total_balance = total_balance_result if total_balance_result is not None else 0

            stats['total_balance_usd'] = round(float(total_balance), 2)
        except Exception as e:
            logger.error(f"獲取餘額失敗: {e}")
            import traceback
            traceback.print_exc()
            stats['base_balance'] = 0
            stats['quote_balance'] = 0
            stats['total_balance_usd'] = 0

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
                except Exception as e:
                    logger.debug(f"處理買單交易失敗: {e}")
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
                except Exception as e:
                    logger.debug(f"處理賣單交易失敗: {e}")
                    continue

            stats['total_volume_usdc'] = round(total_volume_usdc, 2)

        except Exception as e:
            logger.error(f"計算成交額失敗: {e}")
            import traceback
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
                    logger.info(f"PnL返回7個值，使用前3個: {pnl_result[:3]}")
                else:
                    # 其他情況，嘗試使用前面的值
                    logger.warning(f"PnL返回值數量異常: {len(pnl_result)} 個值，嘗試使用前面的值")
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
            import traceback
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

            # 添加調試日志
            if stats.get('total_trades', 0) > 0:
                logger.debug(f"成交統計: 總交易{stats['total_trades']}筆, 成交額${stats.get('total_volume_usdc', 0):.2f}, 累計盈虧{stats.get('cumulative_pnl', 0):.4f}, 磨損率{stats['slippage_rate']:.4f}%")
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

        return stats

    except Exception as e:
        logger.error(f"收集統計數據失敗: {e}")
        import traceback
        traceback.print_exc()
        return {}


def stats_update_worker():
    """統計數據更新工作線程"""
    global stats_update_running, current_strategy, last_stats

    import time

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
    """廣播狀態更新到所有連接的客戶端"""
    global bot_status
    bot_status['last_update'] = datetime.now().isoformat()
    bot_status['stats'] = data
    socketio.emit('status_update', data)


def run_server(host='0.0.0.0', port=5000, debug=False):
    """運行Web服務器"""
    logger.info(f"啟動Web服務器於 http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(debug=True)
