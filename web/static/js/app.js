// WebSocket連接
let socket;
let isConnected = false;
let botRunning = false;
let heartbeatInterval = null;

// DOM元素
const form = document.getElementById('botConfigForm');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const connectionStatus = document.getElementById('connectionStatus');
const botStatus = document.getElementById('botStatus');
const currentConfig = document.getElementById('currentConfig');
const logDisplay = document.getElementById('logDisplay');
const marketTypeSelect = document.getElementById('market_type');
const spotParams = document.getElementById('spotParams');
const perpParams = document.getElementById('perpParams');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    setupEventListeners();
    loadConfig();
});

// 初始化WebSocket連接
function initWebSocket() {
    socket = io({
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: Infinity,
        transports: ['polling', 'websocket'],
        upgrade: true
    });

    socket.on('connect', () => {
        isConnected = true;
        updateConnectionStatus('已連接', true);
        addLog('已連接到服務器', 'success');
        loadStatus();
        startHeartbeat();
    });

    socket.on('disconnect', () => {
        isConnected = false;
        updateConnectionStatus('連接斷開', false);
        addLog('與服務器連接已斷開', 'error');
        stopHeartbeat();
    });

    socket.on('connected', (data) => {
        addLog(data.message, 'info');
    });

    socket.on('status_update', (data) => {
        handleStatusUpdate(data);
    });

    socket.on('stats_update', (data) => {
        updateStatsDisplay(data);
    });

    socket.on('error', (data) => {
        addLog(`錯誤: ${data.message}`, 'error');
    });

    socket.on('pong', () => {
        // 收到心跳響應
    });

    socket.on('reconnect', (attemptNumber) => {
        addLog(`重新連接成功 (嘗試 ${attemptNumber} 次)`, 'success');
    });

    socket.on('reconnect_error', (error) => {
        addLog('重新連接失敗', 'error');
    });
}

// 啟動心跳
function startHeartbeat() {
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
    }

    heartbeatInterval = setInterval(() => {
        if (socket && socket.connected) {
            socket.emit('ping');
        }
    }, 20000); // 每20秒發送一次心跳
}

// 停止心跳
function stopHeartbeat() {
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = null;
    }
}

// 設置事件監聽器
function setupEventListeners() {
    // 表單提交
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await startBot();
    });

    // 停止按鈕
    stopBtn.addEventListener('click', async () => {
        await stopBot();
    });

    // 市場類型切換
    marketTypeSelect.addEventListener('change', () => {
        toggleMarketTypeParams();
    });
}

// 切換市場類型參數顯示
function toggleMarketTypeParams() {
    const marketType = marketTypeSelect.value;
    if (marketType === 'spot') {
        spotParams.style.display = 'block';
        perpParams.style.display = 'none';
    } else {
        spotParams.style.display = 'none';
        perpParams.style.display = 'block';
    }
}

// 加載配置信息
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            const config = await response.json();
            updateConfigUI(config);
        }
    } catch (error) {
        console.error('加載配置失敗:', error);
        addLog('加載配置失敗', 'error');
    }
}

// 更新配置UI
function updateConfigUI(config) {
    // 可以根據配置信息更新UI，例如顯示哪些交易所已配置
    const envConfigured = config.env_configured;
    const exchangeSelect = document.getElementById('exchange');

    Array.from(exchangeSelect.options).forEach(option => {
        const exchange = option.value;
        if (envConfigured[exchange]) {
            option.textContent += ' ✓';
        } else {
            option.textContent += ' (未配置)';
        }
    });
}

// 加載狀態
async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        if (response.ok) {
            const status = await response.json();
            updateBotStatus(status);
        }
    } catch (error) {
        console.error('加載狀態失敗:', error);
    }
}

// 啟動機器人
async function startBot() {
    if (!isConnected) {
        addLog('未連接到服務器', 'error');
        return;
    }

    if (botRunning) {
        addLog('機器人已在運行中', 'warning');
        return;
    }

    // 收集表單數據
    const formData = new FormData(form);
    const data = {
        exchange: formData.get('exchange'),
        symbol: formData.get('symbol'),
        spread: parseFloat(formData.get('spread')),
        quantity: formData.get('quantity') ? parseFloat(formData.get('quantity')) : null,
        market_type: formData.get('market_type'),
        strategy: formData.get('strategy'),
        duration: parseInt(formData.get('duration')),
        interval: parseInt(formData.get('interval')),
        enable_db: formData.get('enable_db') === 'on'
    };

    // 根據市場類型添加額外參數
    if (data.market_type === 'spot') {
        data.max_orders = parseInt(document.getElementById('max_orders').value);
        data.enable_rebalance = formData.get('enable_rebalance') === 'on';
        data.base_asset_target = parseFloat(formData.get('base_asset_target'));
        data.rebalance_threshold = parseFloat(formData.get('rebalance_threshold'));
    } else {
        data.max_orders = parseInt(document.getElementById('max_orders_perp').value);
        data.target_position = parseFloat(formData.get('target_position'));
        data.max_position = parseFloat(formData.get('max_position'));
        data.position_threshold = parseFloat(formData.get('position_threshold'));
        data.inventory_skew = parseFloat(formData.get('inventory_skew'));
        data.stop_loss = formData.get('stop_loss') ? parseFloat(formData.get('stop_loss')) : null;
        data.take_profit = formData.get('take_profit') ? parseFloat(formData.get('take_profit')) : null;
    }

    // 顯示加載狀態
    startBtn.disabled = true;
    startBtn.innerHTML = '<span class="loading"></span> <span>啟動中...</span>';
    addLog('正在啟動機器人...', 'info');

    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            addLog('機器人啟動成功', 'success');
            botRunning = true;
            updateUIForRunningState(true, result.config);
        } else {
            addLog(`啟動失敗: ${result.message}`, 'error');
            startBtn.disabled = false;
            startBtn.innerHTML = '<span>啟動機器人</span>';
        }
    } catch (error) {
        console.error('啟動失敗:', error);
        addLog(`啟動失敗: ${error.message}`, 'error');
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>啟動機器人</span>';
    }
}

// 停止機器人
async function stopBot() {
    if (!botRunning) {
        addLog('機器人未在運行', 'warning');
        return;
    }

    stopBtn.disabled = true;
    stopBtn.innerHTML = '<span class="loading"></span> <span>停止中...</span>';
    addLog('發送停止信號...', 'info');
    addLog('請等待當前循環完成（最多60秒）', 'info');

    try {
        const response = await fetch('/api/stop', {
            method: 'POST'
        });

        const result = await response.json();

        if (result.success) {
            addLog('機器人已成功停止', 'success');
            addLog('統計數據已保留，可查看最終結果', 'info');
            botRunning = false;
            // 不調用updateUIForRunningState(false)，因為在handleStatusUpdate中處理
        } else {
            addLog(`停止失敗: ${result.message}`, 'error');
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<span>停止機器人</span>';
        }
    } catch (error) {
        console.error('停止失敗:', error);
        addLog(`停止失敗: ${error.message}`, 'error');
        stopBtn.disabled = false;
        stopBtn.innerHTML = '<span>停止機器人</span>';
    }
}

// 更新連接狀態
function updateConnectionStatus(text, connected) {
    statusText.textContent = text;
    connectionStatus.textContent = text;

    if (connected) {
        statusDot.classList.add('connected');
        statusDot.classList.remove('running');
    } else {
        statusDot.classList.remove('connected');
        statusDot.classList.remove('running');
    }
}

// 更新機器人狀態
function updateBotStatus(status) {
    if (status.running) {
        botRunning = true;
        updateUIForRunningState(true, status.strategy);
        botStatus.textContent = '運行中';
    } else {
        botRunning = false;
        updateUIForRunningState(false);
        botStatus.textContent = '未啟動';
    }
}

// 處理狀態更新
function handleStatusUpdate(data) {
    if (data.status === 'running') {
        statusDot.classList.add('running');
        addLog(data.message || '機器人運行中', 'info');
    } else if (data.status === 'stopped') {
        statusDot.classList.remove('running');
        botRunning = false;

        // 更新按鈕狀態
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>啟動機器人</span>';
        stopBtn.disabled = true;
        stopBtn.innerHTML = '<span>停止機器人</span>'; 

        // 啟用表單輸入
        Array.from(form.elements).forEach(element => {
            element.disabled = false;
        });

        botStatus.textContent = '已停止（數據保留）';
        addLog(data.message || '機器人已停止，數據已保留', 'warning');

        // 不隱藏統計面板，保留數據顯示
    }

    // 更新統計信息
    if (data.stats) {
        updateStatsDisplay(data.stats);
    }
}

// 更新統計數據顯示
function updateStatsDisplay(stats) {
    const statsSection = document.getElementById('statsSection');

    if (!stats || Object.keys(stats).length === 0) {
        statsSection.style.display = 'none';
        return;
    }

    statsSection.style.display = 'block';

    // 更新運行時間（頭部）
    updateStatValue('statRuntimeHeader', stats.runtime_formatted || '--');

    // 更新各項統計數據
    updateStatValue('statPrice', stats.current_price ? stats.current_price.toFixed(4) : '--');
    updateStatValue('statTotalBalance', stats.total_balance_usd ? `$${stats.total_balance_usd.toFixed(2)}` : '--');

    // 更新累計盈虧（帶顏色）
    if (stats.cumulative_pnl !== undefined && stats.cumulative_pnl !== null) {
        updateStatValue('statCumulativePnL', formatPnL(stats.cumulative_pnl), stats.cumulative_pnl);
    } else {
        updateStatValue('statCumulativePnL', '--');
    }

    // 更新交易統計
    updateStatValue('statTotalTrades', stats.total_trades || '0');
    updateStatValue('statBuySell', `${stats.buy_trades || 0} / ${stats.sell_trades || 0}`);

    // 更新成交額和手續費
    updateStatValue('statVolumeUSDC', stats.total_volume_usdc ? `$${stats.total_volume_usdc.toFixed(2)}` : '--');
    updateStatValue('statFees', stats.total_fees ? `${stats.total_fees.toFixed(4)} ${stats.quote_asset || ''}` : '--');

    // 更新磨損率
    if (stats.slippage_rate !== undefined && stats.slippage_rate !== null) {
        const slippageValue = stats.slippage_rate.toFixed(4) + '%';
        updateStatValue('statSlippage', slippageValue);
    } else {
        updateStatValue('statSlippage', '--');
    }

    // 更新成交量
    const makerTotal = (stats.maker_buy_volume || 0) + (stats.maker_sell_volume || 0);
    const takerTotal = (stats.taker_buy_volume || 0) + (stats.taker_sell_volume || 0);
    updateStatValue('statMakerVolume', makerTotal.toFixed(4));
    updateStatValue('statTakerVolume', takerTotal.toFixed(4));
}

// 更新單個統計項
function updateStatValue(elementId, value, numericValue = null) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.textContent = value;

    // 根據數值設置顏色
    if (numericValue !== null && numericValue !== undefined) {
        element.classList.remove('positive', 'negative');
        if (numericValue > 0) {
            element.classList.add('positive');
        } else if (numericValue < 0) {
            element.classList.add('negative');
        }
    }
}

// 格式化盈虧顯示
function formatPnL(value) {
    if (value === null || value === undefined) return '--';
    const prefix = value >= 0 ? '+' : '';
    return `${prefix}${value.toFixed(4)}`;
}

// 更新UI運行狀態
function updateUIForRunningState(running, config = null) {
    const statsSection = document.getElementById('statsSection');

    if (running) {
        startBtn.disabled = true;
        startBtn.innerHTML = '<span>運行中</span>';
        stopBtn.disabled = false;

        // 禁用表單輸入
        Array.from(form.elements).forEach(element => {
            if (element.type !== 'button' && element.type !== 'submit') {
                element.disabled = true;
            }
        });

        if (config) {
            let strategyName = config.strategy === 'standard' ? '標準' : '對沖';
            let marketName = config.market_type === 'spot' ? '現貨' : '永續';
            currentConfig.textContent = `${config.exchange} - ${config.symbol} (${marketName}/${strategyName})`;
        }
        botStatus.textContent = '運行中';

        // 顯示統計區域
        if (statsSection) {
            statsSection.style.display = 'block';
        }
    } else {
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>啟動機器人</span>';
        stopBtn.disabled = true;

        // 啟用表單輸入
        Array.from(form.elements).forEach(element => {
            element.disabled = false;
        });

        currentConfig.textContent = '無';
        botStatus.textContent = '未啟動';

        // 隱藏統計區域
        if (statsSection) {
            statsSection.style.display = 'none';
        }
    }
}

// 添加日誌
function addLog(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString('zh-TW');
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span>${message}</span>
    `;

    logDisplay.appendChild(logEntry);
    logDisplay.scrollTop = logDisplay.scrollHeight;

    // 限制日誌條目數量
    while (logDisplay.children.length > 100) {
        logDisplay.removeChild(logDisplay.firstChild);
    }
}

// 工具函數：格式化時間
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
}
