// WebSocket連接
let socket;
let isConnected = false;
let botRunning = false;
let heartbeatInterval = null;
let activeEvmProvider = null;
let activeEvmWallet = null;
let activeEvmAccount = null;
let preparedPopdexAgent = null;
let popdexAgentNetworkInitialized = false;
const discoveredEvmWallets = new Map();
const boundEvmProviders = new WeakSet();

// EIP-6963 allows multiple injected EVM wallets to coexist without every
// extension competing for window.ethereum.
window.addEventListener('eip6963:announceProvider', (event) => {
    const detail = event && event.detail;
    if (detail && detail.provider && detail.info) {
        registerEvmWallet(detail.provider, detail.info);
    }
});

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
const strategySelect = document.getElementById('strategy');
const gridParams = document.getElementById('gridParams');
const autoPriceRangeCheckbox = document.getElementById('auto_price_range');
const manualPriceFields = document.getElementById('manualPriceFields');
const manualPriceLower = document.getElementById('manualPriceLower');
const autoPriceField = document.getElementById('autoPriceField');
const gridTypeField = document.getElementById('gridTypeField');
const exchangeSelect = document.getElementById('exchange');
const spreadField = document.getElementById('spreadField');
const maxOrdersPerpField = document.getElementById('maxOrdersPerpField');
const targetPositionField = document.getElementById('targetPositionField');
const positionThresholdField = document.getElementById('positionThresholdField');
const inventorySkewField = document.getElementById('inventorySkewField');
const popdexAgentPanel = document.getElementById('popdexAgentPanel');
const popdexAgentNetwork = document.getElementById('popdexAgentNetwork');
const popdexAgentExpiry = document.getElementById('popdexAgentExpiry');
const popdexAgentGlobal = document.getElementById('popdexAgentGlobal');
const popdexAgentBadge = document.getElementById('popdexAgentBadge');
const popdexAgentBadgeText = document.getElementById('popdexAgentBadgeText');
const popdexAgentMessage = document.getElementById('popdexAgentMessage');
const evmWalletPicker = document.getElementById('evmWalletPicker');
const evmWalletList = document.getElementById('evmWalletList');
const connectEvmWalletBtn = document.getElementById('connectEvmWalletBtn');
const preparePopdexAgentBtn = document.getElementById('preparePopdexAgentBtn');
const authorizePopdexAgentBtn = document.getElementById('authorizePopdexAgentBtn');
const refreshWalletsBtn = document.getElementById('refreshWalletsBtn');
const agentWalletLabel = document.getElementById('agentWalletLabel');
const agentWalletAddress = document.getElementById('agentWalletAddress');
const agentPublicAddress = document.getElementById('agentPublicAddress');
const agentApprovalState = document.getElementById('agentApprovalState');
const agentStepWallet = document.getElementById('agentStepWallet');
const agentStepKey = document.getElementById('agentStepKey');
const agentStepAuthorize = document.getElementById('agentStepAuthorize');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    setupEventListeners();
    loadConfig();

    // 初始化表單狀態
    adjustMarketTypeOptions();
    toggleSpreadField();
    toggleGridTypeField();
    initPopdexAgentSetup();
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
        toggleGridTypeField();
        toggleSpreadField();
    });

    // 策略切換
    strategySelect.addEventListener('change', () => {
        toggleStrategyParams();
        toggleSpreadField();
    });

    // 交易所切換
    exchangeSelect.addEventListener('change', () => {
        // 根據交易所調整可用的市場類型選項
        adjustMarketTypeOptions();
        togglePopdexAgentPanel();
    });

    // 自動價格範圍切換
    autoPriceRangeCheckbox.addEventListener('change', () => {
        togglePriceRangeMode();
    });
}

// ==================== POPDEX Agent / 通用 EVM 錢包 ====================

function initPopdexAgentSetup() {
    if (!popdexAgentPanel) {
        return;
    }

    connectEvmWalletBtn.addEventListener('click', showEvmWalletPicker);
    refreshWalletsBtn.addEventListener('click', () => requestEvmWalletDiscovery(true));
    preparePopdexAgentBtn.addEventListener('click', preparePopdexAgent);
    authorizePopdexAgentBtn.addEventListener('click', authorizePopdexAgent);

    popdexAgentNetwork.addEventListener('change', () => {
        popdexAgentNetworkInitialized = true;
        clearPreparedAgent();
        loadPopdexAgentStatus();
    });
    popdexAgentExpiry.addEventListener('change', clearPreparedAgent);
    popdexAgentGlobal.addEventListener('change', clearPreparedAgent);

    requestEvmWalletDiscovery(false);
    togglePopdexAgentPanel();
}

function togglePopdexAgentPanel() {
    if (!popdexAgentPanel) {
        return;
    }
    const visible = exchangeSelect.value === 'popdex';
    popdexAgentPanel.hidden = !visible;
    if (visible) {
        loadPopdexAgentStatus();
        requestEvmWalletDiscovery(false);
    }
}

function registerEvmWallet(provider, info = {}) {
    if (!provider || typeof provider.request !== 'function') {
        return;
    }
    for (const entry of discoveredEvmWallets.values()) {
        if (entry.provider === provider) {
            if (info.name && entry.info.name === 'EVM Wallet') {
                entry.info = normalizeWalletInfo(provider, info);
            }
            renderEvmWalletList();
            return;
        }
    }

    const normalizedInfo = normalizeWalletInfo(provider, info);
    let key = normalizedInfo.uuid || normalizedInfo.rdns;
    if (!key || discoveredEvmWallets.has(key)) {
        key = `${normalizedInfo.rdns || 'wallet'}-${discoveredEvmWallets.size + 1}`;
    }
    discoveredEvmWallets.set(key, {provider, info: normalizedInfo});
    renderEvmWalletList();
}

function normalizeWalletInfo(provider, info = {}) {
    let detectedName = 'EVM Wallet';
    let rdns = info.rdns || 'injected.evm.wallet';
    if (provider.isRabby) {
        detectedName = 'Rabby Wallet';
        rdns = 'io.rabby';
    } else if (provider.isOkxWallet || provider.isOKExWallet) {
        detectedName = 'OKX Wallet';
        rdns = 'com.okex.wallet';
    } else if (provider.isCoinbaseWallet) {
        detectedName = 'Coinbase Wallet';
        rdns = 'com.coinbase.wallet';
    } else if (provider.isMetaMask) {
        detectedName = 'MetaMask';
        rdns = 'io.metamask';
    }
    return {
        uuid: info.uuid || null,
        name: info.name || detectedName,
        icon: info.icon || null,
        rdns: info.rdns || rdns,
    };
}

function collectLegacyEvmWallets() {
    const candidates = [];
    if (window.ethereum && Array.isArray(window.ethereum.providers)) {
        window.ethereum.providers.forEach(provider => candidates.push(provider));
    }
    if (window.okxwallet) {
        candidates.push(window.okxwallet.ethereum || window.okxwallet);
    }
    if (window.coinbaseWalletExtension) {
        candidates.push(window.coinbaseWalletExtension);
    }
    if (window.ethereum) {
        candidates.push(window.ethereum);
    }
    candidates.forEach(provider => registerEvmWallet(provider));
}

function requestEvmWalletDiscovery(showPicker) {
    try {
        window.dispatchEvent(new Event('eip6963:requestProvider'));
    } catch (error) {
        console.debug('EIP-6963 discovery unavailable:', error);
    }
    collectLegacyEvmWallets();
    window.setTimeout(() => {
        collectLegacyEvmWallets();
        renderEvmWalletList();
        if (showPicker) {
            evmWalletPicker.hidden = false;
        }
    }, 120);
}

function showEvmWalletPicker() {
    requestEvmWalletDiscovery(true);
    evmWalletPicker.hidden = false;
    setAgentMessage('請選擇已安裝的 EVM 錢包。OKX、MetaMask、Rabby 等都可使用。');
}

function renderEvmWalletList() {
    if (!evmWalletList) {
        return;
    }
    evmWalletList.replaceChildren();
    if (discoveredEvmWallets.size === 0) {
        const empty = document.createElement('div');
        empty.className = 'wallet-empty-state';
        empty.textContent = '尚未偵測到 EVM 瀏覽器錢包。請安裝或解鎖 OKX Wallet、MetaMask、Rabby 等錢包後重新偵測。';
        evmWalletList.appendChild(empty);
        return;
    }

    discoveredEvmWallets.forEach((entry, key) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'wallet-provider-card';
        button.addEventListener('click', () => connectEvmWallet(key));

        if (entry.info.icon && entry.info.icon.startsWith('data:image/')) {
            const icon = document.createElement('img');
            icon.className = 'wallet-provider-icon';
            icon.alt = '';
            icon.src = entry.info.icon;
            button.appendChild(icon);
        } else {
            const monogram = document.createElement('span');
            monogram.className = 'wallet-provider-monogram';
            monogram.textContent = (entry.info.name || 'E').charAt(0).toUpperCase();
            button.appendChild(monogram);
        }

        const meta = document.createElement('span');
        meta.className = 'wallet-provider-meta';
        const name = document.createElement('strong');
        name.textContent = entry.info.name || 'EVM Wallet';
        const standard = document.createElement('small');
        standard.textContent = entry.info.rdns || 'EIP-1193 provider';
        meta.append(name, standard);
        button.appendChild(meta);
        evmWalletList.appendChild(button);
    });
}

async function connectEvmWallet(key) {
    const entry = discoveredEvmWallets.get(key);
    if (!entry) {
        setAgentMessage('找不到選擇的錢包，請重新偵測。', 'error');
        return;
    }
    setAgentMessage(`正在等待 ${entry.info.name} 連接確認…`);
    try {
        const accounts = await entry.provider.request({method: 'eth_requestAccounts'});
        if (!Array.isArray(accounts) || !accounts[0]) {
            throw new Error('錢包未提供可用帳戶');
        }
        activeEvmProvider = entry.provider;
        activeEvmWallet = entry.info;
        updateActiveEvmAccount(accounts[0]);
        bindEvmProviderEvents(entry.provider);
        evmWalletPicker.hidden = true;
        setAgentMessage(`${entry.info.name} 已連接。下一步將在本機建立或載入 Agent Key。`, 'success');
        addLog(`已連接 ${entry.info.name}: ${shortAddress(accounts[0])}`, 'success');
    } catch (error) {
        const rejected = Number(error && error.code) === 4001;
        setAgentMessage(rejected ? '使用者取消了錢包連接。' : `錢包連接失敗：${error.message || error}`, 'error');
    }
}

function bindEvmProviderEvents(provider) {
    if (!provider || typeof provider.on !== 'function' || boundEvmProviders.has(provider)) {
        return;
    }
    boundEvmProviders.add(provider);
    provider.on('accountsChanged', (accounts) => {
        if (provider !== activeEvmProvider) {
            return;
        }
        updateActiveEvmAccount(Array.isArray(accounts) ? accounts[0] : null);
        clearPreparedAgent();
        if (activeEvmAccount) {
            setAgentMessage('錢包帳戶已變更，請重新準備 Agent 授權。');
        } else {
            setAgentMessage('錢包已中斷連接。', 'error');
        }
    });
    provider.on('chainChanged', () => {
        if (provider === activeEvmProvider) {
            clearPreparedAgent();
            setAgentMessage('錢包網路已變更；授權前會再次核對 POPDEX 網路。');
        }
    });
    provider.on('disconnect', () => {
        if (provider === activeEvmProvider) {
            updateActiveEvmAccount(null);
            clearPreparedAgent();
        }
    });
}

function updateActiveEvmAccount(account) {
    activeEvmAccount = account || null;
    if (activeEvmAccount) {
        agentWalletLabel.textContent = activeEvmWallet ? activeEvmWallet.name : 'EVM Wallet';
        agentWalletAddress.textContent = activeEvmAccount;
        agentWalletAddress.title = activeEvmAccount;
        setAgentStep(agentStepWallet, 'complete');
        preparePopdexAgentBtn.disabled = false;
        connectEvmWalletBtn.querySelector('span').textContent = '更換 EVM 錢包';
    } else {
        agentWalletLabel.textContent = '連接 EVM 錢包';
        agentWalletAddress.textContent = '等待連接';
        agentWalletAddress.removeAttribute('title');
        setAgentStep(agentStepWallet, 'current');
        preparePopdexAgentBtn.disabled = true;
        authorizePopdexAgentBtn.disabled = true;
        connectEvmWalletBtn.querySelector('span').textContent = '連接 EVM 錢包';
    }
}

function clearPreparedAgent() {
    preparedPopdexAgent = null;
    authorizePopdexAgentBtn.disabled = true;
    if (activeEvmAccount) {
        preparePopdexAgentBtn.disabled = false;
    }
    if (!agentStepAuthorize.classList.contains('is-complete')) {
        setAgentStep(agentStepAuthorize, 'idle');
        agentApprovalState.textContent = '等待建立授權資料';
    }
}

async function loadPopdexAgentStatus() {
    if (!popdexAgentPanel || popdexAgentPanel.hidden) {
        return null;
    }
    try {
        const statusUrl = popdexAgentNetworkInitialized
            ? `/api/popdex-agent/status?network=${encodeURIComponent(popdexAgentNetwork.value)}`
            : '/api/popdex-agent/status';
        const response = await fetch(statusUrl);
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.message || '狀態查詢失敗');
        }
        if (!popdexAgentNetworkInitialized && result.network) {
            popdexAgentNetwork.value = result.network;
        }
        popdexAgentNetworkInitialized = true;
        applyPopdexAgentStatus(result);
        return result;
    } catch (error) {
        setAgentBadge('error', '狀態不可用');
        setAgentMessage(`Agent 狀態查詢失敗：${error.message || error}`, 'error');
        return null;
    }
}

function applyPopdexAgentStatus(status) {
    if (status.agent_address) {
        agentPublicAddress.textContent = status.agent_address;
        agentPublicAddress.title = status.agent_address;
        setAgentStep(agentStepKey, 'complete');
    } else {
        agentPublicAddress.textContent = status.key_valid === false && status.configured ? '設定的私鑰無效' : '尚未產生';
        setAgentStep(agentStepKey, activeEvmAccount ? 'current' : 'idle');
    }

    if (!activeEvmAccount && status.account_address) {
        agentWalletAddress.textContent = status.account_address;
        agentWalletAddress.title = status.account_address;
    }

    if (status.agent_address && status.agent_info && status.agent_info.available === false) {
        setAgentStep(agentStepAuthorize, 'current');
        agentApprovalState.textContent = 'RPC 狀態不可用';
        setAgentBadge('error', 'RPC 不可用');
        setAgentMessage(`無法確認鏈上 Agent 狀態：${status.agent_info.error || '請稍後再試'}`, 'error');
    } else if (status.authorized) {
        setAgentStep(agentStepAuthorize, 'complete');
        agentApprovalState.textContent = '鏈上授權有效';
        authorizePopdexAgentBtn.disabled = true;
        setAgentBadge('ready', '授權完成');
        setAgentMessage('POPDEX Agent 已完成鏈上授權，可以啟動策略。', 'success');
    } else if (status.agent_address) {
        setAgentStep(agentStepAuthorize, 'current');
        agentApprovalState.textContent = '等待錢包授權';
        setAgentBadge('pending', '待鏈上授權');
    } else {
        setAgentStep(agentStepAuthorize, 'idle');
        agentApprovalState.textContent = '等待前兩步';
        setAgentBadge('idle', '尚未設定');
    }
}

async function preparePopdexAgent() {
    if (!activeEvmAccount) {
        showEvmWalletPicker();
        return;
    }
    setButtonBusy(preparePopdexAgentBtn, true, '本機建立中…');
    setAgentMessage('正在本機產生或載入 Agent Key；私鑰不會回傳到瀏覽器。');
    try {
        const response = await fetch('/api/popdex-agent/prepare', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                account_address: activeEvmAccount,
                network: popdexAgentNetwork.value,
                expires_in_days: Number(popdexAgentExpiry.value),
                global_scope: popdexAgentGlobal.checked,
            }),
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.message || 'Agent 建立失敗');
        }
        preparedPopdexAgent = result;
        agentPublicAddress.textContent = result.agent_address;
        agentPublicAddress.title = result.agent_address;
        setAgentStep(agentStepKey, 'complete');

        if (result.already_authorized) {
            setAgentStep(agentStepAuthorize, 'complete');
            agentApprovalState.textContent = '鏈上授權有效';
            authorizePopdexAgentBtn.disabled = true;
            setAgentBadge('ready', '授權完成');
            setAgentMessage('既有 Agent 已正確授權，無需再次簽署。', 'success');
        } else {
            setAgentStep(agentStepAuthorize, 'current');
            agentApprovalState.textContent = '等待錢包簽署';
            authorizePopdexAgentBtn.disabled = false;
            setAgentBadge('pending', '待錢包簽署');
            setAgentMessage(`Agent ${shortAddress(result.agent_address)} 已保存至本機 .env。請由 ${activeEvmWallet.name} 確認鏈上授權。`, 'success');
        }
        addLog(`POPDEX Agent ${result.generated ? '已產生' : '已載入'}: ${shortAddress(result.agent_address)}`, 'success');
    } catch (error) {
        preparedPopdexAgent = null;
        authorizePopdexAgentBtn.disabled = true;
        setAgentBadge('error', '建立失敗');
        setAgentMessage(`Agent 建立失敗：${error.message || error}`, 'error');
        addLog(`POPDEX Agent 建立失敗: ${error.message || error}`, 'error');
    } finally {
        setButtonBusy(preparePopdexAgentBtn, false, '建立本機 Agent');
        preparePopdexAgentBtn.disabled = !activeEvmAccount;
    }
}

async function authorizePopdexAgent() {
    if (!activeEvmProvider || !activeEvmAccount || !preparedPopdexAgent || !preparedPopdexAgent.transaction) {
        setAgentMessage('請先連接 EVM 錢包並建立本機 Agent。', 'error');
        return;
    }
    setButtonBusy(authorizePopdexAgentBtn, true, '等待錢包確認…');
    setAgentBadge('pending', '簽署中');
    try {
        await ensureEvmWalletChain(activeEvmProvider, preparedPopdexAgent.chain);
        const accounts = await activeEvmProvider.request({method: 'eth_accounts'});
        const currentAccount = Array.isArray(accounts) ? accounts[0] : null;
        if (!currentAccount || currentAccount.toLowerCase() !== activeEvmAccount.toLowerCase()) {
            throw new Error('錢包目前選擇的帳戶已改變，請重新連接並準備授權');
        }

        const transaction = {
            from: activeEvmAccount,
            to: preparedPopdexAgent.transaction.to,
            data: preparedPopdexAgent.transaction.data,
            gas: preparedPopdexAgent.transaction.gas,
            gasPrice: preparedPopdexAgent.transaction.gasPrice,
            value: '0x0',
        };
        const txHash = await activeEvmProvider.request({
            method: 'eth_sendTransaction',
            params: [transaction],
        });
        agentApprovalState.textContent = `交易 ${shortAddress(txHash)}`;
        setAgentMessage(`授權交易已送出：${txHash}。正在等待鏈上確認…`);
        addLog(`POPDEX Agent 授權交易已送出: ${txHash}`, 'info');

        const receipt = await waitForEvmReceipt(activeEvmProvider, txHash);
        if (receipt && receipt.status && BigInt(receipt.status) === 0n) {
            throw new Error('授權交易已上鏈但執行失敗');
        }
        const verified = await waitForPopdexAgentAuthorization();
        if (!verified) {
            throw new Error('交易已確認，但尚未讀到 Agent 授權狀態；請稍後重新檢查');
        }
        setAgentStep(agentStepAuthorize, 'complete');
        agentApprovalState.textContent = '鏈上授權有效';
        setAgentBadge('ready', '授權完成');
        setAgentMessage('POPDEX Agent 授權完成，現在可以啟動策略。', 'success');
        addLog('POPDEX Agent 鏈上授權完成', 'success');
    } catch (error) {
        const rejected = Number(error && error.code) === 4001;
        setAgentBadge('error', rejected ? '已取消' : '授權失敗');
        setAgentMessage(rejected ? '使用者取消了鏈上授權。' : `鏈上授權失敗：${error.message || error}`, 'error');
        addLog(`POPDEX Agent 授權失敗: ${error.message || error}`, 'error');
    } finally {
        setButtonBusy(authorizePopdexAgentBtn, false, '簽署鏈上授權');
        authorizePopdexAgentBtn.disabled = agentStepAuthorize.classList.contains('is-complete');
    }
}

async function ensureEvmWalletChain(provider, chain) {
    if (!chain || !chain.chainId) {
        throw new Error('缺少 POPDEX 網路設定');
    }
    const currentChain = await provider.request({method: 'eth_chainId'});
    if (BigInt(currentChain) === BigInt(chain.chainId)) {
        return;
    }
    try {
        await provider.request({
            method: 'wallet_switchEthereumChain',
            params: [{chainId: chain.chainId}],
        });
    } catch (error) {
        const code = Number(error && (error.code || (error.data && error.data.originalError && error.data.originalError.code)));
        if (code !== 4902) {
            throw error;
        }
        await provider.request({
            method: 'wallet_addEthereumChain',
            params: [chain],
        });
    }
}

async function waitForEvmReceipt(provider, txHash, timeoutMs = 120000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const receipt = await provider.request({
            method: 'eth_getTransactionReceipt',
            params: [txHash],
        });
        if (receipt) {
            return receipt;
        }
        await delay(1500);
    }
    throw new Error('等待授權交易確認逾時');
}

async function waitForPopdexAgentAuthorization() {
    for (let attempt = 0; attempt < 12; attempt += 1) {
        const status = await loadPopdexAgentStatus();
        if (status && status.authorized) {
            return true;
        }
        await delay(1000);
    }
    return false;
}

function setAgentStep(element, state) {
    if (!element) {
        return;
    }
    element.classList.remove('is-current', 'is-complete');
    if (state === 'current') {
        element.classList.add('is-current');
    } else if (state === 'complete') {
        element.classList.add('is-complete');
    }
}

function setAgentBadge(state, text) {
    popdexAgentBadge.classList.remove('is-idle', 'is-pending', 'is-ready', 'is-error');
    popdexAgentBadge.classList.add(`is-${state}`);
    popdexAgentBadgeText.textContent = text;
}

function setAgentMessage(message, type = '') {
    popdexAgentMessage.classList.remove('is-success', 'is-error');
    if (type) {
        popdexAgentMessage.classList.add(`is-${type}`);
    }
    popdexAgentMessage.textContent = message;
}

function setButtonBusy(button, busy, label) {
    button.disabled = busy;
    button.innerHTML = busy
        ? `<span class="loading"></span><span>${label}</span>`
        : `<span>${label}</span>`;
}

function shortAddress(value) {
    const text = String(value || '');
    return text.length > 14 ? `${text.slice(0, 8)}…${text.slice(-6)}` : text;
}

function delay(milliseconds) {
    return new Promise(resolve => window.setTimeout(resolve, milliseconds));
}

// 切換市場類型參數顯示
function toggleMarketTypeParams() {
    const marketType = marketTypeSelect.value;
    const strategy = strategySelect.value;

    // 預設隱藏，視情況顯示所需欄位
    spotParams.style.display = 'none';
    perpParams.style.display = 'none';

    // 更新永續合約參數區塊標題
    const perpParamsTitle = perpParams.querySelector('.block-title');
    if (perpParamsTitle) {
        perpParamsTitle.textContent = strategy === 'grid' ? '網格風控參數' : '永續合約參數';
    }

    if (strategy === 'grid') {
        // 現貨網格不需要額外倉位配置；永續網格要顯示倉位參數
        if (marketType === 'perp') {
            perpParams.style.display = 'block';
            togglePerpFields();
        }
        return;
    }

    if (marketType === 'spot') {
        spotParams.style.display = 'block';
    } else {
        perpParams.style.display = 'block';
        togglePerpFields();
    }
}

// 切換策略參數顯示
function toggleStrategyParams() {
    const strategy = strategySelect.value;
    gridParams.style.display = strategy === 'grid' ? 'block' : 'none';

    // 顯示正確的市場參數區塊
    toggleMarketTypeParams();

    // 更新網格類型字段的顯示
    toggleGridTypeField();

    // 更新永續合約參數欄位
    if (marketTypeSelect.value === 'perp') {
        togglePerpFields();
    }
}

// 切換價格範圍模式
function togglePriceRangeMode() {
    const isAuto = autoPriceRangeCheckbox.checked;

    if (isAuto) {
        manualPriceFields.style.display = 'none';
        manualPriceLower.style.display = 'none';
        autoPriceField.style.display = 'block';
    } else {
        manualPriceFields.style.display = 'block';
        manualPriceLower.style.display = 'block';
        autoPriceField.style.display = 'none';
    }
}

// 切換網格類型字段顯示（僅在永續合約網格時顯示）
function toggleGridTypeField() {
    const strategy = strategySelect.value;
    const marketType = marketTypeSelect.value;

    if (strategy === 'grid' && marketType === 'perp') {
        gridTypeField.style.display = 'block';
    } else {
        gridTypeField.style.display = 'none';
    }
}

// 根據交易所調整市場類型選項
function adjustMarketTypeOptions() {
    const exchange = exchangeSelect.value;
    const currentMarketType = marketTypeSelect.value;

    // 獲取市場類型的所有選項
    const spotOption = marketTypeSelect.querySelector('option[value="spot"]');
    const perpOption = marketTypeSelect.querySelector('option[value="perp"]');

    if (exchange === 'backpack') {
        // Backpack 支持現貨和永續合約
        if (spotOption) {
            spotOption.disabled = false;
            spotOption.style.display = 'block';
        }
        if (perpOption) {
            perpOption.disabled = false;
            perpOption.style.display = 'block';
        }
    } else {
        // 其他交易所（Aster、Paradex、Lighter、APEX、StandX、POPDEX）只支持永續合約
        if (spotOption) {
            spotOption.disabled = true;
            spotOption.style.display = 'none';
        }
        if (perpOption) {
            perpOption.disabled = false;
            perpOption.style.display = 'block';
        }

        // 如果當前選擇的是現貨，自動切換到永續合約
        if (currentMarketType === 'spot') {
            marketTypeSelect.value = 'perp';
            // 觸發相關更新
            toggleMarketTypeParams();
            toggleGridTypeField();
            toggleSpreadField();
        }
    }
}

// 切換價差欄位顯示
function toggleSpreadField() {
    const strategy = strategySelect.value;
    const marketType = marketTypeSelect.value;

    // 網格交易不需要設置價差
    if (strategy === 'grid') {
        spreadField.style.display = 'none';
        document.getElementById('spread').required = false;
    }
    // 永續合約對沖不需要設置價差
    else if (strategy === 'maker_hedge' && marketType === 'perp') {
        spreadField.style.display = 'none';
        document.getElementById('spread').required = false;
    }
    // 其他情況需要設置價差
    else {
        spreadField.style.display = 'block';
        document.getElementById('spread').required = true;
    }
}

// 切換永續合約參數欄位顯示
function togglePerpFields() {
    const strategy = strategySelect.value;
    const isGridStrategy = (strategy === 'grid');

    // 永續對沖與永續網格都不需要最大訂單數
    if (maxOrdersPerpField) {
        if (strategy === 'maker_hedge' || strategy === 'grid') {
            maxOrdersPerpField.style.display = 'none';
        } else {
            maxOrdersPerpField.style.display = 'block';
        }
    }

    // 網格策略不需要目標倉位、調整觸發值、偏移係數
    // 因為網格策略只使用最大倉位和止損止盈
    if (targetPositionField) {
        targetPositionField.style.display = isGridStrategy ? 'none' : 'block';
    }
    if (positionThresholdField) {
        positionThresholdField.style.display = isGridStrategy ? 'none' : 'block';
    }
    if (inventorySkewField) {
        inventorySkewField.style.display = isGridStrategy ? 'none' : 'block';
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

    // 根據策略類型添加額外參數
    if (data.strategy === 'grid') {
        // 網格策略參數
        data.auto_price_range = formData.get('auto_price_range') === 'on';
        data.grid_upper_price = formData.get('grid_upper_price') ? parseFloat(formData.get('grid_upper_price')) : null;
        data.grid_lower_price = formData.get('grid_lower_price') ? parseFloat(formData.get('grid_lower_price')) : null;
        data.grid_num = parseInt(formData.get('grid_num'));
        data.grid_mode = formData.get('grid_mode');
        data.price_range_percent = parseFloat(formData.get('price_range_percent'));

        // 永續合約網格特有參數
        if (data.market_type === 'perp') {
            data.grid_type = formData.get('grid_type') || 'neutral';

            const targetPositionRaw = formData.get('target_position');
            const maxPositionRaw = formData.get('max_position');
            const thresholdRaw = formData.get('position_threshold');
            const skewRaw = formData.get('inventory_skew');
            const stopLossRaw = formData.get('stop_loss');
            const takeProfitRaw = formData.get('take_profit');

            data.target_position = targetPositionRaw ? parseFloat(targetPositionRaw) : 0.0;
            data.max_position = maxPositionRaw ? parseFloat(maxPositionRaw) : 1.0;
            // 網格策略：position_threshold 設為 max_position，只有超過最大持倉才觸發風控
            data.position_threshold = thresholdRaw ? parseFloat(thresholdRaw) : data.max_position;
            data.inventory_skew = skewRaw ? parseFloat(skewRaw) : 0.0;
            data.stop_loss = stopLossRaw ? parseFloat(stopLossRaw) : null;
            data.take_profit = takeProfitRaw ? parseFloat(takeProfitRaw) : null;
        }

        // 網格策略不需要 max_orders 和 spread（會使用默認值）
        data.max_orders = 1;
        if (!data.spread) {
            data.spread = 0.1;
        }
    } else {
        // 標準策略和對沖策略參數
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

            // 永續對沖策略設置默認價差為 0.01%
            if (data.strategy === 'maker_hedge' && !data.spread) {
                data.spread = 0.01;
            }
        }
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

    // 更新網格策略統計（如果有）
    const gridStatsSection = document.getElementById('gridStatsSection');
    if (stats.grid_profit !== undefined || stats.grid_count !== undefined) {
        // 顯示網格統計區域
        gridStatsSection.style.display = 'block';

        // 更新網格利潤
        if (stats.grid_profit !== undefined && stats.grid_profit !== null) {
            updateStatValue('statGridProfit', formatPnL(stats.grid_profit), stats.grid_profit);
        } else {
            updateStatValue('statGridProfit', '--');
        }

        // 更新網格數量
        updateStatValue('statGridCount', stats.grid_count || '--');

        // 更新價格區間
        if (stats.grid_upper_price !== undefined && stats.grid_lower_price !== undefined) {
            const priceRange = `${stats.grid_lower_price.toFixed(4)} ~ ${stats.grid_upper_price.toFixed(4)}`;
            updateStatValue('statGridPriceRange', priceRange);
        } else {
            updateStatValue('statGridPriceRange', '--');
        }

        // 更新活躍訂單數
        updateStatValue('statActiveGridOrders', stats.active_grid_orders || '0');

        // 更新買入/賣出成交次數
        let filledText = '--';
        if (stats.grid_buy_filled !== undefined && stats.grid_sell_filled !== undefined) {
            // 現貨網格
            filledText = `${stats.grid_buy_filled || 0} / ${stats.grid_sell_filled || 0}`;
        } else if (stats.grid_long_filled !== undefined && stats.grid_short_filled !== undefined) {
            // 永續合約網格
            filledText = `${stats.grid_long_filled || 0} / ${stats.grid_short_filled || 0}`;
        }
        updateStatValue('statGridFilled', filledText);
    } else {
        // 隱藏網格統計區域
        gridStatsSection.style.display = 'none';
    }
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
