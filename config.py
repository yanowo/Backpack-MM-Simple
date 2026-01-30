"""
配置文件 - 全局配置管理中心
支持多交易所配置（Backpack, Aster, Paradex）
"""
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# ==================== 通用配置 ====================

# HTTP/HTTPS 代理配置（支持所有交易所，同時用於 REST API 和 WebSocket）
HTTP_PROXY = os.getenv('HTTP_PROXY')
HTTPS_PROXY = os.getenv('HTTPS_PROXY')

# 數據庫配置
DB_PATH = os.getenv('DB_PATH', 'orders.db')
ENABLE_DATABASE = os.getenv('ENABLE_DATABASE', '0').strip().lower() in {"1", "true", "yes", "on"}

# 日誌配置
LOG_FILE = os.getenv('LOG_FILE', 'market_maker.log')

# ==================== Backpack 交易所配置 ====================

# Backpack API 憑證
BACKPACK_API_KEY = os.getenv('BACKPACK_KEY') or os.getenv('API_KEY')
BACKPACK_SECRET_KEY = os.getenv('BACKPACK_SECRET') or os.getenv('SECRET_KEY')

# Backpack API 端點
BACKPACK_API_URL = os.getenv('BACKPACK_API_URL', 'https://api.backpack.exchange')
BACKPACK_WS_URL = os.getenv('BACKPACK_WS_URL', 'wss://ws.backpack.exchange')

# Backpack API 設定
BACKPACK_API_VERSION = os.getenv('BACKPACK_API_VERSION', 'v1')
BACKPACK_DEFAULT_WINDOW = os.getenv('BACKPACK_DEFAULT_WINDOW', '5000')

# ==================== Aster 交易所配置 ====================

# Aster API 憑證
ASTER_API_KEY = os.getenv('ASTER_API_KEY') or os.getenv('ASTER_KEY')
ASTER_SECRET_KEY = os.getenv('ASTER_SECRET_KEY') or os.getenv('ASTER_SECRET')

# Aster API 端點
ASTER_BASE_URL = os.getenv('ASTER_BASE_URL', 'https://api.aster.exchange')
ASTER_WS_URL = os.getenv('ASTER_WS_URL', 'wss://fstream.asterdex.com/ws')

# ==================== Paradex 交易所配置 ====================

# Paradex 使用 StarkNet 認證（不同於傳統 API Key）
PARADEX_ACCOUNT_ADDRESS = os.getenv('PARADEX_ACCOUNT_ADDRESS')
PARADEX_PRIVATE_KEY = os.getenv('PARADEX_PRIVATE_KEY')

# Paradex API 端點
PARADEX_BASE_URL = os.getenv('PARADEX_BASE_URL', 'https://api.prod.paradex.trade/v1')
PARADEX_WS_URL = os.getenv('PARADEX_WS_URL', 'wss://ws.api.prod.paradex.trade/v1')

# Paradex 簽名設置
PARADEX_SIGNATURE_TTL_SECONDS = int(os.getenv('PARADEX_SIGNATURE_TTL_SECONDS', '1800'))  # 30分鐘
PARADEX_JWT_REFRESH_BUFFER = int(os.getenv('PARADEX_JWT_REFRESH_BUFFER', '120'))  # 提前2分鐘刷新

# ==================== Lighter 交易所配置 ====================

LIGHTER_WS_URL = os.getenv('LIGHTER_WS_URL', 'wss://api.lighter.xyz/ws')

# ==================== Apex 交易所配置 ====================

APEX_WS_URL = os.getenv('APEX_WS_URL', 'wss://quote.omni.apex.exchange/realtime_public')

# ==================== 向後兼容性（保留舊變數名） ====================
# 注意：這些變數已被標記為 Deprecated，建議使用上面的 BACKPACK_ 前綴變數

# API 憑證（指向 Backpack）
API_KEY = BACKPACK_API_KEY  # Deprecated: 請使用 BACKPACK_API_KEY
SECRET_KEY = BACKPACK_SECRET_KEY  # Deprecated: 請使用 BACKPACK_SECRET_KEY

# API 端點（指向 Backpack）
API_URL = BACKPACK_API_URL  # Deprecated: 請使用 BACKPACK_API_URL
WS_URL = BACKPACK_WS_URL  # Deprecated: 請使用 BACKPACK_WS_URL

# API 設定（指向 Backpack）
API_VERSION = BACKPACK_API_VERSION  # Deprecated: 請使用 BACKPACK_API_VERSION
DEFAULT_WINDOW = BACKPACK_DEFAULT_WINDOW  # Deprecated: 請使用 BACKPACK_DEFAULT_WINDOW
