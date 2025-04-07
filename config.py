"""
配置文件
"""
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# API配置
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
API_URL = "https://api.backpack.exchange"
WS_URL = "wss://ws.backpack.exchange"
API_VERSION = "v1"
DEFAULT_WINDOW = "5000"

# 數據庫配置
DB_PATH = 'orders.db'

# 日誌配置
LOG_FILE = "market_maker.log"