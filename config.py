"""
配置文件
"""
import os
from dotenv import load_dotenv

# 载入环境变数
load_dotenv()

# API配置
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
WS_PROXY = os.getenv('PROXY_WEBSOCKET')
API_URL = "https://api.backpack.exchange"
WS_URL = "wss://ws.backpack.exchange"
API_VERSION = "v1"
DEFAULT_WINDOW = "5000"

# 数据库配置
DB_PATH = 'orders.db'

# 日志配置
LOG_FILE = "market_maker.log"