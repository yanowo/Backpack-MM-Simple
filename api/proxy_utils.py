"""
代理配置工具模塊
統一從環境變量讀取代理配置，供所有 API 客户端使用
"""
import os
from typing import Dict
from logger import setup_logger

logger = setup_logger("api.proxy_utils")


def get_proxy_config() -> Dict[str, str]:
    """
    從環境變量讀取代理配置

    優先級：
    1. 如果設置了 HTTPS_PROXY，則 HTTPS 使用 HTTPS_PROXY
    2. 如果只設置了 HTTP_PROXY，則 HTTP 和 HTTPS 都使用 HTTP_PROXY
    3. 如果都未設置，返回空字典

    Returns:
        Dict[str, str]: 代理配置字典，如 {'http': '...', 'https': '...'}
    """
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')

    proxies = {}

    if http_proxy:
        proxies['http'] = http_proxy
        # 如果沒有單獨設置 https_proxy，HTTPS 也使用 http_proxy
        if not https_proxy:
            proxies['https'] = http_proxy

    if https_proxy:
        proxies['https'] = https_proxy

    if proxies:
        logger.info(f"已從環境變量讀取代理配置: HTTP={'http' in proxies}, HTTPS={'https' in proxies}")
        logger.debug(f"代理配置詳情: {proxies}")
    else:
        logger.debug("未檢測到代理配置")

    return proxies
