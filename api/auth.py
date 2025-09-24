"""
API认证和签名相关模块
"""
import base64
import nacl.signing
import sys
from typing import Optional
from logger import setup_logger

logger = setup_logger("api.auth")

def create_signature(secret_key: str, message: str) -> Optional[str]:
    """
    创建API签名
    
    Args:
        secret_key: API密钥
        message: 要签名的消息
        
    Returns:
        签名字符串或None（如果签名失败）
    """
    try:
        # 尝试对密钥进行解码和签名
        decoded_key = base64.b64decode(secret_key)
        signing_key = nacl.signing.SigningKey(decoded_key)
        signature = signing_key.sign(message.encode('utf-8')).signature
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        logger.error(f"签名创建失败: {e}")
        logger.error("无法创建API签名，程序将终止")
        # 强制终止程序
        sys.exit(1)