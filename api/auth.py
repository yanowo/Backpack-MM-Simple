"""
API認證和簽名相關模塊
"""
import base64
import nacl.signing
from typing import Optional
from logger import setup_logger

logger = setup_logger("api.auth")

def create_signature(secret_key: str, message: str) -> Optional[str]:
    """
    創建API簽名
    
    Args:
        secret_key: API密鑰
        message: 要簽名的消息
        
    Returns:
        簽名字符串或None（如果簽名失敗）
    """
    try:
        decoded_key = base64.b64decode(secret_key)
        signing_key = nacl.signing.SigningKey(decoded_key)
        signature = signing_key.sign(message.encode('utf-8')).signature
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        logger.error(f"簽名創建失敗: {e}")
        return None