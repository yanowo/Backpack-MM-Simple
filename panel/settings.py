"""
面板設定模塊 - 處理設定的保存和加載
"""
import os
import json
import logging
from typing import Dict, Any, Optional

# 設定文件的默認路徑
DEFAULT_SETTINGS_DIR = 'settings'
DEFAULT_SETTINGS_FILE = 'panel_settings.json'

# 默認設定
DEFAULT_SETTINGS = {
    'base_spread_percentage': 0.1,  # 默認價差0.1%
    'order_quantity': None,         # 訂單數量，默認為自動
    'max_orders': 3,                # 每側3個訂單
    'duration': 24*3600,            # 運行24小時
    'interval': 60,                 # 每分鐘更新一次
    'default_symbol': None,         # 默認交易對
    'market_type': 'spot',          # 默認現貨
    'target_position': 0.0,         # 永續目標倉位
    'max_position': 1.0,            # 永續最大倉位
    'position_threshold': 0.1,      # 永續調整閾值
    'inventory_skew': 0.25          # 永續報價偏移
}

class SettingsManager:
    """設定管理器"""
    
    def __init__(self, settings_dir: str = DEFAULT_SETTINGS_DIR, settings_file: str = DEFAULT_SETTINGS_FILE):
        """
        初始化設定管理器
        
        Args:
            settings_dir: 設定目錄
            settings_file: 設定文件名
        """
        self.settings_dir = settings_dir
        self.settings_file = settings_file
        self.settings_path = os.path.join(settings_dir, settings_file)
        self.settings = DEFAULT_SETTINGS.copy()
        
        # 確保設定目錄存在
        os.makedirs(settings_dir, exist_ok=True)
        
        # 嘗試加載設定
        self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """
        從文件加載設定
        
        Returns:
            加載的設定字典
        """
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    
                # 更新設定，保留默認值作為備用
                for key, value in loaded_settings.items():
                    if key in self.settings:
                        self.settings[key] = value
                        
                return self.settings
            else:
                # 文件不存在時創建默認設定文件
                self.save_settings()
                return self.settings
        except Exception as e:
            logging.error(f"加載設定時出錯: {str(e)}")
            return self.settings
    
    def save_settings(self) -> bool:
        """
        保存設定到文件
        
        Returns:
            保存是否成功
        """
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"保存設定時出錯: {str(e)}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        獲取設定值
        
        Args:
            key: 設定鍵
            default: 默認值（如果設定不存在）
            
        Returns:
            設定值
        """
        return self.settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """
        設置設定值
        
        Args:
            key: 設定鍵
            value: 設定值
        """
        self.settings[key] = value
    
    def update_settings(self, settings_dict: Dict[str, Any]) -> None:
        """
        批量更新設定
        
        Args:
            settings_dict: 設定字典
        """
        for key, value in settings_dict.items():
            self.settings[key] = value
        
        # 自動保存更新後的設定
        self.save_settings()
    
    def reset_to_defaults(self) -> None:
        """重置為默認設定"""
        self.settings = DEFAULT_SETTINGS.copy()
        self.save_settings()

# 創建單例實例
settings_manager = SettingsManager()

# 導出便捷函數
def get_setting(key: str, default: Any = None) -> Any:
    """獲取設定值"""
    return settings_manager.get_setting(key, default)

def set_setting(key: str, value: Any) -> None:
    """設置設定值並保存"""
    settings_manager.set_setting(key, value)
    settings_manager.save_settings()

def update_settings(settings_dict: Dict[str, Any]) -> None:
    """批量更新設定並保存"""
    settings_manager.update_settings(settings_dict)

def load_settings() -> Dict[str, Any]:
    """加載設定"""
    return settings_manager.load_settings()

def reset_defaults() -> None:
    """重置為默認設定"""
    settings_manager.reset_to_defaults() 