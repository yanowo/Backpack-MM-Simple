"""
面板设定模块 - 处理设定的保存和加载
"""
import os
import json
import logging
from typing import Dict, Any, Optional

# 设定文件的默认路径
DEFAULT_SETTINGS_DIR = 'settings'
DEFAULT_SETTINGS_FILE = 'panel_settings.json'

# 默认设定
DEFAULT_SETTINGS = {
    'base_spread_percentage': 0.1,  # 默认价差0.1%
    'order_quantity': None,         # 订单数量，默认为自动
    'max_orders': 3,                # 每侧3个订单
    'duration': 24*3600,            # 运行24小时
    'interval': 60,                 # 每分钟更新一次
    'default_symbol': None,         # 默认交易对
    'market_type': 'spot',          # 默认现货 (spot/perp)
    'target_position': 0.0,         # 永续目标仓位 (0.0=中性)
    'max_position': 1.0,            # 永续最大仓位限制
    'position_threshold': 0.1,      # 永续调整阈值 (当偏离target超过此值时调整)
    'inventory_skew': 0          # 永续报价偏移系数 (0-1之间)
}

class SettingsManager:
    """设定管理器"""
    
    def __init__(self, settings_dir: str = DEFAULT_SETTINGS_DIR, settings_file: str = DEFAULT_SETTINGS_FILE):
        """
        初始化设定管理器
        
        Args:
            settings_dir: 设定目录
            settings_file: 设定文件名
        """
        self.settings_dir = settings_dir
        self.settings_file = settings_file
        self.settings_path = os.path.join(settings_dir, settings_file)
        self.settings = DEFAULT_SETTINGS.copy()
        
        # 确保设定目录存在
        os.makedirs(settings_dir, exist_ok=True)
        
        # 尝试加载设定
        self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """
        从文件加载设定
        
        Returns:
            加载的设定字典
        """
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    
                # 更新设定，保留默认值作为备用
                for key, value in loaded_settings.items():
                    if key in self.settings:
                        self.settings[key] = value
                        
                return self.settings
            else:
                # 文件不存在时创建默认设定文件
                self.save_settings()
                return self.settings
        except Exception as e:
            logging.error(f"加载设定时出错: {str(e)}")
            return self.settings
    
    def save_settings(self) -> bool:
        """
        保存设定到文件
        
        Returns:
            保存是否成功
        """
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"保存设定时出错: {str(e)}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        获取设定值
        
        Args:
            key: 设定键
            default: 默认值（如果设定不存在）
            
        Returns:
            设定值
        """
        return self.settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """
        设置设定值
        
        Args:
            key: 设定键
            value: 设定值
        """
        self.settings[key] = value
    
    def update_settings(self, settings_dict: Dict[str, Any]) -> None:
        """
        批量更新设定
        
        Args:
            settings_dict: 设定字典
        """
        for key, value in settings_dict.items():
            self.settings[key] = value
        
        # 自动保存更新后的设定
        self.save_settings()
    
    def reset_to_defaults(self) -> None:
        """重置为默认设定"""
        self.settings = DEFAULT_SETTINGS.copy()
        self.save_settings()

# 创建单例实例
settings_manager = SettingsManager()

# 导出便捷函数
def get_setting(key: str, default: Any = None) -> Any:
    """获取设定值"""
    return settings_manager.get_setting(key, default)

def set_setting(key: str, value: Any) -> None:
    """设置设定值并保存"""
    settings_manager.set_setting(key, value)
    settings_manager.save_settings()

def update_settings(settings_dict: Dict[str, Any]) -> None:
    """批量更新设定并保存"""
    settings_manager.update_settings(settings_dict)

def load_settings() -> Dict[str, Any]:
    """加载设定"""
    return settings_manager.load_settings()

def reset_defaults() -> None:
    """重置为默认设定"""
    settings_manager.reset_to_defaults() 