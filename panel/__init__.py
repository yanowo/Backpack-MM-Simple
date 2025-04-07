"""
交互式面板包 - 提供圖形化操作界面
"""

# 版本信息
__version__ = "1.0.0"

# 導出主要的類和函數
from panel.interactive_panel import InteractivePanel
from panel.key_handler import KeyboardHandler

# 導出設定相關的函數
from panel.settings import (
    get_setting,
    set_setting,
    update_settings,
    load_settings,
    reset_defaults
) 