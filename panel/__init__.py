"""
交互式面板包 - 提供图形化操作界面
"""

# 版本信息
__version__ = "1.0.0"

# 导出主要的类和函数
from panel.interactive_panel import InteractivePanel
from panel.key_handler import KeyboardHandler

# 导出设定相关的函数
from panel.settings import (
    get_setting,
    set_setting,
    update_settings,
    load_settings,
    reset_defaults
) 