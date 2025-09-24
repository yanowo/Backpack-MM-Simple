#!/usr/bin/env python
"""
交互式命令面板主程序
"""
import sys
import os
import argparse

# 检查所需库
try:
    import rich
except ImportError:
    print("错误: 未安装rich库。请执行 pip install rich 安装。")
    sys.exit(1)

# 导入面板模块
from panel.interactive_panel import InteractivePanel
from panel.key_handler import KeyboardHandler
from panel.settings import load_settings, update_settings

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Backpack Exchange 做市交易面板',
        epilog='''
使用示例:
  现货交易:
    python panel/panel_main.py --symbol SOL_USDC
  
  永续合约 (需在面板中设置 market_type perp):
    python panel/panel_main.py --symbol SOL_USDC_PERP
        '''
    )
    
    # 基本参数
    parser.add_argument('--api-key', type=str, help='API Key (可选，默认使用环境变数或配置文件)')
    parser.add_argument('--secret-key', type=str, help='Secret Key (可选，默认使用环境变数或配置文件)')
    parser.add_argument('--symbol', type=str, help='默认交易对 (现货: SOL_USDC, 永续: SOL_USDC_PERP)')
    parser.add_argument('--settings-dir', type=str, default='settings', help='设定目录路径')
    
    return parser.parse_args()

def run_panel(api_key=None, secret_key=None, default_symbol=None):
    """
    启动交互式面板
    
    Args:
        api_key: API密钥
        secret_key: API密钥
        default_symbol: 默认交易对
    """
    # 检查API密钥
    try:
        from config import API_KEY as CONFIG_API_KEY, SECRET_KEY as CONFIG_SECRET_KEY
    except ImportError:
        CONFIG_API_KEY = os.getenv('API_KEY')
        CONFIG_SECRET_KEY = os.getenv('SECRET_KEY')
    
    # 优先使用传入的参数，其次使用配置文件或环境变量
    api_key = api_key or CONFIG_API_KEY
    secret_key = secret_key or CONFIG_SECRET_KEY
    
    if not api_key or not secret_key:
        print("错误: 未设置API密钥。请在config.py中设置、使用环境变量或通过命令行参数提供。")
        sys.exit(1)
    
    # 加载设定并更新默认交易对
    settings = load_settings()
    if default_symbol:
        update_settings({'default_symbol': default_symbol})
    
    # 建立交互式面板
    panel = InteractivePanel()
    
    # 设置键盘处理器
    handler = KeyboardHandler(panel.handle_input)
    handler.start()
    
    try:
        # 启动面板
        panel.start()
    except KeyboardInterrupt:
        print("\n程序已被中断")
    except Exception as e:
        print(f"运行错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 释放资源
        handler.stop()
        if hasattr(panel, 'cleanup'):
            panel.cleanup()

def main():
    """主函数"""
    args = parse_arguments()
    run_panel(
        api_key=args.api_key,
        secret_key=args.secret_key,
        default_symbol=args.symbol
    )

if __name__ == "__main__":
    main()