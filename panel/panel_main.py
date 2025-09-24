#!/usr/bin/env python
"""
交互式命令面板主程序
"""
import sys
import os
import argparse

# 檢查所需庫
try:
    import rich
except ImportError:
    print("錯誤: 未安裝rich庫。請執行 pip install rich 安裝。")
    sys.exit(1)

# 導入面板模塊
from panel.interactive_panel import InteractivePanel
from panel.key_handler import KeyboardHandler
from panel.settings import load_settings, update_settings

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description='Backpack Exchange 做市交易面板',
        epilog='''
使用示例:
  現貨交易:
    python panel/panel_main.py --symbol SOL_USDC
  
  永續合約 (需在面板中設置 market_type perp):
    python panel/panel_main.py --symbol SOL_USDC_PERP
        '''
    )
    
    # 基本參數
    parser.add_argument('--api-key', type=str, help='API Key (可選，默認使用環境變數或配置文件)')
    parser.add_argument('--secret-key', type=str, help='Secret Key (可選，默認使用環境變數或配置文件)')
    parser.add_argument('--symbol', type=str, help='默認交易對 (現貨: SOL_USDC, 永續: SOL_USDC_PERP)')
    parser.add_argument('--settings-dir', type=str, default='settings', help='設定目錄路徑')
    
    return parser.parse_args()

def run_panel(api_key=None, secret_key=None, default_symbol=None):
    """
    啟動交互式面板
    
    Args:
        api_key: API密鑰
        secret_key: API密鑰
        default_symbol: 默認交易對
    """
    # 檢查API密鑰
    try:
        from config import API_KEY as CONFIG_API_KEY, SECRET_KEY as CONFIG_SECRET_KEY
    except ImportError:
        CONFIG_API_KEY = os.getenv('API_KEY')
        CONFIG_SECRET_KEY = os.getenv('SECRET_KEY')
    
    # 優先使用傳入的參數，其次使用配置文件或環境變量
    api_key = api_key or CONFIG_API_KEY
    secret_key = secret_key or CONFIG_SECRET_KEY
    
    if not api_key or not secret_key:
        print("錯誤: 未設置API密鑰。請在config.py中設置、使用環境變量或通過命令行參數提供。")
        sys.exit(1)
    
    # 加載設定並更新默認交易對
    settings = load_settings()
    if default_symbol:
        update_settings({'default_symbol': default_symbol})
    
    # 建立交互式面板
    panel = InteractivePanel()
    
    # 設置鍵盤處理器
    handler = KeyboardHandler(panel.handle_input)
    handler.start()
    
    try:
        # 啟動面板
        panel.start()
    except KeyboardInterrupt:
        print("\n程序已被中斷")
    except Exception as e:
        print(f"運行錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 釋放資源
        handler.stop()
        if hasattr(panel, 'cleanup'):
            panel.cleanup()

def main():
    """主函數"""
    args = parse_arguments()
    run_panel(
        api_key=args.api_key,
        secret_key=args.secret_key,
        default_symbol=args.symbol
    )

if __name__ == "__main__":
    main()