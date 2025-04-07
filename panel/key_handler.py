"""
鍵盤處理模塊 - 提供跨平台鍵盤輸入處理
"""
import os
import sys
import threading
import time
from typing import Callable, Any

# 嘗試導入適合的鍵盤處理庫
try:
    # Windows平台
    import msvcrt
    
    def get_key():
        """
        Windows下獲取按鍵
        """
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # 將bytes轉換為字符串
            key_decoded = key.decode('utf-8', errors='ignore')
            
            # 處理特殊按鍵
            if key == b'\xe0':  # 擴展按鍵
                key = msvcrt.getch()
                if key == b'H':  # 上箭頭
                    return "up"
                elif key == b'P':  # 下箭頭
                    return "down"
                elif key == b'K':  # 左箭頭
                    return "left"
                elif key == b'M':  # 右箭頭
                    return "right"
                else:
                    return None
            elif key == b'\r':  # 回車鍵
                return "enter"
            elif key == b'\x08':  # 退格鍵
                return "backspace"
            elif key == b'\x1b':  # ESC鍵
                return "escape"
            elif key == b'\t':  # Tab鍵
                return "tab"
            else:
                return key_decoded
        return None
    
    WINDOWS = True
    
except ImportError:
    # Unix平台
    try:
        import termios
        import tty
        import select
        
        def get_key():
            """
            Unix下獲取按鍵
            """
            # 設置為非阻塞模式
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                # 檢查是否有輸入
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    
                    # 處理特殊按鍵
                    if key == '\x1b':  # ESC鍵可能是特殊按鍵的開始
                        next_key = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.1)[0] else None
                        if next_key == '[':  # 箭頭按鍵的序列
                            key = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.1)[0] else None
                            if key == 'A':
                                return "up"
                            elif key == 'B':
                                return "down"
                            elif key == 'C':
                                return "right"
                            elif key == 'D':
                                return "left"
                        return "escape"
                    elif key == '\r':  # 回車鍵
                        return "enter"
                    elif key == '\x7f':  # 退格鍵
                        return "backspace"
                    elif key == '\t':  # Tab鍵
                        return "tab"
                    else:
                        return key
                return None
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        WINDOWS = False
        
    except (ImportError, AttributeError):
        # 無法使用特定平台的庫，創建簡單替代方案
        def get_key():
            """
            簡單的輸入捕獲（非實時）
            """
            if sys.stdin.isatty():
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    return key
            return None
            
        WINDOWS = False

class KeyboardHandler:
    """
    跨平台的鍵盤處理類
    """
    def __init__(self, callback: Callable[[str], Any]):
        """
        初始化鍵盤處理器
        
        Args:
            callback: 按鍵處理回調函數
        """
        self.callback = callback
        self.running = False
        self.thread = None
    
    def start(self):
        """
        啟動鍵盤監聽
        """
        self.running = True
        self.thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        self.thread.start()
    
    def stop(self):
        """
        停止鍵盤監聽
        """
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
    
    def _listen_keyboard(self):
        """
        監聽鍵盤輸入
        """
        while self.running:
            key = get_key()
            if key:
                self.callback(key)
            time.sleep(0.01)  # 降低CPU使用率

# 直接測試
if __name__ == "__main__":
    def key_callback(key):
        print(f"按下鍵: {key}")
        if key == 'q':
            print("退出...")
            return False
        return True
    
    print("按鍵測試 (按 'q' 退出)...")
    
    handler = KeyboardHandler(key_callback)
    handler.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        handler.stop()