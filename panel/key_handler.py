"""
键盘处理模块 - 提供跨平台键盘输入处理
"""
import os
import sys
import threading
import time
from typing import Callable, Any

# 尝试导入适合的键盘处理库
try:
    # Windows平台
    import msvcrt
    
    def get_key():
        """
        Windows下获取按键
        """
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # 将bytes转换为字符串
            key_decoded = key.decode('utf-8', errors='ignore')
            
            # 处理特殊按键
            if key == b'\xe0':  # 扩展按键
                key = msvcrt.getch()
                if key == b'H':  # 上箭头
                    return "up"
                elif key == b'P':  # 下箭头
                    return "down"
                elif key == b'K':  # 左箭头
                    return "left"
                elif key == b'M':  # 右箭头
                    return "right"
                else:
                    return None
            elif key == b'\r':  # 回车键
                return "enter"
            elif key == b'\x08':  # 退格键
                return "backspace"
            elif key == b'\x1b':  # ESC键
                return "escape"
            elif key == b'\t':  # Tab键
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
            Unix下获取按键
            """
            # 设置为非阻塞模式
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                # 检查是否有输入
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    
                    # 处理特殊按键
                    if key == '\x1b':  # ESC键可能是特殊按键的开始
                        next_key = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.1)[0] else None
                        if next_key == '[':  # 箭头按键的序列
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
                    elif key == '\r':  # 回车键
                        return "enter"
                    elif key == '\x7f':  # 退格键
                        return "backspace"
                    elif key == '\t':  # Tab键
                        return "tab"
                    else:
                        return key
                return None
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        WINDOWS = False
        
    except (ImportError, AttributeError):
        # 无法使用特定平台的库，创建简单替代方案
        def get_key():
            """
            简单的输入捕获（非实时）
            """
            if sys.stdin.isatty():
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    return key
            return None
            
        WINDOWS = False

class KeyboardHandler:
    """
    跨平台的键盘处理类
    """
    def __init__(self, callback: Callable[[str], Any]):
        """
        初始化键盘处理器
        
        Args:
            callback: 按键处理回调函数
        """
        self.callback = callback
        self.running = False
        self.thread = None
    
    def start(self):
        """
        启动键盘监听
        """
        self.running = True
        self.thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        self.thread.start()
    
    def stop(self):
        """
        停止键盘监听
        """
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
    
    def _listen_keyboard(self):
        """
        监听键盘输入
        """
        while self.running:
            key = get_key()
            if key:
                self.callback(key)
            time.sleep(0.01)  # 降低CPU使用率

# 直接测试
if __name__ == "__main__":
    def key_callback(key):
        print(f"按下键: {key}")
        if key == 'q':
            print("退出...")
            return False
        return True
    
    print("按键测试 (按 'q' 退出)...")
    
    handler = KeyboardHandler(key_callback)
    handler.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        handler.stop()