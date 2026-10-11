"""Windows 鼠标 / 按键 / DPI 封装（纯 ctypes，无第三方依赖）。

非 Windows 平台下所有操作都是 dry-run，方便在其他系统上调试逻辑。
"""
import sys
import time
import ctypes

IS_WIN = sys.platform == "win32"


def init_dpi_awareness():
    """让进程以物理像素工作，保证 mss 截图坐标与鼠标坐标一致（必须在截图前调用）。"""
    if not IS_WIN:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


if IS_WIN:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    INPUT_MOUSE = 0
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    def _send_mouse(flags: int):
        inp = INPUT(type=INPUT_MOUSE)
        inp.u.mi = MOUSEINPUT(0, 0, 0, flags, 0, 0)
        if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise OSError(f"SendInput failed, err={ctypes.get_last_error()}")

    def get_cursor_pos():
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def set_cursor_pos(x: int, y: int):
        user32.SetCursorPos(int(x), int(y))

    def key_down(vk: int) -> bool:
        return bool(user32.GetAsyncKeyState(int(vk)) & 0x8000)

else:

    def get_cursor_pos():
        return 0, 0

    def set_cursor_pos(x: int, y: int):
        pass

    def key_down(vk: int) -> bool:
        return False


def double_click(x: int, y: int, gap_ms: int = 60, restore_cursor: bool = True) -> bool:
    """在屏幕绝对坐标 (x, y) 双击左键。返回 True 表示真实执行，False 表示 dry-run。"""
    if not IS_WIN:
        return False
    old = get_cursor_pos() if restore_cursor else None
    set_cursor_pos(x, y)
    time.sleep(0.015)  # 给游戏一点时间识别光标移动
    for i in range(2):
        _send_mouse(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.012)
        _send_mouse(MOUSEEVENTF_LEFTUP)
        if i == 0:
            time.sleep(max(gap_ms, 10) / 1000)
    if old is not None:
        time.sleep(0.03)
        set_cursor_pos(*old)
    return True
