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

    INPUT_KEYBOARD = 1
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008
    # 需要 E0 前缀的扩展键（方向键、Insert/Delete/Home/End/PgUp/PgDn）
    _EXTENDED_VK = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E}

    def _send_key(vk: int, scan: int, flags: int):
        inp = INPUT(type=INPUT_KEYBOARD)
        inp.u.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
        if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise OSError(f"SendInput failed, err={ctypes.get_last_error()}")

    def _press_key_win(vk: int, hold_ms: int):
        # 优先用扫描码：DirectInput / RawInput 的游戏通常只认扫描码
        scan = user32.MapVirtualKeyW(int(vk), 0)  # MAPVK_VK_TO_VSC
        if scan:
            flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if vk in _EXTENDED_VK else 0)
            _send_key(0, scan, flags)
            time.sleep(hold_ms / 1000)
            _send_key(0, scan, flags | KEYEVENTF_KEYUP)
        else:
            _send_key(int(vk), 0, 0)
            time.sleep(hold_ms / 1000)
            _send_key(int(vk), 0, KEYEVENTF_KEYUP)

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


# 常用按键的虚拟键码，网页下拉框也用这张表
KEY_NAMES = {
    0x20: "空格 Space",
    0x08: "退格 Backspace",
    0x0D: "回车 Enter",
    0x1B: "Esc",
    0x09: "Tab",
    0x45: "E",
    0x46: "F",
    0x51: "Q",
    0x52: "R",
    0x31: "1",
    0x32: "2",
    0x33: "3",
}


def press_key(vk: int, hold_ms: int = 40) -> bool:
    """按下并松开一个键。返回 True 表示真实执行，False 表示 dry-run。"""
    if not IS_WIN:
        return False
    _press_key_win(vk, hold_ms)
    return True


def left_click(x=None, y=None, hold_ms: int = 50, restore_cursor: bool = True) -> bool:
    """单击左键。给了 (x, y) 就先移过去，否则在当前光标位置点。返回 True 表示真实执行。"""
    if not IS_WIN:
        return False
    old = None
    if x is not None and y is not None:
        old = get_cursor_pos() if restore_cursor else None
        set_cursor_pos(x, y)
        time.sleep(0.015)
    _send_mouse(MOUSEEVENTF_LEFTDOWN)
    time.sleep(max(hold_ms, 10) / 1000)
    _send_mouse(MOUSEEVENTF_LEFTUP)
    if old is not None:
        time.sleep(0.03)
        set_cursor_pos(*old)
    return True
