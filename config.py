"""配置加载 / 校验 / 原子保存。"""
import json
import os
import threading
from pathlib import Path

DEFAULTS = {
    "host": "0.0.0.0",
    "port": 8765,
    "token": "",             # 非空时访问网页需带 ?token=xxx
    "monitor": 1,            # mss 显示器编号，1 = 主屏
    "roi": None,             # 检测区域 {left, top, width, height}，屏幕绝对物理像素
    "template": "template.png",
    "threshold": 0.85,       # 匹配分数阈值 (0~1)
    "grayscale": True,       # 灰度匹配更快；图标靠颜色区分时改 False
    "debounce": 2,           # 连续命中 N 帧才触发
    "cooldown_ms": 3000,     # 触发后冷却时间
    "rearm_gone_ms": 500,    # 触发后，图标要连续消失这么久才允许下一次触发
    "interval_ms": 50,       # 检测间隔
    "click_mode": "match_center",  # match_center = 点图标中心；fixed = 点固定坐标
    "click_point": None,     # fixed 模式的坐标 {x, y}
    "click_delay_ms": 0,     # 命中后延迟多久再点
    "double_click_gap_ms": 60,
    "restore_cursor": True,  # 点完把鼠标移回原位
    "hotkey_vk": 119,        # 暂停/恢复热键，119 = F8
    "enabled": False,        # 启动时默认暂停，确认配置后再开
    "save_snapshots": True,  # 每次触发保存一张 ROI 截图，便于排查误触发

    # ---- 入护判定：双击上鱼后，看某区域有没有入护图标 ----
    "keep_enabled": False,
    "keep_roi": None,                  # 入护图标所在区域
    "keep_template": "keep_template.png",
    "keep_threshold": 0.85,
    "keep_grayscale": True,
    "keep_delay_ms": 1500,             # 双击后等多久再开始判定（等结算界面出来）
    "keep_window_ms": 1500,            # 判定窗口：窗口内任一帧命中 → 入护；整段都没命中 → 释放
    "keep_key_present": 0x20,          # 有图标时按的键（空格）
    "keep_key_absent": 0x08,           # 没有图标时按的键（Backspace）
    "key_hold_ms": 40,                 # 按键按住时长

    # ---- 等鱼真正拉上来再判定入护 ----
    "landed_mode": "fixed",            # fixed = 固定等 keep_delay_ms；marker = 等结算标志出现；fish_gone = 等上鱼图标消失
    "landed_roi": None,
    "landed_template": "landed_template.png",
    "landed_threshold": 0.85,
    "landed_grayscale": True,
    "landed_gone_ms": 500,             # fish_gone 模式：上鱼图标连续消失多久算拉上来
    "landed_timeout_ms": 60000,        # 最多等多久；超时不按任何键
}

# 允许通过网页修改的字段及其类型
EDITABLE = {
    "monitor": int,
    "roi": dict,
    "threshold": float,
    "grayscale": bool,
    "debounce": int,
    "cooldown_ms": int,
    "rearm_gone_ms": int,
    "interval_ms": int,
    "click_mode": str,
    "click_point": dict,
    "click_delay_ms": int,
    "double_click_gap_ms": int,
    "restore_cursor": bool,
    "hotkey_vk": int,
    "enabled": bool,
    "save_snapshots": bool,
    "keep_enabled": bool,
    "keep_roi": dict,
    "keep_threshold": float,
    "keep_grayscale": bool,
    "keep_delay_ms": int,
    "keep_window_ms": int,
    "keep_key_present": int,
    "keep_key_absent": int,
    "key_hold_ms": int,
    "landed_mode": str,
    "landed_roi": dict,
    "landed_threshold": float,
    "landed_grayscale": bool,
    "landed_gone_ms": int,
    "landed_timeout_ms": int,
}

RANGES = {
    "threshold": (0.3, 1.0),
    "debounce": (1, 50),
    "cooldown_ms": (0, 600_000),
    "rearm_gone_ms": (0, 60_000),
    "interval_ms": (10, 5000),
    "click_delay_ms": (0, 10_000),
    "double_click_gap_ms": (10, 400),
    "monitor": (0, 16),
    "keep_threshold": (0.3, 1.0),
    "keep_delay_ms": (0, 30_000),
    "keep_window_ms": (50, 30_000),
    "keep_key_present": (1, 254),
    "keep_key_absent": (1, 254),
    "key_hold_ms": (10, 1000),
    "landed_threshold": (0.3, 1.0),
    "landed_gone_ms": (50, 30_000),
    "landed_timeout_ms": (1000, 600_000),
}


class ConfigError(ValueError):
    pass


def _check_rect(v):
    try:
        r = {k: int(v[k]) for k in ("left", "top", "width", "height")}
    except Exception:
        raise ConfigError("roi 需要 left/top/width/height 四个整数")
    if r["width"] < 4 or r["height"] < 4:
        raise ConfigError("区域太小（至少 4x4 像素）")
    return r


def _check_point(v):
    try:
        return {"x": int(v["x"]), "y": int(v["y"])}
    except Exception:
        raise ConfigError("click_point 需要 x/y 两个整数")


class ConfigStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        data = dict(DEFAULTS)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data.update(json.load(f))
        self._data = data
        self._save()

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def update(self, patch: dict) -> dict:
        clean = {}
        for k, v in patch.items():
            if k not in EDITABLE:
                raise ConfigError(f"不可修改的字段: {k}")
            if v is None and k in ("roi", "click_point", "keep_roi", "landed_roi"):
                clean[k] = None
                continue
            typ = EDITABLE[k]
            if k in ("roi", "keep_roi", "landed_roi"):
                v = _check_rect(v)
            elif k == "click_point":
                v = _check_point(v)
            elif typ is bool:
                if not isinstance(v, bool):
                    raise ConfigError(f"{k} 需要布尔值")
            else:
                try:
                    v = typ(v)
                except Exception:
                    raise ConfigError(f"{k} 类型错误")
            if k in RANGES:
                lo, hi = RANGES[k]
                if not lo <= v <= hi:
                    raise ConfigError(f"{k} 超出范围 [{lo}, {hi}]")
            if k == "landed_mode" and v not in ("fixed", "marker", "fish_gone"):
                raise ConfigError("landed_mode 只能是 fixed / marker / fish_gone")
            if k == "click_mode" and v not in ("match_center", "fixed"):
                raise ConfigError("click_mode 只能是 match_center 或 fixed")
            clean[k] = v
        with self._lock:
            self._data.update(clean)
            self._save()
            return json.loads(json.dumps(self._data))

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
