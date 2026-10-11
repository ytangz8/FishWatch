"""截图 → 模板匹配 → 双击 的主循环。"""
import collections
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import win_input

log = logging.getLogger("fishwatch")


# ---------------------------------------------------------------- capture
class MssGrabber:
    """mss 截图（GDI）。mss 实例不能跨线程，每个线程各建一个。"""

    def __init__(self):
        import mss
        self._mss = mss
        self._local = threading.local()

    def _sct(self):
        if not hasattr(self._local, "sct"):
            self._local.sct = self._mss.mss()
        return self._local.sct

    def monitors(self):
        return [dict(m) for m in self._sct().monitors]

    def grab(self, rect) -> np.ndarray:
        shot = self._sct().grab(rect)
        return np.asarray(shot)[:, :, :3]  # BGRA -> BGR（视图，不拷贝）


class ImageGrabber:
    """调试用：把一张图片当作屏幕（设置环境变量 FISHWATCH_FAKE_SCREEN=xxx.png）。"""

    def __init__(self, path):
        self.path = path

    def _img(self):
        img = cv2.imread(self.path)
        if img is None:
            raise RuntimeError(f"无法读取 {self.path}")
        return img

    def monitors(self):
        h, w = self._img().shape[:2]
        m = {"left": 0, "top": 0, "width": w, "height": h}
        return [m, dict(m)]

    def grab(self, rect) -> np.ndarray:
        img = self._img()
        l, t = rect["left"], rect["top"]
        return img[t:t + rect["height"], l:l + rect["width"]].copy()


def make_grabber():
    fake = os.environ.get("FISHWATCH_FAKE_SCREEN")
    return ImageGrabber(fake) if fake else MssGrabber()


# ---------------------------------------------------------------- detect
def match(img_bgr: np.ndarray, tpl_bgr: np.ndarray, tpl_gray: np.ndarray, grayscale: bool):
    """返回 (score, (x, y))，(x, y) 为模板左上角在 img 中的位置。"""
    if grayscale:
        src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        tpl = tpl_gray
    else:
        src = np.ascontiguousarray(img_bgr)
        tpl = tpl_bgr
    res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_v, _, max_loc = cv2.minMaxLoc(res)
    if not np.isfinite(max_v):  # 纯色区域会产生 NaN
        max_v = 0.0
    return float(max_v), max_loc


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, store, base_dir: Path, grabber=None):
        self.store = store
        self.base = base_dir
        self.grabber = grabber or make_grabber()
        self.snap_dir = base_dir / "snapshots"
        self.snap_dir.mkdir(exist_ok=True)
        self.events_path = base_dir / "events.jsonl"

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tpl = None          # (bgr, gray)
        self._tpl_key = None      # (path, mtime)

        self.status = {
            "running": False,
            "score": 0.0,
            "hit": False,
            "fps": 0.0,
            "hits": 0,
            "last_hit": None,
            "error": None,
            "dry_run": not win_input.IS_WIN,
        }
        self.preview_jpeg = None
        self.last_full = None      # (monitor_rect, bgr image) 给网页框选用
        self.last_match = None     # 最近一次匹配到的屏幕坐标中心
        self.events = collections.deque(self._load_events(), maxlen=300)

    # ---- template -------------------------------------------------
    def _template(self, cfg):
        path = self.base / cfg["template"]
        try:
            key = (str(path), path.stat().st_mtime)
        except FileNotFoundError:
            self._tpl, self._tpl_key = None, None
            return None
        if key != self._tpl_key:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            self._tpl = (bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            self._tpl_key = key
            log.info("模板已加载 %s %dx%d", path.name, bgr.shape[1], bgr.shape[0])
        return self._tpl

    def save_template(self, rect) -> dict:
        """从最近一次全屏截图中裁出模板（图标可能已消失，所以不重新截图）。"""
        if self.last_full is None:
            raise ValueError("请先在网页上刷新截图")
        mon, img = self.last_full
        x = rect["left"] - mon["left"]
        y = rect["top"] - mon["top"]
        w, h = rect["width"], rect["height"]
        if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
            raise ValueError("选区超出截图范围")
        crop = img[y:y + h, x:x + w]
        cfg = self.store.snapshot()
        cv2.imwrite(str(self.base / cfg["template"]), crop)
        return {"width": w, "height": h}

    # ---- screenshot for UI ---------------------------------------
    def full_screenshot(self, monitor_idx: int):
        mons = self.grabber.monitors()
        if not 0 <= monitor_idx < len(mons):
            raise ValueError(f"显示器编号无效，可用 0~{len(mons) - 1}")
        mon = {k: mons[monitor_idx][k] for k in ("left", "top", "width", "height")}
        img = np.ascontiguousarray(self.grabber.grab(mon))
        self.last_full = (mon, img)
        return mon, img

    # ---- events --------------------------------------------------
    def _load_events(self):
        if not self.events_path.exists():
            return []
        out = []
        with open(self.events_path, encoding="utf-8") as f:
            for line in f.readlines()[-300:]:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out

    def _record(self, kind, **kw):
        ev = {"ts": datetime.now().isoformat(timespec="milliseconds"), "kind": kind, **kw}
        self.events.append(ev)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        log.info("%s %s", kind, kw)
        return ev

    def _save_snapshot(self, img, box):
        vis = img.copy()
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
        cv2.imwrite(str(self.snap_dir / name), vis)
        snaps = sorted(self.snap_dir.glob("*.jpg"))
        for old in snaps[:-100]:  # 只保留最近 100 张
            old.unlink(missing_ok=True)
        return name

    # ---- actions -------------------------------------------------
    def fire(self, x, y, cfg, reason="hit", score=None):
        if cfg["click_delay_ms"]:
            time.sleep(cfg["click_delay_ms"] / 1000)
        real = win_input.double_click(x, y, cfg["double_click_gap_ms"], cfg["restore_cursor"])
        return {"x": x, "y": y, "dry_run": not real, "reason": reason, "score": score}

    def click_target(self, cfg, match_center=None):
        if cfg["click_mode"] == "fixed" and cfg.get("click_point"):
            return cfg["click_point"]["x"], cfg["click_point"]["y"]
        if match_center:
            return match_center
        if self.last_match:
            return self.last_match
        r = cfg.get("roi")
        if r:
            return r["left"] + r["width"] // 2, r["top"] + r["height"] // 2
        return None

    def set_enabled(self, on: bool, source="web"):
        self.store.update({"enabled": on})
        self._record("resume" if on else "pause", source=source)

    # ---- main loop -----------------------------------------------
    def start(self):
        t = threading.Thread(target=self._run_safe, name="engine", daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def _run_safe(self):
        self.status["running"] = True
        while not self._stop.is_set():
            try:
                self._loop()
            except Exception as e:  # 不让异常杀死线程
                log.exception("engine error")
                self.status["error"] = f"{type(e).__name__}: {e}"
                time.sleep(1)
        self.status["running"] = False

    def _loop(self):
        consec = 0
        last_fire = 0.0
        frames, fps_t0 = 0, time.perf_counter()
        last_preview = 0.0
        hotkey_prev = False

        while not self._stop.is_set():
            t0 = time.perf_counter()
            cfg = self.store.snapshot()

            # 热键：按下沿切换暂停
            hk = win_input.key_down(cfg["hotkey_vk"])
            if hk and not hotkey_prev:
                self.set_enabled(not cfg["enabled"], source="hotkey")
                cfg = self.store.snapshot()
            hotkey_prev = hk

            roi = cfg.get("roi")
            tpl = self._template(cfg)
            if not roi:
                self.status["error"] = "尚未设置检测区域"
                time.sleep(0.3)
                continue
            if tpl is None:
                self.status["error"] = "尚未保存模板"
                time.sleep(0.3)
                continue
            tpl_bgr, tpl_gray = tpl
            th, tw = tpl_gray.shape
            if tw > roi["width"] or th > roi["height"]:
                self.status["error"] = "模板比检测区域还大，请把检测区域框大一些"
                time.sleep(0.3)
                continue

            img = self.grabber.grab(roi)
            score, (mx, my) = match(img, tpl_bgr, tpl_gray, cfg["grayscale"])
            hit = score >= cfg["threshold"]
            consec = consec + 1 if hit else 0
            center = (roi["left"] + mx + tw // 2, roi["top"] + my + th // 2)
            if hit:
                self.last_match = center

            now = time.perf_counter()
            if (cfg["enabled"] and consec >= cfg["debounce"]
                    and (now - last_fire) * 1000 >= cfg["cooldown_ms"]):
                tx, ty = self.click_target(cfg, center)
                snap = self._save_snapshot(img, (mx, my, tw, th)) if cfg["save_snapshots"] else None
                info = self.fire(tx, ty, cfg, reason="hit", score=round(score, 3))
                self._record("double_click", snapshot=snap, **info)
                self.status["hits"] += 1
                self.status["last_hit"] = self.events[-1]["ts"]
                last_fire = time.perf_counter()
                consec = 0

            # 状态 & 预览
            self.status.update(score=round(score, 4), hit=hit, error=None)
            frames += 1
            if now - fps_t0 >= 1.0:
                self.status["fps"] = round(frames / (now - fps_t0), 1)
                frames, fps_t0 = 0, now
            if now - last_preview >= 0.2:
                vis = np.ascontiguousarray(img).copy()
                color = (0, 200, 0) if hit else (160, 160, 160)
                cv2.rectangle(vis, (mx, my), (mx + tw, my + th), color, 2)
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.preview_jpeg = buf.tobytes()
                last_preview = now

            spent = time.perf_counter() - t0
            time.sleep(max(0.0, cfg["interval_ms"] / 1000 - spent))
