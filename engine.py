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


# ---------------------------------------------------------------- targets
# 三类检测目标各自对应的配置字段
TARGETS = {
    "fish":   {"label": "上鱼图标", "roi": "roi", "template": "template",
               "threshold": "threshold", "grayscale": "grayscale"},
    "keep":   {"label": "入护图标 1", "roi": "keep_roi", "template": "keep_template",
               "threshold": "keep_threshold", "grayscale": "keep_grayscale"},
    # 第二个入护模板：与 1 共用阈值和灰度设置，满足任意一个就入护
    "keep2":  {"label": "入护图标 2", "roi": "keep2_roi", "template": "keep2_template",
               "threshold": "keep_threshold", "grayscale": "keep_grayscale"},
    "landed": {"label": "结算标志", "roi": "landed_roi", "template": "landed_template",
               "threshold": "landed_threshold", "grayscale": "landed_grayscale"},
}


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
        self._tpls = {}           # path -> (mtime, bgr, gray)
        self._restore_migrated_templates()

        self.status = {
            "running": False,
            "score": 0.0,
            "hit": False,
            "fps": 0.0,
            "hits": 0,
            "last_hit": None,
            "error": None,
            "dry_run": not win_input.IS_WIN,
            "phase": "idle",       # idle / landing / keep_wait / keep_check
            "keep_score": None,
            "keep1_score": None,
            "keep2_score": None,
            "landed_score": None,
        }
        self.preview_jpeg = None
        self.keep_preview_jpeg = None
        self.keep2_preview_jpeg = None
        self.landed_preview_jpeg = None
        self.last_full = None      # (monitor_rect, bgr image) 给网页框选用
        self.last_match = None     # 最近一次匹配到的屏幕坐标中心
        self.events = collections.deque(self._load_events(), maxlen=300)

    # ---- template -------------------------------------------------
    def _restore_migrated_templates(self):
        """撤销已回退版本（多模板库）做过的迁移：它把 template.png 改名成 template.png.migrated。"""
        cfg = self.store.snapshot()
        for name in {cfg["template"], cfg["keep_template"], cfg["landed_template"]}:  # keep2 是新加的，无需恢复
            dst = self.base / name
            src = dst.with_name(dst.name + ".migrated")
            if not dst.exists() and src.exists():
                src.rename(dst)
                log.info("已恢复模板 %s", name)

    def _load_template(self, name):
        """按文件名加载模板并缓存；文件被重新保存后自动重载。返回 (bgr, gray) 或 None。"""
        path = self.base / name
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            self._tpls.pop(name, None)
            return None
        cached = self._tpls.get(name)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._tpls[name] = (mtime, bgr, gray)
        log.info("模板已加载 %s %dx%d", name, bgr.shape[1], bgr.shape[0])
        return bgr, gray

    def _template(self, cfg):
        return self._load_template(cfg["template"])

    def save_template(self, rect, target="fish") -> dict:
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
        name = cfg[TARGETS[target]["template"]]
        cv2.imwrite(str(self.base / name), crop)
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

    def _save_snapshot(self, img, box, prefix=""):
        vis = img.copy()
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        name = prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
        cv2.imwrite(str(self.snap_dir / name), vis)
        snaps = sorted(self.snap_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in snaps[:-200]:  # 只保留最近 200 张
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

    # ---- 入护判定 -------------------------------------------------
    def target_ready(self, cfg, target):
        """返回 (roi, (tpl_bgr, tpl_gray)) 或抛出 ValueError 说明缺什么。"""
        spec = TARGETS[target]
        roi = cfg.get(spec["roi"])
        if not roi:
            raise ValueError(f"{spec['label']}：尚未设置检测区域")
        tpl = self._load_template(cfg[spec["template"]])
        if tpl is None:
            raise ValueError(f"{spec['label']}：尚未保存模板")
        th, tw = tpl[1].shape
        if tw > roi["width"] or th > roi["height"]:
            raise ValueError(f"{spec['label']}：模板比检测区域还大")
        return roi, tpl

    def target_score(self, cfg, target):
        """对目标区域截一帧并匹配，返回 (score, img, (x, y, w, h))。"""
        spec = TARGETS[target]
        roi, (tpl_bgr, tpl_gray) = self.target_ready(cfg, target)
        img = self.grabber.grab(roi)
        score, (mx, my) = match(img, tpl_bgr, tpl_gray, cfg[spec["grayscale"]])
        th, tw = tpl_gray.shape
        return score, img, (mx, my, tw, th)

    KEEP_SLOTS = ("keep", "keep2")

    def keep_slots(self, cfg):
        """已配置好的入护模板槽位列表。一个都没有时抛 ValueError。"""
        ready, errors = [], []
        for slot in self.KEEP_SLOTS:
            try:
                self.target_ready(cfg, slot)
                ready.append(slot)
            except ValueError as e:
                errors.append(str(e))
        if not ready:
            raise ValueError("入护判定：入护图标 1 和 2 都未设置好（" + "；".join(errors) + "）")
        return ready

    def keep_ready(self, cfg):
        return self.keep_slots(cfg)

    def keep_score(self, cfg):
        """对所有已设置的入护模板各匹配一次，取最高分。返回 (score, img, box, slot)。"""
        best = None
        for slot in self.keep_slots(cfg):
            score, img, box = self.target_score(cfg, slot)
            if best is None or score > best[0]:
                best = (score, img, box, slot)
        return best

    def wait_landed(self, cfg):
        """双击后等鱼真正拉上来。返回 (ok, info)。ok=False 表示超时或被停止，此时不应按任何键。

        landed_mode:
          fixed     —— 不检测，直接视为已拉上来（旧行为，靠 keep_delay_ms 固定等待）
          marker    —— 等「结算标志」模板连续命中 debounce 帧（比如结算界面上的按键提示、边框）
          fish_gone —— 等上鱼图标连续消失 landed_gone_ms（适用于拉鱼时图标一直在、拉上来才消失的情况）
        """
        mode = cfg["landed_mode"]
        if mode == "fixed":
            return True, {"mode": mode}
        t0 = time.perf_counter()
        deadline = t0 + cfg["landed_timeout_ms"] / 1000
        self.status["phase"] = "landing"
        consec, gone_since, best = 0, None, 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            if now >= deadline:
                return False, {"mode": mode, "waited_ms": int((now - t0) * 1000), "best": round(best, 3)}
            if mode == "marker":
                score, _, _ = self.target_score(cfg, "landed")
                best = max(best, score)
                self.status["landed_score"] = round(score, 4)
                consec = consec + 1 if score >= cfg["landed_threshold"] else 0
                if consec >= cfg["debounce"]:
                    return True, {"mode": mode, "waited_ms": int((now - t0) * 1000), "score": round(score, 3)}
            else:  # fish_gone
                score, _, _ = self.target_score(cfg, "fish")
                if score >= cfg["threshold"]:
                    gone_since = None
                elif gone_since is None:
                    gone_since = now
                elif (now - gone_since) * 1000 >= cfg["landed_gone_ms"]:
                    return True, {"mode": mode, "waited_ms": int((now - t0) * 1000)}
            self._sleep(cfg["interval_ms"] / 1000)
        return False, {"mode": mode, "stopped": True}

    def _sleep(self, seconds):
        """可被停止信号打断的 sleep。"""
        self._stop.wait(max(0.0, seconds))

    def run_keep_stage(self, cfg, press=True):
        """双击上鱼后调用：等 keep_delay_ms，然后在 keep_window_ms 内反复检测入护图标。
        任一帧分数 ≥ 阈值 → 按 keep_key_present（入护）；窗口结束仍未命中 → 按 keep_key_absent（释放）。
        """
        try:
            self.keep_ready(cfg)
            if cfg["landed_mode"] == "marker":
                self.target_ready(cfg, "landed")
        except ValueError as e:
            self.status["error"] = str(e)
            return self._record("keep_skipped", reason=str(e))

        landed, info = self.wait_landed(cfg)
        if not landed:
            # 没等到拉上来：宁可什么都不按，也不要在拉鱼途中误按释放
            self.status["phase"] = "idle"
            return self._record("keep_timeout", **info)
        if cfg["landed_mode"] != "fixed":
            self._record("landed", **info)

        self.status["phase"] = "keep_wait"
        self._sleep(cfg["keep_delay_ms"] / 1000)
        self.status["phase"] = "keep_check"

        deadline = time.perf_counter() + cfg["keep_window_ms"] / 1000
        best, best_img, best_box, best_slot = -1.0, None, None, None
        present = False
        checks = 0
        while not self._stop.is_set():
            score, img, box, slot = self.keep_score(cfg)
            checks += 1
            if score > best:
                best, best_img, best_box, best_slot = score, img, box, slot
            if score >= cfg["keep_threshold"]:
                present = True
                break
            if time.perf_counter() >= deadline:
                break
            self._sleep(min(cfg["interval_ms"], 100) / 1000)
        self.status["phase"] = "idle"

        key = cfg["keep_key_present"] if present else cfg["keep_key_absent"]
        real = win_input.press_key(key, cfg["key_hold_ms"]) if press else False
        snap = None
        if cfg["save_snapshots"] and best_img is not None:
            snap = self._save_snapshot(best_img, best_box, prefix="keep_")
        return self._record(
            "keep",
            decision="keep" if present else "release",
            score=round(best, 3),
            matched=TARGETS[best_slot]["label"] if best_slot else None,
            checks=checks,
            key=win_input.KEY_NAMES.get(key, hex(key)),
            dry_run=not real,
            snapshot=snap,
        )

    def set_enabled(self, on: bool, source="web"):
        self.store.update({"enabled": on})
        self._record("resume" if on else "pause", source=source)

    def _update_keep_preview(self, cfg):
        """空闲时也刷新入护 / 结算标志区域的分数和预览，方便在网页上调阈值。"""
        status_key = {"keep": "keep1_score", "keep2": "keep2_score", "landed": "landed_score"}
        for target in ("keep", "keep2", "landed"):
            spec = TARGETS[target]
            try:
                score, img, (x, y, w, h) = self.target_score(cfg, target)
            except ValueError:
                self.status[status_key[target]] = None
                setattr(self, f"{target}_preview_jpeg", None)
                continue
            self.status[status_key[target]] = round(score, 4)
            vis = np.ascontiguousarray(img).copy()
            color = (0, 200, 0) if score >= cfg[spec["threshold"]] else (160, 160, 160)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                setattr(self, f"{target}_preview_jpeg", buf.tobytes())
        ks = [v for v in (self.status["keep1_score"], self.status["keep2_score"]) if v is not None]
        self.status["keep_score"] = max(ks) if ks else None

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
        armed = True          # 触发后上锁，图标消失足够久才重新允许触发
        gone_since = None     # 上锁期间图标开始消失的时刻
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
            # 拉鱼过程中上鱼图标往往还在屏幕上；只靠冷却的话，冷却一结束就会再双击一次，打断拉鱼。
            # 所以触发后上锁，必须看到图标连续消失 rearm_gone_ms 才解锁。
            if not armed:
                if hit:
                    gone_since = None
                elif gone_since is None:
                    gone_since = now
                elif (now - gone_since) * 1000 >= cfg["rearm_gone_ms"]:
                    armed, gone_since = True, None
            self.status["armed"] = armed
            if (cfg["enabled"] and armed and consec >= cfg["debounce"]
                    and (now - last_fire) * 1000 >= cfg["cooldown_ms"]):
                tx, ty = self.click_target(cfg, center)
                snap = self._save_snapshot(img, (mx, my, tw, th)) if cfg["save_snapshots"] else None
                info = self.fire(tx, ty, cfg, reason="hit", score=round(score, 3))
                self._record("double_click", snapshot=snap, **info)
                self.status["hits"] += 1
                self.status["last_hit"] = self.events[-1]["ts"]
                if cfg["keep_enabled"]:
                    self.run_keep_stage(cfg)
                last_fire = time.perf_counter()  # 冷却从整轮（双击 + 入护）结束后开始算
                consec = 0
                armed, gone_since = False, None

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
                self._update_keep_preview(cfg)
                last_preview = now

            spent = time.perf_counter() - t0
            time.sleep(max(0.0, cfg["interval_ms"] / 1000 - spent))
