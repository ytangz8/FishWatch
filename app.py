"""FishWatch：检测屏幕区域出现上鱼图标后自动双击。

运行: python app.py
访问: http://<本机IP>:8765  （Mac / 手机浏览器均可）
"""
import logging
import secrets
import socket
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import win_input

win_input.init_dpi_awareness()  # 必须在任何截图之前

import cv2  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402

from config import ConfigError, ConfigStore  # noqa: E402
from engine import TARGETS, Engine  # noqa: E402

BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- logging
log = logging.getLogger("fishwatch")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_fh = RotatingFileHandler(BASE / "fishwatch.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)

store = ConfigStore(BASE / "config.json")
engine = Engine(store, BASE)
app = FastAPI(title="FishWatch")


# ---------------------------------------------------------------- auth
@app.middleware("http")
async def token_auth(request: Request, call_next):
    token = store.snapshot()["token"]
    path = request.url.path
    if token and path.startswith("/api/"):
        given = request.headers.get("x-token") or request.query_params.get("token") or ""
        if not secrets.compare_digest(given, token):
            return JSONResponse({"detail": "token 错误"}, status_code=401)
    return await call_next(request)


def _jpeg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "jpeg 编码失败")
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


# ---------------------------------------------------------------- api
@app.get("/api/state")
def state():
    cfg = store.snapshot()
    cfg.pop("token", None)
    tpl = BASE / cfg["template"]
    return {
        "status": engine.status,
        "config": cfg,
        "template_exists": tpl.exists(),
        "keep_template_exists": (BASE / cfg["keep_template"]).exists(),
        "keep2_template_exists": (BASE / cfg["keep2_template"]).exists(),
        "landed_template_exists": (BASE / cfg["landed_template"]).exists(),
        "key_names": {str(k): v for k, v in win_input.KEY_NAMES.items()},
        "events": list(engine.events)[-50:][::-1],
    }


@app.post("/api/config")
async def update_config(request: Request):
    patch = await request.json()
    try:
        enabled = patch.pop("enabled", None)
        cfg = store.update(patch) if patch else store.snapshot()
        if enabled is not None and enabled != cfg["enabled"]:
            engine.set_enabled(bool(enabled), source="web")
    except ConfigError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/api/monitors")
def monitors():
    return engine.grabber.monitors()


@app.get("/api/screenshot")
def screenshot(mon: int = 1):
    try:
        rect, img = engine.full_screenshot(mon)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _jpeg(img, 85)


@app.get("/api/screenshot/meta")
def screenshot_meta():
    if engine.last_full is None:
        raise HTTPException(404, "还没有截图")
    return engine.last_full[0]


@app.get("/api/preview.jpg")
def preview():
    if engine.preview_jpeg is None:
        raise HTTPException(404, "暂无预览")
    return Response(engine.preview_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/keep_preview.jpg")
def keep_preview():
    if engine.keep_preview_jpeg is None:
        raise HTTPException(404, "暂无预览")
    return Response(engine.keep_preview_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/keep2_preview.jpg")
def keep2_preview():
    if engine.keep2_preview_jpeg is None:
        raise HTTPException(404, "暂无预览")
    return Response(engine.keep2_preview_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.delete("/api/template")
def clear_template(target: str):
    """清除某个目标的模板和检测区域（目前只开放给可选的 keep2）。"""
    if target != "keep2":
        raise HTTPException(400, "只能清除入护图标 2")
    cfg = store.snapshot()
    (BASE / cfg["keep2_template"]).unlink(missing_ok=True)
    store.update({"keep2_roi": None})
    engine._record("template_cleared", target=target)
    return {"ok": True}


@app.get("/api/landed_preview.jpg")
def landed_preview():
    if engine.landed_preview_jpeg is None:
        raise HTTPException(404, "暂无预览")
    return Response(engine.landed_preview_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/template.png")
def template_img(target: str = "fish"):
    cfg = store.snapshot()
    if target not in TARGETS:
        raise HTTPException(400, "未知目标")
    p = BASE / cfg[TARGETS[target]["template"]]
    if not p.exists():
        raise HTTPException(404, "没有模板")
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.post("/api/template")
async def save_template(request: Request, target: str = "fish"):
    if target not in TARGETS:
        raise HTTPException(400, "target 只能是 " + " / ".join(TARGETS))
    rect = await request.json()
    try:
        r = {k: int(rect[k]) for k in ("left", "top", "width", "height")}
        if r["width"] < 4 or r["height"] < 4:
            raise ValueError("模板太小")
        info = engine.save_template(r, target)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    # 若还没有对应的检测区域，自动以模板为中心向外扩 60px
    roi_key = TARGETS[target]["roi"]
    if not store.snapshot().get(roi_key):
        mon = engine.last_full[0]
        pad = 60
        left = max(mon["left"], r["left"] - pad)
        top = max(mon["top"], r["top"] - pad)
        right = min(mon["left"] + mon["width"], r["left"] + r["width"] + pad)
        bottom = min(mon["top"] + mon["height"], r["top"] + r["height"] + pad)
        store.update({roi_key: {"left": left, "top": top, "width": right - left, "height": bottom - top}})
    engine._record("template_saved", target=target, **info)
    return {"ok": True, **info}


@app.get("/api/snapshot/{name}")
def snapshot(name: str):
    p = (engine.snap_dir / name).resolve()
    if p.parent != engine.snap_dir.resolve() or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.post("/api/test_click")
def test_click(delay: float = 3.0):
    """延迟几秒后在目标位置双击一次，用来确认游戏能接收模拟点击。"""
    cfg = store.snapshot()
    target = engine.click_target(cfg)
    if not target:
        raise HTTPException(400, "没有可用的点击位置（先设置检测区域或固定点）")
    delay = min(max(delay, 0), 10)

    def run():
        import time
        time.sleep(delay)
        info = engine.fire(target[0], target[1], cfg, reason="test")
        engine._record("double_click", **info)

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "x": target[0], "y": target[1], "delay": delay}


@app.post("/api/test_keep")
def test_keep():
    """立即对入护区域判定一次，只返回结果，不按键。"""
    cfg = store.snapshot()
    try:
        score, _, _, slot = engine.keep_score(cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    present = score >= cfg["keep_threshold"]
    key = cfg["keep_key_present"] if present else cfg["keep_key_absent"]
    return {
        "score": round(score, 4),
        "decision": "keep" if present else "release",
        "matched": TARGETS[slot]["label"],
        "key": win_input.KEY_NAMES.get(key, hex(key)),
    }


# ---------------------------------------------------------------- main
def _lan_ips():
    """列出本机所有 IPv4。默认路由那块网卡可能是 VPN / 虚拟网卡，所以全部列出。"""
    ips = set()
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    ips.discard("127.0.0.1")
    # 常见局域网网段排前面
    return sorted(ips, key=lambda ip: (not ip.startswith(("192.168.", "10.")), ip)) or ["127.0.0.1"]


if __name__ == "__main__":
    cfg = store.snapshot()
    engine.start()
    q = f"?token={cfg['token']}" if cfg["token"] else ""
    for ip in _lan_ips():
        log.info("控制台: http://%s:%d/%s", ip, cfg["port"], q)
    if engine.status["dry_run"]:
        log.warning("非 Windows 平台：点击为 dry-run，只记录不执行")
    uvicorn.run(app, host=cfg["host"], port=cfg["port"], log_level="warning")
