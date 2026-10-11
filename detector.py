"""模板库 + 带遮罩的模板匹配。

背景会变的问题：普通模板匹配会把图标周围的背景像素也算进去，背景一换分数就掉。
这里给每个模板自动生成一张遮罩（只保留图标本身的像素），匹配时忽略背景；
另外支持同一目标保存多张模板（不同背景下各截一张），每帧取最高分。
"""
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger("fishwatch")

TARGETS = ("fish", "keep")
_NAME_RE = re.compile(r"^\d{3}\.png$")
_K3 = np.ones((3, 3), np.uint8)


# ---------------------------------------------------------------- mask
def auto_mask(bgr: np.ndarray, sensitivity: float) -> Optional[np.ndarray]:
    """用模板边缘一圈估计背景色，把与背景色差异大的像素当作图标，返回 0/255 遮罩。

    要求截模板时图标四周留几个像素的背景。生成失败（几乎全是背景或几乎没有背景）时返回 None，
    表示退化为不用遮罩。
    """
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(1, min(h, w) // 12)
    ring = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(ring, axis=0)
    # 背景本身有纹理（水面、草地）时自动抬高阈值
    texture = float(np.median(np.linalg.norm(ring - bg, axis=1)))
    dist = np.linalg.norm(lab - bg, axis=2)
    m = (dist > sensitivity + 2.0 * texture).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K3)

    # 填洞：图标内部恰好和背景同色的部分（比如深色背景上的黑色描边）也算图标
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    m = cv2.bitwise_or(m, holes)

    # 去掉一圈抗锯齿边缘：这些像素是图标和背景的混色，会随背景变化
    if min(h, w) >= 16:
        m = cv2.erode(m, _K3)

    ratio = float(np.count_nonzero(m)) / m.size
    if ratio < 0.05 or ratio > 0.97:
        return None
    return m


# ---------------------------------------------------------------- templates
@dataclass
class Template:
    name: str
    bgr: np.ndarray
    gray: np.ndarray
    mask: Optional[np.ndarray]

    @property
    def size(self):
        return self.gray.shape[1], self.gray.shape[0]

    @property
    def mask_ratio(self) -> float:
        return 1.0 if self.mask is None else float(np.count_nonzero(self.mask)) / self.mask.size


class TemplateLibrary:
    """templates/<target>/NNN.png。文件变化或遮罩参数变化时自动重新加载。"""

    def __init__(self, root: Path):
        self.root = root
        for t in TARGETS:
            (root / t).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache = {}  # target -> (key, [Template])

    def _dir(self, target) -> Path:
        if target not in TARGETS:
            raise ValueError(f"未知目标 {target}")
        return self.root / target

    def files(self, target) -> List[Path]:
        return sorted(p for p in self._dir(target).glob("*.png") if _NAME_RE.match(p.name))

    def path(self, target, name) -> Path:
        if not _NAME_RE.match(name):
            raise ValueError("模板名不合法")
        return self._dir(target) / name

    def load(self, target, mask_mode="auto", sensitivity=30) -> List[Template]:
        files = self.files(target)
        key = (tuple((p.name, p.stat().st_mtime) for p in files), mask_mode, float(sensitivity))
        with self._lock:
            cached = self._cache.get(target)
            if cached and cached[0] == key:
                return cached[1]
            out = []
            for p in files:
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                mask = auto_mask(bgr, sensitivity) if mask_mode == "auto" else None
                out.append(Template(p.name, bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask))
            self._cache[target] = (key, out)
            if out:
                log.info("模板已加载 %s: %s", target,
                         ", ".join(f"{t.name} {t.size[0]}x{t.size[1]} 遮罩{t.mask_ratio:.0%}" for t in out))
            return out

    def add(self, target, crop: np.ndarray) -> str:
        nums = [int(p.stem) for p in self.files(target)]
        name = f"{(max(nums) + 1) if nums else 1:03d}.png"
        cv2.imwrite(str(self._dir(target) / name), crop)
        return name

    def delete(self, target, name):
        self.path(target, name).unlink(missing_ok=True)

    def import_legacy(self, legacy: Path, target):
        """把旧版单文件模板（template.png / keep_template.png）迁移成模板库里的第一张。"""
        if legacy.exists() and not self.files(target):
            img = cv2.imread(str(legacy), cv2.IMREAD_COLOR)
            if img is not None:
                self.add(target, img)
                legacy.rename(legacy.with_suffix(".png.migrated"))
                log.info("已迁移旧模板 %s -> templates/%s/", legacy.name, target)

    def preview_png(self, target, name, mask_mode, sensitivity) -> bytes:
        """返回带透明通道的 PNG：被遮罩排除的背景像素透明。"""
        tpl = next((t for t in self.load(target, mask_mode, sensitivity) if t.name == name), None)
        if tpl is None:
            raise FileNotFoundError(name)
        alpha = tpl.mask if tpl.mask is not None else np.full(tpl.gray.shape, 255, np.uint8)
        bgra = np.dstack([tpl.bgr, alpha])
        ok, buf = cv2.imencode(".png", bgra)
        if not ok:
            raise RuntimeError("png 编码失败")
        return buf.tobytes()


# ---------------------------------------------------------------- match
@dataclass
class MatchResult:
    score: float
    x: int            # 最佳位置左上角（相对检测区域）
    y: int
    w: int
    h: int
    name: Optional[str]


def match_set(img_bgr: np.ndarray, templates: List[Template], grayscale: bool) -> MatchResult:
    """用一组模板在 img 中匹配，返回分数最高的那个。"""
    best = MatchResult(0.0, 0, 0, 0, 0, None)
    if not templates:
        return best
    src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if grayscale else np.ascontiguousarray(img_bgr)
    ih, iw = src.shape[:2]
    for t in templates:
        tw, th = t.size
        if tw > iw or th > ih:
            continue
        tpl = t.gray if grayscale else t.bgr
        if t.mask is not None:
            res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED, mask=t.mask)
        else:
            res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
        # 纯色区域会算出 NaN / inf
        np.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(res, -1.0, 1.0, out=res)
        _, max_v, _, (mx, my) = cv2.minMaxLoc(res)
        if max_v > best.score or best.name is None:
            best = MatchResult(float(max_v), mx, my, tw, th, t.name)
    return best
