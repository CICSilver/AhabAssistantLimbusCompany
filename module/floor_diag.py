"""临时诊断模块：定位镜牢楼层 CLEAR 标记漏检问题。

只做旁路采集，不参与任何生产判定：独立复跑一遍模板匹配并落盘，
便于事后离线复现、调阈值、验证改进效果。确认问题后整个文件可直接删除。

输出全部静默写入 diagnostic/floor/floor_diag.log，不经过项目日志器，
以免刷爆滚动日志（上一轮采集就把 debugLog 挤得轮转了两次）。
"""

import os
import time

import cv2
import numpy as np

from module.config import cfg
from utils.image_utils import ImageUtils

DIAG_ROOT = os.path.join(".", "diagnostic", "floor")
DIAG_LOG = os.path.join(DIAG_ROOT, "floor_diag.log")
MAX_CAPTURES = 40  # 防止长时间挂机把磁盘写满
PEAK_FLOOR = 0.45  # 低到足以看清"落选"的峰值到底有多低
PRINT_TOP = 12
UPSCALE = 2

_seq = 0
_saved_templates = set()


def _write(lines):
    """静默追加到独立日志；写失败也不得影响主流程。"""
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(DIAG_LOG, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{stamp} {line}\n")
    except Exception:
        pass


def _peaks(res, template_w, template_h, floor_score):
    """把匹配响应图聚合成互不重叠的局部极大值，返回 [(x, y, score)]，按 x 排序。"""
    ys, xs = np.where(res >= floor_score)
    if len(xs) == 0:
        return []
    scores = res[ys, xs]
    order = np.argsort(scores)[::-1]
    radius = max(template_w, template_h) * 0.6
    kept = []
    for i in order:
        x, y, s = int(xs[i]), int(ys[i]), float(scores[i])
        if all((x - kx) ** 2 + (y - ky) ** 2 > radius**2 for kx, ky, _ in kept):
            kept.append((x, y, s))
        if len(kept) >= PRINT_TOP:
            break
    # 转成与生产一致的中心点坐标
    kept = [(x + template_w // 2, y + template_h // 2, s) for x, y, s in kept]
    return sorted(kept, key=lambda p: p[0])


def capture(tag, target, threshold, min_dist, production_result):
    """采集一次楼层识别现场。

    Args:
        tag: 场景标签，如 "clear" / "not_passed"
        target: 生产代码使用的模板相对路径
        threshold: 生产调用实际使用的阈值
        min_dist: 生产调用实际使用的去重距离
        production_result: 生产代码拿到的结果（坐标列表或 None）
    """
    global _seq
    try:
        if _seq >= MAX_CAPTURES:
            return
        _seq += 1
        seq = _seq
        os.makedirs(DIAG_ROOT, exist_ok=True)

        # 与生产同一帧：上游刚刷新过 auto.screenshot
        from module.automation import auto

        screenshot = auto.get_screenshot_array()
        if screenshot is None or getattr(screenshot, "size", 0) == 0:
            _write([f"[#{seq}] 截图为空，跳过采集"])
            return

        stamp = time.strftime("%H%M%S")
        shot_path = os.path.join(DIAG_ROOT, f"{seq:03d}_{tag}_{stamp}.png")
        cv2.imwrite(shot_path, screenshot)

        # 复刻生产路径：原生模板 + 双方放大（find_multiple_targets_upscaled）
        template = ImageUtils.load_image(target, resize=False)
        if template is None:
            _write([f"[#{seq}] 模板加载失败: {target}"])
            return
        template_scale = cfg.set_win_size * UPSCALE / 1440
        template = cv2.resize(template, None, fx=template_scale, fy=template_scale, interpolation=cv2.INTER_LINEAR)
        screenshot = cv2.resize(screenshot, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LINEAR)
        if tag not in _saved_templates:
            cv2.imwrite(os.path.join(DIAG_ROOT, f"template_{tag}.png"), template)
            _saved_templates.add(tag)

        th, tw = template.shape[:2]
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        peaks = _peaks(res, tw, th, PEAK_FLOOR)
        above = [p for p in peaks if p[2] >= threshold]
        kept = list(production_result) if production_result else []

        lines = [
            f"[#{seq}] tag={tag} 截图={os.path.basename(shot_path)} "
            f"放大后 screenshot={screenshot.shape} template={tw}x{th} upscale={UPSCALE} "
            f"set_win_size={cfg.set_win_size} threshold={threshold} min_dist={min_dist:.1f}",
            f"[#{seq}] 响应图最高分={float(np.max(res)):.4f} 峰值(按x排序, >= {PEAK_FLOOR}):",
        ]
        for x, y, sc in peaks:
            mark = "命中" if sc >= threshold else "落选"
            lines.append(f"[#{seq}]     x={x // UPSCALE:<6} y={y // UPSCALE:<6} score={sc:.4f}  [{mark}]")
        lines.append(
            f"[#{seq}] 过阈值峰值={len(above)} 生产去重后={len(kept)} (两者相等则 min_dist 无影响) 生产坐标={kept}"
        )
        _write(lines)
    except Exception as e:  # 诊断绝不能影响正常流程
        _write([f"[#{_seq}] 采集异常，已忽略: {type(e).__name__}: {e}"])
