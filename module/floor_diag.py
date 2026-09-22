"""临时诊断模块：定位镜牢楼层 CLEAR 标记漏检问题。

只做旁路采集，不参与任何生产判定：独立复跑一遍模板匹配并落盘，
便于事后离线复现、调阈值、验证改进效果。确认问题后整个文件可直接删除。
"""

import os
import time

import cv2
import numpy as np

from module.config import cfg
from module.logger import log
from utils.image_utils import ImageUtils

DIAG_ROOT = os.path.join(".", "diagnostic", "floor")
MAX_CAPTURES = 40  # 防止长时间挂机把磁盘写满
PEAK_FLOOR = 0.45  # 低到足以看清"落选"的峰值到底有多低
PRINT_TOP = 12

_seq = 0
_saved_templates = set()


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

        # 与生产同一帧：find_element(take_screenshot=True) 刚刷新过 auto.screenshot
        from module.automation import auto

        screenshot = auto.get_screenshot_array()
        if screenshot is None or getattr(screenshot, "size", 0) == 0:
            log.warning(f"[FLOORDIAG#{seq}] 截图为空，跳过采集")
            return

        stamp = time.strftime("%H%M%S")
        shot_path = os.path.join(DIAG_ROOT, f"{seq:03d}_{tag}_{stamp}.png")
        cv2.imwrite(shot_path, screenshot)

        # 完全按生产方式加载模板（load_image 内含 set_win_size 缩放）
        template = ImageUtils.load_image(target)
        if template is None:
            log.warning(f"[FLOORDIAG#{seq}] 模板加载失败: {target}")
            return
        if tag not in _saved_templates:
            cv2.imwrite(os.path.join(DIAG_ROOT, f"template_{tag}.png"), template)
            _saved_templates.add(tag)

        th, tw = template.shape[:2]
        log.info(
            f"[FLOORDIAG#{seq}] tag={tag} 截图={shot_path} "
            f"screenshot_shape={screenshot.shape} template={tw}x{th} "
            f"set_win_size={cfg.set_win_size} threshold={threshold} min_dist={min_dist:.1f}"
        )

        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        peaks = _peaks(res, tw, th, PEAK_FLOOR)
        log.info(f"[FLOORDIAG#{seq}] 响应图最高分={float(np.max(res)):.4f} 峰值(按x排序, >= {PEAK_FLOOR}):")
        for x, y, s in peaks:
            mark = "命中" if s >= threshold else "落选"
            log.info(f"[FLOORDIAG#{seq}]     x={x:<6} y={y:<6} score={s:.4f}  [{mark}]")

        above = [p for p in peaks if p[2] >= threshold]
        kept = list(production_result) if production_result else []
        log.info(
            f"[FLOORDIAG#{seq}] 过阈值峰值={len(above)} 生产去重后={len(kept)} "
            f"(两者相等则 min_dist 无影响) 生产坐标={kept}"
        )
    except Exception as e:  # 诊断绝不能影响正常流程
        log.warning(f"[FLOORDIAG] 采集异常，已忽略: {type(e).__name__}: {e}")
