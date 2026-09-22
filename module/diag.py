"""临时全域诊断采集模块。确认问题后整个文件与全部调用点一并删除。

设计目标是「一次采集足够回答尚未想到的问题」，因为每轮完整镜牢要跑半小时，
材料不全就得重新打包再跑一轮，代价远高于多存些数据。

采集内容：
- 滚动帧缓冲：常驻内存保留最近 RING_SIZE 帧（只存引用，零拷贝），触发时整段落盘，
  拿到的是事发前后的连续序列而非孤立单帧——动画时序类问题靠单帧看不出来。
- 日志钩子：任何 WARNING/ERROR 自动触发一次落盘，兜住尚未预料到的故障。
- OCR 调用流水：每次 ocr.run 的输入区域、输入图像与识别结果全部记录。
- 显式采集点：楼层识别、饰品升级确认等已知可疑路径，附带匹配分数等上下文。

刻意不在进程内跑整帧 OCR：整帧 OCR 约 1.8 秒，而饰品升级这类缺陷本身就是动画
时序敏感的，插入重操作会改变被测对象。帧存下来之后离线跑 OCR 更全也更安全。

全部输出静默写入 diagnostic/，不经过项目日志器，避免刷爆滚动日志。
"""

import json
import logging
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

ROOT = os.path.join(".", "diagnostic")
FRAME_DIR = os.path.join(ROOT, "events")
OCR_DIR = os.path.join(ROOT, "ocr")
EVENT_LOG = os.path.join(ROOT, "events.jsonl")
OCR_LOG = os.path.join(OCR_DIR, "ocr_calls.jsonl")
DIAG_LOG = os.path.join(ROOT, "diag.log")

RING_SIZE = 40  # 约 6 秒历史（0.15s 轮询）；1080p 灰度每帧约 2MB
MAX_EVENTS = 120  # 防止长时间挂机撑爆磁盘
MAX_OCR_DUMPS = 400

_ring = deque(maxlen=RING_SIZE)
_lock = threading.Lock()
_event_seq = 0
_ocr_seq = 0
_installed = False


def _log(text):
    try:
        os.makedirs(ROOT, exist_ok=True)
        with open(DIAG_LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
    except Exception:
        pass


def _append_jsonl(path, record):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def record_frame(frame):
    """由 take_screenshot 在提交干净帧后调用。只存引用，不拷贝。"""
    try:
        if frame is None:
            return
        with _lock:
            _ring.append((time.monotonic(), frame))
    except Exception:
        pass


def snap(tag, note="", extra=None, dump_frames=True):
    """落盘一次事件现场：整段帧缓冲 + 元数据。"""
    global _event_seq
    try:
        if _event_seq >= MAX_EVENTS:
            return None
        _event_seq += 1
        seq = _event_seq
        stamp = time.strftime("%H%M%S")
        name = f"{seq:03d}_{tag}_{stamp}"
        written = []

        if dump_frames:
            with _lock:
                frames = list(_ring)
            target = os.path.join(FRAME_DIR, name)
            os.makedirs(target, exist_ok=True)
            now = time.monotonic()
            seen = set()
            for index, (ts, frame) in enumerate(frames):
                if id(frame) in seen:  # 同一帧被多次记录时只写一次
                    continue
                seen.add(id(frame))
                array = np.asarray(frame)
                if array.size == 0:
                    continue
                # 文件名带相对事发时刻的毫秒偏移，负值代表事发之前
                offset = int((ts - now) * 1000)
                path = os.path.join(target, f"{index:03d}_{offset:+06d}ms.png")
                cv2.imwrite(path, array)
                written.append(os.path.basename(path))

        record = {
            "seq": seq,
            "tag": tag,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": note,
            "frames": written,
            "frame_dir": name if written else None,
        }
        if extra:
            record["extra"] = extra
        _append_jsonl(EVENT_LOG, record)
        _log(f"[{name}] {note} (落盘 {len(written)} 帧)")
        return name
    except Exception as e:
        _log(f"snap 异常已忽略: {type(e).__name__}: {e}")
        return None


def note_ocr(image, result, crop=None, elapsed_ms=None, caller=""):
    """记录一次 ocr.run 调用的输入与输出。"""
    global _ocr_seq
    try:
        if _ocr_seq >= MAX_OCR_DUMPS:
            return
        _ocr_seq += 1
        seq = _ocr_seq
        array = np.asarray(image)
        img_name = None
        if array.size:
            os.makedirs(os.path.join(OCR_DIR, "img"), exist_ok=True)
            img_name = f"{seq:04d}.png"
            cv2.imwrite(os.path.join(OCR_DIR, "img", img_name), array)
        texts = list(getattr(result, "txts", None) or [])
        boxes = getattr(result, "boxes", None)
        _append_jsonl(
            OCR_LOG,
            {
                "seq": seq,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "caller": caller,
                "crop": list(crop) if crop else None,
                "shape": list(array.shape),
                "elapsed_ms": elapsed_ms,
                "image": img_name,
                "texts": texts,
                "box_count": 0 if boxes is None else len(boxes),
            },
        )
    except Exception:
        pass


class _LogTrigger(logging.Handler):
    """任何 WARNING/ERROR 都留下现场，兜住尚未预料到的故障。"""

    def emit(self, record):
        try:
            if record.levelno < logging.WARNING:
                return
            message = record.getMessage()
            if message.startswith("[") and "诊断" in message:  # 避免自触发
                return
            snap(
                "log",
                note=f"{record.levelname} {record.module}:{record.lineno} {message}",
                extra={"level": record.levelname, "pathname": record.pathname, "lineno": record.lineno},
            )
        except Exception:
            pass


def install():
    """在程序启动时装载钩子；重复调用无副作用。"""
    global _installed
    if _installed:
        return
    _installed = True
    try:
        os.makedirs(ROOT, exist_ok=True)
        logging.getLogger("AALC").addHandler(_LogTrigger())
        _wrap_ocr()
        _log(f"诊断采集已启动 ring={RING_SIZE} 上限 events={MAX_EVENTS} ocr={MAX_OCR_DUMPS}")
    except Exception as e:
        _log(f"install 异常: {type(e).__name__}: {e}")


def _wrap_ocr():
    """包裹 ocr.run，记录每一次真实调用。"""
    try:
        from module.ocr import ocr as ocr_instance

        original = ocr_instance.run
        if getattr(original, "_diag_wrapped", False):
            return

        def wrapped(image, *args, **kwargs):
            start = time.perf_counter()
            result = original(image, *args, **kwargs)
            try:
                note_ocr(image, result, elapsed_ms=round((time.perf_counter() - start) * 1000, 1))
            except Exception:
                pass
            return result

        wrapped._diag_wrapped = True
        ocr_instance.run = wrapped
    except Exception as e:
        _log(f"包裹 ocr.run 失败: {type(e).__name__}: {e}")


# ---------------- 楼层识别专用：保留匹配分数分析 ----------------

PEAK_FLOOR = 0.45
PRINT_TOP = 12
UPSCALE = 2
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
    kept = [(x + template_w // 2, y + template_h // 2, s) for x, y, s in kept]
    return sorted(kept, key=lambda p: p[0])


def capture_floor(tag, target, threshold, min_dist, production_result):
    """楼层识别现场：复刻生产的放大匹配并记录全部峰值得分。"""
    try:
        from module.automation import auto
        from module.config import cfg
        from utils.image_utils import ImageUtils

        screenshot = auto.get_screenshot_array()
        if screenshot is None or getattr(screenshot, "size", 0) == 0:
            return
        template = ImageUtils.load_image(target, resize=False)
        if template is None:
            return
        template_scale = cfg.set_win_size * UPSCALE / 1440
        scaled_template = cv2.resize(
            template, None, fx=template_scale, fy=template_scale, interpolation=cv2.INTER_LINEAR
        )
        scaled_shot = cv2.resize(screenshot, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LINEAR)
        if tag not in _saved_templates:
            os.makedirs(ROOT, exist_ok=True)
            cv2.imwrite(os.path.join(ROOT, f"template_{tag}.png"), scaled_template)
            _saved_templates.add(tag)

        th, tw = scaled_template.shape[:2]
        res = cv2.matchTemplate(scaled_shot, scaled_template, cv2.TM_CCOEFF_NORMED)
        peaks = _peaks(res, tw, th, PEAK_FLOOR)
        kept = list(production_result) if production_result else []
        detail = [
            {"x": x // UPSCALE, "y": y // UPSCALE, "score": round(sc, 4), "hit": sc >= threshold} for x, y, sc in peaks
        ]
        snap(
            f"floor_{tag}",
            note=f"楼层识别 {tag}: 过阈值 {sum(1 for d in detail if d['hit'])} 生产去重后 {len(kept)}",
            extra={
                "template": f"{tw}x{th}",
                "upscale": UPSCALE,
                "threshold": threshold,
                "min_dist": round(min_dist, 1),
                "best": round(float(np.max(res)), 4),
                "peaks": detail,
                "production": [list(p) for p in kept],
            },
        )
    except Exception as e:
        _log(f"capture_floor 异常已忽略: {type(e).__name__}: {e}")
