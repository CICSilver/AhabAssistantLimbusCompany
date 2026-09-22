"""放大匹配的回归测试。

用真实截图裁片做 fixture，而不是合成图。原因：这个缺陷的成因是游戏在 1080 下
**自行渲染**标记（含引擎的抗锯齿与 hinting），而素材按 1440 制作。任何合成场景要么
把模板降采样后贴进画面（此时旧方式反而完美命中），要么贴无关内容（两边都无意义），
都无法复现真实的亚像素相位差。

fixture 取自真实第 5 层的楼层指示条带，含 4 个 CLEAR 标记：
旧方式（模板缩小到 0.75、阈值 0.8）只能检出 3 个，放大匹配可检出全部 4 个。
"""

from pathlib import Path

import cv2
import pytest

from module.automation import automation as automation_module
from module.automation.automation import Automation
from utils.image_utils import ImageUtils

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE = REPO_ROOT / "tests/fixtures/mirror_floor5_clear_band.png"
TEMPLATE = REPO_ROOT / "assets/images/default/share/mirror/road_in_mir/clear_floor.png"

WIN_SIZE = 1080
EXPECTED_MARKS = 4
# fixture 裁自原截图 x=700 起，四个槽位在原图 x=748/854/959/1065
BAND_X_OFFSET = 700
EXPECTED_X = [748 - BAND_X_OFFSET, 854 - BAND_X_OFFSET, 959 - BAND_X_OFFSET, 1065 - BAND_X_OFFSET]


def _load(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        pytest.skip(f"缺少测试数据 {path}")
    return image


def _make_automation(monkeypatch, screenshot, template):
    instance = object.__new__(Automation)
    instance.screenshot = screenshot
    instance._frame_dirty = False
    instance._screenshot_array = screenshot
    instance._screenshot_array_source = screenshot
    instance._frame_match_cache = {}
    instance.img_cache = {}
    monkeypatch.setattr(instance, "_ensure_frame_cache_current", lambda: None, raising=False)
    monkeypatch.setattr(automation_module.cfg, "set_win_size", WIN_SIZE, raising=False)
    # 生产的 load_image(resize=False) 返回原生尺寸灰度模板，这里照此提供
    monkeypatch.setattr(ImageUtils, "load_image", staticmethod(lambda target, resize=True: template))
    return instance


def test_upscaled_match_finds_every_clear_mark(monkeypatch):
    band, template = _load(FIXTURE), _load(TEMPLATE)
    instance = _make_automation(monkeypatch, band, template)

    hits = instance.find_multiple_targets_upscaled("mirror/road_in_mir/clear_floor.png", threshold=0.70, min_dist=60)

    assert len(hits) == EXPECTED_MARKS, f"应检出 {EXPECTED_MARKS} 个 CLEAR 标记，实得 {len(hits)}: {hits}"


def test_downscaled_template_still_misses_one(monkeypatch):
    """守住前提：旧方式确实漏检，否则这个修复就失去意义了。"""
    band, template = _load(FIXTURE), _load(TEMPLATE)
    small = cv2.resize(template, None, fx=WIN_SIZE / 1440, fy=WIN_SIZE / 1440, interpolation=cv2.INTER_AREA)

    hits = ImageUtils.match_template_with_multiple_targets(band, small, 0.8, min_dist=60)

    assert len(hits) < EXPECTED_MARKS, f"旧方式本应漏检，却检出了 {len(hits)} 个"


def test_hits_are_mapped_back_to_screenshot_space(monkeypatch):
    band, template = _load(FIXTURE), _load(TEMPLATE)
    instance = _make_automation(monkeypatch, band, template)

    hits = sorted(
        instance.find_multiple_targets_upscaled("mirror/road_in_mir/clear_floor.png", threshold=0.70, min_dist=60)
    )

    assert len(hits) == EXPECTED_MARKS
    height, width = band.shape
    for (x, y), want in zip(hits, EXPECTED_X):
        assert 0 <= x < width and 0 <= y < height, f"坐标 {(x, y)} 超出原截图范围，可能未从放大空间换算回来"
        assert abs(x - want) <= 3, f"槽位坐标偏差过大：得到 {x}，期望约 {want}"


def test_missing_template_returns_empty(monkeypatch):
    band, template = _load(FIXTURE), _load(TEMPLATE)
    instance = _make_automation(monkeypatch, band, template)
    monkeypatch.setattr(ImageUtils, "load_image", staticmethod(lambda target, resize=True: None))

    assert instance.find_multiple_targets_upscaled("missing.png") == []


def test_scaled_template_is_cached_across_calls(monkeypatch):
    band, template = _load(FIXTURE), _load(TEMPLATE)
    instance = _make_automation(monkeypatch, band, template)
    target = "mirror/road_in_mir/clear_floor.png"

    instance.find_multiple_targets_upscaled(target, threshold=0.70, min_dist=60)
    cached = [k for k in instance.img_cache if k[0] == "upscaled_template"]

    assert len(cached) == 1, f"放大后的模板应进缓存，现有键：{list(instance.img_cache)}"
    # 缓存的模板尺寸应为原生 × (set_win_size * upscale / 1440)
    expected_w = int(round(template.shape[1] * WIN_SIZE * 2 / 1440))
    assert abs(instance.img_cache[cached[0]].shape[1] - expected_w) <= 1
