"""
测试 lap_video_player：分圈查找、多圈速度图、多圈播放器（OpenCV 用 mock，无需窗口）。
运行: python test/test_lap_video_player.py
或:   pytest test/test_lap_video_player.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import lap_video_player as lvp  # noqa: E402
from lap_utils import Lap  # noqa: E402


def _sample_laps() -> list[Lap]:
  """两圈互不重叠的虚构遥测，仅含 time / speed。"""
  df0 = pd.DataFrame(
      {
          "time": [100.0, 100.5, 101.0],
          "speed": [20.0, 22.0, 24.0],
      }
  )
  df1 = pd.DataFrame(
      {
          "time": [200.0, 200.5, 201.0, 201.5],
          "speed": [30.0, 31.0, 32.0, 33.0],
      }
  )
  return [Lap(df0), Lap(df1)]


def _make_video_capture_mock(
    frame_count: int = 900, fps: float = 30.0
) -> MagicMock:
  cap = MagicMock()
  cap.isOpened.return_value = True

  def _get(prop: int) -> float:
    if prop == lvp.cv2.CAP_PROP_FRAME_COUNT:
      return float(frame_count)
    if prop == lvp.cv2.CAP_PROP_FPS:
      return fps
    return 0.0

  cap.get.side_effect = _get
  cap.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
  return cap


def _patch_cv2_gui(cap: MagicMock):
  """避免 namedWindow / imshow 等弹出真实窗口。"""
  return patch.multiple(
      "lap_video_player.cv2",
      VideoCapture=MagicMock(return_value=cap),
      namedWindow=MagicMock(),
      createTrackbar=MagicMock(),
      setTrackbarPos=MagicMock(),
      imshow=MagicMock(),
      waitKey=MagicMock(return_value=-1),
      destroyWindow=MagicMock(),
  )


class TestFindLapContaining(unittest.TestCase):
  def test_inside_first_lap(self):
    laps = _sample_laps()
    self.assertEqual(lvp._find_lap_containing(laps, 100.25), 0)

  def test_inside_second_lap(self):
    laps = _sample_laps()
    self.assertEqual(lvp._find_lap_containing(laps, 201.0), 1)

  def test_boundary_inclusive(self):
    laps = _sample_laps()
    self.assertEqual(lvp._find_lap_containing(laps, laps[0].t_start), 0)
    self.assertEqual(lvp._find_lap_containing(laps, laps[0].t_end), 0)

  def test_gap_returns_none(self):
    laps = _sample_laps()
    self.assertIsNone(lvp._find_lap_containing(laps, 150.0))


class TestSetupMultiLapSpeedPlot(unittest.TestCase):
  def test_lines_and_vline(self):
    laps = _sample_laps()
    fig, ax = plt.subplots()
    try:
      lines, vline = lvp.setup_multi_lap_speed_plot(ax, laps)
      self.assertEqual(len(lines), 2)
      self.assertTrue(vline.get_visible())
      xs = lines[0].get_xdata()
      np.testing.assert_allclose(xs, [0.0, 0.5, 1.0])
    finally:
      plt.close(fig)


class TestAttachMultiLapVideoPlayer(unittest.TestCase):
  def test_empty_laps_raises(self):
    fig, ax = plt.subplots()
    try:
      with self.assertRaises(ValueError) as ctx:
        lvp.attach_multi_lap_video_player(fig, "dummy.mp4", [], ax)
      self.assertIn("non-empty", str(ctx.exception).lower())
    finally:
      plt.close(fig)

  def test_seek_first_lap_start_with_offset(self):
    laps = _sample_laps()
    cap = _make_video_capture_mock(frame_count=900, fps=30.0)
    offset = laps[0].t_start

    with _patch_cv2_gui(cap):
      player = lvp.CV2MultiLapVideoPlayer(
          "dummy.mp4",
          lambda ts: None,
          lambda: None,
          laps,
          video_time_offset=offset,
          auto_advance_laps=False,
      )
      try:
        self.assertEqual(player._current_frame, 0)
        cap.set.assert_called()
        set_calls = [
            c for c in cap.set.call_args_list if c[0][0] == lvp.cv2.CAP_PROP_POS_FRAMES
        ]
        self.assertTrue(set_calls)
        self.assertEqual(set_calls[-1][0][1], 0)
      finally:
        player.stop()

  def test_vline_moves_with_callback_time(self):
    laps = _sample_laps()
    cap = _make_video_capture_mock()
    offset = laps[0].t_start
    fig, ax = plt.subplots()
    got: dict = {}

    def extra_on_frame(video_ts, telemetry_time=None, lap_index=None, **_k):
      got["telemetry_time"] = telemetry_time
      got["lap_index"] = lap_index

    try:
      with _patch_cv2_gui(cap):
        player, timer, _lines, vline = lvp.attach_multi_lap_video_player(
            fig,
            "dummy.mp4",
            laps,
            ax,
            video_time_offset=offset,
            auto_advance_laps=False,
            extra_on_frame=extra_on_frame,
        )
        try:
          mid_video = (laps[0].t_start - offset) + 0.25
          player._on_frame_callback(mid_video)
          self.assertEqual(got["lap_index"], 0)
          self.assertAlmostEqual(got["telemetry_time"], laps[0].t_start + 0.25)
          self.assertTrue(vline.get_visible())
          xdata = vline.get_xdata()
          self.assertAlmostEqual(float(xdata[0]), 0.25)
        finally:
          player.stop()
          timer.stop()
    finally:
      plt.close(fig)


if __name__ == "__main__":
  unittest.main()
