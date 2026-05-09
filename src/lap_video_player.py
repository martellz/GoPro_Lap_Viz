from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "Need OpenCV (cv2). See https://pypi.org/project/opencv-python/ "
        "or: pip install opencv-python"
    ) from e

if TYPE_CHECKING:
    from lap_utils import Lap

from lap_cache import (
    cache_skip_enabled,
    single_lap_segment_dir,
    try_read_single_lap_segment,
    write_single_lap_segment_manifest,
)

# Inspiration: https://github.com/maximus009/VideoPlayer/blob/master/new_test_3.py


def _cv2_safe_window_title(title: str, max_len: int = 64) -> str:
  """OpenCV Qt highgui 对非 ASCII/过长标题易产生 NULL window handler；仅保留安全字符。"""
  ascii_only = title.encode("ascii", "replace").decode("ascii")
  out = []
  for ch in ascii_only:
    if ch.isalnum() or ch in "._-+":
      out.append(ch)
    elif ch.isspace():
      out.append("_")
  s = "".join(out).strip("_") or "win"
  return s[:max_len]


def _cv2_named_window_ready(name: str) -> None:
  cv2.namedWindow(name, cv2.WINDOW_NORMAL)
  cv2.waitKey(1)


def _lapcompare_ffmpeg_cut(
    src: str,
    t0_sec: float,
    duration_sec: float,
    dest: str,
    *,
    output_width: Optional[int] = None,
) -> None:
  """将 ``src`` 中 ``[t0_sec, t0_sec+duration]`` 截成独立短视频。

  * ``output_width`` > 0：用 libx264 + ``scale=W:-2`` 一次性缩放到目标宽度（播放时不再 resize）。
  * 否则：优先 stream copy，失败则 ultrafast 重编码（全分辨率）。
  """
  duration_sec = max(float(duration_sec), 0.05)
  t0_sec = max(float(t0_sec), 0.0)

  if output_width is not None and int(output_width) > 0:
      w = int(output_width)
      vf = f"scale={w}:-2"
      cmd_scaled = [
          "ffmpeg",
          "-hide_banner",
          "-loglevel",
          "error",
          "-y",
          "-ss",
          f"{t0_sec:.6f}",
          "-i",
          src,
          "-t",
          f"{duration_sec:.6f}",
          "-vf",
          vf,
          "-c:v",
          "libx264",
          "-preset",
          "ultrafast",
          "-crf",
          "28",
          "-an",
          dest,
      ]
      r = subprocess.run(cmd_scaled, capture_output=True, text=True)
      if r.returncode != 0 or not os.path.isfile(dest) or os.path.getsize(dest) < 800:
          raise RuntimeError(
              (r.stderr or r.stdout or "ffmpeg scale encode failed").strip()
          )
      return

  cmd_copy = [
      "ffmpeg",
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-ss",
      f"{t0_sec:.6f}",
      "-i",
      src,
      "-t",
      f"{duration_sec:.6f}",
      "-c",
      "copy",
      "-avoid_negative_ts",
      "make_zero",
      dest,
  ]
  r = subprocess.run(cmd_copy, capture_output=True, text=True)
  if r.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 800:
      return
  cmd_enc = [
      "ffmpeg",
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-ss",
      f"{t0_sec:.6f}",
      "-i",
      src,
      "-t",
      f"{duration_sec:.6f}",
      "-c:v",
      "libx264",
      "-preset",
      "ultrafast",
      "-crf",
      "28",
      "-an",
      dest,
  ]
  r2 = subprocess.run(cmd_enc, capture_output=True, text=True)
  if r2.returncode != 0 or not os.path.isfile(dest) or os.path.getsize(dest) < 800:
      raise RuntimeError(
          (r2.stderr or r2.stdout or r.stderr or r.stdout or "ffmpeg failed").strip()
      )


class FrameChange(Enum):
    NoChange = 0
    Next = 1
    Seek = 2


def _find_lap_containing(laps: List[Any], telemetry_t: float) -> Optional[int]:
  """Return index i with laps[i].t_start <= t <= laps[i].t_end, or None."""
  for i, lap in enumerate(laps):
    if lap.t_start <= telemetry_t <= lap.t_end:
      return i
  return None


def _enu_xy_at_tau_in_lap(lap: Any, tau: float) -> Tuple[Optional[float], Optional[float]]:
  """圈内时间 ``tau`` (s) 时刻在 ENU 平面上的位置；对 telemetry 做线性插值。"""
  seg = lap.segment
  if seg is None or len(seg) < 1:
    return None, None
  t_rel = (seg["time"] - lap.t_start).to_numpy(dtype=float)
  if t_rel.size == 0:
    return None, None
  x = seg["enu_x"].to_numpy(dtype=float)
  y = seg["enu_y"].to_numpy(dtype=float)
  if t_rel.size == 1:
    return float(x[0]), float(y[0])
  tau_eff = float(np.clip(tau, 0.0, float(t_rel[-1])))
  if not np.all(np.diff(t_rel) > 0):
    j = int(np.searchsorted(t_rel, tau_eff, side="right") - 1)
    j = int(np.clip(j, 0, t_rel.size - 1))
    return float(x[j]), float(y[j])
  xi = float(np.interp(tau_eff, t_rel, x))
  yi = float(np.interp(tau_eff, t_rel, y))
  return xi, yi


def setup_multi_lap_speed_plot(
    ax,
    laps: List[Any],
    *,
    lap_labels: Optional[List[str]] = None,
    title: str = "Speed — all laps",
) -> Tuple[List[Any], Any]:
  """Plot speed vs. time-in-lap for each Lap on ``ax``; return (line artists, vertical cursor).

  ``lap_labels`` defaults to ``Lap 1``, ``Lap 2``, … when omitted.
  """
  import matplotlib.pyplot as plt

  n = max(len(laps), 1)
  colors = plt.cm.tab10(np.linspace(0, 1, n, endpoint=False))  # type: ignore[attr-defined]
  lines: List[Any] = []
  for i, lap in enumerate(laps):
    t_rel = (lap.segment["time"] - lap.t_start).to_numpy(dtype=float)
    spd = lap.segment["speed"].to_numpy(dtype=float)
    label = lap_labels[i] if lap_labels is not None else f"Lap {i + 1}"
    (ln,) = ax.plot(
      t_rel,
      spd,
      color=colors[i % 10],
      label=label,
      linewidth=1.2,
    )
    lines.append(ln)
  ax.set_xlabel("Time in lap (s)")
  ax.set_ylabel("Speed")
  ax.set_title(title)
  ax.legend(loc="upper right", fontsize=8)
  ax.grid(True, alpha=0.3)
  vline = ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
  return lines, vline


class CV2VideoPlayer:
    KEY_CODE_STOP = 27
    KEY_CODE_TOGGLE_PLAY = 32
    KEY_CODE_PREV_FRAME = ord("a")
    KEY_CODE_NEXT_FRAME = ord("d")

    CMD_NEXT_FRAME = "next"
    CMD_NOOP = "noop"

    _frame_count: int
    _fps: float
    _on_frame_callback: Callable
    _on_stop_callback: Callable
    _previous_frame_display_timestamp: float
    _window_name: str = "VideoPlayer"

    def __init__(self, filename: str, on_frame: Callable, on_stop: Callable):
        self._cap = cv2.VideoCapture(filename)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not read {filename}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

        self._playback_rate = 1.0

        self._current_frame = 0
        self._on_frame_callback = on_frame
        self._on_stop_callback = on_stop
        self._previous_frame_display_timestamp = 0.0

        self._status = "play"

        self._setup_ui()

    def _setup_ui(self):
        def on_change_frame(x):
            if x == self._current_frame:
                return

            self._current_frame = x
            if self._status == "paused":
                self._status = "seek_frame"

        def on_change_playback_rate(x):
            if x == 0:
                x = 1
            self._playback_rate = x / 100.0

        max_idx = max(self._frame_count - 1, 0)
        _cv2_named_window_ready(self._window_name)
        cv2.createTrackbar("Frame", self._window_name, 0, max_idx, on_change_frame)
        cv2.setTrackbarPos("Frame", self._window_name, 0)

        cv2.createTrackbar("Playback Speed", self._window_name, 1, 400, on_change_playback_rate)
        cv2.setTrackbarPos("Playback Speed", self._window_name, int(self._playback_rate * 100))

    def _handle_keyboard_input(self):
        key = cv2.waitKey(1)
        if key == self.KEY_CODE_TOGGLE_PLAY:
            if self._status == "paused":
                self._status = "play"
            else:
                self._status = "paused"
        elif key == self.KEY_CODE_PREV_FRAME:
            self._status = "prev_frame"
        elif key == self.KEY_CODE_NEXT_FRAME:
            self._status = "next_frame"
        elif key == self.KEY_CODE_STOP:
            self.stop()

    def _calculate_current_frame(self) -> FrameChange:
        if self._status == "play":
            if self._current_frame >= self._frame_count:
                self._status = "paused"
                return FrameChange.NoChange

            now = time.time()
            if self._previous_frame_display_timestamp == 0:
                self._previous_frame_display_timestamp = now
                return FrameChange.NoChange

            frame_display_time = 1 / self._fps / self._playback_rate
            time_delta = now - self._previous_frame_display_timestamp

            frame_delta = int(time_delta / frame_display_time)

            if frame_delta > 0:
                self._current_frame += frame_delta
                self._previous_frame_display_timestamp = now
                if frame_delta > 1:
                    return FrameChange.Seek
                return FrameChange.Next
        elif self._status == "next_frame":
            if self._current_frame < self._frame_count - 1:
                self._current_frame += 1
                self._status = "paused"
                return FrameChange.Next
        elif self._status == "prev_frame":
            if self._current_frame > 0:
                self._current_frame -= 1
                self._status = "paused"
                return FrameChange.Seek
        elif self._status == "seek_frame":
            self._status = "paused"
            return FrameChange.Seek
        elif self._status == "paused":
            self._previous_frame_display_timestamp = 0

        return FrameChange.NoChange

    def _show_current_frame(self):
        ret, im = self._cap.read()
        if im is None:
            return
        r = 720.0 / im.shape[1]
        dim = (720, int(im.shape[0] * r))
        im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow(self._window_name, im)

    def _post_timer_hook(self, video_timestamp_sec: float) -> None:
        """Override in subclass for telemetry sync / auto lap advance."""

    def on_timer(self):
        if self._status == "stopped":
            return

        self._handle_keyboard_input()
        frame_change = self._calculate_current_frame()

        if frame_change == FrameChange.NoChange:
            return
        if frame_change == FrameChange.Next:
            pass
        elif frame_change == FrameChange.Seek:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)

        self._show_current_frame()
        cv2.setTrackbarPos("Frame", self._window_name, self._current_frame)
        ts = self._current_frame / self._fps
        self._on_frame_callback(ts)
        self._post_timer_hook(ts)

    def stop(self):
        if self._status == "stopped":
            return
        self._cap.release()
        del self._cap
        cv2.destroyWindow(self._window_name)
        self._status = "stopped"
        self._on_stop_callback()


class CV2MultiLapVideoPlayer(CV2VideoPlayer):
    """Video clock + optional offset maps to CSV ``time``; optional auto-jump to next lap."""

    def __init__(
        self,
        filename: str,
        on_frame: Callable[[float], None],
        on_stop: Callable[[], None],
        laps: List[Any],
        video_time_offset: float = 0.0,
        auto_advance_laps: bool = True,
    ):
        self._laps = list(laps)
        self._video_time_offset = float(video_time_offset)
        self._auto_advance_laps = bool(auto_advance_laps)
        super().__init__(filename, on_frame, on_stop)
        if self._laps:
            self._seek_to_telemetry_time(self._laps[0].t_start)

    def _seek_to_telemetry_time(self, telemetry_t: float) -> None:
        vid_t = telemetry_t - self._video_time_offset
        frame = int(round(vid_t * self._fps))
        frame = max(0, min(frame, max(self._frame_count - 1, 0)))
        self._current_frame = frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def _post_timer_hook(self, video_timestamp_sec: float) -> None:
        if not self._laps or not self._auto_advance_laps or self._status != "play":
            return
        telemetry_t = video_timestamp_sec + self._video_time_offset
        idx = _find_lap_containing(self._laps, telemetry_t)
        if idx is None:
            return
        if telemetry_t <= self._laps[idx].t_end + 1e-6:
            return
        next_idx = idx + 1
        if next_idx >= len(self._laps):
            self._status = "paused"
            return
        self._seek_to_telemetry_time(self._laps[next_idx].t_start)
        self._show_current_frame()
        cv2.setTrackbarPos("Frame", self._window_name, self._current_frame)
        ts = self._current_frame / self._fps
        self._on_frame_callback(ts)


class CV2LapCompareVideoPlayer:
    """多路对比：每圈独立 VideoCapture；默认可用 ffmpeg 先切片，再顺序解码以流畅播放。"""

    KEY_CODE_STOP = 27
    KEY_CODE_TOGGLE_PLAY = 32
    KEY_CODE_PREV_FRAME = ord("a")
    KEY_CODE_NEXT_FRAME = ord("d")
    _MAX_SEQ_READ_FALLBACK = 28

    def __init__(
        self,
        filename: str,
        compare_laps: List[Any],
        window_titles: List[str],
        video_time_offset: float,
        on_frame: Callable[[float], None],
        on_stop: Callable[[], None],
        *,
        video_panel_width: int = 480,
        use_ffmpeg_segments: bool = True,
        segment_cache_read: Optional[bool] = None,
        segment_cache_write: Optional[bool] = None,
    ):
        if len(compare_laps) != len(window_titles):
            raise ValueError("compare_laps and window_titles must have the same length")
        if not compare_laps:
            raise ValueError("compare_laps must be non-empty")

        self._laps = list(compare_laps)
        raw_titles = list(window_titles)
        self._titles = []
        for i, raw in enumerate(raw_titles):
            base = _cv2_safe_window_title(raw, 48)
            self._titles.append(f"v{i}_{base}")

        self._offset = float(video_time_offset)
        self._on_frame = on_frame
        self._on_stop = on_stop
        self._video_width = int(video_panel_width)
        self._resize_inter = cv2.INTER_LINEAR

        self._segment_tmpdir: Optional[str] = None
        self._using_segments = False
        self._segment_frames_prescaled = False
        paths_to_open: List[str] = []

        do_cache_read = (
            not cache_skip_enabled()
            if segment_cache_read is None
            else segment_cache_read
        )
        do_cache_write = (
            not cache_skip_enabled()
            if segment_cache_write is None
            else segment_cache_write
        )

        ffmpeg_ok = shutil.which("ffmpeg") is not None
        if use_ffmpeg_segments and ffmpeg_ok:
            out_w = max(int(self._video_width), 1)
            tmpdir: Optional[str] = None
            n_from_cache = 0
            try:
                for i, lap in enumerate(self._laps):
                    hit: Optional[str] = None
                    if do_cache_read:
                        try:
                            hit = try_read_single_lap_segment(
                                filename, self._offset, out_w, lap
                            )
                        except Exception:
                            hit = None
                    if hit:
                        paths_to_open.append(hit)
                        n_from_cache += 1
                        continue

                    t0 = max(0.0, lap.t_start - self._offset)
                    dur = max(lap.t_end - lap.t_start, 0.05)
                    wrote_cache = False
                    if do_cache_write:
                        clip_dir_t: Optional[Path] = None
                        try:
                            clip_dir_t = single_lap_segment_dir(
                                filename,
                                self._offset,
                                out_w,
                                lap.t_start,
                                lap.t_end,
                            )
                            clip_dir_t.mkdir(parents=True, exist_ok=True)
                            outp = clip_dir_t / "clip.mp4"
                            _lapcompare_ffmpeg_cut(
                                filename,
                                t0,
                                dur,
                                str(outp),
                                output_width=out_w,
                            )
                            write_single_lap_segment_manifest(
                                clip_dir_t,
                                filename,
                                self._offset,
                                out_w,
                                lap,
                            )
                            paths_to_open.append(str(outp.resolve()))
                            wrote_cache = True
                        except OSError:
                            wrote_cache = False
                        except Exception:
                            wrote_cache = False
                            if clip_dir_t is not None:
                                for orphan in ("clip.mp4", "manifest.json"):
                                    p = clip_dir_t / orphan
                                    if p.is_file():
                                        try:
                                            p.unlink()
                                        except OSError:
                                            pass
                    if not wrote_cache:
                        if tmpdir is None:
                            tmpdir = tempfile.mkdtemp(prefix="lapcompare_seg_")
                            self._segment_tmpdir = tmpdir
                        outp = os.path.join(tmpdir, f"lap_{i}.mp4")
                        _lapcompare_ffmpeg_cut(
                            filename,
                            t0,
                            dur,
                            outp,
                            output_width=out_w,
                        )
                        paths_to_open.append(outp)

                self._using_segments = True
                self._segment_frames_prescaled = True
                if tmpdir is None:
                    self._segment_tmpdir = None
                if n_from_cache == len(self._laps):
                    print(
                        "[LapCompare] 已命中切片缓存（按单圈），"
                        f"{out_w}px 宽，跳过 ffmpeg。"
                    )
                elif n_from_cache > 0:
                    print(
                        "[LapCompare] 部分圈使用切片缓存，"
                        f"{out_w}px 宽；其余已 ffmpeg 切片。"
                    )
                else:
                    print(
                        "[LapCompare] ffmpeg 已切片并缩放到 "
                        f"{out_w}px 宽，播放时不再逐帧 resize。"
                    )
            except Exception as exc:
                print(f"[LapCompare] ffmpeg 切片失败，改用整文件（易卡）：{exc}")
                if tmpdir:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                self._segment_tmpdir = None
                paths_to_open = []
                self._using_segments = False
                self._segment_frames_prescaled = False

        if not paths_to_open:
            paths_to_open = [filename] * len(self._laps)

        self._caps: List[Any] = []
        try:
            for path in paths_to_open:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not read {path}")
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self._caps.append(cap)
        except Exception:
            for c in self._caps:
                c.release()
            self._caps.clear()
            if self._segment_tmpdir:
                shutil.rmtree(self._segment_tmpdir, ignore_errors=True)
                self._segment_tmpdir = None
            raise

        n = len(self._laps)
        self._cap_last_frame: List[int] = [-1] * n
        self._cap_last_bgr: List[Optional[np.ndarray]] = [None] * n

        self._fps = float(self._caps[0].get(cv2.CAP_PROP_FPS)) or 30.0
        self._global_nframes = int(self._caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

        self._lap_frame_ranges: List[Tuple[int, int]] = []
        if self._using_segments:
            for cap in self._caps:
                nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                hi = max(nf - 1, 0)
                self._lap_frame_ranges.append((0, hi))
        else:
            for lap in self._laps:
                lo = int(round((lap.t_start - self._offset) * self._fps))
                hi = int(round((lap.t_end - self._offset) * self._fps))
                gmax = max(self._global_nframes - 1, 0)
                lo = max(0, min(lo, gmax))
                hi = max(0, min(hi, gmax))
                if hi < lo:
                    lo, hi = hi, lo
                self._lap_frame_ranges.append((lo, hi))

        self._durations = [max(lap.t_end - lap.t_start, 0.0) for lap in self._laps]
        self._tau_max = max(self._durations) if self._durations else 0.0
        if self._tau_max <= 1e-9:
            self._tau_max = 1.0

        self._max_play_frames = max(int(round(self._tau_max * self._fps)), 1)
        self._play_frame_idx = 0
        self._frame_time_carry = 0.0

        self._tau = 0.0
        self._playback_rate = 1.0
        self._status = "play"
        self._prev_realtime = 0.0
        self._stopped = False
        self._control_win = "LapCompare_ctrl"
        self._tau_centi_max = max(min(int(round(self._tau_max * 100)), 2_000_000), 1)

        self._max_seq_read = 120 if self._using_segments else self._MAX_SEQ_READ_FALLBACK

        self._matplotlib_tau_slider = None

        self._setup_ui()
        if not ffmpeg_ok and use_ffmpeg_segments:
            print(
                "[LapCompare] 未安装 ffmpeg，无法切片；安装后重试可明显流畅: apt install ffmpeg"
            )
        print(
            "[LapCompare] Space 暂停/继续 | a/d 上/下一帧 | Esc 退出\n"
            "           OpenCV: t_cs=圈内时间(0.01s) | spd=速度% | 图表刷新限速以减压 CPU"
        )
        self._sync_trackbar_from_tau()
        self._refresh_all_displays(trigger_callback=True)

    def set_tau(self, tau: float) -> None:
        """跳转圈内时间；会清空各路的解码缓存。"""
        self._play_frame_idx = int(
            round(float(np.clip(tau, 0.0, self._tau_max)) * self._fps)
        )
        self._play_frame_idx = max(0, min(self._play_frame_idx, self._max_play_frames))
        self._tau = self._play_frame_idx / self._fps
        self._status = "paused"
        self._prev_realtime = 0.0
        self._frame_time_carry = 0.0
        self._invalidate_stream_caches()
        self._sync_trackbar_from_tau()
        self._refresh_all_displays(trigger_callback=True)

    def _invalidate_stream_caches(self) -> None:
        for i in range(len(self._caps)):
            self._cap_last_frame[i] = -1
            self._cap_last_bgr[i] = None

    def _setup_ui(self) -> None:
        _cv2_named_window_ready(self._control_win)

        def on_tau_centi(pos: int) -> None:
            self.set_tau(min(pos / 100.0, self._tau_max))

        def on_rate_bar(x: int) -> None:
            if x == 0:
                x = 1
            self._playback_rate = x / 100.0

        cv2.createTrackbar(
            "t_cs", self._control_win, 0, self._tau_centi_max, on_tau_centi
        )
        cv2.setTrackbarPos("t_cs", self._control_win, 0)
        cv2.createTrackbar("spd", self._control_win, 1, 400, on_rate_bar)
        cv2.setTrackbarPos("spd", self._control_win, 100)

        try:
            cv2.resizeWindow(self._control_win, 520, 120)
        except cv2.error:
            pass

        for t in self._titles:
            _cv2_named_window_ready(t)

    def _sync_trackbar_from_tau(self) -> None:
        pos = int(round(self._tau * 100))
        pos = max(0, min(pos, self._tau_centi_max))
        cv2.setTrackbarPos("t_cs", self._control_win, pos)

    def _decode_and_show_stream(self, i: int, lo: int, hi: int, target_f: int) -> None:
        cap = self._caps[i]
        last = self._cap_last_frame[i]
        target_f = max(lo, min(int(target_f), hi))

        if last == target_f and self._cap_last_bgr[i] is not None:
            cv2.imshow(self._titles[i], self._cap_last_bgr[i])
            return

        im: Optional[np.ndarray] = None
        need_seek = (
            last < 0
            or target_f < last
            or (target_f - last) > self._max_seq_read
        )
        if need_seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ret, im = cap.read()
            if not ret or im is None:
                self._cap_last_frame[i] = -1
                self._cap_last_bgr[i] = None
                return
            self._cap_last_frame[i] = target_f
        else:
            for _ in range(target_f - last):
                ret, im = cap.read()
                if not ret or im is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                    ret, im = cap.read()
                    break
            if im is None:
                self._cap_last_frame[i] = -1
                self._cap_last_bgr[i] = None
                return
            self._cap_last_frame[i] = target_f

        if self._segment_frames_prescaled:
            out = im
        else:
            w = float(self._video_width)
            r = w / im.shape[1]
            dim = (int(w), int(im.shape[0] * r))
            out = cv2.resize(im, dim, interpolation=self._resize_inter)
        self._cap_last_bgr[i] = out
        cv2.imshow(self._titles[i], out)

    def _refresh_all_displays(self, trigger_callback: bool = False) -> None:
        display_f = int(round(self._tau * self._fps))
        for i, (lap, (lo, hi)) in enumerate(zip(self._laps, self._lap_frame_ranges)):
            if self._using_segments:
                target_f = display_f
            else:
                vid_t = (lap.t_start + self._tau) - self._offset
                target_f = int(round(vid_t * self._fps))
            self._decode_and_show_stream(i, lo, hi, target_f)
        if trigger_callback:
            self._on_frame(self._tau)

    def _handle_keyboard_input(self) -> None:
        key = cv2.waitKey(1)
        if key == self.KEY_CODE_TOGGLE_PLAY:
            self._status = "play" if self._status == "paused" else "paused"
            self._prev_realtime = 0.0
            if self._status == "play":
                self._frame_time_carry = 0.0
        elif key == self.KEY_CODE_PREV_FRAME:
            self.set_tau(max(0.0, self._tau - 1.0 / self._fps))
        elif key == self.KEY_CODE_NEXT_FRAME:
            self.set_tau(min(self._tau_max, self._tau + 1.0 / self._fps))
        elif key == self.KEY_CODE_STOP:
            self.stop()

    def _tick_playback(self) -> None:
        if self._status != "play":
            return
        now = time.time()
        if self._prev_realtime == 0.0:
            self._prev_realtime = now
            return
        dt = now - self._prev_realtime
        self._prev_realtime = now
        self._frame_time_carry += dt * self._fps * self._playback_rate
        n_step = int(self._frame_time_carry)
        if n_step <= 0:
            return
        self._frame_time_carry -= n_step
        self._play_frame_idx = min(
            self._play_frame_idx + n_step, self._max_play_frames
        )
        self._tau = self._play_frame_idx / self._fps
        if self._play_frame_idx >= self._max_play_frames:
            self._status = "paused"
        self._sync_trackbar_from_tau()

    def on_timer(self) -> None:
        if self._stopped:
            return
        self._handle_keyboard_input()
        prev_idx = self._play_frame_idx
        if self._status == "play":
            self._tick_playback()
        if self._status == "play" or self._play_frame_idx != prev_idx:
            self._refresh_all_displays(trigger_callback=True)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for cap in self._caps:
            cap.release()
        self._caps.clear()
        cv2.destroyWindow(self._control_win)
        for t in self._titles:
            cv2.destroyWindow(t)
        if self._segment_tmpdir:
            shutil.rmtree(self._segment_tmpdir, ignore_errors=True)
            self._segment_tmpdir = None
        self._on_stop()


def attach_video_player_to_figure(figure, filename: str, on_frame: Callable, **callback_args):
    try:
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError("matplotlib is required.")

    if not isinstance(figure, Figure):
        raise ValueError("`figure` should be a matplotlib Figure instance")

    timer = None

    def on_stop():
        timer.stop()

    video_player = CV2VideoPlayer(
        filename, lambda timestamp: on_frame(timestamp, **callback_args), on_stop
    )

    timer = figure.canvas.new_timer(
        interval=10, callbacks=[(video_player.on_timer, [], {})]
    )
    timer.start()


def attach_multi_lap_video_player(
    figure,
    filename: str,
    laps: List[Any],
    ax_speed,
    *,
    video_time_offset: float = 0.0,
    auto_advance_laps: bool = True,
    draw_speed_plot: bool = True,
    extra_on_frame: Optional[Callable[..., None]] = None,
) -> Tuple[CV2MultiLapVideoPlayer, Any, List[Any], Any]:
    """Attach OpenCV video + matplotlib timer; speed curves for all laps on ``ax_speed``.

    ``telemetry_time = video_timestamp_sec + video_time_offset`` must match CSV ``time``
    (same unit as ``Lap.t_start`` / ``t_end``). Adjust ``video_time_offset`` so that frame 0
    aligns with your telemetry origin.

    Returns ``(video_player, timer, speed_lines, vline)``.
    """
    if not laps:
        raise ValueError("laps must be non-empty")
    try:
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError("matplotlib is required.")

    if not isinstance(figure, Figure):
        raise ValueError("`figure` should be a matplotlib Figure instance")

    if draw_speed_plot:
        lines, vline = setup_multi_lap_speed_plot(ax_speed, laps)
    else:
        lines, vline = [], None

    timer = None

    def on_stop():
        timer.stop()

    def update_plot(video_ts: float):
        telemetry_t = video_ts + video_time_offset
        lap_idx = _find_lap_containing(laps, telemetry_t)
        if vline is not None:
            if lap_idx is None:
                vline.set_visible(False)
            else:
                rel_t = telemetry_t - laps[lap_idx].t_start
                vline.set_xdata([rel_t, rel_t])
                vline.set_visible(True)
            figure.canvas.draw_idle()
        if extra_on_frame is not None:
            extra_on_frame(
                video_ts,
                telemetry_time=telemetry_t,
                lap_index=lap_idx,
            )

    video_player = CV2MultiLapVideoPlayer(
        filename,
        update_plot,
        on_stop,
        laps,
        video_time_offset=video_time_offset,
        auto_advance_laps=auto_advance_laps,
    )

    timer = figure.canvas.new_timer(
        interval=10, callbacks=[(video_player.on_timer, [], {})]
    )
    timer.start()
    return video_player, timer, lines, vline


def attach_lap_compare_video_player(
    figure,
    filename: str,
    laps_subset: List[Any],
    ax_speed,
    *,
    video_time_offset: float = 0.0,
    window_titles: Optional[List[str]] = None,
    lap_labels: Optional[List[str]] = None,
    speed_plot_title: str = "Speed — lap comparison",
    draw_speed_plot: bool = True,
    video_panel_width: int = 480,
    extra_on_frame: Optional[Callable[..., None]] = None,
    add_mpl_tau_slider: bool = True,
    timer_interval_ms: Optional[int] = None,
    use_ffmpeg_segments: Optional[bool] = None,
    ax_track: Any = None,
    track_marker_colors: Optional[List[str]] = None,
    segment_cache_read: Optional[bool] = None,
    segment_cache_write: Optional[bool] = None,
) -> Tuple[CV2LapCompareVideoPlayer, Any, List[Any], Any]:
    """多窗口同步对比：默认 ffmpeg 按圈切短文件再播，顺序解码接近满帧率。

    环境变量 ``LAP_COMPARE_NO_FFMPEG=1``：强制不切片段（整文件 seek，易卡）。

    切片按「单圈时间段」持久化（换对比组合时复用已有切片）；会话缓存仅存矩形与时间偏移。
    默认开启；``LAP_VIZ_SKIP_CACHE=1`` 跳过读/写。

    ``ax_track``: 若提供，将在 ENU 轨迹上绘制每圈当前 ``tau`` 对应的车位点（与轨迹同色）。
    ``track_marker_colors``: 每圈标记颜色，默认 ``C0``..``C9`` 循环。
    """
    if not laps_subset:
        raise ValueError("laps_subset must be non-empty")
    try:
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError("matplotlib is required.")

    if not isinstance(figure, Figure):
        raise ValueError("`figure` should be a matplotlib Figure instance")

    if use_ffmpeg_segments is None:
        use_ffmpeg_segments = os.environ.get(
            "LAP_COMPARE_NO_FFMPEG", ""
        ).strip().lower() not in ("1", "true", "yes")

    n = len(laps_subset)
    if window_titles is None:
        if lap_labels is not None:
            window_titles = [f"Video - {lb}" for lb in lap_labels]
        else:
            window_titles = [f"Video - panel_{i + 1}" for i in range(n)]

    if draw_speed_plot:
        lines, vline = setup_multi_lap_speed_plot(
            ax_speed,
            laps_subset,
            lap_labels=lap_labels,
            title=speed_plot_title,
        )
    else:
        lines, vline = [], None

    track_pos_markers: List[Any] = []
    if ax_track is not None:
        for i in range(n):
            col = (
                track_marker_colors[i]
                if track_marker_colors is not None
                else f"C{i % 10}"
            )
            (mk,) = ax_track.plot(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=9,
                color=col,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=15,
                clip_on=True,
            )
            track_pos_markers.append(mk)

    timer = None
    mpl_slider_holder: dict = {"w": None}
    mpl_times = {"t": 0.0}

    def on_stop():
        timer.stop()

    def update_plot(tau: float):
        need_draw = False
        if vline is not None:
            vline.set_xdata([tau, tau])
            vline.set_visible(True)
            need_draw = True
        if track_pos_markers:
            for i, lap in enumerate(laps_subset):
                x, y = _enu_xy_at_tau_in_lap(lap, tau)
                m = track_pos_markers[i]
                if x is not None and y is not None:
                    m.set_data([x], [y])
                    m.set_visible(True)
                else:
                    m.set_visible(False)
            need_draw = True
        if need_draw:
            now = time.perf_counter()
            if now - mpl_times["t"] >= 0.05:
                mpl_times["t"] = now
                figure.canvas.draw_idle()
        wgt = mpl_slider_holder["w"]
        if wgt is not None and abs(float(wgt.val) - tau) > 0.005:
            wgt.eventson = False
            wgt.set_val(tau)
            wgt.eventson = True
        if extra_on_frame is not None:
            extra_on_frame(tau_in_lap=tau)

    player = CV2LapCompareVideoPlayer(
        filename,
        laps_subset,
        window_titles,
        video_time_offset,
        update_plot,
        on_stop,
        video_panel_width=video_panel_width,
        use_ffmpeg_segments=use_ffmpeg_segments,
        segment_cache_read=segment_cache_read,
        segment_cache_write=segment_cache_write,
    )

    if add_mpl_tau_slider:
        try:
            from matplotlib.widgets import Slider
        except ImportError:
            Slider = None  # type: ignore
        if Slider is not None:
            figure.subplots_adjust(bottom=0.18)
            rax = figure.add_axes([0.1, 0.06, 0.82, 0.03])
            s_mpl = Slider(
                rax,
                "Lap time (s)",
                0.0,
                float(player._tau_max),
                valinit=0.0,
                valstep=max(0.01, float(player._tau_max) / 2000.0),
                valfmt="%.2f",
            )

            def _on_mpl_slider(val: float) -> None:
                player.set_tau(float(val))

            s_mpl.on_changed(_on_mpl_slider)
            mpl_slider_holder["w"] = s_mpl
            player._matplotlib_tau_slider = s_mpl

    fps = max(player._fps, 1.0)
    interval = (
        timer_interval_ms
        if timer_interval_ms is not None
        else int(max(8, min(50, round(1000.0 / fps))))
    )
    timer = figure.canvas.new_timer(interval=interval, callbacks=[(player.on_timer, [], {})])
    timer.start()
    return player, timer, lines, vline


def start_video_player(filename: str, on_frame: Callable):
    _should_stop = False

    def on_stop():
        nonlocal _should_stop
        _should_stop = True

    video_player = CV2VideoPlayer(filename, on_frame, on_stop)

    while not _should_stop:
        video_player.on_timer()
        time.sleep(0.01)
