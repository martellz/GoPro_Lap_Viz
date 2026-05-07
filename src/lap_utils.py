import pandas as pd
import numpy as np
from latlon2xyz import wgs84_to_enu
from typing import List, Tuple


_GPMF_COLUMNS: Tuple[str, ...] = (
    "latitude",
    "longitude",
    "elevation",
    "time",
    "speed",
)


# a single lap of a race
class Lap:
  """一段连续Telemetry：从一次进入起点矩形到下一次进入之前（含末段直到轨迹结束）。"""

  def __init__(self, segment: pd.DataFrame):
    self.segment = segment.reset_index(drop=True)
    self.t_start = float(self.segment["time"].iloc[0])
    self.t_end = float(self.segment["time"].iloc[-1])

  def __len__(self) -> int:
    return len(self.segment)


def _lap_entry_indices(
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
  if xmin > xmax:
    xmin, xmax = xmax, xmin
  if ymin > ymax:
    ymin, ymax = ymax, ymin
  inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
  prev_inside = np.roll(inside, 1)
  prev_inside[0] = False
  entries = np.nonzero(inside & ~prev_inside)[0]
  return entries


def split_trajectory_into_laps(
    df: pd.DataFrame,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    x_col: str = "enu_x",
    y_col: str = "enu_y",
) -> List[Lap]:
  """按「每次进入起点矩形」分圈：第 k 圈为从第 k 次进入矩形到下一次进入前一行（最后一圈直到数据结束）。

  矩形与轨迹点使用同一平面坐标（默认 read_GPMF_from_csv 得到的 enu_x / enu_y）。
  """
  for col in (x_col, y_col):
    if col not in df.columns:
      raise ValueError(f"DataFrame 缺少列 {col!r}，请先计算 ENU 或使用对应列名")

  x = df[x_col].to_numpy(dtype=np.float64, copy=False)
  y = df[y_col].to_numpy(dtype=np.float64, copy=False)
  entries = _lap_entry_indices(x, y, xmin, xmax, ymin, ymax)
  if len(entries) == 0:
    raise ValueError("轨迹从未进入给定矩形区域，无法分圈（请放大矩形或检查坐标列）")

  laps: List[Lap] = []
  print(entries)
  for k in range(len(entries)):
    start = int(entries[k])
    end = int(entries[k + 1]) if k + 1 < len(entries) else len(df)
    laps.append(Lap(df.iloc[start:end]))
  return laps


def _configure_matplotlib_cjk_font() -> bool:
  """若系统已注册可用的中日韩字体，则配置 matplotlib 使用之，避免中文标题缺字形告警。"""
  import matplotlib as mpl
  from matplotlib import font_manager

  prefer = (
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Noto Serif CJK SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Droid Sans Fallback",
  )
  available = {f.name for f in font_manager.fontManager.ttflist}
  for name in prefer:
    if name in available:
      mpl.rcParams["font.family"] = "sans-serif"
      mpl.rcParams["font.sans-serif"] = [name] + [
        x for x in mpl.rcParams["font.sans-serif"] if x != name
      ]
      mpl.rcParams["axes.unicode_minus"] = False
      return True
  for f in font_manager.fontManager.ttflist:
    if "Noto" in f.name and "CJK" in f.name:
      mpl.rcParams["font.family"] = "sans-serif"
      mpl.rcParams["font.sans-serif"] = [f.name] + [
        x for x in mpl.rcParams["font.sans-serif"] if x != f.name
      ]
      mpl.rcParams["axes.unicode_minus"] = False
      return True
  return False


def select_start_rectangle_interactive(
    df: pd.DataFrame,
    *,
    x_col: str = "enu_x",
    y_col: str = "enu_y",
    figsize: Tuple[float, float] = (10.0, 8.0),
) -> Tuple[float, float, float, float]:
  """在轨迹平面图上用鼠标拖拽框选起点矩形，关闭图像窗口后返回 (xmin, xmax, ymin, ymax)。"""
  import matplotlib.pyplot as plt
  from matplotlib.widgets import RectangleSelector

  use_cjk_title = _configure_matplotlib_cjk_font()
  state = {"extents": None}

  fig, ax = plt.subplots(figsize=figsize)
  t = df["time"].to_numpy() if "time" in df.columns else None
  sc = ax.scatter(df[x_col], df[y_col], s=1, c=t, cmap="viridis") if t is not None else ax.scatter(df[x_col], df[y_col], s=1)
  if t is not None:
    fig.colorbar(sc, ax=ax, label="time")
  ax.set_aspect("equal")
  ax.set_xlabel(x_col)
  ax.set_ylabel(y_col)
  if use_cjk_title:
    ax.set_title("拖拽框选起点矩形；关闭本窗口以确认选择")
  else:
    ax.set_title(
      "Drag rectangle: start/finish zone — close window to confirm "
      "(install fonts-noto-cjk or WenQuanYi for Chinese UI)"
    )

  def onselect(eclick, erelease):
    state["extents"] = selector.extents

  selector = RectangleSelector(
    ax,
    onselect,
    useblit=True,
    button=[1],
    minspanx=1e-6,
    minspany=1e-6,
    spancoords="data",
    interactive=True,
  )

  plt.show()

  ex = state["extents"]
  if ex is None:
    ex = selector.extents
  if ex is None:
    raise RuntimeError("未获取到矩形范围，请先拖拽框选后再关闭窗口")

  xmin, xmax, ymin, ymax = (float(ex[0]), float(ex[1]), float(ex[2]), float(ex[3]))
  if xmin > xmax:
    xmin, xmax = xmax, xmin
  if ymin > ymax:
    ymin, ymax = ymax, ymin
  return xmin, xmax, ymin, ymax


def select_start_rectangle_and_split_laps(
    df: pd.DataFrame,
    *,
    x_col: str = "enu_x",
    y_col: str = "enu_y",
    figsize: Tuple[float, float] = (10.0, 8.0),
) -> List[Lap]:
  """交互框选起点矩形并直接分圈。"""
  rect = select_start_rectangle_interactive(df, x_col=x_col, y_col=y_col, figsize=figsize)
  return split_trajectory_into_laps(df, *rect, x_col=x_col, y_col=y_col)


def _normalize_time_series(s: pd.Series) -> pd.Series:
  num = pd.to_numeric(s, errors="coerce")
  if not num.isna().all():
    if num.isna().any():
      raise ValueError("Column 'time' mixes numeric and non-numeric values")
    return num.astype("float64")

  dt = pd.to_datetime(s, utc=True, errors="coerce")
  if dt.isna().any():
    raise ValueError(
      "Column 'time' must be numeric (e.g. seconds) or parseable datetimes"
    )
  return dt.astype("int64").astype("float64") / 1e9


def read_GPMF_from_csv(csv_path: str) -> pd.DataFrame:
  """Read a telemetry CSV and keep only latitude, longitude, elevation, time, speed.

  *time* may be numeric (e.g. seconds / epoch) or ISO-like datetime strings; if datetime,
  it is converted to Unix seconds (UTC) as float64.
  """
  df = pd.read_csv(csv_path)
  missing = [c for c in _GPMF_COLUMNS if c not in df.columns]
  if missing:
    raise ValueError(
      f"CSV '{csv_path}' is missing columns: {missing}; have: {list(df.columns)}"
    )

  out = df.loc[:, list(_GPMF_COLUMNS)].copy()

  for c in ("latitude", "longitude", "elevation", "speed"):
    v = pd.to_numeric(out[c], errors="coerce")
    if v.isna().any():
      raise ValueError(f"CSV column {c!r} has invalid or missing numeric values")
    out[c] = v.astype("float64")

  out["time"] = _normalize_time_series(out["time"])
  out["time"] = out["time"] - out["time"].iloc[0]

  # convert to ENU coordinates
  out["enu_x"], out["enu_y"], out["enu_z"] = wgs84_to_enu(out["latitude"], out["longitude"], np.zeros_like(out["elevation"]))

  return out

