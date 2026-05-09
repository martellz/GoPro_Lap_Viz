"""会话缓存（起点矩形、时间偏移）、按单圈持久化 ffmpeg 切片。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# 会话 JSON（矩形、offset）；不含对比圈号
SESSION_CACHE_VERSION = 1
# 每个切片目录内 manifest.json 的版本号
LAP_CLIP_MANIFEST_VERSION = 1


def cache_root() -> Path:
    env = os.environ.get("LAP_VIZ_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser() / "gopro_lap_viz").resolve()
    return (Path.home() / ".cache" / "gopro_lap_viz").resolve()


def file_fingerprint(path: str) -> Dict[str, Any]:
    p = Path(path).resolve()
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)}


def fingerprint_match(
    stored: Optional[Dict[str, Any]], current: Dict[str, Any]
) -> bool:
    if not stored:
        return False
    return stored.get("size") == current["size"] and stored.get("mtime") == current[
        "mtime"
    ]


def session_cache_path(csv_path: str, video_path: str) -> Path:
    raw = f"{Path(csv_path).resolve()}|{Path(video_path).resolve()}".encode("utf-8")
    key = hashlib.sha256(raw).hexdigest()[:24]
    return cache_root() / "sessions" / f"{key}.json"


def load_session(csv_path: str, video_path: str) -> Optional[Dict[str, Any]]:
    p = session_cache_path(csv_path, video_path)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def session_is_valid(data: Dict[str, Any], csv_path: str, video_path: str) -> bool:
    if int(data.get("version", -1)) != SESSION_CACHE_VERSION:
        return False
    if not fingerprint_match(data.get("csv_fingerprint"), file_fingerprint(csv_path)):
        return False
    if not fingerprint_match(
        data.get("video_fingerprint"), file_fingerprint(video_path)
    ):
        return False
    rect = data.get("rectangle_enu")
    if not rect or not all(k in rect for k in ("xmin", "xmax", "ymin", "ymax")):
        return False
    return True


def save_session(
    csv_path: str,
    video_path: str,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    video_time_offset: float,
) -> None:
    root = cache_root()
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    path = session_cache_path(csv_path, video_path)
    payload: Dict[str, Any] = {
        "version": SESSION_CACHE_VERSION,
        "csv_fingerprint": file_fingerprint(csv_path),
        "video_fingerprint": file_fingerprint(video_path),
        "video_time_offset": float(video_time_offset),
        "rectangle_enu": {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def single_lap_segment_dir(
    video_path: str,
    video_time_offset: float,
    output_width: int,
    t_start: float,
    t_end: float,
) -> Path:
    """单圈切片缓存目录（与对比选了哪几圈无关，只由时间区间与视频指纹决定）。"""
    a, b = round(float(t_start), 4), round(float(t_end), 4)
    vid_fp = file_fingerprint(video_path)
    payload = {
        "clip_v": LAP_CLIP_MANIFEST_VERSION,
        "lap": [a, b],
        "offset": round(float(video_time_offset), 6),
        "video": vid_fp,
        "width": int(output_width),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    bid = hashlib.sha256(raw).hexdigest()[:28]
    return cache_root() / "segments" / "by_lap" / bid


def try_read_single_lap_segment(
    video_path: str,
    video_time_offset: float,
    output_width: int,
    lap: Any,
) -> Optional[str]:
    """若该圈切片已缓存且校验通过，返回 clip.mp4 绝对路径，否则 None。"""
    try:
        d = single_lap_segment_dir(
            video_path,
            video_time_offset,
            output_width,
            lap.t_start,
            lap.t_end,
        )
    except OSError:
        return None
    man_path = d / "manifest.json"
    clip = d / "clip.mp4"
    if not man_path.is_file() or not clip.is_file():
        return None
    try:
        with open(man_path, encoding="utf-8") as f:
            man: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if int(man.get("version", -1)) != LAP_CLIP_MANIFEST_VERSION:
        return None
    if not fingerprint_match(
        man.get("video_fingerprint"), file_fingerprint(video_path)
    ):
        return None
    if int(man.get("output_width", -1)) != int(output_width):
        return None
    if abs(float(man.get("video_time_offset", 0.0)) - float(video_time_offset)) > 1e-5:
        return None
    if abs(float(man.get("t_start", 0.0)) - float(lap.t_start)) > 0.05:
        return None
    if abs(float(man.get("t_end", 0.0)) - float(lap.t_end)) > 0.05:
        return None
    if clip.stat().st_size < 800:
        return None
    return str(clip.resolve())


def write_single_lap_segment_manifest(
    bundle_dir: Path,
    video_path: str,
    video_time_offset: float,
    output_width: int,
    lap: Any,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": LAP_CLIP_MANIFEST_VERSION,
        "video_fingerprint": file_fingerprint(video_path),
        "video_time_offset": float(video_time_offset),
        "output_width": int(output_width),
        "t_start": float(lap.t_start),
        "t_end": float(lap.t_end),
    }
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def cache_skip_enabled() -> bool:
    return os.environ.get("LAP_VIZ_SKIP_CACHE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
