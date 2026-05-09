"""会话缓存（起点矩形、对比圈号）、ffmpeg 切片持久化，避免每次启动重复交互与转码。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CACHE_VERSION = 1


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
    if int(data.get("version", -1)) != CACHE_VERSION:
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
    compare_lap_indices_1based: Optional[List[int]] = None,
) -> None:
    root = cache_root()
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    path = session_cache_path(csv_path, video_path)
    payload: Dict[str, Any] = {
        "version": CACHE_VERSION,
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
    if compare_lap_indices_1based is not None:
        payload["compare_lap_indices_1based"] = [
            int(x) for x in compare_lap_indices_1based
        ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def segment_bundle_dir(
    video_path: str,
    video_time_offset: float,
    output_width: int,
    lap_time_ranges: List[Tuple[float, float]],
) -> Path:
    laps_norm = sorted(
        (round(float(a), 4), round(float(b), 4)) for a, b in lap_time_ranges
    )
    vid_fp = file_fingerprint(video_path)
    payload = {
        "cache_v": CACHE_VERSION,
        "laps": laps_norm,
        "offset": round(float(video_time_offset), 6),
        "video": vid_fp,
        "width": int(output_width),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    bid = hashlib.sha256(raw).hexdigest()[:28]
    return cache_root() / "segments" / bid


def try_read_segment_cache(
    video_path: str,
    video_time_offset: float,
    output_width: int,
    laps: List[Any],
) -> Optional[List[str]]:
    """若缓存完整且视频指纹、圈时间段一致，返回各圈切片绝对路径列表。"""
    try:
        lap_ranges = [(lap.t_start, lap.t_end) for lap in laps]
        d = segment_bundle_dir(
            video_path, video_time_offset, output_width, lap_ranges
        )
    except OSError:
        return None
    man_path = d / "manifest.json"
    if not man_path.is_file():
        return None
    try:
        with open(man_path, encoding="utf-8") as f:
            man: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if int(man.get("version", -1)) != CACHE_VERSION:
        return None
    if not fingerprint_match(
        man.get("video_fingerprint"), file_fingerprint(video_path)
    ):
        return None
    if int(man.get("num_laps", -1)) != len(laps):
        return None
    if int(man.get("output_width", -1)) != int(output_width):
        return None
    if abs(float(man.get("video_time_offset", 0.0)) - float(video_time_offset)) > 1e-5:
        return None
    stored_ranges = man.get("lap_ranges")
    if not stored_ranges or len(stored_ranges) != len(laps):
        return None
    for i, lap in enumerate(laps):
        a, b = stored_ranges[i]
        if abs(float(a) - lap.t_start) > 0.05 or abs(float(b) - lap.t_end) > 0.05:
            return None

    paths: List[str] = []
    for i in range(len(laps)):
        fp = d / f"lap_{i}.mp4"
        if not fp.is_file() or fp.stat().st_size < 800:
            return None
        paths.append(str(fp.resolve()))
    return paths


def write_segment_manifest(
    bundle_dir: Path,
    video_path: str,
    video_time_offset: float,
    output_width: int,
    laps: List[Any],
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": CACHE_VERSION,
        "video_fingerprint": file_fingerprint(video_path),
        "video_time_offset": float(video_time_offset),
        "output_width": int(output_width),
        "num_laps": len(laps),
        "lap_ranges": [[float(lap.t_start), float(lap.t_end)] for lap in laps],
    }
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def cache_skip_enabled() -> bool:
    return os.environ.get("LAP_VIZ_SKIP_CACHE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
