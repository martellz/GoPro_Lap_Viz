"""
读取 GX10025 遥测 → 交互分圈 → 可视化多圈 ENU 轨迹与速度；可选同步视频。

视频路径：
  - 环境变量 LAP_VIZ_VIDEO（默认项目根下 test_video1.mp4）
时间对齐：
  - 环境变量 LAP_VIZ_TIME_OFFSET（默认与 CSV 第一行 time 一致，即假定视频 0s 对应首条遥测）

对比多圈（多窗口同步播放 + 仅所选圈的速度曲线）：
  - LAP_COMPARE_LAPS 设为逗号分隔的「圈号」，从 1 开始，例如 2 与 13：
      LAP_COMPARE_LAPS=2,13
  - 对比模式默认会用 **ffmpeg** 先按圈截取短视频再播放（顺序解码，流畅）。
    若不要切片（整文件 seek）：设置环境变量 LAP_COMPARE_NO_FFMPEG=1
  - 未设置 LAP_COMPARE_LAPS 时仍使用单窗口顺序播放整段素材（所有圈 + attach_multi_lap_video_player）。

运行前请安装 ffmpeg（例如 apt install ffmpeg）。

缓存（跳过下次框选与 ffmpeg 切片）：
  - 默认写入 ~/.cache/gopro_lap_viz（或 XDG_CACHE_HOME，或环境变量 LAP_VIZ_CACHE）
  - LAP_VIZ_SKIP_CACHE=1：禁用会话与切片缓存

运行（项目根目录）：
  python test/test_lap_utils.py
  LAP_VIZ_VIDEO=/path/to/clip.mp4 LAP_COMPARE_LAPS=2,13 python test/test_lap_utils.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt

from lap_utils import read_GPMF_from_csv, select_start_rectangle_and_split_laps, split_trajectory_into_laps
from lap_cache import cache_skip_enabled, load_session, save_session, session_is_valid
from lap_video_player import (
    attach_lap_compare_video_player,
    attach_multi_lap_video_player,
    setup_multi_lap_speed_plot,
)


def main(csv_path=None, video_path=None) -> None:
    if csv_path is None:
        csv_path = ROOT / "test_data" / "GX10025.csv"
    if video_path is None:
        video_path = ROOT / "test_data" / "GX010025.MP4"
    csv_str = str(csv_path.resolve())
    video_str = str(video_path.resolve()) if video_path.is_file() else ""

    df = read_GPMF_from_csv(csv_str)
    print(df)

    default_offset = float(df["time"].iloc[0])
    if os.environ.get("LAP_VIZ_TIME_OFFSET") is not None:
        time_offset = float(os.environ["LAP_VIZ_TIME_OFFSET"])
    else:
        time_offset = default_offset

    use_cache = not cache_skip_enabled()
    session_data = (
        load_session(csv_str, video_str) if (use_cache and video_str) else None
    )
    valid = bool(
        session_data and session_is_valid(session_data, csv_str, video_str)
    )
    if valid and session_data is not None and os.environ.get("LAP_VIZ_TIME_OFFSET") is None:
        time_offset = float(session_data.get("video_time_offset", time_offset))

    if valid and session_data is not None:
        r = session_data["rectangle_enu"]
        rect = (r["xmin"], r["xmax"], r["ymin"], r["ymax"])
        laps = split_trajectory_into_laps(df, *rect)
        print("已用缓存的 ENU 起点矩形分圈，跳过框选。")
    else:
        laps, rect = select_start_rectangle_and_split_laps(df)
    print(f"分圈数量: {len(laps)}")

    for i, lap in enumerate(laps):
        print(
            f"  lap {i + 1}: {len(lap)} 点, "
            f"time {lap.t_start:.3f} .. {lap.t_end:.3f}, "
            f"lap time: {lap.t_end - lap.t_start:.3f} seconds"
        )

    compare_spec = os.environ.get("LAP_COMPARE_LAPS", "").strip()
    picks_1based: list[int] | None = None
    if compare_spec:
        parts = [p.strip() for p in compare_spec.split(",") if p.strip()]
        picks_1based = [int(p) for p in parts]
        picks0 = [p - 1 for p in picks_1based]
        bad = [p for p in picks0 if p < 0 or p >= len(laps)]
        if bad:
            raise SystemExit(
                f"LAP_COMPARE_LAPS 超出范围: {compare_spec!r}，当前共 {len(laps)} 圈（1..{len(laps)}）"
            )
        selected = [laps[i] for i in picks0]
        labels = [f"Lap {picks_1based[j]}" for j in range(len(selected))]
    elif valid and session_data is not None and session_data.get(
        "compare_lap_indices_1based"
    ):
        picks_1based = [int(x) for x in session_data["compare_lap_indices_1based"]]
        picks0 = [p - 1 for p in picks_1based]
        bad = [p for p in picks0 if p < 0 or p >= len(laps)]
        if bad:
            picks_1based = None
            selected = None
            labels = None
        else:
            selected = [laps[i] for i in picks0]
            labels = [f"Lap {picks_1based[j]}" for j in range(len(selected))]
    else:
        selected = None
        labels = None

    track_indices = (
        picks0 if selected is not None else list(range(len(laps)))
    )

    if (
        use_cache
        and video_str
        and (not valid or compare_spec)
    ):
        xmin, xmax, ymin, ymax = rect
        save_session(
            csv_str,
            video_str,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            video_time_offset=time_offset,
            compare_lap_indices_1based=picks_1based,
        )
        print("已更新会话缓存（矩形 / 时间偏移 / 对比圈号）。")

    fig, (ax_track, ax_speed) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    for plot_i, lap_idx in enumerate(track_indices):
        lap = laps[lap_idx]
        seg = lap.segment
        lbl = (
            labels[plot_i]
            if labels is not None
            else f"Lap {lap_idx + 1}"
        )
        ax_track.plot(
            seg["enu_x"],
            seg["enu_y"],
            color=f"C{plot_i % 10}",
            linewidth=1.2,
            label=lbl,
        )
    ax_track.set_aspect("equal")
    ax_track.set_xlabel("enu_x (m)")
    ax_track.set_ylabel("enu_y (m)")
    ax_track.set_title("多圈轨迹 (ENU)")
    ax_track.legend(loc="best", fontsize=8)
    ax_track.grid(True, alpha=0.3)

    if video_path.is_file():
        print(f"视频: {video_path}")
        print(f"video_time_offset = {time_offset} （telemetry = video_ts + offset）")
        if selected is not None:
            print(f"对比模式: {[lb for lb in labels]}")
            attach_lap_compare_video_player(
                fig,
                str(video_path),
                selected,
                ax_speed,
                video_time_offset=time_offset,
                lap_labels=labels,
                speed_plot_title="Speed — selected laps",
                ax_track=ax_track,
            )
        else:
            attach_multi_lap_video_player(
                fig,
                str(video_path),
                laps,
                ax_speed,
                video_time_offset=time_offset,
                auto_advance_laps=True,
            )
    else:
        if selected is not None:
            setup_multi_lap_speed_plot(
                ax_speed,
                selected,
                lap_labels=labels,
                title="Speed — selected laps",
            )
        else:
            setup_multi_lap_speed_plot(ax_speed, laps)
        print(
            f"未找到视频文件: {video_path}\n"
            "已仅绘制速度曲线。指定视频例如：\n"
            '  LAP_VIZ_VIDEO=/your/clip.mp4 python test/test_lap_utils.py'
        )

    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.show()


if __name__ == "__main__":
    main()
