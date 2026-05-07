import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))


from src.lap_utils import read_GPMF_from_csv, select_start_rectangle_and_split_laps

df = read_GPMF_from_csv("test_data/GX10025.csv")
print(df)

laps = select_start_rectangle_and_split_laps(df)
print(f"分圈数量: {len(laps)}")

# plot the laps with different colors
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# for i, lap in enumerate(laps):
#   ax.plot(lap.segment["enu_x"], lap.segment["enu_y"], color=f"C{i}")
# plt.show()

# 将所有 lap 的 segment 拼接起来可视化
# import matplotlib.pyplot as plt
# import pandas as pd
# df_all = pd.concat([lap.segment for lap in laps], ignore_index=True)
# fig, ax = plt.subplots()
# ax.plot(df_all["enu_x"], df_all["enu_y"], color="black")
# plt.show()

for i, lap in enumerate(laps):
  print(f"  lap {i + 1}: {len(lap)} 点, time {lap.t_start:.3f} .. {lap.t_end:.3f}, lap time: {lap.t_end - lap.t_start:.3f} seconds")