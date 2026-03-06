import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# --- Load all CSV files from folder ---
folder_path = "C:/Users/experiment/Desktop/temp_BT_data"

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# --- Remove zero rows (tracking loss) ---
df = df[
    ~(
        (df["RightFoot_pos_x"] == 0) &
        (df["RightFoot_pos_y"] == 0) &
        (df["RightFoot_pos_z"] == 0)
    )
]
df = df[
    ~(
        (df["LeftFoot_pos_x"] == 0) &
        (df["LeftFoot_pos_y"] == 0) &
        (df["LeftFoot_pos_z"] == 0)
    )
]

# --- Extract foot positions ---
rf_x = df["RightFoot_pos_x"]
rf_z = df["RightFoot_pos_z"]
lf_x = df["LeftFoot_pos_x"]
lf_z = df["LeftFoot_pos_z"]

# --- Compute bounding box of visited space ---
all_x = pd.concat([rf_x, lf_x])
all_z = pd.concat([rf_z, lf_z])

margin = 0.1  # adjust if needed

xmin, xmax = all_x.min() - margin, all_x.max() + margin
zmin, zmax = all_z.min() - margin, all_z.max() + margin

# --- Heatmaps with cropping ---
fig = plt.figure(figsize=(12, 5))

# Right Foot
ax1 = fig.add_subplot(1, 2, 1)
hb1 = ax1.hist2d(rf_x, rf_z, bins=60, cmap="hot")
ax1.set_title("Right Foot Position Heatmap (Cropped)")
ax1.set_xlabel("X")
ax1.set_ylabel("Z")
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(zmin, zmax)
fig.colorbar(hb1[3], ax=ax1)

# Left Foot
ax2 = fig.add_subplot(1, 2, 2)
hb2 = ax2.hist2d(lf_x, lf_z, bins=60, cmap="hot")
ax2.set_title("Left Foot Position Heatmap (Cropped)")
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(zmin, zmax)
fig.colorbar(hb2[3], ax=ax2)

plt.tight_layout()
plt.show()
