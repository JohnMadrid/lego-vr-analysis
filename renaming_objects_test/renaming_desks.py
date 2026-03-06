import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# INPUT:
# A CSV or dataframe with at least these two columns:
#   gaze_capture_time   (nanoseconds, int)
#   hit_obj_name        (string)
#
# OUTPUT:
# A time‑series plot showing which object was hit at each time.
# ---------------------------------------------------------

# Example: load your eye‑tracking CSV
df = pd.read_csv("renaming_objects_test/nametest_ET_Data_2026-01-22.csv")

# Convert nanoseconds → seconds (float)
df["time_seconds"] = df["gaze_capture_time"] * 1e-9

# Sort by time (important if data is unordered)
df = df.sort_values("time_seconds")

# Create a categorical mapping for hit_obj_name
# Each unique object gets a numeric ID for plotting
df["obj_id"] = df["hit_obj_name"].astype("category").cat.codes
obj_labels = dict(enumerate(df["hit_obj_name"].astype("category").cat.categories))

# Plot
plt.figure(figsize=(14, 5))
plt.scatter(df["time_seconds"], df["obj_id"], s=10)

plt.yticks(list(obj_labels.keys()), list(obj_labels.values()))
plt.xlabel("Time (seconds)")
plt.ylabel("Hit Object")
plt.title("Eye‑Tracking: hit_obj_name over time")
plt.grid(True, linestyle="--", alpha=0.4)

plt.show()
