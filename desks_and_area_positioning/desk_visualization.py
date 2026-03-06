import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv(f"desks_and_area_positioning/table_marking_BT_Data_2026-01-21.csv")

# Filter rows where all hand buttons are pressed
mask = (
    (df["RightHand_trigger_pressed"] == True) &
    (df["RightHand_grip_pressed"] == True) &
    (df["LeftHand_trigger_pressed"] == True) &
    (df["LeftHand_grip_pressed"] == True)
)

filtered = df[mask]

# Extract tracker positions
right_hand = filtered[["RightHand_pos_x", "RightHand_pos_y", "RightHand_pos_z"]]
left_hand  = filtered[["LeftHand_pos_x", "LeftHand_pos_y", "LeftHand_pos_z"]]
right_foot = filtered[["RightFoot_pos_x", "RightFoot_pos_y", "RightFoot_pos_z"]]
left_foot  = filtered[["LeftFoot_pos_x", "LeftFoot_pos_y", "LeftFoot_pos_z"]]
waist      = filtered[["Waist_pos_x", "Waist_pos_y", "Waist_pos_z"]]

# Plot tracker paths (2D)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

ax.plot(right_hand["RightHand_pos_x"], right_hand["RightHand_pos_z"], label="Right Hand")
ax.plot(left_hand["LeftHand_pos_x"], left_hand["LeftHand_pos_z"], label="Left Hand")

ax.set_title("Tracker Paths (Filtered by Button Presses)")
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.legend()
plt.tight_layout()
plt.show()

# Grabbed objects
grabbed_left  = filtered["LeftHand_grabbed_name"].dropna().unique()
grabbed_right = filtered["RightHand_grabbed_name"].dropna().unique()

print("Objects grabbed with left hand:", grabbed_left)
print("Objects grabbed with right hand:", grabbed_right)

# Plot grab events (2D)
grab_events = filtered[
    (filtered["LeftHand_grabbed_name"].notna()) |
    (filtered["RightHand_grabbed_name"].notna())
]

fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111)

# Left-hand grab markers
left_grabs = grab_events[grab_events["LeftHand_grabbed_name"].notna()]
ax2.scatter(
    left_grabs["LeftHand_pos_x"],
    left_grabs["LeftHand_pos_z"],
    c="blue",
    label="Left-hand grab",
    s=40
)


# Right-hand grab markers
right_grabs = grab_events[grab_events["RightHand_grabbed_name"].notna()]
ax2.scatter(
    right_grabs["RightHand_pos_x"],
    right_grabs["RightHand_pos_z"],
    c="red",
    label="Right-hand grab",
    s=40
)


ax2.set_title("Grab Events (Hand Positions)")
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
ax2.legend()
plt.tight_layout()
plt.show()
