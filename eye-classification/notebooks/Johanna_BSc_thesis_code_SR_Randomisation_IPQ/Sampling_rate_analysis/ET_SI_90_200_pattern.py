import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

"""
Analyzes the temporal relationship between Varjo eye‑tracking timestamps and
Unity frame timestamps. For each model‑building phase, the script computes
intra‑Unity intervals (multiple Varjo samples within the same Unity frame) and
inter‑Unity intervals (gaps between consecutive Unity frames). It visualizes
both interval types using boxplots with jittered samples, computes descriptive
statistics, and performs non‑parametric tests (Mann–Whitney U and Wilcoxon) to
evaluate whether intra‑Unity intervals are significantly shorter than inter‑Unity
intervals. This analysis verifies the expected sampling structure of the
eye‑tracking system.
"""

# Load data
df = pd.read_csv("C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/ET_Data/ET_Cleaned/01_ET_Data_cleaned.csv")

#model_leng = df["model_name"].unique()
#print(len(model_leng))
# Ensure correct temporal order

df = df.sort_values(["model_name", "raw_timestamp"]).reset_index(drop=True)

# Convert units
df["raw_timestamp_ms"] = df["raw_timestamp"] / 1e6
df["raw_timestamp_unity_ms"] = df["raw_timestamp_unity"]

# Compute intra- and inter-group Varjo intervals
results = []

for model in df["model_name"].unique():
    sub = df[df["model_name"] == model].copy()

    # Group by Unity timestamp
    groups = sub.groupby("raw_timestamp_unity_ms")

    intra_dts = []  # intervals within Unity timestamp
    inter_dts = []  # intervals between Unity timestamp groups

    unity_times = sorted(sub["raw_timestamp_unity_ms"].unique())

    for i, t in enumerate(unity_times):
        group_varjo = groups.get_group(t)["raw_timestamp_ms"].values
        if len(group_varjo) > 1:
            intra_dts.extend(np.diff(group_varjo))  # Varjo intervals within this Unity timestamp
        if i < len(unity_times) - 1:
            next_group_varjo = groups.get_group(unity_times[i + 1])["raw_timestamp_ms"].values
            inter_dts.append(next_group_varjo[0] - group_varjo[-1])  # gap to next Unity timestamp group

    results.append({
        "model": model,
        "intra_dts": intra_dts,
        "inter_dts": inter_dts
    })

# Visualize
# Theme colors
intra_color = "#4C72B0"   # blue
inter_color = "#7B4C9A"   # purple
mean_marker_color = "#FFFFFF"
mean_edge_color = "#7B4C9A"
individual_point_color = "#B085C6"   # light purple
outlier_marker_color = "black"

for r in results:

    # Convert ms → seconds
    intra_sec = np.array(r["intra_dts"]) / 1000
    inter_sec = np.array(r["inter_dts"]) / 1000

    fig, ax = plt.subplots(figsize=(8, 6))  # larger figure

    # Create boxplot
    bp = ax.boxplot(
        [intra_sec, inter_sec],
        labels=["Intra-Group Interval", "Inter-Group Interval"],
        patch_artist=True,
        widths=0.25,  # tighter spacing
        showmeans=True,
        meanline=False,
        meanprops=dict(marker='o', markerfacecolor=mean_marker_color,
                       markeredgecolor=mean_edge_color, markersize=8),
        flierprops=dict(marker='o', markerfacecolor=outlier_marker_color,
                        markeredgecolor='none', markersize=5)
    )

    # Color boxes
    bp['boxes'][0].set_facecolor(intra_color)
    bp['boxes'][1].set_facecolor(inter_color)

    # Color medians
    bp['medians'][0].set_color("black")
    bp['medians'][1].set_color("black")

    # Add jittered individual samples
    x1 = np.random.normal(1, 0.03, size=len(intra_sec))
    x2 = np.random.normal(2, 0.03, size=len(inter_sec))
    ax.scatter(x1, intra_sec, color=individual_point_color, alpha=0.6, s=12)
    ax.scatter(x2, inter_sec, color=individual_point_color, alpha=0.6, s=12)

    # Legend
    legend_patches = [
        mpatches.Patch(color=intra_color, label="Intra-Group Interval"),
        mpatches.Patch(color=inter_color, label="Inter-Group Interval"),
        mpatches.Patch(facecolor=mean_marker_color, edgecolor=mean_edge_color,
                       label="Mean"),
        mpatches.Patch(color=individual_point_color, label="Individual Samples"),
        mpatches.Patch(color=outlier_marker_color, label="Outliers")
    ]
    ax.legend(handles=legend_patches, loc="upper left", frameon=False, fontsize=18)

    # Labels and title
    ax.set_ylabel("Varjo timestamp interval (seconds)", fontsize=18)
    ax.set_title(f"Varjo intervals relative to Unity timestamps – {r['model']}", fontsize=20)
    ax.set_ylim(0, 0.04)

    # Tick label size
    ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    plt.show()



# Optional: numeric summary
for r in results:
    print(f"Model: {r['model']}")
    print(f"Mean intra-Group Interval: {np.mean(r['intra_dts']):.2f} ms")
    print(f"Mean inter-Group Interval: {np.mean(r['inter_dts']):.2f} ms")
    print("---")

# Statistical comparison: Mann–Whitney U test across participants

def analyze_intervals(intra_dts, inter_dts):
    # Mann–Whitney U test (one-sided: intra < inter)
    stat, p_value = mannwhitneyu(intra_dts, inter_dts, alternative='less')

    # Rank-biserial correlation (effect size)
    n1 = len(intra_dts)
    n2 = len(inter_dts)
    rank_biserial = 1 - (2 * stat) / (n1 * n2)

    print(f"Mann–Whitney U statistic: {stat:.2f}")
    print(f"One-sided p-value (intra < inter): {p_value:.6f}")
    print(f"Rank-biserial correlation (effect size): {rank_biserial:.3f}")
    print("------------------------------------------------------------")

print("\n=== Statistical Test: Intra vs Inter Group Intervals ===")
for r in results:
    print(f"\nModel: {r['model']}")
    analyze_intervals(r["intra_dts"], r["inter_dts"])



def compute_intervals(df):
    """
    Compute intra-Unity and inter-Unity Varjo timestamp intervals
    for each model in a participant's dataset.

    Returns:
        results: list of dicts with keys:
            - model
            - intra_dts
            - inter_dts
    """

    # Ensure correct temporal order
    df = df.sort_values(["model_name", "raw_timestamp"]).reset_index(drop=True)

    # Convert units
    df["raw_timestamp_ms"] = df["raw_timestamp"] / 1e6
    df["raw_timestamp_unity_ms"] = df["raw_timestamp_unity"]

    results = []

    for model in df["model_name"].unique():
        sub = df[df["model_name"] == model].copy()

        # Group by Unity timestamp
        groups = sub.groupby("raw_timestamp_unity_ms")

        intra_dts = []  # intervals within Unity timestamp
        inter_dts = []  # intervals between Unity timestamp groups

        unity_times = sorted(sub["raw_timestamp_unity_ms"].unique())

        for i, t in enumerate(unity_times):
            group_varjo = groups.get_group(t)["raw_timestamp_ms"].values

            # Intra-Unity intervals
            if len(group_varjo) > 1:
                intra_dts.extend(np.diff(group_varjo))

            # Inter-Unity intervals
            if i < len(unity_times) - 1:
                next_group_varjo = groups.get_group(unity_times[i + 1])["raw_timestamp_ms"].values
                inter_dts.append(next_group_varjo[0] - group_varjo[-1])

        results.append({
            "model": model,
            "intra_dts": intra_dts,
            "inter_dts": inter_dts
        })

    return results


participant_diffs = []  # one value per participant
all_participants = [f"{i:02d}" for i in range(1, 11)]
for participant_id in all_participants:
    # Load participant CSV
    df = pd.read_csv(f"C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/ET_Data/ET_Cleaned/{participant_id}_ET_Data_cleaned.csv")
    df = df.sort_values(["model_name", "raw_timestamp"]).reset_index(drop=True)

    # Compute intervals per model
    results = compute_intervals(df)

    # Combine all models for this participant
    all_intra = []
    all_inter = []

    for r in results:
        all_intra.extend(r["intra_dts"])
        all_inter.extend(r["inter_dts"])

    # Compute participant-level means
    mean_intra = np.mean(all_intra)
    mean_inter = np.mean(all_inter)

    # Store difference (intra - inter)
    participant_diffs.append(mean_intra - mean_inter)

# Group-level statistical test
stat, p_value = wilcoxon(participant_diffs, alternative='less')

print("=== Group-Level Test Across Participants ===")
print(f"Wilcoxon signed-rank statistic: {stat:.3f}")
print(f"One-sided p-value (intra < inter): {p_value:.6f}")
print(f"Participant differences: {participant_diffs}")