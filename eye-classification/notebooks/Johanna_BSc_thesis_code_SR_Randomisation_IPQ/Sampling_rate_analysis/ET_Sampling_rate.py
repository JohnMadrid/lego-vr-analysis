import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

"""
Computes sampling rates for eye‑tracking data on a phase‑by‑phase and
participant‑level basis. The script loads cleaned ET CSV files, removes
irrelevant entries, calculates phase durations and sampling rates, and derives
weighted mean sampling rates per participant. It also produces descriptive
statistics across all phases and participants and generates a comprehensive PDF
report including boxplots, histograms, heatmaps, and barplots. This analysis
supports evaluating sampling‑rate stability, recording quality, and overall
data characteristics for the eye‑tracking dataset.
"""

# Visual theme to match cleaned_uniformity_sampling_intervals
label_fontsize = 12
title_fontsize = 12
tick_fontsize = 10
text_fontsize = 12
legend_fontsize = 10
sns.set_theme(context="talk", style="whitegrid")


def compute_sampling_rates(et_folder: str, output_folder: str, ts_unit="ns"):
    """
    Berechnet Sampling Rates pro Phase, pro Participant, und overall.
    Speichert:
      - CSV pro Participant mit 18 Phasen: Dauer, Samples, mean sampling rate
      - CSV mit allen Phasen aller Participants (10*18 Zeilen)
      - CSV mit mean sampling rate pro Participant + descriptive statistics
    """

    os.makedirs(output_folder, exist_ok=True)
    per_participant_folder = os.path.join(output_folder, "per_participant")
    os.makedirs(per_participant_folder, exist_ok=True)

    unit_div = {"s":1.0, "ms":1e3, "us":1e6, "ns":1e9}[ts_unit]

    all_phases_list = []
    participant_summary_list = []

    # Process each participant CSV
    for csv_file in os.listdir(et_folder):
        if not csv_file.endswith(".csv"):
            continue

        pid = os.path.basename(csv_file)[:2].zfill(2)
        et_path = os.path.join(et_folder, csv_file)
        df = pd.read_csv(et_path)
        # remove invalid models
        df = df[~df["model_name"].isin(["TM","None"])].copy()
        df["raw_timestamp"] = pd.to_numeric(df["raw_timestamp"], errors="coerce")
        df = df.dropna(subset=["raw_timestamp"])

        # get unique model names in temporal order
        model_order = df["model_name"].drop_duplicates().tolist()

        participant_phase_data = []
        total_samples = 0
        total_time = 0.0

        for model_name in model_order:
            phase_rows = df[df["model_name"] == model_name].copy()
            if phase_rows.empty:
                continue

            start_ts = phase_rows["raw_timestamp"].min()
            end_ts = phase_rows["raw_timestamp"].max()
            duration_s = (end_ts - start_ts)/unit_div
            n_samples = phase_rows.shape[0]
            if duration_s <= 0 or n_samples < 2:
                continue
            mean_sr = n_samples/duration_s

            participant_phase_data.append({
                "participant": pid,
                "model_name": model_name,
                "duration_s": duration_s,
                "n_samples": n_samples,
                "mean_sampling_rate_Hz": mean_sr
            })

            all_phases_list.append({
                "participant": pid,
                "model_name": model_name,
                "duration_s": duration_s,
                "n_samples": n_samples,
                "mean_sampling_rate_Hz": mean_sr
            })

            total_samples += n_samples
            total_time += duration_s

        # Mean Sampling Rate pro Participant (gewichtetes Mittel)
        if total_time > 0:
            mean_sr_participant = total_samples/total_time
        else:
            mean_sr_participant = np.nan

        # Save per-participant CSV
        participant_df = pd.DataFrame(participant_phase_data)
        participant_df.to_csv(os.path.join(per_participant_folder, f"{pid}_phases.csv"), index=False)

        # Collect participant summary
        sampling_rates = participant_df["mean_sampling_rate_Hz"].values
        participant_summary_list.append({
            "participant": pid,
            "mean_sampling_rate_Hz": mean_sr_participant,
            "median_sampling_rate_Hz": np.median(sampling_rates),
            "std_sampling_rate_Hz": np.std(sampling_rates, ddof=1),
            "min_sampling_rate_Hz": np.min(sampling_rates),
            "max_sampling_rate_Hz": np.max(sampling_rates),
            "n_phases": len(sampling_rates)
        })

    # CSV mit allen Phasen aller Participants
    all_phases_df = pd.DataFrame(all_phases_list)
    all_phases_df.to_csv(os.path.join(output_folder, "all_participants_phases.csv"), index=False)

    # CSV mit per-Participant mean + descriptive stats
    participant_summary_df = pd.DataFrame(participant_summary_list)
    participant_summary_df.to_csv(os.path.join(output_folder, "participants_summary.csv"), index=False)

    # Grand Mean über alle Participants
    total_samples_all = all_phases_df["n_samples"].sum()
    total_time_all = all_phases_df["duration_s"].sum()
    grand_mean_sr = total_samples_all / total_time_all if total_time_all>0 else np.nan

    # Compute additional descriptive statistics across participants
    all_sampling_rates = all_phases_df["mean_sampling_rate_Hz"].values
    grand_median = np.median(all_sampling_rates)
    grand_std = np.std(all_sampling_rates, ddof=1)
    grand_min = np.min(all_sampling_rates)
    grand_max = np.max(all_sampling_rates)
    grand_cv = (grand_std / grand_mean_sr) * 100 if grand_mean_sr > 0 else np.nan

    # Print summary
    print(f"Grand Mean Sampling Rate across all participants: {grand_mean_sr:.2f} Hz")
    print(f"Grand Median: {grand_median:.2f} Hz")
    print(f"Grand Standard Deviation: {grand_std:.2f} Hz")
    print(f"Grand Min: {grand_min:.2f} Hz")
    print(f"Grand Max: {grand_max:.2f} Hz")
    print(f"Grand Coefficient of Variation (CV): {grand_cv:.2f} %")

    return {
        "all_phases_df": all_phases_df,
        "participant_summary_df": participant_summary_df,
        "grand_mean_sampling_rate_Hz": grand_mean_sr
    }

result = compute_sampling_rates(et_folder="C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/ET_Data/ET_Cleaned", output_folder="C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/Sampling_Rate_Analysis/SR_results_ET")


# FUNCTION: generate plots and PDF report
def generate_sampling_rate_report(per_phase_csv, per_participant_csv, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df_phase = pd.read_csv(per_phase_csv)
    df_part = pd.read_csv(per_participant_csv)

    pdf_path = os.path.join(output_folder, "EyeTracking_SamplingRate_Report.pdf")
    pdf = PdfPages(pdf_path)

    palette_main = "lightblue"
    palette_accent = "mediumpurple"

    # Boxplot per participant (with jitter excluding outliers)
    plt.figure(figsize=(10,6))
    sns.boxplot(
        data=df_phase,
        x="participant",
        y="mean_sampling_rate_Hz",
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":palette_accent, "markeredgecolor":"black"},
        boxprops=dict(facecolor=palette_main)
    )

    # Exclude outliers for jitter
    def exclude_outliers(group):
        q1 = group["mean_sampling_rate_Hz"].quantile(0.25)
        q3 = group["mean_sampling_rate_Hz"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group["mean_sampling_rate_Hz"] >= lower) & (group["mean_sampling_rate_Hz"] <= upper)]

    df_no_outliers = df_phase.groupby("participant", group_keys=False).apply(exclude_outliers)

    sns.stripplot(
        data=df_no_outliers,
        x="participant",
        y="mean_sampling_rate_Hz",
        color=palette_accent,
        alpha=0.6,
        jitter=True
    )

    plt.ylabel("Sampling Rate (Hz)", fontsize=label_fontsize)
    plt.xlabel("Participant", fontsize=label_fontsize)
    plt.title("Mean Sampling Rates for Eye-Tracking Data for each Model Building Phase per Participant (Step B)", fontsize=title_fontsize)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Mean Sampling Rate',
               markerfacecolor=palette_accent, markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Outliers',
               markerfacecolor='white', markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Individual Model Building Phase',
               markerfacecolor=palette_accent, markeredgecolor='black', alpha=0.6, markersize=8)
    ]
    plt.legend(handles=legend_handles, loc='lower left', fontsize=tick_fontsize, frameon=True)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.tight_layout()
    pdf.savefig()
    plt.savefig(os.path.join(output_folder, "boxplot_per_participant.png"), dpi=300)
    plt.close()

    # Histogram + KDE overall
    plt.figure(figsize=(8, 5))
    sns.histplot(df_phase["mean_sampling_rate_Hz"], kde=True, bins=20, color=palette_main)

    # Use provided values
    mean_val = 192.55
    std_val = 3.59

    # Add vertical lines
    plt.axvline(mean_val, color=palette_accent, linestyle="--", label=f"Mean: {mean_val:.2f} Hz")
    plt.axvline(mean_val + std_val, color="gray", linestyle=":", label=f"+1 SD: {mean_val + std_val:.2f} Hz")
    plt.axvline(mean_val - std_val, color="gray", linestyle=":", label=f"-1 SD: {mean_val - std_val:.2f} Hz")

    # Add annotation box
    #plt.text(182, plt.ylim()[1] - 10,
           #  f"Mean: {mean_val:.2f} Hz\nSD: {std_val:.2f} Hz",
            # ha='center', va='center', fontsize=10,
             # bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.title("Distribution of Mean Sampling Rates for Eye-Tracking Data Across Model Building Phases (Step B)", fontsize=title_fontsize)
    plt.xlabel("Sampling Rate (Hz)", fontsize=label_fontsize)
    plt.ylabel("Number of Model Building Phases", fontsize=label_fontsize)
    #plt.legend(loc='upper left', bbox_to_anchor=(0, -0.18), fontsize=tick_fontsize, frameon=True)
    plt.legend(fontsize=legend_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    pdf.savefig()
    plt.savefig(os.path.join(output_folder, "histogram_sampling_rate.png"), dpi=300)
    plt.close()

    # Heatmap: participant × phase
    pivot = df_phase.pivot(index="participant", columns="model_name", values="mean_sampling_rate_Hz")
    plt.figure(figsize=(12,6))
    sns.heatmap(
        pivot,
        cmap="Purples",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": tick_fontsize},
        cbar_kws={'label': 'Sampling Rate (Hz)'}
    )
    plt.title("Mean Sampling Rates for Eye-Tracking Data by Participant and Model Phase (Step B)", fontsize=title_fontsize)
    plt.xlabel("Model Building Phase", fontsize=label_fontsize)
    plt.ylabel("Participant", fontsize=label_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.tight_layout()
    pdf.savefig()
    plt.savefig(os.path.join(output_folder, "heatmap_participant_phase.png"), dpi=300)
    plt.close()

    # Barplot with SD and Mean per participant (annotations below plot)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_part,
        x="participant",
        y="mean_sampling_rate_Hz",
        color=palette_main,
        edgecolor="black",
        width=0.6
    )

    # Add annotations below the x-axis
    plt.xticks(rotation=0)
    plt.title("Mean Sampling Rate for Eye-Tracking Data per Participant (Step B)", fontsize=title_fontsize)
    plt.ylabel("Mean Sampling Rate (Hz)", fontsize=label_fontsize)
    plt.xlabel("Participant", fontsize=label_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)

    # Adjust y-limits to make space below bars (extend downward)
    ymin, ymax = plt.ylim()
    pad = 0.12 * (ymax - ymin)
    plt.ylim(ymin - pad, ymax)

    # Annotate below each bar
    # Place annotations slightly above the new lower bound and stagger to reduce overlap
    new_ymin, new_ymax = plt.ylim()
    span = new_ymax - new_ymin
    base_y = new_ymin + 0.02 * span
    #stagger = 0.03 * span
    for idx, row in df_part.iterrows():
        #y_pos = base_y + (stagger if (idx % 2) else 0.0)
        y_pos = base_y
        plt.text(
            idx,
            y_pos,
            f"M={row['mean_sampling_rate_Hz']:.2f} Hz\nSD={row['std_sampling_rate_Hz']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

    # Legend
    #plt.legend(handles=[
     #   Line2D([0], [0], linestyle='none', marker='s', color='w',
      #         label='Mean Sampling Rate (Bar)', markerfacecolor=palette_main, markeredgecolor='black'),
       # Line2D([0], [0], linestyle='none', marker=' ', color='w',
        #       label='Text below: Mean & SD')
    #])
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    pdf.savefig()
    plt.savefig(os.path.join(output_folder, "barplot_mean_sampling_rate.png"), dpi=300)
    plt.close()

    # Text summary page
    grand_mean = df_phase["mean_sampling_rate_Hz"].mean()
    grand_sd = df_phase["mean_sampling_rate_Hz"].std()
    grand_min = df_phase["mean_sampling_rate_Hz"].min()
    grand_max = df_phase["mean_sampling_rate_Hz"].max()
    grand_cv = (grand_sd / grand_mean) * 100

    summary_text = f"""
Eye Tracking Sampling Rate Summary
=================

Participants: {df_part['participant'].nunique()}
Phases total: {len(df_phase)}

Overall descriptive statistics (across all participants and phases):
  Mean Sampling Rate: {grand_mean:.2f} Hz
  Standard Deviation: {grand_sd:.2f} Hz
  Minimum: {grand_min:.2f} Hz
  Maximum: {grand_max:.2f} Hz
  Coefficient of Variation (CV): {grand_cv:.2f} %

Notes:
- Sampling rates are phase-based (N / duration).
- Weighted means used to account for unequal phase durations.
- Low CV indicates stable recording quality.
"""

    fig, ax = plt.subplots(figsize=(8.5,6))
    ax.axis("off")
    ax.text(0, 1, summary_text, ha="left", va="top", fontsize=10, family="monospace")
    pdf.savefig()
    fig.savefig(os.path.join(output_folder, "summary_text_page.png"))
    plt.close()

    pdf.close()
    print(f"PDF report saved: {pdf_path}")

# FOLDER PATHS
output_folder = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/Sampling_Rate_Analysis/SR_results_ET"
os.makedirs(output_folder, exist_ok=True)

per_phase_csv = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/Sampling_Rate_Analysis/SR_results_ET/all_participants_phases.csv"
per_participant_csv = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/Sampling_Rate_Analysis/SR_results_ET/participants_summary.csv"
per_participant_folder = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/Sampling_Rate_Analysis/SR_results_ET/per_participant"  # optional for individual plots

# CALL FUNCTION
generate_sampling_rate_report(per_phase_csv, per_participant_csv, output_folder)
