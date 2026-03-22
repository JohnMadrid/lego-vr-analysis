import glob
import math
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

"""
Processes body‑tracking or eye‑tracking CSV files to evaluate the stability and
uniformity of sampling intervals during model‑building phases. The pipeline
loads and cleans raw data, isolates each phase, computes per‑phase timing
metrics (mean interval, SD, CV%), and identifies outlier phases using a robust
IQR‑based rule. It then aggregates results across participants and generates
diagnostic visualizations (interval grids, CV grids, and interval‑over‑time
plots) across four preprocessing steps (A–D), enabling assessment of data
quality and sampling‑interval uniformity under different filtering conditions.
"""

label_fontsize = 12
title_fontsize = 12
tick_fontsize = 10
text_fontsize = 12
sns.set_theme(context="talk", style="whitegrid")

# 1) Data loading & preprocessing
def load_and_clean_csv(path: str, remove_next: bool = False, data_mode: str = "BT") -> Tuple[pd.DataFrame, str]:
    """
    Load CSV, remove invalid model_name entries ("TM","None"), optionally remove rows
    where RightHand_grabbed_name or LeftHand_grabbed_name == "NextItemButton",
    convert timestamps from milliseconds to seconds, sort by timestamp and return
    cleaned dataframe and participant id (first two characters of filename).

    Parameters:
    path : str
        Path to csv file.
    remove_next : bool, default False
        If True, also remove rows where RightHand_grabbed_name or LeftHand_grabbed_name == "NextItemButton".

    Returns:
    df_clean : pd.DataFrame
        Cleaned dataframe with timestamps in seconds and original columns preserved.
    participant_id : str
        First two characters of the csv filename (basename).
    """

    # CSV laden
    df = pd.read_csv(path)
    df = df.dropna(subset=["model_name", "raw_timestamp"])

    # Entferne ungültige model_name-Werte "TM" oder "None"
    df = df[~df['model_name'].isin(['TM', 'None'])]

    if remove_next:
        # Masken für echte Phase-Transitions
        # NextItemButton gegriffen + Grip gedrückt
        mask_r = (df.get('RightHand_grabbed_name', pd.Series(dtype=object)) == 'NextItemButton') & \
                 (df.get('RightHand_grip_pressed', pd.Series(dtype=bool)) == True)
        mask_l = (df.get('LeftHand_grabbed_name', pd.Series(dtype=object)) == 'NextItemButton') & \
                 (df.get('LeftHand_grip_pressed', pd.Series(dtype=bool)) == True)

        mask = mask_r | mask_l

        # Alle Zeilen, die entfernt werden sollen
        # für jede echte Phase-Transition:
        # - Entferne die Zeile selbst
        # - Entferne alle folgenden Zeilen, bei denen model_name noch der gleiche ist wie der aktuelle
        indices_to_remove = set()
        for idx in df[mask].index:
            try:
                current_model = df.loc[idx, 'model_name']
            except KeyError:
                continue  # skip if idx is not in index

            indices_to_remove.add(idx)  # Zeile der Transition selbst

            # Schleife für alle nachfolgenden Zeilen
            for follow_idx in range(idx + 1, len(df)):
                if follow_idx not in df.index:
                    continue  # skip if follow_idx is not a valid index

                try:
                    next_model = df.loc[follow_idx, 'model_name']
                except KeyError:
                    continue

                if next_model == current_model:
                    indices_to_remove.add(follow_idx)
                else:
                    break  # stoppt, sobald ein neuer model_name erscheint

        # Entferne markierte Zeilen
        df = df.drop(index=indices_to_remove, errors='ignore')

    # Normalisiere Zeitstempel in Sekunden für Plotting und Analysen
    df = df.copy()
    if data_mode.upper() == "BT":
        # Body tracking: raw timestamps in milliseconds → seconds
        df["raw_timestamp"] = pd.to_numeric(df["raw_timestamp"], errors="coerce").astype(float) / 1e3
    elif data_mode.upper() == "ET":
        # Eye tracking: raw timestamps in nanoseconds → seconds
        df["raw_timestamp"] = pd.to_numeric(df["raw_timestamp"], errors="coerce").astype(float) / 1e9
    else:
        # Assume already in seconds
        df["raw_timestamp"] = pd.to_numeric(df["raw_timestamp"], errors="coerce").astype(float)
    df = df.sort_values("raw_timestamp").reset_index(drop=True)

    participant_id = os.path.basename(path)[:2]
    return df, participant_id


# 2) Phase-wise metrics and outlier detection
def compute_phase_metrics(df: pd.DataFrame, participant_id: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute per-phase metrics (mean Δt, std Δt, CV%) for given dataframe grouped by model_name,
    preserve the occurrence order of phases as they appear in the file.

    Returns:
      - phase_df: DataFrame with one row per phase (model_name) with metrics
      - summary: dict with participant id, overall CV (mean of phase CVs), descriptive stats,
                 and list of outlier phases

    Outlier identification (comment):
      - Outliers are identified on the distribution of per-phase CV (%) for this participant.
      - We use the robust IQR rule: phases with CV > Q3 + 2*IQR or CV < Q1 - 2*IQR are flagged.
        This is stricter than the usual 1.5*IQR to reduce false positives if small numbers of phases.
    """
    rows = []
    # preserve first-occurrence order of model names
    seen = []
    for _, r in df.iterrows():
        mn = r["model_name"]
        if mn not in seen:
            seen.append(mn)

    for model_name in seen:
        phase = df[df["model_name"] == model_name].sort_values("raw_timestamp")
        ts = phase["raw_timestamp"].values
        if len(ts) < 2:
            continue
        delta_t = np.diff(ts)
        mean_dt = float(np.mean(delta_t))
        std_dt = float(np.std(delta_t, ddof=0))
        cv = float((std_dt / mean_dt) * 100) if mean_dt != 0 else np.nan
        rows.append({
            "participant": participant_id,
            "model_name": model_name,
            "n_points": len(ts),
            "mean_dt_s": mean_dt,
            "std_dt_s": std_dt,
            "cv_pct": cv
        })

    phase_df = pd.DataFrame(rows)
    if phase_df.empty:
        summary = {
            "participant": participant_id,
            "overall_cv_mean": np.nan,
            "overall_cv_median": np.nan,
            "n_phases": 0,
            "outlier_phases": []
        }
        return phase_df, summary

    # overall CV as mean of per-phase CVs
    overall_cv_mean = float(phase_df["cv_pct"].mean())
    overall_cv_median = float(phase_df["cv_pct"].median())
    n_phases = len(phase_df)

    # identify outliers using IQR-based rule (robust)
    q1 = phase_df["cv_pct"].quantile(0.25)
    q3 = phase_df["cv_pct"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr
    # phases flagged as outliers if cv_pct outside [lower_bound, upper_bound]
    outliers = phase_df[(phase_df["cv_pct"] < lower_bound) | (phase_df["cv_pct"] > upper_bound)].copy()
    outlier_list = outliers["model_name"].tolist()

    summary = {
        "participant": participant_id,
        "overall_cv_mean": overall_cv_mean,
        "overall_cv_median": overall_cv_median,
        "n_phases": n_phases,
        "q1_cv": float(q1),
        "q3_cv": float(q3),
        "iqr_cv": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_phases": outlier_list,
        "phase_df": phase_df
    }
    return phase_df, summary

# 3) Combine metrics across files and report
def analyze_multiple_files(csv_paths: List[str],
                           remove_next: bool = False,
                           outlier_removal: bool = False,
                           outlier_map: Dict[str, List[str]] = None,
                           cv_uniformity_threshold_pct: float = 50.0,
                           data_mode: str = "BT"):
    """
    Für eine Liste von CSV-Dateien:
      - Lade und bereinige jede Datei (TM/None entfernen, optional NextItemButton)
      - Berechne Phasen-Metriken
      - Optional entferne Phasen, die in outlier_map für diesen Teilnehmer stehen
      - Drucke Teilnehmer- und Gesamtzusammenfassungen
      - Rückgabe aggregierter DataFrame + Teilnehmer-Summaries

    Parameters:
    outlier_map : dict
        Optional: dict {participant_id: [model_names]} der zu entfernenden Outlier-Phasen
        z.B. aus STEP A für STEP B.
    """
    all_phase_rows = []
    participant_summaries = {}
    outlier_map = outlier_map or {}

    for path in csv_paths:
        df_clean, pid = load_and_clean_csv(path, remove_next=remove_next, data_mode=data_mode)
        phase_df, summary = compute_phase_metrics(df_clean, pid)

        # store raw phase rows
        if not phase_df.empty:
            phase_df["file"] = os.path.basename(path)
            all_phase_rows.append(phase_df)

        # participant-level stats
        filtered_phase_df = phase_df.copy()
        removed_outliers = []
        if outlier_removal and pid in outlier_map:
            remove_phases = outlier_map[pid]
            removed_outliers = remove_phases
            filtered_phase_df = filtered_phase_df[~filtered_phase_df["model_name"].isin(remove_phases)]

        mean_dt = float(filtered_phase_df["mean_dt_s"].mean()) if not filtered_phase_df.empty else np.nan
        std_dt = float(filtered_phase_df["mean_dt_s"].std(ddof=0)) if not filtered_phase_df.empty else np.nan
        mean_cv = float(filtered_phase_df["cv_pct"].mean()) if not filtered_phase_df.empty else np.nan
        std_cv = float(filtered_phase_df["cv_pct"].std(ddof=0)) if not filtered_phase_df.empty else np.nan
        uniform = "UNIFORM" if (not np.isnan(mean_cv) and mean_cv < cv_uniformity_threshold_pct) else "NOT UNIFORM"

        participant_summaries[pid] = {
            "participant": pid,
            "n_phases_total": summary["n_phases"],
            "n_outliers": len(summary.get("outlier_phases", [])),
            "outlier_phases": summary.get("outlier_phases", []),
            "removed_outliers": removed_outliers,
            "mean_dt_s": mean_dt,
            "std_dt_s": std_dt,
            "mean_cv_pct": mean_cv,
            "std_cv_pct": std_cv,
            "uniformity": uniform
        }

        print(f"Participant {pid} | phases: {summary['n_phases']} | outliers: {len(summary.get('outlier_phases', []))} "
              f"| mean_dt_s: {mean_dt:.4f} | std_dt_s: {std_dt:.4f} | mean_CV%: {mean_cv:.2f} | {uniform}")

    # aggregate across all participants
    if all_phase_rows:
        phase_df_all = pd.concat(all_phase_rows, ignore_index=True)
    else:
        phase_df_all = pd.DataFrame(columns=["participant", "model_name", "n_points", "mean_dt_s", "std_dt_s", "cv_pct", "file"])

    # overall CV across participants
    overall_cv_values = [ps["mean_cv_pct"] for ps in participant_summaries.values() if not np.isnan(ps["mean_cv_pct"])]
    overall_cv_across = float(np.mean(overall_cv_values)) if overall_cv_values else np.nan
    overall_uniform = "UNIFORM" if (not np.isnan(overall_cv_across) and overall_cv_across < cv_uniformity_threshold_pct) else "NOT UNIFORM"
    print(f"Overall CV across participants: {overall_cv_across:.2f} -> {overall_uniform}")

    return phase_df_all, participant_summaries



# 4) Calculate overall CV over all CSV files (helper)
def compute_overall_cv_from_phase_df(phase_df_all: pd.DataFrame) -> float:
    """
    Compute the overall CV as the mean of per-phase CVs across all participants/files.
    """
    if phase_df_all.empty:
        return np.nan
    return float(phase_df_all["cv_pct"].mean())

# 5) Plotting functions
def plot_mean_interval_grid(phase_df_all: pd.DataFrame,
                            title: str = "Mean Sampling Interval per Phase (ms)",
                            excluded_map: Optional[Dict[str, List[str]]] = None,
                            save_path: Optional[str] = None):
    if phase_df_all.empty:
        print("No phase data to plot.")
        return

    participants = sorted(phase_df_all["participant"].unique())
    n_participants = len(participants)

    n_cols = 5  # maximal 10 plots pro Figur: 2 Reihen
    n_rows = math.ceil(n_participants / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, pid in enumerate(participants):
        ax = axes[i]
        p_data = phase_df_all[phase_df_all["participant"] == pid].copy()
        # Preserve temporal order of phases
        p_data["model_name"] = pd.Categorical(p_data["model_name"], categories=p_data["model_name"], ordered=True)
        # Prepare milliseconds for plotting
        p_data["mean_dt_ms"] = p_data["mean_dt_s"] * 1000.0
        # Mask excluded phases by setting values to NaN so bars are hidden but labels stay
        if excluded_map and pid in excluded_map and len(excluded_map[pid]) > 0:
            p_data.loc[p_data["model_name"].isin(excluded_map[pid]), "mean_dt_ms"] = np.nan
        sns.barplot(data=p_data, x="model_name", y="mean_dt_ms", ax=ax, color="skyblue", errorbar=None)
        ax.set_xlabel("Model Phase", fontsize=label_fontsize)
        ax.set_ylabel("Mean Sampling Interval (ms)", fontsize=label_fontsize)
        ax.set_title(f"Participant {pid}", fontsize=title_fontsize)
        ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=90)
        ax.tick_params(axis='y', labelsize=tick_fontsize)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_cv_grid(phase_df_all: pd.DataFrame,
                 title: str = "CV (%) per Phase",
                 excluded_map: Optional[Dict[str, List[str]]] = None,
                 save_path: Optional[str] = None):
    if phase_df_all.empty:
        print("No phase data to plot.")
        return

    participants = sorted(phase_df_all["participant"].unique())
    n_participants = len(participants)

    n_cols = 5
    n_rows = math.ceil(n_participants / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, pid in enumerate(participants):
        ax = axes[i]
        p_data = phase_df_all[phase_df_all["participant"] == pid].copy()
        p_data["model_name"] = pd.Categorical(p_data["model_name"], categories=p_data["model_name"], ordered=True)
        if excluded_map and pid in excluded_map and len(excluded_map[pid]) > 0:
            p_data.loc[p_data["model_name"].isin(excluded_map[pid]), "cv_pct"] = np.nan
        sns.barplot(data=p_data, x="model_name", y="cv_pct", ax=ax, color="mediumpurple", errorbar=None)
        ax.set_title(f"Participant {pid}", fontsize=title_fontsize)
        ax.set_xlabel("Model Phase", fontsize=label_fontsize)
        ax.set_ylabel("CV (%)", fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=90)
        ax.tick_params(axis='y', labelsize=tick_fontsize)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


# 6) Plot intervals over time for a given participant & phase
def plot_intervals_over_time_for_phase(df: pd.DataFrame, participant_id: str, model_name: str, save_path: Optional[str] = None):
    """
    Plot sampling intervals over time (using timestamp index) for a specific participant and phase.
    Does NOT compute or plot sampling rate.

    The x-axis is the timestamp of the first sample of each interval (i.e., timestamps[:-1]).
    """
    sub = df[df["model_name"] == model_name].sort_values("raw_timestamp")
    if sub.empty:
        print(f"No data for participant {participant_id} phase {model_name}")
        return
    # raw_timestamp already normalized to seconds in load_and_clean_csv
    ts = sub["raw_timestamp"].values
    if len(ts) < 2:
        print(f"Not enough timestamps for participant {participant_id} phase {model_name}")
        return

    delta_t = np.diff(ts)  # interval durations in seconds
    delta_t_ms = delta_t * 1000.0  # convert to milliseconds for plotting
    phase_time = ts[:-1] - ts[0]  # phase-relative time in seconds

    plt.figure(figsize=(10, 4))
    width = 0.02 * max(delta_t) if len(delta_t) else 0.01

    # Determine y-axis cap to reduce impact of extreme outliers
    # Use 100th percentile as cap; if all equal, fallback to max
    if len(delta_t_ms) > 0:
        y_cap = np.percentile(delta_t_ms, 99.7)
        if y_cap <= 0 or np.isclose(y_cap, 0):
            y_cap = np.max(delta_t_ms)
    else:
        y_cap = 1.0

    # Values to plot (clipped)
    clipped_values = np.minimum(delta_t_ms, y_cap)
    bars = plt.bar(
        phase_time,
        clipped_values,
        width=width,
        alpha=0.85,
        color="lightblue",
        edgecolor="skyblue",
        linewidth=1.2
    )

    # Annotate clipped bars with true value above the cap line
    for x, val_ms, bar in zip(phase_time, delta_t_ms, bars):
        if val_ms > y_cap:
            plt.text(
                x,
                y_cap * 0.98,
                f"{val_ms:.0f} ms",
                ha="center",
                va="top",
                fontsize=tick_fontsize,
                rotation=90,
                color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0)
            )

    plt.ylim(0, y_cap * 1.05 if y_cap > 0 else None)
    # Optional helper line indicating clipping level (only if actual clipping occurred)
    if np.any(delta_t_ms > y_cap):
        plt.axhline(y_cap, color="red", linestyle=":", linewidth=1.0, label=f"Clip level ≈ P99 ({y_cap:.0f} ms)")
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.2), fontsize=tick_fontsize, frameon=True)
    plt.xlabel("Phase Time (s)", fontsize=label_fontsize)
    plt.ylabel("Interval Duration (ms)", fontsize=label_fontsize)
    plt.title(f"Participant {participant_id} | Phase {model_name} | Intervals over Phase Time", fontsize=title_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


# Main analysis pipeline performing the requested sequence
def full_pipeline(folder_path: str,
                  cv_uniformity_threshold_pct: float = 50.0,
                  data_mode: str = "BT",
                  output_dir: Optional[str] = None):
    """
    Executes the full requested sequence and prints progression:
      A) Remove TM/None only -> compute metrics (with outliers present)
      B) Remove outliers -> recompute participant summaries (no outliers)
      C) Remove TM/None + NextItemButton -> compute metrics (with outliers)
      D) Remove outliers (after removing NextItemButton) -> recompute summaries (no outliers)
    For each step prints participant-level and overall uniformity statements and plots the two grids
    (mean interval per phase per participant) and CV grid.

    For each outlier phase detected, a plot of intervals over time for that participant+phase is generated.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("No CSV files found in folder.")
        return

    results = {}
    mode = data_mode.upper()
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), f"uniformity_{'ET' if mode=='ET' else 'BT'}")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- STEP A ----------------
    print("\n=== STEP A: Remove TM/None (outliers present) ===")
    phase_df_A, participants_A = analyze_multiple_files(
        csv_files,
        remove_next=False,
        outlier_removal=False,
        cv_uniformity_threshold_pct=cv_uniformity_threshold_pct,
        data_mode=mode
    )
    plot_mean_interval_grid(
        phase_df_A,
        title="STEP A: Mean Sampling Interval per Phase (ms) - TM/None removed",
        excluded_map=None,
        save_path=os.path.join(output_dir, "step_A", "bar_mean_interval.png")
    )
    plot_cv_grid(
        phase_df_A,
        title="STEP A: CV (%) per Phase - TM/None removed",
        excluded_map=None,
        save_path=os.path.join(output_dir, "step_A", "bar_cv.png")
    )
    # ---------------- STEP B ----------------
    # remove outliers detected in STEP A
    outlier_map_A = {pid: ps.get("outlier_phases", []) for pid, ps in participants_A.items()}
    print("\n=== STEP B: Remove outliers detected in STEP A ===")
    phase_df_B, participants_B = analyze_multiple_files(
        csv_files,
        remove_next=False,
        outlier_removal=True,
        outlier_map=outlier_map_A,  # ← this is the fix
        cv_uniformity_threshold_pct=cv_uniformity_threshold_pct,
        data_mode=mode
    )

    # plot interval-over-time for each outlier from Step A
    for path in csv_files:
        df_clean, pid = load_and_clean_csv(path, remove_next=False, data_mode=mode)
        for mn in outlier_map_A.get(pid, []):
            plot_intervals_over_time_for_phase(
                df_clean,
                pid,
                mn,
                save_path=os.path.join(output_dir, "step_B", f"intervals_{pid}_{mn}.png")
            )
    plot_mean_interval_grid(
        phase_df_B,
        title="STEP B: Mean Sampling Interval per Phase (ms) - TM/None + outliers removed",
        excluded_map=outlier_map_A,
        save_path=os.path.join(output_dir, "step_B", "bar_mean_interval.png")
    )
    plot_cv_grid(
        phase_df_B,
        title="STEP B: CV (%) per Phase - TM/None + outliers removed",
        excluded_map=outlier_map_A,
        save_path=os.path.join(output_dir, "step_B", "bar_cv.png")
    )

    # ---------------- STEP C ----------------
    if mode != "ET":
        print("\n=== STEP C: Remove TM/None + NextItemButton (outliers present) ===")
        phase_df_C, participants_C = analyze_multiple_files(
            csv_files,
            remove_next=True,
            outlier_removal=False,
            cv_uniformity_threshold_pct=cv_uniformity_threshold_pct,
            data_mode=mode
        )
        plot_mean_interval_grid(
            phase_df_C,
            title="STEP C: Mean Sampling Interval per Phase (ms) - TM/None & NextItemButton removed",
            excluded_map=None,
            save_path=os.path.join(output_dir, "step_C", "bar_mean_interval.png")
        )
        plot_cv_grid(
            phase_df_C,
            title="STEP C: CV (%) per Phase - TM/None & NextItemButton removed",
            excluded_map=None,
            save_path=os.path.join(output_dir, "step_C", "bar_cv.png")
        )

    # ---------------- STEP D ----------------
    if mode != "ET":
        outlier_map_C = {pid: ps.get("outlier_phases", []) for pid, ps in participants_C.items()}
        print("\n=== STEP D: Remove outliers detected in STEP C ===")
        phase_df_D, participants_D = analyze_multiple_files(
            csv_files,
            remove_next=True,
            outlier_removal=True,
            outlier_map=outlier_map_C,  # ← this is the fix
            cv_uniformity_threshold_pct=cv_uniformity_threshold_pct,
            data_mode=mode
        )

        # plot interval-over-time for each outlier from Step C
        for path in csv_files:
            df_clean, pid = load_and_clean_csv(path, remove_next=True, data_mode=mode)
            for mn in outlier_map_C.get(pid, []):
                plot_intervals_over_time_for_phase(
                    df_clean,
                    pid,
                    mn,
                    save_path=os.path.join(output_dir, "step_D", f"intervals_{pid}_{mn}.png")
                )
        plot_mean_interval_grid(
            phase_df_D,
            title="STEP D: Mean Sampling Interval per Phase (ms) - TM/None & NextItemButton + outliers removed",
            excluded_map=outlier_map_C,
            save_path=os.path.join(output_dir, "step_D", "bar_mean_interval.png")
        )
        plot_cv_grid(
            phase_df_D,
            title="STEP D: CV (%) per Phase - TM/None & NextItemButton + outliers removed",
            excluded_map=outlier_map_C,
            save_path=os.path.join(output_dir, "step_D", "bar_cv.png")
        )

    if mode == "ET":
        results = {
            "step_A": {"phase_df": phase_df_A, "participants": participants_A},
            "step_B": {"phase_df": phase_df_B, "participants": participants_B},
        }
    else:
        results = {
            "step_A": {"phase_df": phase_df_A, "participants": participants_A},
            "step_B": {"phase_df": phase_df_B, "participants": participants_B},
            "step_C": {"phase_df": phase_df_C, "participants": participants_C},
            "step_D": {"phase_df": phase_df_D, "participants": participants_D},
        }
    return results

def create_summary_table(results: dict) -> pd.DataFrame:
    """
    Create a summary table across Steps A–D containing:
      - Mean interval, SD interval
      - Mean CV (%), SD CV (%)
      - Uniformity status
      - Outliers removed: count + a list (participant, model_name)
    """

    def get_stats_from_participants(participants: dict):
        mean_intervals = [ps.get("mean_dt_s", np.nan) for ps in participants.values()]
        mean_cvs = [ps.get("mean_cv_pct", np.nan) for ps in participants.values()]

        mean_interval = np.nanmean(mean_intervals)
        sd_interval = np.nanstd(mean_intervals)
        mean_cv = np.nanmean(mean_cvs)
        sd_cv = np.nanstd(mean_cvs)
        uniform = "UNIFORM" if (not np.isnan(mean_cv) and mean_cv < 50.0) else "NOT UNIFORM"
        return mean_interval, sd_interval, mean_cv, sd_cv, uniform

    def collect_outliers(participants: dict):
        outlier_tuples = []
        for pid, info in participants.items():
            for mn in info.get("outlier_phases", []):
                outlier_tuples.append((pid, mn))
        n_outliers = len(outlier_tuples)
        names = ", ".join([f"({p}, {m})" for p, m in outlier_tuples]) if n_outliers > 0 else "-"
        return f"{n_outliers} removed: {names}"

    # Build columns dynamically based on available steps
    columns = {"Metric": ["Mean interval (s)", "SD interval (s)", "Mean CV (%)", "SD CV (%)", "Uniformity", "Outliers removed"]}

    step_labels = [
        ("step_A", "A: TM/None removed\n(outliers not removed)"),
        ("step_B", "B: TM/None removed\n(outliers removed)"),
        ("step_C", "C: TM/None+NextItemButton removed\n(outliers not removed)"),
        ("step_D", "D: TM/None+NextItemButton removed\n(outliers removed)")
    ]

    for key, label in step_labels:
        if key in results:
            participants = results[key]["participants"]
            stats = get_stats_from_participants(participants)
            outliers = collect_outliers(participants)
            columns[label] = [*stats, outliers]

    summary_table = pd.DataFrame(columns)

    print("\n=== Descriptive statistics for mean sampling intervals and mean CV in percent ===")
    print(summary_table.to_string(index=False))
    return summary_table

def plot_step_summary(step_name: str, participants: dict):
    """
    Creates a 2-panel bar plot for one step (A-D):
      - Left: mean sampling intervals (s) per participant
      - Right: mean CV (%) per participant
    Includes grand mean line and descriptive stats in legend.
    """
    pids = sorted(participants.keys())
    mean_intervals = [participants[pid].get('mean_dt_s', np.nan) for pid in pids]
    mean_cvs = [participants[pid].get('mean_cv_pct', np.nan) for pid in pids]
    outlier_counts = [participants[pid].get('n_outliers', 0) for pid in pids]

    if np.all(np.isnan(mean_intervals)) or np.all(np.isnan(mean_cvs)):
        print(f"No valid data to plot for {step_name}")
        return

    total_outliers = sum(outlier_counts)
    grand_mean_interval = np.nanmean(mean_intervals)
    grand_mean_cv = np.nanmean(mean_cvs)

    # Prepare multi-line stats strings
    interval_stats_lines = [
        f"Mean: {grand_mean_interval*1000.0:.2f} ms",
        f"SD: {np.nanstd(mean_intervals)*1000.0:.2f} ms",
        f"Median: {np.nanmedian(mean_intervals)*1000.0:.2f} ms",
        f"Outliers removed: {total_outliers}"
    ]
    cv_stats_lines = [
        f"Mean: {grand_mean_cv:.2f}%",
        f"SD: {np.nanstd(mean_cvs):.2f}%",
        f"Median: {np.nanmedian(mean_cvs):.2f}%",
        f"Outliers removed: {total_outliers}"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean sampling intervals (plot in ms)
    mean_intervals_ms = [x * 1000.0 if not np.isnan(x) else np.nan for x in mean_intervals]
    axes[0].bar(pids, mean_intervals_ms, color='skyblue')
    axes[0].axhline(grand_mean_interval*1000.0, color='red', linestyle='--', label=f"Grand mean: {grand_mean_interval*1000.0:.2f} ms")
    axes[0].set_xlabel("Participant", fontsize=label_fontsize)
    axes[0].set_ylabel("Mean interval (ms)", fontsize=label_fontsize)
    axes[0].set_title(f"{step_name} - Mean Sampling Interval", fontsize=title_fontsize)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -0.18), fontsize=tick_fontsize, frameon=True)
    axes[0].tick_params(axis='both', labelsize=tick_fontsize)
    axes[0].text(0.02, 0.02, "\n".join(interval_stats_lines), transform=axes[0].transAxes,
                 fontsize=text_fontsize, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.7))

    # Right: mean CV (%)
    axes[1].bar(pids, mean_cvs, color='mediumpurple')
    axes[1].axhline(grand_mean_cv, color='red', linestyle='--', label=f"Grand mean: {grand_mean_cv:.2f}%")
    axes[1].set_xlabel("Participant", fontsize=label_fontsize)
    axes[1].set_ylabel("Mean CV (%)", fontsize=label_fontsize)
    axes[1].set_title(f"{step_name} - Mean CV", fontsize=title_fontsize)
    axes[1].legend(loc='upper left', bbox_to_anchor=(0, -0.18), fontsize=tick_fontsize, frameon=True)
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[1].text(0.02, 0.02, "\n".join(cv_stats_lines), transform=axes[1].transAxes,
                 fontsize=text_fontsize, verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.7))

    # Leave space at the bottom for legends placed outside
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

def export_phase_summary_csv_current_dir(phase_dfs: dict, participant_summaries: dict,
                                         filename: str = "phase_summary.csv"):
    """
    Exports a CSV with per-phase, per-participant data for all steps (A-D)
    into the current working directory.

    Parameters:
    phase_dfs : dict
        Dict with keys 'A','B','C','D' containing phase DataFrames per step.
        Each DataFrame must have columns: participant, model_name, mean_dt_s, cv_pct
    participant_summaries : dict
        Nested dict with keys 'A','B','C','D' containing participant summary dicts
        as returned from analyze_multiple_files.
    filename : str
        Name of the CSV file to save in the current working directory.
    """
    all_rows = []

    for step in ['A', 'B', 'C', 'D']:
        phase_df = phase_dfs.get(step)
        summaries = participant_summaries.get(step)
        if phase_df is None or summaries is None:
            continue

        for idx, row in phase_df.iterrows():
            pid = row['participant']
            is_outlier = False
            n_outliers = 0
            if pid in summaries:
                outlier_phases = summaries[pid].get('outlier_phases', [])
                is_outlier = row['model_name'] in outlier_phases
                n_outliers = summaries[pid].get('n_outliers', 0)

            all_rows.append({
                'step': step,
                'participant': pid,
                'model_name': row['model_name'],
                'mean_dt_s': row['mean_dt_s'],
                'cv_pct': row['cv_pct'],
                'is_outlier': is_outlier,
                'n_outliers': n_outliers
            })

    df_export = pd.DataFrame(all_rows)
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, filename)
    df_export.to_csv(output_path, index=False)
    print(f"CSV exported in current directory: {output_path}")
    return df_export

def export_phase_timestamps_csv(folder_path: str, results: dict, filename: str = "phase_timestamps.csv"):
    """
    Export for each participant and each model phase (Steps A–D) the start and end raw_timestamp.
    Uses the cleaned data of each step (A–D) as defined in full_pipeline.
    """
    all_rows = []

    for step_key, step_data in results.items():
        step_label = step_key.split("_")[-1].upper()  # e.g. "step_A" -> "A"
        remove_next = step_label in ["C", "D"]
        outlier_map = {pid: ps.get("outlier_phases", []) for pid, ps in step_data["participants"].items()} \
                      if step_label in ["B", "D"] else {}

        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        for path in csv_files:
            df_clean, pid = load_and_clean_csv(path, remove_next=remove_next)

            # optionally remove outlier phases for steps B and D
            if pid in outlier_map:
                df_clean = df_clean[~df_clean["model_name"].isin(outlier_map[pid])]

            for model_name, group in df_clean.groupby("model_name"):
                start_ts = group["raw_timestamp"].min()
                end_ts = group["raw_timestamp"].max()
                all_rows.append({
                    "step": step_label,
                    "participant": pid,
                    "model_name": model_name,
                    "start_raw_timestamp": int(start_ts * 1e3),
                    "end_raw_timestamp": int(end_ts * 1e3)
                })

    df_timestamps = pd.DataFrame(all_rows)
    output_path = os.path.join(os.getcwd(), filename)
    df_timestamps.to_csv(output_path, index=False)
    print(f"Timestamps CSV exported: {output_path}")
    return df_timestamps

def clean_eyetracking_data(et_folder: str, timestamp_csv_path: str, output_folder: str = None):
    timestamps_df = pd.read_csv(timestamp_csv_path, dtype={"participant": str, "model_name": str})
    step_c_df = timestamps_df[timestamps_df["step"] == "C"].copy() # take step c to have all phases including outliers for ET analysis but without NextItemButton

    if output_folder is None:
        output_folder = os.path.join(et_folder, "ET_Cleaned")
    os.makedirs(output_folder, exist_ok=True)

    for csv_file in os.listdir(et_folder):
        if not csv_file.endswith(".csv"):
            continue

        pid = os.path.basename(csv_file)[:2] # first 2 chars indicate participant
        pid = pid.zfill(2)
        et_path = os.path.join(et_folder, csv_file)
        try:
            df = pd.read_csv(et_path)
        except Exception:
            continue

        # remove TM and None
        df = df[~df["model_name"].isin(["TM", "None"])].copy()
        if df.empty:
            continue

        # model order as observed in ET data (temporal order in file)
        model_order = df["model_name"].drop_duplicates().tolist()

        # subset Step c timestamps for this participant
        pid_steps = step_c_df[step_c_df["participant"] == pid]
        if pid_steps.empty:
            continue

        collected = []
        for model_name in model_order:
            # find the timestamp row(s) for this participant & model_name (Step D)
            match = pid_steps[pid_steps["model_name"] == model_name]
            if match.empty:
                # no timestamp info for this model -> skip
                continue

            # sort matches by start timestamp
            match = match.sort_values(by=["start_raw_timestamp"])

            # take last start and first end
            start_ts = int(match["start_raw_timestamp"].max()) # last start
            end_ts = int(match["end_raw_timestamp"].min()) # first end

            # ensure raw_timestamp in ET data is numeric
            df["raw_timestamp"] = pd.to_numeric(df["raw_timestamp"], errors="coerce").astype(int)

            # select rows strictly between start and end
            model_rows = df[
                (df["model_name"] == model_name) &
                (df["raw_timestamp"] > start_ts) &
                (df["raw_timestamp"] < end_ts)
                ].copy()

            if not model_rows.empty:
                collected.append(model_rows)

        if not collected:
            continue

        df_clean = pd.concat(collected, ignore_index=True)

        # safe rename if columns exist
        rename_map = {}
        if "raw_timestamp" in df_clean.columns:
            rename_map["raw_timestamp"] = "raw_timestamp_unity"
        if "gaze_capture_time" in df_clean.columns:
            rename_map["gaze_capture_time"] = "raw_timestamp"
        if rename_map:
            df_clean = df_clean.rename(columns=rename_map)

        output_path = os.path.join(output_folder, f"{pid}_ET_Data_cleaned.csv")
        df_clean.to_csv(output_path, index=False)

    return True

def generate_sampling_rate_visualization_stepD(phase_df_D, participant_summary_D, output_folder):
    """
    Generates PDF report and plots for STEP D sampling rates:
      - Automatically removes outlier phases based on participant_summary_D
      - Computes sampling rate as 1 / mean_dt_s
      - Uses mean_dt_s and std from STEP D with the cleaned values
      - Creates per-participant boxplots + overall histogram with mean and ±1 SD
      - Jitter plot excludes boxplot outliers
    """

    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, "SamplingRate_Report_STEP_D.pdf")
    pdf = PdfPages(pdf_path)

    palette_main = "lightblue"
    palette_accent = "mediumpurple"

    # Remove STEP D outlier phases based on participant_summary_D
    outlier_map_D = {pid: ps.get("outlier_phases", []) for pid, ps in participant_summary_D.items()}
    phase_df_D_clean = phase_df_D.copy()
    mask_keep = phase_df_D_clean.apply(
        lambda row: row['model_name'] not in outlier_map_D.get(row['participant'], []),
        axis=1
    )
    phase_df_D_clean = phase_df_D_clean[mask_keep].copy()

    # Compute sampling rate
    phase_df_D_clean["mean_sampling_rate_Hz"] = 1 / phase_df_D_clean["mean_dt_s"]

    # Remove outliers per participant for jitter plot (1.5×IQR rule)
    def remove_boxplot_outliers(group):
        q1 = group["mean_sampling_rate_Hz"].quantile(0.25)
        q3 = group["mean_sampling_rate_Hz"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group["mean_sampling_rate_Hz"] >= lower) & (group["mean_sampling_rate_Hz"] <= upper)]

    df_no_outliers = phase_df_D_clean.groupby("participant", group_keys=False).apply(remove_boxplot_outliers)

    # Boxplot per participant
    plt.figure(figsize=(10,6))
    sns.boxplot(
        data=phase_df_D_clean,
        x="participant",
        y="mean_sampling_rate_Hz",
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":palette_accent, "markeredgecolor":"black"},
        boxprops=dict(facecolor=palette_main)
    )

    sns.stripplot(
        data=df_no_outliers,
        x="participant",
        y="mean_sampling_rate_Hz",
        color=palette_accent,
        alpha=0.6,
        jitter=True
    )

    plt.title("Mean Sampling Rates for Body-Tracking Data for each Model Building Phase per Participant (Step D)", fontsize=title_fontsize)
    plt.xlabel("Participant", fontsize=label_fontsize)
    plt.ylabel("Sampling Rate (Hz)", fontsize=label_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
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
    plt.savefig(os.path.join(output_folder, "boxplot_per_participant_STEP_D.png"), dpi=300)
    plt.close()

    # Histogram + KDE overall
    legend_fontsize = 10

    plt.figure(figsize=(8, 5))
    sns.histplot(phase_df_D_clean["mean_sampling_rate_Hz"], kde=True, bins=20, color=palette_main)

    # Mean and SD from cleaned STEP D
    mean_dt = phase_df_D_clean["mean_dt_s"].mean()
    mean_val = 1 / mean_dt
    std_val = phase_df_D_clean["mean_sampling_rate_Hz"].std()

    plt.axvline(mean_val, color=palette_accent, linestyle="--", label=f"Mean: {mean_val:.2f} Hz")
    plt.axvline(mean_val + std_val, color="gray", linestyle=":", label=f"+1 SD: {mean_val + std_val:.2f} Hz")
    plt.axvline(mean_val - std_val, color="gray", linestyle=":", label=f"-1 SD: {mean_val - std_val:.2f} Hz")

    # 🔹 Adjusted title and y-label
    plt.title("Distribution of Mean Sampling Rates for Body-Tracking Data Across Model Building Phases (Step D)", fontsize=title_fontsize)
    plt.xlabel("Sampling Rate (Hz)", fontsize=label_fontsize)
    plt.ylabel("Number of Model Building Phases", fontsize=label_fontsize)   # clearer than "Frequency"
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    pdf.savefig()
    plt.savefig(os.path.join(output_folder, "histogram_sampling_rate_STEP_D.png"), dpi=300)
    plt.close()

    pdf.close()

    print(f"Sampling rate PDF report saved: {pdf_path}")
    print(f"Histogram datapoints: {len(phase_df_D_clean)} phases included")
    return phase_df_D_clean

# Main:
if __name__ == "__main__":
    folder_path = r"C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/BT_Data"
    results = full_pipeline(folder_path, cv_uniformity_threshold_pct=50.0, data_mode="BT")
    summary = create_summary_table(results)
    plot_step_summary("STEP A", results['step_A']['participants'])
    plot_step_summary("STEP B", results['step_B']['participants'])
    if 'step_C' in results:
        plot_step_summary("STEP C", results['step_C']['participants'])
    if 'step_D' in results:
        plot_step_summary("STEP D", results['step_D']['participants'])

    phase_dfs = {
        'A': results['step_A']['phase_df'],
        'B': results['step_B']['phase_df'],
        'C': results['step_C']['phase_df'],
        'D': results['step_D']['phase_df']
    }

    participant_summaries = {
        'A': results['step_A']['participants'],
        'B': results['step_B']['participants'],
        'C': results['step_C']['participants'],
        'D': results['step_D']['participants']
    }

    df_exported = export_phase_summary_csv_current_dir(phase_dfs, participant_summaries)
    # example how step b would be extracted
    #df_B = df_exported[df_exported['step'] == 'B']
    #df_B = df_B[~df_B['is_outlier']]
    #grand_mean_cv_B = df_B['cv_pct'].mean()
    #grand_mean_interval_B = df_B['mean_dt_s'].mean()

    df_timestamps = export_phase_timestamps_csv(folder_path, results)

    # Now clean ET data and repeat process of BT with ET
    et_folder = r"C:\Users\johan\Desktop\data_pilot_Bricks_VR_2025\ET_Data"
    timestamp_csv = r"C:\Users\johan\Desktop\data_pilot_Bricks_VR_2025\Sampling_Rate_Analysis\phase_timestamps.csv"
    clean_eyetracking_data(et_folder, timestamp_csv)

    results_ET = full_pipeline(os.path.join(et_folder, "ET_Cleaned"), data_mode="ET")
    summary_ET = create_summary_table(results_ET)

    if isinstance(results_ET, dict) and 'step_A' in results_ET:
        plot_step_summary("STEP A", results_ET['step_A']['participants'])
    if isinstance(results_ET, dict) and 'step_B' in results_ET:
        plot_step_summary("STEP B", results_ET['step_B']['participants'])
    if isinstance(results_ET, dict) and 'step_C' in results_ET:
        plot_step_summary("STEP C", results_ET['step_C']['participants'])
    if isinstance(results_ET, dict) and 'step_D' in results_ET:
        plot_step_summary("STEP D", results_ET['step_D']['participants'])

    phase_dfs_ET = {
        'A': results_ET['step_A']['phase_df'],
        'B': results_ET['step_B']['phase_df']
    }
    if isinstance(results_ET, dict) and 'step_C' in results_ET:
        phase_dfs_ET['C'] = results_ET['step_C']['phase_df']
    if isinstance(results_ET, dict) and 'step_D' in results_ET:
        phase_dfs_ET['D'] = results_ET['step_D']['phase_df']

    participant_summaries_ET = {
        'A': results_ET['step_A']['participants'],
        'B': results_ET['step_B']['participants']
    }
    if isinstance(results_ET, dict) and 'step_C' in results_ET:
        participant_summaries_ET['C'] = results_ET['step_C']['participants']
    if isinstance(results_ET, dict) and 'step_D' in results_ET:
        participant_summaries_ET['D'] = results_ET['step_D']['participants']

    # df_exported_ET = export_phase_summary_csv_current_dir(phase_dfs_ET, participant_summaries_ET)

    generate_sampling_rate_visualization_stepD(results['step_D']['phase_df'], results['step_D']['participants'],
                                     output_folder=r"C:\Users\johan\Desktop\data_pilot_Bricks_VR_2025\Sampling_Rate_Analysis\uniformity_BT")
