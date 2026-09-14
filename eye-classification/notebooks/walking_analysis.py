from pathlib import Path
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("../data/merged")
GROUP_COLS = ["participant_id", "condition_number", "trial_number"]
MELNIK_CONDITION_LABELS = {1: "No delay", 2: "0.7 s delay", 3: "2 s delay"}
MELNIK_FIG5_CONDITION_COLORS = {1: "#4C9F70", 2: "#E6A04A", 3: "#D8665B"}
EXCLUSIONS = [
    ("013", "C3M4A"),
    ("016", ["C1M2A", "C2M1F"]),
    ("018", "C2M6A"),
    ("019", "C3M1F"),
    ("025", "C3M2A"),
    ("033", ["C3M1F", "C3M2A"]),
    ("035", "C3M6A"),
    ("037", "C3M4A"),
    ("038", "C1M5F"),
    ("041", "C2M2A"),
    ("043", "C3M1F"),
    ("044", ["C1M3F", "C1M2A", "C3M1F"]),
    ("047", ["C3M4A", "C3M5F"]),
    ("050", "C3M1F"),
    ("036", None),
]

WALKING_MIN_STABLE_VISIT_MS = 250.0
WALKING_SENSITIVITY_THRESHOLDS_MS = (100.0, 250.0, 500.0)
WALKING_PATH_BIN_MS = 100.0
WALKING_MAX_PATH_GAP_MS = 300.0
WALKING_MAX_PLAUSIBLE_SPEED_M_S = 3.0
WALKING_SPEED_THRESHOLD_M_S = 0.20
WALKING_TASK_CODES = ("M", "W", "R")
WALKING_BETWEEN_TYPES = ("MW", "MR", "WM", "WR", "RM", "RW")
WALKING_FOOT_TO_CODE = {"Model": "M", "Work": "W", "Resource": "R"}

WALKING_REQUIRED_COLUMNS = {
    "participant_id",
    "condition_number",
    "trial_number",
    "time_ms",
    "model_name",
    "LeftFootArea",
    "RightFootArea",
    "Waist_pos_x",
    "Waist_pos_z",
    "bad_sample_LeftFoot",
    "is_interpolated_LeftFoot",
    "bad_sample_RightFoot",
    "is_interpolated_RightFoot",
    "bad_sample_Waist",
    "is_interpolated_Waist",
}


def list_available_participants(data_dir=DATA_DIR):
    return sorted(path.stem.replace("_synced", "") for path in data_dir.glob("*_synced.csv"))


def normalize_walking_participant_id(values):
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64").astype("string").str.zfill(3)


def walking_analysis_mask(df, exclusions=EXCLUSIONS):
    """Apply the same training and manual exclusions used by the main pipeline."""
    participant = normalize_walking_participant_id(df["participant_id"])
    model = df["model_name"].astype("string").str.strip()
    keep = model.notna() & ~model.str.contains("TM", case=False, na=False)

    for participant_id, model_names in exclusions:
        participant_match = participant.eq(str(participant_id).zfill(3))
        if model_names is None:
            keep &= ~participant_match
            continue

        names = [model_names] if isinstance(model_names, str) else list(model_names)
        names = {str(name).strip().casefold() for name in names}
        remove = participant_match & model.str.casefold().isin(names)
        keep &= ~remove

    return keep & df["condition_number"].isin(MELNIK_CONDITION_LABELS)


def load_walking_participant(
    participant_id,
    data_dir=DATA_DIR,
    exclusions=EXCLUSIONS,
):
    """Load only columns needed for walking analysis from one participant file."""
    participant_id = str(participant_id).zfill(3)
    file_path = data_dir / f"{participant_id}_synced.csv"
    header = set(pd.read_csv(file_path, nrows=0).columns)
    missing = sorted(WALKING_REQUIRED_COLUMNS - header)
    if missing:
        raise KeyError(f"{participant_id}: missing walking columns: {missing}")

    df = pd.read_csv(
        file_path,
        usecols=lambda col: col in WALKING_REQUIRED_COLUMNS,
        low_memory=False,
    )
    df["participant_id"] = normalize_walking_participant_id(df["participant_id"])
    df = df.loc[walking_analysis_mask(df, exclusions=exclusions)].copy()
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    return df.sort_values([*GROUP_COLS, "time_ms"]).reset_index(drop=True)


def classify_body_area(left_area, right_area):
    """Combine both foot labels into a conservative whole-body area state."""
    left = str(left_area).strip().title() if pd.notna(left_area) else None
    right = str(right_area).strip().title() if pd.notna(right_area) else None

    if left == right and left in WALKING_FOOT_TO_CODE:
        return WALKING_FOOT_TO_CODE[left]
    if left == right == "Middle":
        return "Middle"
    if left in WALKING_FOOT_TO_CODE and right == "Middle":
        return WALKING_FOOT_TO_CODE[left]
    if right in WALKING_FOOT_TO_CODE and left == "Middle":
        return WALKING_FOOT_TO_CODE[right]
    if left in WALKING_FOOT_TO_CODE and right in WALKING_FOOT_TO_CODE:
        return "Transit"
    return "Unknown"


def classify_body_area_samples(df):
    """Classify samples, accepting interpolated foot labels but rejecting unresolved bad data."""
    states = pd.Series(
        [
            classify_body_area(left, right)
            for left, right in zip(df["LeftFootArea"], df["RightFootArea"])
        ],
        index=df.index,
        dtype="string",
    )
    unresolved_left = (
        pd.to_numeric(df["bad_sample_LeftFoot"], errors="coerce").eq(1)
        & ~pd.to_numeric(df["is_interpolated_LeftFoot"], errors="coerce").eq(1)
    )
    unresolved_right = (
        pd.to_numeric(df["bad_sample_RightFoot"], errors="coerce").eq(1)
        & ~pd.to_numeric(df["is_interpolated_RightFoot"], errors="coerce").eq(1)
    )
    states.loc[unresolved_left | unresolved_right] = "Unknown"
    return states


def median_positive_step(values):
    values = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
    steps = np.diff(values)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return float(np.median(steps)) if len(steps) else np.nan


def make_body_area_runs(df, group_cols=GROUP_COLS):
    """Run-length encode whole-body area states within each retained trial."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["body_area_state"] = classify_body_area_samples(work)
    previous = work.groupby(group_cols, dropna=False)["body_area_state"].shift()
    work["body_area_run_id"] = work["body_area_state"].ne(previous).cumsum()

    trial_steps = (
        work.groupby(group_cols, dropna=False)["time_ms"]
        .apply(median_positive_step)
        .rename("sample_period_ms")
        .reset_index()
    )
    runs = (
        work.groupby([*group_cols, "body_area_run_id"], dropna=False)
        .agg(
            body_area=("body_area_state", "first"),
            run_start_ms=("time_ms", "min"),
            run_end_ms=("time_ms", "max"),
            n_samples=("time_ms", "size"),
            start_waist_x=("Waist_pos_x", "first"),
            start_waist_z=("Waist_pos_z", "first"),
            end_waist_x=("Waist_pos_x", "last"),
            end_waist_z=("Waist_pos_z", "last"),
        )
        .reset_index()
        .merge(trial_steps, on=group_cols, how="left", validate="many_to_one")
    )
    runs["run_duration_ms"] = (
        runs["run_end_ms"] - runs["run_start_ms"] + runs["sample_period_ms"]
    )
    return runs.sort_values([*group_cols, "run_start_ms"]).reset_index(drop=True)


def make_stable_walking_visits(
    body_runs,
    min_stable_ms=WALKING_MIN_STABLE_VISIT_MS,
    group_cols=GROUP_COLS,
):
    """Keep stable M/W/R episodes and collapse repeated labels across transit states."""
    if body_runs.empty:
        return pd.DataFrame()

    visits = body_runs[
        body_runs["body_area"].isin(WALKING_TASK_CODES)
        & body_runs["run_duration_ms"].ge(min_stable_ms)
    ].copy()
    if visits.empty:
        return visits

    previous_task_area = visits.groupby(group_cols, dropna=False)["body_area"].shift()
    visits["new_visit"] = previous_task_area.isna() | visits["body_area"].ne(previous_task_area)
    visits["visit_id"] = visits.groupby(group_cols, dropna=False)["new_visit"].cumsum().astype(int)

    # Repeated stable runs in the same task area, separated only by Middle/Transit,
    # are one physical visit. Keep the last run end as the true departure time.
    visits = (
        visits.groupby([*group_cols, "visit_id"], dropna=False)
        .agg(
            body_area=("body_area", "first"),
            run_start_ms=("run_start_ms", "first"),
            run_end_ms=("run_end_ms", "last"),
            stable_area_duration_ms=("run_duration_ms", "sum"),
            n_stable_runs=("body_area_run_id", "size"),
            start_waist_x=("start_waist_x", "first"),
            start_waist_z=("start_waist_z", "first"),
            end_waist_x=("end_waist_x", "last"),
            end_waist_z=("end_waist_z", "last"),
        )
        .reset_index()
    )
    visits["visit_span_duration_ms"] = visits["run_end_ms"] - visits["run_start_ms"]
    return visits.reset_index(drop=True)


def make_walking_transitions(visits, group_cols=GROUP_COLS):
    """Connect consecutive distinct stable task-area visits once per physical journey."""
    if visits.empty:
        return pd.DataFrame()

    transitions = visits.copy()
    previous_columns = [
        "visit_id",
        "body_area",
        "run_start_ms",
        "run_end_ms",
        "end_waist_x",
        "end_waist_z",
    ]
    for col in previous_columns:
        transitions[f"from_{col}"] = transitions.groupby(group_cols, dropna=False)[col].shift()

    transitions = transitions[transitions["from_visit_id"].notna()].copy()
    transitions = transitions.rename(columns={
        "visit_id": "to_visit_id",
        "body_area": "to_area",
        "run_start_ms": "arrival_ms",
        "run_end_ms": "to_visit_end_ms",
        "start_waist_x": "arrival_waist_x",
        "start_waist_z": "arrival_waist_z",
        "from_body_area": "from_area",
        "from_run_end_ms": "departure_ms",
        "from_end_waist_x": "departure_waist_x",
        "from_end_waist_z": "departure_waist_z",
    })
    transitions["transition_type"] = transitions["from_area"] + transitions["to_area"]
    transitions["travel_time_s"] = (
        transitions["arrival_ms"] - transitions["departure_ms"]
    ) / 1000.0
    transitions["transition_displacement_m"] = np.hypot(
        pd.to_numeric(transitions["arrival_waist_x"], errors="coerce")
        - pd.to_numeric(transitions["departure_waist_x"], errors="coerce"),
        pd.to_numeric(transitions["arrival_waist_z"], errors="coerce")
        - pd.to_numeric(transitions["departure_waist_z"], errors="coerce"),
    )

    keep_columns = [
        *group_cols,
        "from_visit_id",
        "to_visit_id",
        "from_area",
        "to_area",
        "transition_type",
        "departure_ms",
        "arrival_ms",
        "travel_time_s",
        "transition_displacement_m",
    ]
    transitions = transitions[keep_columns].copy()
    if not transitions["transition_type"].isin(WALKING_BETWEEN_TYPES).all():
        raise AssertionError("Walking transitions contain an invalid or same-area endpoint pair.")
    if transitions["travel_time_s"].lt(0).any():
        raise AssertionError("Walking transition arrival precedes departure.")
    return transitions.reset_index(drop=True)


def validate_walking_logic():
    """Exercise the intended bridge, return, threshold, and foot-state rules."""
    assert classify_body_area("Work", "Work") == "W"
    assert classify_body_area("Work", "Middle") == "W"
    assert classify_body_area("Model", "Resource") == "Transit"
    assert classify_body_area("Middle", "Middle") == "Middle"

    def synthetic_runs(states_and_durations):
        rows = []
        cursor_ms = 0.0
        for run_id, (state, duration_ms) in enumerate(states_and_durations, start=1):
            end_ms = cursor_ms + duration_ms
            rows.append({
                "participant_id": "test",
                "condition_number": 1,
                "trial_number": 1,
                "body_area_run_id": run_id,
                "body_area": state,
                "run_start_ms": cursor_ms,
                "run_end_ms": end_ms,
                "run_duration_ms": duration_ms,
                "start_waist_x": run_id * 0.1,
                "start_waist_z": 0.0,
                "end_waist_x": run_id * 0.1,
                "end_waist_z": 0.0,
            })
            cursor_ms = end_ms
        return pd.DataFrame(rows)

    bridged = synthetic_runs([
        ("W", 400.0),
        ("Middle", 100.0),
        ("Transit", 100.0),
        ("R", 400.0),
        ("Middle", 100.0),
        ("W", 400.0),
    ])
    bridged_visits = make_stable_walking_visits(bridged, min_stable_ms=250.0)
    bridged_transitions = make_walking_transitions(bridged_visits)
    assert bridged_visits["body_area"].tolist() == ["W", "R", "W"]
    assert bridged_transitions["transition_type"].tolist() == ["WR", "RW"]

    same_area_return = synthetic_runs([
        ("W", 400.0),
        ("Middle", 150.0),
        ("W", 400.0),
    ])
    same_area_visits = make_stable_walking_visits(
        same_area_return,
        min_stable_ms=250.0,
    )
    assert same_area_visits["body_area"].tolist() == ["W"]
    assert make_walking_transitions(same_area_visits).empty

    short_task_blip = synthetic_runs([
        ("W", 400.0),
        ("R", 100.0),
        ("M", 400.0),
    ])
    short_blip_visits = make_stable_walking_visits(
        short_task_blip,
        min_stable_ms=250.0,
    )
    short_blip_transitions = make_walking_transitions(short_blip_visits)
    assert short_blip_visits["body_area"].tolist() == ["W", "M"]
    assert short_blip_transitions["transition_type"].tolist() == ["WM"]
    return True


def make_waist_path_trial_summary(df, group_cols=GROUP_COLS):
    """Estimate horizontal waist path at 100 ms resolution to suppress tracker jitter."""
    columns = [
        *group_cols,
        "horizontal_path_m",
        "walking_time_s",
        "path_observed_time_s",
        "path_coverage_pct",
        "n_path_steps",
        "n_rejected_fast_steps",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    path_rows = []
    for trial_key, trial in df.groupby(group_cols, sort=False, dropna=False):
        trial = trial[[
            "time_ms",
            "Waist_pos_x",
            "Waist_pos_z",
            "bad_sample_Waist",
            "is_interpolated_Waist",
        ]].copy()
        trial["time_ms"] = pd.to_numeric(trial["time_ms"], errors="coerce")
        trial_start_ms = trial["time_ms"].min()
        trial_end_ms = trial["time_ms"].max()
        unresolved_waist = (
            pd.to_numeric(trial["bad_sample_Waist"], errors="coerce").eq(1)
            & ~pd.to_numeric(trial["is_interpolated_Waist"], errors="coerce").eq(1)
        )
        trial.loc[unresolved_waist, ["Waist_pos_x", "Waist_pos_z"]] = np.nan
        trial["Waist_pos_x"] = pd.to_numeric(trial["Waist_pos_x"], errors="coerce")
        trial["Waist_pos_z"] = pd.to_numeric(trial["Waist_pos_z"], errors="coerce")
        trial = trial.dropna(
            subset=["time_ms", "Waist_pos_x", "Waist_pos_z"]
        ).sort_values("time_ms")

        values = dict(zip(group_cols, trial_key if isinstance(trial_key, tuple) else (trial_key,)))
        if trial.empty:
            path_rows.append({**values, **{col: np.nan for col in columns[len(group_cols):]}})
            continue

        trial["path_bin"] = np.floor(
            (trial["time_ms"] - trial_start_ms) / WALKING_PATH_BIN_MS
        ).astype("int64")
        binned = (
            trial.groupby("path_bin", sort=True)
            .agg(
                time_ms=("time_ms", "median"),
                waist_x=("Waist_pos_x", "median"),
                waist_z=("Waist_pos_z", "median"),
            )
            .reset_index(drop=True)
        )

        dt_s = binned["time_ms"].diff() / 1000.0
        step_m = np.hypot(binned["waist_x"].diff(), binned["waist_z"].diff())
        speed_m_s = step_m / dt_s
        connected = dt_s.gt(0) & dt_s.le(WALKING_MAX_PATH_GAP_MS / 1000.0)
        plausible = connected & speed_m_s.le(WALKING_MAX_PLAUSIBLE_SPEED_M_S)
        moving = plausible & speed_m_s.ge(WALKING_SPEED_THRESHOLD_M_S)

        trial_duration_s = max((trial_end_ms - trial_start_ms) / 1000.0, 0.0)
        observed_time_s = float(dt_s.where(plausible, 0).sum())
        path_rows.append({
            **values,
            "horizontal_path_m": float(step_m.where(plausible, 0).sum()),
            "walking_time_s": float(dt_s.where(moving, 0).sum()),
            "path_observed_time_s": observed_time_s,
            "path_coverage_pct": 100 * observed_time_s / trial_duration_s if trial_duration_s > 0 else np.nan,
            "n_path_steps": int(plausible.sum()),
            "n_rejected_fast_steps": int((connected & speed_m_s.gt(WALKING_MAX_PLAUSIBLE_SPEED_M_S)).sum()),
        })

    return pd.DataFrame(path_rows, columns=columns)


def make_walking_trial_summary(df, visits, transitions, path_summary, group_cols=GROUP_COLS):
    """Zero-complete transition counts and combine them with visit/path metrics."""
    trial_keys = df[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)
    type_grid = trial_keys.assign(_key=1).merge(
        pd.DataFrame({"transition_type": WALKING_BETWEEN_TYPES, "_key": 1}),
        on="_key",
        how="inner",
    ).drop(columns="_key")

    if transitions.empty:
        transition_counts = type_grid.assign(n_transitions=0)
    else:
        observed_counts = (
            transitions.groupby([*group_cols, "transition_type"], dropna=False)
            .size()
            .rename("n_transitions")
            .reset_index()
        )
        transition_counts = type_grid.merge(
            observed_counts,
            on=[*group_cols, "transition_type"],
            how="left",
            validate="one_to_one",
        )
        transition_counts["n_transitions"] = transition_counts["n_transitions"].fillna(0).astype(int)

    wide = (
        transition_counts.pivot_table(
            index=group_cols,
            columns="transition_type",
            values="n_transitions",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    for transition_type in WALKING_BETWEEN_TYPES:
        if transition_type not in wide.columns:
            wide[transition_type] = 0

    visit_counts = (
        visits.groupby(group_cols, dropna=False)
        .agg(
            n_stable_task_area_visits=("visit_id", "size"),
            n_stable_model_visits=("body_area", lambda values: values.eq("M").sum()),
            n_stable_work_visits=("body_area", lambda values: values.eq("W").sum()),
            n_stable_resource_visits=("body_area", lambda values: values.eq("R").sum()),
        )
        .reset_index()
        if not visits.empty
        else pd.DataFrame(columns=[
            *group_cols,
            "n_stable_task_area_visits",
            "n_stable_model_visits",
            "n_stable_work_visits",
            "n_stable_resource_visits",
        ])
    )

    summary = trial_keys.merge(wide, on=group_cols, how="left", validate="one_to_one")
    summary = summary.merge(visit_counts, on=group_cols, how="left", validate="one_to_one")
    summary = summary.merge(path_summary, on=group_cols, how="left", validate="one_to_one")
    count_columns = [
        *WALKING_BETWEEN_TYPES,
        "n_stable_task_area_visits",
        "n_stable_model_visits",
        "n_stable_work_visits",
        "n_stable_resource_visits",
    ]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    summary["model_return_trips"] = summary["WM"] + summary["RM"]
    summary["work_resource_trips"] = summary["WR"] + summary["RW"]
    summary["total_between_area_trips"] = summary[list(WALKING_BETWEEN_TYPES)].sum(axis=1)

    expected = summary[list(WALKING_BETWEEN_TYPES)].sum(axis=1)
    if not expected.eq(summary["total_between_area_trips"]).all():
        raise AssertionError("Per-trial walking transition totals do not reconcile.")
    return transition_counts, summary


def summarize_walking_sensitivity(body_runs, thresholds=WALKING_SENSITIVITY_THRESHOLDS_MS):
    """Return compact trial metrics under several stable-visit thresholds."""
    rows = []
    trial_keys = body_runs[GROUP_COLS].drop_duplicates()
    for threshold_ms in thresholds:
        visits = make_stable_walking_visits(body_runs, min_stable_ms=threshold_ms)
        transitions = make_walking_transitions(visits)
        counts = (
            transitions.groupby(GROUP_COLS, dropna=False)
            .agg(
                n_transitions=("transition_type", "size"),
                model_return_trips=("transition_type", lambda values: values.isin(["WM", "RM"]).sum()),
                work_resource_trips=("transition_type", lambda values: values.isin(["WR", "RW"]).sum()),
            )
            .reset_index()
            if not transitions.empty
            else pd.DataFrame(columns=[*GROUP_COLS, "n_transitions", "model_return_trips", "work_resource_trips"])
        )
        counts = trial_keys.merge(counts, on=GROUP_COLS, how="left")
        for col in ["n_transitions", "model_return_trips", "work_resource_trips"]:
            counts[col] = counts[col].fillna(0).astype(int)
        counts["threshold_ms"] = threshold_ms
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def process_walking_participant(
    participant_id,
    data_dir=DATA_DIR,
    exclusions=EXCLUSIONS,
):
    """Produce only compact walking tables for one participant, then release samples."""
    participant_id = str(participant_id).zfill(3)
    raw = load_walking_participant(
        participant_id,
        data_dir=data_dir,
        exclusions=exclusions,
    )
    analysis_rows = len(raw)
    if raw.empty:
        report = {
            "participant_id": participant_id,
            "analysis_rows": 0,
            "n_valid_trials": 0,
            "n_stable_visits": 0,
            "n_transitions": 0,
            "pct_task_area_time": np.nan,
            "pct_transit_time": np.nan,
            "pct_unknown_time": np.nan,
            "pct_interpolated_foot_samples": np.nan,
            "pct_interpolated_waist_samples": np.nan,
            "mean_path_coverage_pct": np.nan,
            "n_rejected_fast_steps": 0,
            "status": "no analysis rows after cleaning/exclusions",
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), report

    runs = make_body_area_runs(raw)
    visits = make_stable_walking_visits(runs)
    transitions = make_walking_transitions(visits)
    path_summary = make_waist_path_trial_summary(raw)
    transition_counts, trial_summary = make_walking_trial_summary(
        raw, visits, transitions, path_summary
    )
    sensitivity = summarize_walking_sensitivity(runs)

    total_run_ms = pd.to_numeric(runs["run_duration_ms"], errors="coerce").sum()
    task_run_ms = runs.loc[
        runs["body_area"].isin(WALKING_TASK_CODES), "run_duration_ms"
    ].sum()
    transit_run_ms = runs.loc[runs["body_area"].eq("Transit"), "run_duration_ms"].sum()
    unknown_run_ms = runs.loc[runs["body_area"].eq("Unknown"), "run_duration_ms"].sum()
    interpolated_foot = (
        pd.to_numeric(raw["is_interpolated_LeftFoot"], errors="coerce").eq(1)
        | pd.to_numeric(raw["is_interpolated_RightFoot"], errors="coerce").eq(1)
    )
    interpolated_waist = pd.to_numeric(
        raw["is_interpolated_Waist"], errors="coerce"
    ).eq(1)

    report = {
        "participant_id": participant_id,
        "analysis_rows": analysis_rows,
        "n_valid_trials": len(trial_summary),
        "n_stable_visits": len(visits),
        "n_transitions": len(transitions),
        "pct_task_area_time": 100 * task_run_ms / total_run_ms if total_run_ms else np.nan,
        "pct_transit_time": 100 * transit_run_ms / total_run_ms if total_run_ms else np.nan,
        "pct_unknown_time": 100 * unknown_run_ms / total_run_ms if total_run_ms else np.nan,
        "pct_interpolated_foot_samples": 100 * interpolated_foot.mean(),
        "pct_interpolated_waist_samples": 100 * interpolated_waist.mean(),
        "mean_path_coverage_pct": trial_summary["path_coverage_pct"].mean(),
        "n_rejected_fast_steps": int(trial_summary["n_rejected_fast_steps"].sum()),
        "status": "ok",
    }

    del raw, runs, path_summary
    gc.collect()
    return visits, transitions, trial_summary, sensitivity, report


def make_all_participant_walking_tables(
    participant_ids=None,
    limit=None,
    data_dir=DATA_DIR,
    exclusions=EXCLUSIONS,
):
    """Stream participant files and retain only compact visit/trip/trial tables."""
    participant_ids = list(
        list_available_participants(data_dir)
        if participant_ids is None
        else participant_ids
    )
    if limit is not None:
        participant_ids = participant_ids[:limit]

    visit_tables = []
    transition_tables = []
    trial_tables = []
    sensitivity_tables = []
    reports = []

    for index, participant_id in enumerate(participant_ids, start=1):
        participant_id = str(participant_id).zfill(3)
        print(f"[{index:02d}/{len(participant_ids):02d}] Walking participant {participant_id}", flush=True)
        try:
            visits, transitions, trials, sensitivity, report = process_walking_participant(
                participant_id,
                data_dir=data_dir,
                exclusions=exclusions,
            )
        except Exception as exc:
            visits = transitions = trials = sensitivity = pd.DataFrame()
            report = {
                "participant_id": participant_id,
                "analysis_rows": pd.NA,
                "n_valid_trials": 0,
                "n_stable_visits": 0,
                "n_transitions": 0,
                "pct_task_area_time": np.nan,
                "pct_transit_time": np.nan,
                "pct_unknown_time": np.nan,
                "pct_interpolated_foot_samples": np.nan,
                "pct_interpolated_waist_samples": np.nan,
                "mean_path_coverage_pct": np.nan,
                "n_rejected_fast_steps": pd.NA,
                "status": f"ERROR: {exc}",
            }
            print(f"  ERROR: {exc}", flush=True)

        if not visits.empty:
            visit_tables.append(visits)
        if not transitions.empty:
            transition_tables.append(transitions)
        if not trials.empty:
            trial_tables.append(trials)
        if not sensitivity.empty:
            sensitivity_tables.append(sensitivity)
        reports.append(report)
        del visits, transitions, trials, sensitivity
        gc.collect()

    return (
        pd.concat(visit_tables, ignore_index=True) if visit_tables else pd.DataFrame(),
        pd.concat(transition_tables, ignore_index=True) if transition_tables else pd.DataFrame(),
        pd.concat(trial_tables, ignore_index=True) if trial_tables else pd.DataFrame(),
        pd.concat(sensitivity_tables, ignore_index=True) if sensitivity_tables else pd.DataFrame(),
        pd.DataFrame(reports),
    )


def participant_condition_walking_means(trial_summary, require_complete_conditions=True):
    """Average trials within participant/condition and optionally keep a paired cohort."""
    metrics = [
        *WALKING_BETWEEN_TYPES,
        "model_return_trips",
        "work_resource_trips",
        "total_between_area_trips",
        "horizontal_path_m",
        "walking_time_s",
        "path_coverage_pct",
    ]
    participant_means = (
        trial_summary.groupby(["participant_id", "condition_number"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    if require_complete_conditions:
        expected_conditions = set(MELNIK_CONDITION_LABELS)
        complete_ids = [
            participant_id
            for participant_id, group in participant_means.groupby("participant_id", sort=False)
            if set(group["condition_number"].dropna()) == expected_conditions
        ]
        participant_means = participant_means[
            participant_means["participant_id"].isin(complete_ids)
        ].copy()
    return participant_means.reset_index(drop=True)


def walking_condition_summary(participant_condition_means):
    """Group mean and SEM across participant-level condition means."""
    metrics = [
        *WALKING_BETWEEN_TYPES,
        "model_return_trips",
        "work_resource_trips",
        "total_between_area_trips",
        "horizontal_path_m",
        "walking_time_s",
    ]
    rows = []
    for condition, group in participant_condition_means.groupby("condition_number", sort=True):
        row = {
            "condition_number": condition,
            "condition_label": MELNIK_CONDITION_LABELS.get(condition, str(condition)),
            "n_participants": group["participant_id"].nunique(),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = values.mean()
            row[f"sem_{metric}"] = values.sem() if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_walking_threshold_sensitivity(sensitivity):
    """Average within participant first, then summarize each stability threshold."""
    participant_means = (
        sensitivity.groupby(
            ["participant_id", "condition_number", "threshold_ms"],
            dropna=False,
        )[["n_transitions", "model_return_trips", "work_resource_trips"]]
        .mean()
        .reset_index()
    )
    expected_conditions = set(MELNIK_CONDITION_LABELS)
    complete_ids = [
        participant_id
        for participant_id, group in participant_means.groupby("participant_id", sort=False)
        if set(group["condition_number"].dropna()) == expected_conditions
    ]
    participant_means = participant_means[
        participant_means["participant_id"].isin(complete_ids)
    ].copy()
    return (
        participant_means.groupby(["condition_number", "threshold_ms"], dropna=False)
        .agg(
            mean_total_trips=("n_transitions", "mean"),
            mean_model_returns=("model_return_trips", "mean"),
            mean_work_resource_trips=("work_resource_trips", "mean"),
            n_participants=("participant_id", "nunique"),
        )
        .reset_index()
    )


def walking_mode_first_seen(values):
    """Return the most frequent state, resolving ties by first occurrence."""
    values = values.dropna()
    if values.empty:
        return "Unknown"
    counts = values.value_counts(sort=False)
    winners = set(counts[counts.eq(counts.max())].index)
    return next(value for value in values if value in winners)


def select_walking_qc_trials(trial_summary, participant_id):
    """Choose the median-path trial for each available condition."""
    participant_id = str(participant_id).zfill(3)
    available = trial_summary[trial_summary["participant_id"].eq(participant_id)].copy()
    selected = []
    for condition, group in available.groupby("condition_number", sort=True):
        finite = group.dropna(subset=["horizontal_path_m"]).copy()
        if finite.empty:
            chosen = group.iloc[0]
        else:
            median_path = finite["horizontal_path_m"].median()
            chosen = finite.loc[(finite["horizontal_path_m"] - median_path).abs().idxmin()]
        selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True) if selected else pd.DataFrame()


def plot_walking_trajectory_qc(
    participant_id,
    trial_summary,
    data_dir=DATA_DIR,
    exclusions=EXCLUSIONS,
):
    """Plot one median-path trial per condition to visually audit area labels."""
    participant_id = str(participant_id).zfill(3)
    selected = select_walking_qc_trials(trial_summary, participant_id)
    if selected.empty:
        print(f"No walking-QC trials available for participant {participant_id}.")
        return None

    raw = load_walking_participant(
        participant_id,
        data_dir=data_dir,
        exclusions=exclusions,
    )
    raw["body_area"] = classify_body_area_samples(raw)
    state_colors = {
        "M": "#4472C4",
        "W": "#4C9F70",
        "R": "#E6A04A",
        "Middle": "#9B8BB4",
        "Transit": "#8B8B8B",
        "Unknown": "#222222",
    }

    paths = []
    for _, selected_trial in selected.iterrows():
        trial_mask = np.logical_and.reduce([
            raw[col].eq(selected_trial[col]).to_numpy()
            for col in GROUP_COLS
        ])
        trial = raw.loc[trial_mask, ["time_ms", "Waist_pos_x", "Waist_pos_z", "body_area"]].copy()
        trial = trial.dropna(subset=["time_ms", "Waist_pos_x", "Waist_pos_z"])
        if trial.empty:
            continue
        start_ms = trial["time_ms"].min()
        trial["path_bin"] = np.floor((trial["time_ms"] - start_ms) / WALKING_PATH_BIN_MS).astype("int64")
        path = (
            trial.groupby("path_bin", sort=True)
            .agg(
                time_ms=("time_ms", "median"),
                waist_x=("Waist_pos_x", "median"),
                waist_z=("Waist_pos_z", "median"),
                body_area=("body_area", walking_mode_first_seen),
            )
            .reset_index(drop=True)
        )
        paths.append((selected_trial, path))

    if not paths:
        print(f"No finite waist trajectories available for participant {participant_id}.")
        return None

    fig, axes = plt.subplots(1, len(paths), figsize=(4.4 * len(paths), 4.3), squeeze=False)
    axes = axes.ravel()
    all_x = pd.concat([path["waist_x"] for _, path in paths])
    all_z = pd.concat([path["waist_z"] for _, path in paths])
    x_pad = max((all_x.max() - all_x.min()) * 0.08, 0.05)
    z_pad = max((all_z.max() - all_z.min()) * 0.08, 0.05)

    for ax, (selected_trial, path) in zip(axes, paths):
        ax.plot(path["waist_x"], path["waist_z"], color="#C8C8C8", linewidth=0.8, zorder=1)
        for state, color in state_colors.items():
            points = path[path["body_area"].eq(state)]
            if not points.empty:
                ax.scatter(
                    points["waist_x"],
                    points["waist_z"],
                    s=9,
                    color=color,
                    alpha=0.75,
                    linewidths=0,
                    label=state,
                    zorder=2,
                )
        ax.scatter(path.iloc[0]["waist_x"], path.iloc[0]["waist_z"], marker="^", s=55, color="black", label="Start", zorder=3)
        ax.scatter(path.iloc[-1]["waist_x"], path.iloc[-1]["waist_z"], marker="s", s=40, color="black", label="End", zorder=3)
        ax.set_title(
            f"{MELNIK_CONDITION_LABELS[selected_trial['condition_number']]}\n"
            f"trial {int(selected_trial['trial_number'])}, path {selected_trial['horizontal_path_m']:.1f} m"
        )
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_z.min() - z_pad, all_z.max() + z_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Waist x position (m)")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Waist z position (m)")

    by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label.update(zip(labels, handles))
    legend_order = [
        label for label in ["M", "W", "R", "Middle", "Transit", "Unknown", "Start", "End"]
        if label in by_label
    ]
    fig.legend(
        [by_label[label] for label in legend_order],
        legend_order,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=min(len(by_label), 8),
        frameon=False,
    )
    fig.suptitle(
        f"Walking-detection QC: participant {participant_id} median-path trials",
        y=1.05,
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    plt.show()
    del raw
    gc.collect()
    return axes


def plot_walking_strategy_summary(condition_summary):
    """Plot model returns, work-resource trips, and horizontal path length."""
    conditions = [c for c in MELNIK_CONDITION_LABELS if c in set(condition_summary["condition_number"])]
    labels = [MELNIK_CONDITION_LABELS[c] for c in conditions]
    colors = [MELNIK_FIG5_CONDITION_COLORS[c] for c in conditions]
    summary = condition_summary.set_index("condition_number").reindex(conditions)
    x = np.arange(len(conditions))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    fig.suptitle("Effect of template delay on physical movement between task areas", y=1.03, fontsize=13)

    ax = axes[0]
    wm = summary["mean_WM"].to_numpy(dtype=float)
    rm = summary["mean_RM"].to_numpy(dtype=float)
    totals = summary["mean_model_return_trips"].to_numpy(dtype=float)
    sems = summary["sem_model_return_trips"].fillna(0).to_numpy(dtype=float)
    ax.bar(x, wm, color=colors, alpha=0.62, edgecolor="white", label="WM: work to model")
    ax.bar(x, rm, bottom=wm, color=colors, alpha=0.96, edgecolor="white", hatch="//", label="RM: resource to model")
    ax.errorbar(x, totals, yerr=sems, fmt="none", ecolor="black", capsize=3, lw=1)
    for xpos, total in zip(x, totals):
        ax.text(xpos, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("A. Physical returns to model")
    ax.set_ylabel("Mean trips per valid trial")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    wr = summary["mean_WR"].to_numpy(dtype=float)
    rw = summary["mean_RW"].to_numpy(dtype=float)
    totals = summary["mean_work_resource_trips"].to_numpy(dtype=float)
    sems = summary["sem_work_resource_trips"].fillna(0).to_numpy(dtype=float)
    ax.bar(x, wr, color=colors, alpha=0.62, edgecolor="white", label="WR: work to resource")
    ax.bar(x, rw, bottom=wr, color=colors, alpha=0.96, edgecolor="white", hatch="//", label="RW: resource to work")
    ax.errorbar(x, totals, yerr=sems, fmt="none", ecolor="black", capsize=3, lw=1)
    for xpos, total in zip(x, totals):
        ax.text(xpos, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("B. Work-resource trips")
    ax.set_ylabel("Mean trips per valid trial")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    means = summary["mean_horizontal_path_m"].to_numpy(dtype=float)
    sems = summary["sem_horizontal_path_m"].fillna(0).to_numpy(dtype=float)
    ax.bar(x, means, yerr=sems, capsize=3, color=colors, edgecolor="white")
    for xpos, mean in zip(x, means):
        ax.text(xpos, mean, f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("C. Horizontal body path")
    ax.set_ylabel("Mean path length per valid trial (m)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

    fig.text(
        0.01,
        -0.045,
        "Task-area visits require at least 250 ms. Paths use 100 ms median waist positions. "
        "Means and SEM use participant-level condition means from participants with all three conditions.",
        ha="left",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()
    return axes
