import os
import csv
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency, chisquare

"""
Validates whether the Unity item‑randomization procedure worked as intended.
The script loads all model‑order CSV files and checks four aspects of
randomization: 
(1) whether models are evenly distributed across conditions,
(2) whether model identity is independent of within‑condition position,
(3) whether complexity levels (C1/C2/C3) are balanced across positions, and
(4) whether familiarity (F/A) is balanced across positions. Chi‑square tests
are used to detect deviations from the expected uniform distributions.
"""

# Helpers: parsing & loading
def extract_complexity(model_name):
    """
    Assumes names like C1M1F, C2M4A, ...
    -> returns 'C1', 'C2', 'C3'
    """
    return model_name[0:2]


def extract_familiarity(model_name):
    """
    Assumes last character is 'F' or 'A'
    -> returns 'F' or 'A'
    """
    return model_name[-1]


def load_all_orders(folder_path):
    """
    Reads all CSVs in folder and returns a list of participants.
    Each participant is a dict: condition_number -> list of 6 experimental models
    (TM is removed; only the 6 experimental models per condition remain).
    """
    participants = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        path = os.path.join(folder_path, filename)
        condition_orders = {1: [], 2: [], 3: []}

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cond = int(row["ConditionNumber"])
                model = row["ModelName"].strip()
                # Skip tutorial
                if model == "TM":
                    continue
                condition_orders[cond].append(model)

        # Optionally sanity-check: each condition should have exactly 6 experimental models
        for cond in (1, 2, 3):
            if len(condition_orders[cond]) != 6:
                print(f"WARNING: {filename}, Condition {cond} has {len(condition_orders[cond])} models (expected 6).")

        participants.append(condition_orders)

    return participants


# 1. Model ↔ Condition
def test_model_condition_balance(participants):
    """
    For each model, count how often it appears in each of the 3 conditions.
    Run a chi-square test of independence: model × condition.
    If randomization is correct, model and condition should be independent.
    """
    # Collect all models and counts per condition
    model_condition_counts = defaultdict(lambda: Counter())

    for p in participants:
        for cond in (1, 2, 3):
            for model in p[cond]:
                model_condition_counts[model][cond] += 1

    models = sorted(model_condition_counts.keys())
    conditions = [1, 2, 3]

    # Build contingency table: rows=models, cols=conditions
    table = []
    for m in models:
        row = [model_condition_counts[m][c] for c in conditions]
        table.append(row)

    chi2, p, dof, expected = chi2_contingency(table)

    print("\n Model × Condition balance ")
    print(f"Models included: {len(models)}")
    print(f"Chi-square = {chi2:.4f}, df = {dof}, p = {p:.4f}")
    if p < 0.05:
        print("  -> Significant: Model assignment may NOT be independent of condition.")
    else:
        print("  -> Not significant: Model assignment appears independent of condition (as expected).")


# 2. Model <-> Position (within each condition)
def test_model_position_within_condition(participants):
    """
    For each condition separately, test whether model identity is independent
    of position (1–6) using chi-square test of independence (model × position).
    """
    print("\n Model × Position within each condition ")

    for cond in (1, 2, 3):
        # Count occurrences of each model at each position
        model_pos_counts = defaultdict(lambda: Counter())
        for p in participants:
            models_in_cond = p[cond]
            if len(models_in_cond) != 6:
                # Skip malformed entries
                continue
            for pos_idx, model in enumerate(models_in_cond):
                pos = pos_idx + 1  # positions 1-6
                model_pos_counts[model][pos] += 1

        models = sorted(model_pos_counts.keys())
        positions = [1, 2, 3, 4, 5, 6]

        # Build contingency table: rows=models, cols=positions
        table = []
        for m in models:
            row = [model_pos_counts[m][pos] for pos in positions]
            table.append(row)

        if len(models) == 0:
            print(f"Condition {cond}: No data found.")
            continue

        chi2, p, dof, expected = chi2_contingency(table)
        print(f"\nCondition {cond}:")
        print(f"  Models included: {len(models)}")
        print(f"  Chi-square = {chi2:.4f}, df = {dof}, p = {p:.4f}")
        if p < 0.05:
            print("    -> Significant: Model identity may NOT be independent of position in this condition.")
        else:
            print("    -> Not significant: Model identity appears independent of position in this condition.")


# 3. Complexity per position (across all conditions)

def test_complexity_by_position(participants):
    """
    For each position (1–6, experimental part only), across all conditions and
    participants, test whether C1/C2/C3 appear equally often (1/3 each).
    Uses chi-square goodness-of-fit per position.
    """
    # positions 1-6, collect complexities across all conditions and participants
    pos_complexities = {pos: [] for pos in range(1, 7)}

    for p in participants:
        for cond in (1, 2, 3):
            models_in_cond = p[cond]
            if len(models_in_cond) != 6:
                continue
            for pos_idx, model in enumerate(models_in_cond):
                pos = pos_idx + 1
                comp = extract_complexity(model)
                pos_complexities[pos].append(comp)

    print("\n Complexity distribution by position (C1/C2/C3) ")

    for pos in range(1, 7):
        comps = pos_complexities[pos]
        counts = Counter(comps)
        total = sum(counts.values())
        if total == 0:
            print(f"Position {pos}: No data.")
            continue

        observed = [counts.get("C1", 0), counts.get("C2", 0), counts.get("C3", 0)]
        expected = [total / 3.0] * 3

        chi2, p = chisquare(f_obs=observed, f_exp=expected)

        print(f"\nPosition {pos}:")
        print(f"  Counts: C1={observed[0]}, C2={observed[1]}, C3={observed[2]}, total={total}")
        print(f"  Chi-square = {chi2:.4f}, p = {p:.4f}")
        if p < 0.05:
            print("    -> Significant: Complexity may NOT be equally likely at this position.")
        else:
            print("    -> Not significant: Complexity appears ~1/3 each at this position.")


# 4. Familiarity per position (across all conditions)

def test_familiarity_by_position(participants):
    """
    For each position (1–6, experimental part only), across all conditions and
    participants, test whether F/A appear with 50/50 probability.
    Uses chi-square goodness-of-fit per position.
    """
    pos_fams = {pos: [] for pos in range(1, 7)}

    for p in participants:
        for cond in (1, 2, 3):
            models_in_cond = p[cond]
            if len(models_in_cond) != 6:
                continue
            for pos_idx, model in enumerate(models_in_cond):
                pos = pos_idx + 1
                fam = extract_familiarity(model)
                pos_fams[pos].append(fam)

    print("\n Familiarity distribution by position (F/A) ")

    for pos in range(1, 7):
        fams = pos_fams[pos]
        counts = Counter(fams)
        total = sum(counts.values())
        if total == 0:
            print(f"Position {pos}: No data.")
            continue

        observed = [counts.get("F", 0), counts.get("A", 0)]
        expected = [total / 2.0, total / 2.0]

        chi2, p = chisquare(f_obs=observed, f_exp=expected)

        print(f"\nPosition {pos}:")
        print(f"  Counts: F={observed[0]}, A={observed[1]}, total={total}")
        print(f"  Chi-square = {chi2:.4f}, p = {p:.4f}")
        if p < 0.05:
            print("    -> Significant: Familiarity may NOT be 50/50 at this position.")
        else:
            print("    -> Not significant: Familiarity appears ~50/50 at this position.")


# Main

def main(folder_path):
    participants = load_all_orders(folder_path)
    print(f"Loaded {len(participants)} participants.")

    if not participants:
        print("No participants found. Check the folder path or CSV format.")
        return

    # 1. Model ↔ Condition
    test_model_condition_balance(participants)

    # 2. Model ↔ Position within each condition
    test_model_position_within_condition(participants)

    # 3. Complexity per position
    test_complexity_by_position(participants)

    # 4. Familiarity per position
    test_familiarity_by_position(participants)


if __name__ == "__main__":
    folder_path = r"C:\Users\johan\Desktop\data_pilot_Bricks_VR_2025\Model_Order_Data"
    main(folder_path)
