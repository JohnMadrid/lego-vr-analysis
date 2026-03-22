import os
import csv
from collections import defaultdict
from scipy.stats import chi2_contingency

"""
Analyzes whether model frequencies are evenly distributed across the three
experimental conditions. The script loads all model‑order CSV files, counts how
often each non‑tutorial model appears in each condition, merges counts across
participants, and applies a chi‑square test to detect deviations from a balanced
(random) distribution.
"""

def extract_model_counts(csv_path):
    """
    Extracts model counts per condition from a CSV file.
    Assumes columns: 'ConditionNumber', 'ModelName'
    """
    counts = defaultdict(lambda: defaultdict(int))  # model -> condition -> count
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = int(row['ConditionNumber'])
            model = row['ModelName']
            if model != 'TM':  # exclude tutorial model
                counts[model][cond] += 1
    return counts

def merge_counts(all_counts):
    """
    Merges model counts from multiple CSVs into one table.
    Returns: dict of model -> [count in cond 1, cond 2, cond 3]
    """
    merged = defaultdict(lambda: [0, 0, 0])
    for file_counts in all_counts:
        for model, cond_counts in file_counts.items():
            for cond, count in cond_counts.items():
                merged[model][cond - 1] += count
    return merged

def run_chi_square_test(merged_counts):
    """
    Runs chi-square test on model frequency across conditions.
    """
    contingency = [counts for counts in merged_counts.values()]
    chi2, p, dof, expected = chi2_contingency(contingency)
    return chi2, p, contingency

def main(folder_path):
    all_counts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            path = os.path.join(folder_path, filename)
            counts = extract_model_counts(path)
            all_counts.append(counts)

    merged_counts = merge_counts(all_counts)
    chi2, p, contingency = run_chi_square_test(merged_counts)

    print("\n Model Frequency Across Conditions")
    print(f"Chi-square statistic: {chi2:.6f}")
    print(f"p-value: {p:.4f}")
    if p < 0.05:
        print("️ Model frequency distribution is likely NOT balanced across conditions.")
    else:
        print(" Model frequency distribution appears balanced across conditions.")

    print("\n Model Counts Per Condition:")
    for model, counts in merged_counts.items():
        print(f"{model}: Condition 1 = {counts[0]}, Condition 2 = {counts[1]}, Condition 3 = {counts[2]}")

if __name__ == "__main__":
    folder_path = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/Model_Order_Data"
    main(folder_path)
