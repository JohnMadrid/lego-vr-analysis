import os
import pandas as pd
import re

"""
This script validates experimental model‑order CSV files by checking whether each
participant's data follows the required structure and constraints. For every CSV
in the specified folder, the script verifies:

1. Model occurrence rules:
   - The training model (TM) appears exactly three times.
   - TM appears at the first position of each condition.
   - All non‑TM models appear exactly once.

2. Resource brick matching:
   - Each model is paired with the correct corresponding resource brick.
   - TM is always paired with TR.

3. Complexity and type distribution:
   - Each condition contains exactly two models per complexity level (C1, C2, C3).
   - Each level contains one abstract (A) and one familiar (F) model.

The script prints a PASS/FAIL summary for each participant in each CSV file.
"""

# Entry point: loops through all CSV files in the specified folder
def validate_csv_files(folder_path):
    print("\n CSV Validation Summary:")
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)
            print(f"\n File: {filename}")
            validate_participants(df)

# Validates all data for each participant in the CSV
def validate_participants(df):
    grouped = df.groupby("ParticipantCode")
    for participant, group in grouped:
        print(f"  Participant: {participant}")
        results = {}

        # Run each check and store result
        results["check_model_occurrences"] = check_model_occurrences(group)
        results["check_resource_bricks"] = check_resource_bricks(group)
        results["check_complexity_and_type_distribution"] = check_complexity_and_type_distribution(group)

        # Print results
        print("    check_model_occurrences: TM appears exactly 3 times →", "PASS" if results["check_model_occurrences"]["tm_count"] else "FAIL")
        print("    check_model_occurrences: TM appears at position 0 of each condition →", "PASS" if results["check_model_occurrences"]["tm_positions"] else "FAIL")
        print("    check_model_occurrences: All models appear only once →", "PASS" if results["check_model_occurrences"]["unique_models"] else "FAIL")

        print("    check_resource_bricks: Each model has matching resource bricks →", "PASS" if results["check_resource_bricks"]["matching_bricks"] else "FAIL")
        print("    check_resource_bricks: TM is always paired with TR →", "PASS" if results["check_resource_bricks"]["tm_with_tr"] else "FAIL")

        print("    check_complexity_and_type_distribution: 2 models per complexity level →", "PASS" if results["check_complexity_and_type_distribution"]["complexity_count"] else "FAIL")
        print("    check_complexity_and_type_distribution: 1 abstract and 1 familiar per level →", "PASS" if results["check_complexity_and_type_distribution"]["type_balance"] else "FAIL")

# Validates model occurrence rules
def check_model_occurrences(group):
    model_counts = group["ModelName"].value_counts()
    tm_positions = group[group["ModelName"] == "TM"]["Order"].tolist()

    result = {
        "tm_count": model_counts.get("TM", 0) == 3,
        "tm_positions": all(pos % 7 == 0 for pos in tm_positions),
        "unique_models": all(count == 1 for model, count in model_counts.items() if model != "TM")
    }
    return result

# Validates resource brick matching
def check_resource_bricks(group):
    result = {
        "matching_bricks": True,
        "tm_with_tr": True
    }
    for _, row in group.iterrows():
        model = row["ModelName"]
        resource = row["ResourceBrickName"]
        if model == "TM":
            if resource != "TR":
                result["tm_with_tr"] = False
        else:
            model_match = re.match(r"^C([1-3])M([1-6])([AF])$", model)
            resource_match = re.match(r"^C([1-3])R([1-6])([AF])$", resource)
            if not model_match or not resource_match or model_match.groups() != resource_match.groups():
                result["matching_bricks"] = False
    return result

# Validates complexity and type distribution
def check_complexity_and_type_distribution(group):
    result = {
        "complexity_count": True,
        "type_balance": True
    }
    grouped_conditions = group[group["ModelName"] != "TM"].groupby("ConditionNumber")

    for condition, trials in grouped_conditions:
        map = {1: {"A": 0, "F": 0}, 2: {"A": 0, "F": 0}, 3: {"A": 0, "F": 0}}
        for model in trials["ModelName"]:
            match = re.match(r"^C([1-3])M\d+([AF])$", model)
            if not match:
                continue
            c = int(match.group(1))
            t = match.group(2)
            map[c][t] += 1
        for level in [1, 2, 3]:
            if sum(map[level].values()) != 2:
                result["complexity_count"] = False
            if map[level]["A"] != 1 or map[level]["F"] != 1:
                result["type_balance"] = False
    return result

# Run validation on all CSV files in the folder
folder_path = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/Model_Order_Data"
validate_csv_files(folder_path)
