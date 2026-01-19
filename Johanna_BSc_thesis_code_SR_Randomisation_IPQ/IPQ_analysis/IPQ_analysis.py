import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Step 1: Load the original IPQ CSV and define functions for analysis
input_path = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/IPQ_analysis/IPQ_responses.csv"
output_path = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/IPQ_analysis/IPQ_responses_named.csv"

def preprocess_ipq_data(filepath):
    """
    Loads and preprocesses IPQ data:
    - Matches verbose column headers to item names.
    - Inverts scores for SP2, SP3, and INV3.
    - Returns cleaned DataFrame and item map.
    """
    df = pd.read_csv(filepath, encoding="utf-8", quotechar='"')

    # Define short codes to match verbose headers
    item_codes = {
        "G1": "IPQ01[SQ001]",
        "SP1": "IPQ02[SQ001]",
        "SP2": "IPQ03[SQ001]",
        "SP3": "IPQ04[SQ001]",
        "SP4": "IPQ05[SQ001]",
        "SP5": "IPQ06[SQ001]",
        "INV1": "IPQ07[SQ001]",
        "INV2": "IPQ08[SQ001]",
        "INV3": "IPQ09[SQ001]",
        "INV4": "IPQ10[SQ001]",
        "REAL1": "IPQ11[SQ001]",
        "REAL2": "IPQ12[SQ001]",
        "REAL3": "IPQ13[SQ001]",
        "REAL4": "IPQ14[SQ001]"
    }

    # Match actual column names
    item_map = {}
    for label, code in item_codes.items():
        match = [col for col in df.columns if code in col]
        if match:
            item_map[label] = match[0]

    # Invert scores for SP2, SP3, INV3
    for item in ["SP2", "SP3", "INV3"]:
        col = item_map.get(item)
        if col in df.columns:
            df[col] = df[col] * -1

    return df, item_map

def analyze_ipq_data(df, item_map):
    """
    IPQ Data Analysis for a single study condition:
    (1) Invert SP2, SP3, INV3.
    (2) Compute mean & std for all 14 items.
    (3) Compute mean & std for each sub-scale.
    (4) Print descriptive stats and plot sub-scale means.
    """
    subscales = {
        "General Presence": ["G1"],
        "Spatial Presence": ["SP1", "SP2", "SP3", "SP4", "SP5"],
        "Involvement": ["INV1", "INV2", "INV3", "INV4"],
        "Realism": ["REAL1", "REAL2", "REAL3", "REAL4"]
    }

    # 1. Invert selected items
    invert_items = ["SP2", "SP3", "INV3"]
    for item in invert_items:
        if item in item_map:
            df[item_map[item]] = df[item_map[item]] * -1

    # 2. Select all item columns
    item_cols = [item_map[i] for i in item_map if i in item_map]

    # 3. Overall stats
    overall_mean = df[item_cols].mean().mean()
    overall_std = df[item_cols].stack().std()
    print(f"Overall Mean: {overall_mean:.2f}")
    print(f"Overall Std Dev: {overall_std:.2f}")

    # 4. Sub-scale stats
    subscale_means = {}
    for scale, items in subscales.items():
        cols = [item_map[i] for i in items if i in item_map]
        mean = df[cols].mean().mean()
        std = df[cols].stack().std()
        subscale_means[scale] = mean
        print(f"{scale}: Mean = {mean:.2f}, Std Dev = {std:.2f}")

    # 5. Plotting
    plt.figure(figsize=(8, 5))
    plt.bar(subscale_means.keys(), subscale_means.values(), color='steelblue')
    plt.ylabel("Mean Score")
    plt.title("IPQ Sub-scale Means – All Participants")
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def print_first_ipq_responses(filepath, item_map):
    """
    Loads the IPQ dataset and prints a table of responses
    given by the first participant for all IPQ items,
    labeled with their item names.
    """
    df = pd.read_csv(filepath, encoding="utf-8", quotechar='"')
    rows = []

    for item_id, col_name in item_map.items():
        if col_name in df.columns:
            value = df.iloc[0][col_name]
            rows.append((item_id, col_name, value))

    print("\n🧾 IPQ Responses – First Participant\n")
    print(pd.DataFrame(rows, columns=["Item Name", "Question", "Response"]))


def plot_ipq_likert_diverging(df, item_map):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    item_ids = list(item_map.keys())
    scores = df[[item_map[i] for i in item_ids]].copy()

    # Convert 1-7 Likert to -3 to 3
    if scores.max().max() > 3:
        scores = scores - 4

    likert_vals = np.arange(-3, 4)
    percentages = pd.DataFrame(index=item_ids, columns=likert_vals)

    # Compute percentages
    for item in item_ids:
        counts = scores[item_map[item]].value_counts().reindex(likert_vals, fill_value=0)
        percentages.loc[item] = counts / counts.sum() * 100

    plt.figure(figsize=(10, len(item_ids) * 0.5))
    colors = sns.color_palette("BuPu", n_colors=len(likert_vals))

    for i, val in enumerate(likert_vals):
        vals = percentages[val].astype(float)
        # Negative values plotted to the left
        left = np.zeros(len(item_ids))
        if val < 0:
            plt.barh(percentages.index, -vals, left=left, color=colors[i], edgecolor='white', label=str(val))
        else:
            plt.barh(percentages.index, vals, left=left, color=colors[i], edgecolor='white', label=str(val))

    plt.xlabel("Percentage (%)")
    plt.ylabel("IPQ Item")
    plt.title("Diverging IPQ Likert Responses")
    plt.legend(title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(-100, 100)
    plt.tight_layout()
    plt.show()


def create_vr_dataframe_and_spearman(df, item_map, vr_experience_list, save_path=None):
    """
    Creates a new DataFrame with VR experience and IPQ subscale & total means per participant,
    calculates Spearman correlations between VR experience and each IPQ score,
    and optionally saves the dataframe as CSV.

    Parameters:
        df: preprocessed IPQ DataFrame
        item_map: dictionary mapping item codes to column names
        vr_experience_list: list of VR experience values (length must match df)
        save_path: optional CSV path to save the dataframe
    Returns:
        df_summary: DataFrame with VR experience and subscale/total means
    """
    assert len(vr_experience_list) == len(df), "Length of VR experience list must match number of participants"

    subscales = {
        "General Presence": ["G1"],
        "Spatial Presence": ["SP1", "SP2", "SP3", "SP4", "SP5"],
        "Involvement": ["INV1", "INV2", "INV3", "INV4"],
        "Realism": ["REAL1", "REAL2", "REAL3", "REAL4"]
    }

    # Calculate means
    data = {
        "VR_experience": vr_experience_list,
    }

    # Total mean
    item_cols = [item_map[i] for i in item_map if i in item_map]
    data["IPQ_total_mean"] = df[item_cols].mean(axis=1)

    # Subscale means
    for scale, items in subscales.items():
        cols = [item_map[i] for i in items if i in item_map]
        data[f"{scale}_mean"] = df[cols].mean(axis=1)

    df_summary = pd.DataFrame(data)

    # Save dataframe if path provided
    if save_path:
        df_summary.to_csv(save_path, index=False)
        print(f"Summary dataframe saved to {save_path}")

    # Calculate Spearman correlations
    print("\nSpearman correlations (VR experience vs IPQ scores):")
    for col in df_summary.columns[1:]:  # skip VR_experience column
        rho, pval = spearmanr(df_summary["VR_experience"], df_summary[col])
        print(f"{col}: rho = {rho:.2f}, p = {pval:.3f}")

    return df_summary



# Step 2: Run preprocessing and analysis
df_clean, item_map = preprocess_ipq_data(input_path)
df_clean.to_csv(output_path, index=False)
print("IPQ column names matched and saved to IPQ_responses_named.csv")

df_named = pd.read_csv(output_path, encoding="utf-8", quotechar='"')
analyze_ipq_data(df_named, item_map)

# === Print first participant's responses with item names ===
print_first_ipq_responses(output_path, item_map)

plot_ipq_likert_diverging(df_named, item_map)

df_clean, item_map = preprocess_ipq_data(input_path)

# Hard-coded VR experience values for 10 participants (1 = little/no experience, 5 = a lot)
vr_experience_values = [2, 2, 2, 1, 2, 1, 1, 2, 3, 1]

df_vr_summary = create_vr_dataframe_and_spearman(
    df_clean,
    item_map,
    vr_experience_values,
    save_path="C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/IPQ_analysis/IPQ_vr_summary.csv"
)