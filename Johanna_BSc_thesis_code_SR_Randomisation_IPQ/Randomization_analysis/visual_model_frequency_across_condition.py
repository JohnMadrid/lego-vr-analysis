import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script loads all CSV files from a specified folder, combines them into a single
DataFrame, and visualizes how often each model appears across different experimental
conditions (excluding the training model). It produces a seaborn bar plot showing
model frequencies summed across participants.
"""

# settings for nice plot
label_fontsize = 16
title_fontsize = 18
tick_fontsize = 14
text_fontsize = 14
legend_fontsize = 12
sns.set_theme(context="talk", style="whitegrid")

# load data
def load_all_csv(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def plot_model_frequency_by_condition(df):
    # Filter out training model
    df_filtered = df[df["ModelName"] != "TM"]

    # Count occurrences of each model per condition
    counts = df_filtered.groupby(["ModelName", "ConditionNumber"]).size().reset_index(name="Frequency")

    # Plot with seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=counts,
        x="ModelName",
        y="Frequency",
        hue="ConditionNumber",
        palette=["powderblue", "lightskyblue", "cornflowerblue"],
        edgecolor = "black"
    )

    plt.title("Model Frequency by Condition (Summed Across Participants)", fontsize=title_fontsize)
    plt.xlabel("Model Name", fontsize=label_fontsize)
    plt.ylabel("Frequency", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    plt.legend(title="Condition Number", fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()


folder_path = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/Model_Order_Data"
df_all = load_all_csv(folder_path)
plot_model_frequency_by_condition(df_all)