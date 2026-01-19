import pandas as pd

# Step 1: Load the original LimeSurvey CSV
input_path = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/IPQ_analysis/results-survey922256.csv"
df = pd.read_csv(input_path, encoding="utf-8", quotechar='"')

# Step 2: Define column prefixes for each questionnaire
ipq_prefix = "IPQ"
sus_prefix = "SUS"
model_prefixes = ["C1", "C2", "C3"]

# Step 3: Identify relevant columns
ipq_cols = [col for col in df.columns if col.startswith(ipq_prefix)]
sus_cols = [col for col in df.columns if col.startswith(sus_prefix)]
model_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in model_prefixes)]

# Step 4: Include metadata columns
metadata_cols = [col for col in df.columns if col.startswith("id.") or col.startswith("submitdate.")]

# Step 5: Create separate DataFrames
ipq_df = df[metadata_cols + ipq_cols]
sus_df = df[metadata_cols + sus_cols]
model_df = df[metadata_cols + model_cols]

# Step 6: Save each to a new CSV file and ake check print
output_dir = "C:/Users/johan/Desktop/data_pilot_Bricks_VR_2025/IPQ_analysis/"

ipq_df.to_csv(f"{output_dir}IPQ_responses.csv", index=False)
sus_df.to_csv(f"{output_dir}SUS_responses.csv", index=False)
model_df.to_csv(f"{output_dir}ModelPairs_responses.csv", index=False)

print("Files saved to:")
print(f"- {output_dir}IPQ_responses.csv")
print(f"- {output_dir}SUS_responses.csv")
print(f"- {output_dir}ModelPairs_responses.csv")
