import pandas as pd

# Upload CSV
df = pd.read_csv("instances_dataset_timefold.csv")

# Input features
input_features = [
    "num_employees",
    "num_tasks",
    "num_skills",
    "total_required_workload",
    "total_available_capacity",
    "capacity_ratio",
    "avg_candidates_per_task",
    "min_candidates_per_task",
    "fraction_single_candidate_tasks",
    "fraction_zero_candidate_tasks"
]

# Removing duplicates based on input features
df_unique = df.drop_duplicates(subset=input_features, keep="first")

# Save the result
df_unique.to_csv("instances_dataset_timefold_unique_inputs.csv", index=False)

print("There were entries:", len(df))
print("There are unique entries:", len(df_unique))
print("Duplicates removed:", len(df) - len(df_unique))