import pandas as pd
df = pd.read_csv("regression/run_eng_20k/training_data/training_data.csv")
print(df["initial_velocity_ms"].describe())
print(df[df["initial_velocity_ms"] > 200].shape[0], "suspicious rows")

df = pd.read_csv("regression/run_eng_20k/training_data/training_data.csv")
print(df[df["initial_velocity_ms"] < 0])