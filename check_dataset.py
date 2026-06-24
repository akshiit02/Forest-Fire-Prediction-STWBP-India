import pandas as pd

df = pd.read_csv("data/interim/balanced_dataset.csv")

print("Columns:")
print(df.columns)

print("\nUnique values in 'type':")
print(df["type"].unique())
