import pandas as pd

df = pd.read_csv("data/interim/balanced_dataset_multitask.csv")

print("\nIntensity Distribution:\n")
print(df["fire_intensity"].value_counts())