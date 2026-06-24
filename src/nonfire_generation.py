import pandas as pd
import numpy as np

# Paths
FIRE_PATH = "data/interim/cleaned_fire.csv"
NONFIRE_OUTPUT = "data/interim/non_fire_samples.csv"
BALANCED_OUTPUT = "data/interim/balanced_dataset.csv"

def generate_nonfire_samples():

    print("Loading cleaned fire dataset...")
    fire_df = pd.read_csv(FIRE_PATH)

    print("Fire dataset shape:", fire_df.shape)

    # Jitter magnitude (approx ~20–30 km max)
    jitter_degree = 0.3

    nonfire_df = fire_df.copy()

    # Apply spatial jitter
    nonfire_df["latitude"] = nonfire_df["latitude"] + np.random.uniform(-jitter_degree, jitter_degree, len(nonfire_df))
    nonfire_df["longitude"] = nonfire_df["longitude"] + np.random.uniform(-jitter_degree, jitter_degree, len(nonfire_df))

    # Keep within India bounding box
    nonfire_df["latitude"] = nonfire_df["latitude"].clip(6, 37)
    nonfire_df["longitude"] = nonfire_df["longitude"].clip(68, 98)

    # Modify intensity slightly to create overlap but not identical
    nonfire_df["brightness"] = nonfire_df["brightness"] * np.random.uniform(0.85, 0.95, len(nonfire_df))
    nonfire_df["bright_t31"] = nonfire_df["bright_t31"] * np.random.uniform(0.90, 0.98, len(nonfire_df))
    nonfire_df["frp"] = nonfire_df["frp"] * np.random.uniform(0.5, 0.8, len(nonfire_df))

    # Assign non-fire label
    nonfire_df["label_occurrence"] = 0

    print("Non-fire dataset shape:", nonfire_df.shape)

    nonfire_df.to_csv(NONFIRE_OUTPUT, index=False)
    print("Non-fire samples saved.")

    # Merge
    balanced_df = pd.concat([fire_df, nonfire_df], ignore_index=True)

    print("Balanced dataset shape:", balanced_df.shape)

    balanced_df.to_csv(BALANCED_OUTPUT, index=False)
    print("Balanced dataset saved.")

if __name__ == "__main__":
    generate_nonfire_samples()
