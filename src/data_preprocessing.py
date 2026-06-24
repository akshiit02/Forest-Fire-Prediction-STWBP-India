import pandas as pd
import numpy as np
import os

# Define paths
RAW_PATH = "data/raw/fire_dataset_2012_2019.csv"
OUTPUT_PATH = "data/interim/cleaned_fire.csv"


def clean_fire_data():

    print("Loading dataset...")
    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)

    # Filter high-confidence fires
    df = df[df["confidence"] >= 80].copy()
    print("After confidence filter:", df.shape)

    # Create datetime column
    df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)

    df["datetime"] = pd.to_datetime(
        df["acq_date"] + " " + df["acq_time"],
        format="%Y-%m-%d %H%M"
    )

    # Extract time features
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print("After duplicate removal:", df.shape)

    # Clip outliers
    for col in ["brightness", "bright_t31", "frp"]:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(low, high)

    print("Outliers handled.")

    # Add label
    df["label_occurrence"] = 1

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print("Cleaned dataset saved at:", OUTPUT_PATH)
    print("Final shape:", df.shape)

if __name__ == "__main__":
    clean_fire_data()
