import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================
# PATHS
# ==============================

DATA_PATH = "data/interim/balanced_dataset.csv"
MODEL_PATH = "models/stwbp_model.h5"
SCALER_PATH = "models/scaler.pkl"

# ==============================
# ANALYSIS FUNCTION
# ==============================

def analyze_weights():
    print("Loading dataset and model...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # -------- Auto detect target --------
    possible_targets = ["label", "fire", "class", "target", "is_fire"]

    target = None
    for col in possible_targets:
        if col in df.columns:
            target = col
            break

    if target is None:
        raise ValueError("No valid target column found.")

    print("Using target column:", target)

    spatial_temporal_features = ["latitude", "longitude", "dayofyear"]
    driver_features = ["brightness", "bright_t31", "frp", "scan", "track", "month"]

    X_st = df[spatial_temporal_features]
    X_driver = df[driver_features]

    # Scale spatial-temporal features
    X_st_scaled = scaler.transform(X_st)

    # -------- Get adaptive weights layer --------
    weight_layer_model = load_model(MODEL_PATH)

    # Extract adaptive weight layer output
    from tensorflow.keras.models import Model
    intermediate_model = Model(
        inputs=weight_layer_model.inputs,
        outputs=weight_layer_model.get_layer("adaptive_weights").output
    )

    weights = intermediate_model.predict([X_st_scaled, X_driver])

    weight_df = pd.DataFrame(weights, columns=driver_features)

    # -------- Region Creation --------
    def assign_region(lat):
        if lat > 28:
            return "Himalayan"
        elif lat > 22:
            return "Central India"
        elif lat > 18:
            return "Western Ghats"
        else:
            return "North-East"

    weight_df["region"] = df["latitude"].apply(assign_region)

    # Keep only numeric columns
    numeric_cols = weight_df.select_dtypes(include=[np.number]).columns
    region_means = weight_df.groupby("region")[numeric_cols].mean()

    print("\nMean Adaptive Weights by Region:")
    print(region_means)

    region_means.to_csv("results/regional_weight_analysis.csv")
    print("\nRegional weight analysis saved.")


if __name__ == "__main__":
    analyze_weights()
