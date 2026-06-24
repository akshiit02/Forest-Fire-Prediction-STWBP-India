import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

print("Loading live satellite data...")
df = pd.read_csv("data/live_data.csv")

# -------------------------------------------------
# Filter India Bounding Box
# -------------------------------------------------
df = df[
    (df["latitude"] >= 6) & (df["latitude"] <= 37) &
    (df["longitude"] >= 68) & (df["longitude"] <= 98)
]

print("Total India points found:", len(df))

if len(df) == 0:
    print("No India data found.")
    exit()

# -------------------------------------------------
# 🔥 IMPORTANT FILTERS (REALISM FIX)
# -------------------------------------------------

# Remove low-confidence detections
if "confidence" in df.columns:
    df = df[df["confidence"] >= 50]

# Clip extreme sensor values
df["brightness"] = df["brightness"].clip(250, 450)
df["frp"] = df["frp"].clip(0, 500)

print("After filtering:", len(df))

# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------
df["acq_date"] = pd.to_datetime(df["acq_date"])
df["month"] = df["acq_date"].dt.month
df["dayofyear"] = df["acq_date"].dt.dayofyear

# -------------------------------------------------
# Features
# -------------------------------------------------
spatial_features = ["latitude", "longitude", "month"]
driver_features = ["brightness", "bright_t31", "frp", "scan", "track", "dayofyear"]

X_spatial = df[spatial_features]
X_driver = df[driver_features]

# -------------------------------------------------
# Load Scalers
# -------------------------------------------------
print("Loading scalers...")
spatial_scaler, driver_scaler = joblib.load("models/stwbp_scaler.pkl")

X_spatial_scaled = spatial_scaler.transform(X_spatial)
X_driver_scaled = driver_scaler.transform(X_driver)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
print("Loading trained STW-BP model...")
model = load_model("models/stwbp_model.h5")

# -------------------------------------------------
# Predict
# -------------------------------------------------
print("Running prediction on live data...")

fire_pred, intensity_pred = model.predict([X_spatial_scaled, X_driver_scaled])

fire_prob = fire_pred.flatten()

# 🔥 THRESHOLD FIX (VERY IMPORTANT)
fire_class = (fire_prob > 0.7).astype(int)

intensity_class = np.argmax(intensity_pred, axis=1) + 1

# -------------------------------------------------
# Store
# -------------------------------------------------
df["fire_probability"] = fire_prob
df["prediction"] = fire_class
df["intensity"] = intensity_class

# -------------------------------------------------
# Display
# -------------------------------------------------
print("\nSample Predictions:\n")

for i, row in df.head(10).iterrows():
    print(f"""
Location: ({row['latitude']:.2f}, {row['longitude']:.2f})
Prediction (0/1): {row['prediction']}
Fire Probability: {row['fire_probability']:.2f}
Intensity Level: {row['intensity']}
""")

print("\n🔥 Total Fire Points Detected:", int(df["prediction"].sum()))
print("🔥 Average Fire Probability:", float(df["fire_probability"].mean()))
print("🔥 Average Intensity:", float(df["intensity"].mean()))