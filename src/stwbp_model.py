import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================
# PATHS
# =========================

DATA_PATH = "data/interim/balanced_dataset_multitask.csv"
MODEL_PATH = "models/stwbp_model.h5"
SCALER_PATH = "models/stwbp_scaler.pkl"


def train_model():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    target_fire = "label_occurrence"
    target_intensity = "fire_intensity"

    # Convert intensity 1–5 → 0–4
    df[target_intensity] = df[target_intensity] - 1

    spatial_temporal_features = ["latitude", "longitude", "month"]
    driver_features = ["brightness", "bright_t31", "frp", "scan", "track", "dayofyear"]

    X_st = df[spatial_temporal_features]
    X_drivers = df[driver_features]

    y_fire = df[target_fire]
    y_intensity = df[target_intensity]

    # =========================
    # SPLIT
    # =========================

    X_st_train, X_st_test, X_drivers_train, X_drivers_test, y_fire_train, y_fire_test, y_int_train, y_int_test = train_test_split(
        X_st, X_drivers, y_fire, y_intensity,
        test_size=0.2,
        random_state=42,
        stratify=y_fire
    )

    # =========================
    # SCALING
    # =========================

    scaler_st = StandardScaler()
    scaler_drivers = StandardScaler()

    X_st_train = scaler_st.fit_transform(X_st_train)
    X_st_test = scaler_st.transform(X_st_test)

    X_drivers_train = scaler_drivers.fit_transform(X_drivers_train)
    X_drivers_test = scaler_drivers.transform(X_drivers_test)

    joblib.dump((scaler_st, scaler_drivers), SCALER_PATH)

    # =========================
    # MODEL
    # =========================

    st_input = Input(shape=(len(spatial_temporal_features),))
    x = Dense(128, activation="relu")(st_input)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)

    adaptive_weights = Dense(len(driver_features), activation="sigmoid")(x)

    driver_input = Input(shape=(len(driver_features),))
    weighted_drivers = Multiply()([adaptive_weights, driver_input])

    # 🔥 Improved network
    x2 = Dense(256, activation="relu")(weighted_drivers)
    x2 = Dropout(0.2)(x2)

    x2 = Dense(128, activation="relu")(x2)
    x2 = Dropout(0.2)(x2)

    x2 = Dense(64, activation="relu")(x2)

    # Outputs
    fire_output = Dense(1, activation="sigmoid", name="fire_output")(x2)
    intensity_output = Dense(5, activation="softmax", name="intensity_output")(x2)

    model = Model(inputs=[st_input, driver_input], outputs=[fire_output, intensity_output])

    # =========================
    # 🔥 CORRECT LOSS WEIGHTS
    # =========================

    loss_weights = {
        "fire_output": 1.5,      # ↑ focus on fire
        "intensity_output": 1.0
    }

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss={
            "fire_output": "binary_crossentropy",
            "intensity_output": "sparse_categorical_crossentropy"
        },
        loss_weights=loss_weights,
        metrics={
            "fire_output": ["accuracy"],
            "intensity_output": ["accuracy"]
        }
    )

    # =========================
    # CALLBACKS
    # =========================

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )

    # =========================
    # TRAIN
    # =========================

    print("Training model...")

    model.fit(
        [X_st_train, X_drivers_train],
        [y_fire_train, y_int_train],
        validation_split=0.2,
        epochs=80,
        batch_size=128,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # =========================
    # EVALUATION
    # =========================

    fire_pred, intensity_pred = model.predict([X_st_test, X_drivers_test])

    fire_prob = fire_pred.flatten()

    # 🔥 IMPORTANT FIX (threshold)
    fire_class = (fire_prob > 0.5).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_fire_test, fire_class))

    print("\n🔥 FIRE CLASSIFICATION REPORT:\n")
    print(classification_report(y_fire_test, fire_class))

    print("ROC-AUC:", roc_auc_score(y_fire_test, fire_prob))

    intensity_class = np.argmax(intensity_pred, axis=1)

    print("\n🔥 INTENSITY ACCURACY:")
    print(accuracy_score(y_int_test, intensity_class))

    # =========================
    # SAVE
    # =========================

    model.save(MODEL_PATH)
    print("Model saved.")


if __name__ == "__main__":
    train_model()