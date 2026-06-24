import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Paths
DATA_PATH = "data/interim/balanced_dataset.csv"
MODEL_PATH = "models/random_forest.pkl"

def train_random_forest():

    print("Loading balanced dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # Select features
    features = [
        "latitude",
        "longitude",
        "month",
        "dayofyear",
        "brightness",
        "bright_t31",
        "frp",
        "scan",
        "track"
    ]

    X = df[features]
    y = df["label_occurrence"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    print("Model trained.")

    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\nEvaluation Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=features)
    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False))

    # Save model
    joblib.dump(rf, MODEL_PATH)
    print("\nRandom Forest model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train_random_forest()
