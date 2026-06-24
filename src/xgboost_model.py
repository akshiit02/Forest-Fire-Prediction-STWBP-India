import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier

# Paths
DATA_PATH = "data/interim/balanced_dataset.csv"
MODEL_PATH = "models/xgboost_model.pkl"


def train_xgboost():

    print("Loading balanced dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # Select features (SAME as RF)
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

    # Train-test split (SAME as RF)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training XGBoost...")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)

    print("Model trained.")

    # Predictions
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

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
    importance = pd.Series(xgb.feature_importances_, index=features)
    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False))

    # Save model
    joblib.dump(xgb, MODEL_PATH)
    print("\nXGBoost model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_xgboost()