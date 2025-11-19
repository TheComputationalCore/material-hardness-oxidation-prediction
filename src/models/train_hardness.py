"""
Train the hardness prediction model.
"""

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.pipelines import (
    build_hardness_pipeline,
    HARDNESS_FEATURES
)
from src.models.utils import load_csv, save_model, save_metadata


DATA_PATH = "data/hardness.csv"
MODEL_PATH = "models/hardness_model.joblib"
META_PATH = "models/hardness_metadata.json"


def train_hardness_model():
    df = load_csv(DATA_PATH)

    X = df[HARDNESS_FEATURES]
    y = df["Hardness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_hardness_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }

    print("\nHardness Model Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    save_model(pipeline, MODEL_PATH)
    save_metadata(META_PATH, "Hardness Model", HARDNESS_FEATURES, metrics)


if __name__ == "__main__":
    train_hardness_model()
