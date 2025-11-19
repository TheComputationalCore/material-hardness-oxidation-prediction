"""
Evaluate trained models on their datasets.
"""

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model_path, df_path, label):
    model = joblib.load(model_path)
    df = pd.read_csv(df_path)

    X = df.drop(columns=[label])
    y = df[label]

    y_pred = model.predict(X)

    print("\nEvaluation Results:")
    print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y, y_pred, squared=False):.4f}")
    print(f"R2: {r2_score(y, y_pred):.4f}")


if __name__ == "__main__":
    evaluate("models/hardness_model.joblib", "data/hardness.csv", "Hardness")
    evaluate("models/oxidation_model.joblib", "data/oxidation.csv", "Oxidation_Rate")
