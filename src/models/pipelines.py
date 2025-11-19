"""
pipelines.py — defines sklearn pipelines for hardness and oxidation models.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# Feature definitions
HARDNESS_FEATURES = ["Material", "Current", "Heat_Input", "Carbon", "Manganese"]
OXIDATION_FEATURES = ["Material", "Current", "Heat_Input", "Soaking_Time", "Carbon", "Manganese"]

# Columns
categorical_cols = ["Material"]
numeric_cols_hardness = ["Current", "Heat_Input", "Carbon", "Manganese"]
numeric_cols_oxidation = ["Current", "Heat_Input", "Soaking_Time", "Carbon", "Manganese"]

# Preprocessor for hardness
preprocessor_hardness = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols_hardness),
    ]
)

# Preprocessor for oxidation
preprocessor_oxidation = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols_oxidation),
    ]
)


def build_hardness_pipeline():
    """Return full sklearn pipeline for hardness prediction."""
    return Pipeline(
        steps=[
            ("preprocess", preprocessor_hardness),
            ("model", LinearRegression())
        ]
    )


def build_oxidation_pipeline():
    """Return full sklearn pipeline for oxidation rate prediction."""
    return Pipeline(
        steps=[
            ("preprocess", preprocessor_oxidation),
            ("model", RandomForestRegressor(random_state=42))
        ]
    )
