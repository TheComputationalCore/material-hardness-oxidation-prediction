"""
validator.py
Validates incoming prediction data.
"""

import pandas as pd

HARDNESS_FEATURES = ["Material", "Current", "Heat_Input", "Carbon", "Manganese"]
OXIDATION_FEATURES = ["Material", "Current", "Heat_Input", "Soaking_Time", "Carbon", "Manganese"]


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


def validate_material(mat):
    """
    Material must be one of the two valid categories.
    Return the original string (not encoded) because the pipeline's
    OneHotEncoder expects the string categories.
    """
    mapping = {"EN-8", "Mild Steel"}
    if mat not in mapping:
        raise ValidationError(
            f"Invalid material '{mat}'. Must be one of: {sorted(mapping)}"
        )
    return mat  # return string, not integer


def validate_numeric(name, value):
    """
    Validate a single numeric input.
    """
    if value is None or value == "":
        raise ValidationError(f"Missing required value for '{name}'")

    try:
        return float(value)
    except ValueError:
        raise ValidationError(f"Field '{name}' must be numeric. Got '{value}'")


def validate_hardness_input(data):
    """
    Validate inputs for hardness prediction.
    Returns sanitized dictionary ready for DataFrame construction.
    """
    validated = {}

    validated["Material"] = validate_material(data.get("Material"))
    validated["Current"] = validate_numeric("Current", data.get("Current"))
    validated["Heat_Input"] = validate_numeric("Heat_Input", data.get("Heat_Input"))
    validated["Carbon"] = validate_numeric("Carbon", data.get("Carbon"))
    validated["Manganese"] = validate_numeric("Manganese", data.get("Manganese"))

    return validated


def validate_oxidation_input(data):
    """
    Validate inputs for oxidation prediction.
    """
    validated = {}

    validated["Material"] = validate_material(data.get("Material"))
    validated["Current"] = validate_numeric("Current", data.get("Current"))
    validated["Heat_Input"] = validate_numeric("Heat_Input", data.get("Heat_Input"))
    validated["Soaking_Time"] = validate_numeric("Soaking_Time", data.get("Soaking_Time"))
    validated["Carbon"] = validate_numeric("Carbon", data.get("Carbon"))
    validated["Manganese"] = validate_numeric("Manganese", data.get("Manganese"))

    return validated


def to_dataframe(data: dict, feature_order: list):
    """
    Convert validated dictionary to DataFrame with correct column ordering.
    """
    return pd.DataFrame([[data[col] for col in feature_order]], columns=feature_order)
