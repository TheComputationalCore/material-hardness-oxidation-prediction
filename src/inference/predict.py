"""
src/inference/predict.py
Model loader + prediction wrappers with robust path resolution and logging.
"""

import os
import joblib
import traceback

from src.models.pipelines import HARDNESS_FEATURES, OXIDATION_FEATURES
from src.inference.validator import (
    validate_hardness_input,
    validate_oxidation_input,
    to_dataframe,
    ValidationError
)

# Resolve top-level project path robustly (project_root/.../models)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # .../src/inference
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # top-level repo root
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

HARDNESS_MODEL_PATH = os.path.join(MODEL_DIR, "hardness_model.joblib")
OXIDATION_MODEL_PATH = os.path.join(MODEL_DIR, "oxidation_model.joblib")


def _safe_load_model(path):
    """Load a model and return (model, error). On success error is None."""
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        # Return None and error string (but also print full traceback to console)
        print(f"ERROR loading model at {path}: {e}")
        traceback.print_exc()
        return None, str(e)


# Load models once
hardness_model, hardness_load_error = _safe_load_model(HARDNESS_MODEL_PATH)
oxidation_model, oxidation_load_error = _safe_load_model(OXIDATION_MODEL_PATH)

if hardness_load_error:
    print("Hardness model load error:", hardness_load_error)
if oxidation_load_error:
    print("Oxidation model load error:", oxidation_load_error)


def predict_hardness(payload: dict):
    if hardness_model is None:
        return {"error": f"Hardness model not available. ({hardness_load_error})"}

    try:
        validated = validate_hardness_input(payload)
        df = to_dataframe(validated, HARDNESS_FEATURES)
        pred = hardness_model.predict(df)[0]
        return {"prediction": float(pred)}
    except ValidationError as ve:
        return {"error": f"Validation error: {ve}"}
    except Exception as e:
        # Log full traceback to console and return readable message to UI
        print("Exception during hardness prediction:", e)
        traceback.print_exc()
        return {"error": "Unexpected error during prediction. See server logs."}


def predict_oxidation(payload: dict):
    if oxidation_model is None:
        return {"error": f"Oxidation model not available. ({oxidation_load_error})"}

    try:
        validated = validate_oxidation_input(payload)
        df = to_dataframe(validated, OXIDATION_FEATURES)
        pred = oxidation_model.predict(df)[0]
        return {"prediction": float(pred)}
    except ValidationError as ve:
        return {"error": f"Validation error: {ve}"}
    except Exception as e:
        print("Exception during oxidation prediction:", e)
        traceback.print_exc()
        return {"error": "Unexpected error during prediction. See server logs."}
