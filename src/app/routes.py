# src/app/routes.py
from flask import Blueprint, render_template, request, jsonify
from src.inference.predict import predict_hardness, predict_oxidation

app_bp = Blueprint("app_bp", __name__)


@app_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app_bp.route("/predict", methods=["POST"])
def predict():
    # Backwards-compatible form POST (server-rendered)
    material = request.form.get("Material")
    current = request.form.get("Current")
    heat_input = request.form.get("Heat_Input")
    soaking_time = request.form.get("Soaking_Time")
    carbon = request.form.get("Carbon")
    manganese = request.form.get("Manganese")

    # Build payloads
    hardness_payload = {
        "Material": material,
        "Current": current,
        "Heat_Input": heat_input,
        "Carbon": carbon,
        "Manganese": manganese
    }

    oxidation_payload = {
        "Material": material,
        "Current": current,
        "Heat_Input": heat_input,
        "Soaking_Time": soaking_time,
        "Carbon": carbon,
        "Manganese": manganese
    }

    hardness_result = predict_hardness(hardness_payload)
    oxidation_result = predict_oxidation(oxidation_payload)

    return render_template(
    "index.html",
    hardness=None,
    oxidation=None,
    hardness_error=None,
    oxidation_error=None,
    selected_material=None,
    form_data=None
)



# New JSON API for async UI
@app_bp.route("/api/v1/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # Expect payload to contain the fields we need; don't fail fast here,
    # predict_* will return sensible errors if validation fails
    hardness_result = predict_hardness(payload)
    oxidation_result = predict_oxidation(payload)

    response = {
        "hardness": hardness_result.get("prediction"),
        "oxidation": oxidation_result.get("prediction"),
        "hardness_error": hardness_result.get("error"),
        "oxidation_error": oxidation_result.get("error"),
    }

    # keep HTTP 200 even with prediction errors (client will display them)
    return jsonify(response), 200
