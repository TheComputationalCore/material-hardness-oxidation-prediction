# tests/test_app.py
import json
from src.app.app import create_app

def test_index_route():
    app = create_app()
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    assert b"Predict" in res.data or b"Prediction" in res.data

def test_api_predict_json():
    app = create_app()
    client = app.test_client()
    payload = {
        "Material": "EN-8",
        "Current": 140,
        "Heat_Input": 0.864,
        "Soaking_Time": 10,
        "Carbon": 0.37,
        "Manganese": 0.8
    }
    res = client.post("/api/v1/predict", data=json.dumps(payload),
                      content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert "hardness" in data and "oxidation" in data
