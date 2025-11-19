# API Reference — Material Hardness & Oxidation Predictor

Base URL (local): `http://localhost:5001`

## Public pages
- `GET /`  
  Renders the UI form for inputting sample parameters and shows predictions.

## Programmatic endpoints
### `POST /predict` (form POST)
- Description: HTML form POST that returns rendered page with predictions.
- Request fields (form fields):
  - `Material` (string): "EN-8" or "Mild Steel"
  - `Current` (number)
  - `Heat_Input` (number)
  - `Soaking_Time` (number) — required for oxidation predictions (ignored by hardness)
  - `Carbon` (number)
  - `Manganese` (number)

### `POST /api/v1/predict` (JSON)
- Description: Lightweight JSON API for programmatic use (if implemented).
- Request example:
```json
{
  "Material": "EN-8",
  "Current": 140,
  "Heat_Input": 0.864,
  "Soaking_Time": 10,
  "Carbon": 0.37,
  "Manganese": 0.8
}
