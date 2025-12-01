# 🌌 Material Hardness & Oxidation Prediction  
### **AI-Driven Microstructure–Property Intelligence Platform for Materials Engineering**  
A research-grade system integrating machine learning, explainable AI, scientific modeling, and fully modular production engineering.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Pipelines-FCC624?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/Explainability-SHAP-ff69b4?style=for-the-badge">
  <img src="https://img.shields.io/badge/Deployment-Render-46E3B7?style=for-the-badge&logo=render">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

# ⭐ Executive Summary
Material Hardness & Oxidation Prediction (**MHOC**) is a high-fidelity materials intelligence platform engineered for precise prediction of **hardness** and **oxidation rate** in ferrous alloys.  

It integrates:
- ML regression models  
- SHAP-based explainability  
- Scientific microstructure–property modeling  
- Production-grade modular architecture  
- Diagnostics, EDA, and a full web interface  

---

# 🔬 Scientific Foundation
Grounded in the peer-reviewed research:

**“Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process”**  
IOP Conference Series: Materials Science and Engineering  
Vol. 998 (2020), 012061  
doi:10.1088/1757-899X/998/1/012061  

This work provides empirical validation for heat input, microstructure, hardness, and oxidation behavior modeling.

---

# 🏗 System Architecture

```
                   ┌───────────────────────────┐
                   │     Web UI (Flask)        │
                   │  HTML • CSS • JS • Charts │
                   └───────────────┬───────────┘
                                   │
                         User Input Validation
                                   │
                   ┌───────────────▼──────────────┐
                   │   Inference Engine (Python)  │
                   │  Pydantic • Feature Builder  │
                   └───────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼──────────────────────────┐
         ▼                         ▼                          ▼
┌────────────────┐      ┌──────────────────┐       ┌──────────────────────┐
│ Hardness Model │      │ Oxidation Model  │       │   Metadata System    │
│ LinearReg / RF │      │ Random Forest    │       │ Versioning • Hashing │
└───────┬────────┘      └──────────┬──────-┘       └──────────┬───────────┘
        │                           │                        │
        └────────────┬──────────────┴──────────────┬─────────┘
                     ▼                             ▼
         ┌──────────────────-─┐          ┌────────────────────────┐
         │ SHAP Explainability│          │ Performance Diagnostics│
         │ Global + Local     │          │ Residuals • R² • MAE   │
         └───────────┬────────┘          └───────────┬────────────┘
                     ▼                             ▼
                   JSON                        UI Charts
                   Plots                       Reports
```

---

# 🖥️ UI Showcase

## Home Interface  
![Home](screenshots/demo-01-home.png)

## Prediction Workflow  
![Predict](screenshots/demo-02-predict.png)

## SHAP — Hardness  
![SHAP Hardness](screenshots/demo-03-hardness-shap.png)

## SHAP — Oxidation  
![SHAP Oxidation](screenshots/demo-04-oxidation-shap.png)

---

# 📊 Exploratory Data Analysis (EDA)

### Hardness Dataset  
![Hardness Corr](src/app/static/plots/eda_hardness_correlation.png)
![Hardness Hist](src/app/static/plots/eda_hardness_hist.png)

### Oxidation Dataset  
![Ox Corr](src/app/static/plots/eda_oxidation_correlation.png)
![Ox Hist](src/app/static/plots/eda_oxidation_hist.png)

---

# 📈 Model Performance & Diagnostics

### Hardness Model  
![Actual vs Pred](src/app/static/plots/perf_hardness_actual_vs_pred.png)
![Residuals](src/app/static/plots/perf_hardness_residuals.png)
![Feature Coefficients](src/app/static/plots/fi_hardness_coefficients.png)

### Oxidation Model  
![Actual vs Pred](src/app/static/plots/perf_oxidation_actual_vs_pred.png)
![Residuals](src/app/static/plots/perf_oxidation_residuals.png)
![Feature Importances](src/app/static/plots/fi_oxidation_importances.png)

---

# 🧠 Machine Learning Pipelines

## Feature Engineering
- Numerical scaling  
- Composition preprocessing  
- Heat input features  
- Outlier handling  
- Pipeline-based reproducibility  

## Models
| Task | Models |
|------|--------|
| Hardness | Linear Regression, Random Forest |
| Oxidation | Random Forest |

## Explainability
- SHAP global importance  
- SHAP local per-sample breakdown  

---

# 📐 Mathematical Formulation

### Hardness
\[
\hat{H} = f(X_{process}, X_{composition})
\]

### Oxidation Rate
\[
\hat{O} = g(T, t, X_{alloy})
\]

### Loss Function
\[
\mathcal{L} = \frac{1}{N} \sum (y_i - \hat{y}_i)^2
\]

---

# 🧩 Directory Structure (Complete)

```
material-hardness-oxidation-prediction/
├── data/
├── models/
├── screenshots/
├── src/
│   ├── app/
│   ├── inference/
│   ├── models/
│   └── utils/
├── tests/
├── requirements.txt
├── render.yaml
├── Procfile
└── runtime.txt
```

---

# ⚙️ Local Development

```
git clone https://github.com/TheComputationalCore/Material-Hardness-Oxidation-Prediction
cd Material-Hardness-Oxidation-Prediction
conda create -n mhoc python=3.10
conda activate mhoc
pip install -r requirements.txt
python src/app/app.py
```

---

# 🚀 Deployment (Render)

### Build  
```
pip install -r requirements.txt
```

### Start  
```
gunicorn "app.app:app" --chdir src --bind 0.0.0.0:$PORT --workers 2
```

---

# 🧪 Testing
```
pytest -q
```

---

# 📘 Documentation
- docs/MODEL_CARD.md  
- docs/ARCHITECTURE.md  
- docs/API_REFERENCE.md  

---

# 🧾 Citation

```
D. Chandra et al.
"Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process."
IOP Conference Series: Materials Science and Engineering,
Vol. 998, 012061, 2020.
doi:10.1088/1757-899X/998/1/012061
```

---

# 👤 Author
**Dinesh Chandra — TheComputationalCore**

---

# 🔒 License
MIT License  
