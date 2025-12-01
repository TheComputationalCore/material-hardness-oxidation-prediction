
# 🌌 **Material Hardness & Oxidation Prediction**
### ⚙️ *A Full-Stack, Research-Grade, Enterprise-Ready Materials Intelligence Platform*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/Framework-Flask-black?logo=flask">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-yellow?logo=scikitlearn">
  <img src="https://img.shields.io/badge/Explainability-SHAP-pink">
  <img src="https://img.shields.io/badge/Deployment-Render-teal">
  <img src="https://img.shields.io/badge/Testing-Pytest-purple">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

---

# 🚀 **Executive Summary**

This repository delivers a **state-of-the-art materials engineering intelligence system** combining:

- High-fidelity **microstructure → property prediction**
- Industrial-grade **ML pipeline automation**
- **SHAP-powered interpretability**
- Full-stack **web deployment**
- **Enterprise architecture**, modular and scalable

Designed for  
✔ Research labs  
✔ Surface engineering teams  
✔ Manufacturing AI groups  
✔ Academic courses  
✔ Industrial R&D centers  

---

# 🌐 **Live Demo**
**https://material-hardness-oxidation-prediction.onrender.com**

---

# 🧬 **Scientific Foundation**

Material hardness & oxidation behavior directly affect:

- Wear resistance  
- Creep strength  
- Oxidation kinetics  
- High-temperature reliability  
- Hardfacing performance  
- Microstructure evolution  

This system functions as an **AI surrogate model**, reducing expensive lab experimentation.

### 📘 Primary Reference  
C. Dinesh Chandra et al.  
*Experimental Studies of Stellite‑6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process*  
IOP Conf. Ser.: Mater. Sci. Eng. 998 (2020) 012061  
DOI: 10.1088/1757-899X/998/1/012061  

---

# 🏗 **Architecture Overview**

```
material-hardness-oxidation-prediction/
│
├── data/                     # Raw & cleaned datasets
├── docs/                     # Full documentation suite
├── models/                   # Trained models, SHAP dumps, metadata
│
├── src/
│   ├── app/                  # Flask UI + routes + templates
│   ├── inference/            # Prediction, validation, preprocessing
│   ├── models/               # Training pipelines + evaluation
│   └── utils/                # Logging, file I/O, helpers
│
├── screenshots/              # UI + SHAP visualization assets
├── tests/                    # Unit + integration tests
│
├── render.yaml               # Render deployment configuration
├── requirements.txt          # Python dependencies
├── Procfile                  # Gunicorn startup
└── runtime.txt               # Deployment runtime
```

---

# 🧩 **System Diagram**

```
               ┌─────────────────────────┐
               │  Web UI (Flask + Jinja) │
               └─────────────┬───────────┘
                             │
                             ▼
               ┌─────────────────────────┐
               │ Request Validator       │
               │ (Pydantic Schema)       │
               └─────────────┬───────────┘
                             │
                             ▼
               ┌─────────────────────────┐
               │ Preprocessing Pipeline  │
               │ Scaling, FE, Encoding   │
               └─────────────┬───────────┘
                             │
                ┌────────────┴────────────┐
                ▼                           ▼
    ┌──────────────────────┐     ┌────────────────────────┐
    │ Hardness Model       │     │ Oxidation Model         │
    │ (RF / LR)            │     │ (RF / LR)               │
    └────────────┬─────────┘     └───────────┬────────────┘
                 │                           │
                 └──────────────┬────────────┘
                                ▼
                 ┌──────────────────────────┐
                 │ SHAP Explainability Layer│
                 └──────────────┬───────────┘
                                ▼
                 ┌──────────────────────────┐
                 │ JSON Response + Visuals  │
                 └──────────────────────────┘
```

---

# 📊 **Exploratory Data Analysis (EDA)**

The system includes full EDA:

- Correlation heatmaps  
- Distribution plots  
- Feature coupling analysis  
- Material-process relationship mapping  
- Outlier detection  
- Feature variance reports  

---

# 📈 **Modeling Approach**

### **Algorithms Supported**
| Target | Models |
|--------|--------|
| Hardness | Linear Regression, Random Forest |
| Oxidation Rate | Linear Regression, Random Forest |

### **Explainability**
SHAP features included:

- Global importance  
- Local per‑sample breakdown  
- Feature interaction effects  
- Sensitivity analysis  

---

# 🧠 **Mathematical Formulation**

### **Hardness Prediction**
Given process variables \( X = \{x_1, x_2, ..., x_n\} \):

\[
\hat{H}(X) = f_{\theta}(X)
\]

Where \( f_{\theta} \) is the trained RF or LR model.

### **Oxidation Rate Prediction**

\[
\hat{O}(X) = g_{\phi}(X)
\]

Where \( g_{\phi} \) is an ML estimator capturing oxidation kinetics.

### **Loss Function (Regression)**

\[
\mathcal{L} = \frac{1}{N} \sum (y_i - \hat{y}_i)^2
\]

---

# 🧱 **Feature Definitions**

| Feature | Description |
|---------|-------------|
| Current | Arc energy / heat input |
| Voltage | Arc voltage |
| Speed | Travel speed |
| Thickness | Coating thickness |
| Temp | Operating temperature |
| Time | Oxidation duration |
| … | Additional derived features |

---

# 🛠 **Development Setup**

```bash
git clone https://github.com/TheComputationalCore/Material-Hardness-Oxidation-Prediction
cd Material-Hardness-Oxidation-Prediction
conda create -n mhoc python=3.10
conda activate mhoc
pip install -r requirements.txt
python src/app/app.py
```

Visit:
```
http://localhost:5000
```

---

# 🧪 **Testing Suite**

```bash
pytest -q
```

---

# 🚀 **Deployment Guide (Render)**

### Build
```
pip install -r requirements.txt
```

### Start
```
gunicorn "app.app:app" --chdir src --bind 0.0.0.0:$PORT --workers 2
```

---

# 📘 **Documentation Suite**

- `docs/MODEL_CARD.md`
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `docs/DATA_DICTIONARY.md`
- `docs/EXPERIMENTS.md`
- `docs/CHANGELOG.md`

---

# 🧾 **Academic Citation**

```
@article{materialHardnessOxidationAI,
  author    = {Dinesh Chandra},
  title     = {AI-driven Prediction of Hardness and Oxidation in Stellite-6 Coatings},
  journal   = {IOP Conference Series: Materials Science and Engineering},
  volume    = {998},
  year      = {2020},
  doi       = {10.1088/1757-899X/998/1/012061}
}
```

---

# 👤 **Author**

**Dinesh Chandra — TheComputationalCore**  
GitHub: https://github.com/TheComputationalCore  
YouTube: https://www.youtube.com/@TheComputationalCore  

---

# 📦 **License**
MIT License — free for academic, commercial, and industrial use.
