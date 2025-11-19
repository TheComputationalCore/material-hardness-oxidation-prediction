# 🔥 Material Hardness & Oxidation Prediction  
### Intelligent Microstructure–Property Modeling for Materials Engineering  
**Author:** Dinesh Chandra  
**Live Demo:** https://your-deployment-url.com  
**Research Backing:**  
*[Machine learning–assisted prediction of mechanical properties of EN-8 alloy steel — IOP Conference Series](https://iopscience.iop.org/article/10.1088/1757-899X/998/1/012061)*

---

<div align="center">

![Home](screenshots/demo-01-home.png)

</div>

---

## 🚀 Overview  

This project delivers a **high-fidelity machine learning system** for predicting:

1. **Material Hardness**  
2. **Oxidation Rate**  

Using advanced data-driven modeling, automated feature validation, SHAP-based interpretability, and a modern browser interface, this system bridges **materials science** with **production-quality machine learning engineering**.

The application enables metallurgists, welding engineers, and researchers to:

- Predict microstructure-driven properties within seconds  
- Understand feature influence using explainable AI  
- Experiment with process conditions digitally  
- Accelerate materials + process optimization  

---

## 🧪 Scientific Foundation  

Hardness and oxidation behaviors heavily influence:

- Heat treatment outcomes  
- Failure mechanisms  
- Wear resistance  
- High-temperature reliability  
- Surface engineering performance  

Experimentation is **time-consuming and resource-heavy**.  
This ML-based platform provides a scientific surrogate model to accelerate experimentation and decision-making.

This work is connected to and inspired by the research:

**“Machine learning–assisted prediction of mechanical properties of EN-8 alloy steel” — Dinesh Chandra, IOP (2020).**  
https://doi.org/10.1088/1757-899X/998/1/012061

---

## 🏗 Architecture  

material-hardness-oxidation-prediction/
│
├── data/ # Datasets with documentation
├── models/ # Trained ML models + metadata
├── src/
│ ├── app/ # Flask app (UI, routes, templates)
│ ├── inference/ # Prediction & validation logic
│ ├── models/ # ML pipelines (training/eval)
│ └── utils/ # Config + shared utilities
│
├── screenshots/ # UI visuals
├── tests/ # pytest suite
├── requirements.txt
├── render.yaml
├── Procfile
└── runtime.txt

---

## 🌐 UI Preview  

### **Home Interface**
<div align="center">
<img src="screenshots/demo-01-home.png" width="750">
</div>

---

### **Prediction Workflow**
<div align="center">
<img src="screenshots/demo-02-predict.png" width="750">
</div>

---

### **Hardness Explainability (SHAP)**
<div align="center">
<img src="screenshots/demo-03-hardness-shap.png" width="750">
</div>

---

### **Oxidation Explainability (SHAP)**
<div align="center">
<img src="screenshots/demo-04-oxidation-shap.png" width="750">
</div>

---

## 📊 Exploratory Data Analysis (EDA)

<details>
<summary><strong>Click to expand EDA visualizations</strong></summary>

### Hardness Dataset
<div align="center">
<img src="src/app/static/plots/eda_hardness_correlation.png" width="420">
<img src="src/app/static/plots/eda_hardness_hist.png" width="420">
</div>

### Oxidation Dataset
<div align="center">
<img src="src/app/static/plots/eda_oxidation_correlation.png" width="420">
<img src="src/app/static/plots/eda_oxidation_hist.png" width="420">
</div>

</details>

---

## 📈 Model Performance & Diagnostics

<details>
<summary><strong>Click to expand performance plots</strong></summary>

### Hardness Model

<div align="center">
<img src="src/app/static/plots/perf_hardness_actual_vs_pred.png" width="420">
<img src="src/app/static/plots/perf_hardness_residuals.png" width="420">
<img src="src/app/static/plots/fi_hardness_coefficients.png" width="420">
</div>

### Oxidation Model

<div align="center">
<img src="src/app/static/plots/perf_oxidation_actual_vs_pred.png" width="420">
<img src="src/app/static/plots/perf_oxidation_residuals.png" width="420">
<img src="src/app/static/plots/fi_oxidation_importances.png" width="420">
</div>

</details>

---

## 🧠 Machine Learning Pipelines  

Each model includes:

- Data validation  
- Preprocessing & feature engineering  
- Scikit-learn pipelines  
- Hyperparameter tuning  
- SHAP attribution  
- Metadata for reproducibility  

Training scripts:

-src/models/train_hardness.py
-src/models/train_oxidation.py


Evaluation:


---

## 🛠 Setup (Local Development)

### **1. Clone repo**

           ```bash
              git clone https://github.com/TheComputationalCore/Material-Hardness-Oxidation-Prediction
              cd Material-Hardness-Oxidation-Prediction
           ```

  2. Environment

             conda create -n mhoc python=3.10
             conda activate mhoc
             pip install -r requirements.txt
   
  3. Run

     python src/app/app.py

     Open in browser:
                       http://localhost:5000

🧪 Testing

    pytest -q
    
   Tests cover:

    API routes

    Prediction logic

    Input schema

    Edge-case handling


🚀 Deployment on Render
       Build Command
                   pip install -r requirements.txt
       Start Command
                   gunicorn "app.app:app" --chdir src --bind 0.0.0.0:$PORT --workers 2

  📘 Documentation

        Located in /docs:

        MODEL_CARD.md — model specs, ethics, limitations

        ARCHITECTURE.md — diagrams

    📄 Citation

     Dinesh Chandra (2020),
     Prediction of mechanical properties of EN-8 alloy steel,
     IOP Conference Series: Materials Science and Engineering, 998 (1), 012061.
     Paper: https://doi.org/10.1088/1757-899X/998/1/012061

     ⭐ Author

     Dinesh Chandra
     Machine Learning Engineer & Materials Science Researcher

     📦 License

     MIT License — open for academic and professional use.
