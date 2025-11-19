# Material Hardness & Oxidation Prediction
[![CI](https://github.com/TheComputationalCore/material-hardness-oxidation-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/TheComputationalCore/material-hardness-oxidation-prediction/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](https://github.com/your-org/your-repo) <!-- replace link -->
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Material Hardness & Oxidation Prediction** is a compact, production-ready project that predicts:
- **Hardness (numeric)** of metal samples (Linear Regression pipeline), and
- **Oxidation rate (numeric)** under given process settings (Random Forest pipeline).

This repository is structured for reproducibility, engineering quality and research clarity:
- Clean data in `data/`
- Training pipelines under `src/models/`
- Inference under `src/inference/`
- Flask web app under `src/app/`
- Models saved to `models/`
- Documentation under `docs/`
- Tests under `tests/`

---

## Quickstart (local)

### 1. Prerequisites
- Python 3.10
- A virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
