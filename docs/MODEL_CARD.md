
---

## `docs/MODEL_CARD.md`

```md
# Model Card — Material Hardness & Oxidation Prediction

## Model Details
- **Model names:** Hardness Model (LinearRegression pipeline), Oxidation Model (RandomForest pipeline)
- **Versioning:** saved in `models/*_metadata.json` after each training run
- **Date trained:** included in metadata file per training run

## Intended Use
- Predict mechanical hardness and oxidation rate for steel samples under controlled process conditions.
- Input features: `Material`, `Current`, `Heat_Input`, `Soaking_Time` (oxidation only), `Carbon`, `Manganese`.
- Typical users: materials scientists, process engineers, researchers evaluating relative trends.

## Not intended use
- Safety-critical control decisions (do not use without domain validation).
- Predictions outside the training domain (e.g., materials other than EN-8 or Mild Steel).
- Extrapolating to drastically different processing conditions than in the dataset.

## Performance
- Training and validation metrics are computed and printed by `src/models/train_*.py`.
- The exact metrics for the latest saved model are stored in `models/*_metadata.json`. Run the training scripts to reproduce.

## Data
- `data/hardness.csv` and `data/oxidation.csv` contain the training records used for the models.
- Each dataset includes the `Material` column with values `EN-8` and `Mild Steel`.
- See `data/README.md` for column descriptions and provenance.

## Factors affecting performance
- Small dataset size means models are likely to generalize poorly without more data.
- Feature distributions and interactions (e.g., Current × Heat_Input) affect accuracy.
- Models are sensitive to correct data scaling and the exact feature ordering — the inference pipeline enforces ordering.

## Ethical considerations and risk
- Ensure results are validated by domain experts before operational use.
- Log model predictions and monitor drift if used in production.

## Maintenance & reproducibility
- Train with `make train` and capture resulting metadata in `models/`.
- Use the same Python environment from `requirements.txt` for reproducibility.

## Contact
- Project author: (add your name & contact)
- Repository: (add GitHub link)
