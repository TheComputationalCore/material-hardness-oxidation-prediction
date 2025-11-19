# Data README

This folder contains the curated training datasets used by the project.

## Files
- `hardness.csv` — training data for hardness model
  - Columns: `Material`, `Current`, `Heat_Input`, `Carbon`, `Manganese`, `Hardness`
  - `Material` values: `EN-8`, `Mild Steel`

- `oxidation.csv` — training data for oxidation model
  - Columns: `Material`, `Current`, `Heat_Input`, `Soaking_Time`, `Carbon`, `Manganese`, `Oxidation_Rate`

## Provenance
These CSVs were created from the original experimental logs (see `scripts/`) and curated for demonstration and training purposes. If you have full datasets, replace these and retrain.

## Usage
- Keep example rows small and representative in `data/` for the demo.
- For real research, add metadata describing measurement conditions, units, and raw data links.
