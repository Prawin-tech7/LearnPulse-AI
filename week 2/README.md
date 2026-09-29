# Week 2: Data Collection and Preprocessing

## Goal

Document the OULAD data source and prepare a processed dataset for later analysis and model development.

## Data Source

The project uses the Open University Learning Analytics Dataset (OULAD). See [dataset source and setup notes](DATASET_SOURCE.md). Raw files are stored locally in `datasets/raw/` and are excluded from Git.

## Preprocessing

Run [`LearnPulse_Preprocessing.ipynb`](LearnPulse_Preprocessing.ipynb) from the repository root or this directory. The notebook loads student information, VLE activity, assessment, and registration files; explores data quality; creates assessment and engagement aggregates; encodes categorical columns; scales selected numeric columns; and writes the processed output.

## Output

- `week 2/processed_student_dataset.csv`

## Reproducibility Note

The notebook currently fits its encoders and scaler on the full dataset. For model evaluation, split the data first and fit preprocessing only on training data to avoid information leakage.

## Status

Data exploration and preprocessing notebook are available. Predictive modeling is planned for Week 3.