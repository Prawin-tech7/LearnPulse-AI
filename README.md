# LearnPulse AI

LearnPulse AI is an early warning system project exploring machine learning approaches to identify student disengagement and dropout risk from educational data.

## Project Goals

- Explore student engagement and academic outcome data.
- Prepare documented, reproducible features for analysis.
- Develop and evaluate risk prediction models in later project phases.
- Treat predictions as decision support for human review, not as automated decisions about students.

## Project Phases

1. [Week 1: Project Planning](week%201/README.md)
2. [Week 2: Data Collection and Preprocessing](week%202/README.md)
3. Week 3: Model Development (planned)
4. Week 4: Model Evaluation (planned)

## Repository Layout

- `week 1/`: project planning and planning report.
- `week 2/`: dataset source notes, preprocessing notebook, and processed dataset.

## Data and Reproducibility

The project uses the Open University Learning Analytics Dataset (OULAD). Raw dataset files are not included in this repository; see [the dataset notes](week%202/DATASET_SOURCE.md) for the source and local setup. Follow the dataset's license and usage terms. Do not commit personal, restricted, or sensitive data.

The preprocessing notebook and its output are documented in [Week 2](week%202/README.md). Raw dataset files belong in the local, Git-ignored `datasets/raw/` directory.

## Current Status

Weeks 1 and 2 contain the current planning and preprocessing work. Model development and evaluation are planned; no model performance claims are made here.

## Responsible Use

Risk estimates can be wrong and may reflect limitations or biases in the source data. Any future system should be evaluated for errors and fairness, and used only to support appropriate human review.
