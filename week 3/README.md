# \# Week 3 — LearnPulse AI: Early Warning System

# 

# \## Objective

# 

# Develop and evaluate machine learning models that predict student academic outcomes and identify students who may need additional academic support.

# 

# \## Dataset

# 

# The project uses the Open University Learning Analytics Dataset (OULAD).

# 

# \- Original Week 2 processed dataset: 32,593 records

# \- Day-30 early-warning dataset: 27,613 records

# \- Target outcomes: Distinction, Fail, Pass, and Withdrawn

# \- Binary risk definition: Fail or Withdrawn = At Risk; Pass or Distinction = Not At Risk

# 

# The day-30 dataset contains early engagement and academic-history features. Feature availability and relative-date conventions must be validated before deployment.

# 

# \## Models Developed

# 

# \### 1. Four-Outcome Random Forest

# 

# Predicts the four recorded final outcomes.

# 

# \- Accuracy: 44.11%

# \- Balanced accuracy: 38.75%

# \- Macro F1-score: 38.73%

# 

# \### 2. Binary Early-Risk Random Forest

# 

# Predicts whether a student belongs to the project-defined at-risk category.

# 

# \- Accuracy: 68.73%

# \- Balanced accuracy: 68.00%

# \- At-risk recall: 60.13%

# \- Macro F1-score: 68.09%

# 

# The binary model improved accuracy by 14.09 percentage points over its majority-class baseline.

# 

# \## Evaluation

# 

# Evaluation includes grouped train-test splitting by student, grouped five-fold cross-validation, baseline comparisons, classification metrics, confusion matrices, and feature importance.

# 

# The binary model's strongest reported features were total clicks and active days within the early observation window.

# 

# \## Files

# 

# \- `build\_early\_warning\_dataset.py` — builds the day-30 dataset

# \- `early\_warning\_30day.csv` — generated dataset

# \- `train\_early\_warning\_model.py` — trains the four-outcome model

# \- `train\_risk\_model.py` — trains the binary risk model

# \- `compare\_current\_baseline.py` — compares the multiclass baseline

# \- `early\_warning\_metrics.txt` — multiclass evaluation results

# \- `early\_warning\_risk\_metrics.txt` — binary risk evaluation results

# \- `early\_warning\_model.joblib` — trained multiclass model

# \- `early\_warning\_risk\_model.joblib` — trained binary model

# \- `LearnPulse\_Model\_Training.ipynb` — training notebook

# \- `Learn Pulse AI - Week 3.docx` — original report, if retained

# \- `LearnPulse\_AI\_Week\_3\_Updated\_Report.docx` — updated report

# 

# \## Limitations and Responsible Use

# 

# The models make errors, including missed at-risk cases. Predictions must not be treated as definitive judgments or used to automatically penalize students. Validate feature timing, assess fairness across student groups, and require human review before any real-world use.

# 

# \## Future Work

# 

# \- Validate the day-30 prediction cutoff

# \- Improve classification and risk-threshold selection

# \- Evaluate fairness and explainability

# \- Integrate the model with the prediction API

# \- Test the system with appropriate monitoring and privacy safeguards



