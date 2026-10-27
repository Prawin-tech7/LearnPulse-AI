# LearnPulse AI — Week 6: AI Ethics, Bias Analysis and Fairness Assessment

## 1. Overview

LearnPulse AI is an early-warning system project that investigates student engagement and academic outcome prediction using machine learning.

Week 6 evaluates the ethical risks, predictive performance, and group-level fairness of a Random Forest classifier. Two complementary audits are conducted:

1. **Initial fairness audit:** Evaluates the model using the existing processed student dataset.
2. **Temporal fairness audit:** Reconstructs engagement and assessment features using information available up to a day-60 prediction cutoff.

The objective is to identify limitations and potential disparities before considering the system for educational support. The model is experimental and must not be used as the sole basis for decisions about individual students.

## 2. Audit objectives

- Evaluate classification performance across Distinction, Fail, Pass, and Withdrawn outcomes.
- Compare recall, precision, false-positive rates, and false-negative rates across recorded gender and disability groups.
- Reduce train/test contamination by splitting records according to student ID.
- Investigate potential temporal leakage in the existing processed dataset.
- Document ethical risks, limitations, and responsible-use recommendations.

## 3. Project structure

```text
LearnPulse-AI/
├── datasets/
│   └── raw/
│       ├── studentInfo.csv
│       ├── studentVle.csv
│       ├── studentAssessment.csv
│       ├── assessments.csv
│       └── studentRegistration.csv
├── week 2/
│   └── processed_student_dataset.csv
└── week 6/
    ├── week6_inspect.py
    ├── week6_labels.py
    ├── week6_verify.py
    ├── week6_fairness_audit.py
    ├── week6_temporal_audit.py
    ├── classification_report.txt
    ├── fairness_metrics.csv
    ├── confusion_matrix.png
    ├── recall_by_gender.png
    ├── recall_by_disability.png
    ├── audit_metadata.json
    ├── temporal_classification_report.txt
    ├── temporal_fairness_metrics.csv
    ├── temporal_confusion_matrix.png
    ├── temporal_audit_metadata.json
    └── README.md
```

## 4. Technology stack

- Python
- pandas and NumPy for data preparation and calculations
- scikit-learn for model training, preprocessing, splitting, and evaluation
- Matplotlib for visualizations
- JSON and CSV files for audit metadata and metrics

## 5. Setup and execution

Run the commands from the LearnPulse-AI project root. Activate the Python environment used for the project first.

```bat
python -m pip install pandas numpy matplotlib scikit-learn
```

Run the supporting inspection and verification scripts:

```bat
python "week 6\week6_inspect.py"
python "week 6\week6_labels.py"
python "week 6\week6_verify.py"
```

Run the initial fairness audit:

```bat
python "week 6\week6_fairness_audit.py"
```

Run the temporal audit:

```bat
python "week 6\week6_temporal_audit.py"
```

**Important:** Both audit scripts currently define the project root using a Windows absolute path. If the project is stored elsewhere, update the `ROOT` path in each script before execution. The raw OULAD files and the processed Week 2 dataset must be available at the paths expected by the scripts.

The audit scripts regenerate their corresponding reports, CSV files, charts, and metadata.

## 6. Methodology

### Initial fairness audit

The initial audit loads `week 2/processed_student_dataset.csv`, uses `final_result` as the target, and excludes `id_student` from the model features.

A Random Forest classifier with 100 trees and random state 42 is trained. `GroupShuffleSplit` creates a 80/20 train/test split grouped by student ID, and the script checks that no student ID appears in both sets.

This audit evaluates the existing feature representation. Because the original preprocessing aggregates student assessment and engagement information without a day-60 cutoff, some features may contain information unavailable at an early prediction time.

### Temporal fairness audit

The temporal audit rebuilds features from the raw data using a cutoff of 60 days after course start.

It includes:
- Total and average VLE clicks recorded on or before day 60.
- Mean assessment score and assessment count from submissions dated on or before day 60.
- Student and course attributes used by the model.

Course records with an unregistration date on or before day 60 are excluded. Records without an unregistration date, or with a later unregistration date, remain eligible.

The audit groups records by student, module, and presentation when constructing engagement and assessment features. A scikit-learn pipeline fits one-hot encoding on training data and applies the trained transformation to test data.

The eventual `final_result` remains the prediction target. The temporal audit is therefore a retrospective experiment using early features to predict eventual outcomes; it is not proof of a fully validated operational early-warning system.

## 7. Initial audit results

The initial audit evaluated 6,471 test records from a dataset of 32,593 records.

| Metric | Result |
|---|---:|
| Test accuracy | 69.17% |
| Distinction recall | 45% |
| Fail recall | 38% |
| Pass recall | 90% |
| Withdrawn recall | 73% |
| Student ID overlap between train and test | 0 |

The model performs substantially better at identifying Pass outcomes than Distinction and Fail outcomes. The aggregate accuracy does not capture these class-level differences.

## 8. Temporal audit results

The temporal audit retained 26,361 eligible course records and evaluated 5,294 test records.

| Metric | Result |
|---|---:|
| Prediction cutoff | Day 60 |
| Test accuracy | 57.14% |
| Distinction recall | 21% |
| Fail recall | 50% |
| Pass recall | 85% |
| Withdrawn recall | 11.2% |
| Student ID overlap between train and test | 0 |

The model correctly identified 90 of 801 actual Withdrawn cases and missed 711. Its low withdrawal recall is a major limitation: it would fail to identify many eventual withdrawal cases in this test set.

The accuracy difference between the two audits must not be attributed entirely to temporal leakage. The eligible cohorts and feature-construction methods differ, so the results are not a controlled comparison isolating one cause.

## 9. Fairness findings

The initial audit found differences in recall across recorded demographic groups. For example, Distinction recall was approximately 35.6% for the group encoded as F and 52.2% for the group encoded as M.

In the temporal audit, Withdrawn recall was approximately:
- 12.8% for F and 10.1% for M.
- 9.2% for N and 23.5% for Y in the recorded disability attribute.

The disability-Y group contained 501 test records, compared with 4,793 for disability-N. These observed differences warrant further investigation, including uncertainty estimates and repeated validation.

Differences in measured performance do not by themselves prove discrimination or explain its cause. Group membership, sample size, data quality, outcome prevalence, and feature behavior all require careful examination.

## 10. Ethical risks and limitations

- **Temporal leakage:** Features derived from the complete course may not be available at the intended prediction time.
- **Unequal error rates:** Poor recall for particular outcomes or groups can lead to unequal access to timely support.
- **False reassurance:** A student not flagged by the model may still need help.
- **Privacy:** Student records and sensitive attributes must be handled with appropriate access controls.
- **Automation bias:** Staff may treat a model prediction as a definitive judgment.
- **Limited validation:** The current results use one grouped train/test split and do not establish performance across other cohorts or institutions.
- **Target interpretation:** The Withdrawn label represents the dataset's recorded course outcome and should not automatically be treated as a verified dropout event.
- **Missing data assumptions:** Replacing missing early activity or assessment aggregates with zero may conflate no recorded activity with unavailable information.
- **Cross-course feature mixing:** The original preprocessing notebook aggregates assessment and VLE features by `id_student` alone. Because a student may have multiple course records, features may combine information across different modules or presentations. The effect of this on the model's predictions has not yet been quantified.
- **Preprocessing leakage:** The original notebook fits `MinMaxScaler` before the Week 3 train/test split. Consequently, the scaling parameters are calculated using the full dataset, including test records. The impact on the reported model performance has not been isolated.

## 11. Recommendations

1. Improve withdrawal detection and investigate class-specific errors before considering deployment.
2. Validate feature availability at the prediction cutoff and test for leakage.
3. Repeat evaluations with multiple grouped splits or suitable cross-validation.
4. Calculate confidence intervals for group metrics and investigate small-group uncertainty.
5. Assess data quality and whether group differences persist after accounting for relevant context.
6. Keep a human review process and use predictions only to support voluntary, constructive student outreach.
7. Protect student privacy and avoid punitive decisions based solely on model outputs.
8. Monitor model performance and fairness over time before any real-world use.

## 12. Output files

| File | Purpose |
|---|---|
| `classification_report.txt` | Initial overall metrics and confusion matrix |
| `fairness_metrics.csv` | Initial class-level group metrics |
| `confusion_matrix.png` | Initial classification error visualization |
| `recall_by_gender.png` | Initial recall comparison by gender |
| `recall_by_disability.png` | Initial recall comparison by disability |
| `audit_metadata.json` | Initial dataset, split, and model metadata |
| `temporal_classification_report.txt` | Temporal overall metrics and confusion matrix |
| `temporal_fairness_metrics.csv` | Temporal class-level group metrics |
| `temporal_confusion_matrix.png` | Temporal classification error visualization |
| `temporal_audit_metadata.json` | Temporal cutoff, features, split, and model metadata |

## 13. Conclusion

The Week 6 audit demonstrates why model accuracy alone is insufficient for evaluating an educational prediction system. The initial audit reports 69.17% test accuracy, while the day-60 experiment reports 57.14% and identifies only 11.2% of actual Withdrawn cases.

The group-level results also show performance differences that deserve additional investigation. These findings support further validation, improved temporal feature engineering, uncertainty analysis, and responsible human oversight.

**LearnPulse AI remains an experimental student-support project. The current evidence is not sufficient to justify autonomous or high-stakes decisions about students.**
