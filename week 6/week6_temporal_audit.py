
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(r"C:\Users\KiTE\Desktop\LearnPulse-AI")
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "week 6"
OUT.mkdir(parents=True, exist_ok=True)

# Experimental prediction cutoff: 60 days after course start.
CUTOFF_DAY = 60
TARGET = "final_result"

CLASSES = ["Distinction", "Fail", "Pass", "Withdrawn"]

print("=" * 65)
print("LEARNPULSE AI - TEMPORAL FAIRNESS AUDIT")
print("Prediction cutoff: course day", CUTOFF_DAY)
print("=" * 65)

# 1. Load source data
info = pd.read_csv(RAW / "studentInfo.csv")
vle = pd.read_csv(RAW / "studentVle.csv")
student_assessment = pd.read_csv(RAW / "studentAssessment.csv")
assessments = pd.read_csv(RAW / "assessments.csv")
registration = pd.read_csv(RAW / "studentRegistration.csv")

KEYS = ["id_student", "code_module", "code_presentation"]

# Confirm each course record has one registration row.
registration = registration[
    KEYS + ["date_unregistration"]
].drop_duplicates()

if registration.duplicated(KEYS).any():
    raise ValueError(
        "Multiple registration rows found for a course record. "
        "Inspect registration data before continuing."
    )

# 2. Keep course records still registered after the cutoff.
base = info.merge(
    registration,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

# An unregistration on or before the cutoff means the student
# may already have left by the prediction date.
base = base[
    base["date_unregistration"].isna()
    | (base["date_unregistration"] > CUTOFF_DAY)
].copy()

print("\nCourse records eligible at cutoff:", len(base))
print("Excluded records already unregistered by cutoff:",
      len(info) - len(base))

if base.empty:
    raise ValueError("No eligible course records remain.")

# 3. Build VLE engagement features using only activity by day 60.
vle_early = vle[vle["date"] <= CUTOFF_DAY].copy()

vle_features = (
    vle_early.groupby(KEYS)
    .agg(
        total_clicks_early=("sum_click", "sum"),
        avg_clicks_early=("sum_click", "mean")
    )
    .reset_index()
)

# 4. Link student submissions to the assessment schedule.
# assessments.csv supplies module and presentation identifiers.
assessment_records = student_assessment.merge(
    assessments[
        [
            "id_assessment",
            "code_module",
            "code_presentation",
            "date"
        ]
    ],
    on="id_assessment",
    how="left",
    validate="many_to_one",
    suffixes=("_submission", "_schedule")
)

# Use actual submission date to avoid using scores submitted later.
early_submissions = assessment_records[
    (assessment_records["date_submitted"] <= CUTOFF_DAY)
    & assessment_records["code_module"].notna()
].copy()

assessment_features = (
    early_submissions.groupby(KEYS)
    .agg(
        average_score_early=("score", "mean"),
        assessment_count_early=("score", "count")
    )
    .reset_index()
)

print("Assessment submissions by cutoff:",
      len(early_submissions))
print("VLE records by cutoff:", len(vle_early))

# 5. Merge early features into each student-course record.
df = base.merge(
    vle_features, on=KEYS, how="left", validate="one_to_one"
).merge(
    assessment_features, on=KEYS, how="left",
    validate="one_to_one"
)

early_numeric = [
    "total_clicks_early",
    "avg_clicks_early",
    "average_score_early",
    "assessment_count_early"
]

df[early_numeric] = df[early_numeric].fillna(0)

# Drop the identifier and unregistration date from model features.
# The target is the eventual course outcome.
excluded = [TARGET, "id_student", "date_unregistration"]
X = df.drop(columns=excluded)
y = df[TARGET].astype(str)
groups = df["id_student"]

# Keep only supported target classes.
valid = y.isin(CLASSES)
df = df.loc[valid].copy()
X = X.loc[valid].copy()
y = y.loc[valid].copy()
groups = groups.loc[valid]

categorical = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Fit category encoders on training data only.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        ("numeric", "passthrough", numeric)
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Group split: no student ID can occur in both sets.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
test_meta = df.iloc[test_idx].copy()

train_ids = set(groups.iloc[train_idx])
test_ids = set(groups.iloc[test_idx])

assert train_ids.isdisjoint(test_ids)

print("\n=== SPLIT ===")
print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Training students:", len(train_ids))
print("Testing students:", len(test_ids))
print("Student ID overlap:", len(train_ids & test_ids))

# 7. Train and evaluate.
model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

report = classification_report(
    y_test,
    pred,
    labels=CLASSES,
    zero_division=0
)

matrix = confusion_matrix(y_test, pred, labels=CLASSES)

print("\n=== OVERALL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.4f}")
print(report)
print("Confusion matrix (actual rows, predicted columns):")
print(matrix)

(OUT / "temporal_classification_report.txt").write_text(
    f"Prediction cutoff: day {CUTOFF_DAY}\n"
    f"Eligible records: {len(df)}\n"
    f"Test accuracy: {accuracy:.4f}\n\n"
    f"{report}\nConfusion matrix:\n{matrix}\n",
    encoding="utf-8"
)

# 8. Compare performance across gender and disability groups.
audit_rows = []

for attribute in ["gender", "disability"]:
    for group_value in sorted(test_meta[attribute].dropna().unique()):
        mask = test_meta[attribute].to_numpy() == group_value
        actual = y_test.to_numpy()[mask]
        predicted = pred[mask]

        for class_name in CLASSES:
            actual_positive = actual == class_name
            predicted_positive = predicted == class_name

            tp = int(np.sum(actual_positive & predicted_positive))
            fp = int(np.sum(~actual_positive & predicted_positive))
            fn = int(np.sum(actual_positive & ~predicted_positive))
            tn = int(np.sum(~actual_positive & ~predicted_positive))

            audit_rows.append({
                "attribute": attribute,
                "group_value": str(group_value),
                "records": int(mask.sum()),
                "class_name": class_name,
                "accuracy": float(
                    accuracy_score(actual, predicted)
                ) if len(actual) else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "false_positive_rate":
                    fp / (fp + tn) if fp + tn else 0.0,
                "false_negative_rate":
                    fn / (fn + tp) if fn + tp else 0.0
            })

metrics = pd.DataFrame(audit_rows)
metrics.to_csv(OUT / "temporal_fairness_metrics.csv", index=False)

print("\n=== FAIRNESS METRICS ===")
for attribute in ["gender", "disability"]:
    subset = metrics[
        (metrics["attribute"] == attribute)
        & (metrics["class_name"] == "Withdrawn")
    ]

    print(f"\nWithdrawal metrics by {attribute}:")
    print(subset.to_string(index=False))

# 9. Save a confusion-matrix chart.
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, interpolation="nearest")
ax.set_title(f"Temporal Audit Confusion Matrix (Day {CUTOFF_DAY})")
ax.set_xlabel("Predicted outcome")
ax.set_ylabel("Actual outcome")
ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, rotation=30, ha="right")
ax.set_yticklabels(CLASSES)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT / "temporal_confusion_matrix.png", dpi=150)
plt.close()

metadata = {
    "cutoff_day": CUTOFF_DAY,
    "eligible_records": int(len(df)),
    "training_records": int(len(train_idx)),
    "testing_records": int(len(test_idx)),
    "student_id_overlap": int(len(train_ids & test_ids)),
    "accuracy": float(accuracy),
    "excluded_features": excluded,
    "categorical_features": categorical,
    "numeric_features": numeric,
    "split_method": "GroupShuffleSplit by id_student",
    "random_state": 42
}

(OUT / "temporal_audit_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\n=== OUTPUT FILES ===")
print("temporal_classification_report.txt")
print("temporal_fairness_metrics.csv")
print("temporal_confusion_matrix.png")
print("temporal_audit_metadata.json")
print("\nTemporal audit completed.")
