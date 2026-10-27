
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("week 3/early_warning_30day.csv")
MODEL_PATH = Path("week 3/early_warning_model.joblib")
METRICS_PATH = Path("week 3/early_warning_metrics.txt")

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200

LABELS = ["Distinction", "Fail", "Pass", "Withdrawn"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

TARGET = "final_result"
GROUP_COLUMN = "id_student"

# Assessment scores are deliberately excluded until we verify
# when those scores become available to the prediction system.
EXCLUDED_FEATURES = [
    TARGET,
    GROUP_COLUMN,
    "average_score_30d",
]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 65)
print("LEARNPULSE AI - DAY-30 EARLY-WARNING MODEL")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run build_early_warning_dataset.py first."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    TARGET,
    GROUP_COLUMN,
    "total_clicks_30d",
    "active_days_30d",
    "assessments_submitted_30d",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

df = df.dropna(subset=[TARGET, GROUP_COLUMN]).copy()

unknown_labels = set(df[TARGET].unique()) - set(LABEL_TO_ID)

if unknown_labels:
    raise ValueError(
        f"Unexpected target labels: {unknown_labels}"
    )

y = df[TARGET].map(LABEL_TO_ID).astype(int)
groups = df[GROUP_COLUMN]

X = df.drop(columns=EXCLUDED_FEATURES)

if X.empty:
    raise ValueError("No prediction features remain.")

# Reject non-finite numeric values. Missing values are handled
# within the preprocessing pipeline.
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

for col in numeric_columns:
    X[col] = X[col].replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

print("\n===== DATASET SUMMARY =====")
print("Rows:", len(df))
print("Unique students:", groups.nunique())
print("Prediction features:", len(X.columns))
print("Excluded features:", EXCLUDED_FEATURES)
print("\nOutcome distribution:")
print(df[TARGET].value_counts())

print("\nFeatures used:")
for column in X.columns:
    print("-", column)


# ============================================================
# 3. PREPROCESSING AND MODEL PIPELINE
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool", "str"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore"),
    ),
])

transformers = []

if numeric_columns:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_columns)
    )

if categorical_columns:
    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)

model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
    ),
])


# ============================================================
# 4. GROUP-AWARE TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_groups = groups.iloc[train_indices]
test_groups = groups.iloc[test_indices]

student_overlap = len(
    set(train_groups) & set(test_groups)
)

if student_overlap != 0:
    raise RuntimeError(
        "Student leakage detected between training and test data."
    )

if y_train.nunique() < len(LABELS):
    raise ValueError(
        "Training split does not contain every outcome class."
    )

print("\n===== TRAIN / TEST SPLIT =====")
print("Training rows:", len(train_indices))
print("Test rows:", len(test_indices))
print("Training students:", train_groups.nunique())
print("Test students:", test_groups.nunique())
print("Student overlap:", student_overlap)


# ============================================================
# 5. TRAIN AND EVALUATE
# ============================================================

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(
    y_test, predictions
)
macro_f1 = f1_score(
    y_test,
    predictions,
    labels=list(range(len(LABELS))),
    average="macro",
    zero_division=0,
)

report = classification_report(
    y_test,
    predictions,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    zero_division=0,
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=list(range(len(LABELS))),
)

print("\n===== HOLDOUT RESULTS =====")
print("Accuracy:", round(accuracy, 4))
print("Balanced accuracy:", round(balanced_accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification report:")
print(report)

print("Confusion matrix (actual rows, predicted columns):")
print(matrix)


# ============================================================
# 6. GROUP-AWARE CROSS-VALIDATION
# ============================================================

number_of_groups = groups.nunique()
fold_count = min(5, number_of_groups)

if fold_count < 2:
    raise ValueError(
        "Not enough unique students for group cross-validation."
    )

cv = GroupKFold(n_splits=fold_count)

print("\nRunning group-aware cross-validation...")

cv_results = cross_validate(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    },
    n_jobs=1,
    error_score="raise",
)

cv_summary = {}

print("\n===== GROUP-AWARE CROSS-VALIDATION =====")
print("Folds:", fold_count)

for metric in [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
]:
    values = cv_results["test_" + metric]

    mean_value = values.mean()
    std_value = values.std()

    cv_summary[metric] = (mean_value, std_value)

    print(
        f"{metric}: mean={mean_value:.4f}, "
        f"std={std_value:.4f}"
    )


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

print("\n===== TOP FEATURE IMPORTANCE =====")

try:
    fitted_preprocessor = model.named_steps["preprocessing"]
    fitted_classifier = model.named_steps["classifier"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(
        "importance",
        ascending=False,
    )

    print(importance_df.head(15).to_string(index=False))

except (AttributeError, ValueError) as exc:
    print("Could not calculate feature importance:", exc)
    importance_df = pd.DataFrame(
        columns=["feature", "importance"]
    )


# ============================================================
# 8. SAVE MODEL AND EVALUATION REPORT
# ============================================================

joblib.dump(model, MODEL_PATH)

report_lines = [
    "LEARNPULSE AI - DAY-30 EARLY-WARNING EVALUATION",
    "=" * 55,
    f"Dataset rows: {len(df)}",
    f"Unique students: {groups.nunique()}",
    f"Training rows: {len(train_indices)}",
    f"Test rows: {len(test_indices)}",
    f"Student overlap: {student_overlap}",
    f"Accuracy: {accuracy:.4f}",
    f"Balanced accuracy: {balanced_accuracy:.4f}",
    f"Macro F1: {macro_f1:.4f}",
    "",
    "Classification report:",
    report,
    "Confusion matrix (actual rows, predicted columns):",
    str(matrix),
    "",
    "Group-aware cross-validation:",
]

for metric, (mean_value, std_value) in cv_summary.items():
    report_lines.append(
        f"{metric}: mean={mean_value:.4f}, "
        f"std={std_value:.4f}"
    )

report_lines.extend([
    "",
    "Features:",
    ", ".join(X.columns),
    "",
    "Excluded features:",
    ", ".join(EXCLUDED_FEATURES),
    "",
    "Limitations:",
    "- Day-30 date conventions must be verified.",
    "- Assessment scores excluded pending availability verification.",
    "- Group splitting prevents student overlap across partitions.",
    "- Results are experimental and do not establish causation.",
    "- Performance should be assessed before real-world use.",
])

if not importance_df.empty:
    report_lines.extend([
        "",
        "Top feature importance:",
        importance_df.head(15).to_string(index=False),
    ])

METRICS_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("\n===== SAVED FILES =====")
print("Model:", MODEL_PATH)
print("Evaluation report:", METRICS_PATH)
print("\nTraining and evaluation completed.")
