
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = "week 3/early_warning_30day.csv"
MODEL_PATH = "week 3/early_warning_risk_model.joblib"
REPORT_PATH = "week 3/early_warning_risk_metrics.txt"

df = pd.read_csv(DATA_PATH)

# Fail and Withdrawn are the outcomes we want to identify.
df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

# Exclude the original outcome, student ID, and assessment score.
# The score is excluded because its availability by day 30 needs validation.
X = df.drop(
    columns=["final_result", "at_risk", "id_student", "average_score_30d"]
)
y = df["at_risk"]
groups = df["id_student"]

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# Baseline: always predict the majority class in training data.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# Train the risk model.
model.fit(X_train, y_train)
pred = model.predict(X_test)

def metrics(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced accuracy": balanced_accuracy_score(actual, predicted),
        "At-risk precision": precision_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
        "At-risk recall": recall_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
        "At-risk F1": f1_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
        "Macro F1": f1_score(
            actual, predicted, average="macro", zero_division=0
        ),
    }

baseline_metrics = metrics(y_test, baseline_pred)
model_metrics = metrics(y_test, pred)

# Confirm there is no student overlap.
overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])

# Grouped cross-validation on the training portion only.
cv = GroupKFold(n_splits=5)
cv_results = cross_validate(
    model,
    X_train,
    y_train,
    groups=groups.iloc[train_idx],
    cv=cv,
    scoring={
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "recall": "recall",
        "precision": "precision",
    },
    n_jobs=1,
)

print("===== BINARY EARLY-WARNING MODEL =====")
print("Rows:", len(df))
print("Unique students:", groups.nunique())
print("At-risk records:", int(y.sum()))
print("Not-at-risk records:", int((y == 0).sum()))
print("Training rows:", len(train_idx))
print("Test rows:", len(test_idx))
print("Student overlap:", len(overlap))

print("\n===== BASELINE =====")
for name, value in baseline_metrics.items():
    print(f"{name}: {value:.4f}")

print("\n===== RANDOM FOREST =====")
for name, value in model_metrics.items():
    print(f"{name}: {value:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test, pred,
    labels=[0, 1],
    target_names=["Not at risk", "At risk"],
    zero_division=0
))

print("===== CONFUSION MATRIX =====")
print("Rows = actual; columns = predicted")
print(confusion_matrix(y_test, pred, labels=[0, 1]))

print("\n===== 5-FOLD GROUPED CV (TRAINING DATA) =====")
for metric_name in [
    "balanced_accuracy", "f1_macro", "recall", "precision"
]:
    values = cv_results[f"test_{metric_name}"]
    print(
        f"{metric_name}: "
        f"mean={values.mean():.4f}, std={values.std():.4f}"
    )

print("\n===== TOP 10 FEATURE IMPORTANCES =====")
fitted_preprocessor = model.named_steps["preprocessor"]
feature_names = fitted_preprocessor.get_feature_names_out()
importances = model.named_steps["classifier"].feature_importances_

for name, value in sorted(
    zip(feature_names, importances),
    key=lambda item: item[1],
    reverse=True
)[:10]:
    print(f"{name}: {value:.4f}")

joblib.dump(model, MODEL_PATH)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("LearnPulse AI — Binary Early-Warning Evaluation\n\n")
    f.write("Target: Fail or Withdrawn = at risk; others = not at risk\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Unique students: {groups.nunique()}\n")
    f.write(f"Student overlap: {len(overlap)}\n\n")

    f.write("Baseline metrics:\n")
    for name, value in baseline_metrics.items():
        f.write(f"{name}: {value:.4f}\n")

    f.write("\nRandom Forest metrics:\n")
    for name, value in model_metrics.items():
        f.write(f"{name}: {value:.4f}\n")

    f.write("\nClassification report:\n")
    f.write(classification_report(
        y_test, pred,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        zero_division=0
    ))

    f.write("\nConfusion matrix (actual rows, predicted columns):\n")
    f.write(str(confusion_matrix(y_test, pred, labels=[0, 1])))

    f.write("\n\nFive-fold grouped CV (training data):\n")
    for metric_name in [
        "balanced_accuracy", "f1_macro", "recall", "precision"
    ]:
        values = cv_results[f"test_{metric_name}"]
        f.write(
            f"{metric_name}: mean={values.mean():.4f}, "
            f"std={values.std():.4f}\n"
        )

print("\nSaved model:", MODEL_PATH)
print("Saved report:", REPORT_PATH)
