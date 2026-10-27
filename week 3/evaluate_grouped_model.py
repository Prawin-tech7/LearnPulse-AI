
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

# Load the processed dataset
df = pd.read_csv(r"week 2\processed_student_dataset.csv")

target = "final_result"
group_column = "id_student"

# Remove target and student ID from model inputs.
# ID is used only to keep each student's records together.
X = df.drop(columns=[target, group_column])
y = df[target]
groups = df[group_column]

# Class labels from the existing processed dataset:
# 0 = Distinction, 1 = Fail, 2 = Pass, 3 = Withdrawn
label_names = ["Distinction", "Fail", "Pass", "Withdrawn"]

# Detect categorical and numeric columns from the data
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

# All preprocessing is fitted inside each training split.
# Numeric scaling is fitted only on training data.
# Categorical values are one-hot encoded without using test data.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

# One final holdout split, keeping each student's records together
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
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

# Train only on the training partition
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n========== GROUPED HOLDOUT EVALUATION ==========")
print("Training rows:", len(train_indices))
print("Test rows:", len(test_indices))
print(
    "Student overlap:",
    len(set(train_groups) & set(test_groups))
)
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print(
    "Macro F1:",
    round(f1_score(y_test, predictions, average="macro", zero_division=0), 4),
)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=label_names,
    zero_division=0,
))

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3]))

# Group-aware cross-validation using the same pipeline
print("\n========== GROUP-AWARE CROSS-VALIDATION ==========")

group_count = groups.nunique()
fold_count = min(5, group_count)

if fold_count < 2:
    print("Not enough distinct students for cross-validation.")
else:
    cv = GroupKFold(n_splits=fold_count)

    scores = cross_validate(
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
        return_train_score=False,
    )

    print("Folds:", fold_count)
    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        values = scores["test_" + metric]
        print(
            metric,
            "mean =", round(values.mean(), 4),
            "std =", round(values.std(), 4),
        )

print("\nEvaluation complete.")
print(
    "Note: This evaluates final course outcomes, not necessarily "
    "early prediction before the outcome is known."
)
