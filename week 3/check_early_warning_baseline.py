
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

df = pd.read_csv("week 3/early_warning_30day.csv")

target = "final_result"
groups = df["id_student"]

X = df.drop(columns=[target, "id_student"])
y = df[target].map({
    "Distinction": 0,
    "Fail": 1,
    "Pass": 2,
    "Withdrawn": 3,
})

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

predictions = baseline.predict(X_test)

print("\n===== SIMPLE BASELINE =====")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print("Macro F1:", round(f1_score(
    y_test, predictions, average="macro", zero_division=0
), 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=["Distinction", "Fail", "Pass", "Withdrawn"],
    zero_division=0,
))
