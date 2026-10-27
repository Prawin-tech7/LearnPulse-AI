
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("week 3/early_warning_30day.csv")

target = "final_result"
group_column = "id_student"

# Use exactly the same feature exclusions as the trained model.
X = df.drop(columns=[
    target,
    group_column,
    "average_score_30d",
])

y = df[target].map({
    "Distinction": 0,
    "Fail": 1,
    "Pass": 2,
    "Withdrawn": 3,
})

groups = df[group_column]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X.iloc[train_idx], y_train)

predictions = baseline.predict(X.iloc[test_idx])

print("===== CURRENT DATASET BASELINE =====")
print("Training rows:", len(train_idx))
print("Test rows:", len(test_idx))
print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, predictions), 4
))
print("Macro F1:", round(
    f1_score(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0,
    ),
    4,
))
