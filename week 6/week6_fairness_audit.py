
from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# LEARNPULSE AI - WEEK 6 FAIRNESS AUDIT
# ============================================================

ROOT = Path(r"C:\Users\KiTE\Desktop\LearnPulse-AI")
DATA_FILE = ROOT / "week 2" / "processed_student_dataset.csv"
RAW_FILE = ROOT / "datasets" / "raw" / "studentInfo.csv"
OUTPUT_DIR = ROOT / "week 6"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "final_result"
GROUP_COLUMNS = ["gender", "disability"]

CLASS_NAMES = {
    0: "Distinction",
    1: "Fail",
    2: "Pass",
    3: "Withdrawn"
}

print("=" * 65)
print("LEARNPULSE AI - WEEK 6 FAIRNESS AUDIT")
print("=" * 65)

# 1. Load the processed dataset
df = pd.read_csv(DATA_FILE)

required_columns = [
    "id_student", TARGET, "gender", "disability"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("\nDataset shape:", df.shape)
print("Outcome classes:", CLASS_NAMES)

# 2. Verify the category encoding using the original data
with RAW_FILE.open("r", encoding="utf-8-sig", newline="") as file:
    raw_rows = list(csv.DictReader(file))

print("\n=== SENSITIVE ATTRIBUTE ENCODING ===")

for column in GROUP_COLUMNS:
    original_labels = sorted({
        row[column] for row in raw_rows
        if row.get(column) not in (None, "")
    })

    print(f"\n{column}:")
    for encoded_value, original_label in enumerate(original_labels):
        print(f"  {encoded_value} = {original_label}")

# 3. Prepare features
# Exclude the outcome and student identifier from model inputs.
X = df.drop(columns=[TARGET, "id_student"])
y = df[TARGET].astype(int)
groups = df["id_student"]

# Check the expected outcome values
if not set(y.unique()).issubset(CLASS_NAMES.keys()):
    raise ValueError("Unexpected encoded outcome labels found.")

# 4. Split by student ID to avoid student overlap
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

test_df = df.iloc[test_indices].copy()

train_ids = set(groups.iloc[train_indices])
test_ids = set(groups.iloc[test_indices])

assert train_ids.isdisjoint(test_ids)

print("\n=== DATA SPLIT ===")
print("Training records:", len(train_indices))
print("Testing records:", len(test_indices))
print("Training unique students:", len(train_ids))
print("Testing unique students:", len(test_ids))
print("Student ID overlap:", len(train_ids & test_ids))

# 5. Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=None,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 6. Overall performance
accuracy = accuracy_score(y_test, predictions)

print("\n=== OVERALL MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification report:")
report = classification_report(
    y_test,
    predictions,
    labels=list(CLASS_NAMES.keys()),
    target_names=list(CLASS_NAMES.values()),
    zero_division=0
)
print(report)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=list(CLASS_NAMES.keys())
)

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(matrix)

# Save the overall report and confusion matrix
(OUTPUT_DIR / "classification_report.txt").write_text(
    f"Accuracy: {accuracy:.4f}\n\n{report}\n"
    f"Confusion matrix:\n{matrix}\n",
    encoding="utf-8"
)

# 7. Per-group fairness metrics
# For each outcome, treat that outcome as positive
# and all other outcomes as negative.
audit_rows = []

print("\n=== GROUP FAIRNESS METRICS ===")

for attribute in GROUP_COLUMNS:
    print(f"\n--- Attribute: {attribute} ---")

    for group_value in sorted(test_df[attribute].dropna().unique()):
        mask = (
            test_df[attribute].to_numpy() == group_value
        )

        actual_group = y_test.to_numpy()[mask]
        predicted_group = predictions[mask]

        print(f"\nGroup {group_value} | Records: {int(mask.sum())}")

        for class_id, class_name in CLASS_NAMES.items():
            actual_positive = actual_group == class_id
            predicted_positive = predicted_group == class_id

            tp = int(np.sum(actual_positive & predicted_positive))
            fp = int(np.sum(~actual_positive & predicted_positive))
            fn = int(np.sum(actual_positive & ~predicted_positive))
            tn = int(np.sum(~actual_positive & ~predicted_positive))

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            fpr = fp / (fp + tn) if fp + tn else 0.0
            fnr = fn / (fn + tp) if fn + tp else 0.0

            group_accuracy = (
                accuracy_score(actual_group, predicted_group)
                if len(actual_group) else 0.0
            )

            positive_rate = (
                float(np.mean(predicted_positive))
                if len(predicted_positive) else 0.0
            )

            row = {
                "attribute": attribute,
                "group_value": int(group_value),
                "class_id": class_id,
                "class_name": class_name,
                "records": int(mask.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "accuracy": group_accuracy,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "predicted_positive_rate": positive_rate
            }

            audit_rows.append(row)

            print(
                f"  {class_name}: "
                f"Precision={precision:.3f}, "
                f"Recall={recall:.3f}, "
                f"FPR={fpr:.3f}, "
                f"FNR={fnr:.3f}"
            )

# 8. Export detailed metrics
metrics_df = pd.DataFrame(audit_rows)

metrics_path = OUTPUT_DIR / "fairness_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

# 9. Plot recall for each outcome and sensitive group
for attribute in GROUP_COLUMNS:
    plot_df = metrics_df[metrics_df["attribute"] == attribute]

    pivot = plot_df.pivot(
        index="class_name",
        columns="group_value",
        values="recall"
    )

    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title(f"Recall by Outcome and {attribute.title()} Group")
    ax.set_xlabel("Actual outcome treated as positive")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.legend(title=f"Encoded {attribute}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    chart_path = OUTPUT_DIR / f"recall_by_{attribute}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

# 10. Plot the confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
image = ax.imshow(matrix, interpolation="nearest")
ax.set_title("LearnPulse AI Confusion Matrix")
ax.set_xlabel("Predicted outcome")
ax.set_ylabel("Actual outcome")
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES.values(), rotation=30, ha="right")
ax.set_yticklabels(CLASS_NAMES.values())

for row_index in range(matrix.shape[0]):
    for col_index in range(matrix.shape[1]):
        ax.text(
            col_index,
            row_index,
            str(matrix[row_index, col_index]),
            ha="center",
            va="center"
        )

fig.colorbar(image, ax=ax)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=150
)
plt.close()

# 11. Save audit metadata
metadata = {
    "dataset_records": int(len(df)),
    "training_records": int(len(train_indices)),
    "testing_records": int(len(test_indices)),
    "unique_training_students": int(len(train_ids)),
    "unique_testing_students": int(len(test_ids)),
    "student_id_overlap": int(len(train_ids & test_ids)),
    "features_excluded": [TARGET, "id_student"],
    "test_accuracy": float(accuracy),
    "class_mapping": CLASS_NAMES,
    "split_method": "GroupShuffleSplit by id_student",
    "random_state": 42
}

(OUTPUT_DIR / "audit_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\n=== AUDIT FILES CREATED ===")
print("Folder:", OUTPUT_DIR)
print(" - classification_report.txt")
print(" - fairness_metrics.csv")
print(" - recall_by_gender.png")
print(" - recall_by_disability.png")
print(" - confusion_matrix.png")
print(" - audit_metadata.json")
print("\nFairness audit completed.")
