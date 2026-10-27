import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\Users\KiTE\Desktop\LearnPulse-AI")
NOTEBOOK = ROOT / "week 3" / "LearnPulse_Model_Training.ipynb"
DATASET = ROOT / "week 2" / "processed_student_dataset.csv"

# 1. Inspect the model-training notebook
print("\n" + "=" * 60)
print("WEEK 3 MODEL NOTEBOOK")
print("=" * 60)

with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

keywords = (
    "target", "final_result", "classifier", ".fit(",
    ".predict(", "classification_report", "train_test_split",
    "feature", "save", "joblib", "pickle", "randomforest",
    "logisticregression", "decisiontree", "accuracy"
)

for number, cell in enumerate(notebook.get("cells", []), start=1):
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))
    if any(word in source.lower() for word in keywords):
        print(f"\n--- Code cell {number} ---")
        print(source[:3500])

# 2. Inspect the processed dataset
print("\n" + "=" * 60)
print("PROCESSED DATASET")
print("=" * 60)

with DATASET.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    counts = Counter()
    total_rows = 0

    print("Columns:", columns)

    for row in reader:
        total_rows += 1
        for col in columns:
            if col.lower() in {
                "final_result", "gender", "age_band",
                "disability", "highest_education", "region"
            }:
                value = (row.get(col) or "").strip()
                counts[(col, value or "<missing>")] += 1

print("Total rows:", total_rows)

for col in columns:
    if col.lower() in {
        "final_result", "gender", "age_band",
        "disability", "highest_education", "region"
    }:
        print(f"\nDistribution for {col}:")
        for (name, value), count in sorted(counts.items()):
            if name == col:
                print(f"  {value}: {count}")

print("\nInspection complete. No dataset rows were modified.")