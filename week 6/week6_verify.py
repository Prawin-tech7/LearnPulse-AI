import csv
from pathlib import Path
from collections import Counter

root = Path(r"C:\Users\KiTE\Desktop\LearnPulse-AI")
raw_file = root / "datasets" / "raw" / "studentInfo.csv"
processed_file = root / "week 2" / "processed_student_dataset.csv"

# Verify original outcome labels
print("=" * 60)
print("ORIGINAL OUTCOME LABELS")
print("=" * 60)

with raw_file.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

outcomes = Counter(row["final_result"] for row in rows)

for label, count in sorted(outcomes.items()):
    print(f"{label}: {count}")

print("\nLabelEncoder's alphabetical mapping:")
for number, label in enumerate(sorted(outcomes)):
    print(f"{number} = {label}")

# Check whether student IDs occur in multiple records
print("\n" + "=" * 60)
print("STUDENT ID REUSE CHECK")
print("=" * 60)

id_counts = Counter(row["id_student"] for row in rows)
repeated = sum(count > 1 for count in id_counts.values())

print("Original rows:", len(rows))
print("Unique student IDs:", len(id_counts))
print("IDs with multiple records:", repeated)

# Check whether the processed target has the expected classes
print("\n" + "=" * 60)
print("PROCESSED TARGET CHECK")
print("=" * 60)

with processed_file.open("r", encoding="utf-8-sig", newline="") as f:
    processed_rows = list(csv.DictReader(f))

print("Processed rows:", len(processed_rows))
print(
    "Encoded final_result counts:",
    dict(Counter(row["final_result"] for row in processed_rows))
)

print("\nVerification complete.")