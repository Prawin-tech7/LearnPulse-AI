import json
from pathlib import Path

root = Path(r"C:\Users\KiTE\Desktop\LearnPulse-AI")

files = [
    root / "week 2" / "LearnPulse_Preprocessing.ipynb",
    root / "week 3" / "LearnPulse_Model_Training.ipynb",
]

keywords = [
    "read_csv", "final_result", "labelencoder",
    "map(", "replace(", "factorize", "categories",
    "drop(", "average_score", "assessment_count"
]

for path in files:
    print("\n" + "=" * 70)
    print(path.name)
    print("=" * 70)

    with path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    for i, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if any(word in source.lower() for word in keywords):
            print(f"\n--- Cell {i} ---")
            print(source[:2500])