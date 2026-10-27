
import pandas as pd

df = pd.read_csv("week 3/early_warning_30day.csv")

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isna().sum().sort_values(ascending=False))

print("\n===== OUTCOME DISTRIBUTION =====")
print(df["final_result"].value_counts())

print("\n===== FIRST FIVE ROWS =====")
print(df.head().to_string(index=False))

print("\n===== EARLY ACTIVITY SUMMARY =====")
print(
    df[
        [
            "total_clicks_30d",
            "active_days_30d",
            "assessments_submitted_30d",
            "average_score_30d",
        ]
    ].describe().round(2)
)

print("\n===== STUDENT ID CHECK =====")
print("Unique student IDs:", df["id_student"].nunique())
print("Rows:", len(df))
print("Repeated student IDs:", df["id_student"].duplicated().sum())
