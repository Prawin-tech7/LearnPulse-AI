
import pandas as pd

df = pd.read_csv("week 3/early_warning_30day.csv")

print("\n===== ASSESSMENT SUBMISSION COUNTS =====")
print(
    df.groupby("assessments_submitted_30d")["average_score_30d"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\n===== MISSING SCORES BY SUBMISSION COUNT =====")
print(
    df.assign(
        score_missing=df["average_score_30d"].isna()
    )
    .groupby("assessments_submitted_30d")["score_missing"]
    .agg(["sum", "mean"])
    .round(3)
)

print("\n===== OUTCOMES BY EARLY ACTIVITY =====")
df["activity_group"] = pd.cut(
    df["total_clicks_30d"],
    bins=[-1, 0, 50, 200, 500, float("inf")],
    labels=[
        "Zero clicks",
        "1-50 clicks",
        "51-200 clicks",
        "201-500 clicks",
        "Over 500 clicks",
    ],
)

print(
    pd.crosstab(
        df["activity_group"],
        df["final_result"],
        normalize="index",
    ).round(3)
)

print("\n===== OUTCOMES BY ASSESSMENT SUBMISSIONS =====")
print(
    pd.crosstab(
        df["assessments_submitted_30d"],
        df["final_result"],
        normalize="index",
    ).round(3)
)

print("\n===== ZERO-ACTIVITY RECORDS =====")
zero = df[df["total_clicks_30d"] == 0]

print("Records with zero clicks:", len(zero))
print("\nTheir outcomes:")
print(zero["final_result"].value_counts())

print("\n===== CHECK SCORE RANGES =====")
print("Scores below 0:", (df["average_score_30d"] < 0).sum())
print("Scores above 100:", (df["average_score_30d"] > 100).sum())
