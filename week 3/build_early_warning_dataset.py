
from pathlib import Path
import pandas as pd

RAW = Path("datasets/raw")
OUT = Path("week 3/early_warning_30day.csv")
CUTOFF_DAY = 30

required = [
    "studentInfo.csv",
    "studentVle.csv",
    "studentAssessment.csv",
    "assessments.csv",
    "studentRegistration.csv",
]

for filename in required:
    if not (RAW / filename).exists():
        raise FileNotFoundError(f"Missing file: {RAW / filename}")

keys = ["code_module", "code_presentation", "id_student"]

print("Loading student outcomes...")
info = pd.read_csv(RAW / "studentInfo.csv")

info = info[
    [
        "code_module", "code_presentation",
        "id_student", "gender", "region",
        "highest_education", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits",
        "disability", "final_result",
    ]
].dropna(subset=["final_result"]).copy()

initial_rows = len(info)

print("Loading registration records...")
registration = pd.read_csv(
    RAW / "studentRegistration.csv",
    usecols=keys + ["date_unregistration"],
)

registration["date_unregistration"] = pd.to_numeric(
    registration["date_unregistration"],
    errors="coerce",
)

# In OULAD, -1 is the sentinel for no recorded unregistration.
# Other valid dates, including negative dates, are real dates.
registration["already_unregistered"] = (
    registration["date_unregistration"].notna()
    & (registration["date_unregistration"] != -1)
    & (registration["date_unregistration"] <= CUTOFF_DAY)
)

registration_status = registration[
    keys + ["already_unregistered"]
]

if registration_status.duplicated(keys).any():
    raise ValueError(
        "Registration data has duplicate student-course keys."
    )

info = info.merge(
    registration_status,
    on=keys,
    how="left",
    validate="one_to_one",
)

unknown_registration = info["already_unregistered"].isna().sum()

print("Rows before registration filter:", initial_rows)
print("Rows with missing registration status:", unknown_registration)

# Exclude records whose registration status cannot be verified.
info = info[info["already_unregistered"].eq(False)].copy()
info.drop(columns=["already_unregistered"], inplace=True)

print("Rows after registration filter:", len(info))
print("Excluded records:", initial_rows - len(info))

print("Loading VLE activity up to day", CUTOFF_DAY, "...")
vle = pd.read_csv(
    RAW / "studentVle.csv",
    usecols=keys + ["date", "sum_click"],
)

vle = vle[vle["date"] <= CUTOFF_DAY]

vle_features = (
    vle.groupby(keys, as_index=False)
    .agg(
        total_clicks_30d=("sum_click", "sum"),
        active_days_30d=("date", "nunique"),
    )
)

print("Loading assessment submissions...")
assessments = pd.read_csv(
    RAW / "assessments.csv",
    usecols=[
        "code_module", "code_presentation",
        "id_assessment", "date",
    ],
)

submissions = pd.read_csv(
    RAW / "studentAssessment.csv",
    usecols=[
        "id_assessment", "id_student",
        "date_submitted", "score",
    ],
)

submissions = submissions.merge(
    assessments,
    on="id_assessment",
    how="inner",
    validate="many_to_one",
)

# Only count assessments scheduled and submitted by day 30.
submissions = submissions[
    submissions["date"].notna()
    & (submissions["date"] <= CUTOFF_DAY)
    & submissions["date_submitted"].notna()
    & (submissions["date_submitted"] <= CUTOFF_DAY)
].copy()

assessment_keys = [
    "code_module", "code_presentation", "id_student"
]

assessment_features = (
    submissions.groupby(assessment_keys, as_index=False)
    .agg(
        assessments_submitted_30d=("id_assessment", "nunique"),
        average_score_30d=("score", "mean"),
    )
)

print("Combining features...")
dataset = info.merge(
    vle_features,
    on=keys,
    how="left",
    validate="one_to_one",
)

dataset = dataset.merge(
    assessment_features,
    on=keys,
    how="left",
    validate="one_to_one",
)

dataset["total_clicks_30d"] = (
    dataset["total_clicks_30d"].fillna(0)
)
dataset["active_days_30d"] = (
    dataset["active_days_30d"].fillna(0)
)
dataset["assessments_submitted_30d"] = (
    dataset["assessments_submitted_30d"].fillna(0)
)

dataset.to_csv(OUT, index=False)

print("\nDataset created:", OUT)
print("Rows:", len(dataset))
print("Columns:", len(dataset.columns))

print("\nOutcome counts:")
print(dataset["final_result"].value_counts(dropna=False))

print("\nMissing values:")
print(dataset.isna().sum())

print("\nPreview:")
print(dataset.head())

print(
    "\nLIMITATION: Verify the OULAD relative-day conventions and "
    "assessment timing before deployment. Average assessment scores "
    "are excluded from model features until their availability by "
    "day 30 is confirmed."
)
