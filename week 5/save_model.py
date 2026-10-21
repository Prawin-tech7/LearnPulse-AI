
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Locate the project and dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "week 2" / "processed_student_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "week 5" / "models"
MODEL_PATH = MODEL_DIR / "learnpulse_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load the same processed dataset used in Week 3
df = pd.read_csv(DATA_PATH)

X = df.drop("final_result", axis=1)
y = df["final_result"]

print("Dataset shape:", df.shape)
print("Number of features:", X.shape[1])
print("Target classes:", sorted(y.unique().tolist()))

# Use the same train/test split as Week 3
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Tune the Random Forest model
params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

print("\nTraining and tuning model...")
grid.fit(X_train, y_train)

# GridSearchCV has already refitted the best estimator on X_train
best_model = grid.best_estimator_

# Evaluate on the held-out test set
predictions = best_model.predict(X_test)

print("\nBest parameters:", grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)
print("Test accuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions))

# Save the model AND the feature information needed by the API
artifact = {
    "model": best_model,
    "feature_names": X.columns.tolist(),
    "target_name": "final_result",
    "target_classes": best_model.classes_.tolist(),
    "best_params": grid.best_params_
}

joblib.dump(artifact, MODEL_PATH)

print("\nModel saved successfully to:")
print(MODEL_PATH)
