
from pathlib import Path
from datetime import datetime, timezone
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "learnpulse_model.joblib"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "api.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("learnpulse")

app = FastAPI(
    title="LearnPulse AI Prediction API",
    description="Student outcome prediction and service monitoring",
    version="1.0.0",
)

# Basic in-memory service metrics
metrics = {
    "total_requests": 0,
    "prediction_requests": 0,
    "prediction_errors": 0,
    "total_prediction_latency_ms": 0.0,
}

artifact = None
model_load_error = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception:
    model_load_error = "Model could not be loaded"
    logger.exception("Model loading failed")


class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float]


@app.middleware("http")
async def track_requests(request: Request, call_next):
    metrics["total_requests"] += 1
    start = time.perf_counter()

    try:
        response = await call_next(request)
        return response
    except Exception:
        logger.exception("Unhandled request error")
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "method=%s path=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )


@app.get("/")
def root():
    return {
        "service": "LearnPulse AI",
        "message": "Student outcome prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Model unavailable",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
def predict(student: StudentFeatures):
    metrics["prediction_requests"] += 1
    start = time.perf_counter()

    try:
        if artifact is None:
            raise HTTPException(
                status_code=503,
                detail="Model unavailable",
            )

        model = artifact["model"]
        expected_features = artifact["feature_names"]
        supplied_features = student.features

        missing = sorted(set(expected_features) - set(supplied_features))
        extra = sorted(set(supplied_features) - set(expected_features))

        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": missing,
                    "unexpected_features": extra,
                },
            )

        input_df = pd.DataFrame(
            [[supplied_features[name] for name in expected_features]],
            columns=expected_features,
        )

        prediction = model.predict(input_df)[0]

        result = {
            "prediction": str(prediction),
            "model": "RandomForestClassifier",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            result["class_probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(model.classes_, probabilities)
            }

        logger.info("Prediction completed successfully")
        return result

    except HTTPException:
        metrics["prediction_errors"] += 1
        raise
    except Exception:
        metrics["prediction_errors"] += 1
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error",
        )
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["total_prediction_latency_ms"] += elapsed_ms


@app.get("/metrics")
def get_metrics():
    count = metrics["prediction_requests"]

    return {
        **metrics,
        "average_prediction_latency_ms": round(
            metrics["total_prediction_latency_ms"] / count, 2
        ) if count else 0.0,
    }
