"""
SolarVision Time Series API
Endpoints: /health, /forecast, /anomaly, /cleaning

Run:
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import json
from io import BytesIO

from PIL import Image
from fastapi import UploadFile, File

MODEL_DIR = Path("models")
PRED_PATH = Path("forecast_test_predictions_24h.csv")

app = FastAPI(
    title="SolarVision Time Series API",
    description="24h forecast helper, anomaly rules, cleaning cost-benefit",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Lazy load
# ---------------------------------------------------------------------------
_model = None
_features: List[str] = []
_blend = {"blend_w_model": 1.0, "persist_col": "P_dusty_W_lag_24h"}


def load_artifacts():
    global _model, _features, _blend
    model_path = MODEL_DIR / "best_forecast_24h.joblib"
    feat_path = MODEL_DIR / "best_forecast_24h_features.joblib"
    blend_path = MODEL_DIR / "blend_config.joblib"
    if not model_path.exists() or not feat_path.exists():
        raise FileNotFoundError(
            "Models not found. Run 01_preprocessing + 02_training first."
        )
    _model = joblib.load(model_path)
    _features = joblib.load(feat_path)
    if blend_path.exists():
        _blend = joblib.load(blend_path)


@app.on_event("startup")
def startup():
    try:
        load_artifacts()
        print(f"Loaded model with {len(_features)} features")
    except FileNotFoundError as e:
        print("WARNING:", e)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    """Feature map matching training columns. Missing keys → 0.0."""
    features: Dict[str, float] = Field(
        ..., description="Dict of feature_name → value"
    )
    use_blend: bool = False
    persist_value: Optional[float] = Field(
        None, description="Same-hour yesterday power if use_blend"
    )


class ForecastResponse(BaseModel):
    predicted_power_W: float
    model: str
    n_features_used: int
    note: str


class AnomalyPoint(BaseModel):
    timestamp: Optional[str] = None
    actual_power_W: float
    predicted_power_W: float


class AnomalyRequest(BaseModel):
    points: List[AnomalyPoint]
    drop_pct: float = 0.20
    min_pred_W: float = 5.0
    window: int = 6
    min_hits: int = 3


class AnomalyResponse(BaseModel):
    n_points: int
    n_anomaly_hours: int
    n_alerts: int
    alerts: List[Dict[str, Any]]


class CleaningRequest(BaseModel):
    expected_energy_kwh: float
    actual_energy_kwh: float
    cleaning_cost: float = 50.0
    electricity_price: float = 2.0
    recoverable_fraction: float = 0.8
    horizon_days: int = 7
    plant_scale: float = 1.0


class CleaningResponse(BaseModel):
    decision: str
    net_benefit: float
    estimated_loss_kwh: float
    recoverable_energy_kwh: float
    revenue_gain: float
    assumptions: Dict[str, float]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "n_features": len(_features) if _features else 0,
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if _model is None:
        try:
            load_artifacts()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    row = {f: float(req.features.get(f, 0.0)) for f in _features}
    X = pd.DataFrame([row], columns=_features)
    pred = float(_model.predict(X)[0])
    pred = max(pred, 0.0)

    if req.use_blend and req.persist_value is not None:
        w = float(_blend.get("blend_w_model", 1.0))
        pred = w * pred + (1 - w) * float(req.persist_value)
        pred = max(pred, 0.0)

    return ForecastResponse(
        predicted_power_W=round(pred, 4),
        model="best_forecast_24h",
        n_features_used=len(_features),
        note="Prediction is module-scale watts; not a plant total.",
    )


@app.post("/anomaly", response_model=AnomalyResponse)
def anomaly(req: AnomalyRequest):
    if not req.points:
        raise HTTPException(400, "points list is empty")

    actual = np.array([p.actual_power_W for p in req.points], dtype=float)
    predicted = np.array([p.predicted_power_W for p in req.points], dtype=float)
    ts = [p.timestamp for p in req.points]

    under = (predicted >= req.min_pred_W) & (
        actual <= predicted * (1.0 - req.drop_pct)
    )
    hits = (
        pd.Series(under.astype(int))
        .rolling(req.window, min_periods=1)
        .sum()
        .values
    )
    alert = hits >= req.min_hits

    alerts = []
    for i, flag in enumerate(alert):
        if flag:
            alerts.append(
                {
                    "timestamp": ts[i],
                    "actual_power_W": float(actual[i]),
                    "predicted_power_W": float(predicted[i]),
                    "residual_W": float(actual[i] - predicted[i]),
                    "anomaly_count_window": float(hits[i]),
                    "reason": (
                        f"Actual ≥{int(req.drop_pct*100)}% below forecast "
                        f"for ≥{req.min_hits} of last {req.window} hours"
                    ),
                }
            )

    return AnomalyResponse(
        n_points=len(req.points),
        n_anomaly_hours=int(under.sum()),
        n_alerts=len(alerts),
        alerts=alerts,
    )


@app.post("/cleaning", response_model=CleaningResponse)
def cleaning(req: CleaningRequest):
    exp = req.expected_energy_kwh * req.plant_scale
    act = req.actual_energy_kwh * req.plant_scale
    loss = max(exp - act, 0.0)
    recovered = loss * req.recoverable_fraction
    revenue = recovered * req.electricity_price
    net = revenue - req.cleaning_cost
    return CleaningResponse(
        decision="Clean" if net > 0 else "Wait",
        net_benefit=round(net, 4),
        estimated_loss_kwh=round(loss, 4),
        recoverable_energy_kwh=round(recovered, 4),
        revenue_gain=round(revenue, 4),
        assumptions={
            "electricity_price": req.electricity_price,
            "cleaning_cost": req.cleaning_cost,
            "recoverable_fraction": req.recoverable_fraction,
            "plant_scale": req.plant_scale,
            "horizon_days": float(req.horizon_days),
        },
    )


@app.get("/demo/test_summary")
def demo_test_summary():
    """Optional: summary from saved test predictions CSV if present."""
    if not PRED_PATH.exists():
        raise HTTPException(404, f"{PRED_PATH} not found — run training notebooks")
    df = pd.read_csv(PRED_PATH, index_col=0, parse_dates=True)
    return {
        "n_rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "mae": float(np.mean(np.abs(df["actual_power_W"] - df["predicted_power_W"]))),
        "columns": list(df.columns),
    }

# =========================
# Thermal Fault Detection
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

THERMAL_DIR = (
    PROJECT_ROOT
    / "Solar-Panel-Anomaly-Detection-main"
    / "Solar-Panel-Anomaly-Detection-main"
)

THERMAL_MODEL_PATH = THERMAL_DIR / "models" / "best_solar_cnn.keras"
CLASS_MAPPING_PATH = THERMAL_DIR / "class_mapping.json"

_thermal_model = None
_class_mapping = None


def load_thermal_model():
    """Load the thermal CNN and its class mapping."""

    global _thermal_model, _class_mapping

    if _thermal_model is not None:
        return

    if not THERMAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Thermal model not found: {THERMAL_MODEL_PATH}"
        )

    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_MAPPING_PATH}"
        )

    import tensorflow as tf

    _thermal_model = tf.keras.models.load_model(
        THERMAL_MODEL_PATH,
        compile=False
    )

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        _class_mapping = json.load(f)


def get_class_names():
    """Return class names ordered by their model output index."""

    mapping = _class_mapping

    if not isinstance(mapping, dict):
        raise ValueError("class_mapping.json must contain a JSON object.")

    # Actual project format: {"idx_to_class": {"0": "Cell", ...}}
    if "idx_to_class" in mapping:
        indexed = mapping["idx_to_class"]

        if isinstance(indexed, dict):
            try:
                indexed = {
                    int(key): value
                    for key, value in indexed.items()
                }

                if sorted(indexed.keys()) == list(range(len(indexed))):
                    return [
                        str(indexed[i])
                        for i in range(len(indexed))
                    ]
            except (ValueError, TypeError):
                pass

        raise ValueError("Invalid idx_to_class mapping.")

    # Alternative format: {"class_to_idx": {"Cell": 0, ...}}
    if "class_to_idx" in mapping:
        indexed = mapping["class_to_idx"]

        try:
            indexed = {
                int(index): name
                for name, index in indexed.items()
            }

            if sorted(indexed.keys()) == list(range(len(indexed))):
                return [
                    str(indexed[i])
                    for i in range(len(indexed))
                ]
        except (ValueError, TypeError):
            pass

        raise ValueError("Invalid class_to_idx mapping.")

    raise ValueError(
        "Could not find idx_to_class or class_to_idx in class_mapping.json."
    )


@app.post("/thermal/predict")
async def predict_thermal(file: UploadFile = File(...)):
    """
    Classify an uploaded thermal solar-panel image.
    """

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        load_thermal_model()

        # Match the CNN's expected input dimensions.
        input_shape = _thermal_model.input_shape

        if isinstance(input_shape, list):
            raise ValueError(
                "This endpoint expects a model with one image input."
            )

        height, width = input_shape[1], input_shape[2]

        if height is None or width is None:
            raise ValueError(
                "The model does not specify fixed image dimensions."
            )

        image = image.resize((int(width), int(height)))

        # Match the app's image normalization.
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = _thermal_model.predict(
            image_array,
            verbose=0
        )[0]

        class_names = get_class_names()

        if len(class_names) != len(predictions):
            raise ValueError(
                f"Class mapping contains {len(class_names)} classes, "
                f"but the model returned {len(predictions)} scores."
            )

        predicted_index = int(np.argmax(predictions))

        return {
            "filename": file.filename,
            "predicted_class": class_names[predicted_index],
            "confidence": float(predictions[predicted_index]),
            "class_scores": {
                class_names[i]: float(score)
                for i, score in enumerate(predictions)
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Thermal prediction failed: {str(e)}"
        )
    