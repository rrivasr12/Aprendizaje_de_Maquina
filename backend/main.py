import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# App Initialization
app = FastAPI(
    title="Flight Price Prediction & Explainability API",
    description="Backend API REST para predicción de tarifas aéreas con alternancia entre Random Forest y Red Neuronal MLP, explicabilidad SHAP real y monitoreo operacional.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Telemetry Metrics Storage
system_metrics = {
    "total_predictions": 0,
    "total_explanations": 0,
    "latencies_ms": [],
    "model_name": "Random Forest Regressor (Tuned)",
    "r2_score": 0.9769,
    "rmse_inr": 3436.21,
    "mae_inr": 1784.55,
    "start_time": time.time(),
}

# Artifacts & Models
preprocessor = None
y_scaler = None
rf_model = None
mlp_model = None
feature_names = []
explainer = None


def load_artifacts():
    global preprocessor, y_scaler, rf_model, mlp_model, feature_names, explainer
    if preprocessor is None:
        try:
            preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
            y_scaler = joblib.load(MODELS_DIR / "y_scaler.joblib")
            rf_model = joblib.load(MODELS_DIR / "best_rf_model.joblib")
            
            if (MODELS_DIR / "best_mlp_model.joblib").exists():
                mlp_model = joblib.load(MODELS_DIR / "best_mlp_model.joblib")
                
            feature_names = list(preprocessor.get_feature_names_out())
            
            # Inicializar SHAP TreeExplainer sobre el modelo Random Forest
            try:
                explainer = shap.TreeExplainer(rf_model)
                print("[OK] SHAP TreeExplainer inicializado correctamente sobre el modelo Random Forest.")
            except Exception as se:
                print(f"[AVISO] SHAP TreeExplainer init: {se}")
                
            print("[OK] Artefactos de modelos (Random Forest & MLP) cargados exitosamente.")
        except Exception as e:
            print(f"Error cargando artefactos: {e}")


# Validaciones Pydantic
VALID_AIRLINES = {"SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"}
VALID_CITIES = {"Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"}
VALID_TIMES = {"Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"}
VALID_STOPS = {"zero", "one", "two_or_more"}
VALID_CLASSES = {"Economy", "Business"}


class FlightPredictionInput(BaseModel):
    airline: str = Field(..., description="Nombre de la aerolínea", example="Vistara")
    source_city: str = Field(..., description="Ciudad de origen", example="Delhi")
    departure_time: str = Field(..., description="Horario de salida", example="Morning")
    stops: str = Field(..., description="Número de escalas", example="one")
    arrival_time: str = Field(..., description="Horario de llegada", example="Night")
    destination_city: str = Field(..., description="Ciudad de destino", example="Mumbai")
    class_name: str = Field(..., alias="class", description="Clase de cabina (Economy/Business)", example="Economy")
    duration: float = Field(..., ge=0.1, le=50.0, description="Duración estimada del vuelo en horas", example=2.17)
    days_left: int = Field(..., ge=1, le=50, description="Días de antelación de la reserva (1-50)", example=1)
    model_type: Optional[str] = Field("rf", description="Modelo a usar: 'rf' (Random Forest) o 'mlp' (Red Neuronal MLP)", example="rf")

    @field_validator("airline")
    def validate_airline(cls, v):
        if v not in VALID_AIRLINES:
            raise ValueError(f"Aerolínea no válida. Opciones permitidas: {VALID_AIRLINES}")
        return v

    @field_validator("source_city", "destination_city")
    def validate_city(cls, v):
        if v not in VALID_CITIES:
            raise ValueError(f"Ciudad no válida. Opciones permitidas: {VALID_CITIES}")
        return v

    @field_validator("departure_time", "arrival_time")
    def validate_time(cls, v):
        if v not in VALID_TIMES:
            raise ValueError(f"Horario no válido. Opciones permitidas: {VALID_TIMES}")
        return v

    @field_validator("stops")
    def validate_stops(cls, v):
        if v not in VALID_STOPS:
            raise ValueError(f"Escala no válida. Opciones permitidas: {VALID_STOPS}")
        return v

    @field_validator("class_name")
    def validate_class(cls, v):
        if v not in VALID_CLASSES:
            raise ValueError(f"Clase no válida. Opciones permitidas: {VALID_CLASSES}")
        return v

    class Config:
        populate_by_name = True


class FlightPredictionResponse(BaseModel):
    predicted_price_inr: float
    formatted_price: str
    predicted_price_usd: float
    predicted_price_clp: float
    model_used: str
    latency_ms: float
    status: str


class SHAPContribution(BaseModel):
    feature: str
    contribution: float
    direction: str  # 'increases_price' or 'decreases_price'


class SHAPExplanationResponse(BaseModel):
    base_price_inr: float
    predicted_price_inr: float
    contributions: List[SHAPContribution]


# Middleware para monitoreo de latencia (X-Process-Time-MS)
@app.middleware("http")
async def track_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-MS"] = f"{process_time_ms:.2f}"
    return response


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_loaded": rf_model is not None,
        "mlp_loaded": mlp_model is not None,
    }


@app.post("/api/predict", response_model=FlightPredictionResponse)
def predict_flight_price(input_data: FlightPredictionInput):
    load_artifacts()
    t0 = time.time()

    if rf_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Los modelos no han sido cargados. Ejecute python scripts/train.py primero.")

    input_dict = {
        "airline": [input_data.airline],
        "source_city": [input_data.source_city],
        "departure_time": [input_data.departure_time],
        "stops": [input_data.stops],
        "arrival_time": [input_data.arrival_time],
        "destination_city": [input_data.destination_city],
        "class": [input_data.class_name],
        "duration": [input_data.duration],
        "days_left": [input_data.days_left],
    }

    df_single = pd.DataFrame(input_dict)
    X_proc = preprocessor.transform(df_single)

    # Seleccionar modelo de inferencia: Random Forest o Red Neuronal MLP
    selected_model_name = "Random Forest Regressor (Tuned)"
    if input_data.model_type in ["mlp", "neural_network", "mlp_regressor"] and mlp_model is not None:
        pred_scaled = mlp_model.predict(X_proc)
        pred_inr = float(y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        selected_model_name = "Red Neuronal Profunda (MLPRegressor)"
    else:
        pred_inr = float(rf_model.predict(X_proc)[0])

    pred_inr = max(0.0, pred_inr)
    latency_ms = (time.time() - t0) * 1000

    system_metrics["total_predictions"] += 1
    system_metrics["latencies_ms"].append(latency_ms)

    pred_usd = round(pred_inr * 0.012, 2)
    pred_clp = round(pred_inr * 11.5, 0)
    formatted_str = f"₹ {pred_inr:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    return FlightPredictionResponse(
        predicted_price_inr=round(pred_inr, 2),
        formatted_price=formatted_str,
        predicted_price_usd=pred_usd,
        predicted_price_clp=pred_clp,
        model_used=selected_model_name,
        latency_ms=round(latency_ms, 2),
        status="success",
    )


@app.post("/api/explain", response_model=SHAPExplanationResponse)
def explain_flight_prediction(input_data: FlightPredictionInput):
    """
    Transforma la entrada mediante el pipeline de scikit-learn y calcula los SHAP values
    reales utilizando shap.TreeExplainer(model) sobre la muestra transformada.
    """
    load_artifacts()

    if rf_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Los modelos no están disponibles.")

    input_dict = {
        "airline": [input_data.airline],
        "source_city": [input_data.source_city],
        "departure_time": [input_data.departure_time],
        "stops": [input_data.stops],
        "arrival_time": [input_data.arrival_time],
        "destination_city": [input_data.destination_city],
        "class": [input_data.class_name],
        "duration": [input_data.duration],
        "days_left": [input_data.days_left],
    }
    df_single = pd.DataFrame(input_dict)
    
    # 1. Transformar input con el pipeline de scikit-learn
    X_transformed = preprocessor.transform(df_single)

    pred_inr = float(rf_model.predict(X_transformed)[0])

    global explainer
    try:
        if explainer is None:
            explainer = shap.TreeExplainer(rf_model)
            
        # 2. Calcular los SHAP values reales sobre X_transformed
        raw_shap = explainer.shap_values(X_transformed)
        
        if isinstance(raw_shap, list):
            sample_shap = raw_shap[0][0]
        elif len(raw_shap.shape) == 2:
            sample_shap = raw_shap[0]
        else:
            sample_shap = raw_shap

        base_val = explainer.expected_value
        if isinstance(base_val, (np.ndarray, list)):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)

        # 3. Retornar mapa de atributos con la contribución exacta en Rupias
        contributions = []
        for name, val in zip(feature_names, sample_shap):
            contrib_val = float(val)
            clean_name = name.replace("cat__", "").replace("num__", "")
            
            if abs(contrib_val) >= 1.0:
                contributions.append(
                    SHAPContribution(
                        feature=clean_name,
                        contribution=round(contrib_val, 2),
                        direction="increases_price" if contrib_val > 0 else "decreases_price",
                    )
                )

        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        system_metrics["total_explanations"] += 1

        return SHAPExplanationResponse(
            base_price_inr=round(base_val, 2),
            predicted_price_inr=round(pred_inr, 2),
            contributions=contributions[:10],
        )

    except Exception as e:
        print(f"Error al calcular SHAP real: {e}")
        raise HTTPException(status_code=500, detail=f"Error al calcular valores SHAP reales: {str(e)}")


@app.get("/api/metrics")
def get_system_telemetry():
    latencies = system_metrics["latencies_ms"]
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    min_lat = float(np.min(latencies)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0

    uptime_sec = round(time.time() - system_metrics["start_time"], 1)

    return {
        "total_predictions": system_metrics["total_predictions"],
        "total_explanations": system_metrics["total_explanations"],
        "avg_latency_ms": round(avg_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "target_latency_rnf02_met": avg_lat < 200.0,
        "model_metadata": {
            "primary_model": system_metrics["model_name"],
            "r2_score": system_metrics["r2_score"],
            "rmse_inr": system_metrics["rmse_inr"],
            "mae_inr": system_metrics["mae_inr"],
        },
        "system_health": "Optimal" if (avg_lat < 200.0 or not latencies) else "Degraded",
        "uptime_seconds": uptime_sec,
    }


# Montar archivos estáticos
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
