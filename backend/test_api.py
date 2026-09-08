import math
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    """Valida el estado de salud del backend y carga de artefactos."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert data["mlp_loaded"] is True


def test_predict_endpoint_rf():
    """Valida la inferencia exitosa utilizando el modelo Random Forest."""
    payload = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "one",
        "arrival_time": "Night",
        "destination_city": "Mumbai",
        "class": "Economy",
        "duration": 2.5,
        "days_left": 15,
        "model_type": "rf",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, f"Error en inferencia RF: {response.text}"
    data = response.json()
    assert "predicted_price_inr" in data
    assert data["predicted_price_inr"] > 0
    assert "predicted_price_usd" in data
    assert "predicted_price_clp" in data
    assert "formatted_price" in data
    assert "latency_ms" in data
    assert data["status"] == "success"
    assert "Random Forest" in data["model_used"]


def test_predict_endpoint_mlp():
    """Valida la inferencia exitosa utilizando la Red Neuronal MLPRegressor."""
    payload = {
        "airline": "Air_India",
        "source_city": "Mumbai",
        "departure_time": "Evening",
        "stops": "zero",
        "arrival_time": "Night",
        "destination_city": "Delhi",
        "class": "Business",
        "duration": 2.1,
        "days_left": 10,
        "model_type": "mlp",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, f"Error en inferencia MLP: {response.text}"
    data = response.json()
    assert "predicted_price_inr" in data
    assert data["predicted_price_inr"] > 0
    assert "predicted_price_usd" in data
    assert "predicted_price_clp" in data
    assert "formatted_price" in data
    assert "latency_ms" in data
    assert data["status"] == "success"
    assert "MLP" in data["model_used"] or "Neuronal" in data["model_used"]


def test_explain_endpoint():
    """Valida la explicabilidad SHAP real mediante TreeExplainer sobre Random Forest."""
    payload = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "one",
        "arrival_time": "Night",
        "destination_city": "Mumbai",
        "class": "Economy",
        "duration": 2.5,
        "days_left": 15,
    }
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 200, f"Error en explicación SHAP: {response.text}"
    data = response.json()
    assert "base_price_inr" in data
    assert "predicted_price_inr" in data
    assert "contributions" in data
    
    # Validar que los valores sean finitos y numéricamente consistentes
    assert not math.isnan(data["base_price_inr"])
    assert not math.isinf(data["base_price_inr"])
    assert data["base_price_inr"] > 0
    assert not math.isnan(data["predicted_price_inr"])
    assert not math.isinf(data["predicted_price_inr"])
    
    contributions = data["contributions"]
    assert isinstance(contributions, list)
    assert len(contributions) > 0
    for item in contributions:
        assert "feature" in item
        assert "contribution" in item
        assert not math.isnan(item["contribution"])
        assert item["direction"] in ["increases_price", "decreases_price"]


def test_predict_invalid_category_returns_400_or_422():
    """Valida el rechazo estricto ante aerolíneas inexistentes."""
    payload = {
        "airline": "AerolineaFalsa",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "one",
        "arrival_time": "Night",
        "destination_city": "Mumbai",
        "class": "Economy",
        "duration": 2.5,
        "days_left": 15,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in [400, 422]


def test_predict_invalid_days_left_out_of_range():
    """Valida el rechazo ante days_left fuera del rango permitido (1-50)."""
    payload = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "one",
        "arrival_time": "Night",
        "destination_city": "Mumbai",
        "class": "Economy",
        "duration": 2.5,
        "days_left": 999,  # Inválido (> 50)
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_duration_negative():
    """Valida el rechazo ante duration menor al límite inferior permitido."""
    payload = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "one",
        "arrival_time": "Night",
        "destination_city": "Mumbai",
        "class": "Economy",
        "duration": -2.0,  # Inválido (< 0.1)
        "days_left": 10,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint():
    """Valida el endpoint de telemetría y métricas operacionales."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "total_explanations" in data
    assert "avg_latency_ms" in data
    assert "target_latency_rnf02_met" in data
    assert "model_metadata" in data
    assert data["model_metadata"]["r2_score"] > 0.95


def test_middleware_process_time_header():
    """Valida que el middleware inyecte la cabecera X-Process-Time-MS."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "X-Process-Time-MS" in response.headers
    val = float(response.headers["X-Process-Time-MS"])
    assert val >= 0.0
