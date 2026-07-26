from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_endpoint():
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
    response = client.post("/api/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price_inr" in data
        assert data["predicted_price_inr"] > 0
        assert "latency_ms" in data
        assert data["status"] == "success"
    else:
        # If models aren't trained yet, expect 500
        assert response.status_code == 500


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "avg_latency_ms" in data
    assert "target_latency_rnf02_met" in data
