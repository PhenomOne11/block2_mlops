import base64
import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from src.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client
        
def test_healthz(client):
    res=client.get("/healthz")
    assert res.status_code==200
    assert res.json()["status"]=="ready"
    
def test_metrics(client):
    res=client.get("/metrics")
    assert res.status_code==200
    assert "inference_request_total" in res.text
    
def test_predict_tensor(client):
    dummy_tensor=np.random.randn(3,32,32).tolist()
    res=client.post("/predict", json={"tensor": dummy_tensor})
    assert res.status_code==200
    data=res.json()
    assert "class_id" in data
    assert "class_name" in data
    assert "confidence" in data
    assert len(data["top_3"]) == 3
    assert data["batched_size"] >= 1
    
def test_predict_b64(client):
    img = Image.new("RGB", (32, 32), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    res = client.post("/predict", json={"image_b64": b64_str})
    assert res.status_code == 200
    data = res.json()
    assert data["confidence"] > 0.0
    assert data["class_name"] in [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]


def test_predict_invalid_payload(client):
    res = client.post("/predict", json={})
    assert res.status_code == 400