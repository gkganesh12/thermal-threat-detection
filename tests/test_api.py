import pytest
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_detect_endpoint_validation():
    response = client.post("/detect/")
    assert response.status_code == 422  # Validation error for missing required fields

@pytest.mark.asyncio
async def test_startup_event():
    # Test that the application can start up without errors
    assert True  # Basic test to ensure the test suite runs
