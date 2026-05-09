"""End-to-end HTTP tests for the FastAPI service.

These tests need a configured DATA_BACKEND (local with a real
LOCAL_DATA_ROOT, or s3) so the lifespan can load the model and time
index. On a dev machine without data access, skip with
`-k 'not requires_data'`.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ready_after_lifespan(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_info_returns_metadata(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()

    assert body["model_name"] == "AIA_MEGS_20_30_epochs_36min"
    assert len(body["aia_wavelengths"]) == 9
    assert len(body["eve_ions"]) == 38
    assert "min" in body["available_dates"]
    assert "max" in body["available_dates"]
