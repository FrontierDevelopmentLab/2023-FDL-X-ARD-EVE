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


def test_predict_returns_38_ions_for_known_timestamp(client):
    info = client.get("/info").json()
    valid_ts = info["available_dates"]["min"]

    resp = client.post("/predict", json={"timestamp": valid_ts})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 38
    for v in body["predictions"].values():
        assert isinstance(v, (int, float))


def test_predict_snaps_to_nearest_indexed_timestamp(client):
    """A request 5 minutes off the index grid snaps onto the grid."""
    info = client.get("/info").json()
    near_min = info["available_dates"]["min"]
    # Add 5 minutes to push the request off the 36-minute grid
    from datetime import datetime, timedelta
    bumped = (datetime.fromisoformat(near_min) + timedelta(minutes=5)).isoformat()

    resp = client.post("/predict", json={"timestamp": bumped})
    assert resp.status_code == 200
    body = resp.json()
    # The response timestamp is the snapped value, not the bumped request
    assert body["timestamp"] != bumped


def test_predict_out_of_range_returns_422(client):
    resp = client.post("/predict", json={"timestamp": "1999-01-01T00:00:00"})
    assert resp.status_code == 422
    assert "out of range" in resp.text.lower() or "outside" in resp.text.lower()


def test_predict_invalid_iso_returns_422(client):
    resp = client.post("/predict", json={"timestamp": "not a date"})
    assert resp.status_code == 422


def test_predict_range_returns_records_for_valid_range(client):
    info = client.get("/info").json()
    start = info["available_dates"]["min"]
    # 1 hour after start -> should include ~2 timestamps at 36-min cadence
    from datetime import datetime, timedelta
    end = (datetime.fromisoformat(start) + timedelta(hours=1)).isoformat()

    resp = client.post("/predict-range", json={"start": start, "end": end})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    assert len(body["predictions"]) == body["count"]
    # Each record carries the timestamp + 38 ions
    first = body["predictions"][0]
    assert "timestamp" in first
    assert len(first) == 39


def test_predict_range_empty_returns_zero_count(client):
    resp = client.post(
        "/predict-range",
        json={"start": "1999-01-01T00:00:00", "end": "1999-01-01T01:00:00"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"count": 0, "predictions": []}


def test_predict_range_inverted_returns_422(client):
    resp = client.post(
        "/predict-range",
        json={"start": "2017-09-06T12:00:00", "end": "2017-09-06T00:00:00"},
    )
    assert resp.status_code == 422
