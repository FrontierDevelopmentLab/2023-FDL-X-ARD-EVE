"""Tests for ui.api_client. These tests don't touch the real API — they use
httpx's MockTransport to assert the client constructs requests correctly."""

from datetime import datetime

import httpx
import pandas as pd
import pytest

from ui.api_client import APIClient


def _client_with(mock_handler) -> APIClient:
    transport = httpx.MockTransport(mock_handler)
    c = APIClient(base_url="http://test")
    c._client = httpx.Client(base_url="http://test", transport=transport)
    return c


def test_info_returns_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/info"
        return httpx.Response(200, json={
            "model_name": "test",
            "aia_wavelengths": ["94A"],
            "eve_ions": ["Fe IX"],
            "available_dates": {"min": "2017-01-01T00:00:00", "max": "2017-01-02T00:00:00"},
        })

    c = _client_with(handler)
    info = c.info()
    assert info["model_name"] == "test"


def test_predict_posts_timestamp_and_returns_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        assert request.method == "POST"
        body = request.read().decode()
        assert "2017-09-06T12:00:00" in body
        return httpx.Response(200, json={
            "timestamp": "2017-09-06T12:00:00",
            "predictions": {"Fe IX": 1.23},
        })

    c = _client_with(handler)
    result = c.predict(datetime(2017, 9, 6, 12, 0, 0))
    assert result["predictions"]["Fe IX"] == 1.23


def test_predict_range_returns_dataframe_with_timestamp_index():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "count": 2,
            "predictions": [
                {"timestamp": "2017-09-06T00:00:00", "Fe IX": 1.0},
                {"timestamp": "2017-09-06T00:36:00", "Fe IX": 2.0},
            ],
        })

    c = _client_with(handler)
    df = c.predict_range(datetime(2017, 9, 6, 0, 0), datetime(2017, 9, 6, 1, 0))
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "timestamp"
    assert len(df) == 2
    assert df.iloc[0]["Fe IX"] == 1.0


def test_predict_range_empty_returns_empty_dataframe():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"count": 0, "predictions": []})

    c = _client_with(handler)
    df = c.predict_range(datetime(1999, 1, 1), datetime(1999, 1, 2))
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_health_returns_status_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ready"})

    c = _client_with(handler)
    assert c.health() == {"status": "ready"}
