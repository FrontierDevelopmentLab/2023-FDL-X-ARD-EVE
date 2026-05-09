"""Thin HTTP client wrapping the FastAPI service.

The client returns plain dicts for /info, /predict, /health and a
pandas.DataFrame for /predict-range so the existing Streamlit plot
code works unchanged.
"""

from datetime import datetime

import httpx
import pandas as pd


class APIClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def info(self) -> dict:
        resp = self._client.get("/info")
        resp.raise_for_status()
        return resp.json()

    def predict(self, timestamp: datetime) -> dict:
        resp = self._client.post("/predict", json={"timestamp": timestamp.isoformat()})
        resp.raise_for_status()
        return resp.json()

    def predict_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        resp = self._client.post(
            "/predict-range",
            json={"start": start.isoformat(), "end": end.isoformat()},
        )
        resp.raise_for_status()
        body = resp.json()
        if body["count"] == 0:
            return pd.DataFrame()
        df = pd.DataFrame(body["predictions"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def health(self) -> dict:
        resp = self._client.get("/health")
        # /health returns 503 during startup; we still want the body to surface
        return resp.json()
