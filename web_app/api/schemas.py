"""Pydantic request / response models for the FastAPI service."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class AvailableDates(BaseModel):
    min: datetime
    max: datetime


class InfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    aia_wavelengths: list[str]
    eve_ions: list[str]
    available_dates: AvailableDates


class PredictRequest(BaseModel):
    timestamp: datetime


class PredictResponse(BaseModel):
    timestamp: datetime
    predictions: dict[str, float]


class PredictRangeRequest(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _check_range(self) -> "PredictRangeRequest":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class PredictRangeResponse(BaseModel):
    count: int
    predictions: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: Literal["ready", "starting"]
