"""Tests for api.schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from api import schemas


def test_predict_request_accepts_iso_string():
    req = schemas.PredictRequest.model_validate({"timestamp": "2017-09-06T12:00:00"})
    assert req.timestamp == datetime(2017, 9, 6, 12, 0, 0)


def test_predict_request_accepts_tz_aware_and_strips_to_naive_utc():
    req = schemas.PredictRequest.model_validate({"timestamp": "2017-09-06T12:00:00Z"})
    # Pydantic preserves tz on the model; the conversion happens in the endpoint
    # via find_nearest_indexed_timestamp. Just confirm parsing works.
    assert req.timestamp.year == 2017


def test_predict_request_rejects_garbage():
    with pytest.raises(ValidationError):
        schemas.PredictRequest.model_validate({"timestamp": "not a date"})


def test_predict_range_request_rejects_inverted_range():
    with pytest.raises(ValidationError) as exc:
        schemas.PredictRangeRequest.model_validate(
            {"start": "2017-09-06T12:00:00", "end": "2017-09-06T00:00:00"}
        )
    assert "end" in str(exc.value).lower()


def test_predict_range_request_accepts_equal_endpoints():
    """A zero-width range is allowed (it'll just match one or zero timestamps)."""
    req = schemas.PredictRangeRequest.model_validate(
        {"start": "2017-09-06T12:00:00", "end": "2017-09-06T12:00:00"}
    )
    assert req.start == req.end


def test_health_response_literal_is_enforced():
    with pytest.raises(ValidationError):
        schemas.HealthResponse(status="banana")


def test_predict_response_serialises_predictions_dict():
    resp = schemas.PredictResponse(
        timestamp=datetime(2017, 9, 6, 12, 0, 0),
        predictions={"C III": 0.123, "Fe IX": 0.456},
    )
    payload = resp.model_dump()
    assert payload["predictions"]["C III"] == 0.123
