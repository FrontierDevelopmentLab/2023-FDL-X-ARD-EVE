# Web App API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a FastAPI service to the Virtual EVE `web_app/` so external clients can query predictions programmatically and the Streamlit UI stops loading the model itself.

**Architecture:** Restructure `web_app/` into `core/` (shared inference + data access), `api/` (FastAPI service), and `ui/` (Streamlit). Two services in `docker-compose.yml`; UI calls API for predictions and metadata, reads AIA images directly from the Zarr store. Time-index cache moves from CSV to parquet.

**Tech Stack:** FastAPI 0.118, uvicorn 0.30, pydantic 2.9, httpx 0.27, pytorch (existing), zarr (existing), streamlit (existing), pyarrow (new for parquet cache).

**Spec:** `docs/superpowers/specs/2026-05-09-web-app-api-design.md`

## Notes for the implementer

- All paths are relative to the repository root unless otherwise stated.
- All `pytest` commands assume CWD = `web_app/`.
- Tests that exercise the FastAPI lifespan need a configured data backend (`DATA_BACKEND=s3` or `DATA_BACKEND=local` with `LOCAL_DATA_ROOT` pointed at a real Zarr). On a fresh machine with `DATA_BACKEND=s3`, the first run builds the time index from S3 metadata (slow — minutes). Subsequent runs read the parquet cache (fast). If you're developing on a machine without data access, deploy first then run integration tests on the server.
- Tests that don't touch the lifespan (schema validators, pure unit tests) run anywhere.
- Pure file moves use `git mv` so history is preserved.
- Each task ends with a commit. Commit messages are concrete; no `Co-Authored-By` trailer (per the repo's global preferences).

## File structure (post-implementation)

```
web_app/
├── conftest.py                  # NEW: makes core/api/ui importable from tests
├── pytest.ini                   # NEW: testpaths = api/tests
├── docker-compose.yml           # REWRITTEN: api + ui services
├── requirements.txt             # DELETED (replaced by api/, ui/ requirements)
├── Dockerfile                   # DELETED (replaced by api/, ui/ Dockerfiles)
├── README.md                    # MINOR UPDATE
├── cache/                       # owned by api (writes); ui mounts read-only
├── core/
│   ├── __init__.py              # NEW: empty
│   ├── inference.py             # MOVED from web_app/inference.py
│   ├── data_access.py           # MOVED, MODIFIED: parquet cache, helpers
│   ├── model.py                 # MOVED from web_app/model.py
│   └── checkpoints/             # MOVED from web_app/checkpoints/
├── api/
│   ├── __init__.py              # NEW: empty
│   ├── main.py                  # NEW: FastAPI app + lifespan
│   ├── schemas.py               # NEW: Pydantic models
│   ├── Dockerfile               # NEW
│   ├── requirements.txt         # NEW
│   └── tests/
│       ├── __init__.py          # NEW: empty
│       ├── conftest.py          # NEW: shared fixtures
│       ├── test_schemas.py      # NEW
│       ├── test_data_access.py  # NEW: parquet cache test
│       └── test_api.py          # NEW: HTTP endpoint tests
└── ui/
    ├── __init__.py              # NEW: empty
    ├── main.py                  # MOVED, REFACTORED to consume API
    ├── api_client.py            # NEW: httpx wrapper
    ├── assets/                  # MOVED from web_app/assets/
    ├── Dockerfile               # MOVED, MODIFIED
    └── requirements.txt         # NEW: httpx, no torch
```

---

## Task 1: Restructure web_app/ into core/, api/, ui/

**Files:**
- Create: `web_app/core/__init__.py`, `web_app/api/__init__.py`, `web_app/ui/__init__.py`, `web_app/api/tests/__init__.py`
- Move: `web_app/inference.py` → `web_app/core/inference.py`
- Move: `web_app/data_access.py` → `web_app/core/data_access.py`
- Move: `web_app/model.py` → `web_app/core/model.py`
- Move: `web_app/checkpoints/` → `web_app/core/checkpoints/`
- Move: `web_app/main.py` → `web_app/ui/main.py`
- Move: `web_app/assets/` → `web_app/ui/assets/`
- Move: `web_app/Dockerfile` → `web_app/ui/Dockerfile`
- Modify: `web_app/core/inference.py` (import + cache-path adjustments)
- Modify: `web_app/core/data_access.py` (cache path adjustment)
- Modify: `web_app/ui/main.py` (imports)

- [ ] **Step 1: Create new package directories**

```bash
cd web_app
mkdir -p core api ui api/tests
touch core/__init__.py api/__init__.py ui/__init__.py api/tests/__init__.py
```

- [ ] **Step 2: Move existing source files with `git mv`**

```bash
cd web_app
git mv inference.py core/inference.py
git mv data_access.py core/data_access.py
git mv model.py core/model.py
git mv checkpoints core/checkpoints
git mv main.py ui/main.py
git mv assets ui/assets
git mv Dockerfile ui/Dockerfile
```

- [ ] **Step 3: Update imports in `core/inference.py`**

Change line 12 from:
```python
import model as _model_module
```
to:
```python
from . import model as _model_module
```

The `sys.modules` aliasing block (lines 14–17) stays exactly as-is — it constructs the legacy module path the pickled checkpoint expects.

- [ ] **Step 4: Update cache path in `core/data_access.py`**

The cache directory now lives at `web_app/cache/` (one level up from `core/`). Change:

```python
CACHE_DIR = Path(__file__).parent / "cache"
```
to:
```python
CACHE_DIR = Path(__file__).parent.parent / "cache"
```

(`__file__` is now `web_app/core/data_access.py`; `.parent.parent` is `web_app/`.)

- [ ] **Step 5: Update imports in `ui/main.py`**

Replace the existing import block (lines 16–24):
```python
from data_access import (
    AIA_WAVELENGTHS,
    build_time_index,
    get_aia_image,
    get_aia_root,
    get_available_dates,
    get_timestamps_in_range,
)
from inference import load_model, predict_eve_timeseries
```

with:
```python
from core.data_access import (
    AIA_WAVELENGTHS,
    build_time_index,
    get_aia_image,
    get_aia_root,
    get_available_dates,
    get_timestamps_in_range,
)
from core.inference import load_model, predict_eve_timeseries
```

(These imports will be replaced again in Task 11. Keeping them working now lets us verify the move in isolation.)

- [ ] **Step 6: Adjust the asset path in `ui/main.py`**

Lines 30 and 75 reference `assets/sdo_icon.jpeg` and `assets/fdlx.png` (and similar). These are relative paths Streamlit resolves from CWD. After the move, when running `streamlit run ui/main.py` from `web_app/`, the path becomes `ui/assets/...`. Update both references:

```python
# Line ~30
page_icon="ui/assets/sdo_icon.jpeg",
# Line ~75
st.image("ui/assets/fdlx.png", width=280)
# Line ~212
st.image("ui/assets/nasa_sdo.png", width=400)
```

- [ ] **Step 7: Verify the file tree**

```bash
cd web_app
find . -maxdepth 3 -not -path '*/cache/*' -not -path '*/__pycache__/*' -not -path '*/.git/*' | sort
```

Expected (truncated):
```
./api
./api/__init__.py
./api/tests
./api/tests/__init__.py
./core
./core/__init__.py
./core/checkpoints
./core/checkpoints/AIA_MEGS_20_30_epochs_36min.ckpt
./core/data_access.py
./core/inference.py
./core/model.py
./ui
./ui/__init__.py
./ui/Dockerfile
./ui/assets
...
./ui/main.py
./docker-compose.yml
./requirements.txt
./README.md
```

- [ ] **Step 8: Commit**

```bash
git add -A web_app/
git commit -m "Restructure web_app/ into core/, api/, ui/ packages

Move inference, data access, model, and checkpoints into core/.
Move Streamlit + assets + Dockerfile into ui/. Add empty api/ that
later tasks populate.

Imports and the data_access cache path are updated to match the new
layout. Streamlit asset paths are prefixed with ui/. The pickle-time
sys.modules aliasing in inference.py is preserved unchanged so the
existing checkpoint still unpickles."
```

---

## Task 2: Add pytest infrastructure

**Files:**
- Create: `web_app/conftest.py`
- Create: `web_app/pytest.ini`
- Create: `web_app/api/tests/conftest.py`

- [ ] **Step 1: Create `web_app/conftest.py`**

This file's presence makes `web_app/` the rootdir for pytest, which adds it to `sys.path` so `from core import ...` and `from api import ...` work from any test.

```python
"""Top-level conftest. Its presence anchors pytest at web_app/ and adds it to sys.path."""
```

(Yes, just the docstring. The file's existence is what matters.)

- [ ] **Step 2: Create `web_app/pytest.ini`**

```ini
[pytest]
testpaths = api/tests
addopts = -ra --strict-markers
```

- [ ] **Step 3: Create `web_app/api/tests/conftest.py`**

```python
"""Shared fixtures for API tests."""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def _check_data_backend():
    """Sanity check: tests that hit the lifespan need a configured backend.

    Tests that mock the lifespan or exercise pure logic don't need this; they
    should still run. We only warn — we don't fail — so unit tests can run
    on a dev machine without data access.
    """
    backend = os.environ.get("DATA_BACKEND", "local")
    if backend == "local":
        root = os.environ.get("LOCAL_DATA_ROOT", "")
        if not root or not os.path.isdir(root):
            print(
                f"\n[pytest] DATA_BACKEND=local but LOCAL_DATA_ROOT={root!r} "
                "is not a directory. Tests that exercise the FastAPI "
                "lifespan will fail. Set DATA_BACKEND=s3 or point "
                "LOCAL_DATA_ROOT at a real Zarr root to enable them."
            )
```

- [ ] **Step 4: Verify pytest discovers the test directory**

Add a temporary throw-away test to confirm discovery:

```bash
cd web_app
cat > api/tests/test_smoke.py <<'EOF'
def test_pytest_runs():
    assert True
EOF
pytest -q
```

Expected output:
```
1 passed in 0.0Xs
```

Then delete the smoke test:
```bash
rm web_app/api/tests/test_smoke.py
```

- [ ] **Step 5: Commit**

```bash
git add web_app/conftest.py web_app/pytest.ini web_app/api/tests/conftest.py
git commit -m "Add pytest config and shared conftest for the web_app API tests

Pytest rootdir is anchored at web_app/ so 'from core import ...' and
'from api import ...' resolve in test files. The fixture in
api/tests/conftest.py prints a soft warning if the data backend is
not configured, since lifespan-exercising tests need real data."
```

---

## Task 3: Switch time-index cache from CSV to parquet (TDD)

**Files:**
- Create: `web_app/api/tests/test_data_access.py`
- Modify: `web_app/core/data_access.py`

- [ ] **Step 1: Write a failing test for the parquet cache**

Create `web_app/api/tests/test_data_access.py`:

```python
"""Unit tests for core.data_access cache I/O."""

from pathlib import Path

import pandas as pd

from core import data_access


def test_build_time_index_reads_parquet_cache(tmp_path, monkeypatch):
    """If the parquet cache exists, build_time_index returns it without rebuilding."""
    cache_path = tmp_path / "aia_time_index.parquet"
    monkeypatch.setattr(data_access, "INDEX_CACHE", cache_path)

    expected = pd.DataFrame(
        {
            "year": [2017, 2017],
            "idx_94A": [0, 1],
            "idx_131A": [0, 1],
            "idx_171A": [0, 1],
            "idx_193A": [0, 1],
            "idx_211A": [0, 1],
            "idx_304A": [0, 1],
            "idx_335A": [0, 1],
            "idx_1600A": [0, 1],
            "idx_1700A": [0, 1],
        },
        index=pd.DatetimeIndex(
            ["2017-09-06 00:00:00", "2017-09-06 00:36:00"], name="Time"
        ),
    )
    expected.to_parquet(cache_path)

    result = data_access.build_time_index(aia_root=None)

    pd.testing.assert_frame_equal(result, expected)


def test_index_cache_filename_is_parquet():
    """The cache constant should point at a .parquet file."""
    assert data_access.INDEX_CACHE.suffix == ".parquet"
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
cd web_app
pytest api/tests/test_data_access.py -v
```

Expected: both tests fail. The first fails with a CSV/parquet mismatch (or `read_csv` choking on a parquet file). The second fails with `AssertionError` because `INDEX_CACHE.suffix == ".csv"`.

- [ ] **Step 3: Update `core/data_access.py` to use parquet**

Change the constant on line ~29:
```python
INDEX_CACHE = CACHE_DIR / "aia_time_index.csv"
```
to:
```python
INDEX_CACHE = CACHE_DIR / "aia_time_index.parquet"
```

In `build_time_index`, replace the CSV read (lines ~103–106):
```python
if INDEX_CACHE.exists():
    df = pd.read_csv(INDEX_CACHE, parse_dates=["Time"])
    df.set_index("Time", inplace=True)
    return df
```
with:
```python
if INDEX_CACHE.exists():
    return pd.read_parquet(INDEX_CACHE)
```

Replace the CSV write near the bottom of the function (line ~155):
```python
join_series.to_csv(INDEX_CACHE)
```
with:
```python
join_series.to_parquet(INDEX_CACHE)
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
cd web_app
pytest api/tests/test_data_access.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Delete any pre-existing CSV cache**

If there's an old `web_app/cache/aia_time_index.csv`, the new code never reads it. Remove it so the next index build creates the parquet from scratch:

```bash
rm -f web_app/cache/aia_time_index.csv
```

(Skip if the file doesn't exist.)

- [ ] **Step 6: Commit**

```bash
git add web_app/core/data_access.py web_app/api/tests/test_data_access.py
git commit -m "Switch time-index cache from CSV to parquet

The cache becomes a shared artefact between the API (writer) and the
UI (reader) once the FastAPI service lands. Parquet preserves dtypes
without a parse_dates hint and round-trips the DatetimeIndex cleanly.
pyarrow is the engine and is already a transitive streamlit dep, so
the UI gets parquet support for free.

Old CSV caches are not migrated; first run rebuilds the index."
```

---

## Task 4: Add a `find_nearest_indexed_timestamp` helper to core/data_access (TDD)

The `/predict` endpoint snaps a request to the nearest indexed timestamp and reports back the snapped value. Today, `get_aia_image` does this snapping internally and returns only the image — no timestamp. Add a small helper that exposes the snapped timestamp so the API can return it.

**Files:**
- Modify: `web_app/api/tests/test_data_access.py`
- Modify: `web_app/core/data_access.py`

- [ ] **Step 1: Add a failing test for the helper**

Append to `web_app/api/tests/test_data_access.py`:

```python
import datetime as _dt


def _build_fake_index() -> pd.DataFrame:
    return pd.DataFrame(
        {"year": [2017] * 3, **{f"idx_{wl}": [0, 1, 2] for wl in data_access.AIA_WAVELENGTHS}},
        index=pd.DatetimeIndex(
            [
                "2017-09-06 00:00:00",
                "2017-09-06 00:36:00",
                "2017-09-06 01:12:00",
            ],
            name="Time",
        ),
    )


def test_find_nearest_indexed_timestamp_snaps_within_range():
    idx = _build_fake_index()
    snapped = data_access.find_nearest_indexed_timestamp(
        idx, _dt.datetime(2017, 9, 6, 0, 30, 0)
    )
    assert snapped == pd.Timestamp("2017-09-06 00:36:00")


def test_find_nearest_indexed_timestamp_returns_none_when_out_of_range():
    idx = _build_fake_index()
    snapped = data_access.find_nearest_indexed_timestamp(
        idx, _dt.datetime(2010, 1, 1, 0, 0, 0)
    )
    assert snapped is None


def test_find_nearest_indexed_timestamp_strips_timezone():
    idx = _build_fake_index()
    aware = _dt.datetime(2017, 9, 6, 0, 30, 0, tzinfo=_dt.timezone.utc)
    snapped = data_access.find_nearest_indexed_timestamp(idx, aware)
    assert snapped == pd.Timestamp("2017-09-06 00:36:00")
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
cd web_app
pytest api/tests/test_data_access.py::test_find_nearest_indexed_timestamp_snaps_within_range -v
```

Expected: `AttributeError: module 'core.data_access' has no attribute 'find_nearest_indexed_timestamp'`.

- [ ] **Step 3: Implement the helper**

Append to `web_app/core/data_access.py` (after `get_timestamps_in_range`):

```python
def find_nearest_indexed_timestamp(
    time_index: pd.DataFrame, ts
) -> pd.Timestamp | None:
    """Round ts to 36 minutes and snap to the nearest indexed timestamp.

    Returns None if the snapped value would fall outside [index.min, index.max].
    """
    ts = pd.to_datetime(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    rounded = ts.round("36min")

    if rounded < time_index.index.min() or rounded > time_index.index.max():
        return None

    idx_loc = time_index.index.get_indexer([rounded], method="nearest")
    if idx_loc[0] == -1:
        return None
    return time_index.index[idx_loc[0]]
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
cd web_app
pytest api/tests/test_data_access.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add web_app/core/data_access.py web_app/api/tests/test_data_access.py
git commit -m "Add find_nearest_indexed_timestamp helper for API snapping

The /predict endpoint needs to return the snapped timestamp it actually
ran inference on, not the raw request. This helper centralises the
36-minute rounding and the in-range bounds check, returning None when
the request falls outside the indexed range so the endpoint can map
that to a 422."
```

---

## Task 5: Add Pydantic schemas (TDD)

**Files:**
- Create: `web_app/api/schemas.py`
- Create: `web_app/api/tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for the schemas**

Create `web_app/api/tests/test_schemas.py`:

```python
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
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cd web_app
pytest api/tests/test_schemas.py -v
```

Expected: collection error (no `api.schemas` module yet).

- [ ] **Step 3: Implement the schemas**

Create `web_app/api/schemas.py`:

```python
"""Pydantic request / response models for the FastAPI service."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, model_validator


class AvailableDates(BaseModel):
    min: datetime
    max: datetime


class InfoResponse(BaseModel):
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
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
cd web_app
pytest api/tests/test_schemas.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add web_app/api/schemas.py web_app/api/tests/test_schemas.py
git commit -m "Add Pydantic schemas for the FastAPI request/response surface

Covers /info, /predict, /predict-range, and /health. The
PredictRangeRequest validator rejects inverted ranges with a 422 at
the framework boundary so endpoint code can assume a valid range.
Equal endpoints are accepted (a zero-width range is valid; it just
matches one or zero timestamps)."
```

---

## Task 6: Add FastAPI app skeleton with lifespan and `/health` (TDD)

**Files:**
- Create: `web_app/api/main.py`
- Create: `web_app/api/tests/test_api.py`

- [ ] **Step 1: Write a failing test for `/health`**

Create `web_app/api/tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
cd web_app
pytest api/tests/test_api.py -v
```

Expected: collection error (no `api.main`).

- [ ] **Step 3: Implement the FastAPI skeleton**

Create `web_app/api/main.py`:

```python
"""FastAPI service exposing Virtual EVE predictions."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core import data_access, inference
from . import schemas


_state: dict = {"ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and time index once at startup."""
    aia_root = data_access.get_aia_root()
    time_index = data_access.build_time_index(aia_root)
    model, aia_norms, wavelengths, eve_ions = inference.load_model()

    app.state.aia_root = aia_root
    app.state.time_index = time_index
    app.state.model = model
    app.state.aia_norms = aia_norms
    app.state.wavelengths = wavelengths
    app.state.eve_ions = eve_ions
    _state["ready"] = True

    yield

    _state["ready"] = False


app = FastAPI(title="Virtual EVE API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=schemas.HealthResponse)
def health():
    if _state["ready"]:
        return schemas.HealthResponse(status="ready")
    return JSONResponse(
        status_code=503,
        content={"status": "starting"},
    )
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
cd web_app
pytest api/tests/test_api.py::test_health_returns_ready_after_lifespan -v
```

Expected: PASS. (May take a minute or more on first run because the lifespan loads the model and builds the time index.)

If it fails with an import or data error: the data backend is not configured. See the "Notes for the implementer" at the top of this plan.

- [ ] **Step 5: Commit**

```bash
git add web_app/api/main.py web_app/api/tests/test_api.py
git commit -m "Add FastAPI app skeleton with lifespan and /health endpoint

Lifespan loads the model + time index once on startup and stores them
on app.state; endpoints read state from there. The _state['ready']
flag flips to true only after the load completes, so /health returns
503 during startup and 200 once the API is ready to serve."
```

---

## Task 7: Add `/info` endpoint (TDD)

**Files:**
- Modify: `web_app/api/main.py`
- Modify: `web_app/api/tests/test_api.py`

- [ ] **Step 1: Write a failing test for `/info`**

Append to `web_app/api/tests/test_api.py`:

```python
def test_info_returns_metadata(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()

    assert body["model_name"] == "AIA_MEGS_20_30_epochs_36min"
    assert len(body["aia_wavelengths"]) == 9
    assert len(body["eve_ions"]) == 38
    assert "min" in body["available_dates"]
    assert "max" in body["available_dates"]
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
cd web_app
pytest api/tests/test_api.py::test_info_returns_metadata -v
```

Expected: 404 (no `/info` route).

- [ ] **Step 3: Implement `/info`**

Append to `web_app/api/main.py`:

```python
@app.get("/info", response_model=schemas.InfoResponse)
def info():
    ti = app.state.time_index
    return schemas.InfoResponse(
        model_name="AIA_MEGS_20_30_epochs_36min",
        aia_wavelengths=app.state.wavelengths,
        eve_ions=app.state.eve_ions,
        available_dates=schemas.AvailableDates(
            min=ti.index.min().to_pydatetime(),
            max=ti.index.max().to_pydatetime(),
        ),
    )
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
cd web_app
pytest api/tests/test_api.py -v
```

Expected: both `test_health_returns_ready_after_lifespan` and `test_info_returns_metadata` pass.

- [ ] **Step 5: Commit**

```bash
git add web_app/api/main.py web_app/api/tests/test_api.py
git commit -m "Add /info endpoint exposing model + dataset metadata

Returns model name, AIA wavelength list, EVE ion list, and the
inclusive bounds of the indexed date range. Clients use this to
discover the queryable surface without trial and error."
```

---

## Task 8: Add `/predict` endpoint (TDD)

**Files:**
- Modify: `web_app/api/main.py`
- Modify: `web_app/api/tests/test_api.py`

- [ ] **Step 1: Write failing tests for `/predict`**

Append to `web_app/api/tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run the tests — confirm they fail**

```bash
cd web_app
pytest api/tests/test_api.py -k predict -v
```

Expected: 404 on `/predict` (no route yet).

- [ ] **Step 3: Implement `/predict`**

Add `import torch` to the top of `web_app/api/main.py` alongside the
existing imports (after `from contextlib import asynccontextmanager`).

Then append the endpoint to the bottom of the file:

```python
@app.post("/predict", response_model=schemas.PredictResponse)
def predict(req: schemas.PredictRequest):
    snapped = data_access.find_nearest_indexed_timestamp(
        app.state.time_index, req.timestamp
    )
    if snapped is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "timestamp is outside the indexed date range "
                f"[{app.state.time_index.index.min().isoformat()}, "
                f"{app.state.time_index.index.max().isoformat()}]"
            ),
        )

    aia_image = data_access.get_aia_image(
        app.state.aia_root, app.state.time_index, snapped
    )
    if aia_image is None:
        raise HTTPException(
            status_code=422,
            detail="no AIA data available at the snapped timestamp",
        )

    x = inference.normalize_aia_image(
        aia_image, app.state.aia_norms, app.state.wavelengths
    )
    with torch.no_grad():
        pred = app.state.model.forward_unnormalize(x).numpy()[0]

    return schemas.PredictResponse(
        timestamp=snapped.to_pydatetime(),
        predictions={ion: float(pred[i]) for i, ion in enumerate(app.state.eve_ions)},
    )
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
cd web_app
pytest api/tests/test_api.py -k predict -v
```

Expected: all four predict tests pass.

- [ ] **Step 5: Commit**

```bash
git add web_app/api/main.py web_app/api/tests/test_api.py
git commit -m "Add /predict endpoint for single-timestamp inference

The endpoint snaps the request to the nearest indexed timestamp via
core.data_access.find_nearest_indexed_timestamp and returns the
snapped value alongside the 38 ion predictions. Out-of-range and
unparseable timestamps return 422; valid in-range requests always
succeed because every indexed timestamp has corresponding data."
```

---

## Task 9: Add `/predict-range` endpoint (TDD)

**Files:**
- Modify: `web_app/api/main.py`
- Modify: `web_app/api/tests/test_api.py`

- [ ] **Step 1: Write failing tests for `/predict-range`**

Append to `web_app/api/tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run the tests — confirm they fail**

```bash
cd web_app
pytest api/tests/test_api.py -k predict_range -v
```

Expected: 404 on `/predict-range`.

- [ ] **Step 3: Implement `/predict-range`**

Append to `web_app/api/main.py`:

```python
@app.post("/predict-range", response_model=schemas.PredictRangeResponse)
def predict_range(req: schemas.PredictRangeRequest):
    timestamps = data_access.get_timestamps_in_range(
        app.state.time_index, req.start, req.end
    ).index.tolist()
    df = inference.predict_eve_timeseries(
        app.state.model,
        app.state.aia_root,
        app.state.time_index,
        app.state.aia_norms,
        app.state.wavelengths,
        app.state.eve_ions,
        timestamps,
    )

    if df.empty:
        return schemas.PredictRangeResponse(count=0, predictions=[])

    df = df.reset_index()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    records = df.to_dict(orient="records")
    return schemas.PredictRangeResponse(count=len(records), predictions=records)
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
cd web_app
pytest api/tests/test_api.py -v
```

Expected: all tests in `test_api.py` pass.

- [ ] **Step 5: Commit**

```bash
git add web_app/api/main.py web_app/api/tests/test_api.py
git commit -m "Add /predict-range endpoint for time-window inference

Filters indexed timestamps to the inclusive [start, end] window and
runs inference on each. Returns count + records; an empty window is
HTTP 200 with count=0 (not an error). Inverted ranges are rejected
by the schema validator with 422."
```

---

## Task 10: Add `api/requirements.txt` and `api/Dockerfile`

**Files:**
- Create: `web_app/api/requirements.txt`
- Create: `web_app/api/Dockerfile`

- [ ] **Step 1: Create `api/requirements.txt`**

```
--extra-index-url https://download.pytorch.org/whl/cpu
fastapi~=0.118.0
uvicorn[standard]~=0.30.0
pydantic~=2.9.0
numpy>=2.0
pandas>=2.0
pyarrow>=15.0
pytorch-lightning>=2.0
s3fs>=2024.0
torch>=2.0
torchvision>=0.15
zarr>=3.0
pytest~=8.0
```

- [ ] **Step 2: Create `api/Dockerfile`**

Patterned on the existing UI Dockerfile (now at `ui/Dockerfile`) — same builder/runtime split, but for uvicorn:

```dockerfile
## -----------  builder stage  -----------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

## -----------  runtime stage  -----------
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY core/ /app/core/
COPY api/ /app/api/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=30 --start-period=60s \
    CMD curl --fail http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The build context is `web_app/` (set by `docker-compose.yml` in Task 13), which lets the `COPY core/ /app/core/` and `COPY api/ /app/api/` lines work.

- [ ] **Step 3: Verify the image builds**

If you have docker available locally:

```bash
cd web_app
docker build -f api/Dockerfile -t virtual-eve-api:dev .
```

Expected: build succeeds. (The image is large because of torch — ~2GB. This is expected.)

If docker is not available locally, skip this step and rely on the docker-compose smoke test in Task 14.

- [ ] **Step 4: Commit**

```bash
git add web_app/api/requirements.txt web_app/api/Dockerfile
git commit -m "Add api/Dockerfile and api/requirements.txt

Two-stage build mirroring the existing ui Dockerfile. New deps for the
service layer use compatible-release pins (~=) so patches flow but
minor-version surprises are blocked. Shared deps stay on >= matching
the existing style. HEALTHCHECK runs curl against /health every 10s
with a 60s start grace period for model load."
```

---

## Task 11: Add `ui/api_client.py` (TDD)

**Files:**
- Create: `web_app/ui/api_client.py`
- Create: `web_app/api/tests/test_api_client.py`

- [ ] **Step 1: Write failing tests for the client**

Create `web_app/api/tests/test_api_client.py`:

```python
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
```

- [ ] **Step 2: Run the tests — confirm they fail**

```bash
cd web_app
pytest api/tests/test_api_client.py -v
```

Expected: collection error (no `ui.api_client`).

- [ ] **Step 3: Implement the client**

Create `web_app/ui/api_client.py`:

```python
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
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
cd web_app
pytest api/tests/test_api_client.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add web_app/ui/api_client.py web_app/api/tests/test_api_client.py
git commit -m "Add ui/api_client.py wrapping the FastAPI service over httpx

predict_range returns a DataFrame indexed by timestamp so the existing
Streamlit plot code works unchanged. The other methods return raw
dicts. health() does not raise on 503 since the UI uses the body to
distinguish 'starting' from 'ready'."
```

---

## Task 12: Refactor `ui/main.py` to consume the API

**Files:**
- Modify: `web_app/ui/main.py`

This task changes existing logic, not adds it. The UI's Streamlit semantics make automated testing impractical; verify by running the app manually after the API is up.

- [ ] **Step 1: Replace the imports**

In `web_app/ui/main.py`, replace:

```python
from core.data_access import (
    AIA_WAVELENGTHS,
    build_time_index,
    get_aia_image,
    get_aia_root,
    get_available_dates,
    get_timestamps_in_range,
)
from core.inference import load_model, predict_eve_timeseries
```

with:

```python
import os

from core.data_access import AIA_WAVELENGTHS, build_time_index, get_aia_image, get_aia_root
from ui.api_client import APIClient
```

(`get_aia_image` + `get_aia_root` + `build_time_index` stay because the UI still reads images directly from Zarr. `load_model`, `predict_eve_timeseries`, `get_timestamps_in_range`, `get_available_dates` go away.)

- [ ] **Step 2: Replace the cached resources**

Replace the existing block (around lines 39–60):

```python
@st.cache_resource(show_spinner="Connecting to S3 data store...")
def init_aia():
    return get_aia_root()


@st.cache_resource(show_spinner="Building time index (first run may take a few minutes)...")
def init_time_index(_aia_root):
    return build_time_index(_aia_root)


@st.cache_resource(show_spinner="Loading Virtual EVE model...")
def init_model():
    return load_model()


aia_root = init_aia()
time_index = init_time_index(aia_root)
model, aia_norms, wavelengths, eve_ions = init_model()
date_min, date_max = get_available_dates(time_index)
```

with:

```python
API_URL = os.environ.get("API_URL", "http://localhost:8000")


@st.cache_resource(show_spinner="Connecting to API...")
def init_client():
    return APIClient(API_URL)


@st.cache_resource(show_spinner="Reading API metadata...")
def init_info(_client: APIClient):
    return _client.info()


@st.cache_resource(show_spinner="Connecting to AIA Zarr store...")
def init_aia_root():
    return get_aia_root()


@st.cache_resource(show_spinner="Loading time index for AIA images...")
def init_time_index_for_images(_aia_root):
    return build_time_index(_aia_root)


client = init_client()
info = init_info(client)
aia_root = init_aia_root()
time_index = init_time_index_for_images(aia_root)

eve_ions = info["eve_ions"]
date_min = pd.to_datetime(info["available_dates"]["min"])
date_max = pd.to_datetime(info["available_dates"]["max"])
```

The UI keeps its own `aia_root` + `time_index` for AIA image lookups in the imagery panel — these read from the parquet cache the API has already built (so on a freshly started ui container, the API must be up first; the docker-compose `depends_on: condition: service_healthy` enforces this in Task 13).

- [ ] **Step 3: Replace the inference call in the Analyze branch**

Find this block (around lines 226–230):

```python
with st.spinner(f"Running inference on {len(timestamps)} images..."):
    eve_data = predict_eve_timeseries(
        model, aia_root, time_index, aia_norms, wavelengths, eve_ions, timestamps
    )
```

Replace with a single API call. Also remove the now-unnecessary `ts_df = get_timestamps_in_range(...)` block above it (the API does the filtering server-side):

```python
with st.spinner("Running inference via API..."):
    eve_data = client.predict_range(start_dt, end_dt)
```

- [ ] **Step 4: Drop the local timestamp-filtering block**

A few lines above the spinner there's:

```python
ts_df = get_timestamps_in_range(time_index, start_dt, end_dt)

if ts_df.empty:
    st.warning("No data available in the selected range.")
    st.stop()

timestamps = ts_df.index.tolist()
st.sidebar.info(f"Found {len(timestamps)} timestamps in range.")
```

Replace with:

```python
# Empty-range check happens after the API call by inspecting eve_data.
```

And after the API call:

```python
if eve_data.empty:
    st.warning("No data available in the selected range.")
    st.stop()

st.sidebar.info(f"Found {len(eve_data)} timestamps in range.")
```

(The "Found N timestamps" message moves below the API call since we don't know the count until then.)

- [ ] **Step 5: Adjust `first_ts` reference**

The first-timestamp line is:

```python
first_ts = timestamps[0]
```

Change to:

```python
first_ts = eve_data.index[0]
```

- [ ] **Step 6: Manual smoke run**

You need both services running for this. The fastest path is the full docker-compose in Task 13. For now, you can verify imports and Streamlit boot in isolation:

```bash
cd web_app
python -c "from ui.api_client import APIClient; from ui import main"
```

Expected: no import errors. (Streamlit will attempt to render at import time only inside `streamlit run`, not via `python -c`.)

- [ ] **Step 7: Commit**

```bash
git add web_app/ui/main.py
git commit -m "Refactor Streamlit UI to consume the FastAPI service for predictions

The UI no longer loads the model or imports core.inference. Date
bounds, model name, and ion list come from /info; range predictions
come from /predict-range via the new APIClient. The AIA image panel
still reads Zarr directly (out of scope for the API per the design
spec). API_URL is configurable via env var; defaults to
http://localhost:8000."
```

---

## Task 13: Update `ui/requirements.txt` and `ui/Dockerfile`

**Files:**
- Create: `web_app/ui/requirements.txt`
- Modify: `web_app/ui/Dockerfile`
- Delete: `web_app/requirements.txt`

- [ ] **Step 1: Create `ui/requirements.txt`**

```
httpx~=0.27.0
numpy>=2.0
pandas>=2.0
plotly>=5.0
pyarrow>=15.0
streamlit>=1.30
zarr>=3.0
s3fs>=2024.0
```

`pyarrow` is explicit so the UI can read the parquet cache (it's a transitive streamlit dep but we list it for clarity). `torch`, `torchvision`, `pytorch-lightning` are dropped — the UI never imports `core.inference` or `core.model` after the refactor.

- [ ] **Step 2: Modify `ui/Dockerfile`**

Update the file paths to match the new layout. Replace the existing content with:

```dockerfile
## -----------  builder stage  -----------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY ui/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

## -----------  runtime stage  -----------
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY core/ /app/core/
COPY ui/ /app/ui/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "ui/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

The build context is `web_app/` (set by `docker-compose.yml` in Task 14), which lets the `COPY core/` and `COPY ui/` lines work.

- [ ] **Step 3: Delete the old top-level `requirements.txt`**

```bash
git rm web_app/requirements.txt
```

(It's superseded by the per-service requirements.)

- [ ] **Step 4: Verify the image builds (optional, requires docker)**

```bash
cd web_app
docker build -f ui/Dockerfile -t virtual-eve-ui:dev .
```

Expected: build succeeds. The image should be substantially smaller than before (no torch).

- [ ] **Step 5: Commit**

```bash
git add web_app/ui/requirements.txt web_app/ui/Dockerfile
git rm web_app/requirements.txt
git commit -m "Slim ui/ requirements: drop torch, add httpx; update Dockerfile

The UI no longer loads the model so torch/torchvision/pytorch-lightning
come out, shrinking the image. httpx is added for the api_client.
pyarrow is explicit so parquet-cache reads are obvious. Dockerfile
paths updated for the new web_app/core + web_app/ui layout; build
context is web_app/ root."
```

---

## Task 14: Rewrite `web_app/docker-compose.yml`

**Files:**
- Modify: `web_app/docker-compose.yml`

- [ ] **Step 1: Replace the file with the two-service compose**

Overwrite `web_app/docker-compose.yml`:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATA_BACKEND=${DATA_BACKEND:-local}
      - LOCAL_DATA_ROOT=/data
    volumes:
      - ${HOST_DATA_PATH:-./DATA_TEST}:/data:ro
      - ./cache:/app/cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 60s

  ui:
    build:
      context: .
      dockerfile: ui/Dockerfile
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
      - DATA_BACKEND=${DATA_BACKEND:-local}
      - LOCAL_DATA_ROOT=/data
    volumes:
      - ${HOST_DATA_PATH:-./DATA_TEST}:/data:ro
      - ./cache:/app/cache:ro
    depends_on:
      api:
        condition: service_healthy
```

`HOST_DATA_PATH` defaults to the empty `DATA_TEST` directory so the file is valid on this dev machine. To deploy on a server with real data, set `HOST_DATA_PATH` (and `DATA_BACKEND` if applicable) via `.env` or environment.

- [ ] **Step 2: Validate the compose file syntax (optional)**

If you have docker available:

```bash
cd web_app
docker compose config
```

Expected: prints the resolved compose configuration with no errors.

- [ ] **Step 3: Commit**

```bash
git add web_app/docker-compose.yml
git commit -m "Rewrite docker-compose.yml for two-service deployment

api + ui services with parameterised HOST_DATA_PATH so the same
compose file works locally and on the deployment server. The ui
service waits for the api healthcheck to pass before starting, so
Streamlit's first request never races the API's model load."
```

---

## Task 15: End-to-end smoke test

**Files:** none (verification only)

- [ ] **Step 1: Start both services**

On a machine with data access (the deployment server, or a dev machine with `LOCAL_DATA_ROOT` set or `DATA_BACKEND=s3`):

```bash
cd web_app
HOST_DATA_PATH=/path/to/sdomlv2a docker compose up --build
```

Expected: both images build. The `api` container starts and logs the model + index load. After the load completes (minutes on cold S3 cache, seconds with parquet cache), the healthcheck flips to healthy and the `ui` container starts.

- [ ] **Step 2: Hit `/health`, `/info`, `/predict`, `/predict-range`**

In another terminal:

```bash
curl -s http://localhost:8000/health | jq
# {"status": "ready"}

curl -s http://localhost:8000/info | jq '.eve_ions | length, .aia_wavelengths | length'
# 38
# 9

curl -s http://localhost:8000/info | jq '.available_dates'
# {"min": "...", "max": "..."}

# Use one of the dates from /info as a known-valid timestamp
TS=$(curl -s http://localhost:8000/info | jq -r '.available_dates.min')
curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' \
    -d "{\"timestamp\": \"$TS\"}" | jq '.predictions | length'
# 38

curl -s -X POST http://localhost:8000/predict-range -H 'content-type: application/json' \
    -d '{"start": "2017-09-06T00:00:00", "end": "2017-09-06T02:00:00"}' | jq '.count'
# Some positive integer
```

If `/predict-range` returns count: 0, pick a date range from `/info`'s `available_dates` window.

- [ ] **Step 3: Open the Streamlit UI**

Navigate to `http://localhost:8501` in a browser. Pick a small date range within `available_dates` and click "Analyze". The page should:

1. Show the AIA image panel for the first timestamp (read directly from Zarr).
2. Show the EVE irradiance time-series plot (predictions from the API).
3. Show the histograms.
4. Offer a CSV download.

If anything is broken or visibly different from the pre-refactor experience, note the issue and fix before declaring success.

- [ ] **Step 4: Run the test suite on the same machine**

```bash
cd web_app
pytest -v
```

Expected: all tests pass. If lifespan-exercising tests fail because the data backend is misconfigured, the conftest should have warned. Fix and re-run.

- [ ] **Step 5: No commit needed**

This task verifies the prior commits work end-to-end.

---

## Done

When all tasks are complete:

- The FastAPI service is running and serving `/info`, `/predict`, `/predict-range`, `/health`.
- The Streamlit UI calls the API for predictions and metadata; it reads AIA images directly from Zarr.
- `web_app/` is split into `core/`, `api/`, `ui/`.
- The time-index cache is parquet.
- All tests pass on a machine with data access.
