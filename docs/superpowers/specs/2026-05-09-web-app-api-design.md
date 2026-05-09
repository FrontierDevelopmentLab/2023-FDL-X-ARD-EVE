# Virtual EVE — Web App API Design

**Date:** 2026-05-09
**Status:** Approved (brainstorm)

## Summary

Add a FastAPI service to `web_app/` so external clients can query Virtual EVE
predictions programmatically and so the Streamlit UI no longer owns the model.
Restructure `web_app/` into `core/` (shared inference + data access),
`api/` (FastAPI service), and `ui/` (Streamlit). Both services run as separate
containers in `docker-compose.yml`; the UI calls the API for all model-derived
data and continues to read raw AIA images directly from the Zarr store.

## Goals

- Expose Virtual EVE predictions via a stable, machine-readable HTTP API.
- Decouple model ownership from the UI: the model is loaded exactly once, in
  the API service.
- Keep the Streamlit demo working with no user-facing regressions.
- Hold scope tight — no auth, no batch jobs, no AIA image streaming.

## Non-goals

- Authentication, API keys, or rate limiting.
- AIA image retrieval over HTTP. Clients can read the public Zarr store
  directly; the UI continues to read it via `core.data_access`.
- BigQuery write-back. The legacy `inference-cloud-function/` is out of scope.
- Async batch inference, queueing, or job APIs.
- Migrating `web_app/` from `requirements.txt` to `pyproject.toml` + `uv.lock`.

## Architecture

Two services in `web_app/docker-compose.yml`:

- **`api`** — FastAPI + uvicorn on port 8000. Owns the model and the time
  index. Loads them once via FastAPI lifespan.
- **`ui`** — Streamlit on port 8501. Calls the API for predictions and
  metadata. Imports `core.data_access` directly for AIA image retrieval.

```
[ user/browser ] ──http──> [ ui : streamlit, 8501 ] ──http──> [ api : fastapi, 8000 ]
                                       │                                │
                                       └──── direct zarr read ──────────┴──> [ AIA Zarr (local fs or S3) ]
```

Each Dockerfile uses `web_app/` as its build context so it can `COPY core/`.
Each container also `COPY`s only its own subdirectory (`api/` or `ui/`).

## File layout

```
web_app/
├── core/
│   ├── __init__.py
│   ├── inference.py            # moved from web_app/inference.py
│   ├── data_access.py          # moved from web_app/data_access.py
│   ├── model.py                # moved from web_app/model.py
│   └── checkpoints/            # moved from web_app/checkpoints/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + lifespan
│   ├── schemas.py              # Pydantic request/response models
│   ├── Dockerfile
│   ├── requirements.txt
│   └── tests/
│       └── test_api.py
├── ui/
│   ├── main.py                 # moved + refactored Streamlit app
│   ├── api_client.py           # httpx client wrapping the API
│   ├── assets/                 # moved from web_app/assets/
│   ├── Dockerfile              # moved + adjusted from web_app/Dockerfile
│   └── requirements.txt
├── cache/                      # mounted into the api container only
└── docker-compose.yml          # api + ui services
```

`core/__init__.py` is empty so importing `core.data_access` does not transitively
load `core.model` (which pulls in torch). This lets the UI drop torch entirely.

## API surface

All datetimes are ISO-8601. Naive datetimes are treated as UTC. Tz-aware
datetimes are converted to UTC and stripped before lookup against the time
index.

### `GET /info`

Returns model and dataset metadata.

Response 200:
```json
{
  "model_name": "AIA_MEGS_20_30_epochs_36min",
  "aia_wavelengths": ["131A", "1600A", "1700A", "171A", "193A", "211A", "304A", "335A", "94A"],
  "eve_ions": ["C III", "Fe IX", "..."],
  "available_dates": {
    "min": "2010-05-13T00:00:00",
    "max": "2024-01-01T00:00:00"
  }
}
```

### `POST /predict`

Single-timestamp prediction.

Request:
```json
{ "timestamp": "2017-09-06T12:00:00" }
```

Response 200:
```json
{
  "timestamp": "2017-09-06T12:00:00",
  "predictions": { "C III": 0.123, "Fe IX": 0.456 }
}
```

`predictions` always contains all 38 EVE ion keys.

The requested timestamp is rounded to the nearest 36-minute slot (the
cadence of the time index) and snapped to the closest indexed entry. The
`timestamp` field in the response reflects the snapped value, which may
differ from the request by up to ~18 minutes.

Errors:
- `422` — malformed request body, unparseable timestamp, or rounded
  timestamp outside `[available_dates.min, available_dates.max]`.

### `POST /predict-range`

Range prediction. Inclusive on both ends.

Request:
```json
{ "start": "2017-09-06T00:00:00", "end": "2017-09-06T23:59:00" }
```

Response 200:
```json
{
  "count": 40,
  "predictions": [
    { "timestamp": "2017-09-06T00:00:00", "C III": 0.123, "Fe IX": 0.456 }
  ]
}
```

The endpoint filters indexed timestamps to those falling within
`[start, end]` (inclusive) and runs inference on each. An empty range
returns `{ "count": 0, "predictions": [] }` with HTTP 200 — empty is not
an error.

Errors:
- `422` — malformed dates or `end < start`.

### `GET /health`

Readiness probe. Returns 200 only after model and time index are loaded.

Ready (200):
```json
{ "status": "ready" }
```

Starting (503):
```json
{ "status": "starting" }
```

## Pydantic schemas (`api/schemas.py`)

- `InfoResponse` — `model_name: str`, `aia_wavelengths: list[str]`,
  `eve_ions: list[str]`, `available_dates: AvailableDates`.
- `AvailableDates` — `min: datetime`, `max: datetime`.
- `PredictRequest` — `timestamp: datetime`.
- `PredictResponse` — `timestamp: datetime`, `predictions: dict[str, float]`.
- `PredictRangeRequest` — `start: datetime`, `end: datetime`. A model
  validator rejects `end < start` with 422.
- `PredictRangeResponse` — `count: int`, `predictions: list[dict[str, Any]]`.
- `HealthResponse` — `status: Literal["ready", "starting"]`.

## Startup (FastAPI lifespan)

`api/main.py` defines an `async def lifespan(app)` context manager that:

1. Sets a module-level `_ready = False`.
2. Calls `core.data_access.get_aia_root()` and
   `core.data_access.build_time_index(...)`.
3. Calls `core.inference.load_model()`.
4. Stores handles on `app.state` (`model`, `aia_norms`, `wavelengths`,
   `eve_ions`, `aia_root`, `time_index`).
5. Sets `_ready = True`.

Endpoints read state from `app.state`. `/health` reads `_ready`.

## Streamlit refactor (`ui/main.py`)

After the refactor:

- The three existing cached resources for AIA root, time index, and model are
  replaced by a single cached call to `api_client.info()`. The UI does not
  load the model.
- The sidebar date bounds come from `info.available_dates`.
- The "Analyze" button calls `api_client.predict_range(start_dt, end_dt)`.
  The client returns a `pandas.DataFrame` indexed by timestamp with one
  column per ion — the same shape the existing plotly code expects.
- The AIA images panel still calls `core.data_access.get_aia_image(...)`. The
  UI keeps a small cached `aia_root` plus a minimal time-index lookup
  (loaded from the same CSV cache as the API) for image retrieval only.
- `API_URL` comes from env. Default `http://api:8000` (compose);
  `http://localhost:8000` for standalone `streamlit run`.

## `api_client.py`

```python
class APIClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None: ...
    def info(self) -> dict: ...
    def predict(self, timestamp: datetime) -> dict: ...
    def predict_range(self, start: datetime, end: datetime) -> pd.DataFrame: ...
    def health(self) -> dict: ...
```

`predict_range` does the JSON → DataFrame conversion so the call site in
Streamlit can swap with no other shape changes.

## Errors and CORS

- FastAPI default error envelopes; no custom error model.
- `CORSMiddleware` allows all origins, methods, and headers (demo-grade).
- Unhandled exceptions surface as HTTP 500 with the default body and a logged
  stack trace.

## Health-check wiring

- `api/Dockerfile` adds
  `HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1`.
- `docker-compose.yml`: `ui` declares
  `depends_on: { api: { condition: service_healthy } }` so Streamlit waits
  for the API to report ready before starting.

## Testing

`api/tests/test_api.py` uses `fastapi.testclient.TestClient`. The TestClient
triggers the lifespan, so the model and time index load on construction.

A small test fixture is needed at `web_app/DATA_TEST/AIA.zarr` containing a
handful of timestamps across a couple of years. The directory exists today
but is empty; populating it (either by extracting a slice from the real
dataset or by generating a synthetic Zarr with the right schema) is part of
the implementation. The fixture is checked in to the repo alongside the
checkpoint so tests run offline.

`conftest.py` sets `DATA_BACKEND=local` and `LOCAL_DATA_ROOT` to point at
`DATA_TEST/` before the lifespan runs.

Coverage:
- `test_info_returns_metadata` — 38 ions, 9 wavelengths, parseable dates.
- `test_predict_returns_38_ions_for_known_timestamp`.
- `test_predict_snaps_to_nearest_indexed_timestamp` — verify the response
  `timestamp` field reflects the snapped value.
- `test_predict_out_of_range_returns_422`.
- `test_predict_invalid_iso_returns_422`.
- `test_predict_range_returns_records_for_valid_range`.
- `test_predict_range_empty_returns_zero_count`.
- `test_predict_range_inverted_returns_422`.
- `test_health_ready_after_lifespan`.

Run from `web_app/`: `pytest api/tests/`.

## Dependencies

`api/requirements.txt` (compatible-release pins for new deps; `>=` for shared
deps to match the current style):

```
--extra-index-url https://download.pytorch.org/whl/cpu
fastapi~=0.118.0
uvicorn[standard]~=0.30.0
pydantic~=2.9.0
numpy>=2.0
pandas>=2.0
pytorch-lightning>=2.0
s3fs>=2024.0
torch>=2.0
torchvision>=0.15
zarr>=3.0
pytest~=8.0
```

`ui/requirements.txt`:

```
httpx~=0.27.0
numpy>=2.0
pandas>=2.0
plotly>=5.0
streamlit>=1.30
zarr>=3.0
s3fs>=2024.0
```

The UI drops `torch`, `torchvision`, and `pytorch-lightning` because it never
imports `core.model` or `core.inference`. This shrinks the UI image
substantially.

## docker-compose.yml (sketch)

```yaml
services:
  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATA_BACKEND=local
      - LOCAL_DATA_ROOT=/data
    volumes:
      - ./DATA_TEST:/data:ro
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
      - DATA_BACKEND=local
      - LOCAL_DATA_ROOT=/data
    volumes:
      - ./DATA_TEST:/data:ro
      - ./cache:/app/cache:ro
    depends_on:
      api:
        condition: service_healthy
```

The UI mounts `cache/` read-only because it consults the same time-index CSV
the API built — a deliberate, narrow shared dependency that keeps image
fetching cheap. The API owns writes.

## Risks and open questions

- **First-startup time on a cold cache.** Building the time index over S3
  takes minutes; `/health` stays at 503 for the duration. This matches the
  current Streamlit startup behavior — moving the work to the API does not
  make it worse, and the new readiness check makes the wait visible to the
  UI rather than a confusing error.
- **Cache directory ownership.** The API writes the time index; the UI reads
  it. The compose mount enforces this with read-only on the UI side.
- **First request after startup may be slow.** The time index is loaded but
  Zarr chunks are not pre-fetched. Acceptable for a demo; revisit if it
  becomes a real concern.
- **Test fixture must be created.** `web_app/DATA_TEST/` exists but is
  empty. The implementation must populate it (a small slice of the real
  Zarr or a synthetic equivalent) before the test suite can run.

## Migration steps (high-level)

The detailed implementation plan will come from writing-plans. The shape:

1. Create `core/`, `api/`, `ui/` under `web_app/`. Move existing files into
   `core/` and `ui/`. Update imports.
2. Add `api/main.py`, `api/schemas.py`, `api/Dockerfile`,
   `api/requirements.txt`, `api/tests/test_api.py`.
3. Add `ui/api_client.py`. Refactor `ui/main.py` to use it. Trim `torch`
   from `ui/requirements.txt`.
4. Rewrite `web_app/docker-compose.yml` with `api` + `ui` services and the
   healthcheck wiring.
5. Smoke-test: `docker compose up` → `/info`, `/health`, `/predict-range`,
   then exercise the Streamlit UI end-to-end.
