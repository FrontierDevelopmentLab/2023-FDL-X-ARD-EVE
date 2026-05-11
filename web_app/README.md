# Virtual EVE — Solar Irradiance Prediction Demo

A two-service deployment that predicts solar extreme ultraviolet (EUV) irradiance using a deep learning model trained on NASA Solar Dynamics Observatory (SDO) imagery. A FastAPI service owns the model and exposes predictions over HTTP; a Streamlit UI consumes the API and renders the visualisation.

## Background

NASA's SDO carries the EVE (EUV Variability Experiment) instrument, which measures solar spectral irradiance. In 2014, the primary MEGS-A module failed, eliminating measurements of 14 key EUV spectral lines. This application uses a hybrid deep learning model to reconstruct those lost measurements — and predict all 38 EVE ion spectral lines — from AIA (Atmospheric Imaging Assembly) images alone.

The model was developed during [FDL-X 2023](https://frontierdevelopmentlab.org/) and is described in [Indaco et al. (2024), arXiv:2408.17430](https://arxiv.org/abs/2408.17430).

## How It Works

The model takes 9-channel AIA images (94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å at 512x512 px) from a Zarr data store and runs inference through a hybrid model that combines:

- **Linear model** — predicts irradiance from per-channel mean/std statistics
- **CNN model** — uses an EfficientNet-B5 backbone for spatial feature extraction
- **Hybrid combination** — a learnable parameter blends both predictions

The API exposes single-timestamp and range predictions; the UI uses the range endpoint to plot a time series.

## Project Structure

```
web_app/
├── core/                       # shared inference + data access (used by both services)
│   ├── inference.py            # model load + forward pass
│   ├── data_access.py          # Zarr reads + time-index cache
│   ├── model.py                # PyTorch Lightning model classes
│   └── checkpoints/            # pre-trained checkpoint (~43 MB)
├── api/
│   ├── main.py                 # FastAPI app + lifespan (loads model once)
│   ├── schemas.py              # Pydantic request/response models
│   ├── Dockerfile
│   ├── requirements.txt
│   └── tests/                  # pytest suite (schemas, helpers, endpoints, client)
├── ui/
│   ├── main.py                 # Streamlit app (consumes the API)
│   ├── api_client.py           # httpx wrapper around the API
│   ├── assets/                 # icons + logos
│   ├── Dockerfile
│   └── requirements.txt
├── cache/                      # parquet time-index cache (api writes, ui reads RO)
├── conftest.py                 # pytest rootdir anchor
├── pytest.ini
└── docker-compose.yml          # api + ui services with healthcheck wiring
```

## Quick Start

### Docker (recommended)

The compose file parameterises the host data path; set `HOST_DATA_PATH` to point at your SDOMLv2 AIA Zarr dataset:

```bash
HOST_DATA_PATH=/path/to/your/sdomlv2a docker compose up --build
```

To use S3 instead of a local mount, set `DATA_BACKEND=s3` (the volume mount becomes irrelevant):

```bash
DATA_BACKEND=s3 docker compose up --build
```

Then:
- API: <http://localhost:8000> (try `/health`, `/info`, `/predict`, `/predict-range`)
- UI: <http://localhost:8501>

The UI service waits for the API healthcheck to pass before starting, so the first UI request never races the model load.

### Local (no Docker)

Run the two services manually. From `web_app/`:

```bash
# Terminal 1 — API
DATA_BACKEND=local LOCAL_DATA_ROOT=/path/to/your/sdomlv2a \
  uvicorn api.main:app --port 8000

# Terminal 2 — UI
API_URL=http://localhost:8000 \
DATA_BACKEND=local LOCAL_DATA_ROOT=/path/to/your/sdomlv2a \
  streamlit run ui/main.py
```

The UI also reads AIA images directly from the Zarr store (for the imagery panel only), so it needs `DATA_BACKEND` / `LOCAL_DATA_ROOT` even though predictions go through the API.

### Run the tests

From `web_app/`:

```bash
DATA_BACKEND=local LOCAL_DATA_ROOT=/path/to/your/sdomlv2a pytest
```

Tests that exercise the FastAPI lifespan need real data. Schema, helper, and `api_client` tests run without it.

## API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns 200 + `{"status": "ready"}` once the model + index are loaded; 503 + `{"status": "starting"}` otherwise |
| `/info` | GET | Model name, AIA wavelengths, EVE ions, available date range |
| `/predict` | POST | Predict at a single (snapped) timestamp |
| `/predict-range` | POST | Predict over an inclusive `[start, end]` window |

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `DATA_BACKEND` | `local` | `local` or `s3` |
| `LOCAL_DATA_ROOT` | `/data` (in containers); see `core/data_access.py` for local default | Path to the SDOMLv2 AIA Zarr dataset |
| `API_URL` | `http://api:8000` (compose) / `http://localhost:8000` (standalone) | UI uses this to reach the API |
| `HOST_DATA_PATH` | `./DATA_TEST` | Compose-only: host path mounted to `/data` in both containers |

When `DATA_BACKEND=s3`, the app reads from the `nasa-radiant-data` S3 bucket (path: `helioai-datasets/us-fdlx-ard/sdomlv2a/AIA.zarr`). No credentials are required for public access.

## Data

The model uses the [SDOML v2 dataset](https://sdoml.org). On first run, the API builds a time index mapping timestamps to image indices and caches it as `cache/aia_time_index.parquet` for subsequent runs. The UI reads the same parquet read-only for its imagery panel.

Timestamps are snapped to a **36-minute grid** (`core.data_access.MODEL_CADENCE`). AIA records its 9 wavelengths a few seconds apart, so the raw per-wavelength timestamps need a common bin to be joined into one row per observation; 36 minutes is also the cadence the model was trained at, and incoming request timestamps are rounded to the same grid so a query always resolves to a definite slot.

## Authors

Manuel Indaco, Daniel Gass, William Fawcett, Richard Galvez, Paul Wright, and Andres Munoz-Jaramillo.
