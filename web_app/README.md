# Virtual EVE — Solar Irradiance Prediction Demo

A Streamlit web application that predicts solar extreme ultraviolet (EUV) irradiance using a deep learning model trained on NASA Solar Dynamics Observatory (SDO) imagery.

## Background

NASA's SDO carries the EVE (EUV Variability Experiment) instrument, which measures solar spectral irradiance. In 2014, the primary MEGS-A module failed, eliminating measurements of 14 key EUV spectral lines. This application uses a hybrid deep learning model to reconstruct those lost measurements — and predict all 38 EVE ion spectral lines — from AIA (Atmospheric Imaging Assembly) images alone.

The model was developed during [FDL-X 2023](https://frontierdevelopmentlab.org/) and is described in [Indaco et al. (2024), arXiv:2408.17430](https://arxiv.org/abs/2408.17430).

## How It Works

The app loads 9-channel AIA images (94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å at 512x512 px) from a Zarr data store and runs inference through a hybrid model that combines:

- **Linear model** — predicts irradiance from per-channel mean/std statistics
- **CNN model** — uses an EfficientNet-B5 backbone for spatial feature extraction
- **Hybrid combination** — a learnable parameter blends both predictions

Users select a date range, the app runs inference on each available timestamp, and displays the predicted EVE irradiance time series alongside the AIA images.

## Project Structure

```
web_app/
├── main.py              # Streamlit UI entry point (pages: Virtual EVE, About)
├── model.py             # PyTorch Lightning model definitions (Linear, CNN, Hybrid)
├── inference.py          # Model loading and inference pipeline
├── data_access.py        # Data access layer (local filesystem or S3)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container image (python:3.12-slim)
├── docker-compose.yml    # Docker service orchestration
├── .env                  # Environment variables (local paths, data backend config)
├── checkpoints/          # Pre-trained model checkpoint (~43 MB)
├── cache/                # Cached time index (built on first run)
└── assets/               # Logo and icon images
```

## Quick Start

### Docker (recommended)

1. Copy `.env.example` or create a `.env` file in the `web_app/` directory and set `HOST_DATA_PATH` to your local copy of the SDOMLv2 AIA Zarr dataset:

   ```env
   HOST_DATA_PATH=/path/to/your/sdomlv2a
   ```

   See the `.env` file for all available settings (data backend, S3 bucket, etc.).

2. Build and run:

   ```bash
   docker compose up --build
   ```

   To pass the env file explicitly (e.g. from a different location):

   ```bash
   docker compose --env-file /path/to/.env up --build
   ```

3. Open http://localhost:8501.

### Local

```bash
pip install -r requirements.txt
export DATA_BACKEND=local
export LOCAL_DATA_ROOT=/path/to/your/sdomlv2a
streamlit run main.py
```

## Configuration

All configuration is managed via the `.env` file (git-ignored). The `.env` file is automatically loaded by `docker compose`.

| Environment Variable | Default | Description |
|---|---|---|
| `DATA_BACKEND` | `local` | Data source: `local` or `s3` |
| `HOST_DATA_PATH` | — | Host path to the SDOMLv2 AIA Zarr dataset (mounted into the container) |
| `LOCAL_DATA_ROOT` | `/data` | Container-internal mount point for the data |
| `S3_BUCKET` | `nasa-radiant-data` | S3 bucket name (used when `DATA_BACKEND=s3`) |
| `AIA_ZARR_PREFIX` | `helioai-datasets/us-fdlx-ard/sdomlv2a/AIA.zarr` | S3 prefix for AIA Zarr store |
| `HMI_ZARR_PREFIX` | `helioai-datasets/us-fdlx-ard/sdomlv2a/HMI.zarr` | S3 prefix for HMI Zarr store |

When `DATA_BACKEND=s3`, the app reads from S3. No credentials are required for public access.

## Data

The model uses the [SDOML v2 dataset](https://sdoml.org). On first run, the app builds a time index mapping timestamps to image indices and caches it to `cache/aia_time_index.csv` for subsequent runs.

## Authors

Manuel Indaco, Daniel Gass, William Fawcett, Richard Galvez, Paul Wright, and Andres Munoz-Jaramillo.
