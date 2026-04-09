# Virtual EVE — Solar EUV Irradiance Prediction

Virtual EVE predicts solar extreme ultraviolet (EUV) irradiance using deep learning applied to imagery from NASA's Solar Dynamics Observatory (SDO). Developed during FDL-X 2023, the model reconstructs measurements originally provided by the SDO/EVE MEGS-A instrument (which failed in 2014) by predicting all 38 EVE ion spectral lines from 9-channel AIA solar images.

## Architecture

The model uses a hybrid approach that blends:

- A **linear model** for statistical baseline predictions
- A **CNN** (EfficientNet-B5 backbone) for learning spatial features from 512x512 px AIA images
- A **learnable blending parameter** that combines both outputs

Training and experiment tracking use PyTorch Lightning and Weights & Biases.

## Project Structure

| Directory | Description |
|---|---|
| `src/irradiance/` | Core library — data loading, model definitions, training |
| `web_app/` | Streamlit application for interactive irradiance visualization |
| `inference-cloud-function/` | Google Cloud Function for serverless inference |
| `notebooks/` | Analysis and experimentation notebooks |

## Data

The model is trained on the [SDOML v2](https://sdoml.org) dataset (Zarr format), with support for both local filesystem and AWS S3 backends. HMI magnetogram data is also integrated into the pipeline.

## Quickstart

```bash
pip install -e .
```

To run the web app:

```bash
cd web_app
streamlit run main.py
```

## Team

FDL-X 2023 ARD EUV Team
