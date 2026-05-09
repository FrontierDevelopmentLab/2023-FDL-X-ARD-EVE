"""FastAPI service exposing Virtual EVE predictions."""

from contextlib import asynccontextmanager

import torch
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
