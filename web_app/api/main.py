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
