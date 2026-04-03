"""On-the-fly inference using the pre-trained Virtual EVE model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure model classes are importable when torch.load unpickles the checkpoint.
# The checkpoint was pickled with the full module path from the training repo.
import model as _model_module

sys.modules["src"] = type(sys)("src")
sys.modules["src.irradiance"] = type(sys)("src.irradiance")
sys.modules["src.irradiance.models"] = type(sys)("src.irradiance.models")
sys.modules["src.irradiance.models.model"] = _model_module

CHECKPOINT_PATH = Path(__file__).parent.parent / "inference-cloud-function" / "checkpoints" / "AIA_MEGS_20_30_epochs_36min.ckpt"


def load_model():
    """Load the pre-trained model and normalizations from checkpoint."""
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = state["model"]
    model.eval()

    aia_norms = state["normalizations"]["AIA"]
    wavelengths = sorted(state["sci_parameters"]["aia_wavelengths"])
    eve_ions = state["sci_parameters"]["eve_ions"]

    return model, aia_norms, wavelengths, eve_ions


def normalize_aia_image(aia_image: dict, aia_norms: dict, wavelengths: list) -> torch.Tensor:
    """Normalize raw AIA images and stack into a model-ready tensor."""
    channels = []
    for wl in wavelengths:
        img = aia_image[wl].astype(np.float32)
        img = (img - aia_norms[wl]["mean"]) / aia_norms[wl]["std"]
        channels.append(img)

    # Shape: [1, n_wavelengths, 512, 512]
    stacked = np.stack(channels, axis=0)[np.newaxis, ...].astype(np.float32)
    return torch.from_numpy(stacked)


def predict_eve(model, aia_image: dict, aia_norms: dict, wavelengths: list, eve_ions: list) -> pd.DataFrame:
    """Run inference on a single AIA image and return EVE irradiance predictions."""
    x = normalize_aia_image(aia_image, aia_norms, wavelengths)

    with torch.no_grad():
        pred = model.forward_unnormalize(x).numpy()[0]

    result = {ion: float(pred[i]) for i, ion in enumerate(eve_ions)}
    return pd.DataFrame([result])


def predict_eve_timeseries(
    model, aia_root, time_index, aia_norms, wavelengths, eve_ions, timestamps
) -> pd.DataFrame:
    """Run inference for multiple timestamps and return a DataFrame."""
    from data_access import get_aia_image

    rows = []
    for ts in timestamps:
        aia_image = get_aia_image(aia_root, time_index, ts)
        if aia_image is None:
            continue
        x = normalize_aia_image(aia_image, aia_norms, wavelengths)
        with torch.no_grad():
            pred = model.forward_unnormalize(x).numpy()[0]
        row = {"timestamp": ts}
        row.update({ion: float(pred[i]) for i, ion in enumerate(eve_ions)})
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df
