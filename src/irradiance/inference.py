import argparse
import os
import numpy as np
import torch
import pandas as pd

from fastapi import FastAPI

app = FastAPI()

import zarr
def test_image(wavelengths):
    aia_data = zarr.group(zarr.DirectoryStore("/mnt/sdomlv2_full/sdomlv2.zarr"))
    aia_image = {}
    for wavelength in wavelengths:
        aia_image[wavelength] = aia_data[2011][wavelength][10,:,:]
    return aia_image

class IrradianceInferenceModel:
    def __init__(self):
        # os.getenv("checkpoint_location")
        checkpoint_path = "../../runs_data/checkpoints/AIA_FULL_MEGS_FULL_30_50_epochs_30min/best_model_normalizations.ckpt" 
        state = torch.load(checkpoint_path)
        self.model = state["model"]
        self.model.eval()
        self.aia_wavelengths = state["sci_parameters"]["aia_wavelengths"]
        self.aia_wavelengths.sort()
        self.eve_ions = state["sci_parameters"]["eve_ions"]
        self.eve_ions.sort()
        self.aia_normalizations = state["normalizations"]["AIA"]


    def predict(self, aia_image, forward_passes=0):
        aia_image = self.prepare_aia(aia_image)
        with torch.no_grad():
            pred_irradiance = self.model.forward_unnormalize(aia_image).numpy()
        
        pred_irradiance = pd.Series({ion: pred_irradiance[0][i] for i, ion in enumerate(self.eve_ions)})
        return pred_irradiance


    def prepare_aia(self, aia_image):
        for wavelength in self.aia_wavelengths:
            aia_image[wavelength] -= self.aia_normalizations[wavelength]["mean"]
            aia_image[wavelength] /= self.aia_normalizations[wavelength]["std"]

        aia_image = np.array([np.stack([aia_image[wavelength] for wavelength in self.aia_wavelengths], axis=0)])
        aia_image = torch.from_numpy(aia_image)
        return aia_image


    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


@app.get("/")
async def root():
    return {"message": "Hello World"}


