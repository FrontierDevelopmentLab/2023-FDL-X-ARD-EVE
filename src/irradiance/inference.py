import argparse
import os

import numpy as np
import torch

from src.irradiance.models.model import HybridIrradianceModel

from fastapi import FastAPI
app = FastAPI()


class IrradianceInferenceModel:

    def __init__(self):
        checkpoint_path = os.getenv("checkpoint_location")
        state = torch.load(checkpoint_path)
        self.model = state["model"]
        self.model.eval()
        self.aia_wavelengths = state["aia_wavelengths"]
        torch.no_grad()

    def predict(self, aia_image, forward_passes=0):
        aia_image = [aia_image]
        pred_irradiance = self.model.forward_unnormalize(aia_image).numpy()
        return pred_irradiance

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


@app.get("/")
async def root():

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint_location', type=str, help='path to the model checkpoint.')
    args = p.parse_args()

\
    aia_wavelengths = state["aia_wavelengths"]

    return {"message": "Hello World"}


