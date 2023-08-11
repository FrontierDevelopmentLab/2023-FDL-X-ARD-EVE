import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import zarr
from datetime import datetime
import pandas as pd

import plotly.express as px

# from fastapi import FastAPI
# app = FastAPI()


class IrradianceInferenceModel:
    def __init__(self):
        # os.getenv("checkpoint_location")
        checkpoint_path = "/home/richardagalvez/2023-FDL-X-ARD-EVE/runs_data/checkpoints/AIA_FULL_MEGS_FULL_30min/best_model.ckpt" 
        state = torch.load(checkpoint_path)
        self.model = state["model"]
        self.model.eval()
        self.aia_wavelengths = state["sci_parameters"]["aia_wavelengths"]
        self.aia_wavelengths.sort()
        self.eve_ions = state["sci_parameters"]["eve_ions"]
        self.eve_ions.sort()
        self.aia_normalizations = state["normalizations"]["AIA"]

        self.aia_data = zarr.group(zarr.DirectoryStore("/mnt/sdomlv2_full/sdomlv2.zarr"))


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


    def load_aia_image(self, time, idx):
        aia_image = {}
        for wavelength in self.aia_wavelengths:
            # time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            year = int(time.year)
            aia_image[wavelength] = self.aia_data[year][wavelength][idx,:,:]
        return aia_image

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

inference_model = IrradianceInferenceModel()


def run_one_off_inferences():

    virtual_eve = IrradianceInferenceModel()
    for year in virtual_eve.aia_data.keys():
        print(f"Processing year: {year}")
        time_keys = pd.to_datetime(virtual_eve.aia_data[year][virtual_eve.aia_wavelengths[0]].attrs['T_OBS']).sort_values()
        pred_irradiance = pd.DataFrame(index=time_keys, columns=virtual_eve.eve_ions)
        for idx, time in enumerate(tqdm(time_keys, total=len(time_keys), desc="Processing time")):
            try:
                aia_image = virtual_eve.load_aia_image(time, idx)
                pred_irradiance.loc[time] = virtual_eve.predict(aia_image)
            except:
                print(f"Failed to process time: {time}")
                continue
        pred_irradiance.to_parquet(f"/home/jupyter/output_inferences/run1/pred_irradiance_{year}.parquet")
        print(f"Saved year: {year}, shape: {pred_irradiance.shape}")


def get_eve_data(year=2011):
    eve_zarr = zarr.group(zarr.DirectoryStore("/mnt/sdomlv2_full/sdomlv2_eve.zarr"))["MEGS-A"]
    eve_timestamps = eve_zarr["Time"][:]
    eve_timestamps = pd.to_datetime(eve_timestamps)
    eve_ions = inference_model.eve_ions
    eve_data = pd.DataFrame(index=eve_timestamps, columns=eve_ions)
    for ion in eve_ions: eve_data[ion] = eve_zarr[ion][:]

    eve_data = eve_data[eve_data>-1]
    eve_data.dropna(inplace=True)
    eve_data.sort_index(inplace=True)

    eve_data = eve_data.loc[f"{year}-01-01":f"{year}-12-31"]

    return eve_data


def get_prediction_data(year=2011):
    predictions = pd.read_parquet(f"/home/jupyter/output_inferences/run1/pred_irradiance_{year}.parquet")
    predictions.dropna(inplace=True)
    return predictions

eve_data = get_eve_data()
predictions = get_prediction_data()

def comparison_plot(date_start, date_end, eve_data, predictions):
    predictions = predictions.loc[date_start:date_end]
    eve_data = eve_data.loc[date_start:date_end]

    fig = px.line()
    for ion in inference_model.eve_ions[:2]:
        fig.add_scatter(x=predictions.index, y=predictions[ion], name=f"Predicted {ion}")
        fig.add_scatter(x=eve_data.index, y=eve_data[ion], name=f"Actual {ion}")
    fig.show()


comparison_plot("2011-01-01", "2011-01-02", eve_data, predictions)



# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

