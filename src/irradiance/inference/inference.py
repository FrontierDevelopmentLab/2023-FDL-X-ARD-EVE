import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import json
import zarr
import gcsfs
import google.cloud.bigquery as bq


class IrradianceInferenceModel:
    def __init__(self):
        self.config = json.load("src/irradiance/inference/config.json")
        self.model_config = self.config["inference_model"]
        self.cloud_config = self.config["gcp_config"]

        checkpoint_path = self.model_config["checkpoint_path"] 
        state = torch.load(checkpoint_path)
        self.model = state["model"]
        self.model.eval()
        
        self.aia_wavelengths = state["sci_parameters"]["aia_wavelengths"]
        self.aia_wavelengths.sort()
        self.eve_ions = state["sci_parameters"]["eve_ions"]
        self.eve_ions.sort()
        self.aia_normalizations = state["normalizations"]["AIA"]

        self.aia_root = self.get_zarr_root(bucket=self.cloud_config["bucket"], path=self.cloud_config["aia_path"])
        self.bq_client = bq.Client(project=self.cloud_config["project"], location=self.cloud_config["region"])


    def get_zarr_root(self, bucket: str, path: str) -> zarr.hierarchy.Group:
        print(f"Connecting to zarr root in path: {path} and bucket: {bucket}")

        gcp_zarr = gcsfs.GCSFileSystem(project=self.cloud_config["project"], bucket=bucket, access="read_only", requester_pays=True)
        store = gcsfs.GCSMap(root=f"{bucket}/{path}", gcs=gcp_zarr, check=False, create=True)
        root = zarr.group(store=store)

        return root


    def get_indices(self, time):
        query = f"""
            SELECT
                *
            FROM
                `{self.cloud_config["project"]}.{self.cloud_config["database"]}.{self.cloud_config["index_table"]}`
            WHERE
                Time = '{time}'
            LIMIT 1
        """
        df = self.bq_client.query(query).to_dataframe()
        return df.index.values


    def predict(self, aia_image):
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


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

