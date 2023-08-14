import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import json
import zarr
import gcsfs
import google.cloud.bigquery as bq
from datetime import datetime


class IrradianceInferenceModel:
    def __init__(self):
        with open("src/irradiance/inference/config.json", "r") as f:
            self.config = json.load(f)
        
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

        self.aia_root = self.get_zarr_root(bucket=self.cloud_config["zarr_bucket"], path=self.cloud_config["aia_path"])
        # self.hmi_root = self.get_zarr_root(bucket=self.cloud_config["zarr_bucket"], path=self.cloud_config["hmi_path"])
        self.bq_client = bq.Client(project=self.cloud_config["gcp_project"], location=self.cloud_config["gcp_region"])


    def get_zarr_root(self, bucket: str, path: str) -> zarr.hierarchy.Group:
        gcp_zarr = gcsfs.GCSFileSystem(project=self.cloud_config["gcp_project"], bucket=bucket, access="read_only", requester_pays=True)
        store = gcsfs.GCSMap(root=f"{bucket}/{path}", gcs=gcp_zarr, check=False, create=True)
        root = zarr.group(store=store)

        return root


    def get_indices(self, time):
        query = f"""
            SELECT *
            FROM `{self.cloud_config["gcp_project"]}.{self.cloud_config["database"]}.{self.cloud_config["index_table"]}`
            WHERE Time = '{time}'
            LIMIT 1
        """
        result = self.bq_client.query(query).result()
        result = result.to_dataframe()
        if result.shape[0] == 0:
            raise ValueError(f"No data found for time: {time}")
        result = result.iloc[0].to_dict()
        return result


    def predict(self, time):
        aia_image = self.get_aia_image(time)
        with torch.no_grad():
            pred_irradiance = self.model.forward_unnormalize(aia_image).numpy()
        pred_irradiance = {ion: pred_irradiance[0][i] for i, ion in enumerate(self.eve_ions)}
        pred_irradiance["timestamp"] = time
        pred_irradiance["inference_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        pred_irradiance["model_name"] = self.model_config["model_name"]
        return pred_irradiance

    
    def write_to_bq(self, prediction):
        pred_irradiance = pd.DataFrame([prediction])
        pred_irradiance.to_gbq(
            destination_table=f"{self.cloud_config['database']}.{self.cloud_config['inference_table']}",
            project_id=self.cloud_config["gcp_project"],
            if_exists="append"
        )


    def get_aia_image(self, time):
        indices = self.get_indices(time)
        time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
        aia_image = {}
        for wavelength in self.aia_wavelengths:
            year = int(time.year)
            idx = indices[f"idx_{wavelength}"]
            aia_image[wavelength] = self.aia_root[year][wavelength][idx,:,:]
            aia_image[wavelength] -= self.aia_normalizations[wavelength]["mean"]
            aia_image[wavelength] /= self.aia_normalizations[wavelength]["std"]

        aia_image = np.array([np.stack([aia_image[wavelength] for wavelength in self.aia_wavelengths], axis=0)])
        aia_image = torch.from_numpy(aia_image)
        
        return aia_image


    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


inference_model = IrradianceInferenceModel()
prediction = inference_model.predict("2011-01-01T00:24:00")
inference_model.write_to_bq(prediction)
print(prediction)

from copy import copy
self = copy(inference_model)





# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

