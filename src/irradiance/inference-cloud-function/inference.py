import numpy as np
import torch
import pandas as pd
import json
import zarr
import gcsfs
import google.cloud.bigquery as bq
from datetime import datetime
import functions_framework

from model import HybridIrradianceModel


with open("normalizations.json", "r") as f:
    normalizations = json.load(f)

eve_norm = np.array(normalizations["EVE"]["eve_norm"], dtype=np.float32)

model = HybridIrradianceModel(
        d_input=9,
        d_output=38,
        eve_norm=eve_norm,
        cnn_model="efficientnet_b3",
        lr_linear=0.01,
        lr_cnn=0.0001,
        cnn_dp=0.2
)


class IrradianceInferenceModel:
    def __init__(self):
        with open("inference_config.json", "r") as f:
            self.config = json.load(f)

        with open(self.config["inference_model"]["normalizations_path"], "r") as f:
            self.normalizations = json.load(f)
        
        self.model = HybridIrradianceModel(
                d_input=9,
                d_output=38,
                eve_norm=np.array(self.normalizations["EVE"]["eve_norm"], dtype=np.float32), 
                cnn_model="efficientnet_b3",
                lr_linear=0.01,
                lr_cnn=0.0001,
                cnn_dp=0.2
        )
        self.model.load_state_dict(torch.load(self.config["inference_model"]["checkpoint_path"])["model"])
        
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

        self.sql_column_order = ["timestamp", "inference_time", "model_name"] + self.eve_ions


    def get_zarr_root(self, bucket: str, path: str) -> zarr.hierarchy.Group:
        gcp_zarr = gcsfs.GCSFileSystem(project=self.cloud_config["gcp_project"], bucket=bucket, access="read_only", requester_pays=True)
        store = gcsfs.GCSMap(root=f"{bucket}/{path}", gcs=gcp_zarr, check=False, create=True)
        root = zarr.group(store=store)

        return root


    def get_indices(self, time):
        query = f"""
            SELECT *
            FROM `{self.cloud_config["gcp_project"]}.{self.cloud_config["dataset"]}.{self.cloud_config["index_table"]}`
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

        pred_irradiance = pd.DataFrame([pred_irradiance])
        pred_irradiance = pred_irradiance[self.sql_column_order]
        pred_irradiance = pred_irradiance.iloc[0].to_dict()

        return pred_irradiance

    
    def write_to_bq(self, pred_irradiance):
        table_ref = self.get_table_ref()
        errors = self.bq_client.insert_rows_json(table_ref, [pred_irradiance])
        return errors


    def get_table_ref(self):
        project = self.cloud_config["gcp_project"]
        dataset_id = self.cloud_config["dataset"]
        table_id = self.cloud_config["inference_table"]

        dataset_ref = self.bq_client.dataset(dataset_id, project=project)
        table_ref = dataset_ref.table(table_id)

        return table_ref


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
prediction = inference_model.predict("2010-05-13T01:48:00")
print(prediction)

# result = inference_model.write_to_bq(prediction)
# print(result)


# @functions_framework.http
# def hello_http(request):

#     message_json = request.get_json(silent=True)
#     message_args = request.args
#     print(message_json)
#     time = message_json["time"]

#     inference_model = IrradianceInferenceModel()
#     prediction = inference_model.predict(time)
#     result = inference_model.write_to_bq(prediction)
#     print(result)

#     response = {
#         "status": "OK",
#         "message": "The message was properly received.",
#         "data": message_json
#     }

#     return response
