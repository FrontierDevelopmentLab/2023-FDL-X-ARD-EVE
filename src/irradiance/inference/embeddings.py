import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import zarr

class AIAEmbeddingsModel:
    def __init__(self):
        checkpoint_path = "/home/richardagalvez/2023-FDL-X-ARD-EVE/checkpoints/MEGS_A_6hr_100epochs/best_model.ckpt"
        state = torch.load(checkpoint_path)
        self.model = state["model"]
        self.model.eval()

        self.aia_wavelengths = state["sci_parameters"]["aia_wavelengths"]
        self.aia_wavelengths.sort()
        self.eve_ions = state["sci_parameters"]["eve_ions"]
        self.eve_ions.sort()
        self.aia_normalizations = state["normalizations"]["AIA"]

        self.aia_data = zarr.group(zarr.DirectoryStore("/mnt/sdomlv2_full/sdomlv2.zarr"))


embeddings_model = AIAEmbeddingsModel()

