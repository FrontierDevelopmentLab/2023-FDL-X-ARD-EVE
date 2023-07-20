import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import dask.array as da

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import json
from tqdm import tqdm


class ZarrIrradianceDataset(Dataset):

    def __init__(self, aligndata, aia_zarr_path, eve_zarr_path, wavelengths, ions, freq, months, transformations=None, use_normalizations=False):
        """
        aia_zarr_path --> path: path to zarr aia data
        eve_zarr_path --> path: path to zarr eve data
        wavelengths   --> list: list of channels for aia
        ions          --> list: list of ions for eve
        freq          --> str: cadence used for rounding time series
        transformation: to be applied to aia in theory, but can stay None here
        normalizations: to be applied to transform input data for training / inference
        """
        
        self.aligndata = aligndata
        self.aia_zarr_path = aia_zarr_path
        self.eve_zarr_path = eve_zarr_path
        self.wavelengths = wavelengths
        self.ions = ions
        self.cadence = freq
        self.months = months
        self.transformations = transformations

        self.normalizations = None
        if use_normalizations:
            self.normalizations = self.__calc_normalizations()

        # get data from path
        self.aia_data = zarr.group(zarr.DirectoryStore(self.aia_zarr_path))
        self.eve_data = zarr.group(zarr.DirectoryStore(self.eve_zarr_path))

        self.aligndata = self.aligndata.loc[self.aligndata.index.month.isin(self.months), :]
        

    def __len__(self):
        return self.aligndata.shape[0]
    

    def __getitem__(self, idx):
        aia_image = self.get_aia_image(idx)
        eve_data = self.get_eve(idx)

        return aia_image, eve_data


    def get_aia_image(self, idx):
        aia_image_dict = {}
        for wavelength in self.wavelengths:
            idx_row_element = self.aligndata.iloc[idx]
            idx_wavelength = idx_row_element[f"idx_{wavelength}"]
            year = str(idx_row_element.name.year)
            aia_image_dict[wavelength] = self.aia_data[year][wavelength][idx_wavelength, :, :]
            aia_image_dict[wavelength] -= self.normalizations["AIA"][wavelength]["mean"]
            # aia_image_list[wavelength] /= self.normalizations["AIA"][wavelength]["max"]

        aia_image =  np.array(list(aia_image_dict.values()))
        aia_image = torch.from_numpy(aia_image)
        
        return aia_image
    

    def get_eve(self, idx):
        eve_ion_dict = {}
        for ion in self.ions:
            idx_eve = self.aligndata.iloc[idx]["idx_eve"]
            eve_ion_dict[ion] = self.eve_data['MEGS-A'][ion][idx_eve]
            eve_ion_dict[ion] -= self.normalizations["EVE"][ion]["mean"]
            eve_ion_dict[ion] /= self.normalizations["EVE"][ion]["std"]

        eve_data = np.array(list(eve_ion_dict.values()))
        eve_data = torch.from_numpy(eve_data)

        return eve_data


    def __calc_normalizations(self):

        if Path(self.normalizations_cache_filename).exists():
            with open(self.normalizations_cache_filename, "r") as json_file:
                return json.load(json_file)

        normalizations = {}

        # EVE Normalization
        normalizations["EVE"] = {}
        for ion in self.ions:
            # Note that selecting on idx self.aligndata['idx_eve'] removes negative values from EVE data.
            channel_data = self.eve_data['MEGS-A'][ion][self.aligndata['idx_eve']][:]
            normalizations["EVE"][ion] = {}
            normalizations["EVE"][ion]["count"] = channel_data.shape[0]
            normalizations["EVE"][ion]["sum"] = channel_data.sum()
            normalizations["EVE"][ion]["mean"] = channel_data.mean()
            normalizations["EVE"][ion]["std"] = channel_data.std()
            normalizations["EVE"][ion]["min"] = channel_data.min()
            normalizations["EVE"][ion]["max"] = channel_data.max()
    
        # AIA Normalization
        normalizations["AIA"] = {}
        for wavelength in tqdm(self.wavelengths):
            normalizations["AIA"][wavelength] = {}
            for year in self.aia_data.keys():
                normalizations["AIA"][wavelength] = {}
                normalizations["AIA"][wavelength][year] = {}
                
                idx_channel = self.aligndata[f'idx_{wavelength}']
                idx_channel = idx_channel.loc[idx_channel.index.year == int(year)]
                wavelength_data = self.aia_data[year][wavelength][idx_channel]

                normalizations["AIA"][wavelength][year]["sum"] = 0.
                normalizations["AIA"][wavelength][year]["count"] = 0
                normalizations["AIA"][wavelength][year]["min"] = float("inf")
                normalizations["AIA"][wavelength][year]["max"] = float("-inf")

                # Because AIA data is too big to fit in memory, we need to calculate the normalization in chunks.
                num_chunks = wavelength_data.shape[0] // self.zarr_chunk_size + 1

                for chunk_num in range(num_chunks):
                    idx_left = chunk_num * self.zarr_chunk_size
                    idx_right = min((chunk_num+1) * self.zarr_chunk_size, wavelength_data.shape[0])

                    chunk = wavelength_data[idx_left:idx_right, :, :].flatten()

                    normalizations["AIA"][wavelength][year]["count"] += chunk.shape[0]
                    normalizations["AIA"][wavelength][year]["sum"] += chunk.sum()
                    normalizations["AIA"][wavelength][year]["min"] = min(normalizations["AIA"][wavelength][year]["min"], chunk.min())
                    normalizations["AIA"][wavelength][year]["max"] = max(normalizations["AIA"][wavelength][year]["max"], chunk.max())


        overall_normalizations = {wavelength: {} for wavelength in self.wavelengths}
        for wavelength in self.wavelengths:
            overall_wavelength_normalization = {"sum": 0., "count": 0, "max": float("-inf"), "mean": 0.}
            for year in self.aia_data.keys():
                idx_channel = self.aligndata[f'idx_{wavelength}']
                idx_channel = idx_channel.loc[idx_channel.index.year == int(year)]

                overall_wavelength_normalization["count"] += normalizations["AIA"][wavelength][year]["count"]
                overall_wavelength_normalization["sum"] += normalizations["AIA"][wavelength][year]["sum"]
                overall_wavelength_normalization["max"] = max(overall_wavelength_normalization["max"], normalizations["AIA"][wavelength][year]["max"])
            
            overall_wavelength_normalization["mean"] = overall_wavelength_normalization["sum"] / overall_wavelength_normalization["count"]
            overall_normalizations[wavelength] = overall_wavelength_normalization

        normalizations["AIA"] = overall_normalizations

        with open(self.normalizations_cache_filename, "w") as json_file:
            save_json = str(normalizations)
            save_json = save_json.replace("'", '"')
            json_file.write(save_json)

        return normalizations


class ZarrIrradianceDataModule(pl.LightningDataModule):
    """ Loads paired data samples of AIA EUV images and EVE irradiance measures.

    Note: Input data needs to be paired.
    Parameters
    ----------
    aia_path: path to aia zarr file
    eve_path: path to the EVE zarr data file
    batch_size: batch size (default is 32)
    num_workers: number of workers (needed for the training)
    train_transforms: transformations to be applied on the training set
    val_transforms: transformations to be applied on the validation set
    val_months: 
    """
    def __init__(
            self, 
            aia_path, 
            eve_path, 
            wavelengths, 
            ions, frequency, 
            batch_size: int = 32, 
            num_workers=None,
            train_transforms=None, 
            val_transforms=None, 
            val_months=[10,1], 
            test_months=[11,12], 
            holdout_months=None,
            cache_dir=""
        ):

        super().__init__()
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.aia_path = aia_path
        self.eve_path = eve_path
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.wavelengths = wavelengths
        self.wavelengths.sort()
        self.ions = ions
        self.ions.sort()
        self.cadence = frequency
        self.val_months = val_months
        self.test_months = test_months
        self.holdout_months = holdout_months

        # Chunk size depends on ram available. 500 is a good number for 16GB of ram and 30min cadence.
        # If you have more ram, at same cadence, you can increase this number to speed up the normalization calculation.
        self.zarr_chunk_size = 300

        # Cache filenames
        wavelength_id = "_".join(self.wavelengths)
        ions_id = "_".join(ions).replace(" ", "_")
        self.cache_id = f"{wavelength_id}_{ions_id}_{self.cadence}"
        if "small" in self.aia_path: 
            self.cache_id += "_small"

        self.index_cache_filename = f"{cache_dir}/aligndata_{self.cache_id}.csv"

        self.aia_data = zarr.group(zarr.DirectoryStore(self.aia_path))
        self.eve_data = zarr.group(zarr.DirectoryStore(self.eve_path))

        # Temporal alignment of aia and eve data
        self.aligndata = self.__aligntime()



    def __aligntime(self):
        """
        This function extracts the common indexes across aia and eve datasets, considering potential missing values.
        """

        if Path(self.index_cache_filename).exists():
            aligndata = pd.read_csv(self.index_cache_filename)
            aligndata["Time"] = pd.to_datetime(aligndata["Time"])
            aligndata.set_index("Time", inplace=True)
            return aligndata


        join_series = pd.DataFrame()
        for wavelength in self.wavelengths:
            df_t_aia = pd.DataFrame()

            for key in self.aia_data.keys():
                aia_channel = self.aia_data[key][wavelength]

                # Get observation time
                t_obs_aia_channel = aia_channel.attrs['T_OBS']

                df_tmp_aia = pd.DataFrame({'Time': pd.to_datetime(t_obs_aia_channel, format='mixed'), f"idx_{wavelength}": np.arange(0, len(t_obs_aia_channel))})
                df_t_aia = pd.concat([df_t_aia, df_tmp_aia], ignore_index=True)

            # Enforcing same datetime format
            df_t_aia['Time'] = pd.to_datetime(df_t_aia['Time'], format='mixed').dt.tz_localize(None).dt.round(self.cadence).sort_values()
            df_t_obs_aia = df_t_aia.drop_duplicates(subset='Time', keep='first').set_index('Time')

            if wavelength == self.wavelengths[0]:
                join_series = df_t_obs_aia
            else:
                join_series = join_series.join(df_t_obs_aia, how='inner')


        df_t_eve = pd.DataFrame({'Time': pd.to_datetime(self.eve_data['MEGS-A']['Time'][:]), 'idx_eve': np.arange(0, len(self.eve_data['MEGS-A']['Time']))})
        df_t_eve['Time'] = pd.to_datetime(df_t_eve['Time']).dt.round(self.cadence)
        df_t_obs_eve = df_t_eve.drop_duplicates(subset='Time', keep='first').set_index('Time')
        join_series = join_series.join(df_t_obs_eve, how='inner')

        # remove missing eve data (missing values are labeled with negative values)
        for ion in self.ions:
            ion_data = self.eve_data['MEGS-A'][ion][:]
            join_series = join_series.loc[ion_data[join_series['idx_eve']] > 0, :]

        join_series.sort_index(inplace=True)
        join_series.to_csv(self.index_cache_filename)
        
        return join_series



    def setup(self):
        self.train_months = [i for i in range(1,13) if i not in self.test_months + self.val_months]

        self.train_ds = ZarrIrradianceDataset(self.aligndata, self.aia_path, self.eve_path, self.wavelengths, 
                                              self.ions, self.cadence, self.train_months, self.train_transforms, self.normalizations)
        
        self.test_ds = ZarrIrradianceDataset(self.aligndata, self.aia_path, self.eve_path, self.wavelengths, 
                                             self.ions, self.cadence, self.test_months, self.normalizations)
        
        self.valid_ds = ZarrIrradianceDataset(self.aligndata, self.aia_path, self.eve_path, self.wavelengths, 
                                              self.ions, self.cadence, self.val_months, self.val_transforms, self.normalizations)

