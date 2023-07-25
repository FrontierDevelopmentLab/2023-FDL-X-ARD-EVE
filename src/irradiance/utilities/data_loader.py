import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import dask.array as da
import dask
from dask.diagnostics import ProgressBar
from dask.array import stats

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import json
from tqdm import tqdm

from functools import lru_cache


class ZarrIrradianceDataset(Dataset):

    def __init__(self, aligndata, aia_data, eve_data, wavelengths, ions, freq, months, transformations=None, normalizations=None):
        """
        aia_path --> path: path to zarr aia data
        eve_path --> path: path to zarr eve data
        wavelengths   --> list: list of channels for aia
        ions          --> list: list of ions for eve
        freq          --> str: cadence used for rounding time series
        transformation: to be applied to aia in theory, but can stay None here
        use_normalizations: to use or not use normalizations, e.g. if this is test data, we don't want to use normalizations
        """
        
        self.aligndata = aligndata
        self.aia_data = aia_data
        self.eve_data = eve_data
        self.wavelengths = wavelengths
        self.wavelengths.sort()
        self.ions = ions
        self.ions.sort()
        self.cadence = freq
        self.months = months
        self.transformations = transformations
        self.normalizations = normalizations

        # get data from path
        self.aligndata = self.aligndata.loc[self.aligndata.index.month.isin(self.months), :]
        

    def __len__(self):
        return self.aligndata.shape[0]
    
    @lru_cache(maxsize=356)
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
            if self.normalizations:
                aia_image_dict[wavelength] -= self.normalizations["AIA"][wavelength]["mean"]
                # aia_image_list[wavelength] /= self.normalizations["AIA"][wavelength]["max"]

        aia_image =  np.array(list(aia_image_dict.values()))
        
        return aia_image
    

    def get_eve(self, idx):
        eve_ion_dict = {}
        for ion in self.ions:
            idx_eve = self.aligndata.iloc[idx]["idx_eve"]
            eve_ion_dict[ion] = self.eve_data['MEGS-A'][ion][idx_eve]
            if self.normalizations:
                eve_ion_dict[ion] -= self.normalizations["EVE"][ion]["mean"]
                eve_ion_dict[ion] /= self.normalizations["EVE"][ion]["std"]

        eve_data = np.array(list(eve_ion_dict.values()), dtype=np.float32)

        return eve_data


    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output



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
    def __init__(self, aia_path, eve_path, wavelengths, ions, frequency, batch_size: int = 32, num_workers=None, train_transforms=None, 
                 val_transforms=None, val_months=[10,1], test_months=[11,12],  holdout_months=[], cache_dir=""):

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
        self.cache_dir = cache_dir

        self.train_months = [i for i in range(1,13) if i not in self.test_months + self.val_months + self.holdout_months]

        self.aia_data = zarr.group(zarr.DirectoryStore(self.aia_path))
        self.eve_data = zarr.group(zarr.DirectoryStore(self.eve_path))

        # Cache filenames
        wavelength_id = "_".join(self.wavelengths)
        ions_id = "_".join(ions).replace(" ", "_")
        self.cache_id = f"{wavelength_id}_{ions_id}_{self.cadence}"
        if "small" in self.aia_path: 
            self.cache_id += "_small"

        self.index_cache_filename = f"{cache_dir}/aligndata_{self.cache_id}.csv"
        self.normalizations_cache_filename = f"{cache_dir}/normalizations_{self.cache_id}.json"

        # Temporal alignment of aia and eve data
        self.aligndata = self.__aligntime()        
        self.normalizations = self.__calc_normalizations()

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output


    def __aligntime(self):
        """
        This function extracts the common indexes across aia and eve datasets, considering potential missing values.
        """

        # Check the cache
        if Path(self.index_cache_filename).exists():
            aligndata = pd.read_csv(self.index_cache_filename)
            aligndata["Time"] = pd.to_datetime(aligndata["Time"])
            aligndata.set_index("Time", inplace=True)
            return aligndata


        # join_series = pd.DataFrame()
        # for wavelength in self.wavelengths:
        #     df_t_aia = pd.DataFrame()

        #     for key in self.aia_data.keys():
        #         aia_channel = self.aia_data[key][wavelength]

        #         # Get observation time
        #         t_obs_aia_channel = aia_channel.attrs['T_OBS']

        #         df_tmp_aia = pd.DataFrame({'Time': pd.to_datetime(t_obs_aia_channel, format='mixed'), f"idx_{wavelength}": np.arange(0, len(t_obs_aia_channel))})
        #         df_t_aia = pd.concat([df_t_aia, df_tmp_aia], ignore_index=True)

        #     # Enforcing same datetime format
        #     df_t_aia['Time'] = pd.to_datetime(df_t_aia['Time'], format='mixed').dt.tz_localize(None).dt.round(self.cadence).sort_values()
        #     df_t_obs_aia = df_t_aia.drop_duplicates(subset='Time', keep='first').set_index('Time')

        #     if wavelength == self.wavelengths[0]:
        #         join_series = df_t_obs_aia
        #     else:
        #         join_series = join_series.join(df_t_obs_aia, how='inner')


        # AIA
        for i,wavelength in enumerate(self.wavelengths):
            for j, year in enumerate(range(2010, 2015)): # EVE data only goes up to 2014
                print(year, wavelength)
                aia_channel = self.aia_data[year][wavelength]

                # get observation time
                t_obs_aia_channel = aia_channel.attrs['T_OBS'] 
                if j == 0:
                    # transform to DataFrame
                    # AIA
                    df_t_aia = pd.DataFrame({'Time': pd.to_datetime(t_obs_aia_channel,format='mixed'), f'idx_{wavelength}': np.arange(0,len(t_obs_aia_channel))})

                else:
                    df_tmp_aia =pd.DataFrame({'Time': pd.to_datetime(t_obs_aia_channel, format='mixed'), f'idx_{wavelength}': np.arange(0,len(t_obs_aia_channel))})
                    df_t_aia = pd.concat([df_t_aia, df_tmp_aia], ignore_index = True)

            # Enforcing same datetime format
            transform_datetime = lambda x: pd.to_datetime(x, format='mixed').strftime('%Y-%m-%d %H:%M:%S')
            df_t_aia['Time'] = df_t_aia['Time'].apply(transform_datetime)
            df_t_aia['Time'] = pd.to_datetime(df_t_aia['Time']).dt.tz_localize(None) # this is needed for timezone-naive type
            df_t_aia['Time'] = df_t_aia['Time'].dt.round(self.cadence)
            df_t_obs_aia = df_t_aia.drop_duplicates(subset='Time', keep='first') # removing potential duplicates derived by rounding
            df_t_obs_aia.set_index('Time', inplace = True)

            if i == 0:
                join_series = df_t_obs_aia
            else:
                join_series = join_series.join(df_t_obs_aia, how='inner')

        # EVE
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


    def __calc_normalizations(self):

        if Path(self.normalizations_cache_filename).exists():
            with open(self.normalizations_cache_filename, "r") as json_file:
                return json.load(json_file)

        normalizations = {}
        normalizations_align = self.aligndata.copy()
        normalizations_align = normalizations_align[normalizations_align.index.month.isin(self.train_months)]

        normalizations['EVE'] = self.__calc_eve_normalizations(normalizations_align)
        normalizations['AIA'] = self.__calc_aia_normalizations(normalizations_align)

        with open(self.normalizations_cache_filename, "w") as json_file:
            save_json = str(normalizations)
            save_json = save_json.replace("'", '"')
            json_file.write(save_json)

        return normalizations

    def __calc_eve_normalizations(self, normalizations_align) -> dict:

        # EVE Normalization
        normalizations_eve = {}
        for ion in self.ions:
            # Note that selecting on idx normalizations_align['idx_eve'] removes negative values from EVE data.
            channel_data = self.eve_data['MEGS-A'][ion][normalizations_align['idx_eve']][:]
            normalizations_eve[ion] = {}
            normalizations_eve[ion]["count"] = channel_data.shape[0]
            normalizations_eve[ion]["sum"] = channel_data.sum()
            normalizations_eve[ion]["mean"] = channel_data.mean()
            normalizations_eve[ion]["std"] = channel_data.std()
            normalizations_eve[ion]["min"] = channel_data.min()
            normalizations_eve[ion]["max"] = channel_data.max()
    
        normalizations_eve["eve_norm"] = [
            [normalizations_eve[key]["mean"] for key in normalizations_eve.keys()],
            [normalizations_eve[key]["std"] for key in normalizations_eve.keys()]
        ]
        return normalizations_eve
        

    def __calc_aia_normalizations(self, normalizations_align) -> dict:
        normalizations_aia = {}

        for wavelength in self.wavelengths:
            wavelength_data = da.from_array(self.aia_data[2010][wavelength])

            for year in range(2011, 2015): # EVE data only goes up to 2014.                    
                wavelength_data_year = da.from_array(self.aia_data[year][wavelength])
                wavelength_data = da.concatenate([wavelength_data, wavelength_data_year], axis=0)

            wavelength_data = wavelength_data[normalizations_align[f'idx_{wavelength}']]
            
            print(f"\nCalculating normalizations for wavelength {wavelength}:")
            print("-"*50)

            normalizations_aia[wavelength] = {}

            print(f"Sum:")
            with ProgressBar():
                normalizations_aia[wavelength]["sum"] = wavelength_data.sum().compute()

            print(f"Max Pixel Value:")
            with ProgressBar():
                normalizations_aia[wavelength]["max"] = wavelength_data.max().compute()

            print(f"Standard Deviation:")
            with ProgressBar():
                normalizations_aia[wavelength]["std"] = wavelength_data.std().compute()

            print(f"Skew:")
            with ProgressBar():
                normalizations_aia[wavelength]["skew"] = stats.skew(wavelength_data.flatten()).compute()

            print(f"Kurtosis:")
            with ProgressBar():
                normalizations_aia[wavelength]["kurtosis"] = stats.kurtosis(wavelength_data.flatten()).compute()

            normalizations_aia[wavelength]["image_count"] = wavelength_data.shape[0]
            normalizations_aia[wavelength]["pixel_count"] = wavelength_data.shape[0] * wavelength_data.shape[1] * wavelength_data.shape[2]
            normalizations_aia[wavelength]["mean"] = normalizations_aia[wavelength]["sum"] / normalizations_aia[wavelength]["pixel_count"]
            
            
        return normalizations_aia

    def setup(self, stage=None):

        self.train_ds = ZarrIrradianceDataset(self.aligndata, self.aia_data, self.eve_data, self.wavelengths, 
                                              self.ions, self.cadence, self.train_months, self.train_transforms,
                                              normalizations=self.normalizations)

        self.valid_ds = ZarrIrradianceDataset(self.aligndata, self.aia_data, self.eve_data, self.wavelengths, 
                                              self.ions, self.cadence, self.val_months, self.val_transforms, 
                                              normalizations=self.normalizations)
        
        self.test_ds = ZarrIrradianceDataset(self.aligndata, self.aia_data, self.eve_data, self.wavelengths, 
                                             self.ions, self.cadence, self.test_months, normalizations=self.normalizations)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True)
    

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)