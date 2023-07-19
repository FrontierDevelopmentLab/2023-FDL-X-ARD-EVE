import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import dask.array as da

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl

from data.utils import loadMapStack


class ZarrIrradianceDataset(Dataset):

    def __init__(self, aligndata, aia_zarr_path, eve_zarr_path, wavelengths, ions, freq, months, transformations=None):
        """
        aia_zarr_path --> path: path to zarr aia data
        eve_zarr_path --> path: path to zarr eve data
        wavelengths   --> list: list of channels for aia
        ions          --> list: list of ions for eve
        freq          --> str: cadence used for rounding time series
        transformation: to be applied to aia in theory, but can stay None here
        """
        
        self.aligndata = aligndata
        self.aia_zarr_path = aia_zarr_path
        self.eve_zarr_path = eve_zarr_path
        self.wavelengths = wavelengths
        self.ions = ions
        self.cadence = freq
        self.months = months
        self.transformations = transformations

        # get data from path
        self.aia_data = zarr.group(zarr.DirectoryStore(self.aia_zarr_path))
        self.eve_data = zarr.group(zarr.DirectoryStore(self.eve_zarr_path))

        self.aligndata = self.aligndata.loc[self.aligndata.index.month.isin(self.months),:]
        
    def __len__(self):
        return self.aligndata.shape[0]
    
    def __getitem__(self, idx):
        index_row = self.aligndata.iloc[idx,:]
        aia_image_list = []

        for wavelength in self.wavelengths:
            # select data from zarr
            aia_channel = self.aia_data[index_row[f'year_{wavelength}']][wavelength]
            
            # convert data to dask
            aia_image_list.append(torch.tensor(np.array(da.from_array(aia_channel)[index_row[f'index_{wavelength}'],:,:])))

        euv_images = torch.stack(aia_image_list)
    
        if self.transformations is not None:
            # transform as RGB + y to use transformations
            euv_images = euv_images.transpose() # this is probably needed for the dimension of the file they are passing
            # transformed = self.transformations(image=euv_images[..., :3], y=euv_images[..., 3:])
            # euv_images = torch.cat([transformed['image'], transformed['y']], dim=0)
            transformed = self.transformations(image=euv_images) # this applies transformations (here it just "toTensor", but 
                                                                # it could be like rotating, change colors etc..)
            euv_images = transformed['image']
        
        eve_ion_list = []
        for ion in self.ions:
            eve_ion_list.append(torch.tensor(np.array(da.from_array(self.eve_data['MEGS-A'][ion])[index_row['index_eve']])))
        
        eve_data = torch.stack(eve_ion_list)

        return euv_images, eve_data


class ZarrIrradianceDataModule(pl.LightningDataModule):
    def __init__(self, aia_path, eve_path, wavelengths, ions, frequency, batch_size: int = 32, num_workers=None,
                 train_transforms=None, val_transforms=None, val_months=[10,1], test_months=[11,12], holdout_months=None):
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

        super().__init__()
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.aia_path = aia_path
        self.eve_path = eve_path
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.wavelengths = wavelengths
        self.ions = ions
        self.cadence = frequency
        self.val_months = val_months
        self.test_months = test_months
        self.holdout_months = holdout_months
        self.zarr_chunk_size = 1000

        self.wavelengths.sort()
        wavelength_id = "_".join(self.wavelengths)

        ions = [ion.replace(" ", "_") for ion in self.ions]
        ions.sort()
        ions_id = "_".join(ions)

        self.cache_id = f"{wavelength_id}_{ions_id}_{self.cadence}"
        self.index_cache_filename = f"aligndata_{self.cache_id}.csv"
        self.normalizations_cache_filename = f"normalizations_{self.cache_id}.json"

        self.aia_data = zarr.group(zarr.DirectoryStore(self.aia_path))
        self.eve_data = zarr.group(zarr.DirectoryStore(self.eve_path))

        # Temporal alignment of aia and eve data
        self.aligndata = self.__aligntime__()

        # Calculate normalization of the data
        self.normalizations = self._calc_normalizations_()
        
    def __aligntime__(self):
        """
        This function extracts the common indexes across aia and eve datasets, considering potential missing values.
        """

        if Path(self.index_cache_filename).exists():
            aligndata = pd.read_csv(self.index_cache_filename)
            aligndata["Time"] = pd.to_datetime(aligndata["Time"])
            aligndata.set_index("Time", inplace=True)
            return aligndata

        # select data from zarr
        for i,wavelength in enumerate(self.wavelengths):
            for j,key in enumerate(self.aia_data.keys()):
                aia_channel = self.aia_data[key][wavelength]

                # get observation time
                t_obs_aia_channel = aia_channel.attrs['T_OBS'] 
                if j == 0:
                    # transform to DataFrame
                    # AIA
                    df_t_aia = pd.DataFrame({'Time': pd.to_datetime(t_obs_aia_channel,format='mixed'), f'index_{wavelength}': np.arange(0,len(t_obs_aia_channel))})
                    df_t_aia[f'year_{wavelength}'] = key

                else:
                    df_tmp_aia =pd.DataFrame({'Time': pd.to_datetime(t_obs_aia_channel, format='mixed'), f'index_{wavelength}': np.arange(0,len(t_obs_aia_channel))})
                    df_tmp_aia[f'year_{wavelength}'] = key
                    df_t_aia = pd.concat([df_t_aia, df_tmp_aia], ignore_index = True)
            
            # enforcing same datetime format
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
        t_obs_eve_channel = np.array(da.from_array(self.eve_data['MEGS-A']['Time'])).tolist()
        df_t_eve = pd.DataFrame({'Time': pd.to_datetime(t_obs_eve_channel), 'index_eve': np.arange(0,len(t_obs_eve_channel))})
        df_t_eve['Time'] = pd.to_datetime(df_t_eve['Time'])  
        df_t_eve['Time'] = df_t_eve['Time'].dt.round(self.cadence)
        df_t_obs_eve = df_t_eve.drop_duplicates(subset='Time', keep='first')
        df_t_obs_eve.set_index('Time', inplace = True)
        join_series = join_series.join(df_t_obs_eve, how='inner')

        # remove missing eve data (missing values are labeled with negative values)
        for ion in self.ions:
            ion_data = np.array(da.from_array(self.eve_data['MEGS-A'][ion]))
            join_series = join_series.loc[ion_data[join_series['index_eve']] > 0,:]

        join_series.to_csv(self.cache_filename)
        return join_series


    def _calc_normalizations_(self):

        if Path(self.normalizations_cache_filename).exists():
            with open(self.normalizations_cache_filename, "r") as json_file:
                return json.load(json_file)

        normalizations = {}

        # EVE Normalization
        normalizations["EVE"] = {}
        for ion in self.ions:
            # Note that selecting on idx self.aligndata['index_eve'] removes negative values from EVE data.
            channel_data = self.eve_data['MEGS-A'][ion][:][self.aligndata['index_eve']]
            normalizations["EVE"][ion] = {}
            normalizations["EVE"][ion]["count"] = channel_data.shape[0]
            normalizations["EVE"][ion]["sum"] = channel_data.sum()
            normalizations["EVE"][ion]["mean"] = channel_data.mean()
            normalizations["EVE"][ion]["std"] = channel_data.std()
            normalizations["EVE"][ion]["min"] = channel_data.min()
            normalizations["EVE"][ion]["max"] = channel_data.max()
    
        # AIA Normalization
        normalizations["AIA"] = {}
        for wavelength in self.wavelengths:
            normalizations["AIA"][wavelength] = {}
            for year in self.aia_data.keys():
                wavelength_data = self.aia_data[year][wavelength]
                normalizations["AIA"][wavelength] = {}
                normalizations["AIA"][wavelength][year] = {}
                
                # TO DO: How can we incorporate aligndata here to filter AIA?
                # I'm OK with this approximation for now, but we should think about it.
                idx_channel = self.aligndata[f'index_{wavelength}']
                idx_channel = idx_channel.loc[idx_channel.index.year == int(year)]

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


        with open(self.normalizations_cache_filename, "w") as json_file:
            save_json = str(normalizations)
            save_json = save_json.replace("'", '"')
            json_file.write(save_json)

        return normalizations


    def setup(self, stage=None):
        train_months = [i for i in range(1,13) if i not in self.test_months]
        self.train_months = [i for i in train_months if i not in self.val_months]

        self.train_ds = ZarrIrradianceDataset(self.aligndata, self.aia_path, self.eve_path, 
                                              self.wavelengths, self.ions, self.cadence, self.train_months, self.train_transforms)
        
        self.test_ds = ZarrIrradianceDataset(self.aligndata, self.aia_path, self.eve_path, 
                                              self.wavelengths, self.ions, self.cadence, self.test_months)
        
        self.valid_ds = ZarrIrradianceDataset(self.aligndata, self.aia_path, self.eve_path, 
                                              self.wavelengths, self.ions, self.cadence, self.val_months, self.val_transforms)


class IrradianceDataModule(pl.LightningDataModule):

    def __init__(self, stacks_csv_path, eve_npy_path, wavelengths, batch_size: int = 32, num_workers=None,
                 train_transforms=None, val_transforms=None, val_months=[10, 1], test_months=[11,12], holdout_months=None):
        """ Loads paired data samples of AIA EUV images and EVE irradiance measures.

        Input data needs to be paired.
        Parameters
        ----------
        stacks_csv_path: path to the matches
        eve_npy_path: path to the EVE data file
        """
        super().__init__()
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.stacks_csv_path = stacks_csv_path
        self.eve_npy_path = eve_npy_path
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.wavelengths = wavelengths
        self.val_months = val_months
        self.test_months = test_months
        self.holdout_months = holdout_months

    # TODO: Extract SDO dats for train/valid/test sets
    def setup(self, stage=None):
        # load data stacks (paired samples)
        df = pd.read_csv(self.stacks_csv_path, parse_dates=['eve_dates'])
        eve_data = np.load(self.eve_npy_path)

        print(len(eve_data), len(df))
        assert len(eve_data) == len(df), 'Inconsistent data state. EVE and AIA stacks do not match.'


        test_cond_2014 = df.eve_dates.dt.year.isin([2014])
        val_condition = df.eve_dates.dt.month.isin(self.val_months)
        test_condition = df.eve_dates.dt.month.isin(self.test_months) | test_cond_2014 

        if self.holdout_months is not None:
            holdout_condition = df.eve_dates.dt.month.isin(self.holdout_months)
            train_condition = ~(val_condition | test_condition | holdout_condition)
        else:
            train_condition = ~(val_condition | test_condition)

        valid_df = df[val_condition]
        test_df = df[test_condition]
        train_df = df[train_condition]

        self.train_ds = IrradianceDataset(np.array(train_df.aia_stack), eve_data[train_df.index], 
                                          self.wavelengths, self.train_transforms)
        self.valid_ds = IrradianceDataset(np.array(valid_df.aia_stack), eve_data[valid_df.index], 
                                          self.wavelengths, self.val_transforms)
        self.test_ds = IrradianceDataset(np.array(test_df.aia_stack), eve_data[test_df.index], 
                                         self.wavelengths, self.val_transforms)

        self.train_dates = train_df['eve_dates']
        self.valid_dates = valid_df['eve_dates']
        self.test_dates = test_df['eve_dates']


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class IrradianceDataset(Dataset):

    def __init__(self, euv_paths, eve_irradiance, wavelengths, transformations=None):
        """ Loads paired data samples of AIA EUV images and EVE irradiance measures.

        Input data needs to be paired.
        Parameters
        ----------
        euv_paths: list of numpy paths as string (n_samples, )
        eve_irradiance: list of the EVE data values (n_samples, eve_channels)
        """
        assert len(euv_paths) == len(eve_irradiance), 'Input data needs to be paired!'
        self.euv_paths = euv_paths
        self.eve_irradiance = eve_irradiance
        self.transformations = transformations
        self.wavelengths = wavelengths

    def __len__(self):
        return len(self.euv_paths)

    def __getitem__(self, idx):
        euv_images = np.load(self.euv_paths[idx])[self.wavelengths, ...]
        eve_data = self.eve_irradiance[idx]
        # TODO: Understand what's happening below???
        if self.transformations is not None:
            # transform as RGB + y to use transformations
            euv_images = euv_images.transpose()
            # transformed = self.transformations(image=euv_images[..., :3], y=euv_images[..., 3:])
            # euv_images = torch.cat([transformed['image'], transformed['y']], dim=0)
            transformed = self.transformations(image=euv_images)
            euv_images = transformed['image']

        return euv_images, torch.tensor(eve_data, dtype=torch.float32)


class FITSDataset(Dataset):
    def __init__(self, paths, resolution=512, map_reproject=False, aia_preprocessing=True):
        """ Loads data samples of AIA EUV images.

        Parameters
        ----------
        euv_paths: list of numpy paths as string (n_samples, )
        """
        self.paths = paths
        self.resolution = resolution
        self.map_reproject = map_reproject
        self.aia_preprocessing = aia_preprocessing

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        euv_images = loadMapStack(self.paths[idx], resolution=self.resolution, remove_nans=True,
                                  map_reproject=self.map_reproject, aia_preprocessing=self.aia_preprocessing)
        return torch.tensor(euv_images, dtype=torch.float32)


class NumpyDataset(Dataset):

    def __init__(self, euv_paths, euv_wavelengths):
        """ Loads data samples of AIA EUV images.

        Parameters
        ----------
        euv_paths: list of numpy paths as string (n_samples, )
        """
        self.euv_paths = euv_paths
        self.euv_wavelengths = euv_wavelengths

    def __len__(self):
        return len(self.euv_paths)

    def __getitem__(self, idx):
        euv_images = np.load(self.euv_paths[idx])[self.euv_wavelengths, ...]
        return torch.tensor(euv_images, dtype=torch.float32)


class ArrayDataset(Dataset):

    def __init__(self, data):
        """ Loads data samples of AIA EUV images.

        Parameters
        ----------
        euv_paths: list of numpy paths as string (n_samples, )
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)