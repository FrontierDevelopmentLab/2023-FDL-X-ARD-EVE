# eve_ions = ['C III', 'Fe IX', 'Fe VIII', 'Fe X', 'Fe XI', 'Fe XII', 'Fe XIII', 'Fe XIV', 'Fe XIX', 'Fe XV', 'Fe XVI', 'Fe XVIII', 'Fe XVI_2', 'Fe XX', 'Fe XX_2', 'Fe XX_3', 'H I', 'H I_2', 'H I_3', 'He I', 'He II', 'He II_2', 'He I_2', 'Mg IX', 'Mg X', 'Mg X_2', 'Ne VII', 'Ne VIII', 'O II', 'O III', 'O III_2', 'O II_2', 'O IV', 'O IV_2', 'O V', 'O VI', 'S XIV', 'Si XII', 'Si XII_2']

from tqdm import tqdm
import numpy as np
import pandas as pd
import zarr
import dask.array as da


aia_data = zarr.group(zarr.DirectoryStore("/mnt/sdomlv2_small/sdomlv2_small.zarr"))
eve_data = zarr.group(zarr.DirectoryStore("/mnt/sdomlv2_small/sdomlv2_eve.zarr"))["MEGS-A"]

aia_wavelengths = list(aia_data["2010"].keys())
eve_ions = list(eve_data.keys())[:-1]

eve_profile = {}
for ion in tqdm(eve_ions):
    ion_data = eve_data[ion][:]
    nonull_data = ion_data[ion_data>0.]
    eve_profile[ion] = nonull_data.shape[0] / ion_data.shape[0]
    print(f"ion: {ion} has {nonull_data.shape[0]} measurements out of {ion_data.shape[0]} total.")

eve_profile = pd.Series(eve_profile).sort_values(ascending=False)
eve_profile

dense_wavelengths = eve_profile[eve_profile > 0.9].index.tolist()
# ['Fe XX', 'Fe VIII', 'Fe X', 'Fe XI', 'Fe XII', 'Fe XIII', 'Fe XIV', 'Fe XV', 'Fe XVIII', 'He II_2', 'He II', 'Fe IX', 'Mg IX', 'Fe XVI']


[print(f'"{w}"', end=", ") for w in eve_ions]


full_eve = ['C III', 'Fe IX', 'Fe VIII', 'Fe X', 'Fe XI', 'Fe XII', 'Fe XIII', 'Fe XIV', 'Fe XIX', 'Fe XV', 'Fe XVI', 'Fe XVIII', 'Fe XX', 'Fe XX_2', 'Fe XX_3', 'H I', 'H I_2', 'H I_3', 'He I', 'He II', 'He II_2', 'He I_2', 'Mg IX', 'Mg X', 'Mg X_2', 'Ne VII', 'Ne VIII', 'O II', 'O III', 'O III_2', 'O II_2', 'O IV', 'O IV_2', 'O V', 'O VI', 'S XIV', 'Si XII', 'Si XII_2']
