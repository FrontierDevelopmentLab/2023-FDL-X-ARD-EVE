import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch
from iti.data.editor import LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor
from sunpy.visualization.colormaps import cm

from maps.utilities.reprojection import transform

sdo_img_norm = ImageNormalize(vmin=0, vmax=1, stretch=LinearStretch(), clip=True)

# !stretch is connected to NeRF!
sdo_norms = {171: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=False),
             193: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=False),
             211: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=False),
             304: ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.005), clip=False), }

psi_norms = {171: ImageNormalize(vmin=0, vmax=22348.267578125, stretch=AsinhStretch(0.005), clip=True),
             193: ImageNormalize(vmin=0, vmax=50000, stretch=AsinhStretch(0.005), clip=True),
             211: ImageNormalize(vmin=0, vmax=13503.1240234375, stretch=AsinhStretch(0.005), clip=True), }

sdo_cmaps = {171: cm.sdoaia171, 193: cm.sdoaia193, 211: cm.sdoaia211, 304: cm.sdoaia304}

# sdo_norms = {94: ImageNormalize(vmin=0, vmax=340, stretch=AsinhStretch(0.005), clip=False),
#              131: ImageNormalize(vmin=0, vmax=1400, stretch=AsinhStretch(0.005), clip=False),
#              171: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=False),
#              193: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=False),
#              211: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=False),
#              304: ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.001), clip=False),
#              335: ImageNormalize(vmin=0, vmax=600, stretch=AsinhStretch(0.005), clip=False),
#              1600: ImageNormalize(vmin=0, vmax=4000, stretch=AsinhStretch(0.005), clip=False),
#              1700: ImageNormalize(vmin=0, vmax=4000, stretch=AsinhStretch(0.005), clip=False)
#              }

sdo_norms = {94: ImageNormalize(vmin=0, vmax=2.41, clip=False),
             131: ImageNormalize(vmin=0, vmax=11.6, clip=False),
             171: ImageNormalize(vmin=0, vmax=305, clip=False),
             193: ImageNormalize(vmin=0, vmax=417, clip=False),
             211: ImageNormalize(vmin=0, vmax=151, clip=False),
             304: ImageNormalize(vmin=0, vmax=83.1, clip=False),
             335: ImageNormalize(vmin=0, vmax=7.80, clip=False),
             1600: ImageNormalize(vmin=0, vmax=94.5, clip=False),
             1700: ImageNormalize(vmin=0, vmax=94.5, clip=False)
             }


def loadAIAMap(file_path, resolution=1024, map_reproject=False, calibration='auto'):
    """Load and preprocess AIA file to make them compatible to ITI.


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels of 2.2 solar radii.
    map_reproject: apply preprocessing to remove off-limb (map to heliographic map and transform back to original view).
    calibration: calibration mode for AIAPrepEditor

    Returns
    -------
    the preprocessed SunPy Map
    """
    s_map, _ = LoadMapEditor().call(file_path)
    assert s_map.meta['QUALITY'] == 0, f'Invalid quality flag while loading AIA Map: {s_map.meta["QUALITY"]}'
    s_map = NormalizeRadiusEditor(resolution, padding_factor=0.225).call(s_map)
    try:
        s_map = AIAPrepEditor(calibration=calibration).call(s_map)
    except:
        s_map = AIAPrepEditor(calibration='aiapy').call(s_map)

    if map_reproject:
        s_map = transform(s_map, lat=s_map.heliographic_latitude,
                          lon=s_map.heliographic_longitude, distance=1 * u.AU)
    return s_map

def loadMap(file_path, resolution=1024, map_reproject=False, calibration=None):
    """Load and resample a FITS file (no pre-processing).


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels of 2.2 solar radii.
    map_reproject: apply preprocessing to remove off-limb (map to heliographic map and transform back to original view).
    calibration: calibration mode

    Returns
    -------
    the preprocessed SunPy Map
    """
    s_map, _ = LoadMapEditor().call(file_path)
    s_map = s_map.resample((resolution, resolution) * u.pix)
    if map_reproject:
        s_map = transform(s_map, lat=s_map.heliographic_latitude,
                          lon=s_map.heliographic_longitude, distance=1 * u.AU)
    return s_map

def loadMapStack(file_paths, resolution=1024, remove_nans=True, map_reproject=False, aia_preprocessing=True, calibration='auto', percentile_clip=0.25):
    """Load a stack of FITS files, resample ot specific resolution, and stackt hem.


    Parameters
    ----------
    file_paths: list of files to stack.
    resolution: target resolution in pixels of 2.2 solar radii.
    remove_off_limb: set all off-limb pixels to NaN (optional).

    Returns
    -------
    numpy array with AIA stack
    """
    load_func = loadAIAMap if aia_preprocessing else loadMap
    s_maps = [load_func(file, resolution=resolution, map_reproject=map_reproject, calibration=calibration) for file in file_paths]
    stack = np.stack([sdo_norms[s_map.wavelength.value](s_map.data) for s_map in s_maps]).astype(np.float32)

    if remove_nans:
        stack[np.isnan(stack)] = 0
        stack[np.isinf(stack)] = 0

    if percentile_clip:
        for i in range(stack.shape[0]):
            percentiles = np.percentile(stack[i,:,:].reshape(-1), [percentile_clip,100-percentile_clip])
            stack[i,:,:][stack[i,:,:]<percentiles[0]] = percentiles[0]
            stack[i,:,:][stack[i,:,:]>percentiles[1]] = percentiles[1]

    return stack.data


def str2bool(v):
    """converts string to boolean

        arguments
        ----------
        v: string
            string to convert to boolean

        Returns
        -------
        a boolean value based on the string placed
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # test code
    o_map = loadAIAMap('/mnt/aia-jsoc/171/aia171_2010-05-13T00:00:07.fits')
    o_map.plot()
    plt.savefig('/home/robert_jarolim/results/original_map.jpg')
    plt.close()

    s_map = loadAIAMap('/mnt/aia-jsoc/171/aia171_2010-05-13T00:00:07.fits', map_reproject=True)
    s_map.plot(**o_map.plot_settings)
    plt.savefig('/home/robert_jarolim/results/projected_map.jpg')
    plt.close()