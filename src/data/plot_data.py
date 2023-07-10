# quality check for NeRF input
import glob
import multiprocessing
import os

import matplotlib.pyplot as plt
import shutil

from astropy.io.fits import getheader
from tqdm import tqdm

from s4pi.data.utils import loadAIAMap, loadMap, sdo_norms

sdo_file_path = '/mnt/nerf-data/sdo_2012_11/304/*'
iti_a_file_path = '/mnt/nerf-data/stereo_2012_11_converted/304/*_A.fits'
iti_b_file_path = '/mnt/nerf-data/stereo_2012_11_converted/304/*_B.fits'

sdo_paths = sorted(glob.glob(sdo_file_path))
iti_a_paths = sorted(glob.glob(iti_a_file_path))
iti_b_paths = sorted(glob.glob(iti_b_file_path))

sdo_paths = [p for p in sdo_paths if getheader(p, 1)['QUALITY'] == 0]

out_path = '/home/robert_jarolim/verification'
shutil.rmtree(out_path)
os.makedirs(out_path)

with multiprocessing.Pool(os.cpu_count() // 2) as p:
    for path, s_maps in tqdm(zip(sdo_paths, p.imap(loadAIAMap, sdo_paths))):
        s_maps.plot(norm=sdo_norms[304])
        bn = os.path.basename(path)
        plt.savefig(f'{out_path}/sdo_{bn}.jpg')
        plt.close()
    for path, s_maps in tqdm(zip(iti_a_paths, p.imap(loadMap, iti_a_paths))):
        s_maps.plot(norm=sdo_norms[304])
        bn = os.path.basename(path)
        plt.savefig(f'{out_path}/stereo_a_{bn}.jpg')
        plt.close()
    for path, s_maps in tqdm(zip(iti_b_paths, p.imap(loadMap, iti_b_paths))):
        s_maps.plot(norm=sdo_norms[304])
        bn = os.path.basename(path)
        plt.savefig(f'{out_path}/stereo_b_{bn}.jpg')
        plt.close()


shutil.make_archive(out_path, 'zip', out_path)