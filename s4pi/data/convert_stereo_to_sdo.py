import gc
import glob
import os
import shutil
from warnings import simplefilter

import torch
from astropy import units as u
from dateutil.parser import parse
from tqdm import tqdm

from iti.translate import STEREOToSDO

# set data paths for reading and writing
prediction_path = '/mnt/data/stereo_iti_converted'
data_path = '/mnt/data/stereo_synchronic_prep'
os.makedirs(prediction_path, exist_ok=True)

# find all existing files (converted)
existing = [os.path.basename(f) for f in glob.glob(os.path.join(prediction_path, '171', '*.fits'))]


# find all alinged stereo files (grouped by filename)
basenames_stereo = [[os.path.basename(f) for f in glob.glob('%s/%s/*.fits' % (data_path, wl))] for
                  wl in ['171', '195', '284', '304']]
basenames_stereo = set(basenames_stereo[0]).intersection(*basenames_stereo[1:])
basenames_stereo = sorted(list(basenames_stereo))
basenames_stereo = [b for b in basenames_stereo if b not in existing]

# print device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using: %s' % str(device), torch.cuda.get_device_name(0))

# create ITI translator
translator = STEREOToSDO(n_workers=16, device=device)

# create directories for converted data
dirs = ['171', '195', '284', '304', ]
[os.makedirs(os.path.join(prediction_path, d), exist_ok=True) for d in dirs]

# start the translation as python generator
iti_maps = translator.translate(data_path, basenames=basenames_stereo)

# iteratively process results
simplefilter('ignore')  # ignore int conversion warning
for iti_cube, bn in tqdm(zip(iti_maps, basenames_stereo), total=len(basenames_stereo)):
    for s_map, d in zip(iti_cube, dirs):
        path = os.path.join(os.path.join(prediction_path, d, bn))
        if os.path.exists(path): # skip existing data
            continue
        s_map.save(path) # save map to disk
        gc.collect()