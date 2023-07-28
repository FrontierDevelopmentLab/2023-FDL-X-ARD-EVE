from src.irradiance.utilities.data_loader import ZarrIrradianceDataModule
import json
import os
import zarr



HOME_DIR = os.getenv("HOME")
PROJECT_DIR = f"{HOME_DIR}/2023-FDL-X-ARD-EVE"

import argparse

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='run_test.json', required=False)
args = parser.parse_args()




with open(args.config_file, 'r') as config_file:
    run_config = json.load(config_file)

data_loader = ZarrIrradianceDataModule(
    aia_path=run_config['paths']['aia_path'], 
    eve_path=run_config['paths']['eve_path'],
    wavelengths=run_config["sci_parameters"]["aia_wavelengths"],
    ions=run_config["sci_parameters"]["eve_ions"],
    frequency=run_config["sci_parameters"]["frequency"],
    batch_size=run_config["training_parameters"]["batch_size"],
    num_workers=run_config["training_parameters"]["num_workers"],
    train_transforms=None,
    val_transforms=None, 
    val_months=run_config["training_parameters"]["val_months"], 
    test_months=run_config["training_parameters"]["test_months"], 
    holdout_months=run_config["training_parameters"]["holdout_months"],
    cache_dir=f"{PROJECT_DIR}/{run_config['paths']['cache_directory']}"
)

data_loader.setup()

print(data_loader)
train_ds = data_loader.train_ds
aia_path = data_loader.aia_path



