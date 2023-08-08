
import argparse
import os
from pathlib import Path
import json
import sys
import wandb

import albumentations as A
import argparse
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback

from src.irradiance.models.model import HybridIrradianceModel
from src.irradiance.utilities.callback import ImagePredictionLoggerHMI
from src.irradiance.utilities.data_loader import ZarrIrradianceDataModuleHMI

HOME_DIR = os.getenv("HOME")
PROJECT_DIR = f"{HOME_DIR}/2023-FDL-X-ARD-EVE"

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='runs_data/configs/run_test_manuel.json', required=False)
args = parser.parse_args()
with open(args.config_file, 'r') as config_file:
    run_config = json.load(config_file)

wandb_logger = WandbLogger(
    project=run_config['wandb']['project'],
    tags=run_config['wandb']['tags'],
    name=run_config['run_name'],
    notes=run_config['wandb']['notes'],
    config=run_config
)

random_seed = run_config["training_parameters"]["random_seed"]
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.set_default_dtype(torch.float32)

# Data augmentation
if run_config["training_parameters"]["ln_model"]:
    train_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })
else:
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=180, p=0.9, value=0, border_mode=1),
        ToTensorV2()], 
        additional_targets={'y': 'image'}
    )

val_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })

# Initialize data loader (test HMI included)
#data_loader = ZarrIrradianceDataModuleHMI(
#    aia_path=run_config['paths']['aia_path'], 
#    eve_path=run_config['paths']['eve_path'],
#    wavelengths=run_config["sci_parameters"]["aia_wavelengths"],
#    ions=run_config["sci_parameters"]["eve_ions"],
#    frequency=run_config["sci_parameters"]["frequency"],
#    batch_size=run_config["training_parameters"]["batch_size"],
#    num_workers=run_config["training_parameters"]["num_workers"],
#    val_months=run_config["training_parameters"]["val_months"], 
#    test_months=run_config["training_parameters"]["test_months"], 
#    holdout_months=run_config["training_parameters"]["holdout_months"],
#    cache_dir=f"{PROJECT_DIR}/{run_config['paths']['cache_directory']}"
#)

data_loader = ZarrIrradianceDataModuleHMI(
    hmi_path=run_config['paths']['hmi_path'],
    aia_path=run_config['paths']['aia_path'], 
    eve_path=run_config['paths']['eve_path'],
    components=run_config["sci_parameters"]["hmi_components"],
    wavelengths=run_config["sci_parameters"]["aia_wavelengths"],
    ions=run_config["sci_parameters"]["eve_ions"],
    frequency=run_config["sci_parameters"]["frequency"],
    batch_size=run_config["training_parameters"]["batch_size"],
    num_workers=run_config["training_parameters"]["num_workers"],
    val_months=run_config["training_parameters"]["val_months"], 
    test_months=run_config["training_parameters"]["test_months"], 
    holdout_months=run_config["training_parameters"]["holdout_months"],
    cache_dir=f"{PROJECT_DIR}/{run_config['paths']['cache_directory']}"
)
data_loader.setup()


# Initalize model (assuming that if AIA/HMI are not included there is an empty list)
model = HybridIrradianceModel(
        d_input=len(run_config["sci_parameters"]["aia_wavelengths"]) + len(run_config["sci_parameters"]["hmi_components"]),
        d_output=len(run_config["sci_parameters"]["eve_ions"]),
        eve_norm=np.array(data_loader.normalizations["EVE"]["eve_norm"]), 
        cnn_model=run_config["training_parameters"]['cnn_model'], 
        ln_model=run_config["training_parameters"]['ln_model'],
        lr=run_config["training_parameters"]['lr'],
        cnn_dp=run_config["training_parameters"]['cnn_dp']
)

# Plot callback
total_n_valid = data_loader.valid_ds.aligndata.shape[0]
val_data = [data_loader.valid_ds.__getitem__(idx) for idx in range(0, total_n_valid, total_n_valid//4)]
if run_config['paths']['hmi_path'] is not None and run_config['paths']['aia_path'] is not None:
    hmi_aia_stack_val = torch.tensor(np.array([val_data[idx][0] for idx, _ in enumerate(val_data)]))    
    eve_val = torch.tensor(np.array([val_data[idx][1] for idx, _ in enumerate(val_data)]))
    hmi_aia_config = run_config["sci_parameters"]["hmi_components"] + run_config["sci_parameters"]["aia_wavelengths"] 
    image_callback = ImagePredictionLoggerHMI(hmi_aia_stack_val, eve_val, run_config["sci_parameters"]["eve_ions"], hmi_aia_config)
    
elif run_config['paths']['hmi_path'] is not None and run_config['paths']['aia_path'] is None:
    hmi_val = torch.tensor(np.array([val_data[idx][0] for idx, _ in enumerate(val_data)]))    
    eve_val = torch.tensor(np.array([val_data[idx][1] for idx, _ in enumerate(val_data)]))
    image_callback = ImagePredictionLoggerHMI(hmi_val, eve_val, run_config["sci_parameters"]["eve_ions"], run_config["sci_parameters"]["hmi_components"])

else: # HMI is None but AIA is not  
    aia_val = torch.tensor(np.array([val_data[idx][0] for idx, _ in enumerate(val_data)]))
    eve_val = torch.tensor(np.array([val_data[idx][1] for idx, _ in enumerate(val_data)]))
    image_callback = ImagePredictionLoggerHMI(aia_val, eve_val, run_config["sci_parameters"]["eve_ions"], run_config["sci_parameters"]["aia_wavelengths"])


# Checkpoint callback
checkpoint_path = f"{PROJECT_DIR}/{run_config['paths']['checkpoint_path']}/{run_config['run_name']}"
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path,
    monitor='valid_loss',
    mode='min',
    save_top_k=1,
    filename=run_config["paths"]["checkpoint_file_name"]
)


# TODO: Make this more flexible (only linear, only cnn, both, etc.)
if run_config["training_parameters"]['hybrid_loop']:

    # Lambda/Mode callback
    model.set_train_mode("linear")
    model.lr = run_config["training_parameters"]["ln_lr"]
    switch_mode_callback = LambdaCallback(
        on_train_epoch_start=(
            lambda trainer, 
            pl_module: model.set_train_mode("cnn") if trainer.current_epoch > run_config["training_parameters"]["ln_epochs"] else None
        )
    )

    trainer = Trainer(
        default_root_dir=checkpoint_path,
        accelerator="auto",
        devices="auto", # torch.cuda.device_count() if torch.cuda.is_available() else 0,
        max_epochs=(run_config["training_parameters"]["ln_epochs"] + run_config["training_parameters"]["cnn_epochs"]),
        callbacks=[image_callback, checkpoint_callback, switch_mode_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        )

else:
    trainer = Trainer(
        default_root_dir=checkpoint_path,
        accelerator="auto",
        devices="auto", #torch.cuda.device_count() if torch.cuda.is_available() else 0,
        max_epochs=run_config["training_parameters"]["epochs"],
        callbacks=[image_callback, checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
        gradient_clip_val=0.5,
        )

# Train the model ⚡
trainer.fit(model, data_loader)


run_config["model"] = model
run_config["normalizations"] = data_loader.normalizations
full_checkpoint_path = f"{checkpoint_path}/{run_config['paths']['checkpoint_file_name']}.ckpt"
torch.save(run_config, full_checkpoint_path)

trainer.test(model, data_loader, verbose=True)

# Finalize logging
wandb.finish()

