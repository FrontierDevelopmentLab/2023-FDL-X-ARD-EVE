import argparse
import os
import json
import sys
import wandb

import logging
logging.basicConfig(filename="run_log.txt", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logger = logging.getLogger(__name__)

import albumentations as A
import argparse
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback

from src.irradiance.models.model import HybridIrradianceModel
from src.irradiance.utilities.callback import ImagePredictionLogger
from src.irradiance.utilities.data_loader import ZarrIrradianceDataModule

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='run_test.json', required=False)
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

# logger.info(f"Run config: {run_config}")
    
random_seed = run_config["training_parameters"]['seed']
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.set_default_dtype(torch.float32)

# Data augmentation
if run_config["training_parameters"]['ln_model']: # or any complex models
    train_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })
else:
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=180, p=0.9, value=0, border_mode=1),
        ToTensorV2()], 
        additional_targets={'y': 'image'}
    )


val_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })

# Initialize data loader
data_loader = ZarrIrradianceDataModule(
    aia_path=run_config["paths"]["aia_path"], 
    eve_path=run_config["paths"]["eve_path"],
    wavelengths=run_config["sci_parameters"]["aia_wavelengths"],
    ions=run_config["sci_parameters"]["ions"],
    frequency=run_config["sci_parameters"]["frequency"],
    batch_size=run_config["training_parameters"]["batch_size"],
    num_workers=run_config["training_parameters"]["num_workers"],
    train_transforms=None,
    val_transforms=None, 
    val_months=run_config["training_parameters"]["val_months"], 
    test_months=run_config["training_parameters"]["test_months"], 
    holdout_months=run_config["training_parameters"]["holdout_months"],
    cache_dir=run_config["paths"]["cache_directory"],
)

data_loader.setup()

# Initalize model
model = HybridIrradianceModel(
        d_input=len(run_config["sci_parameters"]["aia_wavelengths"]),
        d_output=len(run_config["sci_parameters"]["ions"]),
        eve_norm=np.array(data_loader.normalizations["EVE"]["eve_norm"]), 
        cnn_model=run_config["training_parameters"]['cnn_model'], 
        ln_model=run_config["training_parameters"]['ln_model'],
        cnn_dp=run_config["training_parameters"]['cnn_dp'],
        lr=run_config["training_parameters"]['lr']
)


# Plot callback
total_n_valid = data_loader.valid_ds.aligndata.shape[0]
aia_images = torch.tensor(np.array([
    data_loader.valid_ds.get_aia_image(idx) for idx in range(0, total_n_valid, total_n_valid // 4)
    ]))
eve_data = torch.tensor(np.array([
    data_loader.valid_ds.get_eve(idx) for idx in range(0, total_n_valid, total_n_valid // 4)
    ]))
image_callback = ImagePredictionLogger(
    aia_images, 
    eve_data,
    run_config["sci_parameters"]["ions"],
    run_config["sci_parameters"]["aia_wavelengths"]
)

# Checkpoint callback
checkpoint_path = os.path.join(run_config['paths']['checkpoint_path'], str(random_seed))
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path,
    monitor='valid_loss',
    mode='min',
    save_top_k=1,
    filename=run_config['paths']['checkpoint_file_name']
)


# TODO: Make this more flexible (only linear, only cnn, both, etc.)
if run_config["training_parameters"]['hybrid_loop']:

    # Lambda/Mode callback
    model.set_train_mode("linear")
    model.lr = run_config["training_parameters"]["ln_lr"]
    switch_mode_callback = LambdaCallback(
        on_train_epoch_start=(
            lambda trainer, 
            pl_module: model.set_train_mode('cnn') if trainer.current_epoch > run_config["training_parameters"]["ln_epochs"] else None
        )
    )

    trainer = Trainer(
        default_root_dir=checkpoint_path,
        accelerator="gpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        max_epochs=(
            run_config["training_parameters"]["ln_epochs"] + run_config["training_parameters"]["cnn_epochs"]
        ),
        callbacks=[image_callback, checkpoint_callback, switch_mode_callback],
        logger=wandb_logger,
        log_every_n_steps=10
        )

else:
    trainer = Trainer(
        default_root_dir=checkpoint_path,
        accelerator="gpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        max_epochs=run_config["training_parameters"]["epochs"],
        callbacks=[image_callback, checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10
        )

# Train the model ⚡
trainer.fit(model, data_loader)


save_dictionary = run_config
save_dictionary['model'] = model
full_checkpoint_path = os.path.join(checkpoint_path, f"{run_config['paths']['checkpoint_file_name']}.ckpt")
torch.save(save_dictionary, full_checkpoint_path)

# Evaluate on test set
# Load model from checkpoint
# TODO: Correct: KeyError: 'pytorch-lightning_version'
if run_config['paths']['checkpoint_file_name'] is not None:
    state = torch.load(full_checkpoint_path)
    model = state['model']
    trainer.test(model, data_loader)

# Finalize logging
wandb.finish()

