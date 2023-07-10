import argparse
import os
import sys

import albumentations as A
import argparse
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pathlib import Path

from irradiance.models.model import IrradianceModel
from irradiance.utilities.callback import ImagePredictionLogger
from irradiance.utilities.data_loader import IrradianceDataModule

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

p.add_argument('-stack_csv_path', type=str,
               default='/mnt/converted_data_1hr/matches/merged_256_limb.csv',
               help='path to the CSV with the AIA image stack paths.')

p.add_argument('-eve_norm_path', type=str,
               default='/mnt/converted_data_1hr/eve_normalization.npy',
               help='path to the EVE normalization.')

p.add_argument('-eve_wl_names', type=str,
               default='/mnt/converted_data_1hr/eve_wl_names.npy',
               help='path to the EVE norm')

p.add_argument('-eve_npy_path', type=str,
               default='/mnt/converted_data_1hr/eve_converted.npy',
               help='path to converted EVE data')

p.add_argument('-model_checkpoints', type=str,
               default="/mnt/training/data_1hr_cad",
               help='path to the output directory.')

p.add_argument('-max_epochs', type=int,
               default=200,
               help= 'epochs for the training')

p.add_argument('-seed', type=int, default=3110, help='seed for the training.')

p.add_argument('-checkpoint_name', type=str, default=None, help='Filename to load checkpoint')

args = p.parse_args()

# # For repeatability
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

print('Seed: ', seed)

checkpoint_path = os.path.join(args.model_checkpoints, str(seed))

eve_norm = np.load(args.eve_norm_path)

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=180, p=0.9, value=0, border_mode=1),
    ToTensorV2()], additional_targets={'y': 'image', })

val_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })

# Init our model
data_loader = IrradianceDataModule(args.stack_csv_path, args.eve_npy_path, num_workers=os.cpu_count() // 2,
                                   train_transforms=train_transforms, val_transforms=val_transforms)
data_loader.setup()
model = IrradianceModel(d_input=4, d_output=14, eve_norm=eve_norm)

# initialize logger
wandb_logger = WandbLogger(name=None, entity='4pi-euv',  project='irradiance-test',  group=f'seed-{seed}')

# initialize plot callback - change to valid_ds
total_n_valid = len(data_loader.valid_ds)
plot_data = [data_loader.valid_ds[i] for i in range(0, total_n_valid, total_n_valid // 8)]
plot_images = torch.stack([image for image, eve in plot_data])
plot_eve = torch.stack([eve for image, eve in plot_data])
wl_names = np.load(args.eve_wl_names, allow_pickle=True)
image_callback = ImagePredictionLogger(plot_images, plot_eve, wl_names)

checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                        monitor='valid_loss',
                                        mode='min',
                                        save_top_k=1)

# Initialize a train
trainer = Trainer(
    default_root_dir=checkpoint_path,
    accelerator="gpu",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=args.max_epochs,
    callbacks=[image_callback, checkpoint_callback],
    logger=wandb_logger,
    log_every_n_steps=10
    )

# Train the model ⚡
trainer.fit(model, data_loader)

# Evaluate on test set
# Load model from checkpoint
if checkpoint_name is not None:
    full_chkpt_path = os.path.join(checkpoint_path, args.checkpoint_name)
    model = IrradianceModel.load_from_checkpoint(full_chkpt_path, d_input=4, d_output=14, eve_norm=eve_norm)
    trainer.test(model, data_loader)

trainer.save_checkpoint(os.path.join(checkpoint_path, "final_model.ckpt"), weights_only=False)



