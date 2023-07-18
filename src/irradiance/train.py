import argparse
import os
import yaml
import itertools
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
from src.irradiance.utilities.callback import ImagePredictionLogger
from src.irradiance.utilities.data_loader import IrradianceDataModule, ZarrIrradianceDataModule

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default='hyperparams.yaml', required=False)
args = parser.parse_args()
with open(args.config_path, 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.SafeLoader)

dic_values = [i for i in config_data['config'].values()]
print(dic_values)

combined_parameters = list(itertools.product(*dic_values))

n = 0
for parameter_set in combined_parameters:

    run_config= {}
    for key, item in zip(list(config_data['config'].keys()), parameter_set):
        run_config[key] = item

    for key, item in config_data['paths'].items():
        run_config[key] = item

    print(run_config)
    # Seed: For reproducibility
    print('Seed: ', run_config['seed'])
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed: ', seed)

    # EVE: Normalization data
    eve_norm = np.load(run_config['eve_norm'])

    # Data augmentation
    # TODO: Incorporate augmentation or not, depending on branch that is being trained
    # if run_config['linear_architecture == 'linear' or run_config['model_architecture == 'complex':
    #     train_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })
    # else:
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=180, p=0.9, value=0, border_mode=1),
        ToTensorV2()], additional_targets={'y': 'image', })
    val_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })

    if (run_config['val_months'][0] in run_config['test_months']) is False and (run_config['val_months'][1] in run_config['test_months']) is False: 

        # Initialize data loader
        data_loader = IrradianceDataModule(run_config['aia_csv'], 
                                        run_config['eve_npy'], 
                                        run_config['aia_wl'], 
                                        num_workers=os.cpu_count() // 2,
                                        train_transforms=train_transforms, 
                                        val_transforms=val_transforms,
                                        val_months=run_config['val_months'], 
                                        test_months=run_config['test_months'],
                                        holdout_months=run_config['holdout_months'])
        
        data_loader = ZarrIrradianceDataModule(path_2_aia = '/mnt/sdomlv2_full/sdomlv2.zarr', 
                                        path_2_eve = '/mnt/sdomlv2_full/sdomlv2_eve.zarr',
                                        wavelengths = ['171A','304A','193A'],
                                        ions = ['Fe IX','Fe VIII'],
                                        frequency = '30min',
                                        batch_size = 32, 
                                        num_workers=None,
                                        train_transforms=None, 
                                        val_transforms=None, 
                                        val_months=[10,1], 
                                        test_months=[11,12], 
                                        holdout_months=None)
        
        data_loader.setup()



        # Initalize model
        model = HybridIrradianceModel(d_input=len(run_config['aia_wl']), 
                                    d_output=14, 
                                    eve_norm=eve_norm, 
                                    cnn_model=run_config['cnn_model'], 
                                    ln_model=run_config['ln_model'],
                                    cnn_dp=run_config['cnn_dp'],
                                    lr=run_config['lr'])

        # Initialize logger
        wandb_logger = WandbLogger(entity=config_data['wandb_init']['entity'],
                                project=config_data['wandb_init']['project'],                            
                                group=config_data['wandb_init']['group'],
                                job_type=config_data['wandb_init']['job_type'],
                                tags=config_data['wandb_init']['tags'],
                                name=f"{config_data['wandb_init']['name']}_{n}",
                                notes=config_data['wandb_init']['notes'],
                                config=run_config)                           

        # Plot callback
        total_n_valid = len(data_loader.valid_ds)
        plot_data = [data_loader.valid_ds[i] for i in range(0, total_n_valid, total_n_valid // 4)]
        plot_images = torch.stack([image for image, eve in plot_data])
        plot_eve = torch.stack([eve for image, eve in plot_data])
        eve_wl = np.load(run_config['eve_wl'], allow_pickle=True)
        image_callback = ImagePredictionLogger(plot_images, plot_eve, eve_wl, run_config['aia_wl'])

        # Checkpoint callback
        checkpoint_path = os.path.join(config_data['paths']['checkpoint_path'], str(seed))
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                                monitor='valid_loss',
                                                mode='min',
                                                save_top_k=1,
                                                filename=f"{config_data['paths']['checkpoint_file_name']}_{n}")

        # TODO: Make this more flexible (only linear, only cnn, both, etc.)
        if run_config['hybrid_loop']:

            # Lambda/Mode callback
            model.set_train_mode('linear')
            model.lr = run_config['ln_lr']
            switch_mode_callback = LambdaCallback(on_train_epoch_start=lambda trainer, pl_module: model.set_train_mode('cnn') if trainer.current_epoch > run_config['ln_epochs'] else None)

            # Initialize trainer
            trainer = Trainer(
                default_root_dir=checkpoint_path,
                accelerator="gpu",
                devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                max_epochs=(run_config['ln_epochs']+run_config['cnn_epochs']),
                callbacks=[image_callback, checkpoint_callback, switch_mode_callback],
                logger=wandb_logger,
                log_every_n_steps=10
                )

        else:

            # Initialize trainer
            trainer = Trainer(
                default_root_dir=checkpoint_path,
                accelerator="gpu",
                devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                max_epochs=run_config['epochs'],
                callbacks=[image_callback, checkpoint_callback],
                logger=wandb_logger,
                log_every_n_steps=10
                )

        # Train the model ⚡
        trainer.fit(model, data_loader)

        save_dictionary = run_config
        save_dictionary['model'] = model
        full_checkpoint_path = os.path.join(checkpoint_path, f"{config_data['paths']['checkpoint_file_name']}_{n}.ckpt")
        torch.save(save_dictionary, full_checkpoint_path)

        # Evaluate on test set
        # Load model from checkpoint
        # TODO: Correct: KeyError: 'pytorch-lightning_version'
        if config_data['paths']['checkpoint_file_name'] is not None:
            state = torch.load(full_checkpoint_path)
            model = state['model']
            trainer.test(model, data_loader)

        # Finalize logging
        wandb.finish()

        n = n + 1




