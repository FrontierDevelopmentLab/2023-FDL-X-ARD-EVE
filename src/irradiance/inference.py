import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.irradiance.models.model import HybridIrradianceModel
from src.irradiance.utilities.data_loader import ZarrIrradianceDataset

def ipredict(model, dataset, return_images=False, batch_size=2, num_workers = None):
    """Predict irradiance for a given set of npy image stacks using a generator.

    Parameters
    ----------
    chk_path: model save point.
    dataset: pytorch dataset for streaming the input data.
    return_images: set True to return input images.
    batch_size: number of samples to process in parallel.
    num_workers: number of workers for data preprocessing (default cpu_count / 2).

    Returns
    -------
    predicted irradiance as numpy array and corresponding image if return_images==True
    """
    # use model after training or load weights and drop into the production system
    model.eval()
    # load data
    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(model.device)
            pred_irradiance = model.forward_unnormalize(imgs)
            for pred, img in zip(pred_irradiance, imgs):
                if return_images:
                    yield pred.cpu(), img.cpu()
                else:
                    yield pred.cpu()

def ipredict_uncertainty(model, dataset, return_images=False, forward_passes=100, batch_size=2, num_workers = None):
    """Predict irradiance and uncertainty for a given set of npy image stacks using a generator.

    Parameters
    ----------
    chk_path: model save point.
    dataset: pytorch dataset for streaming the input data.
    return_images: set True to return input images.
    batch_size: number of samples to process in parallel.
    num_workers: number of workers for data preprocessing (default cpu_count / 2).

    Returns
    -------
    predicted irradiance as numpy array and corresponding image if return_images==True
    """
    # load data
    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(model.device)
            dropout_predictions = []
            for _ in range(forward_passes):
                model.eval()
                enable_dropout(model)
                pred_irradiance = model.forward_unnormalize(imgs)
                dropout_predictions += [pred_irradiance.cpu()]

            dropout_predictions = torch.stack(dropout_predictions).numpy() # shape (forward_passes, n_samples, n_classes)

            mean_batch = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

            # Calculating std across multiple MCD forward passes
            std_batch = np.std(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

            epsilon = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes
            entropy = -np.sum(mean_batch * np.log(mean_batch + epsilon), axis=-1)  # shape (n_samples,)

            # Calculating mutual information across multiple MCD forward passes
            # TODO check mutual information can be used for regression here
            mutual_info_batch = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                                   axis=-1), axis=0)  # shape (n_samples,)


            # iterate through batch
            for img, mean, std, mutual_info in zip(imgs, mean_batch, std_batch, mutual_info_batch):
                if return_images:
                    yield (mean, std, mutual_info), img.cpu().numpy()
                else:
                    yield (mean, std, mutual_info)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint_filepath', type=str,
                   default='/mnt/model_checkpoints/best_model.ckpt',
                   help='path to the model checkpoint.')
    p.add_argument('-checkpoint_file', type=str, 
                   default='best_model.ckpt', 
                   help='Filename to load checkpoint.')
    p.add_argument('-aia_files', type=str,
                   default=None,
                   help='path to the EVE normalization.')
    args = p.parse_args()

    # Initalize model
    full_chkpt_path = os.path.join(args.checkpoint_path, args.checkpoint_file)
    state = torch.load(args["checkpoint_filepath"])
    model = state["model"]
    aia_wavelengths = state["aia_wavelengths"]




