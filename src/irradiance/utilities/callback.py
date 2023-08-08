import wandb
from pytorch_lightning import Callback
import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps

from src.irradiance.models.model import unnormalize

# Custom Callback
class ImagePredictionLogger(Callback):
    def __init__(self, val_imgs, val_eve, names, aia_wavelengths):
        super().__init__()
        self.val_imgs, self.val_eve = val_imgs, val_eve
        self.names = names
        self.aia_wavelengths = aia_wavelengths

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        # Get model prediction
        # pred_eve = pl_module.forward(val_imgs).cpu().numpy()
        pred_eve = pl_module.forward_unnormalize(val_imgs).cpu().numpy()
        val_eve = unnormalize(self.val_eve, pl_module.eve_norm).numpy()
        val_imgs = val_imgs.cpu().numpy()

        # create matplotlib figure
        fig = self.plot_aia_eve(val_imgs, val_eve, pred_eve)
        # Log the images to wandb
        trainer.logger.experiment.log({"AIA Images and EVE bar plots": wandb.Image(fig)})
        plt.close(fig)

    def plot_aia_eve(self, val_imgs, val_eve, pred_eve):
        """
        Function to plot a 4 channel AIA stack and the EVE barplots

        Arguments:
        ----------
            val_imgs: numpy array
                Stack with 4 image channels
            val_eve: numpy array
                Stack of ground-truth eve channels
            pred_eve: numpy array
                Stack of predicted eve channels
        Returns:
        --------
            fig: matplotlib figure
                figure with plots
        """
        samples = pred_eve.shape[0]
        n_aia_wavelengths = len(self.aia_wavelengths)

        if n_aia_wavelengths < 3:
            nrows = 1
            ncols = n_aia_wavelengths
            fig = plt.figure(figsize=( 9+9/4*n_aia_wavelengths, 3*samples), dpi=150)
            gs = fig.add_gridspec(samples, n_aia_wavelengths+3, wspace=0.4, hspace=0.25)
        elif n_aia_wavelengths < 5:
            nrows = 2
            ncols = 2
            fig = plt.figure(figsize=( 9+9/4*2, 6*samples), dpi=150)
            gs = fig.add_gridspec(2*samples, 5, wspace=0.4, hspace=0.25)
        elif n_aia_wavelengths < 7:
            nrows = 2
            ncols = 3
            fig = plt.figure(figsize=( 9+9/4*3, 6*samples), dpi=150)
            gs = fig.add_gridspec(2*samples, 6, wspace=0.4, hspace=0.25)
        else:
            nrows = 2
            ncols = 4
            fig = plt.figure(figsize=( 18, 6*samples), dpi=150)
            gs = fig.add_gridspec(2*samples, 7, wspace=0.4, hspace=0.25)
        
        cmaps_all = [f"sdoaia{wavelength.split('A')[0]}" for wavelength in self.aia_wavelengths]
        
        cmaps = [cmaps_all[i] for i, _ in enumerate(self.aia_wavelengths)]
        n_plots = 0

        for s in range(samples):
            for i in range(nrows):
                for j in range(ncols):
                    if n_plots < n_aia_wavelengths: 
                        ax = fig.add_subplot(gs[s*nrows+i, j])
                        ax.imshow(val_imgs[s, i*ncols+j], cmap = plt.get_cmap(cmaps[i*ncols+j]), vmin = 0, vmax = 1)
                        ax.text(0.01, 0.99, cmaps[i*ncols+j], horizontalalignment='left', verticalalignment='top', color = 'w', transform=ax.transAxes)
                        ax.set_axis_off()
                        n_plots += 1
            n_plots = 0
            #eve data
            ax5 = fig.add_subplot(gs[s*nrows, ncols:])
            ax5.bar(np.arange(0,len(val_eve[s,:])), val_eve[s,:], label='ground truth')
            ax5.bar(np.arange(0,len(pred_eve[s,:])), pred_eve[s,:], width = 0.5, label='prediction', alpha=0.5)
            ax5.set_xticks(np.arange(0,len(val_eve[s,:])))
            ax5.set_xticklabels(self.names,rotation = 45)
            ax5.set_yscale('log')
            ax5.legend()

        # fig.tight_layout()
        return fig
    

class ImagePredictionLoggerHMI(Callback):

    def __init__(self, val_imgs, val_eve, ions, channels, path_HMI, path_AIA):
        super().__init__()
                    
        self.val_eve = val_eve
        self.names = ions

        self.val_imgs, self.val_eve = val_imgs, val_eve
        
        # either HMI + AIA or any of them individually
        self.channels = channels

        self.check_HMI = True if path_HMI is not None else False
        self.check_AIA = True if path_AIA is not None else False

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        # Get model prediction
        # pred_eve = pl_module.forward(val_imgs).cpu().numpy()
        pred_eve = pl_module.forward_unnormalize(val_imgs).cpu().numpy()
        val_eve = unnormalize(self.val_eve, pl_module.eve_norm).numpy()
        val_imgs = val_imgs.cpu().numpy()

        # create matplotlib figure
        fig = self.plot_channel_eve(val_imgs, val_eve, pred_eve)
        # Log the images to wandb
        trainer.logger.experiment.log({"AIA Images and EVE bar plots": wandb.Image(fig)})
        plt.close(fig)

    def plot_channel_eve(self, val_imgs, val_eve, pred_eve):
        """
        Function to plot a 4 channel AIA stack and the EVE barplots

        Arguments:
        ----------
            val_imgs: numpy array
                Stack with 4 image channels
            val_eve: numpy array
                Stack of ground-truth eve channels
            pred_eve: numpy array
                Stack of predicted eve channels
        Returns:
        --------
            fig: matplotlib figure
                figure with plots
        """
        cmap_dict = {
            'By': 'hmimag',
            'Bz': 'hmimag',
            'Bx': 'hmimag',
            "131A": 'sdoaia131',
            "1600A": 'sdoaia1600',
            "1700A": 'sdoaia1700',
            "171A": 'sdoaia171',
            "193A": 'sdoaia193',
            "211A": 'sdoaia211',
            "304A": 'sdoaia304',
            "335A": 'sdoaia335',
            "94A": 'sdoaia94'}
        
        if self.check_HMI and self.check_AIA:

            fig = plt.figure(figsize=(10,14), dpi=150)
            
            nrows = 4
            ncols = 4

            # Bx
            plt.subplot(nrows,ncols,1)
            plt.imshow(val_imgs[0,0], cmap = plt.get_cmap(cmap_dict['Bx']), vmin = 0, vmax = 1)
            plt.title('HMI - Bx')

            # By
            plt.subplot(nrows,ncols,2)
            plt.imshow(val_imgs[0,1], cmap = plt.get_cmap(cmap_dict['By']), vmin = 0, vmax = 1)
            plt.title('HMI - By')

            # Bz
            plt.subplot(nrows,ncols,3)
            plt.imshow(val_imgs[0,2], cmap = plt.get_cmap(cmap_dict['Bz']), vmin = 0, vmax = 1)
            plt.title('HMI - Bz')

            # 131A
            plt.subplot(nrows,ncols,4)
            plt.imshow(val_imgs[0,3], cmap = plt.get_cmap(cmap_dict['131A']), vmin = 0, vmax = 1)
            plt.title('AIA - 131 Å')

            # 1600A
            plt.subplot(nrows,ncols,5)
            plt.imshow(val_imgs[0,4], cmap = plt.get_cmap(cmap_dict['1600A']), vmin = 0, vmax = 1)
            plt.title('AIA - 1600 Å')

            # 1700A
            plt.subplot(nrows,ncols,6)
            plt.imshow(val_imgs[0,5], cmap = plt.get_cmap(cmap_dict['1700A']), vmin = 0, vmax = 1)
            plt.title('AIA - 1700 Å')

            # 171A
            plt.subplot(nrows,ncols,7)
            plt.imshow(val_imgs[0,6], cmap = plt.get_cmap(cmap_dict['171A']), vmin = 0, vmax = 1)
            plt.title('AIA - 171 Å')

            # 193A
            plt.subplot(nrows,ncols,8)
            plt.imshow(val_imgs[0,7], cmap = plt.get_cmap(cmap_dict['193A']), vmin = 0, vmax = 1)
            plt.title('AIA - 193 Å')

            # 211A
            plt.subplot(nrows,ncols,9)
            plt.imshow(val_imgs[0,8], cmap = plt.get_cmap(cmap_dict['211A']), vmin = 0, vmax = 1)
            plt.title('AIA - 211 Å')

            # 304A
            plt.subplot(nrows,ncols,10)
            plt.imshow(val_imgs[0,9], cmap = plt.get_cmap(cmap_dict['304A']), vmin = 0, vmax = 1)
            plt.title('AIA - 304 Å')

            # 335A
            plt.subplot(nrows,ncols,11)
            plt.imshow(val_imgs[0,10], cmap = plt.get_cmap(cmap_dict['335A']), vmin = 0, vmax = 1)
            plt.title('AIA - 335 Å')

            # 94A
            plt.subplot(nrows,ncols,12)
            plt.imshow(val_imgs[0,11], cmap = plt.get_cmap(cmap_dict['94A']), vmin = 0, vmax = 1)
            plt.title('AIA - 94 Å')

            # EVE
            plt.subplot(nrows,1,nrows)
            plt.bar(np.arange(0,val_eve.shape[1]), val_eve[0,:], label='ground truth')
            plt.bar(np.arange(0,val_eve.shape[1]), pred_eve[0,:], width = 0.5, label='prediction', alpha=0.5)
            plt.xticks(np.arange(0,val_eve.shape[1]),["C III", "Fe IX", "Fe VIII", "Fe X", "Fe XI", "Fe XII", "Fe XIII", "Fe XIV", "Fe XIX", "Fe XV", "Fe XVI", "Fe XVIII", "Fe XX", "Fe XX_2", "Fe XX_3", "H I", "H I_2", "H I_3", "He I", "He II", "He II_2", "He I_2", "Mg IX", "Mg X", "Mg X_2", "Ne VII", "Ne VIII", "O II", "O III", "O III_2", "O II_2", "O IV", "O IV_2", "O V", "O VI", "S XIV", "Si XII", "Si XII_2"],rotation = 90)
            plt.yscale('log')

        elif not self.check_HMI and self.check_AIA:

            fig = plt.figure(figsize=(10,14), dpi=150)
            
            nrows = 4
            ncols = 3

            # 131A
            plt.subplot(nrows,ncols,1)
            plt.imshow(val_imgs[0,0], cmap = plt.get_cmap(cmap_dict['131A']), vmin = 0, vmax = 1)
            plt.title('AIA - 131 Å')

            # 1600A
            plt.subplot(nrows,ncols,2)
            plt.imshow(val_imgs[0,1], cmap = plt.get_cmap(cmap_dict['1600A']), vmin = 0, vmax = 1)
            plt.title('AIA - 1600 Å')

            # 1700A
            plt.subplot(nrows,ncols,3)
            plt.imshow(val_imgs[0,2], cmap = plt.get_cmap(cmap_dict['1700A']), vmin = 0, vmax = 1)
            plt.title('AIA - 1700 Å')

            # 171A
            plt.subplot(nrows,ncols,4)
            plt.imshow(val_imgs[0,3], cmap = plt.get_cmap(cmap_dict['171A']), vmin = 0, vmax = 1)
            plt.title('AIA - 171 Å')

            # 193A
            plt.subplot(nrows,ncols,5)
            plt.imshow(val_imgs[0,4], cmap = plt.get_cmap(cmap_dict['193A']), vmin = 0, vmax = 1)
            plt.title('AIA - 193 Å')

            # 211A
            plt.subplot(nrows,ncols,6)
            plt.imshow(val_imgs[0,5], cmap = plt.get_cmap(cmap_dict['211A']), vmin = 0, vmax = 1)
            plt.title('AIA - 211 Å')

            # 304A
            plt.subplot(nrows,ncols,7)
            plt.imshow(val_imgs[0,6], cmap = plt.get_cmap(cmap_dict['304A']), vmin = 0, vmax = 1)
            plt.title('AIA - 304 Å')

            # 335A
            plt.subplot(nrows,ncols,8)
            plt.imshow(val_imgs[0,7], cmap = plt.get_cmap(cmap_dict['335A']), vmin = 0, vmax = 1)
            plt.title('AIA - 335 Å')

            # 94A
            plt.subplot(nrows,ncols,9)
            plt.imshow(val_imgs[0,8], cmap = plt.get_cmap(cmap_dict['94A']), vmin = 0, vmax = 1)
            plt.title('AIA - 94 Å')

            # EVE
            plt.subplot(nrows,1,nrows)
            plt.bar(np.arange(0,val_eve.shape[1]), val_eve[0,:], label='ground truth')
            plt.bar(np.arange(0,val_eve.shape[1]), pred_eve[0,:], width = 0.5, label='prediction', alpha=0.5)
            plt.xticks(np.arange(0,val_eve.shape[1]),["C III", "Fe IX", "Fe VIII", "Fe X", "Fe XI", "Fe XII", "Fe XIII", "Fe XIV", "Fe XIX", "Fe XV", "Fe XVI", "Fe XVIII", "Fe XX", "Fe XX_2", "Fe XX_3", "H I", "H I_2", "H I_3", "He I", "He II", "He II_2", "He I_2", "Mg IX", "Mg X", "Mg X_2", "Ne VII", "Ne VIII", "O II", "O III", "O III_2", "O II_2", "O IV", "O IV_2", "O V", "O VI", "S XIV", "Si XII", "Si XII_2"],rotation = 90)
            plt.yscale('log') 

        else:

            fig = plt.figure(figsize=(10,14), dpi=150)
            
            nrows = 2
            ncols = 3

            # Bx
            plt.subplot(nrows,ncols,1)
            plt.imshow(val_imgs[0,0], cmap = plt.get_cmap(cmap_dict['Bx']), vmin = 0, vmax = 1)
            plt.title('HMI - Bx')

            # By
            plt.subplot(nrows,ncols,2)
            plt.imshow(val_imgs[0,1], cmap = plt.get_cmap(cmap_dict['By']), vmin = 0, vmax = 1)
            plt.title('HMI - By')

            # Bz
            plt.subplot(nrows,ncols,3)
            plt.imshow(val_imgs[0,2], cmap = plt.get_cmap(cmap_dict['Bz']), vmin = 0, vmax = 1)
            plt.title('HMI - Bz')

            # EVE
            plt.subplot(nrows,1,nrows)
            plt.bar(np.arange(0,val_eve.shape[1]), val_eve[0,:], label='ground truth')
            plt.bar(np.arange(0,val_eve.shape[1]), pred_eve[0,:], width = 0.5, label='prediction', alpha=0.5)
            plt.xticks(np.arange(0,val_eve.shape[1]),["C III", "Fe IX", "Fe VIII", "Fe X", "Fe XI", "Fe XII", "Fe XIII", "Fe XIV", "Fe XIX", "Fe XV", "Fe XVI", "Fe XVIII", "Fe XX", "Fe XX_2", "Fe XX_3", "H I", "H I_2", "H I_3", "He I", "He II", "He II_2", "He I_2", "Mg IX", "Mg X", "Mg X_2", "Ne VII", "Ne VIII", "O II", "O III", "O III_2", "O II_2", "O IV", "O IV_2", "O V", "O VI", "S XIV", "Si XII", "Si XII_2"],rotation = 90)
            plt.yscale('log')          
        
        #plt.tight_layout()
        
        return fig
