"""
Script to make some basic plots of EVE data
- plot showing the observed and predicted irradiance for each ion, as well as the residual
- plots showing normalized 2d histograms of the observed vs. predicted irradiance for each ion
- a grid of plots showing the residual irradiance for each ion
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from glob import glob

import matplotlib as mpl
mpl.use('Agg')

DATA_DIR = "/Users/wfawcett/Documents/FDL/2023/data/EVE"
PLOT_DIR = f"{DATA_DIR}/plots"


def main():

    # ion_df_dict = process_data()
    ion_df_dict = load_data()
    ion_list = list(ion_df_dict.keys())
    print(f"Available ions: {ion_list}")

    # Make plots for each ion separately
    make_single_plots = False   
    if make_single_plots:
        for ion in ion_list:
            if "Fe_XX" not in ion: # skip iron ions
                continue
            ion_df = ion_df_dict[ion]
            ion_name = ion.replace("_", " ")

            for plot_log in [True, False]:
                make_plot_with_subplot(
                    x=ion_df['timestamp'], 
                    y1=ion_df[f"real_{ion}"],
                    y2=ion_df[f"virt_{ion}"],
                    label1=f"Observed {ion_name}",
                    label2=f"Predicted {ion_name}",
                    subplot_y=ion_df['residual'],
                    subplot_label=None,
                    log_y=plot_log,
                    save_path=f"{PLOT_DIR}/{ion}_residual.png"
                    )

            for plot_log in [True, False]:
                if plot_log:
                    units = "log$_{10}$[Wm$^{-2}$]"
                else:
                    units = "[Wm$^{-2}$]"

                make_2d_hist(
                    x_hist=np.array(ion_df[f"real_{ion}"]),
                    y_hist=np.array(ion_df[f"virt_{ion}"]),
                    xlabel=f"Observed Irradiance {units}",
                    ylabel=f"Predicted Irradiance {units}",
                    # title=f"{ion} Observed vs. Predicted Irradiance",           
                    title=ion_name,
                    save_path=f"{PLOT_DIR}/{ion}_2d_hist.png", 
                    plot_log=plot_log
                )
            # break # just one ion


    # Select ions with good statistics
    for ion in ion_list:
        ion_df = ion_df_dict[ion]
        print(f"{ion}: {ion_df.shape[0]}")
        if ion_df.shape[0] < 1000:
            ion_list.remove(ion)
    print(f"There are {len(ion_list)} ions with >1000 measurements")

    plot_residuals_grid(ion_list, ion_df_dict)

    


def load_data():
    df_dict = {}
    files = glob(f"{DATA_DIR}/pandas/*_merged.parquet")
    for f in files:
        ion = f.replace("_merged.parquet", "").split("/")[-1]
        df_dict[ion] = pd.read_parquet(f)
    return df_dict


def process_data():

    real_eve_data = pd.read_parquet("/home/willfaw/data/EVE/evedata_6min.parquet")
    virtual_eve_data = pd.read_parquet("/home/willfaw/data/EVE/virtual_eve_6min.parquet")
    real_eve_data.rename(columns={"Time": "timestamp"}, inplace=True)

    real_cols = real_eve_data.columns.tolist()
    real_eve_data.rename(columns={c: c.replace(" ", "_") for c in real_cols}, inplace=True)
    real_eve_data['timestamp'] = pd.to_datetime(real_eve_data['timestamp'], utc=True)  # Convert to UTC

    # Remove second precision and timezone awareness from timestamps
    real_eve_data['timestamp'] = real_eve_data['timestamp'].dt.tz_convert(None).dt.floor('min')
    virtual_eve_data['timestamp'] = virtual_eve_data['timestamp'].dt.tz_convert(None).dt.floor('min')

    ion_list = virtual_eve_data.columns.tolist()
    ion_list.remove('timestamp')

    # Create a dictionary of pandas dataframes, one entry for each ion
    ion_df_dict = {}
    for ion in ion_list:
        ion_df_dict[ion] = get_merged_df_for_ion(real_eve_data, virtual_eve_data, ion)
        ion_df_dict[ion].to_parquet(f"/home/willfaw/data/EVE/pandas/{ion}_merged.parquet")
    
    return ion_df_dict


def make_2d_hist(x_hist, y_hist, xlabel, ylabel, title, save_path, plot_log=False):

    if plot_log:
        x = np.log(x_hist)
        y = np.log(y_hist)
    else:
        x = x_hist
        y = y_hist
    
    print("log:", plot_log)
    print(x.min(), y.min())
    print(x.max(), y.max())
    # hist, xedges, yedges, image = plt.hist2d(x, y, bins=100, cmap='Blues', density=True)

    # Normalize the histogram so that the maximum value is 1
    hist, xedges, yedges = np.histogram2d(x, y, bins=100)
    hist_normalized = hist / hist.max()
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Create a plot
    # plt.imshow(hist_normalized.T, origin='lower', cmap='Blues')
    plt.imshow(hist_normalized.T, extent=extent, cmap='Blues', origin='lower')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axline((xedges[0], yedges[0]), slope=1, color='red', linestyle='--', linewidth=1)


    # norm = mcolors.Normalize(vmin=0, vmax=1)
    cbar = plt.colorbar(label='Normalized Counts')
    # cbar.set_clim(0, 1)
    plt.tight_layout()

    if plot_log:
        save_path = save_path.replace(".png", "_log.png")
    print(f"saving 2h hist to {save_path} with log {plot_log}")
    plt.savefig(save_path, dpi=300)
    plt.close()


def make_plot_with_subplot(x, y1, y2, label1, label2, subplot_y, subplot_label, log_y, save_path):

    # Create the main plot (top) and ratio plot (bottom)
    fig = plt.figure(figsize=(9, 8))  # Adjust the figure size as needed
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])  # Create a 2x1 grid with a height ratio of 3:1

    # Add the main plot to the top grid
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, y1, label=label1, color='red')
    ax1.plot(x, y2, label=label2, color='blue')
    ax1.set_xlabel('Date')
    if log_y:
        ax1.set_ylabel('Irradience log$_{10}$[Wm$^{-2}$]')
        ax1.set_yscale('log')

    else:
        ax1.set_ylabel('Irradience [Wm$^{-2}$]')
    ax1.legend()

    # Add the ratio plot to the bottom grid
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Share the same x-axis with the main plot
    ax2.plot(x, subplot_y, label=subplot_label, color='black')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('(Obs $-$ Pred)/Obs')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Add a horizontal line at y=1
    if subplot_label is not None:
        ax2.legend()

    # Adjust the spacing between the subplots
    plt.tight_layout()

    if log_y:
        save_path = save_path.replace(".png", "_log.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saving plot to {save_path}")
    # Show the combined plot
    plt.show()
    plt.close()


def get_merged_df_for_ion(real_eve_data, virtual_eve_data, ion):

    print(f"Processing {ion}")
    real_ion_df = real_eve_data[['timestamp', ion]]
    real_ion_df = real_ion_df[(real_ion_df[ion] >= 0)]
    real_ion_df.rename(columns={ion: f"real_{ion}"}, inplace=True)
    print(real_ion_df.shape)

    virt_ion_df = virtual_eve_data[['timestamp', ion]]
    virt_ion_df = virt_ion_df[(virt_ion_df[ion] >= 0)]
    virt_ion_df.rename(columns={ion: f"virt_{ion}"}, inplace=True)
    print(virt_ion_df.shape)

    # Merge the dataframes
    merged_df = pd.merge(real_ion_df, virt_ion_df, on='timestamp', how='inner')
    print(merged_df.shape)

    merged_df['ratio'] = merged_df[f"real_{ion}"] / merged_df[f"virt_{ion}"]
    merged_df['residual'] = (merged_df[f"real_{ion}"] - merged_df[f"virt_{ion}"]) / merged_df[f"real_{ion}"]
    # print(merged_df.head())
    print('-'*50)
    return merged_df


def plot_residuals_grid(ion_list, ion_df_dict):
    # Plot the residuals for each ion
    fig, axs = plt.subplots(6, 4, figsize=(20, 20))
    axs = axs.ravel()
    for i, ion in enumerate(ion_list):
        ion_df = ion_df_dict[ion]
        axs[i].plot(ion_df['timestamp'], ion_df['residual'])
        axs[i].set_title(ion.replace("_", " "))
        axs[i].set_ylim([-1, 1])

    # adjust x-axis just to show year tick marks

    for ax in axs:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))



    plt.tight_layout()
    print(f"Saving grid to {PLOT_DIR}/residuals_grid.png")
    plt.savefig(f"{PLOT_DIR}/residuals_grid.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()