import os
import sys
import time
import cv2
import traceback
import pandas as pd
import numpy as np
from scipy import interpolate
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects

from shared.shared_functions import round_up_to_odd, read_images, load_all_budj_data, get_whole_cell_mask, load_events, create_excel_dir, get_nuc_and_cyt_gfp_av_signal
from shared.signal_analysis import generate_plots  # import file that allows for generating plots

#######################
### IMPORTANT PATHS ###
#######################
data_dir = "C:/Users/Olle de Jong/Documents/MSc Biology/MSB Research/Adriana/data/"
tiff_files_dir = f"{data_dir}Processed_Tiffs/"  # relative path from data directory to tiff directory
output_dir = f"{data_dir}output/"  # relative path from data directory to image output directory
budding_data_path = f"{data_dir}buddings.txt"  # budding events
kario_data_path = f"{data_dir}kariokinesis.txt"  # kariokinesis events

###############
### GLOBALS ###
###############
bloc_size_frac_to_use = 0.095  # fraction of the total cell mask pixels that is used as local thresholding block size
offset_to_use = -15  # offset for local thresholding
pd.options.mode.chained_assignment = None  # default='warn'

#################
### FUNCTIONS ###
#################
def get_nuc_thresh_mask(imageGFP_at_frame, whole_cell_mask):
    imageGFP_cell_mask = imageGFP_at_frame * whole_cell_mask  # keep only data within the whole cell mask
    num_cell_pixels = np.count_nonzero(whole_cell_mask == True)  # count number of pixels in that mask
    bloc_size_cell_size_dependent = round_up_to_odd(bloc_size_frac_to_use * num_cell_pixels)
    nucl_thresh_mask_local = threshold_local(
        image=imageGFP_cell_mask,
        block_size=bloc_size_cell_size_dependent,
        offset=offset_to_use
    )
    imageGFP_nuc_mask_local = remove_small_objects(imageGFP_cell_mask > nucl_thresh_mask_local)
    return imageGFP_nuc_mask_local


def get_data_for_single_cell(image_gfp, single_cell_data):
    """
    This function, for the given cell, calculates the GFP values for every time-frame.
    :param image_gfp:
    :param single_cell_data:
    :return:
    """
    GFP_total, GFP_cyto, GFP_nucleus, GFP_ratio, GFP_std = [], [], [], [], []  # GFP lists
    for t in single_cell_data.TimeID:
        t_in_tiff = t - 1  # skew the time by one for tiff dataframe
        imageGFP_at_frame = image_gfp[t_in_tiff, :, :]  # get the GFP data
        nrows, ncols = imageGFP_at_frame.shape

        # WHOLE CELL #
        # get whole-cell mask based on budj data
        whole_cell_mask, x_pos, y_pos = get_whole_cell_mask(t, single_cell_data, imageGFP_at_frame.shape)

        # get mean GFP signal and the standard deviation
        mean_GFP = np.mean(imageGFP_at_frame[whole_cell_mask == True])
        if mean_GFP < 5:
            mean_GFP = np.nan
        GFP_total.append(mean_GFP)
        GFP_std.append(np.std(imageGFP_at_frame[whole_cell_mask == True]))

        # NUCLEUS #
        imageGFP_nuc_mask_local = get_nuc_thresh_mask(imageGFP_at_frame, whole_cell_mask)

        # NUCLEAR / CYTO RFP CALCULATIONS #
        cyto_mean, nucleus_mean = get_nuc_and_cyt_gfp_av_signal(imageGFP_at_frame, imageGFP_nuc_mask_local, ncols,
                                                                nrows, whole_cell_mask)
        GFP_nucleus.append(nucleus_mean)
        GFP_cyto.append(cyto_mean)
        GFP_ratio.append(nucleus_mean / cyto_mean)

    return GFP_total, GFP_std, GFP_cyto, GFP_nucleus, GFP_ratio


def get_data_for_all_cells():
    """
    The most complex, and by far most compute intensive, function of this script. It goes through all tiff movies and
    all cells, and for each cell it loops over all frames in the movie. For every frame, local thresholding is
    performed. Based on that, the GFP datapoints are generated.
    :return:
    """
    budj_data = load_all_budj_data(data_dir)
    tiff_images = read_images(tiff_files_dir)

    individual_cells = sorted([x for x in budj_data["Cell_pos"].unique()])

    nth_cell = 0
    final_data = pd.DataFrame({})

    # for every cell that was tracked using BudJ, we generate the data
    for cell in individual_cells:
        print(f"Generating data ( {round(nth_cell / len(individual_cells) * 100)}% )", end="\r", flush=True)

        # get right image from images dictionary and select the right channel (GFP)
        image = tiff_images[cell[3:5]]
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)  # enhance contrast
        image_gfp = image[:, :, :, 1]

        # get this cell it's budj data
        cell_data = budj_data[budj_data["Cell_pos"] == cell]

        # get the GFP datapoints for every frame that this cell was tracked
        GFP_total, GFP_std, GFP_cyto, GFP_nucleus, GFP_ratio = get_data_for_single_cell(image_gfp, cell_data)

        # store data in this cell its dataframe
        cell_data.loc[:, "GFP_total"] = pd.Series(GFP_total, index=cell_data.index)
        cell_data.loc[:, "GFP_std"] = pd.Series(GFP_std, index=cell_data.index)
        cell_data.loc[:, "GFP_nucleus"] = pd.Series(GFP_nucleus, index=cell_data.index)
        cell_data.loc[:, "GFP_cyto"] = pd.Series(GFP_cyto, index=cell_data.index)
        cell_data.loc[:, "GFP_ratio"] = pd.Series(GFP_ratio, index=cell_data.index)

        # store this cell's data in the final dataframe which eventually contains data of all cells
        final_data = pd.concat([final_data, cell_data])
        nth_cell += 1

    # reset the index and save the final dataframe
    final_data = final_data.reset_index(drop=True)
    final_data.to_excel(f"{output_dir}excel/sfp1_GFP_only_data.xlsx")

    print("Generating data.. Done!")
    return final_data


def split_cycles_and_interpolate(final_data, kario_events):
    """
    Function responsible splitting the data of each cell on karyokinesis events, where after the data per
    cycle is interpolated to 100 datapoints.
    :param final_data:
    :return interpolated_dataframes: a list with the interpolated dataframes
    """
    print("Performing interpolation on data..", end="\r", flush=True)

    individual_cells = sorted(list(set(final_data["Cell_pos"])))
    cycle_durations = []

    min_dp_for_int = 10
    desired_datapoints = 100
    cols = ["cell", "cycle", "start", "end"]
    cols += range(desired_datapoints)

    interpolated_dataframes = {}
    tot_under = 0
    for data_type in ["GFP_total", "GFP_nucleus", "GFP_cyto", "GFP_ratio"]:
        data_interpolated = pd.DataFrame(columns=cols)

        for cell in individual_cells:
            single_cell_data = final_data[final_data.Cell_pos == cell]
            single_cell_data = single_cell_data[~single_cell_data.GFP_nucleus.isnull()]

            count = 0
            while count < len(kario_events[cell]) - 1:
                tp1, tp2 = kario_events[cell][count], kario_events[cell][count + 1]
                cycle_durations.append(tp2 - tp1)
                cell_cycle_dat = single_cell_data[single_cell_data.TimeID.between(tp1, tp2)]

                # when there are not enough datapoints (because of outlier removal) for this cycle, ignore it
                if len(cell_cycle_dat) < min_dp_for_int:
                    tot_under += 1
                    count += 1
                    continue

                # get the original x and y data
                x = np.linspace(0, len(cell_cycle_dat), len(cell_cycle_dat))
                y = cell_cycle_dat[data_type]

                # interpolate the data towards 100 datapoints
                x_new = np.linspace(0, len(cell_cycle_dat), desired_datapoints)
                f2 = interpolate.interp1d(x, y, kind='linear')

                # add the interpolated data to a dataframe
                data_interpolated.loc[len(data_interpolated)] = [cell, count + 1, tp1, tp2] + f2(x_new).tolist()
                count += 1

        data_interpolated.to_excel(f"{output_dir}excel/cycles_interpolated_{data_type}s.xlsx")
        interpolated_dataframes[data_type] = data_interpolated

    av_cycle_duration = np.average(cycle_durations)
    print("Performing interpolation on data.. Done!")
    print(f"There were {tot_under} cycles removed that had not enough datapoints (less than {min_dp_for_int}).")
    print(f"Average cycle duration: {av_cycle_duration} frames, which is equal to {av_cycle_duration * 5} minutes.")
    return interpolated_dataframes


def main():
    print("Getting started!")
    tic = time.perf_counter()  # start counter
    create_excel_dir(data_dir)

    # if dataset already exists, prevent generating this again and load it
    final_dataframe_path = f"{output_dir}excel/sfp1_GFP_only_data.xlsx"
    if not os.path.exists(final_dataframe_path):
        final_data = get_data_for_all_cells()
    else:
        print("Data has been generated already. The output file exists.")
        final_data = pd.read_excel(final_dataframe_path)

    # load event data
    kario_events, budding_events = load_events(kario_data_path, budding_data_path)

    # check if cycles have been split and interpolated, if not, do this
    interpolated_dataframes = split_cycles_and_interpolate(final_data, kario_events)

    # generate a big overview image that shows gfp intensities
    generate_plots.dynamics_overview_all_cells(final_data, kario_events, budding_events, output_dir, "GFP")

    # generate a plot per interpolated cycle
    generate_plots.separate_interpolated_cycles(interpolated_dataframes, output_dir, "GFP")

    generate_plots.averaged_plots(interpolated_dataframes, output_dir, "GFP")

    toc = time.perf_counter()
    secs = round(toc - tic, 4)
    print(f"Done. Runtime was {secs} seconds ({round(secs / 60, 2)} minutes)")


# SCRIPT STARTS HERE
if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception:
        print("\nSomething went wrong while running the model:\n", traceback.format_exc())
        sys.exit(1)

