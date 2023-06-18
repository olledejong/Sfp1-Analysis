import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.ndimage import center_of_mass, binary_erosion
from skimage.io import imread
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects

# import the shared needed functions
from shared.shared_functions import round_up_to_odd, read_images, load_all_budj_data, get_whole_cell_mask, load_events, create_excel_dir
from shared.signal_analysis import generate_plots  # import file that allows for generating plots

#######################
### IMPORTANT PATHS ###
#######################
data_dir = "C:/Users/Olle de Jong/Documents/MSc Biology/MSB Research/Adriana/scarlet_data/"
tiff_files_dir = f"{data_dir}Processed_Tiffs/"  # relative path from data directory to tiff directory
output_dir = f"{data_dir}output/"  # relative path from data directory to image output directory
budding_data_path = f"{data_dir}buddings.txt"  # budding events
kario_data_path = f"{data_dir}cytokinesis.txt"  # kariokinesis events

###############
### GLOBALS ###
###############
bloc_size_frac_to_use = 0.09  # fraction of the total cell mask pixels that is used as local thresholding block size
offset_to_use = -10  # offset for local thresholding
pd.options.mode.chained_assignment = None  # default='warn'

#################
### FUNCTIONS ###
#################
def get_nuc_and_cyt_gfp_av_signal(imageRFP_at_frame, imageRFP_nuc_mask_local, ncols, nrows, whole_cell_mask):
    # get the centroid of the nuclear mask when there is one
    a, b = np.nan_to_num(center_of_mass(imageRFP_nuc_mask_local))
    r, r1 = 3, 9
    x1, y1 = np.ogrid[-a: nrows - a, -b: ncols - b]
    disk_mask_nuc = x1 * x1 + y1 * y1 < r * r
    disk_mask_cyto = x1 * x1 + y1 * y1 < r1 * r1

    # get RFP signal mean in the nucleus
    nucleus_mean = np.mean(imageRFP_at_frame[disk_mask_nuc == True])
    if nucleus_mean < 5:
        nucleus_mean = np.nan

    # RFP in the cytoplasm
    diff = np.logical_and(disk_mask_cyto, whole_cell_mask)
    mask_of_cytoplasm = whole_cell_mask ^ diff
    cyto_mean = np.mean(imageRFP_at_frame[mask_of_cytoplasm == True])
    if cyto_mean < 5:
        cyto_mean = np.nan

    return cyto_mean, nucleus_mean


def get_nuc_thresh_mask(imageRFP_at_frame, whole_cell_mask):
    imageRFP_cell_mask = imageRFP_at_frame * whole_cell_mask  # keep only data within the whole cell mask
    num_cell_pixels = np.count_nonzero(whole_cell_mask == True)  # count number of pixels in that mask
    bloc_size_cell_size_dependent = round_up_to_odd(bloc_size_frac_to_use * num_cell_pixels)
    nucl_thresh_mask_local = threshold_local(
        image=imageRFP_cell_mask,
        block_size=bloc_size_cell_size_dependent,
        offset=offset_to_use
    )
    imageRFP_nuc_mask_local = remove_small_objects(imageRFP_cell_mask > nucl_thresh_mask_local)
    imageRFP_nuc_mask_local = binary_erosion(imageRFP_nuc_mask_local, structure=np.ones((3, 3)))
    return imageRFP_nuc_mask_local


def get_data_for_single_cell(image_gfp, single_cell_data):
    """
    This function, for the given cell, calculates the RFP values for every time-frame.
    :param image_gfp:
    :param single_cell_data:
    :return:
    """
    RFP_total, RFP_cyto, RFP_nucleus, RFP_ratio, RFP_std = [], [], [], [], []  # RFP lists
    for t in single_cell_data.TimeID:
        t_in_tiff = t - 1  # skew the time by one for tiff dataframe
        imageRFP_at_frame = image_gfp[t_in_tiff, :, :]  # get the RFP data
        imageRFP_at_frame = cv2.GaussianBlur(imageRFP_at_frame, (3, 3), cv2.BORDER_DEFAULT)
        nrows, ncols = imageRFP_at_frame.shape

        # WHOLE CELL #
        # get whole-cell mask based on budj data
        whole_cell_mask, x_pos, y_pos = get_whole_cell_mask(t, single_cell_data, imageRFP_at_frame.shape)

        # get mean RFP signal and the standard deviation
        mean_RFP = np.mean(imageRFP_at_frame[whole_cell_mask == True])
        if mean_RFP < 5:
            mean_RFP = np.nan
        RFP_total.append(mean_RFP)
        RFP_std.append(np.std(imageRFP_at_frame[whole_cell_mask == True]))

        imageRFP_nuc_mask_local = get_nuc_thresh_mask(imageRFP_at_frame, whole_cell_mask)

        # NUCLEAR / CYTO RFP CALCULATIONS #
        cyto_mean, nucleus_mean = get_nuc_and_cyt_gfp_av_signal(imageRFP_at_frame, imageRFP_nuc_mask_local, ncols,
                                                                nrows, whole_cell_mask)
        RFP_nucleus.append(nucleus_mean)
        RFP_cyto.append(cyto_mean)
        RFP_ratio.append(nucleus_mean / cyto_mean)

    return RFP_total, RFP_std, RFP_cyto, RFP_nucleus, RFP_ratio


def get_data_for_all_cells():
    """
    The most complex, and by far most compute intensive, function of this script. It goes through all tiff movies and
    all cells, and for each cell it loops over all frames in the movie. For every frame, local thresholding is
    performed. Based on that, the RFP datapoints are generated.
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

        # get right image from images dictionary and select the right channel (RFP)
        image = tiff_images[cell[3:5]]
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)  # enhance contrast
        image_gfp = image[:, :, :, 2]

        # get this cell it's budj data
        cell_data = budj_data[budj_data["Cell_pos"] == cell]

        # get the RFP datapoints for every frame that this cell was tracked
        RFP_total, RFP_std, RFP_cyto, RFP_nucleus, RFP_ratio = get_data_for_single_cell(image_gfp, cell_data)

        # store data in this cell its dataframe
        cell_data.loc[:, "RFP_total"] = pd.Series(RFP_total, index=cell_data.index)
        cell_data.loc[:, "RFP_std"] = pd.Series(RFP_std, index=cell_data.index)
        cell_data.loc[:, "RFP_nucleus"] = pd.Series(RFP_nucleus, index=cell_data.index)
        cell_data.loc[:, "RFP_cyto"] = pd.Series(RFP_cyto, index=cell_data.index)
        cell_data.loc[:, "RFP_ratio"] = pd.Series(RFP_ratio, index=cell_data.index)

        # store this cell's data in the final dataframe which eventually contains data of all cells
        final_data = pd.concat([final_data, cell_data])
        nth_cell += 1

    # reset the index and save the final dataframe
    final_data = final_data.reset_index(drop=True)
    final_data.to_excel(f"{output_dir}excel/sfp1_RFP_only_data.xlsx")

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
    for data_type in ["RFP_total", "RFP_nucleus", "RFP_cyto", "RFP_ratio"]:
        data_interpolated = pd.DataFrame(columns=cols)

        for cell in individual_cells:
            try:  # if the cell does not have any kario events and thus isn't present in the dictionary
                kario_events[cell]
            except KeyError:
                continue

            single_cell_data = final_data[final_data.Cell_pos == cell]
            single_cell_data = single_cell_data[~single_cell_data.RFP_nucleus.isnull()]

            count = 0
            while count < len(kario_events[cell]) - 1:
                tp1, tp2 = kario_events[cell][count], kario_events[cell][count + 1]  # go from cyto- to kariokinesis
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
    final_dataframe_path = f"{output_dir}excel/sfp1_RFP_only_data.xlsx"
    if not os.path.exists(final_dataframe_path):
        final_data = get_data_for_all_cells()
    else:
        print("Data has been generated already. The output file exists.")
        final_data = pd.read_excel(final_dataframe_path)

    # get event data
    kario_events, budding_events = load_events(kario_data_path, budding_data_path)

    generate_plots.combined_channel_data(final_data, output_dir, "RFP")

    # check if cycles have been split and interpolated, if not, do this
    interpolated_dataframes = split_cycles_and_interpolate(final_data, kario_events)

    # generate a big overview image that shows RFP intensities
    generate_plots.dynamics_overview_all_cells(final_data, kario_events, budding_events, output_dir, "RFP")

    # generate a plot per interpolated cycle
    generate_plots.separate_interpolated_cycles(interpolated_dataframes, output_dir, "RFP")

    generate_plots.averaged_plots(interpolated_dataframes, output_dir, "RFP")

    toc = time.perf_counter()
    secs = round(toc - tic, 4)
    print(f"Done. Runtime was {secs} seconds ({round(secs / 60, 2)} minutes)")


# SCRIPT STARTS HERE
if __name__ == "__main__":
    main()
    sys.exit(0)
