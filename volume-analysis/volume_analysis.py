import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
from math import pi
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects

import generate_plots  # file that contains the functions that generate plots
from shared.shared_functions import ellipse_from_budj, get_whole_cell_mask, round_up_to_odd, read_images, \
                                    get_time_conversion, load_events, create_excel_dir

#######################
### IMPORTANT PATHS ###
#######################
data_dir = "C:/Users/Olle de Jong/Documents/MSc Biology/MSB Research/Code and data/Data/"
tiff_files_dir = f"{data_dir}processed_tiffs_nup133/"  # relative path from data directory to tiff directory
output_dir = f"{data_dir}output/"  # relative path from data directory to image output directory
budj_data_folder = f"{data_dir}Input/Nup133/BudJ/"  # folder that holds the BudJ info on all cells
budding_data_path = f"{data_dir}Input/Nup133/buddings.txt"  # budding events
kario_data_path = f"{data_dir}Input/Nup133/kariokinesis.txt"  # kariokinesis events

############################
### THRESHOLDING GLOBALS ###
############################
scaling_factor = 0.16  # microns per pixel ---> 100x objective
bloc_size_frac_to_use = 0.09  # fraction of the total cell mask pixels that is used as local thresholding block size
offset_to_use = -50  # offset for local thresholding

pd.options.mode.chained_assignment = None  # default='warn'

###########################
### Optimal cell cycles ###
###########################
# We only want to average over the cycles that include daughter bud data. This list is for that purpose.
# In addition, a few cycles have been removed manually because of low data quality.
# These are: 06_2c1, 08_1c1, 10_3c2, 18_1c1 and 18_2c1
cycles_to_average = {
    "pos02_2": [1, 2], "pos03_1": [1], "pos03_3": [1], "pos05_1": [1], "pos07_1": [1], "pos07_2": [1, 2],
    "pos08_1": [2], "pos09_1": [1, 2, 3], "pos10_1": [1], "pos10_3": [1], "pos11_2": [2], "pos12_3": [1, 2],
    "pos13_1": [1], "pos13_2": [1], "pos13_3": [1], "pos14_1": [1], "pos14_3": [1], "pos15_1": [1], "pos15_3": [1, 2],
    "pos16_1": [1], "pos16_2": [1], "pos20_2": [1], "pos20_3": [1, 2], "pos20_4": [1, 2],
}


#########################
### GENERAL FUNCTIONS ###
#########################
def get_ellipse(image_gfp, mask):
    """
    Function that generates and fits the ellipse using the opencv package.
    :param image_gfp: the GFP image data
    :param mask: nuclear mask determined by budj data
    :return:
    """
    masked_image = image_gfp * mask
    thresh = masked_image.astype(np.uint8)  # change type to uint8

    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # if there is a contour found
    if len(contours) > 0:
        cnt = max(contours, key=len)
        if len(cnt) > 4:
            return cv2.fitEllipse(cnt)


def load_budj_files():
    # find all the files that hold budj data
    files = []
    for filename in os.listdir(budj_data_folder):
        if "with_daughters.csv" in filename:
            cell_name = f"pos{filename[2:6]}"
            files.append((filename, cell_name))
    return files


def load_all_budj_data():
    """
    Collects all file names and using that, all BudJ data is loaded from the files
    :return:
    """
    print("Loading all BudJ excel files..", end="\r", flush=True)
    files = load_budj_files()  # get the separate files

    # add the data of all files to one single dataframe
    budj_data = pd.DataFrame({})
    for filename, cell_name in files:
        # mother with daughter(s) data
        pos_data = pd.read_csv(budj_data_folder + filename, header=0, index_col=0)
        pos_data["Cell_pos"] = None
        temp_data = pd.DataFrame(columns=pos_data.columns)
        # name the cells correctly, the mother's name is unchanged, but the daughters are now recognizable
        for index, row in pos_data.iterrows():
            # when cell is not cell 1, then it is a daughter cell
            row["Cell_pos"] = cell_name if row["Cell"] == 1 else f"{cell_name}_d{round(row['Cell'] - 1)}"
            temp_data.loc[len(temp_data) + 1] = row.values

        # keep only the following columns
        temp_data = temp_data.loc[:, ["TimeID", "Cell_pos", "Volume", "x", "y", "Major R", "Minor r", "Angle"]]

        # save the mother + daughter data to the bigger dataframe holding data for all mothers + daughters
        budj_data = pd.concat([budj_data, temp_data])

    # sort on cell pos and time-frame
    budj_data = pd.merge(budj_data, get_time_conversion(), on="TimeID")
    budj_data = budj_data.sort_values(["Cell_pos", "TimeID"]).reset_index(drop=True)

    print("Loading all BudJ excel files.. Done!")
    return budj_data


def get_area_and_vol(minor_r, major_r):
    """
    Calculates the surface area and the volume of an ellipse using the major and minor radii
    :param minor_r:
    :param major_r:K
    :return:
    """
    # NOTE: When using OpenCV / cv2 for ellipses, the axes that you give the package to draw an ellipse are radii or
    # semi-axes however, what the ellipse fitting provides you with are diameters. In order to calculate areas and
    # volumes we need to divide the axes by two to get the semi-axes.
    r1c = (major_r * scaling_factor) / 2  # from pixels to µm using scaling factor
    r2c = (minor_r * scaling_factor) / 2  # from pixels to µm using scaling factor
    r3c = (np.average((major_r, minor_r)) * scaling_factor) / 2  # third axis of ellipse is average of two known ones
    VOL = (4 / 3) * pi * r1c * r2c * r3c  # VOL: µm * µm * µm --> cubic µm
    SA = 4 * pi * (((r1c * r2c) ** 1.6075 + (r1c * r3c) ** 1.6075 + (r2c * r3c) ** 1.6075) / 3) ** (1 / 1.6075)
    # SA: µm *  µm --> squared µm
    return SA, VOL


def remove_outliers(individual_cells, vol_data):
    """
    This function removes the outliers based on the Inter Quartile Range approach. This is done both on the
    nuclear and whole-cell volume columns.
    :param individual_cells:
    :param vol_data:
    :return:
    """
    print("Removing outliers from the volume data based on the Inter Quartile Range approach..", end="\r", flush=True)
    data_wo_outliers = pd.DataFrame({})
    for cell in individual_cells:
        cell_data = vol_data[vol_data["Cell_pos"] == cell]
        # for whole-cell volume
        for data_column in [cell_data.Cell_volume, cell_data.Nucleus_volume]:
            Q1 = data_column.quantile(0.25)
            Q3 = data_column.quantile(0.75)
            IQR = Q3 - Q1
            cell_data = cell_data[~((data_column < (Q1 - 1.5 * IQR)) | (data_column > (Q3 + 1.5 * IQR)))]

        data_wo_outliers = pd.concat([data_wo_outliers, cell_data])
    print("Removing outliers from the volume data based on the Inter Quartile Range approach.. Done!")
    return data_wo_outliers


def get_extrapolated_datapoints(time_extra_frames, time_partial, data_partial):
    """
    Takes a timerange that holds time-points for which extrapolated data is generated. The linear fit is made
    based on the original time and volume/area axis
    :param time_extra_frames:
    :param time_partial:
    :param data_partial:
    :return:
    """
    model = LinearRegression()
    model.fit(time_partial, data_partial)  # linear fit on (partial) original data
    pred = model.predict(time_extra_frames)

    extra_values = [
        pred[0][0] if pred[0][0] > 0 else 0,
        pred[1][0] if pred[1][0] > 0 else 0,
        pred[2][0] if pred[2][0] > 0 else 0
    ]

    return extra_values


def get_extrapolated_bud_data(image_gfp, daughters_data):
    """
    This function takes the original budj daughter dataframe and extrapolates the daughter data backwards until the
    volume hits zero. This is done through linear extrapolation on the first few datapoints of the daughter volume.
    :return: altered daughters_data
    """

    # first we need to get the daughters area and volume on the time-points that are defined.
    daughters_area_vol_data = {}

    individual_daughters = sorted([x for x in daughters_data["Cell_pos"].unique()])
    for cell in individual_daughters:

        daughter_data = daughters_data[daughters_data["Cell_pos"] == cell]
        daughter_area_vol_data = []

        # generate the area and volumes for the know data-points of the daughter cell
        for t in daughter_data.TimeID:
            t_in_tiff = t - 1
            imageGFP_at_frame = image_gfp[t_in_tiff, :, :]
            whole_cell_mask_daughter, x_pos, y_pos = get_whole_cell_mask(t, daughters_data, imageGFP_at_frame.shape)
            (d_x, d_y), (d_MA, d_ma), d_angle = get_ellipse(imageGFP_at_frame, whole_cell_mask_daughter)
            daughter_area, daughter_volume = get_area_and_vol(d_ma, d_MA)
            daughter_area_vol_data.append((t, daughter_area, daughter_volume))

        # now we need to fit a linear regression line and extrapolate until the volume hits zero
        time_o = [time_point[0] for time_point in daughter_area_vol_data]
        areas_o = [time_point[1] for time_point in daughter_area_vol_data]
        vols_o = [time_point[2] for time_point in daughter_area_vol_data]

        # perform extrapolation only on first 4 datapoints
        time_partial = np.array(time_o[:4]).reshape(-1, 1)
        vols_partial = np.array(vols_o[:4]).reshape(-1, 1)
        areas_partial = np.array(areas_o[:4]).reshape(-1, 1)

        # generate a new time-axis that has three additional frames at the beginning of the known time-axis
        # extrapolation is performed using this numpy array
        time_extra_frames = np.arange(time_partial[0] - 3, time_partial[-1]).reshape(-1, 1)

        final_vols = get_extrapolated_datapoints(time_extra_frames, time_partial, vols_partial) + vols_o
        final_areas = get_extrapolated_datapoints(time_extra_frames, time_partial, areas_partial) + areas_o

        # loop over a list holding all time-points for this daughter and store the known ánd extrapolated areas/vols
        i = 0
        time_points = [time_o[0] - 3, time_o[0] - 2, time_o[0] - 1] + time_o
        for t in time_points:
            daughters_area_vol_data[t] = (final_areas[i], final_vols[i])
            i += 1

    return daughters_area_vol_data


def get_data_for_single_cell(image_gfp, single_cell_data, daughters_data):
    """
    This function, for the given cell, calculates the nuclear and cell areas and nuclear and cell volumes at all
    time-frames. The whole-cell area and volume are incremented with daughter's area and volume when one exists.
    :param daughters_data:
    :param image_gfp:
    :param single_cell_data:
    :return:
    """
    cell_areas, nuc_areas, cell_volumes, nuc_volumes = [], [], [], []
    bud_perc = 0
    for t in single_cell_data.TimeID:
        t_in_tiff = t - 1  # skew the time by one for tiff dataframe
        imageGFP_at_frame = image_gfp[t_in_tiff, :, :]  # get the GFP data

        # WHOLE CELL #

        # get whole-cell area and volume
        whole_cell_mask, x_pos, y_pos = get_whole_cell_mask(t, single_cell_data, imageGFP_at_frame.shape)
        (c_x, c_y), (c_MA, c_ma), c_angle = get_ellipse(imageGFP_at_frame, whole_cell_mask)
        cell_area, cell_volume = get_area_and_vol(c_ma, c_MA)

        # check if there is a daughter at this frame; if so, add the area/volume of the daughter to the mother
        if t in daughters_data.keys():
            if t == list(daughters_data)[-1]:
                # save percentage of whole volume that is the bud for modelling purposes
                bud_perc = daughters_data[t][1] / (daughters_data[t][1] + cell_volume)
            cell_area += daughters_data[t][0]
            cell_volume += daughters_data[t][1]

        cell_areas.append(cell_area)
        cell_volumes.append(cell_volume)

        # NUCLEUS #

        imageGFP_cell_mask = imageGFP_at_frame * whole_cell_mask  # keep only data within the whole cell mask
        num_cell_pixels = np.count_nonzero(whole_cell_mask == True)  # count number of pixels in that mask
        bloc_size_cell_size_dependent = round_up_to_odd(bloc_size_frac_to_use * num_cell_pixels)
        nucl_thresh_mask_local = threshold_local(
            image=imageGFP_cell_mask,
            block_size=bloc_size_cell_size_dependent,
            offset=offset_to_use
        )
        imageGFP_nuc_mask_local = remove_small_objects(imageGFP_cell_mask > nucl_thresh_mask_local)

        # try to fit the ellipse on the nucleus, if the nucleus is out of focus, then add None to the list
        try:
            (x, y), (MA, ma), angle = get_ellipse(imageGFP_at_frame, imageGFP_nuc_mask_local)
        except Exception:
            # thresholding did not lead to an ellipse, nucleus was probably out of focus
            nuc_volumes.append(None)
            nuc_areas.append(None)
            continue

        # save nucleus area and volume to dataframe
        nuc_area, nuc_volume = get_area_and_vol(ma, MA)
        nuc_areas.append(nuc_area)
        nuc_volumes.append(nuc_volume)
    return cell_areas, cell_volumes, nuc_areas, nuc_volumes, bud_perc


def get_data_for_all_cells():
    """
    The most complex, and by far most compute intensive, function of this script. It goes through all tiff movies and
    all cells, and for each cell it loops over all frames in the movie. For every frame, local thresholding is
    performed and an ellipse is fitted. Using this ellipse, the volume is calculated. All the information is stored
    within a single dataframe.
    :return:
    """
    budj_data = load_all_budj_data()
    tiff_images = read_images(tiff_files_dir)

    individual_cells = sorted([x for x in budj_data["Cell_pos"].unique() if 'd' not in x])

    nth_cell = 0
    final_volume_data = pd.DataFrame({})
    bud_percentages = []

    # for every cell that was tracked using BudJ, we generate the area and volume data
    for cell in individual_cells:
        print(f"Generating volume data ( {round(nth_cell / len(individual_cells) * 100)}% )", end="\r", flush=True)

        # get right image from images dictionary and select the right channel (GFP)
        image = tiff_images[cell[3:5]]
        image_gfp = image[:, 1, :, :]

        # separate the data in mother and daughter
        cell_data = budj_data[budj_data["Cell_pos"] == cell].sort_values(by="TimeID")
        daughters_data = budj_data[budj_data["Cell_pos"].str.contains(f"{cell}_d")].sort_values(by="TimeID")

        # extrapolate daughter cell data in such a way that the first timeframe has a volume of 0
        daughters_data = get_extrapolated_bud_data(image_gfp, daughters_data)

        # get the cell and nuclear areas and volumes for every frame that this cell was tracked
        cell_areas, cell_vols, nuc_areas, nuc_vols, bud_perc = get_data_for_single_cell(image_gfp, cell_data, daughters_data)
        bud_percentages.append(bud_perc)

        # get the ratios
        ratios = []
        for i in range(0, len(nuc_vols)):
            ratios.append(None) if nuc_vols[i] is None else ratios.append(nuc_vols[i] / cell_vols[i] - nuc_vols[i])

        # store data in this cell its dataframe
        cell_data["Cell_area"] = pd.Series(cell_areas, index=cell_data.index, dtype="float64")
        cell_data["Cell_volume"] = pd.Series(cell_vols, index=cell_data.index, dtype="float64")
        cell_data["Nucleus_area"] = pd.Series(nuc_areas, index=cell_data.index, dtype="float64")
        cell_data["Nucleus_volume"] = pd.Series(nuc_vols, index=cell_data.index, dtype="float64")
        cell_data["N/C_ratio"] = pd.Series(ratios, index=cell_data.index, dtype="float64")

        # store this cell's data in the final dataframe which eventually contains data of all cells
        final_volume_data = pd.concat([final_volume_data, cell_data])
        nth_cell += 1

    # remove outliers, reset the index and save the final dataframe
    final_volume_data = remove_outliers(individual_cells, final_volume_data)
    final_volume_data = final_volume_data.reset_index(drop=True)
    final_volume_data.to_excel(f"{output_dir}excel/nup133_volume_data.xlsx")
    print(f"On average, a bud makes up {round(np.average(bud_percentages) * 100, 2)}% of the total cell volume just "
          f"before separation")
    print("Generating volume data.. Done!")
    return final_volume_data


def split_cycles_and_interpolate(final_volume_data):
    """
    Function responsible splitting the data of each cell on karyokinesis events, where after the data per
    cycle is interpolated to 100 datapoints.
    :param final_volume_data:
    :return interpolated_dataframes: a list with the interpolated dataframes
    """
    print("Performing interpolation on cell volume, nuc volume and n/c ratio data..", end="\r", flush=True)

    kario_events, budding_events = load_events(kario_data_path, budding_data_path)
    individual_cells = sorted(list(set(final_volume_data["Cell_pos"])))
    cycle_durations = []

    min_dp_for_int = 10
    desired_datapoints = 100
    cols = ["cell", "cycle", "start", "end"]
    cols += range(desired_datapoints)

    interpolated_dataframes = []
    tot_under = 0
    for data_type in ["Cell_area", "Cell_volume", "Nucleus_area", "Nucleus_volume", "N/C_ratio"]:
        data_interpolated = pd.DataFrame(columns=cols)

        for cell in individual_cells[1:]:  # remove cell pos01_1, since it has too many missing frames
            single_cell_data = final_volume_data[final_volume_data.Cell_pos == cell]
            single_cell_data = single_cell_data[~single_cell_data.Nucleus_volume.isnull()]

            count = 0
            while count < len(kario_events[cell]) - 1:
                tp1, tp2 = kario_events[cell][count] + 2, kario_events[cell][count + 1]
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

        if data_type == "N/C_ratio": data_type = "NC_ratio"
        data_interpolated.to_excel(f"{output_dir}excel/cycles_interpolated_{data_type}s.xlsx")
        interpolated_dataframes.append(data_interpolated)

    av_cycle_duration = np.average(cycle_durations)
    print("Performing interpolation on cell volume, nuc volume and n/c ratio data.. Done!")
    print(f"There were {tot_under} cycles removed that had not enough datapoints (less than {min_dp_for_int}).")
    print(f"Average cycle duration: {av_cycle_duration} frames, which is equal to {av_cycle_duration * 5} minutes.")
    return interpolated_dataframes


def keep_wanted_cycles(interpolated_dataframes):
    """
    Removes the cycles from the dataframes which do not include daughter bud data. Which cycles are
    wanted (per cell) are held in the cycles_to_average global dictionary.
    :param interpolated_dataframes:
    :return:
    """
    interpolated_dataframes_wanted_cycles = []

    for df in interpolated_dataframes:
        for index, row in df.iterrows():
            cell_pos = row['cell']
            if not (cell_pos in cycles_to_average.keys() and row['cycle'] in cycles_to_average[cell_pos]):
                df = df.drop([index])

        df = df.reset_index(drop=True)
        interpolated_dataframes_wanted_cycles.append(df)

    return interpolated_dataframes_wanted_cycles


def save_data_for_model(interpolated_dataframes_wanted_cycles):
    """
    Saves the averaged surface area, whole-cell volume and nuclear volume series to a csv file that will be used
    for modelling purposes.
    :param interpolated_dataframes_wanted_cycles:
    :return:
    """

    final_averages = pd.DataFrame({
        'cell_volumes': interpolated_dataframes_wanted_cycles[1].mean(axis=0, numeric_only=True)[4:].values,
        'nuc_surface_areas': interpolated_dataframes_wanted_cycles[2].mean(axis=0, numeric_only=True)[4:].values,
        'nuc_volumes': interpolated_dataframes_wanted_cycles[3].mean(axis=0, numeric_only=True)[4:].values
    })
    final_averages.to_excel(f"{output_dir}excel/final_averages_for_model.xlsx")


#####################
### Main function ###
#####################
def main():
    print("Getting started!")
    tic = time.perf_counter()  # start counter
    create_excel_dir(data_dir)

    print(load_events(kario_data_path, budding_data_path))
    # if volume dataset already exists, prevent generating this again and load it
    final_dataframe_path = f"{output_dir}excel/nup133_volume_data.xlsx"
    if not os.path.exists(final_dataframe_path):
        final_volume_data = get_data_for_all_cells()
    else:
        print("Volume data has been generated already. The output file exists.")
        final_volume_data = pd.read_excel(final_dataframe_path)

    # generate a combined volumes plot for all cells separate
    generate_plots.combined_volumes(final_volume_data)

    # check if cycles have been split and interpolated, if not, do this
    file_count = sum(1 for s in os.listdir(f"{output_dir}excel/") if 'cycles_interpolated' in s)
    if file_count != 5:
        interpolated_dataframes = split_cycles_and_interpolate(final_volume_data)
    else:
        print("Interpolation has already been performed. Output files exist. Loading them.")
        interpolated_dataframes = [
            pd.read_excel(f"{output_dir}excel/cycles_interpolated_Cell_areas.xlsx"),
            pd.read_excel(f"{output_dir}excel/cycles_interpolated_Cell_volumes.xlsx"),
            pd.read_excel(f"{output_dir}excel/cycles_interpolated_Nucleus_areas.xlsx"),
            pd.read_excel(f"{output_dir}excel/cycles_interpolated_Nucleus_volumes.xlsx"),
            pd.read_excel(f"{output_dir}excel/cycles_interpolated_NC_ratios.xlsx")
        ]

    # generate a plot per interpolated cycle
    generate_plots.separate_interpolated_cycles(interpolated_dataframes)

    # keep only the cycles that include daughter bud data and are of good quality
    interpolated_dataframes_wanted_cycles = keep_wanted_cycles(interpolated_dataframes)

    # create two figures; one that holds all whole-cell volume cycles, and one that only holds the 'best' ones
    generate_plots.combined_interpolated_volume_cycles(interpolated_dataframes, interpolated_dataframes_wanted_cycles)

    # average the interpolated data and plot the result (cell volume, nucleus volume and N/C ratio)
    generate_plots.averaged_plots(interpolated_dataframes_wanted_cycles)

    save_data_for_model(interpolated_dataframes_wanted_cycles)

    toc = time.perf_counter()
    secs = round(toc - tic, 4)
    print(f"Done. Runtime was {secs} seconds ({round(secs / 60, 2)} minutes)")


# SCRIPT STARTS HERE
if __name__ == "__main__":
    main()
    sys.exit(0)
