import os
import time
import re
from math import pi, sin, cos
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage.io import imread
from skimage.filters import threshold_local, threshold_minimum, threshold_multiotsu
from skimage.morphology import remove_small_objects, disk, erosion

plt.style.use('seaborn-v0_8')

#######################
### IMPORTANT PATHS ###
#######################
data_dir = "C:/Users/Olle de Jong/Documents/MSc Biology/MSB Research/Code and data/Data"
tiff_files_dir = data_dir + "/processed_tiffs_nup133/"  # relative path from data directory to tiff directory
output_dir = data_dir + "/Output/"  # relative path from data directory to image output directory
budj_data_folder = data_dir + "/Other/Nup133/BudJ/"  # folder that holds the BudJ info on all cells
budding_data_path = data_dir + "/Other/Nup133/buddings.txt"  # budding events
kario_data_path = data_dir + "/Other/Nup133/kariokinesis.txt"  # kariokinesis events

############################
### THRESHOLDING GLOBALS ###
############################
scaling_factor = 0.16  # microns per pixel ---> 100x objective
scaling_factor2 = 0.016  # microns per pixel ---> 100x objective
bloc_size_frac_to_use = 0.09  # fraction of the total cell mask pixels that is used as local thresholding block size
offset_to_use = -50  # offset for local thresholding


#################
### FUNCTIONS ###
#################
def ellipse_from_budj(t, cell_data):
    """
    Define the function that can extract parameters of the ellipse using the data
    Function to extract parameters of the ellipse from the BudJ table
    """
    data_at_frame = cell_data[cell_data["TimeID"] == t]
    x_pos = float(data_at_frame['x']) / scaling_factor
    y_pos = float(data_at_frame['y']) / scaling_factor
    majorR = float(data_at_frame["Major R"]) / scaling_factor
    minorR = float(data_at_frame["Minor r"]) / scaling_factor
    angle = float(data_at_frame['Angle']) * (pi / 180)  # convert to radians
    return x_pos, y_pos, majorR, minorR, angle


def get_ellipse(imageGFP, nuc_mask):
    """
    Function that generates and fits the ellipse using the opencv package.
    :param imageGFP: the GFP image data
    :param nuc_mask: nuclear mask determined by budj data
    :return:
    """
    mask = imageGFP * nuc_mask
    thresh = mask.astype(np.uint8)  # change type to uint8

    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # if there is a contour found
    if len(contours) > 0:
        cnt = max(contours, key=len)
        if len(cnt) > 4:
            return cv2.fitEllipse(cnt)


def round_up_to_odd(f):
    """
    Rounds up any number to a whole odd integer
    """
    return int(np.ceil(f) // 2 * 2 + 1)


def load_budj_files():
    # find all the files that hold budj data
    files = []
    for filename in os.listdir(budj_data_folder):
        if ".csv" in filename:
            prefix_position = f"pos{filename[-25:-23]}_"
            files.append((filename, prefix_position))
    return files


def load_all_budj_data():
    """
    Collects all file names and using that, all BudJ data is loaded from the files
    :return:
    """
    files = load_budj_files()  # get the separate files

    # convert time_ids to real time minutes
    TimeIDs = range(1, 151)  # TimeIDs range 1 through 150
    time_conversion = pd.DataFrame({
        "TimeID": TimeIDs,
        "Time": [x * 5 for x in TimeIDs]
    })

    # add the data of all files to one single dataframe
    budj_data = pd.DataFrame({})
    for f in files:
        pos_data = pd.read_csv(budj_data_folder + f[0], header=0, index_col=0)
        pos_data["Cell_pos"] = f[1] + pos_data["Cell"].map(str)
        pos_data = pos_data.loc[:, ["TimeID", "Cell_pos", "Volume", "x", "y", "Major R", "Minor r", "Angle"]]
        budj_data = pd.concat([budj_data, pos_data])

    budj_data = pd.merge(budj_data, time_conversion, on="TimeID")
    budj_data = budj_data.sort_values(["Cell_pos", "TimeID"])
    return budj_data


def load_events():
    """
    Using BudJ, the kyrokinesis and budding events have been tracked. This function loads that
    data and stores it in a dictionary object.
    :return:
    """
    events_files = {"budding": budding_data_path, "kariokinesis": kario_data_path}

    budding_events = {}
    kario_events = {}
    for event_type in events_files:  # for either budding or kariokinesis events
        opened_file = open(events_files[event_type])  # open the file
        events = {}  # temp storage object
        for line in opened_file:  # every line in the file is a cell
            if line == "\n":
                continue
            # process the two parts of the line by removing characters
            parts = line.split(':')
            cell_id = parts[0].replace("\"", "").strip()  # first part of the line is the individual cell
            timepoints = re.sub('[\[\]]', "", parts[1])  # second part are the timepoints
            # split timepoints on space to capture them in a list
            split_timepoints = timepoints.split(",")
            events[cell_id] = [int(x.strip()) for x in split_timepoints[:-1]]  # remove redundant last comma

        # save the events to the right dictionary
        if event_type == "budding":
            budding_events = events
        if event_type == "kariokinesis":
            kario_events = events
        opened_file.close()
    return budding_events, kario_events


def get_area_and_vol(minor_r, major_r):
    """
    Calculates the area and the volume of an ellipse using the major and minor radii
    :param minor_r:
    :param major_r:
    :return:
    """
    # NOTE: When using OpenCV / cv2 for ellipses, the axes that you give the package to drawn an ellipse are radii,
    # however, what the ellipse fitting provides you with are diameters. In order to calculate areas and volumes we
    # need to divide the axes by two to get the semi-axes.
    r1c = (major_r * scaling_factor) / 2  # from pixels to µm using scaling factor
    r2c = (minor_r * scaling_factor) / 2  # from pixels to µm using scaling factor
    r3c = (np.average((major_r, minor_r)) * scaling_factor) / 2  # third axis of ellipse is average of two known ones
    volume = (4 / 3) * pi * r1c * r2c * r3c  # µm * µm * µm --> cubic µm
    area = pi * r1c * r2c  # µm * µm --> squared µm
    return area, volume


def get_whole_cell_mask(t, single_cell_data, image_shape):
    """
    Given a certain time-point in the tiff movie, the single cell BudJ data, and the image shape,
    the area of the cell is calculated.
    :param t:
    :param single_cell_data:
    :param image_shape:
    :return: whole_cell_mask, the x pos and the y pos of the cell
    """
    # get/calculate the ellipse information
    x_pos, y_pos, majorR, minorR, A = ellipse_from_budj(t, single_cell_data)

    y_dim, x_dim = image_shape  # get the dimensions (512x512)

    # create an ogrid that helps us select/'mask' the info we want
    row, col = np.ogrid[:y_dim, :x_dim]

    # get the mask of the whole cell
    whole_cell_mask = (
            (
                    (((col - x_pos) * cos(A) + (row - y_pos) * sin(A)) ** 2) / (majorR ** 2)
                    +
                    (((col - x_pos) * sin(A) - (row - y_pos) * cos(A)) ** 2) / (minorR ** 2) - 1
            )  # if this sum
            < 0  # is smaller than zero
    )
    return whole_cell_mask, x_pos, y_pos


def remove_outliers(individual_cells, vol_data):
    """
    This function removes the outliers based on the Inter Quartile Range approach. This is done both on the
    nuclear and whole-cell volume columns.
    :param individual_cells:
    :param vol_data:
    :return:
    """
    print("Removing outliers from the volume data based on the Inter Quartile Range approach..")
    data_wo_outliers = pd.DataFrame({})
    for cell in individual_cells:
        cell_data = vol_data[vol_data["Cell_pos"] == cell]
        # for whole-cell volume
        Q1 = cell_data.Cell_volume.quantile(0.25)
        Q3 = cell_data.Cell_volume.quantile(0.75)
        IQR = Q3 - Q1
        cell_data = cell_data[~((cell_data.Cell_volume < (Q1 - 1.5 * IQR)) | (cell_data.Cell_volume > (Q3 + 1.5 * IQR)))]

        # now for nuclear volume
        Q1 = cell_data.Nucleus_volume.quantile(0.25)
        Q3 = cell_data.Nucleus_volume.quantile(0.75)
        IQR = Q3 - Q1
        cell_data = cell_data[~((cell_data.Nucleus_volume < (Q1 - 1.5 * IQR)) | (cell_data.Nucleus_volume > (Q3 + 1.5 * IQR)))]

        data_wo_outliers = pd.concat([data_wo_outliers, cell_data])

    return data_wo_outliers


def get_volume_data(budj_data, individual_cells):
    """
    The most complex, and by far most compute intensive, function of this script. It goes through all tiff movies and
    all cells, and for each cell it loops over all frame in the movie. For every frame, local thresholding is performed,
    and an ellipse is fitted. Using this ellipse, the volume is calculated. It all is stored within one dataframe.

    :param budj_data:
    :param individual_cells:
    :return:
    """
    nth_cell = 0
    final_volume_data = pd.DataFrame({})
    for pos in range(1, 21):
        perc_ranges = pos / 20
        if pos < 10:
            pos = "0" + str(pos)
        pos = str(pos)
        image = imread(os.path.join(f"{tiff_files_dir}2022_12_06_nup133_yegfp_xy{pos}.nd2.tif"))  # load the image
        imageGFP = image[:, 1, :, :]  # get the GFP data

        cells_in_pos = [i for i in individual_cells if pos in i]
        for cell in cells_in_pos:
            print(f"Working.. Now at {round(nth_cell / len(individual_cells) * 100)}%", end="\r", flush=True)
            single_cell_data = budj_data[budj_data["Cell_pos"] == cell].sort_values(by="Time")  # get single cell data
            could_not_fit = 0

            cell_areas, nuc_areas, cell_volumes, nuc_volumes = [], [], [], []
            for t in single_cell_data.TimeID:
                t_in_tiff = t - 1  # skew the time by one for tiff dataframe
                imageGFP_at_frame = imageGFP[t_in_tiff, :, :]  # get the GFP data

                # get the cell mask through budj data
                whole_cell_mask, x_pos, y_pos = get_whole_cell_mask(t, single_cell_data, imageGFP_at_frame.shape)
                imageGFP_cell_mask = imageGFP_at_frame * whole_cell_mask  # keep only cell data

                num_cell_pixels = np.count_nonzero(whole_cell_mask == True)
                bloc_size_cell_size_dependent = round_up_to_odd(bloc_size_frac_to_use * num_cell_pixels)

                nucl_thresh_mask_local = threshold_local(
                    image=imageGFP_cell_mask,
                    block_size=bloc_size_cell_size_dependent,
                    offset=offset_to_use
                )
                imageGFP_nuc_mask_local = remove_small_objects(imageGFP_cell_mask > nucl_thresh_mask_local)

                # save cell area and volume to dataframe
                (c_x, c_y), (c_MA, c_ma), c_angle = get_ellipse(imageGFP_at_frame, whole_cell_mask)  # fit ellipse cell
                cell_area, cell_volume = get_area_and_vol(c_ma, c_MA)  # get the area and volume of the cell
                cell_areas.append(cell_area)
                cell_volumes.append(cell_volume)

                # try to fit the ellipse on the nucleus, if there is none, then add None to the list
                try:
                    (x, y), (MA, ma), angle = get_ellipse(imageGFP_at_frame, imageGFP_nuc_mask_local)
                except Exception:
                    # thresholding did not lead to an ellipse, nucleus was probably out of focus
                    could_not_fit += 1
                    nuc_volumes.append(None)
                    nuc_areas.append(None)
                    continue

                # save nucleus area and volume to dataframe
                nuc_area, nuc_volume = get_area_and_vol(ma, MA)
                nuc_areas.append(nuc_area)
                nuc_volumes.append(nuc_volume)

            # calculate the ratios
            ratios = []
            for i in range(0, len(nuc_volumes)):
                ratios.append(None) if nuc_volumes[i] is None else ratios.append(nuc_volumes[i] / cell_volumes[i])

            # store data in this cell its dataframe
            single_cell_data["Cell_area"] = pd.Series(cell_areas, index=single_cell_data.index, dtype="float64")
            single_cell_data["Cell_volume"] = pd.Series(cell_volumes, index=single_cell_data.index, dtype="float64")
            single_cell_data["Nucleus_area"] = pd.Series(nuc_areas, index=single_cell_data.index, dtype="float64")
            single_cell_data["Nucleus_volume"] = pd.Series(nuc_volumes, index=single_cell_data.index, dtype="float64")
            single_cell_data["N/C_ratio"] = pd.Series(ratios, index=single_cell_data.index, dtype="float64")

            # store this cell's data in the final dataframe which eventually contains data of all cells
            final_volume_data = pd.concat([final_volume_data, single_cell_data])
            # print(f"{cell} - Couldn't fit ellipse {could_not_fit} out of {len(single_cell_data['TimeID'])} times")
            nth_cell += 1

    # final_volume_data = remove_outliers(individual_cells, final_volume_data)  # remove outliers
    final_volume_data = final_volume_data.reset_index(drop=True)  # reset index of dataframe
    final_volume_data.to_excel(f"{output_dir}excel/nup133_volume_data.xlsx")  # save the final file
    return final_volume_data


def generate_separate_volume_plots(individual_cells, final_volume_data):
    """
    Creates a separate plot for every cell that displays the nuclear and whole-cell volumes
    together. This might give a good overview on the data quality per cell.
    :param individual_cells:
    :param final_volume_data:
    :return:
    """
    for cell in individual_cells:
        fig, axs = plt.subplots(2, figsize=(5, 5))
        fig.supxlabel("Time (frames)")
        fig.supylabel("Volume (µm\u00b3)")
        fig.subplots_adjust(hspace=0.35, wspace=0.4)
        fig.suptitle(f"{cell} - Whole cell and nuclear volumes over time")
        single_cell_data = final_volume_data[final_volume_data.Cell_pos == cell]
        single_cell_data = single_cell_data[~single_cell_data.Nucleus_volume.isnull()]

        axs[0].plot(single_cell_data.TimeID, single_cell_data.Cell_volume, 'or')
        axs[0].set_title("Whole cell volume", fontstyle='italic', y=1.02)
        axs[1].plot(single_cell_data.TimeID, single_cell_data.Nucleus_volume, 'oy')
        axs[1].set_title("Nuclear volume", fontstyle='italic', y=1.02)
        plt.savefig(f"{output_dir}/plots/separate_cell_plots/with_outliers/{cell}_volumes_overT.png", bbox_inches='tight', dpi=300)
        plt.close(fig)


def split_cycles_and_interpolate(final_volume_data, individual_cells, kario_events):
    """
    Function responsible splitting the data of each cell on kariokinesis events, where after the data per
    cycle is interpolated to 100 datapoints.

    :param final_volume_data:
    :param individual_cells:
    :param kario_events:
    :return:
    """
    min_datapoints_for_interpolation = 14
    desired_datapoints = 100
    cols = ["cell", "cycle", "start", "end"]
    cols += range(desired_datapoints)

    interpolated_dataframes = []
    for data_type in ["Cell_volume", "Nucleus_volume", "N/C_ratio"]:  # interpolate the columns in this list
        tot_under = 0
        data_interpolated = pd.DataFrame(columns=cols)

        for cell in individual_cells[1:]:  # remove cell pos01_1, since it has too many missing frames
            single_cell_data = final_volume_data[final_volume_data.Cell_pos == cell]
            single_cell_data = single_cell_data[~single_cell_data.Nucleus_volume.isnull()]

            count = 0
            while count < len(kario_events[cell]) - 1:
                tp1 = kario_events[cell][count]
                tp2 = kario_events[cell][count + 1]
                cell_cycle_dat = single_cell_data[single_cell_data.TimeID.between(tp1, tp2)]

                # when there are not enough datapoints (because of outlier removal) for this cycle, ignore it
                if len(cell_cycle_dat) < min_datapoints_for_interpolation:
                    tot_under += 1
                    count += 1
                    continue

                # get the original x and y data
                x = np.linspace(0, len(cell_cycle_dat), len(cell_cycle_dat))
                y = cell_cycle_dat[data_type]

                # interpolate the data towards 100 datapoints
                x_new = np.linspace(0, len(cell_cycle_dat), desired_datapoints)
                f1 = interpolate.interp1d(x, y, kind='linear')
                f2 = interpolate.interp1d(x, y, kind='cubic')

                # add the interpolated data to a dataframe
                data_interpolated.loc[len(data_interpolated)] = [cell, count + 1, tp1, tp2] + f2(x_new).tolist()
                count += 1

        if data_type == "N/C_ratio": data_type = "NC_ratio"  # cannot save file
        data_interpolated.to_excel(f"{output_dir}excel/cycles_interpolated_{data_type}s.xlsx")
        interpolated_dataframes.append(data_interpolated)
        print(f"For the datatype {data_type}, there were {tot_under} cycles removed from the final dataframe since "
              f"they had less than {min_datapoints_for_interpolation} datapoints")
    return interpolated_dataframes


def generate_averaged_plots(cell_volumes_interpolated, nuc_volumes_interpolated, nc_ratios_interpolated):
    """
    Takes the interpolated data files (Excel files) and averages these. The averages are plotted over time and
    saved to the user's system.

    :return:
    """
    data_list = [  # average the interpolated cycles
        ("Cell volume", cell_volumes_interpolated.mean(axis=0, numeric_only=True), "µm\u00b3"),
        ("Nuclear volume", nuc_volumes_interpolated.mean(axis=0, numeric_only=True), "µm\u00b3"),
        ("N/C ratio", nc_ratios_interpolated.mean(axis=0, numeric_only=True), None)
    ]

    for data in data_list:
        plt.plot(data[1][4:].values, 'r', linewidth=3)
        plt.title(f"Average {data[0]} over time", fontstyle='italic', y=1.02)
        plt.xlabel("Time")
        if data[2] is not None:
            plt.ylabel(f"{data[0]} ({data[2]})")
        else:
            plt.ylabel(data[0])
        filename = "NC ratio" if data[0] == "N/C ratio" else data[0]
        plt.savefig(
            f"{output_dir}plots/interpolated_averaged/with_outliers/{filename}.png",
            bbox_inches='tight',
            dpi=350
        )
        plt.close()

    generate_combined_volumes_plot(cell_volumes_interpolated, nuc_volumes_interpolated)


def generate_combined_volumes_plot(cell_volumes_interpolated, nuc_volumes_interpolated):
    """

    :param cell_volumes_interpolated:
    :param nuc_volumes_interpolated:
    :return:
    """
    # take the mean of the dataframes again
    cell_volumes_interpolated = cell_volumes_interpolated.mean(axis=0, numeric_only=True)
    nuc_volumes_interpolated = nuc_volumes_interpolated.mean(axis=0, numeric_only=True)

    # also generate one with nuc and cell volume together
    fig, ax1 = plt.subplots()
    fig.suptitle("Whole-cell and nuclear volumes over time (interpolated & averaged)")

    ax1.set_xlabel('Time')
    ax1.grid(False)
    ax1.set_ylabel("Cell volume (µm\u00b3)", color='tab:red')
    ax1.plot(cell_volumes_interpolated[4:].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.grid(False)
    ax2.set_ylabel("Nucleus volume (µm\u00b3)", color='tab:blue')
    ax2.plot(nuc_volumes_interpolated[4:].values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.savefig(
        f"{output_dir}plots/interpolated_averaged/Volumes_together.png",
        bbox_inches='tight',
        dpi=350
    )
    plt.close()


def main():
    tic = time.perf_counter()
    budj_data = load_all_budj_data()  # load the budj data from all the separate files
    budding_events, kario_events = load_events()  # load the kyrokinesis and budding events
    individual_cells = sorted(list(set(budj_data["Cell_pos"])))  # how many cells are there in total

    # if volume dataset already exists, prevent generating this again and load it
    final_dataframe_path = f"{output_dir}excel/nup133_volume_data_with_outliers.xlsx"
    if os.path.exists(final_dataframe_path):
        print("Volume data has been generated already. The output file exists. loading it from file..")
        final_volume_data = pd.read_excel(final_dataframe_path)
    else:
        print("Generating volume data..")
        final_volume_data = get_volume_data(budj_data, individual_cells)
    
    # generate a combined volumes plot for all cells separate
    generate_separate_volume_plots(individual_cells, final_volume_data)
    
    # check if cycles have been split and interpolated, if not, do this
    count = 0
    for filename in os.listdir(f"{output_dir}excel/"):
        if "cycles_interpolated" in filename and "~$" not in filename:
            count += 1
    if count == 3:
        # load the interpolated files
        print("Interpolation has already been performed. Output files exist. Loading them..")
        cell_volumes_interpolated = pd.read_excel(f"{output_dir}excel/cycles_interpolated_Cell_volumes.xlsx")
        nuc_volumes_interpolated = pd.read_excel(f"{output_dir}excel/cycles_interpolated_Nucleus_volumes.xlsx")
        nc_ratios_interpolated = pd.read_excel(f"{output_dir}excel/cycles_interpolated_NC_ratios.xlsx")
    else:
        print("Performing interpolation on cell volume, nuc volume and n/c ratio data..")
        interpolated_dataframes = split_cycles_and_interpolate(final_volume_data, individual_cells, kario_events)
        cell_volumes_interpolated = interpolated_dataframes[0]
        nuc_volumes_interpolated = interpolated_dataframes[1]
        nc_ratios_interpolated = interpolated_dataframes[2]

    # average the interpolated data and plot the result (cell volume, nucleus volume and N/C ratio)
    print("Generating the averaged interpolated data plots..")
    generate_averaged_plots(cell_volumes_interpolated, nuc_volumes_interpolated, nc_ratios_interpolated)

    toc = time.perf_counter()
    secs = round(toc - tic, 4)
    print(f"Done. Runtime was {secs} seconds ({round(secs / 60, 2)} minutes)")


# SCRIPT STARTS HERE
if __name__ == "__main__":
    main()
