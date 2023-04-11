import os
import sys
import time
import re
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from skimage.io import imread
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects

plt.style.use('seaborn-v0_8')

#######################
### IMPORTANT PATHS ###
#######################
data_dir = "C:/Users/Olle de Jong/Documents/MSc Biology/MSB Research/Code and data/Data"
tiff_files_dir = data_dir + "/processed_tiffs_nup133/"  # relative path from data directory to tiff directory
output_dir = data_dir + "/Output/"  # relative path from data directory to image output directory
budj_data_folder = data_dir + "/Input/Nup133/BudJ/"  # folder that holds the BudJ info on all cells
budding_data_path = data_dir + "/Input/Nup133/buddings.txt"  # budding events
kario_data_path = data_dir + "/Input/Nup133/kariokinesis.txt"  # kariokinesis events

############################
### THRESHOLDING GLOBALS ###
############################
scaling_factor = 0.16  # microns per pixel ---> 100x objective
bloc_size_frac_to_use = 0.09  # fraction of the total cell mask pixels that is used as local thresholding block size
offset_to_use = -50  # offset for local thresholding


#################
### FUNCTIONS ###
#################
def save_figure(path, bbox_inches='tight', dpi=300):
    """
    Custom function that lets you save a pyplot figure and creates the directory where necessary
    """
    directory = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    if directory == '':
        directory = '.'

    if not os.path.exists(directory):
        os.makedirs(directory)

    savepath = os.path.join(directory, filename)

    # Actually save the figure
    plt.savefig(savepath, bbox_inches=bbox_inches, dpi=dpi)
    plt.close()


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


def get_ellipse(image_gfp, nuc_mask):
    """
    Function that generates and fits the ellipse using the opencv package.
    :param image_gfp: the GFP image data
    :param nuc_mask: nuclear mask determined by budj data
    :return:
    """
    mask = image_gfp * nuc_mask
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
        temp_data = temp_data.loc[:, ["TimeID", "Time (min)", "Cell_pos", "Volume", "x", "y", "Major R", "Minor r", "Angle"]]

        # save the mother + daughter data to the bigger dataframe holding data for all mothers + daughters
        budj_data = pd.concat([budj_data, temp_data])

    # sort on cell pos and time-frame
    budj_data = budj_data.sort_values(["Cell_pos", "TimeID"]).reset_index(drop=True)
    print("Loading all BudJ excel files.. Done!")
    return budj_data


def load_events():
    """
    Using BudJ, the kyrokinesis and budding events have been tracked. This function loads that
    data and stores it in a dictionary object.
    :return:
    """
    kario_events = {}
    opened_file = open(kario_data_path)
    events = {}

    for line in opened_file:  # every line in the file is a cell
        if line == "\n":
            continue
        # process the two parts of the line by removing characters
        parts = line.split(':')
        cell_id = parts[0].replace("\"", "").strip()
        timepoints = re.sub('[\[\]]', "", parts[1])

        # split timepoints on space to capture them in a list
        split_timepoints = timepoints.split(",")
        kario_events[cell_id] = [int(x.strip()) for x in split_timepoints[:-1]]

    opened_file.close()
    return kario_events


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


def read_images():
    print("Reading tiff images..", end="\r", flush=True)
    images = {}
    for pos in range(1, 21):
        if pos < 10:
            pos = "0" + str(pos)
        pos = str(pos)
        images[pos] = imread(os.path.join(f"{tiff_files_dir}2022_12_06_nup133_yegfp_xy{pos}.nd2.tif"))  # load the image
    print("Reading tiff images.. Done!")
    return images


def extrapolate_daughter_data(image_gfp, daughters_data):
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
        time_new = np.array(time_o[:4]).reshape(-1, 1)
        vols_new = np.array(vols_o[:4]).reshape(-1, 1)
        areas_new = np.array(areas_o[:4]).reshape(-1, 1)

        # generate a new time-axis that has three additional frames at the beginning of the known time-axis
        # extrapolation is performed using this numpy array
        time_extra_frames = np.arange(time_new[0] - 3, time_new[-1]).reshape(-1, 1)

        model = LinearRegression()  # create object for the class
        model.fit(time_new, vols_new)  # perform linear regression
        pred = model.predict(time_extra_frames)
        extra_vol_values = [
            pred[0][0] if pred[0][0] > 0 else 0,
            pred[1][0] if pred[1][0] > 0 else 0,
            pred[2][0] if pred[2][0] > 0 else 0
        ]
        final_vols = extra_vol_values + vols_o  # make a prediction for the preceding time-frame

        model = LinearRegression()  # create object for the class
        model.fit(time_new, areas_new)  # perform linear regression
        pred = model.predict(time_extra_frames)
        extra_area_values = [
            pred[0][0] if pred[0][0] > 0 else 0,
            pred[1][0] if pred[1][0] > 0 else 0,
            pred[2][0] if pred[2][0] > 0 else 0
        ]
        final_areas = extra_area_values + areas_o  # make a prediction for the preceding time-frame

        # loop over a list holding all time-points for this daughter and store the known ánd extrapolated areas/vols
        i = 0
        for t in [time_o[0] - 3, time_o[0] - 2, time_o[0] - 1] + time_o:
            daughters_area_vol_data[t] = (final_areas[i], final_vols[i])
            i += 1

    return daughters_area_vol_data


def get_data_for_all_timeframes(image_gfp, single_cell_data, daughters_data):
    """
    This function, for the given cell, calculates the nuclear and cell areas and nuclear and cell volumes at all
    time-frames. The whole-cell area and volume are incremented with daughter's area and volume when one exists.
    :param daughters_data:
    :param image_gfp:
    :param single_cell_data:
    :return:
    """
    cell_areas, nuc_areas, cell_volumes, nuc_volumes = [], [], [], []
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
    return cell_areas, cell_volumes, nuc_areas, nuc_volumes


def get_data_for_all_cells():
    """
    The most complex, and by far most compute intensive, function of this script. It goes through all tiff movies and
    all cells, and for each cell it loops over all frames in the movie. For every frame, local thresholding is
    performed and an ellipse is fitted. Using this ellipse, the volume is calculated. All the information is stored
    within a single dataframe.
    :return:
    """
    budj_data = load_all_budj_data()
    # budj_data = extrapolate_daughter_buds(budj_data)
    tiff_images = read_images()
    individual_cells = sorted([x for x in budj_data["Cell_pos"].unique() if 'd' not in x])

    nth_cell = 0
    final_volume_data = pd.DataFrame({})

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
        daughters_data = extrapolate_daughter_data(image_gfp, daughters_data)

        # get the cell and nuclear areas and volumes for every frame that this cell was tracked
        cell_areas, cell_vols, nuc_areas, nuc_vols = get_data_for_all_timeframes(image_gfp, cell_data, daughters_data)

        # get the ratios
        ratios = []
        for i in range(0, len(nuc_vols)):
            ratios.append(None) if nuc_vols[i] is None else ratios.append(nuc_vols[i] / cell_vols[i])

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

    print("Generating volume data.. Done!")
    return final_volume_data


def generate_separate_volume_plots(final_volume_data):
    """
    Creates a separate plot for every cell that displays the nuclear and whole-cell volumes
    together. This might give a good overview on the data quality per cell.
    :param final_volume_data:
    :return:
    """
    individual_cells = sorted(list(set(final_volume_data["Cell_pos"])))  # how many cells are there in total
    for cell in individual_cells:
        print(f"Generating volume plot per cell, {round(individual_cells.index(cell) / len(individual_cells) * 100)}% "
              f"done..", end="\r", flush=True)
        fig, axs = plt.subplots(2, figsize=(5, 5))
        fig.supxlabel("Time (frames)")
        fig.supylabel("Volume (µm\u00b3)")
        fig.subplots_adjust(hspace=0.35, wspace=0.4)
        fig.suptitle(f"{cell} - Whole cell and nuclear volumes over time")
        single_cell_data = final_volume_data[final_volume_data.Cell_pos == cell]
        single_cell_data = single_cell_data[~single_cell_data.Nucleus_volume.isnull()]

        axs[0].plot(single_cell_data.TimeID, single_cell_data.Cell_volume, 'r')
        axs[0].set_title("Whole cell volume", fontstyle='italic', y=1.02)
        axs[1].plot(single_cell_data.TimeID, single_cell_data.Nucleus_volume, 'y')
        axs[1].set_title("Nuclear volume", fontstyle='italic', y=1.02)
        save_figure(f"{output_dir}/plots/separate_cell_plots/{cell}_volumes_overT.png")


def split_cycles_and_interpolate(final_volume_data):
    """
    Function responsible splitting the data of each cell on karyokinesis events, where after the data per
    cycle is interpolated to 100 datapoints.
    :param final_volume_data:
    :param kario_events:
    :return:
    """
    kario_events = load_events()  # load the karyokinesis and budding events

    print("Performing interpolation on cell volume, nuc volume and n/c ratio data..", end="\r", flush=True)
    individual_cells = sorted(list(set(final_volume_data["Cell_pos"])))
    min_datapoints_for_interpolation = 10
    desired_datapoints = 100
    cols = ["cell", "cycle", "start", "end"]
    cols += range(desired_datapoints)

    interpolated_dataframes = []
    tot_under = 0
    for data_type in ["Cell_volume", "Nucleus_volume", "N/C_ratio"]:  # interpolate the columns in this list
        data_interpolated = pd.DataFrame(columns=cols)

        for cell in individual_cells[1:]:  # remove cell pos01_1, since it has too many missing frames
            single_cell_data = final_volume_data[final_volume_data.Cell_pos == cell]
            single_cell_data = single_cell_data[~single_cell_data.Nucleus_volume.isnull()]

            count = 0
            while count < len(kario_events[cell]) - 1:
                tp1 = kario_events[cell][count] + 1
                tp2 = kario_events[cell][count + 1] + 2  # extend by a frame to see real impact of karyokinesis
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
                f2 = interpolate.interp1d(x, y, kind='cubic')

                # add the interpolated data to a dataframe
                data_interpolated.loc[len(data_interpolated)] = [cell, count + 1, tp1, tp2] + f2(x_new).tolist()
                count += 1

        if data_type == "N/C_ratio": data_type = "NC_ratio"  # cannot save file
        data_interpolated.to_excel(f"{output_dir}excel/cycles_interpolated_{data_type}s.xlsx")
        interpolated_dataframes.append(data_interpolated)

    print("Performing interpolation on cell volume, nuc volume and n/c ratio data.. Done!")
    print(f"There were {tot_under} cycles removed among the three dataframes because they had less than "
          f"{min_datapoints_for_interpolation} datapoints.")
    return interpolated_dataframes


def generate_interpolated_cycle_plots(cell_volumes_interpolated, nuc_volumes_interpolated, nc_ratios_interpolated):
    """
    This function takes the data of the three main datatypes of interest and creates a plot of each one for every cycle.
    :param cell_volumes_interpolated:
    :param nuc_volumes_interpolated:
    :param nc_ratios_interpolated:
    :return:
    """
    interpolated_dataframes = [
        ("Cell volume", cell_volumes_interpolated, "Cell volume (µm\u00b3)"),
        ("Nucleus volume", nuc_volumes_interpolated, "Nucleus volume (µm\u00b3)"),
        ("NC ratio", nc_ratios_interpolated, "N/C ratio")
    ]
    progress = 1
    total_runs = len(cell_volumes_interpolated) + len(nuc_volumes_interpolated) + len(nc_ratios_interpolated)
    for interpolated_dataframe in interpolated_dataframes:
        count = 0
        data_type = interpolated_dataframe[0]
        while count < len(interpolated_dataframe[1]):
            print(f"Generating interpolated cycle plots, {round(progress / total_runs * 100)}% done..", end="\r",
                  flush=True)
            single_cycle_data = interpolated_dataframe[1].iloc[count]
            plt.plot(single_cycle_data[5:].values, 'r', linewidth=3)
            cell_name = single_cycle_data['cell']
            cycle_id = single_cycle_data['cycle']
            plt.title(f"{cell_name}, cycle {cycle_id}, {data_type} (interpolated)", fontstyle='italic', y=1.02)
            plt.xlabel("Time")
            plt.ylabel(interpolated_dataframe[2])
            save_figure(
                f"{output_dir}plots/separate_cell_plots/interpolated_cycles/{data_type}/{cell_name}_cycle{cycle_id}.png"
            )
            count += 1
            progress += 1


def generate_combined_volumes_plot(cell_volumes_interpolated, nuc_volumes_interpolated):
    """
    This function generates one of the final products of this script, namely a plot which shows the nuclear
    and whole-cell volumes over time. This data is interpolated to 100 datapoints per cycle, where after an average
    was taken of all cycles.
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

    save_figure(f"{output_dir}plots/interpolated_averaged/Volumes_together.png")


def generate_averaged_plots(cell_volumes_interpolated, nuc_volumes_interpolated, nc_ratios_interpolated):
    """
    Takes the interpolated data files (Excel files) and averages these. The averages are plotted over time and
    saved to the user's system.
    :return:
    """
    print("Generating the averaged interpolated data plots..", end="\r", flush=True)
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
        save_figure(f"{output_dir}plots/interpolated_averaged/{filename}.png")

    generate_combined_volumes_plot(cell_volumes_interpolated, nuc_volumes_interpolated)
    print("Generating the averaged interpolated data plots.. Done!")


def main():
    print("Getting started!")
    tic = time.perf_counter()  # start counter

    # if volume dataset already exists, prevent generating this again and load it
    final_dataframe_path = f"{output_dir}excel/nup133_volume_data.xlsx"
    if not os.path.exists(final_dataframe_path):
        final_volume_data = get_data_for_all_cells()
    else:
        print("Volume data has been generated already. The output file exists.")
        final_volume_data = pd.read_excel(final_dataframe_path)

    # generate a combined volumes plot for all cells separate
    generate_separate_volume_plots(final_volume_data)

    # check if cycles have been split and interpolated, if not, do this
    file_count = sum(1 for s in os.listdir(f"{output_dir}excel/") if 'cycles_interpolated' in s)
    if file_count != 3:
        interpolated_dataframes = split_cycles_and_interpolate(final_volume_data)
    else:
        print("Interpolation has already been performed. Output files exist. Loading them.")
        interpolated_dataframes = [
            pd.read_excel(f"{output_dir}excel/cycles_interpolated_Cell_volumes.xlsx"),
            pd.read_excel(f"{output_dir}excel/cycles_interpolated_Nucleus_volumes.xlsx"),
            pd.read_excel(f"{output_dir}excel/cycles_interpolated_NC_ratios.xlsx")
        ]

    # generate a plot per interpolated cycle
    generate_interpolated_cycle_plots(interpolated_dataframes[0], interpolated_dataframes[1],
                                      interpolated_dataframes[2])

    # average the interpolated data and plot the result (cell volume, nucleus volume and N/C ratio)
    generate_averaged_plots(interpolated_dataframes[0], interpolated_dataframes[1], interpolated_dataframes[2])

    toc = time.perf_counter()
    secs = round(toc - tic, 4)
    print(f"Done. Runtime was {secs} seconds ({round(secs / 60, 2)} minutes)")


# SCRIPT STARTS HERE
if __name__ == "__main__":
    main()
    sys.exit(0)
