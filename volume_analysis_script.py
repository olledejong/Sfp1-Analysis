import os
import re
from math import pi, sin, cos
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import binary_erosion
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
    r1c = (major_r * scaling_factor)  # to get the semi-major axis of the whole cell
    r2c = (minor_r * scaling_factor)  # to get the semi-minor axis of the whole cell
    r3c = (np.average((major_r, minor_r)) * scaling_factor)  # to get the third axis
    volume = (4 / 3) * pi * r1c * r2c * r3c
    area = pi * r1c * r2c
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


def get_volume_data(budj_data, individual_cells):
    """

    :param budj_data:
    :param individual_cells:
    :return:
    """
    final_dataframe = pd.DataFrame({})
    for pos in range(1, 21):
        if pos < 10:
            pos = "0" + str(pos)
        pos = str(pos)
        image = imread(os.path.join(f"{tiff_files_dir}2022_12_06_nup133_yegfp_xy{pos}.nd2.tif"))  # load the image
        imageGFP = image[:, 1, :, :]  # get the GFP data

        for cell in [i for i in individual_cells if pos in i]:
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
            final_dataframe = pd.concat([final_dataframe, single_cell_data])
            print(f"{cell} - Couldn't fit ellipse {could_not_fit} out of {len(single_cell_data['TimeID'])} times")

    # remove unrealistic N/C ratios
    final_dataframe = final_dataframe.drop(final_dataframe[final_dataframe["N/C_ratio"] > .32].index)
    final_dataframe = final_dataframe.reset_index(drop=True)  # reset index of dataframe
    final_dataframe.to_excel(f"{output_dir}excel/nup133_final_data_script_removed_outliers.xlsx")  # save the final file
    return final_dataframe


def split_cycles_and_interpolate(final_dataframe, individual_cells, kario_events):
    """

    :param final_dataframe:
    :param individual_cells:
    :param kario_events:
    :return:
    """
    desired_datapoints = 100
    cols = ["cell", "cycle", "start", "end"]
    cols += range(desired_datapoints)
    data_interpolated = pd.DataFrame(columns=cols)

    for cell in individual_cells[1:]:  # remove cell pos01_1, since it has too many missing frames
        single_cell_data = final_dataframe[final_dataframe.Cell_pos == cell]
        single_cell_data = single_cell_data[~single_cell_data.Nucleus_volume.isnull()]

        # print(f"Kario events for {cell}: {kario_events[cell]}")
        count = 0
        while count < len(kario_events[cell]) - 1:
            tp1 = kario_events[cell][count]
            tp2 = kario_events[cell][count + 1]
            # print(f"{cell} - Getting N/C ratio data from {tp1} through {tp2}")
            cell_cycle_dat = single_cell_data[single_cell_data.TimeID.between(tp1, tp2)]

            # when there are not enough datapoints for this cycle, because of outlier cleaning, then ignore that cycle
            if len(cell_cycle_dat) < 10:
                count += 1
                continue

            # get the original x and y data
            x = np.linspace(0, len(cell_cycle_dat), len(cell_cycle_dat))
            y = cell_cycle_dat['N/C_ratio']

            # interpolate the data towards 100 datapoints
            x_new = np.linspace(0, len(cell_cycle_dat), 100)
            f1 = interpolate.interp1d(x, y, kind='linear')
            f2 = interpolate.interp1d(x, y, kind='cubic')

            # add the interpolated data to a dataframe
            data_interpolated.loc[len(data_interpolated)] = [cell, count + 1, tp1, tp2] + f2(x_new).tolist()
            count += 1

    data_interpolated.to_excel(f"{output_dir}excel/cycles_interpolated_script_removed_outliers.xlsx")  # save the final file
    return data_interpolated


def main():
    budj_data = load_all_budj_data()  # load the budj data from all the separate files
    budding_events, kario_events = load_events()  # load the kyrokinesis and budding events
    individual_cells = sorted(list(set(budj_data["Cell_pos"])))  # how many cells are there in total

    final_dataframe = get_volume_data(budj_data, individual_cells)
    interpolated_data = split_cycles_and_interpolate(final_dataframe, final_dataframe, kario_events)
    print("Done.")


# SCRIPT STARTS HERE
if __name__ == "__main__":
    main()
