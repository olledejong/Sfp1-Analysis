import os
import re
import pandas as pd
import numpy as np
from math import pi, cos, sin
from skimage.io import imread

scaling_factor = 0.16  # microns per pixel ---> 100x objective

def round_up_to_odd(f):
    """
    Rounds up any number to a whole odd integer
    """
    return int(np.ceil(f) // 2 * 2 + 1)


def read_images(tiff_files_dir):
    """
    Reads the images from the 'tiff_files_dir' directory. These are stored in a dictonary and returned.
    :return:
    """
    print("Reading tiff images..", end="\r", flush=True)
    images = {}
    for file in os.listdir(tiff_files_dir):
        if file.endswith(".tif"):
            pos = re.findall("(?<=xy)[0-9]{2}", file)
            if not pos: pos = re.findall("(?<=ser)[0-9]{2}", file)
            images[pos[0]] = imread(f"{tiff_files_dir}{file}")  # load the image
    print("Reading tiff images.. Done!")
    return images


def get_time_conversion():
    TimeIDs = 150  # maximum amount of timepoints for a single cell
    time_step = 5  # every time-point equals 5 real time minutes

    return pd.DataFrame({
        "TimeID": range(1, TimeIDs + 1),
        "Time": [x * time_step for x in TimeIDs]
    })


def load_budj_files(data_dir):
    # find all the files that hold budj data
    files = []
    for filename in os.listdir(data_dir):
        if "Cell data .csv" in filename:
            pos = re.findall("(?<=xy)[0-9]{2}", filename)
            if not pos: pos = re.findall("(?<=ser)[0-9]{2}", filename)
            cell_name = f"pos{pos[0]}_"
            files.append((filename, cell_name))
    return files


def load_all_budj_data(data_dir):
    """
    Collects all file names and using that, all BudJ data is loaded from the files
    :return:
    """
    print("Loading all BudJ excel files..", end="\r", flush=True)
    files = load_budj_files(data_dir)  # get the separate files

    # add the data of all files to one single dataframe
    budj_data = pd.DataFrame({})
    for filename, cell_name_prefix in files:
        # mother with daughter(s) data
        pos_data = pd.read_csv(data_dir + filename, header=0, index_col=0)
        pos_data["Cell_pos"] = cell_name_prefix + pos_data["Cell"].map(str)
        temp_data = pd.DataFrame(columns=pos_data.columns)
        # name the cells correctly, the mother's name is unchanged, but the daughters are now recognizable
        for index, row in pos_data.iterrows():
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


def ellipse_from_budj(t, cell_data):
    """
    Define the function that can extract parameters of the ellipse using the data
    Function to extract parameters of the ellipse from the BudJ table
    """
    data_at_frame = cell_data[cell_data["TimeID"] == t]
    x_pos = float(data_at_frame['x'].iloc[0]) / scaling_factor
    y_pos = float(data_at_frame['y'].iloc[0]) / scaling_factor
    majorR = float(data_at_frame["Major R"].iloc[0]) / scaling_factor
    minorR = float(data_at_frame["Minor r"].iloc[0]) / scaling_factor
    angle = float(data_at_frame['Angle'].iloc[0]) * (pi / 180)  # convert to radians
    return x_pos, y_pos, majorR, minorR, angle


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

    y_dim, x_dim = image_shape  # get the dimensions

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