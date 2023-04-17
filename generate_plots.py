import os
import matplotlib.pyplot as plt
from volume_analysis_script import output_dir

plt.style.use('seaborn-v0_8')


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

    save_path = os.path.join(directory, filename)

    # Actually save the figure
    plt.savefig(save_path, bbox_inches=bbox_inches, dpi=dpi)
    plt.close()


def combined_volumes(final_volume_data):
    """
    Creates a separate plot for every cell that displays the nuclear and whole-cell volumes
    together. This might give a good overview on the data quality per cell.
    :param final_volume_data:
    :return:
    """
    individual_cells = sorted(list(set(final_volume_data["Cell_pos"])))  # amount of cells

    for cell in individual_cells:
        progress = round(individual_cells.index(cell) / len(individual_cells) * 100)
        print(f"Generating volume plot per cell ( {progress}% )", end="\r", flush=True)

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


def interpolated_cycles(interpolated_dataframes):
    """
    This function takes the data of the three main datatypes of interest and creates a plot of each one for every cycle.
    :param interpolated_dataframes:
    :return:
    """
    dataframes = [
        ("Cell area", interpolated_dataframes[0], "Cell area (µm\u00b2)"),
        ("Cell volume", interpolated_dataframes[1], "Cell volume (µm\u00b3)"),
        ("Nucleus area", interpolated_dataframes[2], "Nucleus area (µm\u00b2)"),
        ("Nucleus volume", interpolated_dataframes[3], "Nucleus volume (µm\u00b3)"),
        ("NC ratio", interpolated_dataframes[4], "N/C ratio")
    ]
    progress = 1
    runs = sum(len(df) for df in interpolated_dataframes)
    for interpolated_dataframe in dataframes:
        count = 0
        data_type = interpolated_dataframe[0]
        while count < len(interpolated_dataframe[1]):
            print(f"Generating interpolated cycle plots ( {round(progress / runs * 100)}% )", end="\r", flush=True)
            single_cycle_data = interpolated_dataframe[1].iloc[count]
            cell_name = single_cycle_data['cell']
            cycle_id = single_cycle_data['cycle']

            plt.plot(single_cycle_data[5:].values, 'r', linewidth=3)
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


def averaged_plots(interpolated_dataframes):
    """
    Takes the interpolated data files (Excel files) and averages these. The averages are plotted over time and
    saved to the user's system.
    :return:
    """
    print("Generating the averaged interpolated data plots..", end="\r", flush=True)
    # keep only the cycles that include daughter data

    to_plot = [
        ("Cell area", interpolated_dataframes[0].mean(axis=0, numeric_only=True), "µm\u00b2"),
        ("Cell volume", interpolated_dataframes[1].mean(axis=0, numeric_only=True), "µm\u00b3"),
        ("Nuclear area", interpolated_dataframes[2].mean(axis=0, numeric_only=True), "µm\u00b2"),
        ("Nuclear volume", interpolated_dataframes[3].mean(axis=0, numeric_only=True), "µm\u00b3"),
        ("N/C ratio", interpolated_dataframes[4].mean(axis=0, numeric_only=True), None)
    ]

    for item in to_plot:
        data_type = item[0]
        unit = item[2]

        plt.plot(item[1][4:].values, 'r', linewidth=3)
        plt.title(f"Average {data_type} over time", fontstyle='italic', y=1.02)
        plt.xlabel("Time")
        if unit is not None:
            plt.ylabel(f"{data_type} ({unit})")
        else:
            plt.ylabel(data_type)
        filename = "NC ratio" if data_type == "N/C ratio" else data_type
        save_figure(f"{output_dir}plots/interpolated_averaged/{filename}.png")

    generate_combined_volumes_plot(interpolated_dataframes[1], interpolated_dataframes[3])
    print("Generating the averaged interpolated data plots.. Done!")

