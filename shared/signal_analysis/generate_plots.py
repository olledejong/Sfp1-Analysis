import matplotlib.pyplot as plt
import numpy as np

from shared.shared_functions import save_figure

plt.style.use('seaborn-v0_8')

def dynamics_overview_all_cells(final_data, kario_events, budding_events, output_dir, channel_name):
    variables = [f"{channel_name}_total", f"{channel_name}_nucleus", f"{channel_name}_cyto", f"{channel_name}_ratio"]
    colors = ["grey", "green", "red", "blue"]

    individual_cells = sorted(list(set(final_data["Cell_pos"])))

    L = len(variables)

    fig = plt.figure(1, (len(individual_cells) * 11, 7 * L), dpi=16)

    cell_count = 1
    for cell in individual_cells:
        # extracting data corresponding to one cell
        ss_data = final_data[final_data["Cell_pos"] == cell]
        ss_data = ss_data.sort_values(by="Time")

        # extracting time column
        x = np.array(ss_data["Time"])

        for i in range(L):
            ax = plt.subplot(L, len(individual_cells), cell_count + i * len(individual_cells))

            if cell in budding_events:
                for b in budding_events[cell]:
                    if len(ss_data[ss_data["TimeID"] == b]["Time"]) == 0: continue  # TimeID in event txt file but not in budj data
                    budding_time = float(ss_data[ss_data["TimeID"] == b]["Time"].item())
                    ax.axvline(budding_time, color='purple', linewidth=2, linestyle='-')

            if cell in kario_events:
                for b in kario_events[cell]:
                    if len(ss_data[ss_data["TimeID"] == b]["Time"]) == 0: continue  # TimeID in event txt file but not in budj data
                    kario_time = float(ss_data[ss_data["TimeID"] == b]["Time"].item())
                    ax.axvline(kario_time, color='black', linewidth=2, linestyle='-')

            # plotting the variable
            y = ss_data[variables[i]]
            plt.plot(x, y, 'o-', color=colors[i], linewidth=1)
            plt.xlim(0, 700)
            plt.xlabel("Time, min")
            ax.set_ylabel(variables[i])

            if i == 0:
                plt.title(cell)
        cell_count += 1

    save_figure(f"{output_dir}plots/individual_cells_Sfp1_2_PKA2D_agar_b2b_{channel_name}_only.pdf", bbox_inches='tight')


def separate_interpolated_cycles(interpolated_dataframes, output_dir, channel_name):
    """
    This function takes the data of the three main datatypes of interest and creates a plot of each one for every cycle.
    :param interpolated_dataframes:
    :param output_dir:
    :param channel_name:
    :return:
    """
    dataframes = {
        f"Total {channel_name}": interpolated_dataframes[f"{channel_name}_total"],
        f"Nuclear {channel_name}": interpolated_dataframes[f"{channel_name}_nucleus"],
        f"Cytoplasmic {channel_name}": interpolated_dataframes[f"{channel_name}_cyto"],
        "NC Ratio": interpolated_dataframes[f"{channel_name}_ratio"],
    }
    progress = 1
    for dataframe in dataframes:
        count = 0
        while count < len(dataframes[dataframe]):
            print(f"Generating interpolated cycle plots ..", end="\r", flush=True)
            single_cycle_data = dataframes[dataframe].iloc[count]
            cell_name = single_cycle_data['cell']
            cycle_id = single_cycle_data['cycle']

            plt.plot(single_cycle_data[5:].values, 'r', linewidth=3)
            plt.title(f"{cell_name}, cycle {cycle_id}, {dataframe} (interpolated)", fontstyle='italic', y=1.02)
            plt.xlabel("Time")
            plt.ylabel(dataframe)
            save_figure(
                f"{output_dir}plots/separate_cell_plots/interpolated_cycles/{dataframe}/{cell_name}_cycle{cycle_id}.png"
            )
            count += 1
            progress += 1
    print(f"Generating interpolated cycle plots .. Done!")


def averaged_plots(interpolated_dataframes, output_dir, channel_name):
    """
    Takes the interpolated data files (Excel files) and averages these. The averages are plotted over time and
    saved to the user's system.
    :return:
    """
    print("Generating the averaged interpolated data plots..", end="\r", flush=True)

    nuclear_averages = interpolated_dataframes[f"{channel_name}_nucleus"].mean(axis=0, numeric_only=True)
    cyto_averages = interpolated_dataframes[f"{channel_name}_cyto"].mean(axis=0, numeric_only=True)
    to_plot = {
        f"Total {channel_name}": interpolated_dataframes[f"{channel_name}_total"].mean(axis=0, numeric_only=True),
        f"Nuclear {channel_name}": nuclear_averages,
        f"Cytoplasmic {channel_name}": cyto_averages,
        f"{channel_name} nuclear-to-cytosolic ratio": nuclear_averages / cyto_averages,
    }
    t_span = np.linspace(0, 100, 99)

    for item in to_plot:
        averages = to_plot[item][4:].values
        polfit = np.polyfit(t_span / 100, averages, 10)
        poly_y = np.polyval(polfit, t_span / 100)

        plt.plot(t_span / 100, averages, c='grey', lw=3, alpha=0.6, label=f"Raw average")
        plt.plot(t_span / 100, poly_y, c='darkred', lw=4, alpha=0.8, label=f"Polynomial average")
        plt.title(f"Average Sfp1 {item} signal over the cell cycle", fontstyle='italic', y=1.02)
        plt.xlabel("Cell cycle progression")
        plt.ylabel(item)
        plt.legend()
        save_figure(f"{output_dir}plots/interpolated_averaged/{item}.png")

    print("Generating the averaged interpolated data plots.. Done!")


def combined_channel_data(final_data, output_dir, channel_name):
    """
    Creates a separate plot for every cell that displays the nuclear and whole-cell volumes
    together. This might give a good overview on the data quality per cell.
    :param output_dir:
    :param channel_name:
    :param final_data:
    :return:
    """
    individual_cells = sorted(list(set(final_data["Cell_pos"])))  # amount of cells

    for cell in individual_cells:
        progress = round(individual_cells.index(cell) / len(individual_cells) * 100)
        print(f"Generating combined nuc/cyt signal plot per cell ( {progress}% )", end="\r", flush=True)

        fig, axs = plt.subplots(2, figsize=(5, 5))
        fig.supxlabel("Time (frames)")
        fig.supylabel(f"{channel_name} signal")
        fig.subplots_adjust(hspace=0.35, wspace=0.4)
        fig.suptitle(f"{cell} - Cytoplasmic and nuclear signal over time")

        single_cell_data = final_data[final_data.Cell_pos == cell]
        single_cell_data = single_cell_data[~single_cell_data[f"{channel_name}_nucleus"].isnull()]

        axs[0].plot(single_cell_data.TimeID, single_cell_data[f"{channel_name}_cyto"], 'r')
        axs[0].set_title(f"Cytoplasmic {channel_name} signal", fontstyle='italic', y=1.02)
        axs[1].plot(single_cell_data.TimeID, single_cell_data[f"{channel_name}_nucleus"], 'y')
        axs[1].set_title(f"Nuclear {channel_name} signal", fontstyle='italic', y=1.02)

        save_figure(f"{output_dir}/plots/separate_cell_plots/{cell}_signal_overT.png")