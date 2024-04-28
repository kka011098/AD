# plotting_script.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

def get_index_to_cut(column_index, cut, array):
    """Given an array column index and a threshold, this function returns the index of the
        entries not passing the threshold.

    Args:
        column_index (int): The index for the column where cuts should be applied
        cut (float): Threshold for which values below will have the whole entry removed
        array (np.array): The full array to be edited

    Returns:
        _type_: returns the index of the rows to be removed
    """
    indices_to_cut = np.argwhere(array[column_index] < cut).flatten()
    return indices_to_cut

def plot_box_and_whisker(names, residual, pdf):
    """Plots Box and Whisker plots of 1D data

    Args:
        project_path (string): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
    """
    column_names = [i.split(".")[-1] for i in names]

    fig1, ax1 = plt.subplots()

    boxes = ax1.boxplot(list(residual), showfliers=False, vert=False)
    whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
    edges = max([abs(min(whiskers)), max(whiskers)])

    ax1.set_yticks(np.arange(1, len(column_names) + 1, 1))
    ax1.set_yticklabels(column_names)

    ax1.grid()
    fig1.tight_layout()
    ax1.set_xlabel("Residual")
    ax1.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
    pdf.savefig()

def plot_1D(output_path: str, file_path: str):
    """General plotting for 1D data, for example data from a '.csv' file. This function generates a pdf
        document where each page contains the before/after performance
        of each column of the 1D data

    Args:
        output_path (path): The path to the project directory
        file_path (str): The path to the NPZ file containing the data
    """

    before_path = file_path
    after_path = file_path

    before = np.transpose(np.load(before_path, allow_pickle=True)["data"])
    after = np.transpose(np.load(after_path, allow_pickle=True)["data"])
    names = np.load(file_path, allow_pickle=True)["names"]

    index_to_cut = get_index_to_cut(3, 1e-6, before)
    before = np.delete(before, index_to_cut, axis=1)
    after = np.delete(after, index_to_cut, axis=1)

    response = np.divide(np.subtract(after, before), before) * 100
    residual = np.subtract(after, before)

    with PdfPages(os.path.join(output_path, "plotting", "comparison.pdf")) as pdf:
        plot_box_and_whisker(names, residual, pdf)
        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1])

        axsLeft = subfigs[0].subplots(2, 1, sharex=True)
        ax1 = axsLeft[0]
        ax3 = axsLeft[1]
        axsRight = subfigs[1].subplots(2, 1, sharex=False)
        ax2 = axsRight[0]
        ax4 = axsRight[1]

        number_of_columns = len(names)

        print("=== Plotting ===")

        for index, column in enumerate(tqdm(names)):
            column_name = column.split(".")[-1]
            rms = np.sqrt(np.mean(np.square(response[index])))
            residual_RMS = np.sqrt(np.mean(np.square(residual[index])))

            x_min = min(before[index] + after[index])
            x_max = max(before[index] + after[index])
            x_diff = abs(x_max - x_min)

            # Before Histogram
            counts_before, bins_before = np.histogram(
                before[index],
                bins=np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, 200),
            )
            ax1.hist(
                bins_before[:-1], bins_before, weights=counts_before, label="Before"
            )

            # After Histogram
            counts_after, bins_after = np.histogram(
                after[index],
                bins=np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, 200),
            )
            ax1.hist(
                bins_after[:-1],
                bins_after,
                weights=counts_after,
                label="After",
                histtype="step",
            )

            ax1.set_ylabel("Counts", ha="right", y=1.0)
            ax1.set_yscale("log")
            ax1.legend(loc="best")
            ax1.set_xlim(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff)
            ax1.set_ylim(ymin=1)

            data_bin_centers = bins_after[:-1] + (bins_after[1:] - bins_after[:-1]) / 2
            ax3.scatter(
                data_bin_centers, (counts_after - counts_before), marker="."
            )  # FIXME: Dividing by zero
            ax3.axhline(y=0, linewidth=0.2, color="black")
            ax3.set_xlabel(f"{column_name}", ha="right", x=1.0)
            ax3.set_ylim(
                -max(counts_after - counts_before)
                - 0.05 * max(counts_after - counts_before),
                max(counts_after - counts_before)
                + 0.05 * max(counts_after - counts_before),
            )
            ax3.set_ylabel("Residual")

            # Response Histogram
            counts_response, bins_response = np.histogram(
                response[index], bins=np.arange(-20, 20, 0.1)
            )
            ax2.hist(
                bins_response[:-1],
                bins_response,
                weights=counts_response,
                label="Response",
            )
            ax2.axvline(
                np.mean(response[index]),
                color="k",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean {round(np.mean(response[index]),4)} %",
            )
            ax2.plot([], [], " ", label=f"RMS: {round(rms,4)} %")

            ax2.set_xlabel(f"{column_name} Response [%]", ha="right", x=1.0)
            ax2.set_ylabel("Counts", ha="right", y=1.0)
            ax2.legend(loc="best", bbox_to_anchor=(1, 1.05))

            # Residual Histogram
            counts_residual, bins_residual = np.histogram(
                residual[index], bins=np.arange(-1, 1, 0.01)
            )
            ax4.hist(
                bins_residual[:-1],
                bins_residual,
                weights=counts_residual,
                label="Residual",
            )
            ax4.axvline(
                np.mean(residual[index]),
                color="k",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean {round(np.mean(residual[index]),6)}",
            )
            ax4.plot([], [], " ", label=f"RMS: {round(residual_RMS,6)}")
            ax4.plot([], [], " ", label=f"Max: {round(max(residual[index]),6)}")
            ax4.plot([], [], " ", label=f"Min: {round(min(residual[index]),6)}")

            ax4.set_xlabel(f"{column_name} Residual", ha="right", x=1.0)
            ax4.set_ylabel("Counts", ha="right", y=1.0)
            ax4.set_xlim(-1, 1)
            ax4.legend(loc="best", bbox_to_anchor=(1, 1.05))

            pdf.savefig()
            ax2.clear()
            ax1.clear()
            ax3.clear()
            ax4.clear()
