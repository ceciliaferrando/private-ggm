import os
import json
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

# def plot_results(results_dir, dataset_name, d_values):
#     """
#     Reads JSON files containing error metrics and plots results across dimensions with confidence intervals.
#
#     Args:
#     - results_dir: Path to the directory containing JSON files.
#     - dataset_name: Name of the dataset (used for filenames).
#     - d_values: List of dimensions to include in the plots.
#     """
#
#     json_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")]
#
#     results_dict = {}
#     for json_file in json_files:
#         with open(json_file, "r") as f:
#             data = json.load(f)
#             d_key = int(json_file.split("_d")[1].split("_")[0])  # Extract dimension from filename
#             results_dict[d_key] = data
#
#     metrics = ["mahalanobis", "l2", "mse"]
#     for metric in metrics:
#         plt.figure(figsize=(8, 6))
#
#         maxent_means, maxent_stds, ssp_means, ssp_stds = [], [], [], []
#
#         for d in d_values:
#             if d in results_dict:
#
#                 maxent_data = np.array(results_dict[d][f"MaxEnt_{metric}"])
#                 ssp_data = np.array(results_dict[d][f"SSP_{metric}"])
#
#                 maxent_means.append(np.mean(maxent_data))
#                 maxent_stds.append(np.std(maxent_data))
#                 ssp_means.append(np.mean(ssp_data))
#                 ssp_stds.append(np.std(ssp_data))
#
#         plt.errorbar(d_values, maxent_means, yerr=maxent_stds, fmt='-o', label='MaxEnt', color='blue', capsize=5)
#         plt.errorbar(d_values, ssp_means, yerr=ssp_stds, fmt='-o', label='SSP', color='orange', capsize=5)
#
#         plt.xlabel('d')
#         plt.ylabel(f'{metric.upper()} Error')
#         plt.title(f'MaxEnt vs SSP {metric.upper()} Error Across Dimensions')
#         plt.yscale('log')
#         plt.legend()
#         plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#
#         plot_filename = os.path.join(results_dir, f"{dataset_name}_{metric}_plot.png")
#         plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to {plot_filename}")
#         plt.close()


def plot_results(results_dir, dataset_name, x_values, x_axis="d"):
    """
    Reads JSON files containing error metrics and plots results across dimensions or rho values
    with empirical error bands using seaborn.

    Args:
    - results_dir: Path to the directory containing JSON files.
    - dataset_name: Name of the dataset (used for filenames).
    - x_values: List of values (either dimensions `d` or `rho`) to include in the plots.
    - x_axis: The variable to use on the x-axis. Either "d" or "rho".
    """

    # Load JSON files from the results directory
    json_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")]
    results_dict = {}

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

            # Extract x-axis variable from filename
            if x_axis == "d":
                key = int(json_file.split("_d")[1].split("_")[0])  # Extract dimension d
            elif x_axis == "rho":
                key = float(json_file.split("_rho")[2].split("_")[0])  # Extract rho
            else:
                raise ValueError("x_axis must be either 'd' or 'rho'.")

            results_dict[key] = data

    metrics = ["mahalanobis", "l2", "mse"]
    for metric in metrics:
        # Create a list to store raw data for seaborn
        data_list = []
        for x in x_values:
            if x in results_dict:
                for method in ['MaxEnt', 'SSP']:
                    # Get the list of error measurements for the current method and metric
                    values = results_dict[x][f"{method}_{metric}"]
                    # Append each measurement along with its associated x value and method
                    for val in values:
                        data_list.append({
                            x_axis: x,
                            'Error': val,
                            'Method': method
                        })

        # Create a DataFrame from the raw data
        df = pd.DataFrame(data_list)

        # Set up the figure
        plt.figure(figsize=(8, 6))
        # Use seaborn lineplot to automatically compute and plot the confidence interval bands.
        sns.lineplot(data=df, x=x_axis, y="Error", hue="Method", errorbar=('ci', 95), marker="o")

        plt.xlabel(x_axis)
        plt.ylabel(f'{metric.upper()} Error')
        plt.title(f'MaxEnt vs SSP {metric.upper()} error across {x_axis.lower()}')
        plt.yscale('log')
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Save the plot
        plot_filename = os.path.join(results_dir, f"{dataset_name}_{metric}_vs_{x_axis}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")
        plt.close()


results_dir = "results/rho1off50seed/"
dataset_name = "synthdata"
d_values = [2, 5, 10, 20, 30, 40, 50]
rho_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
plot_results(results_dir, dataset_name, d_values, x_axis="d")