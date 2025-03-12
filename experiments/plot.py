import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_results(results_dir, dataset_name, d_values):
    """
    Reads JSON files containing error metrics and plots results across dimensions with confidence intervals.

    Args:
    - results_dir: Path to the directory containing JSON files.
    - dataset_name: Name of the dataset (used for filenames).
    - d_values: List of dimensions to include in the plots.
    """

    json_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")]

    results_dict = {}
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            d_key = int(json_file.split("_d")[1].split("_")[0])  # Extract dimension from filename
            results_dict[d_key] = data

    metrics = ["mahalanobis", "l2", "mse"]
    for metric in metrics:
        plt.figure(figsize=(8, 6))

        maxent_means, maxent_stds, ssp_means, ssp_stds = [], [], [], []

        for d in d_values:
            if d in results_dict:
                print(results_dict)
                maxent_data = np.array(results_dict[d][f"MaxEnt_{metric}"])
                ssp_data = np.array(results_dict[d][f"SSP_{metric}"])

                maxent_means.append(np.mean(maxent_data))
                maxent_stds.append(np.std(maxent_data))
                ssp_means.append(np.mean(ssp_data))
                ssp_stds.append(np.std(ssp_data))

        plt.errorbar(d_values, maxent_means, yerr=maxent_stds, fmt='-o', label='MaxEnt', color='blue', capsize=5)
        plt.errorbar(d_values, ssp_means, yerr=ssp_stds, fmt='-o', label='SSP', color='orange', capsize=5)

        plt.xlabel('d')
        plt.ylabel(f'{metric.upper()} Error')
        plt.title(f'MaxEnt vs SSP {metric.upper()} Error Across Dimensions')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        plot_filename = os.path.join(results_dir, f"{dataset_name}_{metric}_plot.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")
        plt.close()


results_dir = "results"
dataset_name = "synthdata"
d_values = [2, 5, 10, 20, 50]
plot_results(results_dir, dataset_name, d_values)