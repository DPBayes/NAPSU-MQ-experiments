import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confidence_interval_comparison(algo_cis, variable_names, filename=None, xlim=None):
    """Plot a comparison of confidence intervals for several methods.

    Args:
        algo_cis (list(tuple)): List of (algorithm, confidence intervals).
        variable_names (list(str)): Variable names.
        filename (str, optional): Filename to save plot. Defaults to None.
        xlim ((float, float), optional): X-axis limits for plot. Defaults to None.
    """
    algo_ci_centers = [(algo, cis.mean(axis=1)) for algo, cis in algo_cis]
    algo_ci_widths = [(algo, cis[:,1] - cis[:,0]) for algo, cis in algo_cis]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    num_coefficients = len(variable_names)
    num_algos = len(algo_cis)
    plt.subplots(figsize=(15, 0.5 * num_algos * num_coefficients))
    for i in range(num_coefficients):
        variable_name = variable_names[i]
        for j in range(len(algo_cis)):
            center = algo_ci_centers[j][1][i]
            radius = algo_ci_widths[j][1][i] * 0.5
            algo = algo_cis[j][0]
            plt.errorbar(center, y=["{} - {}".format(algo, variable_name)], xerr=radius, fmt=".", color=colors[j])

    plt.axvline(0, linestyle="dashed", color="black")
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()