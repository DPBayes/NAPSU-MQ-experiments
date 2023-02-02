import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def value_count_list(df, col, type):
    counts = df[col].value_counts()
    return [(value, counts[value], type) for value in counts.index]

def value_count_df(orig_df, syn_dfs, col):
    vl_list = value_count_list(orig_df, col, "Original")
    for syn_df in syn_dfs:
        vl_list += (value_count_list(syn_df, col, "Synthetic"))

    return pd.DataFrame.from_records(vl_list, columns=["Value", "Count", "Type"])

def plot_dataframe_marginal_comparison(orig_df, syn_dfs, filename=None, estimator=np.mean):
    """Plot a comparison of 1-way marginals between original and synthetic data.

    Args:
        orig_df (DataFrame): Original dataframe.
        syn_dfs (list(DataFrame)): List of synthetic dataframes.
        filename (str, optional): Filename to save plot. Defaults to None.
        estimator (function, optional): Aggregation function for synthetic data values. Defaults to np.mean.
    """
    value_count_dfs = { col: value_count_df(orig_df, syn_dfs, col) for col in orig_df.columns }

    fig, axes = plt.subplots(1, len(value_count_dfs.items()), figsize=(3 * len(value_count_dfs.items()), 5.5))
    for i, (key, value) in enumerate(value_count_dfs.items()):
        ax = axes[i]
        ax.set_title(key)
        ax.tick_params("x", labelrotation=90)
        sns.barplot(x="Value", y="Count", data=value_count_dfs[key], hue="Type", estimator=estimator, ax=ax)
        if i < len(value_count_dfs.items()) - 1: 
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.00, 1))

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()