import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def plot_workload_comparison(workload, workload_values, syn_df_query_values, xscale=None, xlim=None):
    """Plot a comparison of workload query values between original and synthetic data.

    Args:
        workload (QueryList): Workload queries.
        workload_values (torch.tensor): Values of workload queries for original data.
        syn_df_query_values (list(torch.tensor)): Workload query values for synthetic data.
        xscale (str, optional): Scale of the x-axis. Defaults to None.
        xlim ((float, float), optional): X-axis limits. Defaults to None.
    """
    suff_stat_by_query = pd.DataFrame.from_records([
        (
            "{} = {}".format(query.inds, query.value), 
            workload_values[i].item(),
            "Real Data"
        ) for i, query in enumerate(workload.queries)
    ] + [
        (
            "{} = {}".format(query.inds, query.value), 
            syn_df_query_value[i].item(),
            "Synthetic Data"
        )
        for i, query in enumerate(workload.queries)
        for syn_df_query_value in syn_df_query_values
    ], columns=["Query", "Count", "Type"])

    sns.catplot(x="Count", y="Query", hue="Type", kind="bar", data=suff_stat_by_query)
    if xscale is not None: plt.xscale(xscale)
    if xlim is not None: plt.xlim(xlim)
    plt.show()