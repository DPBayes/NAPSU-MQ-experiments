import numpy as np 
from scipy import stats
from scipy import integrate

def overlap(orig_conf_ints, syn_conf_ints):
    """Compute the overlaps between two sets of confidence intervals.

    Args:
        orig_conf_ints (ndarray): Original confidence intervals as (n x 2) array.
        syn_conf_ints (ndarray): Synthetic confidence intervals as (n x 2) array-

    Returns:
        (ndarray, ndarray): Starting and ending points of the overlaps. 
    """
    overlap_starts = np.max([orig_conf_ints[:,0], syn_conf_ints[:,0]], axis=0)
    overlap_ends = np.min([orig_conf_ints[:,1], syn_conf_ints[:,1]], axis=0)
    return overlap_starts, overlap_ends

def interval_overlap(orig_conf_ints, syn_conf_ints):
    """ Compute the interval overlap metrics.

    Args:
        orig_conf_ints (ndarray): Original confidence intervals as (n x 2) array.
        syn_conf_ints (ndarray): Synthetic confidence intervals as (n x 2) array.

    Returns:
        (ndarray, ndarray): The interval overlap metrics.
    """
    overlap_starts, overlap_ends = overlap(orig_conf_ints, syn_conf_ints)
    overlap_metric_1 = np.clip((overlap_ends - overlap_starts) / (orig_conf_ints[:,1] - orig_conf_ints[:,0]), a_min=0, a_max=None)
    overlap_metric_2 = np.clip((overlap_ends - overlap_starts) / (syn_conf_ints[:, 1] - syn_conf_ints[:, 0]), a_min=0, a_max=None)
    return overlap_metric_1, overlap_metric_2

def interval_overlap_weighted(orig_distributions, syn_distributions, conf_level):
    """Compute weighted interval overlap metrics.

    Args:
        orig_distributions (list(function)): Original distribution density functions.
        syn_distributions (list(function)): Synthetic distribution density functions.
        conf_level (float): Confidence level.

    Returns:
        (ndarray, ndarray): Weighted interval overlap metrics.
    """
    overlap_metrics_1 = np.zeros(len(orig_distributions))
    overlap_metrics_2 = np.zeros(len(orig_distributions))

    for i, (orig_distribution, syn_distribution) in enumerate(zip(orig_distributions, syn_distributions)):
        orig_conf_int = orig_distribution.interval(conf_level)
        syn_conf_int = syn_distribution.interval(conf_level)

        overlap_metric_1 = orig_distribution.cdf(syn_conf_int[1]) - orig_distribution.cdf(syn_conf_int[0])
        overlap_metric_2 = syn_distribution.cdf(orig_conf_int[1]) - syn_distribution.cdf(orig_conf_int[0])
        overlap_metrics_1[i] = overlap_metric_1
        overlap_metrics_2[i] = overlap_metric_2

    return overlap_metrics_1, overlap_metrics_2

def interval_kldivergence(orig_distributions, syn_distributions):
    """Compute KL-divergence between original and synthetic CI distribution.

    Args:
        orig_distributions (list(function)): Original distribution density functions.
        syn_distributions (list(function)): Synthetic distribution density functions.

    Returns:
        ndarray: KL-divergences.
    """
    results = np.zeros(len(orig_distributions))
    for i, (orig_distribution, syn_distribution) in enumerate(zip(orig_distributions, syn_distributions)):
        p = orig_distribution
        q = syn_distribution
        results[i] = integrate.quad(lambda x: p.pdf(x) * (p.logpdf(x) - q.logpdf(x)), -np.inf, np.inf)[0]
    return results

def statsmodels_result_to_distributions(result):
    """Convert statsmodels result object to a corresponding density functions.

    Args:
        result (statsmodels result object): The result object.

    Returns:
        list(function): The density functions.
    """
    means = result.params
    scales = result.bse
    distributions = []
    for i in range(len(means)):
        if result.use_t:
            dof = result.df_resid_inference if result.df_resid_inference is not None else result.df_resid
            distributions.append(stats.t(loc=means[i], scale=scales[i], df=dof))
        else:
            distributions.append(stats.norm(loc=means[i], scale=scales[i]))
    return distributions

    