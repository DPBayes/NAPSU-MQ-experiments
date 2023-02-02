import warnings
import numpy as np
import scipy.stats as stats

def compute_aggregates(q, u):
    """Compute aggregates for Rubin's rules from point and variance estimates.

    Args:
        q (ndarray): Point estimates.
        u (ndarray): Variance estimates.

    Returns:
        tuple: (q_m, b_m, u_m).
    """
    q_m = np.mean(q)
    b_m = np.var(q, ddof=1)
    u_m = np.mean(u)
    return q_m, b_m, u_m

def conf_int(q, u, conf_level):
    """Compute confidence interval with Rubin's rules.

    Args:
        q (ndarray): Point estimates.
        u (ndarray): Variance estimates.
        conf_level (float): Confidence level.

    Returns:
        ndarray: The confidence interval.
    """
    dist = conf_int_distribution(q, u)
    return dist.interval(conf_level) if dist is not None else np.repeat(np.nan, 2)

def non_negative_conf_int(q, u, conf_level, n, n_orig):
    """Compute confidence interval with Rubin's rules and non-negative variance estimate.

    Args:
        q (ndarray): Point estimates.
        u (ndarray): Variance estimates.
        conf_level (float): Confidence level.
        n (int): Number of synthetic datapoints.
        n_orig (int): Number of original datapoints.

    Returns:
        ndarray: The confidence interval.
    """
    dist = conf_int_distribution(q, u, True, n, n_orig)
    return dist.interval(conf_level) if dist is not None else np.repeat(np.nan, 2)

def conf_int_distribution(q, u, use_nonnegative_variance=False, n=None, n_orig=None):
    """Compute the estimator distribution with Rubin's rules used for confidence intervals and hypothesis tests.

    Args:
        q (ndarray): Point estimates.
        u (ndarray): Variance estimates.
        use_nonnegative_variance (bool, optional): Use the non-negative variance estimate. Defaults to False.
        n (int, optional): Number of synthetic datapoints. Required with non-negative variance estimate. Defaults to None.
        n_orig (int, optional): Number of original datapoints. Required with non-negative variance estimate. Defaults to None.

    Returns:
        scipy distribution: The distribution as a scipy.stats distribution object.
    """
    q_m, b_m, u_m = compute_aggregates(q, u)
    m = q.size
    T_m = (1 + 1 / m) * b_m - u_m
    if use_nonnegative_variance:
        T_m = T_m if T_m > 0 else n / n_orig * u_m
    degree = (m - 1) * (1 - u_m / ((1 + 1 / m) * b_m))**2
    if not np.isfinite(degree):
        print("m: {}, u_m: {}, b_m: {}".format(m, u_m, b_m))

    if T_m < 0: return None
    else: return stats.t(loc=q_m, scale=np.sqrt(T_m), df=degree)