import torch
import pyro
import pyro.distributions as dist
import numpyro
import numpyro.distributions as numdist
import jax.numpy as jnp
import numpy as np

def normal_prior_model(dp_suff_stat, n, sigma_DP, prior_mu, prior_sigma, med):
    """Pyro model for NAPSU-MQ with isotropic Gaussian prior.

    Args:
        dp_suff_stat (torch.tensor): Noisy sufficient statistic.
        n (int): Number of datapoints.
        sigma_DP (float): Noise standard deviation.
        prior_mu (torch.tensor): Prior mean.
        prior_sigma (float): Prior standard deviation.
        med (MaximumEntropyDistribution or MarkovNetworkTorch): An implementation of MED.
    """
    d = med.suff_stat_d
    lambda_d = med.lambda_d

    lambdas = pyro.sample("lambdas", dist.MultivariateNormal(torch.ones(lambda_d) * prior_mu, covariance_matrix=torch.eye(lambda_d) * prior_sigma**2))
    mean, cov = med.suff_stat_mean_and_cov(lambdas)
    return pyro.sample("a_hat", dist.MultivariateNormal(n * mean, covariance_matrix=n * cov + torch.eye(d) * sigma_DP**2), obs=dp_suff_stat)

def normal_prior_model_numpyro(dp_suff_stat, n, sigma_DP, prior_mu, prior_sigma, med):
    """NumPyro model for NAPSU-MQ with isotropic Gaussian prior.

    Args:
        dp_suff_stat (jax.ndarray): Noisy sufficient statistic.
        n (int): Number of datapoints.
        sigma_DP (float): Noise standard deviation.
        prior_mu (jax.ndarray): Prior mean.
        prior_sigma (float): Prior standard deviation.
        med (MarkovNetworkJax): An implementation of MED.
    """
    d = med.suff_stat_d
    lambda_d = med.lambda_d

    lambdas = numpyro.sample("lambdas", numdist.MultivariateNormal(jnp.ones(lambda_d) * prior_mu, covariance_matrix=jnp.eye(lambda_d) * prior_sigma**2))
    mean, cov = med.suff_stat_mean_and_cov(lambdas)
    return numpyro.sample("a_hat", numdist.MultivariateNormal(n * mean, covariance_matrix=n * cov + jnp.eye(d) * sigma_DP**2), obs=jnp.array(dp_suff_stat))

def normal_prior_normalised_model(dp_suff_stat, n, sigma_DP, prior_sigma, med, mean_guess, L_guess):
    """Pyro model for NAPSU-MQ with isotropic Gaussian prior and normalisation.

    Args:
        dp_suff_stat (torch.tensor): Noisy sufficient statistic.
        n (int): Number of datapoints.
        sigma_DP (float): Noise standard deviation.
        prior_sigma (float): Prior standard deviation.
        med (MaximumEntropyDistribution or MarkovNetworkTorch): An implementation of MED.
        mean_guess (torch.tensor): A guess for the posterior mean.
        L_guess (torch.tensor): A guess for the Cholesky decomposition of posterior covariance.
    """
    d = med.suff_stat_d
    lambda_d = med.lambda_d

    prior_cov_inner = torch.linalg.solve(L_guess, torch.eye(lambda_d) * prior_sigma**2)
    norm_prior_cov = torch.linalg.solve(L_guess, prior_cov_inner.t()).t()

    norm_prior_mu = torch.linalg.solve(L_guess, -mean_guess.view(-1, 1)).view(-1)

    norm_lambdas = pyro.sample("norm_lambdas", dist.MultivariateNormal(norm_prior_mu, covariance_matrix=norm_prior_cov))
    lambdas = L_guess @ norm_lambdas + mean_guess
    mean, cov = med.suff_stat_mean_and_cov(lambdas)
    return pyro.sample("a_hat", dist.MultivariateNormal(n * mean, covariance_matrix=n * cov + torch.eye(d) * sigma_DP**2), obs=dp_suff_stat)

def normal_prior_normalised_model_numpyro(dp_suff_stat, n, sigma_DP, prior_sigma, med, mean_guess, L_guess):
    """NumPyro model for NAPSU-MQ with isotropic Gaussian prior and normalisation.

    Args:
        dp_suff_stat (jax.ndarray): Noisy sufficient statistic.
        n (int): Number of datapoints.
        sigma_DP (float): Noise standard deviation.
        prior_sigma (float): Prior standard deviation.
        med (MarkovNetworkJax): An implementation of MED.
        mean_guess (jax.ndarray): A guess for the posterior mean.
        L_guess (jax.ndarray): A guess for the Cholesky decomposition of posterior covariance.
    """
    d = med.suff_stat_d
    lambda_d = med.lambda_d

    prior_cov_inner = jnp.linalg.solve(L_guess, jnp.eye(lambda_d) * prior_sigma**2)
    norm_prior_cov = jnp.linalg.solve(L_guess, prior_cov_inner.transpose()).transpose()

    norm_prior_mu = jnp.linalg.solve(L_guess, -mean_guess.reshape((-1, 1))).flatten()

    norm_lambdas = numpyro.sample("norm_lambdas", numdist.MultivariateNormal(norm_prior_mu, covariance_matrix=norm_prior_cov))
    lambdas = L_guess @ norm_lambdas + mean_guess
    mean, cov = med.suff_stat_mean_and_cov(lambdas)
    return numpyro.sample("a_hat", numdist.MultivariateNormal(n * mean, covariance_matrix=n * cov + jnp.eye(d) * sigma_DP**2), obs=dp_suff_stat)

def mvnormal_prior_model(dp_suff_stat, n, sigma_DP, prior_mu, prior_cov, med):
    """Pyro model for NAPSU-MQ with multivariate Gaussian prior.

    Args:
        dp_suff_stat (torch.tensor): Noisy sufficient statistic.
        n (int): Number of datapoints.
        sigma_DP (float): Noise standard deviation.
        prior_mu (torch.tensor): Prior mean.
        prior_cov (torch.tensor): Prior covariance.
        med (MaximumEntropyDistribution or MarkovNetworkTorch): An implementation of MED.
    """
    d = dp_suff_stat.shape[0]
    lambdas = pyro.sample("lambdas", dist.MultivariateNormal(prior_mu, covariance_matrix=prior_cov))
    mean, cov = med.suff_stat_mean_and_cov(lambdas)
    pyro.sample("a_hat", dist.MultivariateNormal(n * mean, covariance_matrix=n * cov + torch.eye(d) * sigma_DP**2), obs=dp_suff_stat)

def conjugate_prior_potential(lambdas, dp_suff_stat, n, sigma_DP, prior_chi, prior_nu, med):
    """Pyro potential function for NAPSU-MQ with conjugate prior.

    Args:
        lambdas (torch.tensor): Parameter value.
        dp_suff_stat (torch.tensor): Noisy sufficient statistic.
        n (int): Number of datapoints.
        sigma_DP (float): Noise standard deviation.
        prior_chi (torch.tensor): Prior hyperparameter.
        prior_nu (float): Prior hyperparameter.
        med (MaximumEntropyDistribution): An implementation of MED.

    Returns:
        _type_: _description_
    """
    d = dp_suff_stat.shape[0]
    log_prior = med.conjugate_unnorm_logpdf(lambdas, prior_chi, prior_nu)
    mean, cov = med.suff_stat_mean_and_cov(lambdas)
    log_likelihood = dist.MultivariateNormal(n * mean, covariance_matrix=n * cov + torch.eye(d) * sigma_DP**2).log_prob(dp_suff_stat)
    return -(log_prior + log_likelihood)
 