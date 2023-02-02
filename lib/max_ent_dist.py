import torch
import itertools
from functools import reduce
from operator import mul

class MaximumEntropyDistribution:
    """Implementation of MED without using graphical models.

    This class computes all quantities naively, so it is only useful for testing
    with small data domains.
    """
    def __init__(self, values_by_feature, queries):
        """Create The MaximumEntropyDistribution.

        Args:
            values_by_feature (dict): A dict of the possible values for each variable.
            queries (MarginalQueryList): The queries forming the sufficient statistic.
        """
        self.values_by_feature = values_by_feature
        self.d = len(values_by_feature.keys())
        self.suff_stat_d = len(queries.queries)
        self.lambda_d = self.suff_stat_d
        self.queries = queries

        self.x_values = self.get_x_values(list(values_by_feature.keys()))
        self.compute_suff_stat_cache()

    def lambda0(self, lambdas):
        return torch.logsumexp(self.suff_stat_array @ lambdas + self.suff_stat_log_count_array, 0)

    def suff_stat_mean_and_lambda0(self, lambdas):
        lambda0 = self.lambda0(lambdas)
        sum_term = torch.log(self.suff_stat_array) + (self.suff_stat_array @ lambdas).view(-1, 1) + self.suff_stat_log_count_array.view(-1, 1)
        suff_stat_mean = torch.sum(torch.exp(sum_term - lambda0), axis=0)
        return suff_stat_mean, lambda0

    def suff_stat_mean(self, lambdas):
        return self.suff_stat_mean_and_lambda0(lambdas)[0]

    def suff_stat_cov(self, lambdas):
        return self.suff_stat_mean_and_cov_explicit(lambdas)[1]
        
    def suff_stat_mean_and_cov(self, lambdas):
        if len(lambdas.shape) > 1:
            n, d = lambdas.shape
            mean = torch.zeros(lambdas.shape)
            cov = torch.zeros(n, d, d)
            for i in range(n):
                mean_l, cov_l = self.suff_stat_mean_and_cov_explicit(lambdas[i, :])
                mean[i, :] = mean_l
                cov[i, :, :] = cov_l
            return mean, cov

        else:
            return self.suff_stat_mean_and_cov_explicit(lambdas)

    def sample_inds(self, lambdas, n=1, generator=None):
        random_values = torch.rand(n, generator=generator).view(-1, 1)
        thresholds = self.pmf_all_values(lambdas).cumsum(0)
        inds = torch.argmax((random_values < thresholds).int(), dim=1).view(-1)
        return inds

    def sample(self, lambdas, n=1, generator=None):
        return self.x_values[self.sample_inds(lambdas, n, generator), :]

    def compute_suff_stat_cache(self):
        suff_stats_all_int = self.queries(self.x_values)
        self.suff_stats_all = suff_stats_all_int.double()

        self.suff_stat_array, self.suff_stat_count_array = torch.unique(suff_stats_all_int, return_counts=True, dim=0)
        self.suff_stat_array = self.suff_stat_array.double()
        self.suff_stat_count_array = self.suff_stat_count_array.double()

        self.suff_stat_log_count_array = torch.log(self.suff_stat_count_array)

    def get_x_values(self, variables):
        x_val_count = reduce(mul, (len(self.values_by_feature[var]) for var in variables), 1)
        x_values_all = torch.zeros((x_val_count, self.d), dtype=torch.long)

        for i, val in enumerate(itertools.product(*[self.values_by_feature[var] for var in variables])):
            x_values_all[i, variables] = torch.tensor(val, dtype=torch.long)
        return x_values_all

    def get_subset_suff_stat(self, variables):
        subset_x_values = self.get_x_values(variables)
        return self.queries.get_subset(self.variable_associations[variables])(subset_x_values).double()

    def suff_stat_mean_and_cov_explicit(self, lambdas):
        d = lambdas.shape[0]
        mean, lambda0 = self.suff_stat_mean_and_lambda0(lambdas)
        bmm_term = torch.bmm(self.suff_stat_array.view(-1, d, 1), self.suff_stat_array.view(-1, 1, d))
        sum_term = torch.log(bmm_term) + (self.suff_stat_array @ lambdas).reshape((-1, 1, 1)) + self.suff_stat_log_count_array.reshape((-1, 1, 1))
        cov = torch.sum(torch.exp(sum_term - lambda0), axis=0) - torch.outer(mean, mean)
        # Ensure returned matrix is positive-definite
        return mean, cov + torch.eye(lambdas.shape[0]) * 1e-12

    def pmf_all_values(self, lambdas):
        return torch.exp(self.suff_stats_all @ lambdas - self.lambda0(lambdas))

    def mean_query_values(self, queries, lambdas):
        query_values = queries(self.x_values)
        pmf = self.pmf_all_values(lambdas).view(-1, 1)
        return (pmf * query_values).sum(dim=0)

    def conjugate_unnorm_logpdf(self, lambdas, chi, nu):
        lambda0 = self.lambda0(lambdas)
        return torch.dot(chi, lambdas) - nu * lambda0
