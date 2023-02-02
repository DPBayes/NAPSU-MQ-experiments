import unittest 
import torch
torch.set_default_dtype(torch.float64)
from jax.config import config
config.update("jax_enable_x64", True) 
import jax
import jax.numpy as jnp
from lib.markov_network import *
from lib.binary_lr_data import BinaryLogisticRegressionDataGenerator
from lib.marginal_query import *
from lib.max_ent_dist import MaximumEntropyDistribution
# import lib.markov_network_jax as markov_network_jax

class MarkovNetworkTest(unittest.TestCase):
    def setUp(self):
        self.data_gen = BinaryLogisticRegressionDataGenerator(torch.arange(4).double())
        self.queries = FullMarginalQuerySet([(0, 3), (1, 3), (2, 3), (4, 3), (0, 1)], self.data_gen.values_by_feature)
        self.full_queries = all_marginals([(0, 1, 2, 3, 4)], self.data_gen.values_by_feature)
        self.mn = MarkovNetworkTorch(self.data_gen.values_by_feature, self.queries)
        self.med = MaximumEntropyDistribution(self.data_gen.values_by_feature, self.queries.flatten())

        self.n_queries = len(self.queries.flatten().queries)
        self.lambda1 = torch.zeros(self.n_queries)
        self.lambda2 = torch.ones(self.n_queries)
        self.lambda3 = torch.arange(self.n_queries).double()
        self.lambda4 = torch.arange(self.n_queries).double() / self.n_queries

    def test_lambda0(self):
        self.assertAlmostEqual(self.mn.lambda0(self.lambda1).item(), self.med.lambda0(self.lambda1).item())
        self.assertAlmostEqual(self.mn.lambda0(self.lambda2).item(), self.med.lambda0(self.lambda2).item())
        self.assertAlmostEqual(self.mn.lambda0(self.lambda3).item(), self.med.lambda0(self.lambda3).item())
        self.assertAlmostEqual(self.mn.lambda0(self.lambda4).item(), self.med.lambda0(self.lambda4).item())

    def test_suff_stat_mean(self):
        self.assertTrue(torch.allclose(self.mn.suff_stat_mean(self.lambda1), self.med.suff_stat_mean(self.lambda1)))
        self.assertTrue(torch.allclose(self.mn.suff_stat_mean(self.lambda2), self.med.suff_stat_mean(self.lambda2)))
        self.assertTrue(torch.allclose(self.mn.suff_stat_mean(self.lambda3), self.med.suff_stat_mean(self.lambda3)))
        self.assertTrue(torch.allclose(self.mn.suff_stat_mean(self.lambda4), self.med.suff_stat_mean(self.lambda4)))

    def test_suff_stat_cov(self):
        self.assertTrue(torch.allclose(self.mn.suff_stat_cov(self.lambda1), self.med.suff_stat_cov(self.lambda1)))
        self.assertTrue(torch.allclose(self.mn.suff_stat_cov(self.lambda2), self.med.suff_stat_cov(self.lambda2)))
        self.assertTrue(torch.allclose(self.mn.suff_stat_cov(self.lambda3), self.med.suff_stat_cov(self.lambda3)))
        self.assertTrue(torch.allclose(self.mn.suff_stat_cov(self.lambda4), self.med.suff_stat_cov(self.lambda4)))

    def test_suff_stat_mean_and_cov(self):
        mean, cov = self.mn.suff_stat_mean_and_cov(self.lambda1)
        self.assertTrue(torch.allclose(mean, self.mn.suff_stat_mean(self.lambda1)))
        self.assertTrue(torch.allclose(cov, self.mn.suff_stat_cov(self.lambda1)))
        mean, cov = self.mn.suff_stat_mean_and_cov(self.lambda2)
        self.assertTrue(torch.allclose(mean, self.mn.suff_stat_mean(self.lambda2)))
        self.assertTrue(torch.allclose(cov, self.mn.suff_stat_cov(self.lambda2)))
        mean, cov = self.mn.suff_stat_mean_and_cov(self.lambda3)
        self.assertTrue(torch.allclose(mean, self.mn.suff_stat_mean(self.lambda3)))
        self.assertTrue(torch.allclose(cov, self.mn.suff_stat_cov(self.lambda3)))

    def test_sample(self):
        n = 2000
        sample = torch.tensor(self.mn.sample(self.lambda4, n).values)
        sample_one_hot = self.full_queries(sample)
        sample_pmf = sample_one_hot.sum(dim=0) / n
        self.assertTrue(torch.allclose(sample_pmf, self.med.pmf_all_values(self.lambda4), atol=0.05))

class MarkovNetworkJAXTest(unittest.TestCase):
    def setUp(self):
        self.data_gen = BinaryLogisticRegressionDataGenerator(torch.arange(4).double())
        self.queries = FullMarginalQuerySet([(0, 3), (1, 3), (2, 3), (4, 3), (0, 1)], self.data_gen.values_by_feature)
        self.full_queries = all_marginals([(0, 1, 2, 3, 4)], self.data_gen.values_by_feature)
        self.mn = MarkovNetworkJax(self.data_gen.values_by_feature, self.queries)
        self.med = MaximumEntropyDistribution(self.data_gen.values_by_feature, self.queries.flatten())

        self.n_queries = len(self.queries.flatten().queries)
        self.lambda1 = torch.zeros(self.n_queries)
        self.lambda2 = torch.ones(self.n_queries)
        self.lambda3 = torch.arange(self.n_queries).double()
        self.lambda4 = torch.arange(self.n_queries).double() / self.n_queries

    def test_lambda0(self):
        self.assertAlmostEqual(self.mn.lambda0(jnp.array(self.lambda1)), self.med.lambda0(self.lambda1).item())
        self.assertAlmostEqual(self.mn.lambda0(jnp.array(self.lambda2)), self.med.lambda0(self.lambda2).item())
        self.assertAlmostEqual(self.mn.lambda0(jnp.array(self.lambda3)), self.med.lambda0(self.lambda3).item())
        self.assertAlmostEqual(self.mn.lambda0(jnp.array(self.lambda4)), self.med.lambda0(self.lambda4).item())

    def test_suff_stat_mean(self):
        self.assertTrue(jnp.allclose(self.mn.suff_stat_mean(jnp.array(self.lambda1)), jnp.array(self.med.suff_stat_mean(self.lambda1))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_mean(jnp.array(self.lambda2)), jnp.array(self.med.suff_stat_mean(self.lambda2))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_mean(jnp.array(self.lambda3)), jnp.array(self.med.suff_stat_mean(self.lambda3))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_mean(jnp.array(self.lambda4)), jnp.array(self.med.suff_stat_mean(self.lambda4))))

    def test_suff_stat_cov(self):
        self.assertTrue(jnp.allclose(self.mn.suff_stat_cov(jnp.array(self.lambda1)), jnp.array(self.med.suff_stat_cov(self.lambda1))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_cov(jnp.array(self.lambda2)), jnp.array(self.med.suff_stat_cov(self.lambda2))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_cov(jnp.array(self.lambda3)), jnp.array(self.med.suff_stat_cov(self.lambda3))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_cov(jnp.array(self.lambda4)), jnp.array(self.med.suff_stat_cov(self.lambda4))))

    def test_suff_stat_mean_and_cov(self):
        mean, cov = self.mn.suff_stat_mean_and_cov(jnp.array(self.lambda1))
        self.assertTrue(jnp.allclose(mean, self.mn.suff_stat_mean(jnp.array(self.lambda1))))
        self.assertTrue(jnp.allclose(cov, self.mn.suff_stat_cov(jnp.array(self.lambda1))))
        mean, cov = self.mn.suff_stat_mean_and_cov(jnp.array(self.lambda2))
        self.assertTrue(jnp.allclose(mean, self.mn.suff_stat_mean(jnp.array(self.lambda2))))
        self.assertTrue(jnp.allclose(cov, self.mn.suff_stat_cov(jnp.array(self.lambda2))))
        mean, cov = self.mn.suff_stat_mean_and_cov(jnp.array(self.lambda3))
        self.assertTrue(jnp.allclose(mean, self.mn.suff_stat_mean(jnp.array(self.lambda3))))
        self.assertTrue(jnp.allclose(cov, self.mn.suff_stat_cov(jnp.array(self.lambda3))))

    def test_sample(self):
        n = 2000
        sample = torch.tensor(self.mn.sample(jax.random.PRNGKey(7438742), jnp.array(self.lambda4), n).values)
        sample_one_hot = self.full_queries(sample)
        sample_pmf = sample_one_hot.sum(dim=0) / n
        self.assertTrue(torch.allclose(sample_pmf, self.med.pmf_all_values(self.lambda4), atol=0.05))

if __name__ == "__main__":
    unittest.main()