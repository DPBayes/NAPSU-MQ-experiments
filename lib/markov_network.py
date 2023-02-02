import jax
import jax.numpy as jnp
import torch
import functorch
import numpy as np
import pandas as pd
import itertools
import functools
from abc import ABC, abstractmethod
from lib.junction_tree import JunctionTree
from lib.undirected_graph import UndirectedGraph, greedy_ordering
from lib.log_factor import LogFactorJax, LogFactorTorch, LogFactor

class MarkovNetwork(ABC):
    """A Markov network representation for MED.
    """
    def __init__(self, domain, queries, elimination_order=None, debug_checks=True):
        """Create the Markov network.

        Args:
            domain (dict): A dict containing the possible values for each variable.
            queries (FullMarginalQuerySet): The queries forming the sufficient statistic.
            elimination_order (list, optional): Elimination order for variable elimination. Defaults to using a greedy ordering.
            debug_checks (bool, optional): Whether to enable debug checks for factor computations. Defaults to True.
        """
        self.debug_checks = debug_checks
        self.domain = domain 
        self.queries = queries
        self.d = len(self.domain.keys())
        variables_in_queries = set.union(*[set(feature_set) for feature_set in self.queries.feature_sets])
        self.variables_not_in_queries = set(domain.keys()).difference(variables_in_queries)

        self.graph = UndirectedGraph.from_clique_list(queries.feature_sets + [(var,) for var in self.variables_not_in_queries])
        if elimination_order is None:
            elimination_order = greedy_ordering(self.graph)
        self.elimination_order = elimination_order
        self.junction_tree = JunctionTree.from_variable_elimination(queries.feature_sets, self.elimination_order)
        self.junction_tree.remove_redundant_nodes()

        self.flat_queries = self.queries.flatten()
        self.variable_associations = self.flat_queries.get_variable_associations(use_features=True)
        self.suff_stat_d = len(self.flat_queries.queries)
        self.lambda_d = self.suff_stat_d

    @abstractmethod
    def lambda0(self, lambdas):
        pass

    @abstractmethod
    def suff_stat_mean_bp(self, lambdas):
        pass

    @abstractmethod
    def sample(self, lambdas, n_sample=1):
        """Sample the distribution with given parameter values.

        Args:
            lambdas (array): The parameter values.
            n_sample (int, optional): The number of samples to generate. Defaults to 1.

        Returns:
            array: The generated samples.
        """
        pass

    @abstractmethod
    def log_factor_vector(self, lambdas, variables):
        pass

    @abstractmethod
    def compute_factors(self, lambdas):
        """Compute the LogFactor objects used by other methods for given parameters.

        Args:
            lambdas (array): The parameters.

        Returns:
            list(LogFactor): The resulting factors.
        """ 
        pass

    def suff_stat_mean_and_cov_bp(self, lambdas):
        """Compute sufficient statistic mean and covariance for given parameter values with belief propagation.

        Args:
            lambdas (array): The parameter values.

        Returns:
            (array, array): (mean, covariance)
        """
        return self.suff_stat_mean_bp(lambdas), self.suff_stat_cov_bp(lambdas)

    def suff_stat_mean_and_cov(self, lambdas):
        """Compute sufficient statistic mean and covariance for given parameter values with.

        Args:
            lambdas (array): The parameter values.

        Returns:
            (array, array): (mean, covariance)
        """
        return self.suff_stat_mean(lambdas), self.suff_stat_cov(lambdas)

    def marginal_distribution_logits(self, factors, variables):
        to_eliminate = [var for var in self.elimination_order if var not in variables]
        result_factor = self.variable_elimination(factors, to_eliminate)
        proj_factor = result_factor.project(variables)
        proj_factor.ensure_batch_is_first_dim()
        return proj_factor.values

    def variable_elimination(self, factors, variables_to_eliminate):
        """Run variable elimination.

        Args:
            factors (list(LogFactor)): The factors used in variable elimination from self.compute_factors.
            variables_to_eliminate (list): The variables to eliminate.

        Returns:
            LogFactor: The result of variable elimination as a LogFactor.
        """
        for variable in variables_to_eliminate:
            factors = self.eliminate_var(factors, variable)

        return LogFactor.list_product(factors)

    def eliminate_var(self, factors, variable):
        factors_in_prod = [factor for factor in factors if variable in factor.scope]
        factors_not_in_prod = [factor for factor in factors if variable not in factor.scope]

        if len(factors_in_prod) == 0: return factors_not_in_prod
        log_prod_factor = LogFactor.list_product(factors_in_prod)

        summed_log_factor = log_prod_factor.marginalise(variable)
        factors_not_in_prod.append(summed_log_factor)
        return factors_not_in_prod

    def belief_propagation(self, factors):
        """Run belief propagation.

        Args:
            factors (list(LogFactor)): The factors used in belief propagation from self.compute_factors

        Returns:
            dict: Dict containing the LogFactor for each set factor scope in the Markov network.
        """
        for node in self.junction_tree.upward_order:
            node.reset()
            node.potential = LogFactor.list_product(
                factor for factor in factors if factor.scope in self.junction_tree.factors_in_node[node.variables]
            )
        for node in self.junction_tree.upward_order:
            if node.parent is not None:
                self.bp_message(node, node.parent)
        for node in self.junction_tree.downward_order:
            for child in node.children:
                self.bp_message(node, child)
        for node in self.junction_tree.downward_order:
            node.result = LogFactor.list_product([node.potential] + [message for _, message in node.messages])
            
        return {node.variables: node.result for node in self.junction_tree.downward_order}

    def bp_message(self, sender, receiver):
        product = LogFactor.list_product([sender.potential] + [message for mes_sender, message in sender.messages if mes_sender is not receiver])
        edges = self.junction_tree.edges
        n1 = sender.variables
        n2 = receiver.variables
        separator = edges[(n1, n2)] if (n1, n2) in edges.keys() else edges[(n2, n1)]
        for variable in set(sender.variables).difference(set(separator)):
            product = product.marginalise(variable)
        receiver.messages.append((sender, product))

class MarkovNetworkJax(MarkovNetwork):
    """Jax implementation of MarkovNetwork.
    """
    def __init__(self, domain, queries, elimination_order=None, debug_checks=True):
        super().__init__(domain, queries, elimination_order, debug_checks)
        self.suff_stat_mean = jax.jit(jax.grad(self.lambda0))
        self.suff_stat_cov = jax.jit(jax.hessian(self.lambda0))
        self.suff_stat_mean_bp = jax.jit(self.suff_stat_mean_bp)
        self.suff_stat_cov_bp = jax.jit(jax.jacrev(self.suff_stat_mean_bp))
        self.log_factor_class = LogFactorJax

    @functools.partial(jax.jit, static_argnums=0)
    def lambda0(self, lambdas):
        factors = self.compute_factors(lambdas)
        result_factor = self.variable_elimination(factors, self.elimination_order)
        return result_factor.values

    def suff_stat_mean_bp(self, lambdas):
        factors = self.compute_factors(lambdas)
        result_factors = self.belief_propagation(factors)
        result = jnp.zeros(self.suff_stat_d)
        for clique, indices in self.variable_associations.items():
            node_variables = self.junction_tree.node_for_factor[clique]
            factor = result_factors[node_variables]
            for variable in set(node_variables).difference(clique):
                factor = factor.marginalise(variable)
            result = result.at[jnp.array(indices)].set(factor.query(self.queries.queries[clique]))
        return result

    def sample(self, rng, lambdas, n_sample=1):
        n_cols = len(self.domain.keys())
        cols = self.domain.keys()
        data = np.zeros((n_sample, n_cols), dtype=jnp.int64)
        df = pd.DataFrame(data, columns=cols, dtype=int)

        order = self.elimination_order[::-1]
        batch_factors = [factor.add_batch_dim(n_sample) for factor in self.compute_factors(lambdas)]
        for variable in order:
            marginal = self.marginal_distribution_logits(batch_factors, [variable])
            rng, key = jax.random.split(rng)
            values = jax.random.categorical(key, marginal)
            batch_factors = [factor.batch_condition(variable, values) if variable in factor.scope else factor for factor in batch_factors]
            df.loc[:, variable] = np.array(values)

        return df

    def log_factor_vector(self, lambdas, variables):
        vec = jnp.zeros(tuple(len(self.domain[var]) for var in variables))
        for query_ind in self.variable_associations[variables]:
            query_val = jnp.array(self.flat_queries.queries[query_ind].value)
            vec = vec.at[tuple(query_val)].set(lambdas[query_ind])
        return vec

    def compute_factors(self, lambdas):
        return [
            LogFactorJax(factor_scope, self.log_factor_vector(lambdas, factor_scope), self.debug_checks)
            for factor_scope in self.variable_associations.keys()
        ] + [
            LogFactorJax((variable,), jnp.zeros(len(self.domain[variable])), self.debug_checks) 
            for variable in self.variables_not_in_queries
        ]


class MarkovNetworkTorch(MarkovNetwork):
    """PyTorch implementation of MarkovNetwork.
    """
    def __init__(self, domain, queries, elimination_order=None, debug_checks=True):
        super().__init__(domain, queries, elimination_order, debug_checks)
        self.suff_stat_mean = functorch.grad(self.lambda0)
        self.suff_stat_cov = functorch.jacrev(functorch.jacrev(self.lambda0))
        self.suff_stat_cov_bp = functorch.jacrev(self.suff_stat_mean_bp)
        self.log_factor_class = LogFactorTorch

    def lambda0(self, lambdas):
        factors = self.compute_factors(lambdas)
        result_factor = self.variable_elimination(factors, self.elimination_order)
        return result_factor.values

    def suff_stat_mean_bp(self, lambdas):
        factors = self.compute_factors(lambdas)
        result_factors = self.belief_propagation(factors)
        result = torch.zeros(self.suff_stat_d)
        for clique, indices in self.variable_associations.items():
            node_variables = self.junction_tree.node_for_factor[clique]
            factor = result_factors[node_variables]
            for variable in set(node_variables).difference(clique):
                factor = factor.marginalise(variable)
            result[torch.tensor(indices)] = factor.query(self.queries.queries[clique])
        return result

    def sample(self, lambdas, n_sample=1):
        n_cols = len(self.domain.keys())
        cols = self.domain.keys()
        data = torch.zeros((n_sample, n_cols), dtype=torch.long)
        df = pd.DataFrame(data, columns=cols, dtype=int)

        order = self.elimination_order[::-1]
        batch_factors = [factor.add_batch_dim(n_sample) for factor in self.compute_factors(lambdas)]
        for variable in order:
            marginal = self.marginal_distribution_logits(batch_factors, [variable])
            values = torch.distributions.Categorical(logits=marginal).sample((1,)).flatten()
            batch_factors = [factor.batch_condition(variable, values) if variable in factor.scope else factor for factor in batch_factors]
            df.loc[:, variable] = values.numpy()

        return df

    def log_factor_vector(self, lambdas, variables):
        vec = torch.zeros(tuple(len(self.domain[var]) for var in variables))
        for query_ind in self.variable_associations[variables]:
            query_val = self.flat_queries.queries[query_ind].value
            vec[tuple(query_val)] = lambdas[query_ind]
        return vec

    def compute_factors(self, lambdas):
        return [
            LogFactorTorch(factor_scope, self.log_factor_vector(lambdas, factor_scope), self.debug_checks)
            for factor_scope in self.variable_associations.keys()
        ] + [
            LogFactorTorch((variable,), torch.zeros(len(self.domain[variable])), self.debug_checks) 
            for variable in self.variables_not_in_queries
        ]
