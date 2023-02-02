import jax 
import jax.numpy as jnp
import torch
from abc import ABC, abstractmethod

class LogFactor(ABC):
    """Base class for log factors of Markov networks.
    """
    def __init__(self, scope, values, debug_checks=True):
        """Create the log factor.

        Args:
            scope (tuple): Scope of the factor as a tuple of nodes.
            values (array): Values of the factor as an array where values[i, j, k] is the value for data values (i, j, k).
            debug_checks (bool, optional): If enabled, check that some operations are valid for the factor. Defaults to True.
        """
        self.scope = tuple(scope)
        self.variable_inds = {var: i for i, var in enumerate(scope)}
        self.values = values
        self.debug_checks = debug_checks

    def self_type(self):
        return type(self)

    @abstractmethod
    def tile_values(self, repeat_shape):
        pass

    @abstractmethod
    def permute_axes(self, value, permutation):
        pass

    @abstractmethod
    def logsumexp(self, values, axis):
        pass

    @abstractmethod
    def take_values(self, index, axis):
        pass

    @abstractmethod
    def compute_batch_condition_values(self, variable, values, to_remove_index, result_scope, result_shape):
        pass

    @abstractmethod
    def move_values_axis(self, axis, place):
        pass

    @abstractmethod
    def query(self, queries):
        pass

    def get_variable_range(self, variable):
        return self.values.shape[self.variable_inds[variable]]

    def repeat_to_shape(self, result_shape, result_scope):
        variables_not_in_scope = [var for var in result_scope if var not in self.scope]
        repeat_shape = [result_shape[result_scope.index(var)] for var in variables_not_in_scope] + [1 for var in self.scope]
        repeat_variable_order = variables_not_in_scope + list(self.scope)
        permutation = [repeat_variable_order.index(var) for var in result_scope]
        repeated_values = self.tile_values(repeat_shape)
        return self.permute_axes(repeated_values, permutation)

    def product(self, factor):
        """Product of self with another factor.

        Args:
            factor (LogFactor): The other factor.

        Raises:
            ValueError: If debug checks are enabled, check that the ranges for the variables of both factors are equal.

        Returns:
            LogFactor: The resulting product as a new LogFactor object.
        """
        if self.debug_checks:
            common_variables = set(self.scope).intersection(set(factor.scope))
            for common_var in common_variables:
                if self.get_variable_range(common_var) != factor.get_variable_range(common_var):
                    raise ValueError("Ranges for variable {} are not equal. {} != {}".format(
                        common_var, self.get_variable_range(common_var), factor.get_variable_range(common_var)
                    ))

        if set(self.scope) == set(factor.scope):
            permutation = [factor.scope.index(var) for var in self.scope]
            result_scope = self.scope
            result_values = self.values + factor.permute_axes(factor.values, permutation)
        else:
            result_scope = tuple(set(self.scope).union(set(factor.scope)))
            result_values_ranges = [
                self.get_variable_range(var) if var in self.scope else factor.get_variable_range(var) for var in result_scope
            ]
            self_expanded_values = self.repeat_to_shape(result_values_ranges, result_scope)
            other_expanded_values = factor.repeat_to_shape(result_values_ranges, result_scope)
            result_values = self_expanded_values + other_expanded_values

        return self.self_type()(result_scope, result_values, self.debug_checks)

    def marginalise(self, variable):
        """Marginalise a variable from the factor.

        Args:
            variable (str): The variable to marginalise.

        Returns:
            LogFactor: The marginalised factor as a new LogFactor object.
        """
        to_remove_index = self.scope.index(variable)
        result_scope = self.scope[:to_remove_index] + self.scope[to_remove_index + 1:]
        result_values = self.logsumexp(self.values, axis=self.variable_inds[variable])
        return self.self_type()(result_scope, result_values, self.debug_checks)

    def log_sum_total(self):
        """Compute the logsumexp of all values in self.

        Returns:
            float: The resulting logsumexp value.
        """
        return self.logsumexp(self.values, tuple(range(len(self.values.shape))))

    def list_product(factors):
        """The product of a list of LogFactors.

        Args:
            factors (list(LogFactor)): The list of factors in the product.

        Returns:
            LogFactor: The product of the factors as a new LogFactor object.
        """
        factors = list(factors)
        product = factors[0]
        for i in range(1, len(factors)):
            product = product.product(factors[i])
        return product

    def project(self, variables):
        """Marginalise all variables except the given variables.

        Args:
            variables (list): The variables that are not marginalised.

        Raises:
            ValueError: If debug checks are enabled, checks that variables is a subset of self.scope.

        Returns:
            LogFactor: The result as a new LogFactor object.
        """
        if self.debug_checks and not set(variables).issubset(set(self.scope)):
            raise ValueError("Variables {} are not a subset of scope {}".format(variables, self.scope))

        to_eliminate = set(self.scope).difference(set(variables + ["batch"]))
        factor = self
        for var in to_eliminate:
            factor = factor.marginalise(var)
        return factor

    def condition(self, variable, value):
        """Condition the factor on a given value for a given variable.

        Args:
            variable (str): The variable to condition.
            value (int): The value to condition on.

        Returns:
            LogFactor: The conditioned factor as a new LogFactor object.
        """
        to_remove_index = self.scope.index(variable)
        result_scope = self.scope[:to_remove_index] + self.scope[to_remove_index + 1:]
        result_values = self.take_values(value, to_remove_index)
        return self.self_type()(result_scope, result_values, self.debug_checks)

    def add_batch_dim(self, n_batches):
        """Add a dimension named 'batch' the scope for batched operations.

        self.values is copied so that each batch has the same values.
        Args:
            n_batches (int): The number of batches.

        Raises:
            ValueError: If the 'batch' variable is already in scope.

        Returns:
            LogFactor: The batched factor as a new LogFactor object.
        """
        batch_dim_name = "batch"
        if batch_dim_name in self.scope: raise ValueError("{} is already in scope".format(batch_dim_name))
        result_scope = (batch_dim_name,) + self.scope
        result_values = self.tile_values((n_batches,) + tuple(1 for _ in range(len(self.scope))))
        return self.self_type()(result_scope, result_values, self.debug_checks)

    def batch_condition(self, variable, values):
        """Condition a batched LogFactor on an array of values for a given variable.

        Args:
            variable (str): The variable to condition on.
            values (array): The values for each batch to condition on.

        Returns:
            LogFactor: The result as a new LogFactor object.
        """
        self.ensure_batch_is_first_dim()
        to_remove_index = self.scope.index(variable)
        result_scope = self.scope[:to_remove_index] + self.scope[to_remove_index + 1:]
        result_shape = [self.get_variable_range(var) for var in result_scope]
        result_values = self.compute_batch_condition_values(variable, values, to_remove_index, result_scope, result_shape)
        return self.self_type()(result_scope, result_values, self.debug_checks)

    def ensure_batch_is_first_dim(self):
        if self.scope[0] != "batch" and "batch" in self.scope:
            batch_index = self.variable_inds["batch"]
            self.scope = ("batch",) + self.scope[:batch_index] + self.scope[batch_index + 1:]
            self.move_values_axis(batch_index, 0)
            self.variable_inds = {var: i for i, var in enumerate(self.scope)}


class LogFactorJax(LogFactor):
    """Jax implementation of LogFactor.
    """
    def tile_values(self, repeat_shape):
        return jnp.tile(self.values, repeat_shape)

    def permute_axes(self, value, permutation):
        return value.transpose(permutation)

    def logsumexp(self, values, axis):
        return jax.scipy.special.logsumexp(values, axis=axis)

    def take_values(self, index, axis):
        return jnp.take(self.values, index, axis)

    def compute_batch_condition_values(self, variable, values, to_remove_index, result_scope, result_shape):
        return jax.vmap(lambda i: jnp.take(self.values[i], values[i], to_remove_index - 1), 0, 0)(jnp.arange(self.get_variable_range("batch")))

    def move_values_axis(self, axis, place):
        self.values = jnp.moveaxis(self.values, axis, place)

    def query(self, queries):
        result = jnp.zeros(len(queries.queries))
        for i, query in enumerate(queries.queries):
            query_permutation = [query.features.index(variable) for variable in self.scope]
            query_value_tuple = query.value_tuple()
            result = result.at[i].set(self.values[tuple(query_value_tuple[i] for i in query_permutation)])
        return jnp.exp(result - self.log_sum_total())

class LogFactorTorch(LogFactor):
    """PyTorch implementation of LogFactor.
    """
    def tile_values(self, repeat_shape):
        return self.values.repeat(repeat_shape)

    def permute_axes(self, value, permutation):
        return value.permute(permutation)

    def logsumexp(self, values, axis):
        return torch.logsumexp(values, dim=axis)

    def take_values(self, index, axis):
        return torch.select(self.values, axis, index)

    def expand_indices(self, tensor, result_shape, i):
        shape = result_shape[:i] + result_shape[i + 1:]
        return tensor.expand(shape + [-1]).movedim(-1, i)

    def compute_batch_condition_values(self, variable, values, to_remove_index, result_scope, result_shape):
        take_indices = torch.zeros(result_shape, dtype=torch.long)
        for i in range(len(self.scope)):
            if i == to_remove_index:
                index_term = self.expand_indices(values, result_shape, 0)
            else:
                index_term = self.expand_indices(torch.arange(self.values.shape[i]), result_shape, i - 1 if i > to_remove_index else i)
            mul = 1
            for j in range(i+1, len(self.scope)):
                mul *= self.values.shape[j]
            take_indices += index_term * mul

        return self.values.take(take_indices)

    def move_values_axis(self, axis, place):
        self.values = self.values.movedim(axis, place)

    def query(self, queries):
        result = torch.zeros(len(queries.queries))
        for i, query in enumerate(queries.queries):
            query_permutation = [query.features.index(variable) for variable in self.scope]
            query_value_tuple = query.value_tuple()
            result[i] = self.values[tuple(query_value_tuple[i] for i in query_permutation)]
        return torch.exp(result - self.log_sum_total())