import torch
import pandas as pd
import itertools
from functools import reduce
from operator import mul

class DataFrameData:
    """Converter between categorical and integer formatted dataframes.
    """
    def __init__(self, base_df):
        """Initialise.

        Args:
            base_df (DataFrame): Base categorical dataframe.
        """
        self.base_df = base_df
        self.int_df = self.base_df.copy()
        for col in base_df.columns:
            self.int_df[col] = self.base_df[col].cat.codes
        self.int_tensor = torch.tensor(self.int_df.values)

        self.n, self.d = base_df.shape

        self.values_by_col = {col: list(range(len(self.base_df[col].cat.categories))) for col in self.int_df.columns}
        self.values_by_int_feature = {i: list(self.values_by_col[col]) for i, col in enumerate(self.int_df.columns)}

    def get_x_values(self):
        """Enumerate all possible datapoints for the categories of the dataset.

        Returns:
            torch.tensor: Enumeration of all possible datapoints.
        """
        x_values = torch.zeros((self.get_domain_size(), self.d))
        for i, val in enumerate(itertools.product(*self.values_by_col.values())):
            x_values[i, :] = torch.tensor(val)
        return x_values

    def get_domain_size(self):
        """Compute the number of possible datapoints in the domain of the dataset.

        Returns:
            int: The number of possible datapoints in the domain.
        """
        return reduce(mul, [len(col_values) for col_values in self.values_by_col.values()])
    
    def int_df_to_cat_df(self, int_df):
        """Convert interger-valued dataframe to categorical dataframe.

        Args:
            int_df (DataFrame): The interger valued dataframe.

        Returns:
            DataFrame: Categorical valued dataframe.
        """
        cat_df = int_df.copy()
        for col in int_df.columns:
            cat_df[col] = self.base_df[col].cat.categories[int_df[col]]

        return cat_df.astype("category")

    def ndarray_to_cat_df(self, ndarray):
        """Convert integer-valued ndarray to categorical dataframe.

        Args:
            ndarray (ndarray): The interger-valued array to convert.

        Returns:
            DataFrame: The categorical dataframe.
        """
        int_df = pd.DataFrame(ndarray, columns=self.base_df.columns, dtype=int)
        return self.int_df_to_cat_df(int_df)

    def int_query_to_str_query(self, inds, value):
        """Convert marginal query for integer dataframe to query for categorical dataframe.

        Args:
            inds (tuple): Query indices.
            value (tuple): Query value.

        Returns:
            (tuple, tuple): String-valued indices and value.
        """
        value = tuple(value.numpy()) if type(value) is torch.Tensor else value
        str_inds = [self.base_df.columns[ind] for ind in inds]
        str_value = [self.base_df[str_inds[i]].cat.categories[val] for i, val in enumerate(value)]
        return str_inds, str_value

    def str_query_to_int_query(self, feature_set, value):
        """Convert marginal query for categorical dataframe to query for integer dataframe.

        Args:
            feature_set (tuple): Query indices.
            value (tuple): Query values.

        Returns:
            (tuple, tuple): Converted query indices and value.
        """
        def index(list, value):
            if value in list: return list.index(value)
            else: raise ValueError("{} not in {}".format(value, list))

        int_inds = [index(list(self.values_by_col.keys()), feature) for feature in feature_set]
        int_values = [index(list(self.base_df[feature].cat.categories), value[i]) for i, feature in enumerate(feature_set)]
        return int_inds, torch.tensor(int_values)
        