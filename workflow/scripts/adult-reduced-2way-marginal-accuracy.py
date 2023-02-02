import sys
import os
import pickle
sys.path.append(os.getcwd())
import itertools 
import numpy as np 
import torch
torch.set_default_dtype(torch.float64)
import pandas as pd 
from lib import marginal_query
from lib.DataFrameData import DataFrameData

def fix_categories(df, orig_df):
    for col in df.columns:
        df[col] = df[col].cat.set_categories(orig_df[col].cat.categories)

if __name__ == "__main__":
    with open(str(snakemake.input.syn_data), "rb") as file:
        input_object = pickle.load(file)

    syn_dfs_dict = input_object.synth_data
    cat_df = pd.read_csv(str(snakemake.input.data), dtype="category")
    df_data = DataFrameData(cat_df)

    all_1way_feature_sets = [(col,) for col in df_data.values_by_col.keys()]
    all_2way_feature_sets = list(itertools.combinations(df_data.values_by_col.keys(), 2))
    all_1way_marginals = marginal_query.FullMarginalQuerySet(all_1way_feature_sets, df_data.values_by_col).flatten()
    all_2way_marginals = marginal_query.FullMarginalQuerySet(all_2way_feature_sets, df_data.values_by_col).flatten()
    orig_1way_values = all_1way_marginals(df_data.int_tensor).double().mean(axis=0)
    orig_2way_values = all_2way_marginals(df_data.int_tensor).double().mean(axis=0)

    n_syn_datasets = max(list(syn_dfs_dict.keys()))
    syn_dfs = syn_dfs_dict[n_syn_datasets]

    all_syn_values_1way = torch.zeros((n_syn_datasets, orig_1way_values.shape[0]))
    all_syn_values_2way = torch.zeros((n_syn_datasets, orig_2way_values.shape[0]))
    for i, syn_df in enumerate(syn_dfs):
        fix_categories(syn_df, cat_df)
        int_tensor = DataFrameData(syn_df).int_tensor
        syn_values_1way = all_1way_marginals(int_tensor).double().mean(axis=0)
        syn_values_2way = all_2way_marginals(int_tensor).double().mean(axis=0)
        all_syn_values_1way[i, :] = syn_values_1way
        all_syn_values_2way[i, :] = syn_values_2way

    results = []

    accuracy_1way = 0.5 * torch.abs(all_syn_values_1way.mean(axis=0) - orig_1way_values).sum() / len(all_1way_feature_sets)
    accuracy_2way = 0.5 * torch.abs(all_syn_values_2way.mean(axis=0) - orig_2way_values).sum() / len(all_2way_feature_sets)
    results.append({
        "repeat_ind": int(snakemake.wildcards.repeat_ind),
        "epsilon": input_object.epsilon,
        "data_generation_algorithm": input_object.generator_algorithm,
        "marginal-1way-accuracy": accuracy_1way.item(),
        "marginal-2way-accuracy": accuracy_2way.item(),
    })

    df = pd.DataFrame.from_records(results)
    df.to_csv(str(snakemake.output), index=False)