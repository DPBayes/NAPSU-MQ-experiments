import pickle 
import numpy as np 
import pandas as pd 
import adult_reduced_problem as problem

if __name__ == "__main__":
    cols_in_lr = problem.cols_in_lr
    with open(str(snakemake.input.syn_data), "rb") as file:
        input_object = pickle.load(file)
    with open(str(snakemake.input.orig_result), "rb") as file:
        orig_result_object = pickle.load(file)

    syn_dfs_dict = input_object.synth_data
    orig_oh_df = orig_result_object["orig_oh_df"]

    results = []
    for n_syn_datasets, syn_dfs in syn_dfs_dict.items():
        oh_syn_dfs = [problem.df_transform(syn_df, orig_oh_df.columns) for syn_df in syn_dfs]
        for oh_syn_df in oh_syn_dfs:
            accuracy = problem.logistic_regression_accuracy(oh_syn_df, orig_oh_df)
            results.append({
                "repeat_ind": int(snakemake.wildcards.repeat_ind),
                "epsilon": input_object.epsilon,
                "data_generation_algorithm": input_object.generator_algorithm,
                "downstream-accuracy": accuracy,
            })

    df = pd.DataFrame.from_records(results)
    df.to_csv(str(snakemake.output), index=False)