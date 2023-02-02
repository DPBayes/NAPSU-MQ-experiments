import sys
import os
sys.path.append(os.getcwd())
sys.path.append("../")

from lib import lr
from lib import rubin
from lib import confidence_interval_comparison as cic
import problem
import numpy as np
import pandas as pd
import re
import problem

def conf_int_record(conf_int, ci_type, i, conf_level, m, real_m, true_params, orig_conf_ints):
    orig_ci = orig_conf_ints[conf_level].values[i]
    conf_int_length = conf_int[1] - conf_int[0]
    orig_ci_len = orig_ci[1] - orig_ci[0]
    io1, io2 = cic.interval_overlap(np.stack([orig_ci]), np.stack([conf_int]))
    return {
        "repeat_ind": int(snakemake.wildcards.repeat_ind),
        "conf_level": conf_level,
        "epsilon": input_object.epsilon,
        "data_generation_algorithm": input_object.generator_algorithm,
        "dimension": i,
        "n_syn_datasets": m,
        "final_n_syn_datasets": real_m,
        "type": ci_type,
        "start": conf_int[0],
        "end": conf_int[1],
        "length": conf_int_length,
        "length_ratio": conf_int_length / orig_ci_len,
        "has_coverage": conf_int[0] <= true_params[i] <= conf_int[1],
        "IO1": io1[0],
        "IO2": io2[0],
    }

if __name__ == "__main__":
    conf_levels = problem.all_conf_levels
    
    with open(str(snakemake.input.syn_data), "rb") as file:
        input_object = pickle.load(file)
    with open(str(snakemake.input.orig_result), "rb") as file:
        orig_result_object = pickle.load(file)

    syn_dfs_dict = input_object.synth_data
    n_orig = input_object.n_orig
    orig_result = orig_result_object["orig_result"]
    orig_oh_df = orig_result_object["orig_oh_df"]
    orig_oh_df_columns = orig_oh_df.columns

    orig_conf_ints = {}
    for conf_level in conf_levels:
        orig_conf_ints[conf_level] = orig_result.conf_int(1 - conf_level)

    true_params = orig_result.params

    results = []
    for m, syn_dfs in syn_dfs_dict.items():
        n, syn_data_d = syn_dfs[0].shape
        oh_syn_dfs = [problem.df_transform(syn_df, orig_oh_df_columns) for syn_df in syn_dfs]
        syn_datasets = np.stack([df.values for df in oh_syn_dfs])
        if m == 1:
            q_single, u_single, conf_ints_single = lr.logistic_regression(
                syn_datasets, 0, add_constant=True, return_intervals=True, return_results=False, 
                conf_levels=conf_levels, weight=n_orig/n
            )
        else:
            q, u = lr.logistic_regression(syn_datasets, 0, add_constant=True)

        _, d = q.shape if m != 1 else q_single.shape

        for conf_level in conf_levels:
            for i in range(d):
                if m == 1:
                    results.append(
                        conf_int_record(
                            conf_ints_single[conf_level][0, i,:], 
                            "non-split", i, conf_level, m, m, true_params, orig_conf_ints
                        )
                    )
                else:
                    qi = q[:, i]
                    ui = u[:, i]
                    mi = qi.shape[0]
                    
                    conf_int_nn = rubin.non_negative_conf_int(
                        qi, ui, conf_level, n, n_orig
                    )
                    results.append(conf_int_record(
                        conf_int_nn, "non-negative", i, conf_level, m, mi,
                        true_params, orig_conf_ints
                    ))

    df = pd.DataFrame.from_records(results)
    df.to_csv(str(snakemake.output), index=False)
