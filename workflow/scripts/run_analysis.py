import sys
import os
sys.path.append(os.getcwd())

from lib import lr
from lib import rubin
import problem
import numpy as np
import pandas as pd

import synth_data_output
import problem

conf_levels = [0.8, 0.9, 0.95]

with open(str(snakemake.input), "rb") as file:
    input_object = pickle.load(file)

syn_datasets = input_object.synth_data
n_orig = input_object.n_orig
    
m, n, syn_data_d = syn_datasets.shape
if m == 1:
    split_syn_data = np.stack(np.split(syn_datasets.reshape((n, syn_data_d)), problem.n_syn_datasets, axis=0), axis=0)
else:
    split_syn_data = syn_datasets

single_syn_data = split_syn_data[0, :, :]
single_syn_data = single_syn_data.reshape((1, *single_syn_data.shape))

q, u = lr.logistic_regression(split_syn_data, add_constant=False, weight=n_orig/split_syn_data.shape[1])
q_single, u_single, conf_ints_single = lr.logistic_regression(
    single_syn_data, add_constant=False, return_intervals=True, conf_levels=conf_levels, weight=n_orig/single_syn_data.shape[1]
)

_, d = q.shape

data_gen = problem.get_problem()
true_params = data_gen.true_params.numpy()

def conf_int_record(conf_int, ci_type, i, conf_level):
    return {
        "repeat_ind": int(snakemake.wildcards.repeat_ind),
        "conf_level": conf_level,
        "epsilon": input_object.epsilon,
        "data_generation_algorithm": input_object.generator_algorithm,
        "dimension": i,
        "type": ci_type,
        "start": conf_int[0],
        "end": conf_int[1],
        "length": conf_int[1] - conf_int[0],
        "has_coverage": conf_int[0] <= true_params[i] <= conf_int[1],
    }

results = []
for conf_level in conf_levels:
    for i in range(d):
        qi = q[:, i]
        ui = u[:, i]
        inds = (np.isfinite(qi) & np.isfinite(ui))
        qi = qi[inds]
        ui = ui[inds]
        if len(qi) == 0:
            qi = np.array((np.nan))
            ui = np.array((np.nan))

        conf_int = rubin.conf_int(qi, ui, conf_level)
        results.append(conf_int_record(conf_int, "regular", i, conf_level))
        
        conf_int_nn = rubin.non_negative_conf_int(
            qi, ui, conf_level, n, n_orig
        )
        results.append(conf_int_record(conf_int_nn, "non-negative", i, conf_level))

        results.append(conf_int_record(conf_ints_single[conf_level][0, i,:], "non-split", i, conf_level))

        
df = pd.DataFrame.from_records(results)
df.to_csv(str(snakemake.output), index=False)