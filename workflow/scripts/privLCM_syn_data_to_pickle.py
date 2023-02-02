import pandas as pd 
import pickle
import numpy as np
import os
from synth_data_output import SynthDataOutput

syn_dfs = [pd.read_csv(str(file)) for file in snakemake.input]
syn_datasets = np.stack([syn_df.values for syn_df in syn_dfs])
epsilon = float(snakemake.wildcards.epsilon)
n = 2000

output_object = SynthDataOutput(
    n_orig=n, generator_algorithm="privLCM", epsilon=epsilon, delta=n**(-2), 
    synth_data=syn_datasets, generator_diagnostics=None, runtime=np.nan
)
output = str(snakemake.output)
os.makedirs(os.path.dirname(output), exist_ok=True)
with open(output, "wb") as file:
    pickle.dump(output_object, file)