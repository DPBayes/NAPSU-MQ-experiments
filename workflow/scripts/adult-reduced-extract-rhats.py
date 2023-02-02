import pandas as pd 
import arviz as az 
import pickle 

with open(str(snakemake.input), "rb") as file:
    input_obj = pickle.load(file)

mcmc_table = input_obj.generator_diagnostics["mcmc_table"]
r_hats = mcmc_table["r_hat"]
records = []
for i, r_hat in enumerate(r_hats):
    records.append({
        "epsilon": input_obj.epsilon,
        "data generation algorithm": input_obj.generator_algorithm,
        "repeat_ind": int(snakemake.wildcards.repeat_ind),
        "parameter": i,
        "r_hat": r_hat
    })

df = pd.DataFrame.from_records(records)
df.to_csv(str(snakemake.output), index=False)