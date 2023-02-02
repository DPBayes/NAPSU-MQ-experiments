import pandas as pd 
import pickle 

with open(str(snakemake.input), "rb") as file:
    input_obj = pickle.load(file)

diagnostics = input_obj.generator_diagnostics
if "fail_count" in diagnostics.keys():
    fails = diagnostics["fail_count"]
else:
    fails = diagnostics["fails"]
records = []
records.append({
    "epsilon": input_obj.epsilon,
    "data generation algorithm": input_obj.generator_algorithm,
    "repeat_ind": int(snakemake.wildcards.repeat_ind),
    "fail_count": fails
})

df = pd.DataFrame.from_records(records)
df.to_csv(str(snakemake.output), index=False)