import pickle 
import pandas as pd 
import numpy as np
import synth_data_output

input_objects = []
for filename in snakemake.input:
    with open(str(filename), "rb") as file:
        obj = pickle.load(file)
        input_objects.append(obj)

mcmc_runtimes = {}
laplace_runtimes = {}
other_runtimes = {}
for input_obj in input_objects:
    algo = input_obj.generator_algorithm
    eps = input_obj.epsilon
    if eps not in mcmc_runtimes:
        mcmc_runtimes[eps] = []
        laplace_runtimes[eps] = []
        other_runtimes[eps] = {}
    if algo == "max-ent":
        mcmc_runtimes[eps].append(input_obj.runtime["mcmc"])
        laplace_runtimes[eps].append(input_obj.runtime["laplace"])
    else:
        if algo not in other_runtimes[eps].keys():
            other_runtimes[eps][algo] = []
        if "total" in input_obj.runtime.keys():
            runtime = input_obj.runtime["total"]
        elif "RAP" in input_obj.runtime.keys():
            runtime = input_obj.runtime["RAP"]
        elif "PGM" in input_obj.runtime.keys():
            runtime = input_obj.runtime["PGM"]
        other_runtimes[eps][algo].append(runtime)

records = []
for eps in mcmc_runtimes.keys():
    records.append({
        "Epsilon": eps,
        "Algorithm": "LA",
        "Mean": np.array(laplace_runtimes[eps]).mean(),
        "Std": np.array(laplace_runtimes[eps]).std(),
    })
    records.append({
        "Epsilon": eps,
        "Algorithm": "NUTS",
        "Mean": np.array(mcmc_runtimes[eps]).mean(),
        "Std": np.array(mcmc_runtimes[eps]).std(),
    })
    for algo, runtimes in other_runtimes[eps].items():
        records.append({
            "Epsilon": eps,
            "Algorithm": algo,
            "Mean": np.array(runtimes).mean(),
            "Std": np.array(runtimes).std(),
        })
combined_df = pd.DataFrame.from_records(records)

def format_timedelta(td):
    if td.components.hours > 0:
        return "{} h {} min {} s".format(td.components.hours, td.components.minutes, td.components.seconds)
    elif td.components.minutes > 0:
        return "{} min {} s".format(td.components.minutes, td.components.seconds)
    else:
        return "{} s".format(td.components.seconds)

combined_df["Mean"] = pd.to_timedelta(combined_df["Mean"], unit="seconds").apply(format_timedelta)
combined_df["Std"] = combined_df["Std"].apply(lambda s: "{} s".format("{:.1f}".format(s) if s < 100 else str(round(s))))
combined_df.rename(columns={"Std": "Standard Deviation"}, inplace=True)
combined_df = combined_df.groupby(["Algorithm", "Epsilon"]).first()
print(combined_df)

caption="""
Runtimes of each inference run for the Adult experiment. Does not include the time 
taken to generate synthetic data, or run any downstream analysis. The LA rows record the 
runtime for obtaining the Laplace approximation for NAPSU-MQ that is used in the NUTS inference, so the 
total runtime for a NAPSU-MQ run with NUTS is the sum of the LA and NUTS rows.
Experiments were run on 4 CPU cores of a cluster.
"""
combined_df.to_csv(str(snakemake.output.csv), index=True, header=True)
combined_df.to_latex(
    str(snakemake.output.latex), index=True, header=True, label="tab:runtime", 
    caption=caption, escape=False, sparsify=True, multirow=True
)