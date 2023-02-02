import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
torch.set_default_dtype(torch.float64)
import pickle
import arviz as az
import time

from lib.DataFrameData import DataFrameData
from lib import rng_initialisation
from lib import marginal_query
from lib import max_ent_inference as mei
from lib import max_ent_dist
from lib import privacy_accounting
from lib import rap_interface
from lib import privbayes
import problem
from synth_data_output import SynthDataOutput

if __name__ == "__main__":
    epsilon = float(snakemake.wildcards.epsilon)

    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat_ind), epsilon)
    torch.manual_seed(seed)

    data_gen = problem.get_problem()
    data = data_gen.generate_data(2000)
    n, d = data.shape
    n_syn_datasets = problem.n_syn_datasets
    delta = n**(-2)

    start_time = time.time()
    algo = snakemake.wildcards.algo
    if algo in ["rap", "rap-all-queries"]:
        kway_attrs = [("X1", "X2", "X3")]
        n_syn_dataset = 1000
        rapi = rap_interface.RapInterface(
            data, kway_attrs, kway_attrs[0], {"X1": [0, 1], "X2": [0, 1], "X3": [0, 1]}
        )
        use_all_queries = algo == "rap-all-queries"
        rapi.run_rap(
            num_points=n_syn_dataset, epochs=6, top_q=1, iterations=5000, epsilon=epsilon, delta=delta, 
            learning_rate=0.1, use_all_queries=use_all_queries, oversample=n_syn_datasets * 50
        )
        syn_datasets = rapi.un_one_hot_code_D_prime()
        syn_datasets = syn_datasets.reshape((1, *syn_datasets.shape))
        diagnostics = None
    elif algo in ["max-ent-all-queries", "max-ent-no-noise-aware"]:
        queries = marginal_query.all_marginals([(0, 1, 2)], data_gen.values_by_feature)
        queries.queries = queries.queries[:-1]
        med = max_ent_dist.MaximumEntropyDistribution(data_gen.values_by_feature, queries)
        suff_stat = torch.sum(queries(data), axis=0)
        sensitivity = np.sqrt(2)
        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)
        dp_suff_stat = torch.normal(mean=suff_stat.double(), std=sigma_DP)

        if algo == "max-ent-no-noise-aware":
            sigma_DP = 0.005
        n_syn_dataset = n
        laplace_approx, lap_losses, fail_count = mei.laplace_approximation_normal_prior(
            dp_suff_stat, n, sigma_DP, med, max_retries=4, max_iters=500
        )
        posterior_values = laplace_approx.sample((n_syn_datasets,)).numpy()
        syn_datasets = mei.generate_synthetic_data(posterior_values, n_syn_dataset, med, show_progressbar=False)
        diagnostics = {"losses": lap_losses, "fail_count": fail_count}
    elif algo == "PGM":
        from mbi import Dataset, Domain, FactoredInference
        data_df = pd.DataFrame(data, columns=["X1", "X2", "X3"])
        domain = Domain(["X1", "X2", "X3"], [2, 2, 2])
        dataset = Dataset(data_df, domain)
        suff_stat = torch.tensor(dataset.datavector())
        sensitivity = np.sqrt(2)
        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)
        dp_suff_stat = torch.normal(mean=suff_stat.double(), std=sigma_DP).numpy()
        measurement = [(np.eye(suff_stat.shape[0]), dp_suff_stat, sigma_DP, ("X1", "X2", "X3"))]

        engine = FactoredInference(domain, log=False)
        model = engine.estimate(measurement, engine="MD")

        n_syn_dataset = n * n_syn_datasets
        syn_datasets = model.synthetic_data(n_syn_dataset).df.values
        syn_datasets = syn_datasets.reshape((1, *syn_datasets.shape))
        diagnostics = None
    elif algo == "PEP":
        sys.path.insert(0, "iterative-dp")
        import pep
        from mbi import Domain, Dataset
        sys.path = sys.path[1:]
        
        df = pd.DataFrame(data, columns=("X1", "X2", "X3"), dtype=int)

        queries = marginal_query.all_marginals([(0, 1, 2)], data_gen.values_by_feature)
        suff_stat = torch.sum(queries(data), axis=0)
        sensitivity = np.sqrt(2)
        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)
        dp_suff_stat = torch.normal(mean=suff_stat.double(), std=sigma_DP)

        domain = Domain(("X1", "X2", "X3"), (2, 2, 2))
        marginals = pep.MyMarginals(domain, [("X1", "X2", "X3")])
        generator = pep.PEP(domain, marginals, 10000)
        generator.query_measurements = [(i, x.item()) for i, x in enumerate(dp_suff_stat / n)]
        generator.project(1000)
        probs = generator.synthethic_weights

        n_syn_dataset = n * n_syn_datasets
        syn_data_inds = torch.distributions.Categorical(probs=torch.tensor(probs)).sample((n_syn_dataset,))
        syn_datasets = data_gen.x_values[syn_data_inds]
        syn_datasets = syn_datasets.reshape((1, *syn_datasets.shape)).numpy()
        diagnostics = None
    elif algo == "PrivBayes":
        from mbi import Dataset, Domain, FactoredInference
        data_df = pd.DataFrame(data, columns=["X1", "X2", "X3"], dtype="category")
        domain = Domain(["X1", "X2", "X3"], [2, 2, 2])
        df_data = DataFrameData(data_df)
        fmqs = marginal_query.FullMarginalQuerySet([("X1",), ("X2", "X1"), ("X3", "X1", "X2")], df_data.values_by_col)
        suff_stat = fmqs.query(data)
        suff_stat = {key: val.sum(axis=0) for key, val in suff_stat.items()}

        sensitivity = np.sqrt(2 * 3)
        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)
        dp_suff_stat = {key: torch.normal(mean=val.double(), std=sigma_DP) for key, val in suff_stat.items()}
        measurements = [
            (None, val.numpy(), None, key) for key, val in dp_suff_stat.items()
        ]
        n_syn_dataset = n * n_syn_datasets
        syn_data = privbayes.privbayes_inference(domain, measurements, n_syn_dataset)
        syn_datasets = syn_data.df.values
        syn_datasets = syn_datasets.reshape((1, *syn_datasets.shape))
        diagnostics = None
    else:
        raise ValueError("Given algorithm {} is not implemented".format(algo))

    end_time = time.time()
    output_object = SynthDataOutput(
        n_orig=n, generator_algorithm=algo, epsilon=epsilon, delta=delta, 
        synth_data=syn_datasets, generator_diagnostics=diagnostics, runtime=end_time - start_time
    )
    output = str(snakemake.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as file:
        pickle.dump(output_object, file)