import pandas as pd
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from jax.config import config
config.update("jax_enable_x64", True) 
import jax
import jax.numpy as jnp
import arviz as az

import sys
import os
import pickle
import time
sys.path.append(os.getcwd())

from lib import rng_initialisation
from lib import marginal_query
from lib.DataFrameData import DataFrameData
from lib.markov_network import MarkovNetworkJax, MarkovNetworkTorch
from lib import max_ent_inference as mei
from lib import privacy_accounting
from synth_data_output import SynthDataOutput
import problem

if __name__ == "__main__":

    algo = str(snakemake.wildcards.algo)
    epsilon = float(snakemake.wildcards.epsilon)
    seed = rng_initialisation.get_seed(int(snakemake.wildcards.repeat_ind), epsilon, algo, "us-census")
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    cat_df = pd.read_csv(str(snakemake.input.data), dtype="category")
    df_data = DataFrameData(cat_df)
    int_tensor = df_data.int_tensor

    query_sets = [
        ("dPoverty", "iSex", "iMilitary"),
        ("dPoverty", "iVietnam", "iKorean"),
        ("dPoverty", "iMilitary", "dYrsserv"),
        ("iMilitary", "iVietnam", "iKorean"),
        ("iMilitary", "iMarital"),
        ("iMilitary", "iMobillim"),
        ("iMilitary", "iEnglish")
    ]
    queries = marginal_query.FullMarginalQuerySet(query_sets, df_data.values_by_col)

    delta = df_data.n ** (-2)
    sensitivity = np.sqrt(2 * len(query_sets))
    sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)

    n_syn_dataset = df_data.n

    syn_dfs_dict = {}
    if algo == "max-ent":
        queries = queries.get_canonical_queries()
        mntorch = MarkovNetworkTorch(df_data.values_by_col, queries)
        mnjax = MarkovNetworkJax(df_data.values_by_col, queries)

        suff_stat = torch.sum(queries.flatten()(int_tensor), axis=0)
        dp_suff_stat = torch.normal(mean=suff_stat.double(), std=sigma_DP)

        lap_start_time = time.time()
        laplace_approx, lap_losses, fail_count = mei.laplace_approximation_normal_prior(dp_suff_stat, df_data.n, sigma_DP, mntorch, max_retries=10, max_iters=6000)
        lap_end_time = time.time()
        rng, mcmc_key = jax.random.split(rng)
        mcmc, backtransform = mei.run_numpyro_mcmc_normalised(
            # mcmc_key, dp_suff_stat, df_data.n, sigma_DP, mnjax, laplace_approx, num_samples=40, num_warmup=8, num_chains=4, disable_progressbar=True
            mcmc_key, dp_suff_stat, df_data.n, sigma_DP, mnjax, laplace_approx, num_samples=4000, num_warmup=800, num_chains=4, disable_progressbar=True
        )
        inf_data = az.from_numpyro(mcmc, log_likelihood=False)
        mcmc_end_time = time.time()
        mcmc_table = az.summary(inf_data)
        posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
        posterior_values = backtransform(posterior_values.norm_lambdas.values.transpose())

        diagnostics = {"losses": lap_losses, "fails": fail_count, "mcmc_table": mcmc_table}
        runtime = {"laplace": lap_end_time - lap_start_time, "mcmc": mcmc_end_time - lap_end_time}
        for n_syn_datasets in problem.max_ent_n_syn_datasets:
            rng, posterior_key = jax.random.split(rng)
            inds = jax.random.choice(posterior_key, posterior_values.shape[0], (n_syn_datasets,))
            posterior_sample = posterior_values[inds, :]
            rng, *syn_data_keys = jax.random.split(rng, n_syn_datasets + 1)
            syn_int_dfs = [mnjax.sample(syn_data_key, jnp.array(posterior_value), n_syn_dataset) for syn_data_key, posterior_value in zip(syn_data_keys, posterior_sample)]
            syn_dfs = [df_data.int_df_to_cat_df(syn_data) for syn_data in syn_int_dfs]
            syn_dfs_dict[n_syn_datasets] = syn_dfs

    elif algo == "PGM":
        from mbi import Dataset, Domain, FactoredInference
        pgm_queries = marginal_query.FullMarginalQuerySet(queries.feature_sets, df_data.values_by_col)
        measurements = []
        for query_set in pgm_queries.feature_sets:
            q = pgm_queries.query_feature_set_sum(query_set, int_tensor).numpy()
            q_dp = q + np.random.normal(loc=0, scale=sigma_DP, size=q.shape[0])
            I_q = np.eye(q.shape[0])
            measurements.append((I_q, q_dp, sigma_DP, query_set))

        domain_key_list = list(df_data.values_by_col.keys())
        domain_value_count_list = [len(df_data.values_by_col[key]) for key in domain_key_list]
        domain = Domain(domain_key_list, domain_value_count_list)
        start_time = time.time()
        # engine = FactoredInference(domain, log=False, iters=100)
        engine = FactoredInference(domain, log=False, iters=10000)
        model = engine.estimate(measurements, engine="MD")
        end_time = time.time()
        syn_data_pgm = model.synthetic_data(n_syn_dataset).df
        syn_dfs = [df_data.ndarray_to_cat_df(syn_data_pgm.values)]
        syn_dfs_dict[len(syn_dfs)] = syn_dfs
        diagnostics = None
        runtime = {"total": end_time - start_time}
    elif algo.startswith("PGM-repeat-"):
        from mbi import Dataset, Domain, FactoredInference
        repeats = int(algo[11:])
        pgm_queries = marginal_query.FullMarginalQuerySet(queries.feature_sets, df_data.values_by_col)
        sensitivity = np.sqrt(2 * len(query_sets) * repeats)
        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)

        syn_dfs = []
        diagnostics = None
        runtimes = []
        for i in range(repeats):
            measurements = []
            for query_set in pgm_queries.feature_sets:
                q = pgm_queries.query_feature_set_sum(query_set, int_tensor).numpy()
                q_dp = q + np.random.normal(loc=0, scale=sigma_DP, size=q.shape[0])
                I_q = np.eye(q.shape[0])
                measurements.append((I_q, q_dp, sigma_DP, query_set))

            domain_key_list = list(df_data.values_by_col.keys())
            domain_value_count_list = [len(df_data.values_by_col[key]) for key in domain_key_list]
            domain = Domain(domain_key_list, domain_value_count_list)
            start_time = time.time()
            # engine = FactoredInference(domain, log=False, iters=100)
            engine = FactoredInference(domain, log=False, iters=10000)
            model = engine.estimate(measurements, engine="MD")
            end_time = time.time()
            runtimes.append(end_time - start_time)
            syn_data_pgm = model.synthetic_data(n_syn_dataset).df
            syn_dfs.append(df_data.ndarray_to_cat_df(syn_data_pgm.values))
        syn_dfs_dict[len(syn_dfs)] = syn_dfs
        runtime = {"total": sum(runtimes)}
    elif algo == "RAP":
        from lib import rap_interface
        kway_attrs = [
            ("dPoverty", "iSex", "iMilitary"),
            ("dPoverty", "iVietnam", "iKorean"),
            ("dPoverty", "iMilitary", "dYrsserv"),
            ("iMilitary", "iVietnam", "iKorean"),
            ("iMilitary", "iMarital", "iEnglish"),
            ("iMilitary", "iMobillim", "iVietnam"),
        ]
        n_syn_dataset_rap = 1000
        rapi = rap_interface.RapInterface(df_data.int_tensor, kway_attrs, df_data.int_df.columns, df_data.values_by_col)
        start_time = time.time()
        rapi.run_rap(
            num_points=n_syn_dataset_rap, epochs=6, top_q=1, iterations=10000, epsilon=epsilon, delta=delta, learning_rate=0.10, use_all_queries=True, oversample=int(n_syn_dataset / n_syn_dataset_rap)
        )
        end_time = time.time()

        syn_dfs = [df_data.int_df_to_cat_df(rapi.un_one_hot_code_D_prime(True))]
        syn_dfs_dict[len(syn_dfs)] = syn_dfs
        diagnostics = None
        runtime = {"total": end_time - start_time}
    elif algo == "PEP":
        sys.path.insert(0, "iterative-dp")
        import pep
        from mbi import Domain, Dataset
        sys.path = sys.path[1:]

        x_values = df_data.get_x_values()

        domain_key_list = list(df_data.values_by_col.keys())
        domain_value_count_list = [len(df_data.values_by_col[key]) for key in domain_key_list]
        domain = Domain(domain_key_list, domain_value_count_list)
        marginals = pep.MyMarginals(domain, query_sets)
        dataset = Dataset(df_data.int_df, domain)
        generator = pep.PEP(domain, marginals, 100000000)

        suff_stat = marginals.get_answers(dataset)
        dp_suff_stat = torch.normal(mean=torch.tensor(suff_stat), std=sigma_DP / df_data.n)
        generator.query_measurements = [(i, x.item()) for i, x in enumerate(dp_suff_stat)]
        start_time = time.time()
        generator.project(10000)
        end_time = time.time()

        probs = generator.synthethic_weights
        syn_data_inds = torch.distributions.Categorical(probs=torch.tensor(probs)).sample((n_syn_dataset,))
        syn_data = x_values[syn_data_inds]
        syn_dfs = [df_data.ndarray_to_cat_df(syn_data.numpy())]
        syn_dfs_dict[len(syn_dfs)] = syn_dfs
        diagnostics = None
        runtime = {"total": end_time - start_time}
    else:
        raise ValueError("Given algorithm {} is not implemented".format(algo))

    output_object = SynthDataOutput(
        n_orig=df_data.n, generator_algorithm=algo, epsilon=epsilon, delta=delta, 
        synth_data=syn_dfs_dict, generator_diagnostics=diagnostics, runtime=runtime
    )
    output = str(snakemake.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as file:
        pickle.dump(output_object, file)

