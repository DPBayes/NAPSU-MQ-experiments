
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import sys
import numpy as onp
import warnings
import torch

with warnings.catch_warnings():
    sys.path.append("relaxed-adaptive-projection")
    warnings.filterwarnings("ignore", message="jax.experimental.optimizers is deprecated")
    import relaxed_adaptive_projection as rap
    import datasets
    import statistickway as stat_module
    sys.path.pop()

from lib import binary_lr_data
import pandas as pd

class RapInterface:
    """Plumbing between the original RAP implementation and our experimental code.
    """
    def __init__(self, data, kway_attrs, features, categories_by_feature):
        self.data = data.numpy()
        self.kway_attrs = kway_attrs
        self.features = features 
        self.categories_by_feature = categories_by_feature

    def run_rap(
        self, num_points, epsilon, delta, epochs, top_q, iterations, learning_rate=0.01, use_all_queries=False, 
        rap_stopping_condition=1e-7, oversample=5, generator=None
        ):
        self.dataset = datasets.adult.Adult(False, None, False, None, None)
        D = self.dataset.convert_dataset(self.data, {feature: len(values) for feature, values in self.categories_by_feature.items()})
        n, d = D.shape

        kway_compact_queries, _ = self.dataset.get_queries(self.kway_attrs)
        self.all_statistic_fn = stat_module.preserve_statistic(kway_compact_queries)
        self.true_statistics = self.all_statistic_fn(D)
        feats_csum = np.array([0] + list(self.dataset.domain.values())).cumsum()
        feats_idx = [
            list(range(feats_csum[i], feats_csum[i + 1]))
            for i in range(len(feats_csum) - 1)
        ]

        rap_config = rap.RAPConfiguration(
            num_points=num_points,
            num_generated_points=num_points,
            num_dimensions=d,
            statistic_function=self.all_statistic_fn,
            preserve_subset_statistic=stat_module.preserve_subset_statistic,
            get_queries=self.dataset.get_queries,
            get_sensitivity=stat_module.get_sensitivity,

            verbose=True,
            silent=False,

            epochs=epochs if not use_all_queries else 1,
            iterations=iterations,
            epsilon=epsilon,
            delta=delta,
            norm=rap.Norm("L2"),
            projection_interval=None,
            optimizer_learning_rate=learning_rate,
            lambda_l1=0,
            k=len(self.kway_attrs[0]),
            top_q=top_q,
            use_all_queries=use_all_queries,
            rap_stopping_condition=rap_stopping_condition,
            initialize_binomial=False,
            feats_idx=feats_idx,
        )

        rng = jax.random.PRNGKey(torch.randint(0, 2**32, (1,), generator=generator).item())
        rng, init_key, train_key, round_key = jax.random.split(rng, 4)
        alg = rap.RAP(rap_config, init_key)
        alg.train(D, self.kway_attrs, train_key)
        self.D_prime = alg.D_prime
        self.D_prime_rounded = alg.generate_rounded_dataset(round_key, oversample=oversample)

    def un_one_hot_code_D_prime(self, return_dataframe=False):
        n, d = self.D_prime_rounded.shape
        D_prime = self.D_prime_rounded.astype(int)

        data = np.zeros((n, len(self.categories_by_feature.keys())), dtype=int)

        current_feature_start_index = 0
        for feature_index, feature in enumerate(self.features):
            feature_data = D_prime[:, current_feature_start_index:current_feature_start_index+len(self.categories_by_feature[feature])]
            category_indices = np.argmax(feature_data, axis=1)
            categories = np.array(self.categories_by_feature[feature])[category_indices]
            data = data.at[:, feature_index].set(categories)
            current_feature_start_index += len(self.categories_by_feature[feature])

        if not return_dataframe:
            return onp.asarray(data)
        else:
            return pd.DataFrame(onp.asarray(data), columns=self.features, dtype=int)

    def print_rap_performance(self):
        all_synth_statistics = self.all_statistic_fn(self.D_prime)
        all_synth_statistics_rounded = self.all_statistic_fn(self.D_prime_rounded)

        max_base = np.max(np.absolute(self.true_statistics - np.zeros(self.true_statistics.shape)))
        l1_base = np.linalg.norm(self.true_statistics - np.zeros(self.true_statistics.shape), ord=1)
        l2_base = np.linalg.norm(self.true_statistics - np.zeros(self.true_statistics.shape), ord=2)
        print("Baseline max abs error", max_base)
        print("Baseline L1 error", l1_base)
        print("Baseline L2 error", l2_base)

        max_final = np.max(np.absolute(self.true_statistics - all_synth_statistics))
        l1_final = np.linalg.norm(self.true_statistics - all_synth_statistics, ord=1)
        l2_final = np.linalg.norm(self.true_statistics - all_synth_statistics, ord=2)
        print("Final max abs error", max_final)
        print("Final L1 error", l1_final)
        print("Final L2 error", l2_final)

        max_final_rounded = np.max(np.absolute(self.true_statistics - all_synth_statistics_rounded))
        l1_final_rounded = np.linalg.norm(self.true_statistics - all_synth_statistics_rounded, ord=1)
        l2_final_rounded = np.linalg.norm(self.true_statistics - all_synth_statistics_rounded, ord=2)
        print("Final rounded max abs error", max_final_rounded)
        print("Final rounded L1 error", l1_final_rounded)
        print("Final rounded L2 error", l2_final_rounded)