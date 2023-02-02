import pandas as pd
import numpy as np
import pickle

import sys
import os
sys.path.append(os.getcwd())

from lib.mst import MST_selection
from lib.DataFrameData import DataFrameData
from lib.undirected_graph import UndirectedGraph
from mbi import Dataset, Domain, FactoredInference

if __name__ == "__main__":
    cat_df = pd.read_csv(str(snakemake.input), dtype="category")
    df_data = DataFrameData(cat_df)
    delta = df_data.n ** (-2)

    feature_sets = [("age", "income"), ("race", "income"), ("gender", "income"), ("race", "gender"), ("hours-per-week", "income")]

    domain_key_list = list(df_data.values_by_col.keys())
    domain_value_count_list = [len(df_data.values_by_col[key]) for key in domain_key_list]
    domain = Domain(domain_key_list, domain_value_count_list)
    cliques = MST_selection(Dataset(df_data.int_df, domain), 0.5, delta, cliques_to_include=feature_sets)
    graph = UndirectedGraph.from_clique_list(cliques)
    graphviz = graph.visualize()
    graphviz.render(filename="latex/figures/adult-reduced/queries", format="pdf", cleanup=True)

    output = str(snakemake.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as file:
        pickle.dump(cliques, file)